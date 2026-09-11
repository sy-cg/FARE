# -*- coding: utf-8 -*-
"""
scripts/run_findrec.py

Train and evaluate the FindRec baseline under this project's protocol.

The released FindRec code is RecBole-based. This runner adapts its model
structure to the existing project pipeline:
    - Reuses data/Processed_<Dataset>/train.txt val.txt test.txt
    - Reuses text_features.npy and vision_features.npy
    - Reuses RankingEvaluator and FairnessEvaluator
    - Saves val_ranking.npz / test_ranking.npz and metrics_summary.json

Example:
    python scripts/run_findrec.py \
      --dataset Video_Games \
      --config configs/findrec_3090.yaml \
      --seed 2024 \
      --run_id findrec_video_seed2024
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import build_sasrec_train_loader
from src.evaluator_fairness import FairnessEvaluator, save_eval_tables
from src.evaluator_ranking import (
    RankingEvaluator,
    infer_num_items_from_processed_dir,
    save_ranking_result_npz,
    score_fn_from_model_method,
)
try:
    from src.io_utils import safe_torch_load
except ModuleNotFoundError:  # pragma: no cover - standalone copied-script fallback
    def safe_torch_load(path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)
from src.model_findrec import FindRec


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_dataset_dir(project_root: Path, datasets_cfg: Dict[str, Any], dataset_name: str) -> tuple[Path, Dict[str, Any]]:
    datasets = datasets_cfg.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Dataset {dataset_name!r} not found. Available: {list(datasets)}")
    ds_info = datasets[dataset_name]
    data_path = Path(ds_info["path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")
    return data_path, ds_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FindRec baseline")
    parser.add_argument("--config", type=str, default="configs/findrec_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_fairness_eval", action="store_true")
    parser.add_argument("--skip_test_after_training", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["eval"]["batch_size"] = args.eval_batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.max_seq_len is not None:
        cfg["model"]["max_seq_len"] = args.max_seq_len
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.no_fairness_eval:
        cfg["eval"]["run_fairness_eval"] = False
    if args.skip_test_after_training:
        cfg["eval"]["run_test_after_training"] = False
    return cfg


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=tuple(float(x) for x in opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
    model: FindRec,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_mveb_loss = 0.0
    total_alignment_loss = 0.0
    total_kl_loss = 0.0
    total_examples = 0
    grad_norms = []

    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 2.0))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    log_every = int(cfg["train"].get("log_every", 100))

    pbar = tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.training_loss(user_ids, sequences, lengths, targets, label_smoothing=label_smoothing)
        part_summary = ", ".join(f"{name}={value:.6g}" for name, value in parts.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite FindRec loss at epoch={epoch}, step={step}: "
                f"loss={loss.item():.6g}, {part_summary}"
            )

        loss.backward()
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                f"Non-finite FindRec gradient at epoch={epoch}, step={step}: {part_summary}"
            ) from exc
        optimizer.step()

        batch_size = int(targets.numel())
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_rec_loss += float(parts["rec_loss"]) * batch_size
        total_mveb_loss += float(parts["mveb_loss"]) * batch_size
        total_alignment_loss += float(parts["alignment_loss"]) * batch_size
        total_kl_loss += float(parts["kl_loss"]) * batch_size
        total_examples += batch_size
        grad_norms.append(float(norm.detach().cpu().item()))

        if log_every > 0 and step % log_every == 0:
            pbar.set_postfix({"loss": total_loss / max(total_examples, 1)})

    return {
        "train_loss": total_loss / max(total_examples, 1),
        "rec_loss": total_rec_loss / max(total_examples, 1),
        "mveb_loss": total_mveb_loss / max(total_examples, 1),
        "alignment_loss": total_alignment_loss / max(total_examples, 1),
        "kl_loss": total_kl_loss / max(total_examples, 1),
        "avg_grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "num_train_examples": int(total_examples),
    }


def evaluate_split(
    model: FindRec,
    data_dir: Path,
    split: str,
    cfg: Dict[str, Any],
    num_items: int,
    device: torch.device,
    run_dir: Path,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    eval_cfg = cfg["eval"]
    ranking_eval = RankingEvaluator(
        num_items=num_items,
        ks=tuple(eval_cfg.get("ks", [5, 10, 20, 100])),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    )
    ranking_result = ranking_eval.evaluate_from_file(
        model=model,
        split_path=str(data_dir / f"{split}.txt"),
        score_fn=score_fn_from_model_method,
        batch_size=int(eval_cfg.get("batch_size", 128)),
        device=device,
        split_name=split,
        mask_seen_items=bool(eval_cfg.get("mask_seen_items", True)),
        show_progress=True,
    )
    out: Dict[str, Any] = {"ranking_metrics": ranking_result.metrics}

    if save_outputs and bool(eval_cfg.get("save_topk_npz", True)):
        save_ranking_result_npz(ranking_result, run_dir / f"{split}_ranking.npz")

    if bool(eval_cfg.get("run_fairness_eval", True)):
        fair_eval = FairnessEvaluator(data_dir=str(data_dir), ks=tuple(eval_cfg.get("ks", [5, 10, 20, 100])))
        tables = fair_eval.evaluate(ranking_result)
        if save_outputs:
            save_eval_tables(tables, output_dir=str(run_dir), prefix=f"{split}_fairness")
        out["fairness_tables"] = {name: df.to_dict(orient="records") for name, df in tables.items()}
    return out


def save_checkpoint(
    path: Path,
    model: FindRec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model: FindRec, device: torch.device) -> Dict[str, Any]:
    ckpt = safe_torch_load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_yaml(PROJECT_ROOT / args.config), args)
    dataset_name = args.dataset or cfg.get("default_dataset")
    if not dataset_name:
        raise ValueError("Dataset must be provided by --dataset or default_dataset in config")

    datasets_cfg = load_yaml(PROJECT_ROOT / cfg["paths"].get("datasets_config", "configs/datasets.yaml"))
    data_dir, ds_info = resolve_dataset_dir(PROJECT_ROOT, datasets_cfg, str(dataset_name))
    if args.max_seq_len is None and "max_seq_len" in ds_info:
        cfg["model"]["max_seq_len"] = int(ds_info["max_seq_len"])

    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    device_name = "cpu" if args.cpu else str(cfg.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / cfg["paths"].get("result_root", "results") / str(dataset_name) / cfg.get("run_name", "findrec") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    num_items = infer_num_items_from_processed_dir(data_dir)
    cfg_resolved = copy.deepcopy(cfg)
    cfg_resolved["dataset_name"] = str(dataset_name)
    cfg_resolved["data_dir"] = str(data_dir.relative_to(PROJECT_ROOT) if data_dir.is_relative_to(PROJECT_ROOT) else data_dir)
    cfg_resolved["num_items"] = num_items
    cfg_resolved["device_resolved"] = str(device)
    save_json(cfg_resolved, run_dir / "config_resolved.json")

    print("=" * 90)
    print(f"Dataset: {dataset_name}")
    print(f"Data dir: {data_dir}")
    print(f"Run dir:  {run_dir}")
    print(f"Device:   {device}")
    print(f"Items:    {num_items}")
    print("=" * 90)

    train_dataset, train_loader = build_sasrec_train_loader(
        train_path=str(data_dir / "train.txt"),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        seed=seed,
        pin_memory=bool(cfg["train"].get("pin_memory", True)) and device.type == "cuda",
        drop_last=False,
    )
    print(f"Training examples: {len(train_dataset)}")

    model = FindRec(num_items=num_items, data_dir=data_dir, config=cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    print(f"Model parameters: total={model.num_parameters:,}, trainable={model.num_trainable_parameters:,}")

    metric_for_best = str(cfg["eval"].get("metric_for_best", "ndcg@10"))
    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 5))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    eval_every = int(cfg["train"].get("eval_every", 1))
    epochs = int(cfg["train"]["epochs"])
    best_ckpt_path = run_dir / "best_model.pt"

    fieldnames = [
        "epoch",
        "train_loss",
        "rec_loss",
        "mveb_loss",
        "alignment_loss",
        "kl_loss",
        "avg_grad_norm",
        "val_metric",
        "best_metric",
        "best_epoch",
        "epoch_time_sec",
    ]
    train_log_path = run_dir / "train_log.csv"
    with train_log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    metrics_summary: Dict[str, Any] = {
        "dataset": str(dataset_name),
        "method": "FindRec",
        "run_dir": str(run_dir),
        "best_metric_name": metric_for_best,
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        epoch_time = time.time() - t0

        val_metric = None
        if epoch % eval_every == 0:
            val_out = evaluate_split(model, data_dir, "val", cfg, num_items, device, run_dir, save_outputs=False)
            val_metrics = val_out["ranking_metrics"]
            if metric_for_best not in val_metrics:
                raise KeyError(f"metric_for_best={metric_for_best} not found in val metrics: {val_metrics.keys()}")
            val_metric = float(val_metrics[metric_for_best])

            if val_metric > best_metric + min_delta:
                best_metric = val_metric
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_metric, cfg_resolved)
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:03d} | loss={train_stats['train_loss']:.6f} | "
                f"rec={train_stats['rec_loss']:.6f} mveb={train_stats['mveb_loss']:.6f} | "
                f"align={train_stats['alignment_loss']:.6f} kl={train_stats['kl_loss']:.6f} | "
                f"val {metric_for_best}={val_metric:.6f} | best={best_metric:.6f}@{best_epoch} | "
                f"time={epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d} | loss={train_stats['train_loss']:.6f} | "
                f"rec={train_stats['rec_loss']:.6f} mveb={train_stats['mveb_loss']:.6f} | "
                f"align={train_stats['alignment_loss']:.6f} kl={train_stats['kl_loss']:.6f} | "
                f"time={epoch_time:.1f}s"
            )

        row = {
            "epoch": epoch,
            **train_stats,
            "val_metric": val_metric if val_metric is not None else "",
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_time_sec": epoch_time,
        }
        with train_log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({k: row.get(k, "") for k in fieldnames})
        metrics_summary["epochs"].append(row)
        save_json(metrics_summary, run_dir / "metrics_summary.json")

        if epoch % eval_every == 0 and bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if not best_ckpt_path.exists():
        print("No validation checkpoint was saved; saving final model as best_model.pt")
        save_checkpoint(best_ckpt_path, model, optimizer, epochs, best_metric, cfg_resolved)

    print(f"Loading best checkpoint: {best_ckpt_path}")
    ckpt = load_checkpoint(best_ckpt_path, model, device)
    metrics_summary["best_epoch"] = int(ckpt.get("epoch", best_epoch))
    metrics_summary["best_metric"] = float(ckpt.get("best_metric", best_metric))

    print("Evaluating best model on validation split...")
    val_final = evaluate_split(model, data_dir, "val", cfg, num_items, device, run_dir, save_outputs=True)
    metrics_summary["val_final"] = val_final["ranking_metrics"]

    if bool(cfg["eval"].get("run_test_after_training", True)):
        print("Evaluating best model on test split...")
        test_final = evaluate_split(model, data_dir, "test", cfg, num_items, device, run_dir, save_outputs=True)
        metrics_summary["test_final"] = test_final["ranking_metrics"]

    save_json(metrics_summary, run_dir / "metrics_summary.json")
    print("Done.")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
