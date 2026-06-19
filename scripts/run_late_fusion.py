# -*- coding: utf-8 -*-
"""
scripts/run_late_fusion.py

Train and evaluate late-fusion multimodal SASRec baseline.

Run from project root:
    python scripts/run_late_fusion.py --dataset Video_Games --config configs/late_fusion_3090.yaml

Recommended 3090 examples:
    python scripts/run_late_fusion.py --dataset Video_Games --batch_size 256 --eval_batch_size 512
    python scripts/run_late_fusion.py --dataset Baby_Products --batch_size 256 --eval_batch_size 512
    python scripts/run_late_fusion.py --dataset Sports_and_Outdoors --batch_size 128 --eval_batch_size 512 --eval_every 2

Optional initialization from a SASRec-ID checkpoint:
    python scripts/run_late_fusion.py \
        --dataset Video_Games \
        --init_sasrec_checkpoint results/Video_Games/sasrec_id/<run_id>/best_model.pt

Outputs:
    results/<Dataset>/late_fusion_sasrec/<run_id>/
        config_resolved.json
        train_log.csv
        best_model.pt
        val_ranking.npz / test_ranking.npz
        val_fairness_*.csv / test_fairness_*.csv
        metrics_summary.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

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
from src.io_utils import safe_torch_load
from src.losses import clip_grad_norm_, full_sort_ce_loss
from src.model_late_fusion import LateFusionSASRecID


# ==========================================================
# Config utilities
# ==========================================================


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_dataset_dir(project_root: Path, datasets_cfg: Dict[str, Any], dataset_name: str) -> Tuple[Path, Dict[str, Any]]:
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


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

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
    if args.hidden_size is not None:
        cfg["model"]["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        cfg["model"]["num_layers"] = args.num_layers
    if args.num_heads is not None:
        cfg["model"]["num_heads"] = args.num_heads
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.mm_score_dim is not None:
        cfg["model"]["mm_score_dim"] = args.mm_score_dim
    if args.text_weight_init is not None:
        cfg["model"]["text_weight_init"] = args.text_weight_init
    if args.vision_weight_init is not None:
        cfg["model"]["vision_weight_init"] = args.vision_weight_init
    if args.max_modal_weight is not None:
        cfg["model"]["max_modal_weight"] = args.max_modal_weight
    if args.text_only:
        cfg["model"]["use_text"] = True
        cfg["model"]["use_vision"] = False
    if args.vision_only:
        cfg["model"]["use_text"] = False
        cfg["model"]["use_vision"] = True
    if args.init_sasrec_checkpoint is not None:
        cfg["model"]["init_sasrec_checkpoint"] = args.init_sasrec_checkpoint
    if args.freeze_id_backbone:
        cfg["model"]["freeze_id_backbone"] = True
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.no_fairness_eval:
        cfg["eval"]["run_fairness_eval"] = False

    return cfg


# ==========================================================
# Model / optimizer
# ==========================================================


def build_model(num_items: int, data_dir: Path, cfg: Dict[str, Any], device: torch.device) -> LateFusionSASRecID:
    m = cfg["model"]
    model = LateFusionSASRecID(
        num_items=num_items,
        data_dir=data_dir,
        max_seq_len=int(m["max_seq_len"]),
        hidden_size=int(m["hidden_size"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        activation=str(m.get("activation", "gelu")),
        layer_norm_eps=float(m.get("layer_norm_eps", 1e-12)),
        tie_output_embedding=bool(m.get("tie_output_embedding", True)),
        mm_score_dim=int(m.get("mm_score_dim", 64)),
        projection_seed=int(m.get("projection_seed", 2026)),
        use_text=bool(m.get("use_text", True)),
        use_vision=bool(m.get("use_vision", True)),
        cache_projected_features=bool(m.get("cache_projected_features", True)),
        feature_chunk_size=int(m.get("feature_chunk_size", 32768)),
        text_weight_init=float(m.get("text_weight_init", 0.05)),
        vision_weight_init=float(m.get("vision_weight_init", 0.05)),
        max_modal_weight=float(m.get("max_modal_weight", 0.5)),
        learnable_modal_weights=bool(m.get("learnable_modal_weights", True)),
        normalize_queries=bool(m.get("normalize_queries", True)),
        modal_dropout=float(m.get("modal_dropout", 0.0)),
    )
    model.to(device)
    return model


def load_sasrec_checkpoint_if_needed(model: LateFusionSASRecID, cfg: Dict[str, Any], device: torch.device) -> None:
    ckpt_path = cfg["model"].get("init_sasrec_checkpoint", None)
    if not ckpt_path:
        return

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"init_sasrec_checkpoint not found: {ckpt_path}")

    print(f"Loading SASRec-ID checkpoint for initialization: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded with strict=False. missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        print(f"Unexpected keys example: {unexpected[:5]}")
    if missing:
        print(f"Missing keys example: {missing[:5]}")


def freeze_id_backbone_if_needed(model: LateFusionSASRecID, cfg: Dict[str, Any]) -> None:
    if not bool(cfg["model"].get("freeze_id_backbone", False)):
        return

    print("Freezing ID backbone parameters; training only late-fusion modal components.")
    modal_keywords = {
        "text_query",
        "vision_query",
        "raw_text_weight",
        "raw_vision_weight",
    }
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in modal_keywords)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        betas = tuple(float(x) for x in opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


# ==========================================================
# Training / evaluation
# ==========================================================


def train_one_epoch(
    model: LateFusionSASRecID,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_grad_norm = 0.0
    num_steps = 0

    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    grad_clip_norm = cfg["train"].get("grad_clip_norm", None)
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    log_every = int(cfg["train"].get("log_every", 100))

    pbar = tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model.full_sort_scores(user_ids, sequences, lengths)

        if not torch.isfinite(logits).all():
            tw, vw = model.get_modal_weights()
            raise FloatingPointError(
                f"Non-finite logits detected at epoch={epoch}, step={step}. "
                f"text_weight={tw}, vision_weight={vw}"
            )

        loss = full_sort_ce_loss(logits, targets, label_smoothing=label_smoothing)

        if not torch.isfinite(loss):
            tw, vw = model.get_modal_weights()
            raise FloatingPointError(
                f"Non-finite loss detected at epoch={epoch}, step={step}. "
                f"loss={loss.item()}, text_weight={tw}, vision_weight={vw}"
            )

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradient detected at epoch={epoch}, step={step}, param={name}"
                )

        grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)

        if not np.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite grad_norm detected at epoch={epoch}, step={step}, grad_norm={grad_norm}"
            )

        optimizer.step()

        with torch.no_grad():
            if hasattr(model, "raw_text_weight") and model.raw_text_weight.requires_grad:
                model.raw_text_weight.clamp_(min=-20.0, max=20.0)
            if hasattr(model, "raw_vision_weight") and model.raw_vision_weight.requires_grad:
                model.raw_vision_weight.clamp_(min=-20.0, max=20.0)

        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError(
                    f"Non-finite parameter detected after optimizer step: "
                    f"epoch={epoch}, step={step}, param={name}"
                )

        tw, vw = model.get_modal_weights()
        if not np.isfinite(tw) or not np.isfinite(vw):
            raise FloatingPointError(
                f"Non-finite modal weights after optimizer step: "
                f"text_weight={tw}, vision_weight={vw}, epoch={epoch}, step={step}"
            )

        bs = targets.shape[0]
        total_loss += float(loss.item()) * bs
        total_examples += bs
        total_grad_norm += float(grad_norm)
        num_steps += 1

        if step % log_every == 0:
            tw, vw = model.get_modal_weights()
            pbar.set_postfix({
                "loss": total_loss / max(1, total_examples),
                "tw": tw,
                "vw": vw,
            })

    tw, vw = model.get_modal_weights()
    return {
        "train_loss": total_loss / max(1, total_examples),
        "avg_grad_norm": total_grad_norm / max(1, num_steps),
        "num_train_examples": int(total_examples),
        "text_weight": tw,
        "vision_weight": vw,
    }


def evaluate_split(
    model: LateFusionSASRecID,
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
        ks=tuple(eval_cfg.get("ks", [5, 10, 20])),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    )

    ranking_result = ranking_eval.evaluate_from_file(
        model=model,
        split_path=str(data_dir / f"{split}.txt"),
        score_fn=score_fn_from_model_method,
        batch_size=int(eval_cfg.get("batch_size", 512)),
        device=device,
        split_name=split,
        mask_seen_items=bool(eval_cfg.get("mask_seen_items", True)),
        show_progress=True,
    )
    out: Dict[str, Any] = {"ranking_metrics": ranking_result.metrics}

    if save_outputs and bool(eval_cfg.get("save_topk_npz", True)):
        save_ranking_result_npz(ranking_result, str(run_dir / f"{split}_ranking.npz"))

    if bool(eval_cfg.get("run_fairness_eval", True)):
        fair_eval = FairnessEvaluator(data_dir=str(data_dir), ks=tuple(eval_cfg.get("ks", [5, 10, 20])))
        tables = fair_eval.evaluate(ranking_result)
        if save_outputs:
            save_eval_tables(tables, output_dir=str(run_dir), prefix=f"{split}_fairness")
        out["fairness_tables"] = {name: df.to_dict(orient="records") for name, df in tables.items()}

    return out


def save_checkpoint(path: Path, model: LateFusionSASRecID, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "modal_weights": {
                "text_weight": model.get_modal_weights()[0],
                "vision_weight": model.get_modal_weights()[1],
            },
        },
        path,
    )


def load_checkpoint(path: Path, model: LateFusionSASRecID, device: torch.device) -> Dict[str, Any]:
    ckpt = safe_torch_load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train late-fusion multimodal SASRec baseline")
    parser.add_argument("--config", type=str, default="configs/late_fusion_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--mm_score_dim", type=int, default=None)
    parser.add_argument("--text_weight_init", type=float, default=None)
    parser.add_argument("--vision_weight_init", type=float, default=None)
    parser.add_argument("--max_modal_weight", type=float, default=None)
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--vision_only", action="store_true")
    parser.add_argument("--init_sasrec_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_id_backbone", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_fairness_eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    cfg = apply_cli_overrides(cfg, args)

    dataset_name = args.dataset or cfg.get("default_dataset")
    if not dataset_name:
        raise ValueError("Dataset must be provided by --dataset or default_dataset in config")

    datasets_cfg = load_yaml(PROJECT_ROOT / cfg["paths"].get("datasets_config", "configs/datasets.yaml"))
    data_dir, ds_info = resolve_dataset_dir(PROJECT_ROOT, datasets_cfg, dataset_name)
    if args.max_seq_len is None and "max_seq_len" in ds_info:
        cfg["model"]["max_seq_len"] = int(ds_info["max_seq_len"])

    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    device_name = "cpu" if args.cpu else str(cfg.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = PROJECT_ROOT / cfg["paths"].get("result_root", "results")
    run_dir = result_root / dataset_name / cfg.get("run_name", "late_fusion_sasrec") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    num_items = infer_num_items_from_processed_dir(str(data_dir))
    cfg_resolved = copy.deepcopy(cfg)
    cfg_resolved["dataset_name"] = dataset_name
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

    model = build_model(num_items=num_items, data_dir=data_dir, cfg=cfg, device=device)
    load_sasrec_checkpoint_if_needed(model, cfg, device)
    freeze_id_backbone_if_needed(model, cfg)
    optimizer = build_optimizer(model, cfg)
    print(f"Model parameters: total={model.num_parameters:,}, trainable={model.num_trainable_parameters:,}")
    print(f"Initial modal weights: text={model.get_modal_weights()[0]:.6f}, vision={model.get_modal_weights()[1]:.6f}")

    metric_for_best = str(cfg["eval"].get("metric_for_best", "ndcg@20"))
    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 10))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    eval_every = int(cfg["train"].get("eval_every", 1))
    epochs = int(cfg["train"]["epochs"])
    best_ckpt_path = run_dir / "best_model.pt"
    train_log_path = run_dir / "train_log.csv"

    fieldnames = [
        "epoch", "train_loss", "avg_grad_norm", "text_weight", "vision_weight",
        "val_metric", "best_metric", "best_epoch", "epoch_time_sec",
    ]
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    metrics_summary: Dict[str, Any] = {
        "dataset": dataset_name,
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
                f"val {metric_for_best}={val_metric:.6f} | best={best_metric:.6f}@{best_epoch} | "
                f"tw={train_stats['text_weight']:.4f} vw={train_stats['vision_weight']:.4f} | "
                f"time={epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d} | loss={train_stats['train_loss']:.6f} | "
                f"tw={train_stats['text_weight']:.4f} vw={train_stats['vision_weight']:.4f} | time={epoch_time:.1f}s"
            )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "avg_grad_norm": train_stats["avg_grad_norm"],
            "text_weight": train_stats["text_weight"],
            "vision_weight": train_stats["vision_weight"],
            "val_metric": val_metric if val_metric is not None else "",
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_time_sec": epoch_time,
        }
        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
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
    metrics_summary["best_modal_weights"] = {
        "text_weight": model.get_modal_weights()[0],
        "vision_weight": model.get_modal_weights()[1],
    }

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
