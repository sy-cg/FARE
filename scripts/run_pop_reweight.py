# -*- coding: utf-8 -*-
"""
scripts/run_pop_reweight.py

Popularity-reweighted training baseline for ID-only sequential recommendation.

This is a fairness baseline, not a new architecture. It trains an ID backbone
with popularity-weighted full-sort cross entropy:

    loss_i = CE(logits_i, target_i) * w[target_i]

where w is computed from item_popularity_train.npy using training data only.

Supported backbones:
    --backbone sasrec
    --backbone gru4rec
    --backbone bert4rec

Legacy-compatible:
    --model sasrec
    --model gru4rec
    --model bert4rec

Output naming:
    sasrec  -> method = SASRec + PopReweight,  model_name = sasrec_pop_reweight
    gru4rec -> method = GRU4Rec + PopReweight, model_name = gru4rec_pop_reweight
    bert4rec -> method = BERT4Rec + PopReweight, model_name = bert4rec_pop_reweight

Typical usage:

    python scripts/run_pop_reweight.py \
      --backbone sasrec \
      --dataset Video_Games \
      --config configs/pop_reweight_3090.yaml \
      --run_id sasrec_pop_reweight_video

    python scripts/run_pop_reweight.py \
      --backbone gru4rec \
      --dataset Video_Games \
      --config configs/pop_reweight_3090.yaml \
      --run_id gru4rec_pop_reweight_video

    python scripts/run_pop_reweight.py \
      --backbone bert4rec \
      --dataset Video_Games \
      --config configs/pop_reweight_3090.yaml \
      --run_id bert4rec_pop_reweight_video
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in [str(SRC_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


try:
    from run_id_backbone import (  # type: ignore
        EvalNextItemDataset,
        NextItemTrainDataset,
        build_model,
        build_optimizer,
        build_seen_items,
        evaluate_full_sort,
        get_dataset_dir,
        get_metric,
        infer_num_items,
        load_yaml,
        make_collate_fn,
        read_user_sequences,
        resolve_path,
        save_json,
        safe_torch_load,
        set_seed,
        unwrap_state_dict,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "run_pop_reweight.py depends on scripts/run_id_backbone.py. "
        "Please make sure it exists and passes py_compile."
    ) from exc


try:
    from losses import (  # type: ignore
        build_popularity_weights,
        describe_popularity_weights,
        load_item_popularity_counts,
        popularity_weighted_ce_loss_with_stats,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import src/losses.py. "
        "Please make sure popularity-weighted CE utilities are added."
    ) from exc


# ==========================================================
# Naming
# ==========================================================


def method_name_from_backbone(backbone: str) -> str:
    backbone = str(backbone).lower().strip()
    if backbone == "sasrec":
        return "SASRec + PopReweight"
    if backbone == "gru4rec":
        return "GRU4Rec + PopReweight"
    if backbone == "bert4rec":
        return "BERT4Rec + PopReweight"
    return f"{backbone} + PopReweight"


def run_name_from_backbone(backbone: str) -> str:
    backbone = str(backbone).lower().strip()
    return f"{backbone}_pop_reweight"


def resolve_backbone(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    backbone = (
        args.backbone
        or args.model
        or cfg.get("model", {}).get("backbone")
        or cfg.get("backbone")
        or "sasrec"
    )
    backbone = str(backbone).lower().strip()
    if backbone not in {"sasrec", "gru4rec", "bert4rec"}:
        raise ValueError(f"Unsupported backbone={backbone!r}. Expected sasrec/gru4rec/bert4rec.")
    return backbone


# ==========================================================
# Config / args
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PopReweight fairness baseline")

    # Prefer --backbone. Keep --model for backward compatibility.
    parser.add_argument("--backbone", type=str, default=None, choices=["sasrec", "gru4rec", "bert4rec"])
    parser.add_argument("--model", type=str, default=None, choices=["sasrec", "gru4rec", "bert4rec"])

    parser.add_argument("--config", type=str, default="configs/pop_reweight_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    # Common overrides.
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
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Popularity-reweight overrides.
    parser.add_argument("--pop_scheme", type=str, default=None, choices=["inverse_log", "inverse_sqrt", "inverse_power"])
    parser.add_argument("--pop_alpha", type=float, default=None)
    parser.add_argument("--pop_min_weight", type=float, default=None)
    parser.add_argument("--pop_max_weight", type=float, default=None)
    parser.add_argument("--no_normalize_pop_weights", action="store_true")
    parser.add_argument("--no_normalize_batch_weight", action="store_true")
    parser.add_argument("--popularity_file", type=str, default=None)

    parser.add_argument("--no_fairness_eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace, backbone: str) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("pop_reweight", {})

    cfg["model"]["backbone"] = backbone

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
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    if args.pop_scheme is not None:
        cfg["pop_reweight"]["scheme"] = args.pop_scheme
    if args.pop_alpha is not None:
        cfg["pop_reweight"]["alpha"] = args.pop_alpha
    if args.pop_min_weight is not None:
        cfg["pop_reweight"]["min_weight"] = args.pop_min_weight
    if args.pop_max_weight is not None:
        cfg["pop_reweight"]["max_weight"] = args.pop_max_weight
    if args.no_normalize_pop_weights:
        cfg["pop_reweight"]["normalize_to_mean"] = False
    if args.no_normalize_batch_weight:
        cfg["pop_reweight"]["normalize_batch_weight"] = False
    if args.popularity_file is not None:
        cfg["pop_reweight"]["popularity_file"] = args.popularity_file

    if args.no_fairness_eval:
        cfg["eval"]["run_fairness_eval"] = False

    if args.cpu:
        cfg["device"] = "cpu"

    return cfg


# ==========================================================
# Popularity helpers
# ==========================================================


def compute_popularity_counts_from_train(train_rows, num_items: int) -> torch.Tensor:
    counts = torch.zeros(num_items + 1, dtype=torch.float32)
    for _, seq in train_rows:
        for item in seq:
            item = int(item)
            if 0 < item <= num_items:
                counts[item] += 1.0
    counts[0] = 0.0
    return counts


def load_or_compute_popularity_counts(
    data_dir: Path,
    train_rows,
    num_items: int,
    pop_cfg: Dict[str, Any],
) -> torch.Tensor:
    filename = str(pop_cfg.get("popularity_file", "item_popularity_train.npy"))
    fallback_to_train = bool(pop_cfg.get("fallback_to_train_sequences", True))
    path = data_dir / filename

    if path.exists():
        return load_item_popularity_counts(data_dir=data_dir, num_items=num_items, filename=filename)

    if not fallback_to_train:
        raise FileNotFoundError(f"Popularity file not found and fallback disabled: {path}")

    print(f"[Warning] {path} not found; computing popularity counts from train.txt.")
    return compute_popularity_counts_from_train(train_rows, num_items=num_items)


# ==========================================================
# Training
# ==========================================================


def train_one_epoch_pop_reweight(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    item_weights: torch.Tensor,
    grad_clip_norm: float,
    label_smoothing: float,
    normalize_batch_weight: bool,
    log_every: int = 100,
) -> Tuple[float, float, Dict[str, float]]:
    model.train()

    total_loss = 0.0
    total_examples = 0

    grad_norms: List[float] = []
    weight_means: List[float] = []
    weight_mins: List[float] = []
    weight_maxs: List[float] = []

    item_weights = item_weights.to(device=device, dtype=torch.float32)

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model.full_sort_scores(user_ids, sequences, lengths)

        loss, stats = popularity_weighted_ce_loss_with_stats(
            logits=logits,
            targets=targets,
            item_weights=item_weights,
            label_smoothing=float(label_smoothing),
            normalize_batch_weight=bool(normalize_batch_weight),
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            grad_norms.append(float(norm.detach().cpu().item()))

        optimizer.step()

        bs = int(targets.numel())
        total_loss += float(loss.detach().cpu().item()) * bs
        total_examples += bs

        weight_means.append(float(stats["target_weight_mean_raw"]))
        weight_mins.append(float(stats["target_weight_min_raw"]))
        weight_maxs.append(float(stats["target_weight_max_raw"]))

        if log_every > 0 and step % log_every == 0:
            avg = total_loss / max(total_examples, 1)
            print(
                f"  step={step:6d} "
                f"train_loss={avg:.6f} "
                f"target_w_mean={np.mean(weight_means):.4f}",
                flush=True,
            )

    avg_loss = total_loss / max(total_examples, 1)
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")

    epoch_stats = {
        "target_weight_mean_raw": float(np.mean(weight_means)) if weight_means else float("nan"),
        "target_weight_min_raw": float(np.min(weight_mins)) if weight_mins else float("nan"),
        "target_weight_max_raw": float(np.max(weight_maxs)) if weight_maxs else float("nan"),
    }

    return avg_loss, avg_grad_norm, epoch_stats


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    args = parse_args()

    raw_cfg = load_yaml(resolve_path(args.config))
    backbone = resolve_backbone(args, raw_cfg)
    cfg = apply_overrides(raw_cfg, args, backbone=backbone)

    dataset = args.dataset or cfg.get("default_dataset")
    if not dataset:
        raise ValueError("Dataset must be provided via --dataset or default_dataset in config")

    paths_cfg = cfg.get("paths", {})
    datasets_config = resolve_path(paths_cfg.get("datasets_config", "configs/datasets.yaml"))
    datasets_cfg = load_yaml(datasets_config) if datasets_config.exists() else {}

    data_dir = get_dataset_dir(str(dataset), datasets_cfg)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {data_dir}")

    seed = int(cfg.get("seed", 2026))
    set_seed(seed)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg.get("model", {})
    pop_cfg = cfg.get("pop_reweight", {})

    device_str = str(cfg.get("device", "cuda"))
    device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    batch_size = int(train_cfg.get("batch_size", 512))
    eval_batch_size = int(eval_cfg.get("batch_size", 512))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"

    train_rows = read_user_sequences(data_dir / "train.txt")
    val_rows = read_user_sequences(data_dir / "val.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")

    num_items = infer_num_items(data_dir)

    pop_counts = load_or_compute_popularity_counts(
        data_dir=data_dir,
        train_rows=train_rows,
        num_items=num_items,
        pop_cfg=pop_cfg,
    )

    item_weights = build_popularity_weights(
        pop_counts=pop_counts,
        num_items=num_items,
        scheme=str(pop_cfg.get("scheme", "inverse_log")),
        alpha=float(pop_cfg.get("alpha", 0.5)),
        min_weight=float(pop_cfg.get("min_weight", 0.5)),
        max_weight=float(pop_cfg.get("max_weight", 3.0)),
        normalize_to_mean=bool(pop_cfg.get("normalize_to_mean", True)),
        zero_padding=True,
    )

    pop_weight_stats = describe_popularity_weights(item_weights, pop_counts=pop_counts)

    train_ds = NextItemTrainDataset(train_rows, max_seq_len=max_seq_len)
    val_ds = EvalNextItemDataset(val_rows, max_seq_len=max_seq_len)
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)

    collate_fn = make_collate_fn(max_seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    seen_train = build_seen_items(train_rows)

    model = build_model(backbone, num_items=num_items, cfg=cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    method_name = method_name_from_backbone(backbone)
    run_name = run_name_from_backbone(backbone)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / str(dataset) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["dataset"] = str(dataset)
    resolved["method"] = method_name
    resolved["model_name"] = run_name
    resolved["data_dir"] = str(data_dir)
    resolved["num_items"] = int(num_items)
    resolved["num_train_samples"] = int(len(train_ds))
    resolved["num_val_samples"] = int(len(val_ds))
    resolved["num_test_samples"] = int(len(test_ds))
    resolved["model_arg"] = backbone
    resolved["backbone"] = backbone
    resolved["run_name"] = run_name
    resolved["run_id"] = run_id
    resolved["device_resolved"] = str(device)
    resolved["popularity_weight_stats"] = pop_weight_stats

    save_json(resolved, run_dir / "config_resolved.json")

    np.save(
        run_dir / "popularity_weights.npy",
        item_weights.detach().cpu().numpy().astype(np.float32),
    )

    print("========== PopReweight Training ==========")
    print(f"dataset:       {dataset}")
    print(f"method:        {method_name}")
    print(f"backbone:      {backbone}")
    print(f"run_name:      {run_name}")
    print(f"run_id:        {run_id}")
    print(f"data_dir:      {data_dir}")
    print(f"num_items:     {num_items}")
    print(f"train samples: {len(train_ds)}")
    print(f"val samples:   {len(val_ds)}")
    print(f"test samples:  {len(test_ds)}")
    print(f"parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"device:        {device}")
    print("pop weights:")
    print(json.dumps(pop_weight_stats, indent=2, ensure_ascii=False))

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    log_every = int(train_cfg.get("log_every", 100))
    normalize_batch_weight = bool(pop_cfg.get("normalize_batch_weight", True))

    metric_for_best = str(eval_cfg.get("metric_for_best", "ndcg@20")).lower()
    ks = [int(x) for x in eval_cfg.get("ks", [5, 10, 20])]
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))
    save_topk_npz = bool(eval_cfg.get("save_topk_npz", True))

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    logs: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, avg_grad_norm, train_stats = train_one_epoch_pop_reweight(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            item_weights=item_weights,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            normalize_batch_weight=normalize_batch_weight,
            log_every=log_every,
        )

        epoch_time = time.time() - start

        val_metric_value = float("nan")
        val_metrics: Dict[str, float] = {}

        if epoch % eval_every == 0:
            val_metrics = evaluate_full_sort(
                model=model,
                loader=val_loader,
                device=device,
                ks=ks,
                num_items=num_items,
                mask_seen_items=mask_seen_items,
                seen_items=seen_train,
                save_topk_path=None,
            )
            val_metric_value = get_metric(val_metrics, metric_for_best)

            improved = val_metric_value > best_metric + min_delta
            if improved:
                best_metric = val_metric_value
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "config": resolved,
                    },
                    run_dir / "best_model.pt",
                )
            else:
                bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "avg_grad_norm": avg_grad_norm,
            "target_weight_mean_raw": train_stats.get("target_weight_mean_raw", float("nan")),
            "target_weight_min_raw": train_stats.get("target_weight_min_raw", float("nan")),
            "target_weight_max_raw": train_stats.get("target_weight_max_raw", float("nan")),
            "val_metric": val_metric_value,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_time_sec": epoch_time,
        }

        for k_name, v in val_metrics.items():
            row[f"val/{k_name}"] = float(v)

        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"val_{metric_for_best}={val_metric_value:.6f} "
            f"best={best_metric:.6f}@{best_epoch} "
            f"target_w={row['target_weight_mean_raw']:.4f} "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

        if epoch % eval_every == 0 and bad_epochs >= patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break

    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        ckpt = safe_torch_load(best_path, map_location=device)
        state = unwrap_state_dict(ckpt)
        model.load_state_dict(state, strict=True)
    else:
        print("[Warning] best_model.pt was not saved; using last epoch model.")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epochs}, best_path)

    summary: Dict[str, object] = {
        "dataset": str(dataset),
        "method": method_name,
        "model_name": run_name,
        "run_id": run_id,
        "backbone": backbone,
        "best_metric_name": metric_for_best,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "num_items": int(num_items),
        "num_train_samples": int(len(train_ds)),
        "num_val_samples": int(len(val_ds)),
        "num_test_samples": int(len(test_ds)),
        "popularity_weight_stats": pop_weight_stats,
    }

    if bool(eval_cfg.get("run_test_after_training", True)):
        val_topk_path = run_dir / "topk_val.npz" if save_topk_npz else None
        test_topk_path = run_dir / "topk_test.npz" if save_topk_npz else None

        val_final = evaluate_full_sort(
            model=model,
            loader=val_loader,
            device=device,
            ks=ks,
            num_items=num_items,
            mask_seen_items=mask_seen_items,
            seen_items=seen_train,
            save_topk_path=val_topk_path,
        )

        test_final = evaluate_full_sort(
            model=model,
            loader=test_loader,
            device=device,
            ks=ks,
            num_items=num_items,
            mask_seen_items=mask_seen_items,
            seen_items=seen_train,
            save_topk_path=test_topk_path,
        )

        summary["val"] = val_final
        summary["test"] = test_final

    save_json(summary, run_dir / "metrics_summary.json")

    print("========== Finished ==========")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
