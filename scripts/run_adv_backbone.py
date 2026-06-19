# -*- coding: utf-8 -*-
"""
scripts/run_adv_backbone.py

Train adversarial ID-backbone fairness baselines:

    Adv-SASRec
    Adv-GRU4Rec
    Adv-BERT4Rec

Example:

python scripts/run_adv_backbone.py \
  --dataset Video_Games \
  --config configs/adv_sasrec_3090.yaml \
  --backbone sasrec \
  --init_backbone_checkpoint results/Video_Games/sasrec_id/20260515_194126/best_model.pt \
  --run_id adv_sasrec_video

python scripts/run_adv_backbone.py \
  --dataset Video_Games \
  --config configs/adv_gru4rec_3090.yaml \
  --backbone gru4rec \
  --init_backbone_checkpoint results/Video_Games/gru4rec_id/<RUN_ID>/best_model.pt \
  --run_id adv_gru4rec_video

python scripts/run_adv_backbone.py \
  --dataset Video_Games \
  --config configs/adv_bert4rec_3090.yaml \
  --backbone bert4rec \
  --init_backbone_checkpoint results/Video_Games/bert4rec_id/<RUN_ID>/best_model.pt \
  --run_id adv_bert4rec_video
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in [str(SRC_DIR), str(SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from run_id_backbone import (  # type: ignore
        EvalNextItemDataset,
        NextItemTrainDataset,
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
except ImportError as exc:
    raise ImportError(
        "scripts/run_adv_backbone.py depends on scripts/run_id_backbone.py."
    ) from exc

try:
    from run_fare import load_training_group_labels, make_class_weight  # type: ignore
except ImportError as exc:
    raise ImportError(
        "scripts/run_adv_backbone.py reuses group-label loading helpers from scripts/run_fare.py."
    ) from exc

try:
    from model_adv_backbone import AdvIDBackbone  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import AdvIDBackbone from src/model_adv_backbone.py."
    ) from exc


FULL_GROUPS = [
    "popularity_group",
    "text_quality_group",
    "vision_quality_group",
    "category_proxy_group",
    "brand_store_proxy_group",
    "multimodal_cluster_proxy_group",
]


# ==========================================================
# CLI / config
# ==========================================================


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_dict_arg(value: Optional[str]) -> Dict[str, float]:
    if value is None:
        return {}
    value = str(value).strip()
    if not value:
        return {}

    out: Dict[str, float] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid mapping item {part!r}; expected name:value")
        k, v = part.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def method_name_from_backbone(backbone: str) -> str:
    b = str(backbone).lower()
    if b == "sasrec":
        return "Adv-SASRec"
    if b == "gru4rec":
        return "Adv-GRU4Rec"
    if b == "bert4rec":
        return "Adv-BERT4Rec"
    return f"Adv-{backbone}"


def run_name_from_backbone(backbone: str) -> str:
    b = str(backbone).lower()
    if b == "sasrec":
        return "adv_sasrec"
    if b == "gru4rec":
        return "adv_gru4rec"
    if b == "bert4rec":
        return "adv_bert4rec"
    return f"adv_{b}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adv-SASRec / Adv-GRU4Rec / Adv-BERT4Rec")

    parser.add_argument("--config", type=str, default="configs/adv_sasrec_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--method_name", type=str, default=None)

    parser.add_argument("--backbone", type=str, default=None, choices=["sasrec", "gru4rec", "bert4rec"])
    parser.add_argument("--init_backbone_checkpoint", type=str, default=None)
    parser.add_argument("--init_sasrec_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_id_backbone", action="store_true")

    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--group_loss_weights", type=str, default=None)
    parser.add_argument("--adv_loss_weight", type=float, default=None)
    parser.add_argument("--grl_lambda", type=float, default=None)
    parser.add_argument("--classifier_hidden_dim", type=int, default=None)
    parser.add_argument("--aux_dropout", type=float, default=None)
    parser.add_argument("--min_group_count", type=int, default=None)
    parser.add_argument("--drop_rare_classes", action="store_true")
    parser.add_argument("--use_class_balanced_aux_loss", action="store_true")
    parser.add_argument("--normalize_user_repr", action="store_true")

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
    parser.add_argument("--no_fairness_eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("adv", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})

    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone

    for attr, section, key in [
        ("epochs", "train", "epochs"),
        ("batch_size", "train", "batch_size"),
        ("eval_batch_size", "eval", "batch_size"),
        ("lr", "train", "learning_rate"),
        ("weight_decay", "train", "weight_decay"),
        ("max_seq_len", "model", "max_seq_len"),
        ("hidden_size", "model", "hidden_size"),
        ("num_layers", "model", "num_layers"),
        ("num_heads", "model", "num_heads"),
        ("dropout", "model", "dropout"),
        ("patience", "train", "patience"),
        ("eval_every", "train", "eval_every"),
        ("num_workers", "train", "num_workers"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            cfg[section][key] = value

    if args.groups is not None:
        groups = parse_list_arg(args.groups)
        if groups:
            cfg["adv"]["groups"] = groups

    if args.group_loss_weights is not None:
        cfg["adv"]["group_loss_weights"] = parse_dict_arg(args.group_loss_weights)
    if args.adv_loss_weight is not None:
        cfg["adv"]["adv_loss_weight"] = args.adv_loss_weight
    if args.grl_lambda is not None:
        cfg["adv"]["grl_lambda"] = args.grl_lambda
    if args.classifier_hidden_dim is not None:
        cfg["adv"]["classifier_hidden_dim"] = args.classifier_hidden_dim
    if args.aux_dropout is not None:
        cfg["adv"]["aux_dropout"] = args.aux_dropout
    if args.min_group_count is not None:
        cfg["adv"]["min_group_count"] = args.min_group_count
    if args.drop_rare_classes:
        cfg["adv"]["drop_rare_classes"] = True
    if args.use_class_balanced_aux_loss:
        cfg["adv"]["use_class_balanced_aux_loss"] = True
    if args.normalize_user_repr:
        cfg["adv"]["normalize_user_repr"] = True

    if args.no_fairness_eval:
        cfg["eval"]["run_fairness_eval"] = False
    if args.cpu:
        cfg["device"] = "cpu"

    return cfg


# ==========================================================
# Loss
# ==========================================================


def adv_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = labels >= 0
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)

    weight = class_weight.to(device=logits.device, dtype=logits.dtype) if class_weight is not None else None
    return F.cross_entropy(logits[valid], labels[valid].long(), weight=weight)


# ==========================================================
# Model helpers
# ==========================================================


def build_model(
    num_items: int,
    group_num_classes: Dict[str, int],
    cfg: Dict[str, Any],
) -> AdvIDBackbone:
    model_cfg = cfg.get("model", {})
    adv_cfg = cfg.get("adv", {})
    backbone = str(model_cfg.get("backbone", "sasrec")).lower()

    return AdvIDBackbone(
        num_items=num_items,
        group_num_classes=group_num_classes,
        backbone_type=backbone,
        max_seq_len=int(model_cfg.get("max_seq_len", 50)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        activation=str(model_cfg.get("activation", "gelu")),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-12)),
        tie_output_embedding=bool(model_cfg.get("tie_output_embedding", True)),
        classifier_hidden_dim=int(adv_cfg.get("classifier_hidden_dim", 128)),
        aux_dropout=float(adv_cfg.get("aux_dropout", 0.1)),
        normalize_user_repr=bool(adv_cfg.get("normalize_user_repr", False)),
    )


def load_initial_backbone(model: AdvIDBackbone, checkpoint_path: Optional[str], device: torch.device) -> None:
    if not checkpoint_path:
        return

    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Initial backbone checkpoint not found: {path}")

    ckpt = safe_torch_load(path, map_location=device)
    state = unwrap_state_dict(ckpt)

    missing, unexpected = model.load_backbone_state_dict(state, strict=False)

    print(f"Loaded backbone checkpoint: {path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")


# ==========================================================
# Training
# ==========================================================


def train_one_epoch(
    model: AdvIDBackbone,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    group_labels: Dict[str, torch.Tensor],
    class_weights: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.train()

    train_cfg = cfg.get("train", {})
    adv_cfg = cfg.get("adv", {})

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    log_every = int(train_cfg.get("log_every", 100))

    adv_loss_weight = float(adv_cfg.get("adv_loss_weight", 0.05))
    grl_lambda = float(adv_cfg.get("grl_lambda", 1.0))
    group_loss_weights = dict(adv_cfg.get("group_loss_weights", {}) or {})
    use_class_balanced = bool(adv_cfg.get("use_class_balanced_aux_loss", True))

    label_tensors = {g: group_labels[g].to(device=device, non_blocking=True) for g in group_labels.keys()}

    total_examples = 0
    sums = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "adv_loss": 0.0,
    }
    grad_norms: List[float] = []

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model.full_sort_scores(user_ids, sequences, lengths)
        rec_loss = F.cross_entropy(logits, targets.long(), label_smoothing=label_smoothing)

        user_repr = model.encode_sequence(sequences, lengths)
        adv_logits = model.group_logits(user_repr, grl_lambda=grl_lambda)

        adv_loss_total = logits.new_tensor(0.0)

        for group in group_labels.keys():
            labels = label_tensors[group][targets]
            class_weight = class_weights.get(group) if use_class_balanced else None
            w = float(group_loss_weights.get(group, 1.0))
            adv_loss_total = adv_loss_total + w * adv_ce_loss(adv_logits[group], labels, class_weight)

        loss = rec_loss + adv_loss_weight * adv_loss_total

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_norms.append(float(norm.detach().cpu().item()))

        optimizer.step()

        batch_size = int(targets.numel())
        total_examples += batch_size

        sums["loss"] += float(loss.detach().cpu().item()) * batch_size
        sums["rec_loss"] += float(rec_loss.detach().cpu().item()) * batch_size
        sums["adv_loss"] += float(adv_loss_total.detach().cpu().item()) * batch_size

        if log_every > 0 and step % log_every == 0:
            denom = max(total_examples, 1)
            print(
                f"  step={step:6d} "
                f"loss={sums['loss'] / denom:.6f} "
                f"rec={sums['rec_loss'] / denom:.6f} "
                f"adv={sums['adv_loss'] / denom:.6f}",
                flush=True,
            )

    denom = max(total_examples, 1)
    out = {k: v / denom for k, v in sums.items()}
    out["avg_grad_norm"] = float(np.mean(grad_norms)) if grad_norms else float("nan")
    return out


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    cfg = apply_overrides(cfg, args)

    dataset = args.dataset or cfg.get("default_dataset")
    if not dataset:
        raise ValueError("Dataset must be provided via --dataset or default_dataset in config.")

    paths_cfg = cfg.get("paths", {})
    datasets_config = resolve_path(paths_cfg.get("datasets_config", "configs/datasets.yaml"))
    datasets_cfg = load_yaml(datasets_config) if datasets_config.exists() else {}

    data_dir = get_dataset_dir(str(dataset), datasets_cfg)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {data_dir}")

    seed = int(cfg.get("seed", 2026))
    set_seed(seed)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    adv_cfg = cfg.get("adv", {})

    backbone = str(model_cfg.get("backbone", "sasrec")).lower()
    method_name = args.method_name or str(cfg.get("method_name", method_name_from_backbone(backbone)))
    run_name = str(cfg.get("run_name", run_name_from_backbone(backbone)))

    device_str = str(cfg.get("device", "cuda"))
    device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    batch_size = int(train_cfg.get("batch_size", 256))
    eval_batch_size = int(eval_cfg.get("batch_size", 512))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"

    train_rows = read_user_sequences(data_dir / "train.txt")
    val_rows = read_user_sequences(data_dir / "val.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")
    num_items = infer_num_items(data_dir)

    groups = adv_cfg.get("groups") or FULL_GROUPS
    if isinstance(groups, str):
        groups = parse_list_arg(groups) or FULL_GROUPS
    groups = list(groups)

    group_labels, group_num_classes, group_info = load_training_group_labels(
        data_dir=data_dir,
        requested_groups=groups,
        num_items=num_items,
        min_group_count=int(adv_cfg.get("min_group_count", 5)),
        drop_rare_classes=bool(adv_cfg.get("drop_rare_classes", True)),
    )

    class_weights = {
        g: make_class_weight(labels, group_num_classes[g])
        for g, labels in group_labels.items()
    }

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

    model = build_model(
        num_items=num_items,
        group_num_classes=group_num_classes,
        cfg=cfg,
    ).to(device)

    init_ckpt = args.init_backbone_checkpoint or args.init_sasrec_checkpoint or cfg.get("init_backbone_checkpoint")
    load_initial_backbone(model, init_ckpt, device=device)

    if bool(args.freeze_id_backbone or adv_cfg.get("freeze_id_backbone", False)):
        model.freeze_backbone()
        print(f"Frozen ID backbone: {backbone}")

    optimizer = build_optimizer(model, cfg)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / str(dataset) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved.update(
        {
            "dataset": str(dataset),
            "data_dir": str(data_dir),
            "num_items": int(num_items),
            "num_train_samples": int(len(train_ds)),
            "num_val_samples": int(len(val_ds)),
            "num_test_samples": int(len(test_ds)),
            "method": method_name,
            "model_name": run_name,
            "run_id": run_id,
            "backbone": backbone,
            "init_backbone_checkpoint": str(init_ckpt) if init_ckpt else None,
            "device_resolved": str(device),
            "fairness_groups_used": list(group_labels.keys()),
            "group_num_classes": group_num_classes,
            "group_info": group_info,
        }
    )
    save_json(resolved, run_dir / "config_resolved.json")

    print("========== Adv ID-Backbone Training ==========")
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
    print(f"groups:        {list(group_labels.keys())}")
    print(f"group classes: {group_num_classes}")
    print(f"parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"trainable:     {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"device:        {device}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))

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

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            group_labels=group_labels,
            class_weights=class_weights,
            cfg=cfg,
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

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_stats["loss"],
            "rec_loss": train_stats["rec_loss"],
            "adv_loss": train_stats["adv_loss"],
            "avg_grad_norm": train_stats["avg_grad_norm"],
            "val_metric": val_metric_value,
            "best_metric": best_metric,
            "best_epoch": float(best_epoch),
            "epoch_time_sec": epoch_time,
        }
        for k_name, v in val_metrics.items():
            row[f"val/{k_name}"] = float(v)

        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"rec={row['rec_loss']:.6f} "
            f"adv={row['adv_loss']:.4f} "
            f"val_{metric_for_best}={val_metric_value:.6f} "
            f"best={best_metric:.6f}@{best_epoch} "
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
        print("[Warning] best_model.pt was not saved; saving last epoch model.")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epochs}, best_path)

    summary: Dict[str, Any] = {
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
        "fairness_groups_used": list(group_labels.keys()),
        "group_num_classes": group_num_classes,
        "init_backbone_checkpoint": str(init_ckpt) if init_ckpt else None,
    }

    if bool(eval_cfg.get("run_test_after_training", True)):
        val_topk_path = run_dir / "topk_val.npz" if save_topk_npz else None
        test_topk_path = run_dir / "topk_test.npz" if save_topk_npz else None

        summary["val"] = evaluate_full_sort(
            model=model,
            loader=val_loader,
            device=device,
            ks=ks,
            num_items=num_items,
            mask_seen_items=mask_seen_items,
            seen_items=seen_train,
            save_topk_path=val_topk_path,
        )
        summary["test"] = evaluate_full_sort(
            model=model,
            loader=test_loader,
            device=device,
            ks=ks,
            num_items=num_items,
            mask_seen_items=mask_seen_items,
            seen_items=seen_train,
            save_topk_path=test_topk_path,
        )

    save_json(summary, run_dir / "metrics_summary.json")

    print("========== Finished ==========")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
