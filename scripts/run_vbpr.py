# -*- coding: utf-8 -*-
"""
scripts/run_vbpr.py

Train and evaluate VBPR baseline on this project's processed Amazon datasets.

VBPR is a non-sequential multimodal baseline. It uses:
    - user latent factors
    - item latent factors
    - fixed visual item features
    - learned visual projection
    - item bias and visual bias
    - BPR pairwise ranking loss

This script reuses the project utilities from scripts/run_id_backbone.py:
    - dataset path resolution
    - sequence reading
    - full-sort evaluation
    - seen-item masking
    - top-k npz saving

Typical usage:

    python scripts/run_vbpr.py \
      --dataset Video_Games \
      --config configs/vbpr_3090.yaml \
      --run_id vbpr_video

For Baby:

    python scripts/run_vbpr.py \
      --dataset Baby_Products \
      --config configs/vbpr_3090.yaml \
      --run_id vbpr_baby

After training, evaluate fairness:

    python scripts/eval_fairness.py \
      --dataset Video_Games \
      --run_dir results/Video_Games/vbpr/vbpr_video \
      --split test \
      --groups all \
      --append_global
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

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
    from model_vbpr import VBPR  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import src/model_vbpr.py. Make sure it exists.") from exc

try:
    from run_id_backbone import (  # type: ignore
        EvalNextItemDataset,
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
        "run_vbpr.py depends on scripts/run_id_backbone.py. "
        "Please make sure it exists and passes py_compile."
    ) from exc


# ==========================================================
# Dataset
# ==========================================================


def infer_num_users(*rows_list: Sequence[Tuple[int, Sequence[int]]]) -> int:
    max_uid = 0
    for rows in rows_list:
        for uid, _ in rows:
            max_uid = max(max_uid, int(uid))
    return max_uid


class VBPRTrainDataset(torch.utils.data.Dataset):
    """
    Pairwise BPR dataset.

    Each sample returns:
        user_id, positive_item, sampled_negative_item

    Positive items are all train interactions.
    Negative items are sampled uniformly from items not interacted by the user
    in train/val/test sets if all_user_items is provided; otherwise train only.
    """

    def __init__(
        self,
        train_rows: Sequence[Tuple[int, Sequence[int]]],
        num_items: int,
        all_user_items: Optional[Dict[int, set[int]]] = None,
        seed: int = 2026,
    ) -> None:
        self.num_items = int(num_items)
        self.seed = int(seed)

        self.samples: List[Tuple[int, int]] = []
        self.user_pos_train: Dict[int, set[int]] = {}

        for uid, seq in train_rows:
            uid = int(uid)
            pos_set = self.user_pos_train.setdefault(uid, set())
            for item in seq:
                item = int(item)
                if 0 < item <= self.num_items:
                    self.samples.append((uid, item))
                    pos_set.add(item)

        if not self.samples:
            raise ValueError("No positive train interactions found for VBPR.")

        if all_user_items is None:
            self.user_forbidden = self.user_pos_train
        else:
            self.user_forbidden = all_user_items

        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def sample_negative(self, uid: int) -> int:
        forbidden = self.user_forbidden.get(uid, set())

        # Fast rejection sampling. For very dense users, fallback to candidate list.
        for _ in range(100):
            neg = self._rng.randint(1, self.num_items)
            if neg not in forbidden:
                return neg

        candidates = [i for i in range(1, self.num_items + 1) if i not in forbidden]
        if not candidates:
            # Degenerate fallback. Should be extremely rare.
            return self._rng.randint(1, self.num_items)
        return self._rng.choice(candidates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, pos = self.samples[idx]
        neg = self.sample_negative(uid)

        return {
            "user_ids": torch.tensor(uid, dtype=torch.long),
            "pos_items": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(neg, dtype=torch.long),
        }


def vbpr_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "user_ids": torch.stack([x["user_ids"] for x in batch], dim=0),
        "pos_items": torch.stack([x["pos_items"] for x in batch], dim=0),
        "neg_items": torch.stack([x["neg_items"] for x in batch], dim=0),
    }


def build_all_user_items(
    *rows_list: Sequence[Tuple[int, Sequence[int]]],
    num_items: int,
) -> Dict[int, set[int]]:
    out: Dict[int, set[int]] = {}
    for rows in rows_list:
        for uid, seq in rows:
            uid = int(uid)
            s = out.setdefault(uid, set())
            for item in seq:
                item = int(item)
                if 0 < item <= num_items:
                    s.add(item)
    return out


# ==========================================================
# Config
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VBPR multimodal baseline")

    parser.add_argument("--config", type=str, default="configs/vbpr_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    # Model overrides.
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--visual_dim", type=int, default=None)
    parser.add_argument("--feature_file", type=str, default=None)
    parser.add_argument("--mask_file", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--no_normalize_features", action="store_true")
    parser.add_argument("--no_zero_missing_features", action="store_true")

    # Training overrides.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw", "sgd"])
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)

    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("optimizer", {})

    if args.latent_dim is not None:
        cfg["model"]["latent_dim"] = args.latent_dim
    if args.visual_dim is not None:
        cfg["model"]["visual_dim"] = args.visual_dim
    if args.feature_file is not None:
        cfg["model"]["feature_file"] = args.feature_file
    if args.mask_file is not None:
        cfg["model"]["mask_file"] = args.mask_file
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.no_normalize_features:
        cfg["model"]["normalize_features"] = False
    if args.no_zero_missing_features:
        cfg["model"]["zero_missing_features"] = False

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
    if args.optimizer is not None:
        cfg["optimizer"]["name"] = args.optimizer
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.grad_clip_norm is not None:
        cfg["train"]["grad_clip_norm"] = args.grad_clip_norm

    if args.cpu:
        cfg["device"] = "cpu"

    return cfg


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    opt_cfg = cfg.get("optimizer", {})

    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-6))

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1.0e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1.0e-8))
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.0))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    raise ValueError(f"Unsupported optimizer={name!r}")


# ==========================================================
# Training
# ==========================================================


def train_one_epoch(
    model: VBPR,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 5.0,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    grad_norms: List[float] = []

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        pos_items = batch["pos_items"].to(device, non_blocking=True)
        neg_items = batch["neg_items"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        margin = model(user_ids, pos_items, neg_items)
        loss = -torch.nn.functional.logsigmoid(margin).mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite VBPR loss at step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            grad_norms.append(float(norm.detach().cpu().item()))

        optimizer.step()

        bs = int(user_ids.numel())
        total_loss += float(loss.detach().cpu().item()) * bs
        total_examples += bs
        total_correct += int((margin.detach() > 0).sum().cpu().item())

        if log_every > 0 and step % log_every == 0:
            avg_loss = total_loss / max(total_examples, 1)
            acc = total_correct / max(total_examples, 1)
            print(f"  step={step:6d} train_loss={avg_loss:.6f} pair_acc={acc:.4f}", flush=True)

    return {
        "train_loss": total_loss / max(total_examples, 1),
        "pair_acc": total_correct / max(total_examples, 1),
        "avg_grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
    }


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    cfg = apply_overrides(cfg, args)

    dataset = args.dataset or cfg.get("default_dataset")
    if not dataset:
        raise ValueError("Dataset must be provided by --dataset or default_dataset in config.")

    dataset = str(dataset)

    paths_cfg = cfg.get("paths", {})
    datasets_config = resolve_path(paths_cfg.get("datasets_config", "configs/datasets.yaml"))
    datasets_cfg = load_yaml(datasets_config) if datasets_config.exists() else {}
    data_dir = get_dataset_dir(dataset, datasets_cfg)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {data_dir}")

    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg.get("model", {})

    device_str = str(cfg.get("device", "cuda"))
    device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_rows = read_user_sequences(data_dir / "train.txt")
    val_rows = read_user_sequences(data_dir / "val.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")

    num_items = infer_num_items(data_dir)
    num_users = infer_num_users(train_rows, val_rows, test_rows)

    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    batch_size = int(train_cfg.get("batch_size", 2048))
    eval_batch_size = int(eval_cfg.get("batch_size", 512))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"

    all_user_items = build_all_user_items(train_rows, val_rows, test_rows, num_items=num_items)

    train_ds = VBPRTrainDataset(
        train_rows=train_rows,
        num_items=num_items,
        all_user_items=all_user_items,
        seed=seed,
    )

    val_ds = EvalNextItemDataset(val_rows, max_seq_len=max_seq_len)
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vbpr_collate_fn,
        drop_last=False,
    )

    eval_collate_fn = make_collate_fn(max_seq_len)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eval_collate_fn,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eval_collate_fn,
        drop_last=False,
    )

    model = VBPR(
        num_users=num_users,
        num_items=num_items,
        data_dir=data_dir,
        feature_file=str(model_cfg.get("feature_file", "vision_features.npy")),
        mask_file=model_cfg.get("mask_file", "vision_mask.npy"),
        latent_dim=int(model_cfg.get("latent_dim", 64)),
        visual_dim=int(model_cfg.get("visual_dim", 64)),
        normalize_features=bool(model_cfg.get("normalize_features", True)),
        zero_missing_features=bool(model_cfg.get("zero_missing_features", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        init_std=float(model_cfg.get("init_std", 0.01)),
    ).to(device)

    optimizer = build_optimizer(model, cfg)

    run_name = str(cfg.get("run_name", "vbpr"))
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / dataset / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["dataset"] = dataset
    resolved["method"] = "VBPR"
    resolved["model_name"] = run_name
    resolved["run_id"] = run_id
    resolved["data_dir"] = str(data_dir)
    resolved["num_users"] = int(num_users)
    resolved["num_items"] = int(num_items)
    resolved["num_train_samples"] = int(len(train_ds))
    resolved["num_val_samples"] = int(len(val_ds))
    resolved["num_test_samples"] = int(len(test_ds))
    resolved["device_resolved"] = str(device)
    save_json(resolved, run_dir / "config_resolved.json")

    print("========== VBPR Training ==========")
    print(f"dataset:       {dataset}")
    print(f"method:        VBPR")
    print(f"run_name:      {run_name}")
    print(f"run_id:        {run_id}")
    print(f"data_dir:      {data_dir}")
    print(f"num_users:     {num_users}")
    print(f"num_items:     {num_items}")
    print(f"train samples: {len(train_ds)}")
    print(f"val samples:   {len(val_ds)}")
    print(f"test samples:  {len(test_ds)}")
    print(f"parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"device:        {device}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    log_every = int(train_cfg.get("log_every", 100))

    metric_for_best = str(eval_cfg.get("metric_for_best", "ndcg@20")).lower()
    ks = [int(x) for x in eval_cfg.get("ks", [5, 10, 20])]
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))
    save_topk_npz = bool(eval_cfg.get("save_topk_npz", True))

    seen_train = build_seen_items(train_rows)

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
            grad_clip_norm=grad_clip_norm,
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
            "train_loss": float(train_stats["train_loss"]),
            "pair_acc": float(train_stats["pair_acc"]),
            "avg_grad_norm": float(train_stats["avg_grad_norm"]),
            "val_metric": float(val_metric_value),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "epoch_time_sec": float(epoch_time),
        }

        for k_name, v in val_metrics.items():
            row[f"val/{k_name}"] = float(v)

        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"pair_acc={row['pair_acc']:.4f} "
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
        print("[Warning] best_model.pt was not saved; using last epoch model.")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epochs}, best_path)

    summary: Dict[str, object] = {
        "dataset": dataset,
        "method": "VBPR",
        "model_name": run_name,
        "run_id": run_id,
        "best_metric_name": metric_for_best,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_train_samples": int(len(train_ds)),
        "num_val_samples": int(len(val_ds)),
        "num_test_samples": int(len(test_ds)),
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
