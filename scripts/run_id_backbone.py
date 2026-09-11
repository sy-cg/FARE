# -*- coding: utf-8 -*-
"""
scripts/run_id_backbone.py

Unified runner for ID-only sequential backbones:
- SASRec-ID
- GRU4Rec-ID
- BERT4Rec-style ID

This script is intentionally self-contained and compatible with the existing SDR
project layout:

SDR/
├── data/Processed_<Dataset>/train.txt val.txt test.txt
├── configs/datasets.yaml
├── configs/sasrec_3090.yaml / gru4rec_3090.yaml / bert4rec_3090.yaml
├── src/model_sasrec.py
├── src/model_gru4rec.py
├── src/model_bert4rec.py
└── results/<Dataset>/<run_name>/<run_id>/

Expected txt format:
    user_id item_1 item_2 ... item_n

Training samples are generated from train.txt by next-item expansion:
    prefix = item_1 ... item_{t-1}, target = item_t
Validation/test use the last item in val.txt/test.txt as target.

The runner saves:
- best_model.pt
- train_log.csv
- metrics_summary.json
- topk_val.npz / topk_test.npz when enabled

Usage examples:
    python scripts/run_id_backbone.py \
      --model gru4rec \
      --dataset Video_Games \
      --config configs/gru4rec_3090.yaml

    python scripts/run_id_backbone.py \
      --model bert4rec \
      --dataset Video_Games \
      --config configs/bert4rec_3090.yaml \
      --batch_size 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from model_sasrec import SASRecID  # type: ignore
    from model_gru4rec import GRU4RecID  # type: ignore
    from model_bert4rec import BERT4RecID  # type: ignore
    from io_utils import safe_torch_load  # type: ignore
    from eval_protocol import evaluation_history_items  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import backbone models or evaluation helpers. Make sure "
        "src/model_sasrec.py, src/model_gru4rec.py, src/model_bert4rec.py, "
        "src/io_utils.py, and src/eval_protocol.py exist."
    ) from exc


# ==========================================================
# Basic utilities
# ==========================================================


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def deep_update(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


# ==========================================================
# Dataset path resolution
# ==========================================================


def get_dataset_dir(dataset: str, cfg: Dict) -> Path:
    """Resolve processed dataset directory from configs/datasets.yaml.

    Supported formats:
    1) datasets:
         Video_Games:
           data_dir: data/Processed_Video_Games
    2) Video_Games:
         data_dir: data/Processed_Video_Games
    3) fallback: data/Processed_<Dataset>
    """
    candidates = []
    datasets_block = cfg.get("datasets")
    if isinstance(datasets_block, dict) and isinstance(datasets_block.get(dataset), dict):
        d = datasets_block[dataset]
        candidates.extend([d.get("data_dir"), d.get("path"), d.get("processed_dir"), d.get("dir")])

    if isinstance(cfg.get(dataset), dict):
        d = cfg[dataset]
        candidates.extend([d.get("data_dir"), d.get("path"), d.get("processed_dir"), d.get("dir")])

    for c in candidates:
        if c:
            p = resolve_path(str(c))
            if p.exists():
                return p
            return p

    return PROJECT_ROOT / "data" / f"Processed_{dataset}"


def infer_num_items(data_dir: Path) -> int:
    item2id_path = data_dir / "item2id.json"
    if item2id_path.exists():
        with open(item2id_path, "r", encoding="utf-8") as f:
            item2id = json.load(f)
        return int(max(int(v) for v in item2id.values()))

    max_item = 0
    for name in ["train.txt", "val.txt", "test.txt"]:
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    vals = [int(x) for x in parts[1:]]
                    if vals:
                        max_item = max(max_item, max(vals))
    if max_item <= 0:
        raise ValueError(f"Could not infer num_items from {data_dir}")
    return max_item


# ==========================================================
# Data loading
# ==========================================================


def read_user_sequences(path: Path) -> List[Tuple[int, List[int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    rows: List[Tuple[int, List[int]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = [int(x) for x in line.split()]
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {path}:{line_no}: {line[:120]}") from exc
            if len(parts) < 2:
                continue
            user_id, seq = parts[0], parts[1:]
            if seq:
                rows.append((user_id, seq))
    return rows


class NextItemTrainDataset(Dataset):
    """Expanded next-item training samples from train.txt."""

    def __init__(self, rows: Sequence[Tuple[int, List[int]]], max_seq_len: int) -> None:
        self.rows = rows
        self.max_seq_len = int(max_seq_len)
        self.index: List[Tuple[int, int]] = []
        for row_idx, (_, seq) in enumerate(rows):
            # Need at least one prefix item and one target item.
            for target_pos in range(1, len(seq)):
                self.index.append((row_idx, target_pos))
        if not self.index:
            raise ValueError("No train samples were generated. Check train.txt sequence lengths.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int]:
        row_idx, target_pos = self.index[idx]
        user_id, seq = self.rows[row_idx]
        prefix = seq[:target_pos]
        target = seq[target_pos]
        return user_id, prefix, target


class EvalNextItemDataset(Dataset):
    """One next-item evaluation sample per user from val.txt/test.txt."""

    def __init__(self, rows: Sequence[Tuple[int, List[int]]], max_seq_len: int) -> None:
        self.max_seq_len = int(max_seq_len)
        self.samples: List[Tuple[int, List[int], int]] = []
        for user_id, seq in rows:
            if len(seq) < 2:
                continue
            self.samples.append((user_id, seq[:-1], seq[-1]))
        if not self.samples:
            raise ValueError("No eval samples were generated. Check val/test file sequence lengths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int]:
        return self.samples[idx]


def make_collate_fn(max_seq_len: int):
    max_seq_len = int(max_seq_len)

    def collate(batch: Sequence[Tuple[int, List[int], int]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        users = torch.zeros(batch_size, dtype=torch.long)
        seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        targets = torch.zeros(batch_size, dtype=torch.long)

        for i, (u, prefix, target) in enumerate(batch):
            prefix = list(prefix)[-max_seq_len:]
            l = len(prefix)
            users[i] = int(u)
            if l > 0:
                seqs[i, :l] = torch.tensor(prefix, dtype=torch.long)
            lengths[i] = max(l, 1)
            targets[i] = int(target)

        return {
            "user_ids": users,
            "sequences": seqs,
            "lengths": lengths,
            "targets": targets,
        }

    return collate


def build_seen_items(rows: Sequence[Tuple[int, List[int]]]) -> Dict[int, set[int]]:
    seen: Dict[int, set[int]] = {}
    for u, seq in rows:
        seen.setdefault(int(u), set()).update(int(x) for x in seq if int(x) > 0)
    return seen


# ==========================================================
# Metrics / evaluation
# ==========================================================


def compute_ranking_metrics(ranks: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    ranks = np.asarray(ranks, dtype=np.int64)
    n = max(len(ranks), 1)
    for k in ks:
        hit_mask = ranks <= k
        metrics[f"hit@{k}"] = float(hit_mask.mean())
        metrics[f"recall@{k}"] = float(hit_mask.mean())
        ndcg = np.where(hit_mask, 1.0 / np.log2(ranks + 1.0), 0.0)
        metrics[f"ndcg@{k}"] = float(ndcg.mean())
        rr = np.where(hit_mask, 1.0 / ranks, 0.0)
        metrics[f"mrr@{k}"] = float(rr.mean())
    return metrics


@torch.no_grad()
def evaluate_full_sort(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ks: Sequence[int],
    num_items: int,
    mask_seen_items: bool = True,
    seen_items: Optional[Dict[int, set[int]]] = None,
    save_topk_path: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    max_k = int(max(ks))
    ranks: List[int] = []
    topk_items_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    users_all: List[np.ndarray] = []

    for batch in loader:
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        scores = model.full_sort_scores(user_ids, sequences, lengths)
        scores[:, 0] = -1e30

        if mask_seen_items:
            # Mask the actual per-example prefix plus any supplied older
            # history. Test prefixes include the validation interaction, so
            # using train-only history here would produce protocol leakage.
            scores = scores.clone()
            user_cpu = user_ids.detach().cpu().tolist()
            target_cpu = targets.detach().cpu().tolist()
            sequence_cpu = sequences.detach().cpu().tolist()
            length_cpu = lengths.detach().cpu().tolist()
            for row_idx, (u, tgt, seq, length) in enumerate(
                zip(user_cpu, target_cpu, sequence_cpu, length_cpu)
            ):
                prefix = seq[: max(int(length), 0)]
                history = evaluation_history_items(
                    user_id=int(u),
                    prefix=prefix,
                    target=int(tgt),
                    seen_items=seen_items,
                    num_items=num_items,
                )
                for item in history:
                    scores[row_idx, item] = -1e30

        target_scores = scores.gather(1, targets.view(-1, 1))
        # Rank is 1 + number of candidates with strictly larger score.
        batch_ranks = (scores > target_scores).sum(dim=1) + 1
        ranks.extend(batch_ranks.detach().cpu().numpy().astype(np.int64).tolist())

        if save_topk_path is not None:
            topk = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu().numpy().astype(np.int64)
            topk_items_all.append(topk)
            targets_all.append(targets.detach().cpu().numpy().astype(np.int64))
            users_all.append(user_ids.detach().cpu().numpy().astype(np.int64))

    metrics = compute_ranking_metrics(np.asarray(ranks, dtype=np.int64), ks=ks)

    if save_topk_path is not None and topk_items_all:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_topk_path,
            user_ids=np.concatenate(users_all, axis=0),
            targets=np.concatenate(targets_all, axis=0),
            topk_items=np.concatenate(topk_items_all, axis=0),
            ranks=np.asarray(ranks, dtype=np.int64),
        )

    return metrics


# ==========================================================
# Model / optimizer construction
# ==========================================================


def build_model(model_name: str, num_items: int, cfg: Dict) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_name.lower()

    common = dict(
        num_items=num_items,
        max_seq_len=int(model_cfg.get("max_seq_len", 50)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        tie_output_embedding=bool(model_cfg.get("tie_output_embedding", True)),
    )

    if model_name in {"sasrec", "sasrec_id"}:
        return SASRecID(
            **common,
            num_heads=int(model_cfg.get("num_heads", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
            layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-12)),
        )

    if model_name in {"gru4rec", "gru4rec_id", "gru"}:
        return GRU4RecID(**common)

    if model_name in {"bert4rec", "bert4rec_id", "bert"}:
        return BERT4RecID(
            **common,
            num_heads=int(model_cfg.get("num_heads", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
            layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-12)),
            use_cls_token=bool(model_cfg.get("use_cls_token", False)),
        )

    raise ValueError(f"Unknown --model {model_name!r}; choose sasrec, gru4rec, bert4rec")


def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    raise ValueError(f"Unsupported optimizer: {name}")


def get_metric(metrics: Dict[str, float], metric_name: str) -> float:
    key = metric_name.lower()
    if key in metrics:
        return float(metrics[key])
    # Accept forms like ndcg@20 or NDCG@20.
    for k, v in metrics.items():
        if k.lower() == key:
            return float(v)
    raise KeyError(f"Metric {metric_name!r} not found in metrics: {sorted(metrics.keys())}")


# ==========================================================
# Training
# ==========================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    label_smoothing: float = 0.0,
    log_every: int = 100,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    grad_norms: List[float] = []

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model.full_sort_scores(user_ids, sequences, lengths)
        loss = F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step={step}: {loss.item()}")

        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            grad_norms.append(float(norm.detach().cpu().item()))
        optimizer.step()

        batch_size = int(targets.numel())
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_examples += batch_size

        if log_every > 0 and step % log_every == 0:
            avg = total_loss / max(total_examples, 1)
            print(f"  step={step:6d} train_loss={avg:.6f}", flush=True)

    avg_loss = total_loss / max(total_examples, 1)
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")
    return avg_loss, avg_grad_norm


# ==========================================================
# CLI / main
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ID-only sequential backbones")
    parser.add_argument("--model", type=str, required=True, choices=["sasrec", "gru4rec", "bert4rec"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

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
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(cfg)
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
    if args.cpu:
        cfg["device"] = "cpu"
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    cfg = apply_overrides(cfg, args)

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

    train_ds = NextItemTrainDataset(train_rows, max_seq_len=max_seq_len)
    val_ds = EvalNextItemDataset(val_rows, max_seq_len=max_seq_len)
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)
    collate_fn = make_collate_fn(max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    seen_train = build_seen_items(train_rows)

    model = build_model(args.model, num_items=num_items, cfg=cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    run_name = str(cfg.get("run_name", f"{args.model}_id"))
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = now_timestamp()

    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / str(dataset) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["dataset"] = str(dataset)
    resolved["data_dir"] = str(data_dir)
    resolved["num_items"] = int(num_items)
    resolved["num_train_samples"] = int(len(train_ds))
    resolved["num_val_samples"] = int(len(val_ds))
    resolved["num_test_samples"] = int(len(test_ds))
    resolved["model_arg"] = args.model
    resolved["run_name"] = run_name
    resolved["run_id"] = run_id
    resolved["device_resolved"] = str(device)
    save_json(resolved, run_dir / "config_resolved.json")

    print("========== ID Backbone Training ==========")
    print(f"dataset:      {dataset}")
    print(f"model:        {args.model}")
    print(f"run_name:     {run_name}")
    print(f"run_id:       {run_id}")
    print(f"data_dir:     {data_dir}")
    print(f"num_items:    {num_items}")
    print(f"train samples:{len(train_ds)}")
    print(f"val samples:  {len(val_ds)}")
    print(f"test samples: {len(test_ds)}")
    print(f"parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"device:       {device}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    log_every = int(train_cfg.get("log_every", 100))
    metric_for_best = str(eval_cfg.get("metric_for_best", "ndcg@20")).lower()
    ks = [int(x) for x in eval_cfg.get("ks", [5, 10, 20])]
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))
    save_topk_npz = bool(eval_cfg.get("save_topk_npz", True))

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    logs: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, avg_grad_norm = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
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
        else:
            improved = False

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "avg_grad_norm": avg_grad_norm,
            "val_metric": val_metric_value,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_time_sec": epoch_time,
        }
        for k_name, v in val_metrics.items():
            row[f"val/{k_name}"] = v
        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} loss={train_loss:.6f} "
            f"val_{metric_for_best}={val_metric_value:.6f} "
            f"best={best_metric:.6f}@{best_epoch} time={epoch_time:.1f}s",
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
        "model_name": run_name,
        "run_id": run_id,
        "backbone": args.model,
        "best_metric_name": metric_for_best,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "num_items": int(num_items),
        "num_train_samples": int(len(train_ds)),
        "num_val_samples": int(len(val_ds)),
        "num_test_samples": int(len(test_ds)),
        "efficiency": {
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "mean_epoch_sec": float(np.mean([row["epoch_time_sec"] for row in logs])),
            "total_train_sec": float(sum(row["epoch_time_sec"] for row in logs)),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }

    if bool(eval_cfg.get("run_test_after_training", True)):
        val_topk_path = run_dir / "topk_val.npz" if save_topk_npz else None
        test_topk_path = run_dir / "topk_test.npz" if save_topk_npz else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_start = time.perf_counter()
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_inference_sec = time.perf_counter() - val_start
        test_start = time.perf_counter()
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_inference_sec = time.perf_counter() - test_start
        summary["val"] = val_final
        summary["test"] = test_final
        efficiency = summary["efficiency"]
        assert isinstance(efficiency, dict)
        efficiency.update(
            {
                "val_inference_sec": float(val_inference_sec),
                "val_user_count": int(len(val_ds)),
                "val_ms_per_user": float(1000.0 * val_inference_sec / max(len(val_ds), 1)),
                "test_inference_sec": float(test_inference_sec),
                "test_user_count": int(len(test_ds)),
                "test_ms_per_user": float(1000.0 * test_inference_sec / max(len(test_ds), 1)),
                "test_users_per_sec": float(len(test_ds) / max(test_inference_sec, 1.0e-12)),
                "peak_cuda_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / (1024**2))
                    if device.type == "cuda"
                    else 0.0
                ),
            }
        )

    save_json(summary, run_dir / "metrics_summary.json")
    print("========== Finished ==========")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
