# -*- coding: utf-8 -*-
"""
scripts/run_lattice.py

Train and evaluate LATTICE multimodal baseline.

LATTICE is a graph-based multimodal recommendation model:
    - builds image/text item-item kNN graphs
    - learns modality-aware latent item graph
    - combines learned and original item graphs with lambda_coeff
    - propagates item embeddings on item graph
    - combines item graph representation with LightGCN/NGCF/MF collaborative graph
    - trains with BPR loss

This script adapts LATTICE to the current project pipeline:
    - Reuses train/val/test txt files.
    - Reuses full-sort evaluator from run_id_backbone.py.
    - Saves topk_val.npz and topk_test.npz for fairness evaluation.

Typical usage:

    python scripts/run_lattice.py \
      --dataset Video_Games \
      --config configs/lattice_3090.yaml \
      --run_id lattice_video

Smoke test:

    python scripts/run_lattice.py \
      --dataset Video_Games \
      --config configs/lattice_3090.yaml \
      --epochs 2 \
      --run_id lattice_video_smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from model_lattice import LATTICE, build_user_item_norm_adj  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import src/model_lattice.py. Make sure it exists.") from exc

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
        "run_lattice.py depends on scripts/run_id_backbone.py. "
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


class LATTICETrainDataset(torch.utils.data.Dataset):
    """
    Pairwise BPR dataset for LATTICE.
    """

    def __init__(
        self,
        train_rows: Sequence[Tuple[int, Sequence[int]]],
        num_items: int,
        all_user_items: Optional[Dict[int, set[int]]] = None,
        seed: int = 2026,
    ) -> None:
        self.num_items = int(num_items)
        self.samples: List[Tuple[int, int]] = []
        self.user_pos_train: Dict[int, set[int]] = {}
        self.user_forbidden = all_user_items if all_user_items is not None else self.user_pos_train
        self.rng = random.Random(seed)

        for uid, seq in train_rows:
            uid = int(uid)
            if uid <= 0:
                continue

            pos_set = self.user_pos_train.setdefault(uid, set())
            for item in seq:
                item = int(item)
                if 0 < item <= self.num_items:
                    self.samples.append((uid, item))
                    pos_set.add(item)

        if not self.samples:
            raise ValueError("No valid train interactions found for LATTICE.")

    def __len__(self) -> int:
        return len(self.samples)

    def sample_negative(self, uid: int) -> int:
        forbidden = self.user_forbidden.get(uid, set())

        for _ in range(100):
            neg = self.rng.randint(1, self.num_items)
            if neg not in forbidden:
                return neg

        candidates = [i for i in range(1, self.num_items + 1) if i not in forbidden]
        if not candidates:
            return self.rng.randint(1, self.num_items)
        return self.rng.choice(candidates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, pos = self.samples[idx]
        neg = self.sample_negative(uid)

        return {
            "user_ids": torch.tensor(uid, dtype=torch.long),
            "pos_items": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(neg, dtype=torch.long),
        }


def lattice_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "user_ids": torch.stack([x["user_ids"] for x in batch], dim=0),
        "pos_items": torch.stack([x["pos_items"] for x in batch], dim=0),
        "neg_items": torch.stack([x["neg_items"] for x in batch], dim=0),
    }


# ==========================================================
# Config
# ==========================================================


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LATTICE multimodal baseline")

    parser.add_argument("--config", type=str, default="configs/lattice_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    # Model overrides.
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--embed_size", type=int, default=None, help="Alias of --embedding_dim.")
    parser.add_argument("--feat_embed_dim", type=int, default=None)
    parser.add_argument("--weight_size", type=str, default=None, help="Comma-separated hidden sizes, e.g. 64 or 64,64")
    parser.add_argument("--mess_dropout", type=str, default=None, help="Comma-separated dropout list, e.g. 0.0 or 0.1,0.1")
    parser.add_argument("--n_item_layers", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--lambda_coeff", type=float, default=None)
    parser.add_argument("--cf_model", type=str, default=None, choices=["lightgcn", "ngcf", "mf"])
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--text_feature_file", type=str, default=None)
    parser.add_argument("--vision_feature_file", type=str, default=None)
    parser.add_argument("--text_mask_file", type=str, default=None)
    parser.add_argument("--vision_mask_file", type=str, default=None)
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--no_vision", action="store_true")
    parser.add_argument("--no_normalize_features", action="store_true")
    parser.add_argument("--no_zero_missing_features", action="store_true")
    parser.add_argument("--freeze_features", action="store_true")
    parser.add_argument("--no_cache_item_graph", action="store_true")
    parser.add_argument("--item_graph_cache_prefix", type=str, default=None)
    parser.add_argument("--knn_block_size", type=int, default=None)
    parser.add_argument("--dynamic_graph_max_items", type=int, default=None)
    parser.add_argument("--clamp_sim_min", type=float, default=None)

    # Training overrides.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
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

    if args.embedding_dim is not None:
        cfg["model"]["embedding_dim"] = args.embedding_dim
    if args.embed_size is not None:
        cfg["model"]["embedding_dim"] = args.embed_size
    if args.feat_embed_dim is not None:
        cfg["model"]["feat_embed_dim"] = args.feat_embed_dim
    if args.weight_size is not None:
        cfg["model"]["weight_size"] = parse_int_list(args.weight_size)
    if args.mess_dropout is not None:
        cfg["model"]["mess_dropout"] = parse_float_list(args.mess_dropout)
    if args.n_item_layers is not None:
        cfg["model"]["n_item_layers"] = args.n_item_layers
    if args.topk is not None:
        cfg["model"]["topk"] = args.topk
    if args.lambda_coeff is not None:
        cfg["model"]["lambda_coeff"] = args.lambda_coeff
    if args.cf_model is not None:
        cfg["model"]["cf_model"] = args.cf_model
    if args.decay is not None:
        cfg["model"]["decay"] = args.decay
    if args.text_feature_file is not None:
        cfg["model"]["text_feature_file"] = args.text_feature_file
    if args.vision_feature_file is not None:
        cfg["model"]["vision_feature_file"] = args.vision_feature_file
    if args.text_mask_file is not None:
        cfg["model"]["text_mask_file"] = args.text_mask_file
    if args.vision_mask_file is not None:
        cfg["model"]["vision_mask_file"] = args.vision_mask_file
    if args.no_text:
        cfg["model"]["use_text"] = False
    if args.no_vision:
        cfg["model"]["use_vision"] = False
    if args.no_normalize_features:
        cfg["model"]["normalize_features"] = False
    if args.no_zero_missing_features:
        cfg["model"]["zero_missing_features"] = False
    if args.freeze_features:
        cfg["model"]["freeze_features"] = True
    if args.no_cache_item_graph:
        cfg["model"]["cache_item_graph"] = False
    if args.item_graph_cache_prefix is not None:
        cfg["model"]["item_graph_cache_prefix"] = args.item_graph_cache_prefix
    if args.knn_block_size is not None:
        cfg["model"]["knn_block_size"] = args.knn_block_size
    if args.dynamic_graph_max_items is not None:
        cfg["model"]["dynamic_graph_max_items"] = args.dynamic_graph_max_items
    if args.clamp_sim_min is not None:
        cfg["model"]["clamp_sim_min"] = args.clamp_sim_min

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

    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1.0e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1.0e-8))
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    raise ValueError(f"Unsupported optimizer={name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "lambda")).lower()

    if name in {"none", "null", "off"}:
        return None

    if name == "lambda":
        gamma = float(sched_cfg.get("gamma", 0.96))
        interval = float(sched_cfg.get("interval", 50.0))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: gamma ** (epoch / interval))

    raise ValueError(f"Unsupported scheduler={name!r}")


# ==========================================================
# Training
# ==========================================================


def train_one_epoch_lattice(
    model: LATTICE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 5.0,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    model.clear_eval_cache()

    total_examples = 0
    total_pair_correct = 0
    agg: Dict[str, float] = {}
    grad_norms: List[float] = []

    build_item_graph = True

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        pos_items = batch["pos_items"].to(device, non_blocking=True)
        neg_items = batch["neg_items"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss, stats = model.calculate_loss(
            user_ids=user_ids,
            pos_items=pos_items,
            neg_items=neg_items,
            build_item_graph=build_item_graph,
        )

        build_item_graph = False

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite LATTICE loss at step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            grad_norms.append(float(norm.detach().cpu().item()))

        optimizer.step()

        with torch.no_grad():
            user_all, item_all = model.graph_forward(model.norm_adj, build_item_graph=False)
            u = user_all[user_ids]
            pos = item_all[pos_items]
            neg = item_all[neg_items]
            margin = (u * pos).sum(dim=-1) - (u * neg).sum(dim=-1)
            total_pair_correct += int((margin > 0).sum().detach().cpu().item())

        bs = int(user_ids.numel())
        total_examples += bs

        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs

        if log_every > 0 and step % log_every == 0:
            avg_loss = agg.get("loss_total", 0.0) / max(total_examples, 1)
            acc = total_pair_correct / max(total_examples, 1)
            print(f"  step={step:6d} train_loss={avg_loss:.6f} pair_acc={acc:.4f}", flush=True)

    out = {k: v / max(total_examples, 1) for k, v in agg.items()}
    out["pair_acc"] = total_pair_correct / max(total_examples, 1)
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

    print("Building LATTICE normalized user-item graph...")
    norm_adj = build_user_item_norm_adj(
        train_rows=train_rows,
        num_users=num_users,
        num_items=num_items,
    )

    all_user_items = build_all_user_items(train_rows, val_rows, test_rows, num_items=num_items)

    train_ds = LATTICETrainDataset(
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
        collate_fn=lattice_collate_fn,
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

    model = LATTICE(
        num_users=num_users,
        num_items=num_items,
        norm_adj=norm_adj,
        data_dir=data_dir,
        embedding_dim=int(model_cfg.get("embedding_dim", model_cfg.get("embed_size", 64))),
        feat_embed_dim=int(model_cfg.get("feat_embed_dim", 64)),
        weight_size=model_cfg.get("weight_size", [64]),
        mess_dropout=model_cfg.get("mess_dropout", [0.0]),
        n_item_layers=int(model_cfg.get("n_item_layers", model_cfg.get("n_layers", 1))),
        topk=int(model_cfg.get("topk", 10)),
        lambda_coeff=float(model_cfg.get("lambda_coeff", 0.9)),
        cf_model=str(model_cfg.get("cf_model", "lightgcn")),
        decay=float(model_cfg.get("decay", 1.0e-4)),
        text_feature_file=model_cfg.get("text_feature_file", "text_features.npy"),
        vision_feature_file=model_cfg.get("vision_feature_file", "vision_features.npy"),
        text_mask_file=model_cfg.get("text_mask_file", "text_mask.npy"),
        vision_mask_file=model_cfg.get("vision_mask_file", "vision_mask.npy"),
        use_text=bool(model_cfg.get("use_text", True)),
        use_vision=bool(model_cfg.get("use_vision", True)),
        normalize_features=bool(model_cfg.get("normalize_features", True)),
        zero_missing_features=bool(model_cfg.get("zero_missing_features", True)),
        freeze_features=bool(model_cfg.get("freeze_features", False)),
        cache_item_graph=bool(model_cfg.get("cache_item_graph", True)),
        item_graph_cache_prefix=str(model_cfg.get("item_graph_cache_prefix", "lattice")),
        knn_block_size=int(model_cfg.get("knn_block_size", 1024)),
        dynamic_graph_max_items=int(model_cfg.get("dynamic_graph_max_items", 20000)),
        clamp_sim_min=float(model_cfg.get("clamp_sim_min", 0.0)),
        init_std=float(model_cfg.get("init_std", 0.01)),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    run_name = str(cfg.get("run_name", "lattice"))
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / dataset / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["dataset"] = dataset
    resolved["method"] = "LATTICE"
    resolved["model_name"] = run_name
    resolved["run_id"] = run_id
    resolved["data_dir"] = str(data_dir)
    resolved["num_users"] = int(num_users)
    resolved["num_items"] = int(num_items)
    resolved["num_train_samples"] = int(len(train_ds))
    resolved["num_val_samples"] = int(len(val_ds))
    resolved["num_test_samples"] = int(len(test_ds))
    resolved["device_resolved"] = str(device)
    resolved["graph_num_nodes"] = int(norm_adj.shape[0])
    resolved["graph_num_edges"] = int(norm_adj._nnz())
    save_json(resolved, run_dir / "config_resolved.json")

    print("========== LATTICE Training ==========")
    print(f"dataset:       {dataset}")
    print(f"method:        LATTICE")
    print(f"run_name:      {run_name}")
    print(f"run_id:        {run_id}")
    print(f"data_dir:      {data_dir}")
    print(f"num_users:     {num_users}")
    print(f"num_items:     {num_items}")
    print(f"graph nodes:   {norm_adj.shape[0]}")
    print(f"graph edges:   {norm_adj._nnz()}")
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

        train_stats = train_one_epoch_lattice(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start

        val_metric_value = float("nan")
        val_metrics: Dict[str, float] = {}

        if epoch % eval_every == 0:
            model.eval()
            model.clear_eval_cache()

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

            model.clear_eval_cache()
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
            "val_metric": float(val_metric_value),
            "best_metric": float(best_metric),
            "best_epoch": float(best_epoch),
            "epoch_time_sec": float(epoch_time),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        for k, v in train_stats.items():
            row[f"train/{k}"] = float(v)

        for k_name, v in val_metrics.items():
            row[f"val/{k_name}"] = float(v)

        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"loss={train_stats.get('loss_total', float('nan')):.6f} "
            f"mf={train_stats.get('loss_mf', float('nan')):.6f} "
            f"reg={train_stats.get('loss_reg', float('nan')):.6f} "
            f"pair_acc={train_stats.get('pair_acc', float('nan')):.4f} "
            f"val_{metric_for_best}={val_metric_value:.6f} "
            f"best={best_metric:.6f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
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
        "method": "LATTICE",
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
        "graph_num_nodes": int(norm_adj.shape[0]),
        "graph_num_edges": int(norm_adj._nnz()),
    }

    if bool(eval_cfg.get("run_test_after_training", True)):
        model.eval()
        model.clear_eval_cache()

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

        model.clear_eval_cache()

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

        model.clear_eval_cache()

        summary["val"] = val_final
        summary["test"] = test_final

    save_json(summary, run_dir / "metrics_summary.json")

    print("========== Finished ==========")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
