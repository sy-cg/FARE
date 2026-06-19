# -*- coding: utf-8 -*-
"""
scripts/run_modality_debias.py

Train ModalityDebias variants for multimodal baselines.

Supported base models:
    --base_model vbpr
    --base_model bm3
    --base_model freedom
    --base_model lattice
    --base_model latefusion_sasrec
    --base_model latefusion_gru4rec
    --base_model latefusion_bert4rec

Recommended usage for pretrained multimodal baselines:

    python scripts/run_modality_debias.py \
      --dataset Video_Games \
      --base_model vbpr \
      --config configs/modality_debias_3090.yaml \
      --init_base_checkpoint results/Video_Games/vbpr/vbpr_video/best_model.pt \
      --freeze_base \
      --run_id vbpr_modality_debias_video

Recommended usage for LateFusion sequential baselines:

    python scripts/run_modality_debias.py \
      --dataset Video_Games \
      --base_model latefusion_sasrec \
      --config configs/modality_debias_3090.yaml \
      --init_backbone_checkpoint results/Video_Games/sasrec_id/<RUN_ID>/best_model.pt \
      --run_id latefusion_sasrec_modality_debias_video

Notes:
- --init_backbone_checkpoint is for ID backbone checkpoints used by LateFusion-*.
- --init_base_checkpoint is for loading a fully trained base model such as VBPR/BM3/FREEDOM/LATTICE,
  or an already trained LateFusion wrapper if you have one.
- --freeze_base freezes the base model and trains only the ModalityDebias unimodal branches.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

from modality_debias import (  # type: ignore
    LateFusionSequentialBase,
    ModalityDebiasWrapper,
)

try:
    from run_id_backbone import (  # type: ignore
        EvalNextItemDataset,
        build_model as build_id_backbone,
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
    raise ImportError("run_modality_debias.py depends on scripts/run_id_backbone.py") from exc


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


class PairwiseSequenceDataset(torch.utils.data.Dataset):
    """
    Pairwise next-item BPR samples.

    For each train sequence [i1, i2, ..., in], produce:
        prefix [i1 ... i_{t-1}], pos i_t, sampled neg

    This works for both sequential and graph models. Graph models ignore sequences.
    """

    def __init__(
        self,
        rows: Sequence[Tuple[int, Sequence[int]]],
        num_items: int,
        all_user_items: Dict[int, set[int]],
        max_seq_len: int,
        seed: int = 2026,
        use_all_prefixes: bool = True,
    ) -> None:
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.all_user_items = all_user_items
        self.rng = random.Random(seed)
        self.samples: List[Tuple[int, List[int], int]] = []

        for uid, seq in rows:
            uid = int(uid)
            seq = [int(x) for x in seq if 0 < int(x) <= self.num_items]
            if len(seq) < 2:
                continue

            indices = range(1, len(seq)) if use_all_prefixes else [len(seq) - 1]
            for t in indices:
                prefix = seq[:t]
                pos = seq[t]
                self.samples.append((uid, prefix[-self.max_seq_len :], pos))

        if not self.samples:
            raise ValueError("No pairwise sequence samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def sample_negative(self, uid: int) -> int:
        forbidden = self.all_user_items.get(uid, set())
        for _ in range(100):
            neg = self.rng.randint(1, self.num_items)
            if neg not in forbidden:
                return neg
        candidates = [i for i in range(1, self.num_items + 1) if i not in forbidden]
        return self.rng.choice(candidates) if candidates else self.rng.randint(1, self.num_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, prefix, pos = self.samples[idx]
        neg = self.sample_negative(uid)
        return {
            "user_ids": torch.tensor(uid, dtype=torch.long),
            "prefix": torch.tensor(prefix, dtype=torch.long),
            "targets": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(neg, dtype=torch.long),
        }


def pairwise_collate_fn(max_seq_len: int):
    def collate(batch):
        bs = len(batch)
        seqs = torch.zeros((bs, max_seq_len), dtype=torch.long)
        lengths = torch.zeros((bs,), dtype=torch.long)

        for i, row in enumerate(batch):
            prefix = row["prefix"][-max_seq_len:]
            l = int(prefix.numel())
            lengths[i] = l
            if l > 0:
                seqs[i, -l:] = prefix

        return {
            "user_ids": torch.stack([x["user_ids"] for x in batch], dim=0),
            "sequences": seqs,
            "lengths": lengths,
            "targets": torch.stack([x["targets"] for x in batch], dim=0),
            "neg_items": torch.stack([x["neg_items"] for x in batch], dim=0),
        }

    return collate


# ==========================================================
# Args / config
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModalityDebias baseline")

    parser.add_argument("--config", type=str, default="configs/modality_debias_3090.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        choices=[
            "vbpr",
            "bm3",
            "freedom",
            "lattice",
            "latefusion_sasrec",
            "latefusion_gru4rec",
            "latefusion_bert4rec",
        ],
    )
    parser.add_argument("--run_id", type=str, default=None)

    # Checkpoint loading.
    parser.add_argument(
        "--init_backbone_checkpoint",
        type=str,
        default=None,
        help="ID backbone checkpoint for LateFusion-* variants, e.g. SASRec/GRU4Rec/BERT4Rec best_model.pt.",
    )
    parser.add_argument(
        "--init_base_checkpoint",
        type=str,
        default=None,
        help="Pretrained base model checkpoint, e.g. VBPR/BM3/FREEDOM/LATTICE/LateFusion best_model.pt.",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze base_model parameters and train only ModalityDebias branches.",
    )

    # Training overrides.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--use_last_prefix_only", action="store_true")
    parser.add_argument("--train_on_fused", action="store_true")

    # ModalityDebias overrides.
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--debias_lambda", type=float, default=None)
    parser.add_argument("--counterfactual_mode", type=str, default=None, choices=["paper_product", "rank_subtract"])
    parser.add_argument("--branch_loss_weight", type=float, default=None)
    parser.add_argument("--branch_dim", type=int, default=None)
    parser.add_argument("--branch_dropout", type=float, default=None)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("modality_debias", {})

    if args.base_model is not None:
        cfg["base_model"] = args.base_model

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
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.grad_clip_norm is not None:
        cfg["train"]["grad_clip_norm"] = args.grad_clip_norm
    if args.use_last_prefix_only:
        cfg["train"]["use_all_prefixes"] = False
    if args.train_on_fused:
        cfg["train"]["train_on_fused"] = True

    if args.alpha is not None:
        cfg["modality_debias"]["alpha"] = args.alpha
    if args.debias_lambda is not None:
        cfg["modality_debias"]["debias_lambda"] = args.debias_lambda
    if args.counterfactual_mode is not None:
        cfg["modality_debias"]["counterfactual_mode"] = args.counterfactual_mode
    if args.branch_loss_weight is not None:
        cfg["modality_debias"]["branch_loss_weight"] = args.branch_loss_weight
    if args.branch_dim is not None:
        cfg["modality_debias"]["branch_dim"] = args.branch_dim
    if args.branch_dropout is not None:
        cfg["modality_debias"]["branch_dropout"] = args.branch_dropout

    if args.cpu:
        cfg["device"] = "cpu"

    return cfg


# ==========================================================
# Checkpoint helpers
# ==========================================================


def _strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


def _best_matching_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], str, int]:
    model_state = model.state_dict()
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [("raw", state)]

    for prefix in ["module.", "model.", "base_model.", "id_backbone.", "backbone."]:
        candidates.append((f"strip:{prefix}", _strip_prefix_if_present(state, prefix)))

    best_name = "raw"
    best_state = state
    best_count = -1

    for name, cand in candidates:
        count = 0
        for k, v in cand.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                count += 1
        if count > best_count:
            best_name = name
            best_state = cand
            best_count = count

    return best_state, best_name, best_count


def load_checkpoint_if_needed(
    model: nn.Module,
    path: Optional[str],
    device: torch.device,
    label: str = "Checkpoint",
    strict: bool = False,
) -> nn.Module:
    if not path:
        return model

    ckpt_path = resolve_path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{label} not found: {ckpt_path}")

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = unwrap_state_dict(ckpt)
    state, transform_name, matched = _best_matching_state(model, state)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[{label}] loaded: {ckpt_path}")
    print(f"[{label}] transform={transform_name}, matched_keys={matched}")
    print(f"[{label}] missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}")

    if matched == 0:
        print(f"[Warning] {label} matched 0 keys. Check whether the checkpoint belongs to this model.")

    return model


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================================
# Base model builders
# ==========================================================


def build_base_model(
    base_model_name: str,
    cfg: Dict[str, Any],
    data_dir: Path,
    train_rows,
    num_users: int,
    num_items: int,
    device: torch.device,
    init_backbone_checkpoint: Optional[str] = None,
) -> nn.Module:
    model_cfg = cfg.get("model", {})
    base_cfg = cfg.get("base", {})

    if base_model_name == "vbpr":
        from model_vbpr import VBPR  # type: ignore

        return VBPR(
            num_users=num_users,
            num_items=num_items,
            data_dir=data_dir,
            feature_file=base_cfg.get("vision_feature_file", "vision_features.npy"),
            mask_file=base_cfg.get("vision_mask_file", "vision_mask.npy"),
            latent_dim=int(base_cfg.get("latent_dim", 64)),
            visual_dim=int(base_cfg.get("visual_dim", 64)),
            normalize_features=bool(base_cfg.get("normalize_features", True)),
            zero_missing_features=bool(base_cfg.get("zero_missing_features", True)),
            dropout=float(base_cfg.get("dropout", 0.0)),
        )

    if base_model_name == "bm3":
        from model_bm3 import BM3, build_bm3_norm_adj  # type: ignore

        norm_adj = build_bm3_norm_adj(
            train_rows=train_rows,
            num_users=num_users,
            num_items=num_items,
            add_self_loops=bool(base_cfg.get("add_self_loops", False)),
        )
        return BM3(
            num_users=num_users,
            num_items=num_items,
            norm_adj=norm_adj,
            data_dir=data_dir,
            embedding_dim=int(base_cfg.get("embedding_dim", 64)),
            n_layers=int(base_cfg.get("n_layers", 1)),
            dropout=float(base_cfg.get("dropout", 0.5)),
            reg_weight=float(base_cfg.get("reg_weight", 0.01)),
            cl_weight=float(base_cfg.get("cl_weight", 1.0)),
            text_feature_file=base_cfg.get("text_feature_file", "text_features.npy"),
            vision_feature_file=base_cfg.get("vision_feature_file", "vision_features.npy"),
            text_mask_file=base_cfg.get("text_mask_file", "text_mask.npy"),
            vision_mask_file=base_cfg.get("vision_mask_file", "vision_mask.npy"),
            use_text=bool(base_cfg.get("use_text", True)),
            use_vision=bool(base_cfg.get("use_vision", True)),
        )

    if base_model_name == "freedom":
        from model_freedom import FREEDOM, build_edge_info, build_user_item_norm_adj  # type: ignore

        norm_adj = build_user_item_norm_adj(train_rows, num_users=num_users, num_items=num_items)
        edge_indices, edge_values = build_edge_info(train_rows, num_users=num_users, num_items=num_items)
        return FREEDOM(
            num_users=num_users,
            num_items=num_items,
            norm_adj=norm_adj,
            edge_indices=edge_indices,
            edge_values=edge_values,
            data_dir=data_dir,
            embedding_dim=int(base_cfg.get("embedding_dim", 64)),
            feat_embed_dim=int(base_cfg.get("feat_embed_dim", 64)),
            n_mm_layers=int(base_cfg.get("n_mm_layers", 1)),
            n_ui_layers=int(base_cfg.get("n_ui_layers", 2)),
            knn_k=int(base_cfg.get("knn_k", 10)),
            mm_image_weight=float(base_cfg.get("mm_image_weight", 0.1)),
            edge_dropout=float(base_cfg.get("edge_dropout", 0.8)),
            degree_ratio=float(base_cfg.get("degree_ratio", 1.0)),
            reg_weight=float(base_cfg.get("reg_weight", 0.0001)),
            l2_weight=float(base_cfg.get("l2_weight", 0.0)),
            text_feature_file=base_cfg.get("text_feature_file", "text_features.npy"),
            vision_feature_file=base_cfg.get("vision_feature_file", "vision_features.npy"),
            text_mask_file=base_cfg.get("text_mask_file", "text_mask.npy"),
            vision_mask_file=base_cfg.get("vision_mask_file", "vision_mask.npy"),
            use_text=bool(base_cfg.get("use_text", True)),
            use_vision=bool(base_cfg.get("use_vision", True)),
        )

    if base_model_name == "lattice":
        from model_lattice import LATTICE, build_user_item_norm_adj  # type: ignore

        norm_adj = build_user_item_norm_adj(train_rows, num_users=num_users, num_items=num_items)
        return LATTICE(
            num_users=num_users,
            num_items=num_items,
            norm_adj=norm_adj,
            data_dir=data_dir,
            embedding_dim=int(base_cfg.get("embedding_dim", 64)),
            feat_embed_dim=int(base_cfg.get("feat_embed_dim", 64)),
            weight_size=base_cfg.get("weight_size", [64]),
            mess_dropout=base_cfg.get("mess_dropout", [0.0]),
            n_item_layers=int(base_cfg.get("n_item_layers", 1)),
            topk=int(base_cfg.get("topk", 10)),
            lambda_coeff=float(base_cfg.get("lambda_coeff", 0.9)),
            cf_model=str(base_cfg.get("cf_model", "lightgcn")),
            decay=float(base_cfg.get("decay", 0.0001)),
            text_feature_file=base_cfg.get("text_feature_file", "text_features.npy"),
            vision_feature_file=base_cfg.get("vision_feature_file", "vision_features.npy"),
            text_mask_file=base_cfg.get("text_mask_file", "text_mask.npy"),
            vision_mask_file=base_cfg.get("vision_mask_file", "vision_mask.npy"),
            use_text=bool(base_cfg.get("use_text", True)),
            use_vision=bool(base_cfg.get("use_vision", True)),
        )

    if base_model_name.startswith("latefusion_"):
        backbone = base_model_name.replace("latefusion_", "")
        id_model = build_id_backbone(backbone, num_items=num_items, cfg=cfg)
        id_model = load_checkpoint_if_needed(
            id_model,
            init_backbone_checkpoint,
            device,
            label=f"{backbone.upper()} backbone checkpoint",
            strict=False,
        )

        hidden_size = int(model_cfg.get("hidden_size", 128))
        return LateFusionSequentialBase(
            id_backbone=id_model,
            num_items=num_items,
            data_dir=data_dir,
            hidden_size=hidden_size,
            mm_dim=int(base_cfg.get("mm_dim", 64)),
            use_text=bool(base_cfg.get("use_text", True)),
            use_vision=bool(base_cfg.get("use_vision", True)),
            text_feature_file=base_cfg.get("text_feature_file", "text_features.npy"),
            vision_feature_file=base_cfg.get("vision_feature_file", "vision_features.npy"),
            text_mask_file=base_cfg.get("text_mask_file", "text_mask.npy"),
            vision_mask_file=base_cfg.get("vision_mask_file", "vision_mask.npy"),
            text_weight=float(base_cfg.get("text_weight", 0.05)),
            vision_weight=float(base_cfg.get("vision_weight", 0.05)),
        )

    raise ValueError(f"Unsupported base_model={base_model_name!r}")


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Did you freeze the whole model?")

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1.0e-8)),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1.0e-8)),
        )
    raise ValueError(f"Unsupported optimizer={name!r}")


# ==========================================================
# Training
# ==========================================================


def train_one_epoch(
    model: ModalityDebiasWrapper,
    loader,
    optimizer,
    device: torch.device,
    grad_clip_norm: float,
    train_on_fused: bool,
    freeze_base: bool,
    log_every: int,
) -> Dict[str, float]:
    model.train()

    if freeze_base:
        model.base_model.eval()

    total_examples = 0
    agg: Dict[str, float] = {}
    grad_norms: List[float] = []

    # Graph models with per-epoch preprocessing. Keep this even when frozen,
    # because FREEDOM's pruned graph is part of its forward state.
    if hasattr(model.base_model, "pre_epoch_processing"):
        model.base_model.pre_epoch_processing()

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        pos_items = batch["targets"].to(device, non_blocking=True)
        neg_items = batch["neg_items"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss, stats = model.training_loss(
            user_ids=user_ids,
            sequences=sequences,
            lengths=lengths,
            pos_items=pos_items,
            neg_items=neg_items,
            train_on_fused=train_on_fused,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite ModalityDebias loss at step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(grad_clip_norm),
            )
            grad_norms.append(float(norm.detach().cpu().item()))

        optimizer.step()

        bs = int(user_ids.numel())
        total_examples += bs
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs

        if log_every > 0 and step % log_every == 0:
            avg = agg.get("loss_total", 0.0) / max(total_examples, 1)
            print(f"  step={step:6d} train_loss={avg:.6f}", flush=True)

    out = {k: v / max(total_examples, 1) for k, v in agg.items()}
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
        raise ValueError("Dataset must be provided by --dataset or default_dataset.")
    dataset = str(dataset)

    base_model_name = str(cfg.get("base_model", "vbpr"))
    if args.base_model:
        base_model_name = str(args.base_model)

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

    train_rows = read_user_sequences(data_dir / "train.txt")
    val_rows = read_user_sequences(data_dir / "val.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")

    num_items = infer_num_items(data_dir)
    num_users = infer_num_users(train_rows, val_rows, test_rows)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg.get("model", {})
    md_cfg = cfg.get("modality_debias", {})

    device_str = str(cfg.get("device", "cuda"))
    device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    batch_size = int(train_cfg.get("batch_size", 512))
    eval_batch_size = int(eval_cfg.get("batch_size", 512))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"

    all_user_items = build_all_user_items(train_rows, val_rows, test_rows, num_items=num_items)

    train_ds = PairwiseSequenceDataset(
        rows=train_rows,
        num_items=num_items,
        all_user_items=all_user_items,
        max_seq_len=max_seq_len,
        seed=seed,
        use_all_prefixes=bool(train_cfg.get("use_all_prefixes", True)),
    )
    val_ds = EvalNextItemDataset(val_rows, max_seq_len=max_seq_len)
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pairwise_collate_fn(max_seq_len),
        drop_last=False,
    )

    eval_collate = make_collate_fn(max_seq_len)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eval_collate,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eval_collate,
        drop_last=False,
    )

    base_model = build_base_model(
        base_model_name=base_model_name,
        cfg=cfg,
        data_dir=data_dir,
        train_rows=train_rows,
        num_users=num_users,
        num_items=num_items,
        device=device,
        init_backbone_checkpoint=args.init_backbone_checkpoint,
    ).to(device)

    base_model = load_checkpoint_if_needed(
        base_model,
        args.init_base_checkpoint,
        device,
        label=f"{base_model_name} base checkpoint",
        strict=False,
    )

    model = ModalityDebiasWrapper(
        base_model=base_model,
        num_users=num_users,
        num_items=num_items,
        data_dir=data_dir,
        train_rows=train_rows,
        branch_dim=int(md_cfg.get("branch_dim", 64)),
        use_text=bool(md_cfg.get("use_text", True)),
        use_vision=bool(md_cfg.get("use_vision", True)),
        text_feature_file=md_cfg.get("text_feature_file", "text_features.npy"),
        vision_feature_file=md_cfg.get("vision_feature_file", "vision_features.npy"),
        text_mask_file=md_cfg.get("text_mask_file", "text_mask.npy"),
        vision_mask_file=md_cfg.get("vision_mask_file", "vision_mask.npy"),
        normalize_features=bool(md_cfg.get("normalize_features", True)),
        zero_missing_features=bool(md_cfg.get("zero_missing_features", True)),
        branch_dropout=float(md_cfg.get("branch_dropout", 0.0)),
        branch_loss_weight=float(md_cfg.get("branch_loss_weight", 1.0)),
        alpha=float(md_cfg.get("alpha", 5.0)),
        debias_lambda=float(md_cfg.get("debias_lambda", 1.0)),
        counterfactual_mode=str(md_cfg.get("counterfactual_mode", "paper_product")),
        base_counterfactual=str(md_cfg.get("base_counterfactual", "user_mean")),
        cf_prob=float(md_cfg.get("cf_prob", 0.5)),
        rank_normalize=bool(md_cfg.get("rank_normalize", True)),
        debias_min=float(md_cfg.get("debias_min", 0.0)),
        debias_max=float(md_cfg.get("debias_max", 1.0)),
    ).to(device)

    if args.freeze_base:
        freeze_module(model.base_model)
        model.base_model.eval()

    optimizer = build_optimizer(model, cfg)

    default_run_name = f"{base_model_name}_modality_debias"
    run_name = str(cfg.get("run_name", default_run_name))
    # If config keeps a generic run_name, make it base-specific to avoid mixing folders.
    if run_name in {"modality_debias", "mdebias"}:
        run_name = default_run_name

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    result_root = resolve_path(paths_cfg.get("result_root", "results"))
    run_dir = result_root / dataset / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved.update(
        {
            "dataset": dataset,
            "method": "ModalityDebias",
            "base_model": base_model_name,
            "model_name": run_name,
            "run_id": run_id,
            "data_dir": str(data_dir),
            "num_users": int(num_users),
            "num_items": int(num_items),
            "num_train_samples": int(len(train_ds)),
            "num_val_samples": int(len(val_ds)),
            "num_test_samples": int(len(test_ds)),
            "device_resolved": str(device),
            "init_backbone_checkpoint": args.init_backbone_checkpoint,
            "init_base_checkpoint": args.init_base_checkpoint,
            "freeze_base": bool(args.freeze_base),
            "trainable_parameters": int(count_trainable_parameters(model)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    save_json(resolved, run_dir / "config_resolved.json")

    print("========== ModalityDebias Training ==========")
    print(f"dataset:       {dataset}")
    print(f"base_model:    {base_model_name}")
    print(f"run_name:      {run_name}")
    print(f"run_id:        {run_id}")
    print(f"data_dir:      {data_dir}")
    print(f"num_users:     {num_users}")
    print(f"num_items:     {num_items}")
    print(f"train samples: {len(train_ds)}")
    print(f"parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"trainable:     {count_trainable_parameters(model):,}")
    print(f"freeze_base:   {args.freeze_base}")
    print(f"device:        {device}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    log_every = int(train_cfg.get("log_every", 100))
    train_on_fused = bool(train_cfg.get("train_on_fused", False))

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
            train_on_fused=train_on_fused,
            freeze_base=bool(args.freeze_base),
            log_every=log_every,
        )

        epoch_time = time.time() - start
        val_metric_value = float("nan")
        val_metrics: Dict[str, float] = {}

        if epoch % eval_every == 0:
            model.eval()
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
            "val_metric": val_metric_value,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_time_sec": epoch_time,
        }
        for k, v in train_stats.items():
            row[f"train/{k}"] = v
        for k, v in val_metrics.items():
            row[f"val/{k}"] = v

        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"loss={train_stats.get('loss_total', float('nan')):.6f} "
            f"main={train_stats.get('loss_main', float('nan')):.6f} "
            f"text={train_stats.get('loss_text', float('nan')):.6f} "
            f"vision={train_stats.get('loss_vision', float('nan')):.6f} "
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
        state, _, _ = _best_matching_state(model, state)
        model.load_state_dict(state, strict=False)
    else:
        print("[Warning] best_model.pt was not saved; using last epoch model.")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epochs}, best_path)

    summary: Dict[str, object] = {
        "dataset": dataset,
        "method": "ModalityDebias",
        "base_model": base_model_name,
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
        "init_backbone_checkpoint": args.init_backbone_checkpoint,
        "init_base_checkpoint": args.init_base_checkpoint,
        "freeze_base": bool(args.freeze_base),
    }

    if bool(eval_cfg.get("run_test_after_training", True)):
        model.eval()

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
