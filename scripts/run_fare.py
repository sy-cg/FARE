# -*- coding: utf-8 -*-
"""
scripts/run_fare.py

Unified backbone-agnostic FARE trainer.

Supported backbones:
    --backbone sasrec
    --backbone gru4rec
    --backbone bert4rec

Backward compatibility:
    --init_sasrec_checkpoint is kept as an alias of --init_backbone_checkpoint.

Important:
    This script is the single FARE trainer. FARE uses exposure-aware
    recommendation-loss reweighting as its default fairness mechanism.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scripts/run_fare.py depends on scripts/run_id_backbone.py. "
        "Please make sure run_id_backbone.py exists and passes py_compile."
    ) from exc

try:
    from model_fare import FARE  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import FARE from src/model_fare.py. "
        "Please make sure src/model_fare.py exists and passes py_compile."
    ) from exc


CORE_GROUPS = [
    "popularity_group",
    "text_quality_group",
    "vision_quality_group",
]

PROXY_GROUPS = [
    "category_proxy_group",
    "brand_store_proxy_group",
    "multimodal_cluster_proxy_group",
]

FULL_GROUPS = CORE_GROUPS + PROXY_GROUPS

KNOWN_GROUP_ORDER = [
    "popularity_group",
    "modality_availability_group",
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


def parse_args(default_config: str = "configs/fare_3090.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train backbone-agnostic FARE")

    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Display method name written to metrics_summary.json",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["sasrec", "gru4rec", "bert4rec"],
        help="ID backbone wrapped by FARE",
    )
    parser.add_argument("--init_backbone_checkpoint", type=str, default=None)
    parser.add_argument(
        "--init_sasrec_checkpoint",
        type=str,
        default=None,
        help="Backward-compatible alias of --init_backbone_checkpoint",
    )
    parser.add_argument("--freeze_id_backbone", action="store_true")

    # Model overrides.
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
    parser.add_argument("--fair_dim", type=int, default=None)
    parser.add_argument("--encoder_hidden_dim", type=int, default=None)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.add_argument("--fair_weight_init", type=float, default=None)
    parser.add_argument("--max_fair_weight", type=float, default=None)

    # Fairness objective overrides.
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--min_group_count", type=int, default=None)
    parser.add_argument("--drop_rare_classes", action="store_true")

    # Residual scoring and ranking-loss reweighting overrides.
    parser.add_argument(
        "--residual_score_weight",
        type=float,
        default=None,
        help=(
            "Weight of the FARE residual branch in final ranking scores. "
            "The default FARE setting is 1.0."
        ),
    )
    parser.add_argument(
        "--fair_rec_reweight_weight",
        type=float,
        default=None,
        help=(
            "Strength of group-aware reweighting on recommendation CE loss. "
            "Use 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--fair_rec_reweight_groups",
        type=str,
        default=None,
        help=(
            "Comma-separated group names used for ranking-loss reweighting, e.g., "
            "popularity_group,brand_store_proxy_group,multimodal_cluster_proxy_group"
        ),
    )
    parser.add_argument(
        "--fair_rec_reweight_min",
        type=float,
        default=None,
        help="Minimum clipping value for group-aware recommendation sample weights.",
    )
    parser.add_argument(
        "--fair_rec_reweight_max",
        type=float,
        default=None,
        help="Maximum clipping value for group-aware recommendation sample weights.",
    )
    parser.add_argument(
        "--fair_rec_exposure_topk_path",
        type=str,
        default=None,
        help="Path to a top-k .npz file used to estimate exposure distribution.",
    )
    parser.add_argument(
        "--fair_rec_exposure_target",
        type=str,
        default=None,
        choices=["catalog", "uniform"],
        help="Target group distribution for exposure-aware reweighting.",
    )
    parser.add_argument(
        "--fair_rec_exposure_k",
        type=int,
        default=None,
        help="Use only the first K recommendations from the top-k file to estimate exposure.",
    )
    parser.add_argument(
        "--allow_test_exposure_topk",
        action="store_true",
        help="Allow using topk_test.npz for exposure reweighting. Disabled by default to avoid test leakage.",
    )

    # Train/eval.
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_fairness_eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def apply_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("fare", {})

    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone

    scalar_train_overrides = {
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "eval_batch_size": ("eval", "batch_size"),
        "lr": ("train", "learning_rate"),
        "weight_decay": ("train", "weight_decay"),
        "max_seq_len": ("model", "max_seq_len"),
        "hidden_size": ("model", "hidden_size"),
        "num_layers": ("model", "num_layers"),
        "num_heads": ("model", "num_heads"),
        "dropout": ("model", "dropout"),
        "patience": ("train", "patience"),
        "eval_every": ("train", "eval_every"),
        "num_workers": ("train", "num_workers"),
    }
    for attr, (section, key) in scalar_train_overrides.items():
        value = getattr(args, attr)
        if value is not None:
            cfg[section][key] = value

    for key in [
        "mm_score_dim",
        "fair_dim",
        "encoder_hidden_dim",
        "encoder_dropout",
        "fair_weight_init",
        "max_fair_weight",
        "residual_score_weight",
        "fair_rec_reweight_weight",
        "fair_rec_reweight_min",
        "fair_rec_reweight_max",
        "fair_rec_exposure_k",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg["fare"][key] = value

    if not cfg["fare"].get("groups"):
        cfg["fare"]["groups"] = list(FULL_GROUPS)
    cfg["fare"]["fair_rec_reweight_mode"] = "exposure"
    cfg.setdefault("method_name", "FARE")

    if args.groups is not None:
        groups = parse_list_arg(args.groups)
        if groups:
            cfg["fare"]["groups"] = groups

    if args.fair_rec_reweight_groups is not None:
        reweight_groups = parse_list_arg(args.fair_rec_reweight_groups)
        if reweight_groups:
            cfg["fare"]["fair_rec_reweight_groups"] = reweight_groups

    if args.fair_rec_exposure_topk_path is not None:
        cfg["fare"]["fair_rec_exposure_topk_path"] = args.fair_rec_exposure_topk_path
    if args.fair_rec_exposure_target is not None:
        cfg["fare"]["fair_rec_exposure_target"] = args.fair_rec_exposure_target
    if args.allow_test_exposure_topk:
        cfg["fare"]["allow_test_exposure_topk"] = True

    if args.min_group_count is not None:
        cfg["fare"]["min_group_count"] = args.min_group_count
    if args.drop_rare_classes:
        cfg["fare"]["drop_rare_classes"] = True

    if args.no_fairness_eval:
        cfg["eval"]["run_fairness_eval"] = False
    if args.cpu:
        cfg["device"] = "cpu"

    return cfg


# ==========================================================
# Group-label loading
# ==========================================================


def find_metadata_json(data_dir: Path) -> Optional[Path]:
    candidates = [
        "group_schema.json",
        "fairness_groups.json",
        "fairness_metadata.json",
        "group_metadata.json",
        "item_group_metadata.json",
        "item_group_info.json",
        "fairness_assets.json",
        "dataset_stats.json",
    ]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    return None


def load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def infer_group_order(metadata: Optional[Dict[str, Any]]) -> List[str]:
    if metadata:
        matrix_columns = metadata.get("item_group_matrix_columns")
        if isinstance(matrix_columns, list):
            names = []
            for entry in matrix_columns:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("group_name")
                if isinstance(name, str) and name not in names:
                    names.append(name)
            if names:
                return names

        for key in ["group_names", "fairness_group_names", "leakage_probe_groups"]:
            names = metadata.get(key)
            if isinstance(names, list) and all(isinstance(x, str) for x in names):
                return list(names)

        for key in ["groups", "fairness_groups", "group_info", "group_metadata"]:
            groups = metadata.get(key)
            if isinstance(groups, list):
                out = []
                for g in groups:
                    if isinstance(g, dict):
                        name = g.get("name") or g.get("group_name") or g.get("key")
                        if isinstance(name, str):
                            out.append(name)
                if out:
                    return out

    return list(KNOWN_GROUP_ORDER)


def _parse_column_spec(spec: Any) -> Optional[List[int]]:
    if spec is None:
        return None
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, tuple):
        return [int(x) for x in spec]
    if isinstance(spec, dict):
        for key in ["columns", "cols", "indices", "column_indices"]:
            if key in spec:
                return _parse_column_spec(spec[key])
        if "start" in spec and "end" in spec:
            return list(range(int(spec["start"]), int(spec["end"])))
        if "offset" in spec and "dim" in spec:
            s = int(spec["offset"])
            return list(range(s, s + int(spec["dim"])))
        if "start_idx" in spec and "num_classes" in spec:
            s = int(spec["start_idx"])
            return list(range(s, s + int(spec["num_classes"])))
    return None


def columns_from_metadata(metadata: Optional[Dict[str, Any]], group_name: str) -> Optional[List[int]]:
    if not metadata:
        return None

    matrix_columns = metadata.get("item_group_matrix_columns")
    if isinstance(matrix_columns, list):
        cols = []
        for entry in matrix_columns:
            if not isinstance(entry, dict):
                continue
            if entry.get("group_name") == group_name and "column" in entry:
                cols.append(int(entry["column"]))
        if cols:
            return sorted(cols)

    for key in [
        "group_slices",
        "group_columns",
        "group_name_to_columns",
        "group_column_indices",
        "item_group_slices",
    ]:
        block = metadata.get(key)
        if isinstance(block, dict) and group_name in block:
            cols = _parse_column_spec(block[group_name])
            if cols is not None:
                return cols

    for key in ["groups", "group_info", "fairness_groups", "group_metadata"]:
        block = metadata.get(key)
        if isinstance(block, list):
            for entry in block:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("group_name") or entry.get("key")
                if name == group_name:
                    cols = _parse_column_spec(entry)
                    if cols is not None:
                        return cols
        elif isinstance(block, dict) and group_name in block:
            cols = _parse_column_spec(block[group_name])
            if cols is not None:
                return cols

    return None


def infer_contiguous_group_blocks(group_matrix: np.ndarray) -> List[List[int]]:
    mat = (group_matrix > 0).astype(np.int8)
    if mat.shape[0] > 1:
        mat = mat[1:]

    n_cols = mat.shape[1]
    blocks: List[List[int]] = []
    start = 0
    while start < n_cols:
        end = start + 1
        while end < n_cols:
            candidate = mat[:, start : end + 1]
            row_sums = candidate.sum(axis=1)
            if row_sums.size == 0 or int(row_sums.max()) <= 1:
                end += 1
            else:
                break
        blocks.append(list(range(start, end)))
        start = end
    return blocks


def labels_from_columns(group_matrix: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    valid_cols = [int(c) for c in cols if 0 <= int(c) < group_matrix.shape[1]]
    labels = np.full(group_matrix.shape[0], -100, dtype=np.int64)
    if not valid_cols:
        return labels

    sub = group_matrix[:, valid_cols]
    active = sub > 0
    has_label = active.any(axis=1)
    labels[has_label] = active[has_label].argmax(axis=1).astype(np.int64)
    labels[0] = -100
    return labels


def try_load_label_dict(data_dir: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    for name in ["item_group_labels.npz", "fairness_group_labels.npz", "group_labels.npz"]:
        path = data_dir / name
        if path.exists():
            with np.load(path, allow_pickle=False) as obj:
                for key in obj.files:
                    out[str(key)] = np.asarray(obj[key]).reshape(-1).astype(np.int64)
            if out:
                return out

    for name in ["item_group_labels.npy", "fairness_group_labels.npy", "group_labels.npy"]:
        path = data_dir / name
        if path.exists():
            try:
                obj = np.load(path, allow_pickle=False)
            except ValueError:
                print(
                    f"[Warning] Skipped unsafe object-array label file: {path}. "
                    "Use item_group_labels.npz or item_group_labels.json instead."
                )
                continue
            if isinstance(obj, np.ndarray) and obj.dtype != object:
                print(
                    f"[Warning] Skipped unlabeled numeric label array: {path}. "
                    "Use .npz/.json so each fairness group has an explicit key."
                )

    for name in ["item_group_labels.json", "fairness_group_labels.json", "group_labels.json", "fairness_groups.json"]:
        path = data_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, list):
                        out[str(key)] = np.asarray(value, dtype=np.int64).reshape(-1)
                    elif isinstance(value, dict):
                        numeric_items: List[Tuple[int, int]] = []
                        for item_id, label in value.items():
                            try:
                                numeric_items.append((int(item_id), int(label)))
                            except (TypeError, ValueError):
                                continue
                        if numeric_items:
                            max_item_id = max(item_id for item_id, _ in numeric_items)
                            arr = np.full(max_item_id + 1, -100, dtype=np.int64)
                            for item_id, label in numeric_items:
                                if item_id >= 0:
                                    arr[item_id] = label
                            if arr.size > 0:
                                arr[0] = -100
                            out[str(key)] = arr
                if out:
                    return out

    return out


def load_training_group_labels(
    data_dir: Path,
    requested_groups: Sequence[str],
    num_items: int,
    min_group_count: int = 5,
    drop_rare_classes: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, Dict[str, Any]]]:
    label_dict = try_load_label_dict(data_dir)
    metadata = load_json_if_exists(find_metadata_json(data_dir))

    matrix_path = data_dir / "item_group_matrix.npy"
    group_matrix = np.load(matrix_path).astype(np.float32) if matrix_path.exists() else None

    group_order = infer_group_order(metadata)
    inferred_blocks: Dict[str, List[int]] = {}
    if group_matrix is not None:
        blocks = infer_contiguous_group_blocks(group_matrix)
        for name, cols in zip(group_order, blocks):
            inferred_blocks[name] = cols

    labels_out: Dict[str, torch.Tensor] = {}
    num_classes_out: Dict[str, int] = {}
    info: Dict[str, Dict[str, Any]] = {}

    for group in requested_groups:
        labels: Optional[np.ndarray] = None
        source = "unknown"

        if group in label_dict:
            labels = label_dict[group]
            source = "label_file"
        elif group_matrix is not None:
            cols = columns_from_metadata(metadata, group)
            if cols is None:
                cols = inferred_blocks.get(group)
                if cols is not None:
                    source = "inferred_contiguous_block"
            else:
                source = "metadata_columns"

            if cols is not None:
                labels = labels_from_columns(group_matrix, cols)

        if labels is None:
            print(f"[Warning] Could not load labels for group={group}; skipped.")
            continue

        expected = num_items + 1
        if labels.shape[0] < expected:
            padded = np.full(expected, -100, dtype=np.int64)
            padded[: labels.shape[0]] = labels
            labels = padded
        elif labels.shape[0] > expected:
            labels = labels[:expected]

        labels[0] = -100

        valid = labels >= 0
        unique, counts = np.unique(labels[valid], return_counts=True)
        original_counts = {int(k): int(v) for k, v in zip(unique, counts)}

        if len(unique) <= 1:
            print(f"[Warning] group={group} has <=1 valid class; skipped. counts={original_counts}")
            continue

        if drop_rare_classes and min_group_count > 1:
            rare = {int(k) for k, v in original_counts.items() if int(v) < int(min_group_count)}
            if rare:
                labels[np.isin(labels, list(rare))] = -100
                valid = labels >= 0
                unique, counts = np.unique(labels[valid], return_counts=True)

        if len(unique) <= 1:
            print(f"[Warning] group={group} skipped after rare-class filtering.")
            continue

        remap = {int(old): i for i, old in enumerate(sorted(int(x) for x in unique))}
        remapped = np.full_like(labels, -100)
        for old, new in remap.items():
            remapped[labels == old] = new
        labels = remapped

        valid = labels >= 0
        unique2, counts2 = np.unique(labels[valid], return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique2, counts2)}
        num_classes = int(len(unique2))

        labels_out[group] = torch.from_numpy(labels.astype(np.int64))
        num_classes_out[group] = num_classes
        info[group] = {
            "source": source,
            "num_classes": num_classes,
            "class_counts": class_counts,
            "original_class_counts": original_counts,
            "num_valid_items": int(valid.sum()),
            "min_group_count": int(min_group_count),
            "drop_rare_classes": bool(drop_rare_classes),
        }

    if not labels_out:
        raise RuntimeError(
            "No usable fairness group labels were loaded. "
            "Please check item_group_labels.* or item_group_matrix.npy + metadata."
        )

    return labels_out, num_classes_out, info


def make_class_weight(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = labels[labels >= 0]
    counts = torch.bincount(valid, minlength=num_classes).float().clamp(min=1.0)
    inv = 1.0 / counts
    return (inv / inv.mean().clamp(min=1e-8)).float()


# ==========================================================
# Model helpers
# ==========================================================


def _constructor_accepts(cls: Any, name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


def build_fare_model(
    num_items: int,
    data_dir: Path,
    group_num_classes: Dict[str, int],
    cfg: Dict[str, Any],
) -> FARE:
    model_cfg = cfg.get("model", {})
    fair_cfg = cfg.get("fare", {})
    backbone = str(model_cfg.get("backbone", "sasrec")).lower()

    kwargs: Dict[str, Any] = {
        "num_items": num_items,
        "data_dir": data_dir,
        "group_num_classes": group_num_classes,
        "max_seq_len": int(model_cfg.get("max_seq_len", 50)),
        "hidden_size": int(model_cfg.get("hidden_size", 128)),
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "num_heads": int(model_cfg.get("num_heads", 2)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "activation": str(model_cfg.get("activation", "gelu")),
        "layer_norm_eps": float(model_cfg.get("layer_norm_eps", 1e-12)),
        "tie_output_embedding": bool(model_cfg.get("tie_output_embedding", True)),
        "mm_score_dim": int(fair_cfg.get("mm_score_dim", 64)),
        "fair_dim": int(fair_cfg.get("fair_dim", model_cfg.get("hidden_size", 128))),
        "projection_seed": int(fair_cfg.get("projection_seed", cfg.get("seed", 2026))),
        "use_text": bool(fair_cfg.get("use_text", True)),
        "use_vision": bool(fair_cfg.get("use_vision", True)),
        "cache_projected_features": bool(fair_cfg.get("cache_projected_features", True)),
        "feature_chunk_size": int(fair_cfg.get("feature_chunk_size", 32768)),
        "encoder_hidden_dim": int(fair_cfg.get("encoder_hidden_dim", 128)),
        "encoder_dropout": float(fair_cfg.get("encoder_dropout", 0.1)),
        "fair_weight_init": float(fair_cfg.get("fair_weight_init", 0.01)),
        "max_fair_weight": float(fair_cfg.get("max_fair_weight", 0.1)),
        "residual_score_weight": float(fair_cfg.get("residual_score_weight", 1.0)),
        "learnable_fair_weight": bool(fair_cfg.get("learnable_fair_weight", True)),
        "normalize_representations": bool(fair_cfg.get("normalize_representations", True)),
    }

    # Generic FARE should accept backbone_type/backbone.
    # Keep fallback compatibility with old SASRec-only FARE.
    if _constructor_accepts(FARE, "backbone_type"):
        kwargs["backbone_type"] = backbone
    elif _constructor_accepts(FARE, "backbone"):
        kwargs["backbone"] = backbone
    elif backbone != "sasrec":
        raise RuntimeError(
            "Current src/model_fare.py does not expose backbone_type/backbone in FARE.__init__. "
            "Please apply the generic FARE model update before using --backbone gru4rec/bert4rec."
        )

    valid_kwargs = {}
    try:
        sig = inspect.signature(FARE.__init__)
        for k, v in kwargs.items():
            if k in sig.parameters:
                valid_kwargs[k] = v
    except (TypeError, ValueError):
        valid_kwargs = kwargs

    return FARE(**valid_kwargs)


def load_initial_backbone(model: FARE, checkpoint_path: Optional[str], device: torch.device) -> None:
    if not checkpoint_path:
        return

    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Initial backbone checkpoint not found: {path}")

    ckpt = safe_torch_load(path, map_location=device)
    state = unwrap_state_dict(ckpt)

    if hasattr(model, "load_backbone_state_dict"):
        missing, unexpected = model.load_backbone_state_dict(state, strict=False)  # type: ignore[attr-defined]
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Loaded backbone checkpoint: {path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")


def freeze_id_backbone(model: torch.nn.Module) -> None:
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()  # type: ignore[attr-defined]
        return

    fair_keywords = (
        "fair_item_encoder",
        "user_fair_query",
        "raw_fair_weight",
    )
    for name, param in model.named_parameters():
        if not any(key in name for key in fair_keywords):
            param.requires_grad = False


def load_topk_items_from_npz(path: str) -> np.ndarray:
    """Load top-K recommended item ids from a .npz file.

    The function first tries common key names and then falls back to the first
    two-dimensional integer array. The expected output shape is [num_users, K].
    """
    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Exposure top-k file not found: {path_obj}")

    candidate_keys = [
        "topk_items",
        "items",
        "item_ids",
        "indices",
        "pred_items",
        "topk",
    ]

    with np.load(path_obj, allow_pickle=False) as obj:
        for key in candidate_keys:
            if key in obj.files:
                arr = obj[key]
                if arr.ndim == 2:
                    return arr.astype(np.int64)
                if arr.ndim == 1:
                    return arr.reshape(1, -1).astype(np.int64)

        for key in obj.files:
            arr = obj[key]
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int64)

        available = list(obj.files)

    raise ValueError(
        f"Cannot find a top-k item array in {path_obj}. Available keys: {available}"
    )


def _normalize_group_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return parse_list_arg(value) or []
    return [str(x).strip() for x in value if str(x).strip()]


def build_exposure_aware_class_weights(
    label_tensors: Dict[str, torch.Tensor],
    fair_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build group-level class weights from top-K exposure imbalance.

    For each configured group, we compute:

        q_g: target distribution over group classes.
             By default this is the item-catalog distribution.
        e_g: exposure distribution over group classes in a reference top-K file.

    The class weight is:

        raw_w_g = 1 + gamma * (q_g - e_g) / max(q_g, eps)

    Therefore under-exposed groups receive weights greater than 1, while
    over-exposed groups receive weights smaller than 1. These weights are
    clipped and normalized to mean 1 over valid classes.

    This function is intentionally called once before training. For formal
    experiments, the reference top-K file should come from validation or
    training exposure rather than the test split to avoid leakage.
    """
    mode = str(fair_cfg.get("fair_rec_reweight_mode", "exposure")).lower()
    gamma = float(fair_cfg.get("fair_rec_reweight_weight", 0.0))

    if mode != "exposure" or gamma <= 0:
        return {}

    topk_path = fair_cfg.get("fair_rec_exposure_topk_path", None)
    if not topk_path:
        raise ValueError(
            "FARE exposure reweighting requires --fair_rec_exposure_topk_path "
            "or fare.fair_rec_exposure_topk_path in the config."
        )

    topk_path_resolved = resolve_path(str(topk_path))
    if (
        "test" in topk_path_resolved.name.lower()
        and not bool(fair_cfg.get("allow_test_exposure_topk", False))
    ):
        raise ValueError(
            f"Refusing to use test-split exposure file for training: {topk_path_resolved}. "
            "Use a validation/train exposure file, or pass --allow_test_exposure_topk only for diagnostics."
        )

    groups = _normalize_group_list(
        fair_cfg.get("fair_rec_reweight_groups", None),
        default=["popularity_group"],
    )
    if not groups:
        groups = ["popularity_group"]

    target_type = str(fair_cfg.get("fair_rec_exposure_target", "catalog")).lower()
    exposure_k = int(fair_cfg.get("fair_rec_exposure_k", 10))
    w_min = float(fair_cfg.get("fair_rec_reweight_min", 0.5))
    w_max = float(fair_cfg.get("fair_rec_reweight_max", 2.0))
    if w_max < w_min:
        w_min, w_max = w_max, w_min

    topk_items_np = load_topk_items_from_npz(str(topk_path))
    if topk_items_np.ndim != 2:
        raise ValueError(f"Expected a 2-D top-k array, got shape={topk_items_np.shape}")

    if exposure_k > 0:
        topk_items_np = topk_items_np[:, : min(exposure_k, topk_items_np.shape[1])]

    topk_items_cpu = torch.as_tensor(topk_items_np.reshape(-1), dtype=torch.long)
    exposure_weights: Dict[str, torch.Tensor] = {}

    print(f"[EXPOSURE] topk_path={topk_path_resolved}")
    print(f"[EXPOSURE] topk_shape={tuple(topk_items_np.shape)} exposure_k={exposure_k}")
    print(f"[EXPOSURE] groups={groups} target={target_type} gamma={gamma}")
    print(f"[EXPOSURE] clip=[{w_min}, {w_max}]")

    for group in groups:
        if group not in label_tensors:
            print(f"[EXPOSURE][WARN] group={group} is not available in label_tensors; skip.")
            continue

        labels_all = label_tensors[group].detach().cpu().long()
        if labels_all.numel() <= 1:
            print(f"[EXPOSURE][WARN] group={group} has empty item labels; skip.")
            continue

        max_item_id = labels_all.numel() - 1
        valid_topk_mask = (topk_items_cpu > 0) & (topk_items_cpu <= max_item_id)
        invalid_count = int((~valid_topk_mask).sum().item())
        if invalid_count:
            print(f"[EXPOSURE][WARN] group={group} ignored {invalid_count} invalid/padding item ids.")
        safe_topk = topk_items_cpu[valid_topk_mask]
        if safe_topk.numel() == 0:
            print(f"[EXPOSURE][WARN] group={group} has no valid exposure item ids; skip.")
            continue

        exposed_labels = labels_all[safe_topk]
        exposed_labels = exposed_labels[exposed_labels >= 0]

        catalog_labels = labels_all[1:]  # item 0 is padding.
        catalog_labels = catalog_labels[catalog_labels >= 0]

        if exposed_labels.numel() == 0 or catalog_labels.numel() == 0:
            print(f"[EXPOSURE][WARN] group={group} has no valid catalog/exposure labels; skip.")
            continue

        num_classes = int(max(catalog_labels.max().item(), exposed_labels.max().item())) + 1
        catalog_counts = torch.bincount(catalog_labels, minlength=num_classes).float()
        exposure_counts = torch.bincount(exposed_labels, minlength=num_classes).float()
        valid = catalog_counts > 0

        if int(valid.sum().item()) <= 1:
            print(f"[EXPOSURE][WARN] group={group} has <=1 valid class; skip.")
            continue

        if target_type == "uniform":
            q = torch.zeros_like(catalog_counts)
            q[valid] = 1.0 / valid.sum().clamp_min(1)
        else:
            q = catalog_counts / catalog_counts.sum().clamp_min(1.0)

        e = exposure_counts / exposure_counts.sum().clamp_min(1.0)

        eps = 1e-8
        imbalance = torch.zeros_like(q)
        imbalance[valid] = (q[valid] - e[valid]) / q[valid].clamp_min(eps)

        raw_w = torch.ones_like(q)
        raw_w[valid] = 1.0 + gamma * imbalance[valid]
        raw_w = raw_w.clamp(min=w_min, max=w_max)

        # Keep the global scale stable.
        raw_w[valid] = raw_w[valid] / raw_w[valid].mean().clamp_min(1e-8)
        raw_w = torch.nan_to_num(raw_w, nan=1.0, posinf=w_max, neginf=w_min)

        exposure_weights[group] = raw_w.to(device=device, dtype=torch.float32)

        print(f"[EXPOSURE] group={group}")
        print(f"  catalog_counts={catalog_counts.tolist()}")
        print(f"  exposure_counts={exposure_counts.tolist()}")
        print(f"  target_dist={q.tolist()}")
        print(f"  exposure_dist={e.tolist()}")
        print(f"  class_weights={raw_w.tolist()}")
        print(
            f"  weight_min={raw_w[valid].min().item():.4f} "
            f"weight_max={raw_w[valid].max().item():.4f}"
        )

    if not exposure_weights:
        print("[EXPOSURE][WARN] No exposure-aware weights were created; CE weights will fall back to 1.")

    return exposure_weights


def build_fair_rec_sample_weights(
    targets: torch.Tensor,
    label_tensors: Dict[str, torch.Tensor],
    fair_cfg: Dict[str, Any],
    device: torch.device,
    exposure_class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Build FARE exposure-aware per-sample weights for recommendation CE loss."""
    gamma = float(fair_cfg.get("fair_rec_reweight_weight", 0.0))
    mode = str(fair_cfg.get("fair_rec_reweight_mode", "exposure")).lower()

    weights = torch.ones_like(targets, dtype=torch.float32, device=device)
    if gamma <= 0 or mode in {"none", "off", "false", "0"}:
        return weights
    if mode != "exposure":
        raise ValueError(f"FARE only supports fair_rec_reweight_mode='exposure', got {mode!r}.")

    groups = _normalize_group_list(
        fair_cfg.get("fair_rec_reweight_groups", None),
        default=["popularity_group"],
    )
    groups = [g for g in groups if g in label_tensors]
    if not groups:
        return weights

    components: List[torch.Tensor] = []

    for group in groups:
        label_all = label_tensors[group].to(device=device, non_blocking=True).long()
        labels = label_all[targets.long()]
        valid = labels >= 0

        if not bool(valid.any().item()):
            continue

        cw = (exposure_class_weights or {}).get(group)

        if cw is None or cw.numel() == 0:
            continue

        cw = cw.to(device=device, dtype=torch.float32)
        safe_labels = labels.clamp(min=0, max=cw.numel() - 1)
        group_weight = cw[safe_labels]

        # Normalize each group component within the current valid batch.
        group_weight = group_weight / group_weight[valid].mean().clamp_min(1e-8)

        comp = torch.ones_like(weights)
        comp[valid] = group_weight[valid]
        components.append(comp)

    if not components:
        return weights

    group_component = torch.stack(components, dim=0).mean(dim=0)

    # Exposure-aware class weights already include gamma at class level.
    weights = group_component

    w_min = float(fair_cfg.get("fair_rec_reweight_min", 0.5))
    w_max = float(fair_cfg.get("fair_rec_reweight_max", 2.0))
    if w_max < w_min:
        w_min, w_max = w_max, w_min

    weights = weights.clamp(min=w_min, max=w_max)
    weights = weights / weights.mean().clamp_min(1e-8)
    weights = torch.nan_to_num(weights, nan=1.0, posinf=w_max, neginf=w_min)

    return weights


def train_one_epoch(
    model: FARE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    group_labels: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    epoch: int,
    exposure_class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    model.train()

    fair_cfg = cfg.get("fare", {})
    train_cfg = cfg.get("train", {})

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    log_every = int(train_cfg.get("log_every", 100))

    all_groups = list(group_labels.keys())
    label_tensors = {g: group_labels[g].to(device=device, non_blocking=True) for g in all_groups}

    total_examples = 0
    sums = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "fair_weight": 0.0,
        "rec_weight_mean": 0.0,
        "rec_weight_max": 0.0,
    }
    grad_norms: List[float] = []

    for step, batch in enumerate(loader, start=1):
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        sequences = batch["sequences"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model.full_sort_scores(user_ids, sequences, lengths)

        per_sample_rec_loss = F.cross_entropy(
            logits,
            targets.long(),
            label_smoothing=label_smoothing,
            reduction="none",
        )
        rec_sample_weights = build_fair_rec_sample_weights(
            targets=targets,
            label_tensors=label_tensors,
            fair_cfg=fair_cfg,
            device=device,
            exposure_class_weights=exposure_class_weights,
        )
        rec_loss = (per_sample_rec_loss * rec_sample_weights).sum() / rec_sample_weights.sum().clamp_min(1e-8)
        loss = rec_loss

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
        sums["fair_weight"] += float(model.get_fair_weight()) * batch_size
        sums["rec_weight_mean"] += float(rec_sample_weights.detach().mean().cpu().item()) * batch_size
        sums["rec_weight_max"] += float(rec_sample_weights.detach().max().cpu().item()) * batch_size

        if log_every > 0 and step % log_every == 0:
            denom = max(total_examples, 1)
            print(
                f"  step={step:6d} "
                f"loss={sums['loss'] / denom:.6f} "
                f"rec={sums['rec_loss'] / denom:.6f} "
                f"rw_mean={sums['rec_weight_mean'] / denom:.4f} "
                f"rw_max={sums['rec_weight_max'] / denom:.4f}",
                flush=True,
            )

    denom = max(total_examples, 1)
    out = {k: v / denom for k, v in sums.items()}
    out["avg_grad_norm"] = float(np.mean(grad_norms)) if grad_norms else float("nan")
    out["active_group_count"] = 0.0
    return out


# ==========================================================
# Optional fairness evaluation from saved Top-K rankings
# ==========================================================


def _resolve_fairness_eval_groups(data_dir: Path, group_spec: Any) -> Optional[List[str]]:
    """Resolve eval.fairness_groups.

    Returns None for FairnessEvaluator's mandatory defaults. The string "all"
    expands to all groups present in fairness_groups.json.
    """
    if group_spec is None:
        return None

    if isinstance(group_spec, (list, tuple)):
        groups = [str(x).strip() for x in group_spec if str(x).strip()]
    else:
        spec = str(group_spec).strip()
        if not spec or spec.lower() in {"mandatory", "default", "required"}:
            return None
        if spec.lower() == "all":
            path = data_dir / "fairness_groups.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing fairness_groups.json: {path}")
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                raise ValueError(f"fairness_groups.json must be an object: {path}")
            return list(obj.keys())
        groups = parse_list_arg(spec) or []

    if not groups:
        return None

    path = data_dir / "fairness_groups.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            missing = [g for g in groups if g not in obj]
            if missing:
                raise KeyError(f"Unknown fairness eval groups: {missing}. Available: {list(obj)}")

    return groups


def run_saved_topk_fairness_eval(
    data_dir: Path,
    run_dir: Path,
    dataset: str,
    run_name: str,
    run_id: str,
    splits: Sequence[str],
    ks: Sequence[int],
    group_spec: Any = "all",
) -> Dict[str, Any]:
    """Evaluate fairness tables from topk_<split>.npz files saved by training."""
    try:
        from evaluator_fairness import FairnessEvaluator, save_eval_tables  # type: ignore
        from evaluator_ranking import load_ranking_result_npz  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[Warning] fairness evaluation imports failed: {exc}")
        return {"status": "skipped", "message": str(exc)}

    try:
        group_names = _resolve_fairness_eval_groups(data_dir, group_spec)
    except Exception as exc:
        print(f"[Warning] fairness group resolution failed: {exc}")
        return {"status": "skipped", "message": str(exc)}

    saved: Dict[str, Any] = {
        "status": "ok",
        "groups": group_names if group_names is not None else "mandatory",
        "splits": {},
    }

    for split in splits:
        ranking_path = run_dir / f"topk_{split}.npz"
        if not ranking_path.exists():
            saved["splits"][split] = {
                "status": "skipped",
                "message": f"missing ranking file: {ranking_path}",
            }
            continue

        try:
            ranking_result = load_ranking_result_npz(ranking_path)
            evaluator = FairnessEvaluator(
                data_dir=str(data_dir),
                ks=ks,
                group_names=group_names,
            )
            tables = evaluator.evaluate(ranking_result)

            prefix = f"{split}_fairness"
            save_eval_tables(tables, output_dir=str(run_dir), prefix=prefix)

            # Keep a compact JSON for quick inspection without requiring the
            # global fairness.csv aggregation step.
            flat: Dict[str, Any] = {
                "dataset": dataset,
                "split": split,
                "model_name": run_name,
                "run_id": run_id,
            }
            summary_df = tables.get("summary")
            if summary_df is not None and not summary_df.empty:
                for _, row in summary_df.iterrows():
                    k = int(row["k"])
                    for col in summary_df.columns:
                        if col == "k":
                            continue
                        val = row[col]
                        if pd.notna(val):
                            flat[f"summary@{k}/{col}"] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val

            exposure_df = tables.get("exposure")
            if exposure_df is not None and not exposure_df.empty and "group_value" in exposure_df.columns:
                agg = exposure_df[exposure_df["group_value"].astype(str) == "__aggregate__"]
                for _, row in agg.iterrows():
                    k = int(row["k"])
                    group_name = str(row["group_name"])
                    for col in [
                        "exposure_share_gap",
                        "exposure_share_gini",
                        "utility_aware_gap",
                        "utility_aware_l1",
                    ]:
                        if col in row and pd.notna(row[col]):
                            flat[f"{group_name}@{k}/{col}"] = float(row[col])

            save_json(flat, run_dir / f"{prefix}_flat_metrics.json")

            saved["splits"][split] = {
                "status": "ok",
                "ranking_file": str(ranking_path),
                "tables": {
                    name: str(run_dir / f"{prefix}_{name}.csv")
                    for name in tables.keys()
                },
                "flat_metrics": str(run_dir / f"{prefix}_flat_metrics.json"),
            }
        except Exception as exc:
            print(f"[Warning] fairness evaluation failed for split={split}: {exc}")
            saved["splits"][split] = {"status": "failed", "message": str(exc)}

    return saved


# ==========================================================
# Main
# ==========================================================


def main(
    default_config: str = "configs/fare_3090.yaml",
    default_method_name: str = "FARE",
) -> None:
    args = parse_args(default_config=default_config)

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

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    fair_cfg = cfg.get("fare", {})

    method_name = args.method_name or str(cfg.get("method_name", default_method_name))

    backbone = str(model_cfg.get("backbone", "sasrec")).lower()

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

    requested_groups = fair_cfg.get("groups") or FULL_GROUPS
    if isinstance(requested_groups, str):
        requested_groups = parse_list_arg(requested_groups) or FULL_GROUPS
    requested_groups = list(requested_groups)

    group_labels, group_num_classes, group_info = load_training_group_labels(
        data_dir=data_dir,
        requested_groups=requested_groups,
        num_items=num_items,
        min_group_count=int(fair_cfg.get("min_group_count", 5)),
        drop_rare_classes=bool(fair_cfg.get("drop_rare_classes", True)),
    )

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

    model = build_fare_model(
        num_items=num_items,
        data_dir=data_dir,
        group_num_classes=group_num_classes,
        cfg=cfg,
    ).to(device)

    init_ckpt = args.init_backbone_checkpoint or args.init_sasrec_checkpoint or cfg.get("init_backbone_checkpoint")
    load_initial_backbone(model, init_ckpt, device=device)

    if bool(args.freeze_id_backbone or fair_cfg.get("freeze_id_backbone", False)):
        freeze_id_backbone(model)
        print(f"Frozen ID backbone: {backbone}")

    optimizer = build_optimizer(model, cfg)

    run_name = str(cfg.get("run_name", "fare"))
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

    print("========== FARE Training ==========")
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
    print(f"residual_w:    {float(fair_cfg.get('residual_score_weight', 1.0))}")
    print(f"rec_reweight:  {float(fair_cfg.get('fair_rec_reweight_weight', 0.0))}")
    print(f"rec_rw_mode:   {fair_cfg.get('fair_rec_reweight_mode', 'exposure')}")
    print(f"rec_rw_groups: {fair_cfg.get('fair_rec_reweight_groups', None)}")
    print(f"exp_topk_path: {fair_cfg.get('fair_rec_exposure_topk_path', None)}")
    print(f"exp_target:    {fair_cfg.get('fair_rec_exposure_target', 'catalog')}")
    print(f"exp_k:         {int(fair_cfg.get('fair_rec_exposure_k', 10))}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    eval_every = int(train_cfg.get("eval_every", 1))
    min_delta = float(train_cfg.get("min_delta", 0.0))

    metric_for_best = str(eval_cfg.get("metric_for_best", "ndcg@20")).lower()
    ks = [int(x) for x in eval_cfg.get("ks", [5, 10, 20])]
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))
    save_topk_npz = bool(eval_cfg.get("save_topk_npz", True))

    exposure_class_weights = build_exposure_aware_class_weights(
        label_tensors=group_labels,
        fair_cfg=fair_cfg,
        device=device,
    )

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
            cfg=cfg,
            epoch=epoch,
            exposure_class_weights=exposure_class_weights,
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
            "avg_grad_norm": train_stats["avg_grad_norm"],
            "fair_weight": train_stats["fair_weight"],
            "rec_weight_mean": train_stats.get("rec_weight_mean", 1.0),
            "rec_weight_max": train_stats.get("rec_weight_max", 1.0),
            "active_group_count": train_stats["active_group_count"],
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
            f"fair_w={row['fair_weight']:.4f} "
            f"rw_mean={row['rec_weight_mean']:.4f} "
            f"rw_max={row['rec_weight_max']:.4f} "
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
        "fair_weight_init": float(fair_cfg.get("fair_weight_init", 0.01)),
        "max_fair_weight": float(fair_cfg.get("max_fair_weight", 0.1)),
        "residual_score_weight": float(fair_cfg.get("residual_score_weight", 1.0)),
        "fair_rec_reweight_weight": float(fair_cfg.get("fair_rec_reweight_weight", 0.0)),
        "fair_rec_reweight_groups": fair_cfg.get("fair_rec_reweight_groups", None),
        "fair_rec_reweight_mode": str(fair_cfg.get("fair_rec_reweight_mode", "exposure")),
        "fair_rec_reweight_min": float(fair_cfg.get("fair_rec_reweight_min", 0.5)),
        "fair_rec_reweight_max": float(fair_cfg.get("fair_rec_reweight_max", 2.0)),
        "fair_rec_exposure_topk_path": fair_cfg.get("fair_rec_exposure_topk_path", None),
        "fair_rec_exposure_target": str(fair_cfg.get("fair_rec_exposure_target", "catalog")),
        "fair_rec_exposure_k": int(fair_cfg.get("fair_rec_exposure_k", 10)),
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

    if bool(eval_cfg.get("run_fairness_eval", False)):
        if save_topk_npz and bool(eval_cfg.get("run_test_after_training", True)):
            fairness_splits = ["val", "test"]
            fairness_result = run_saved_topk_fairness_eval(
                data_dir=data_dir,
                run_dir=run_dir,
                dataset=str(dataset),
                run_name=run_name,
                run_id=run_id,
                splits=fairness_splits,
                ks=ks,
                group_spec=eval_cfg.get("fairness_groups", "all"),
            )
            summary["fairness_eval"] = fairness_result
        else:
            summary["fairness_eval"] = {
                "status": "skipped",
                "message": "Fairness evaluation requires run_test_after_training=true and save_topk_npz=true.",
            }

    save_json(summary, run_dir / "metrics_summary.json")

    print("========== Finished ==========")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
