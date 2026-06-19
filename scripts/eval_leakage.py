# -*- coding: utf-8 -*-
"""
scripts/eval_leakage.py

Evaluate representation leakage for item-side fairness groups.

This script loads a trained run, extracts item representations, trains simple
linear probes to predict group labels, and saves leakage metrics.

Supported model dirs:
    results/<Dataset>/fare/<run_id>
    results/<Dataset>/sasrec_id/<run_id>
    results/<Dataset>/late_fusion_sasrec/<run_id>

Recommended examples:

FARE leakage on Video_Games:
    python scripts/eval_leakage.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/fare/fare_Video_Games_sasrec_<RUN_TAG> \
        --representations mm_raw,z_fair,id_embedding \
        --groups all \
        --append_global

SASRec-ID item embedding leakage:
    python scripts/eval_leakage.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/sasrec_id/20260515_194126 \
        --representations id_embedding \
        --groups all \
        --append_global

Late-fusion leakage:
    python scripts/eval_leakage.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/late_fusion_sasrec/text_vision_freeze \
        --representations mm_raw,id_embedding \
        --groups all \
        --append_global

Outputs:
    <run_dir>/leakage_probe.csv
    <run_dir>/leakage_probe_summary.csv
    results/leakage.csv              if --append_global
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_leakage import LeakageProbeEvaluator, add_relative_leakage_columns
from src.io_utils import safe_torch_load
from src.model_fare import FARE
from src.model_late_fusion import LateFusionSASRecID, build_or_load_projected_features
from src.model_sasrec import SASRecID

try:
    from src.evaluator_ranking import infer_num_items_from_processed_dir
except Exception:  # pragma: no cover
    infer_num_items_from_processed_dir = None


# ==========================================================
# Basic utilities
# ==========================================================


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return {} if obj is None else obj


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def infer_run_identity(run_dir: Path) -> Tuple[str, str]:
    """Return model_name, run_id from results/<Dataset>/<model_name>/<run_id>."""
    run_id = run_dir.name
    model_name = run_dir.parent.name
    return model_name, run_id


def resolve_data_dir(dataset: str, run_dir: Path, datasets_config: Path) -> Path:
    cfg_path = run_dir / "config_resolved.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        data_dir = cfg.get("data_dir")
        if data_dir:
            p = Path(data_dir)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return p

    datasets_cfg = load_yaml(datasets_config)
    datasets = datasets_cfg.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Dataset {dataset!r} not found in {datasets_config}. Available: {list(datasets)}")
    p = Path(datasets[dataset]["path"])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Processed data dir not found: {p}")
    return p


def infer_num_items(data_dir: Path) -> int:
    if infer_num_items_from_processed_dir is not None:
        return int(infer_num_items_from_processed_dir(str(data_dir)))
    # Fallback: use feature matrix shape [num_items+1, dim].
    for fname in ["item_popularity_train.npy", "text_features.npy", "vision_features.npy", "item_group_matrix.npy"]:
        path = data_dir / fname
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            return int(arr.shape[0] - 1)
    raise FileNotFoundError(f"Cannot infer num_items from {data_dir}")


# ==========================================================
# Fairness groups
# ==========================================================


def load_fairness_group_labels(
    data_dir: Path,
    group_names: Sequence[str],
    num_items: int,
    ignore_index: int = -1,
) -> Dict[str, np.ndarray]:
    path = data_dir / "fairness_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fairness_groups.json: {path}")
    raw = load_json(path)

    if len(group_names) == 1 and group_names[0].lower() == "all":
        selected = list(raw.keys())
    else:
        selected = list(group_names)

    labels: Dict[str, np.ndarray] = {}
    for group in selected:
        if group not in raw:
            raise KeyError(f"Group {group!r} not found in fairness_groups.json. Available: {list(raw)}")
        arr = np.full(num_items + 1, int(ignore_index), dtype=np.int64)
        values = raw[group]
        if not isinstance(values, dict):
            raise ValueError(f"fairness_groups.json[{group!r}] must be a dict item_id -> label")
        for item_str, label in values.items():
            try:
                item_id = int(item_str)
                if 0 <= item_id <= num_items:
                    arr[item_id] = int(label)
            except Exception:
                continue
        valid = arr[arr != ignore_index]
        if len(valid) == 0:
            print(f"[Warning] group={group} has no valid labels; skipped.")
            continue
        if len(np.unique(valid)) < 2:
            print(f"[Warning] group={group} has fewer than 2 classes; skipped.")
            continue
        labels[group] = arr
    if not labels:
        raise ValueError("No valid fairness group labels loaded")
    return labels


def derive_group_num_classes(labels: Dict[str, np.ndarray], ignore_index: int = -1) -> Dict[str, int]:
    out = {}
    for group, arr in labels.items():
        valid = arr[arr != ignore_index]
        out[group] = int(valid.max() + 1)
    return out


# ==========================================================
# Model loading
# ==========================================================


def load_config_resolved(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "config_resolved.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config_resolved.json: {path}")
    cfg = load_json(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"config_resolved.json must contain a JSON object: {path}")
    return cfg


def load_checkpoint_state(run_dir: Path, device: torch.device, checkpoint_name: str = "best_model.pt") -> Dict[str, torch.Tensor]:
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint state at {ckpt_path}")
    return state


def build_fare_model(
    cfg: Dict[str, Any],
    data_dir: Path,
    num_items: int,
    group_num_classes: Dict[str, int],
    device: torch.device,
) -> FARE:
    m = cfg.get("model", {})
    f = cfg.get("fare", {})

    def fair_value(key: str, default: Any) -> Any:
        return f.get(key, m.get(key, default))

    backbone = str(cfg.get("backbone", m.get("backbone", "sasrec"))).lower()

    model = FARE(
        num_items=num_items,
        data_dir=data_dir,
        group_num_classes=group_num_classes,
        max_seq_len=int(m.get("max_seq_len", 50)),
        hidden_size=int(m.get("hidden_size", 128)),
        num_layers=int(m.get("num_layers", 2)),
        num_heads=int(m.get("num_heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
        activation=str(m.get("activation", "gelu")),
        layer_norm_eps=float(m.get("layer_norm_eps", 1e-12)),
        tie_output_embedding=bool(m.get("tie_output_embedding", True)),
        backbone_type=backbone,
        mm_score_dim=int(fair_value("mm_score_dim", 64)),
        fair_dim=int(fair_value("fair_dim", 128)),
        projection_seed=int(fair_value("projection_seed", cfg.get("seed", 2026))),
        use_text=bool(fair_value("use_text", True)),
        use_vision=bool(fair_value("use_vision", True)),
        cache_projected_features=bool(fair_value("cache_projected_features", True)),
        feature_chunk_size=int(fair_value("feature_chunk_size", 32768)),
        encoder_hidden_dim=int(fair_value("encoder_hidden_dim", 128)),
        encoder_dropout=float(fair_value("encoder_dropout", 0.1)),
        fair_weight_init=float(fair_value("fair_weight_init", 0.01)),
        max_fair_weight=float(fair_value("max_fair_weight", 0.1)),
        residual_score_weight=float(fair_value("residual_score_weight", 1.0)),
        learnable_fair_weight=bool(fair_value("learnable_fair_weight", True)),
        normalize_representations=bool(fair_value("normalize_representations", True)),
    )
    model.to(device)
    return model


def build_sasrec_model(cfg: Dict[str, Any], num_items: int, device: torch.device) -> SASRecID:
    m = cfg.get("model", {})
    model = SASRecID(
        num_items=num_items,
        max_seq_len=int(m.get("max_seq_len", 50)),
        hidden_size=int(m.get("hidden_size", 128)),
        num_layers=int(m.get("num_layers", 2)),
        num_heads=int(m.get("num_heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
        activation=str(m.get("activation", "gelu")),
        layer_norm_eps=float(m.get("layer_norm_eps", 1e-12)),
        tie_output_embedding=bool(m.get("tie_output_embedding", True)),
    )
    model.to(device)
    return model


def build_late_fusion_model(cfg: Dict[str, Any], data_dir: Path, num_items: int, device: torch.device) -> LateFusionSASRecID:
    m = cfg.get("model", {})
    model = LateFusionSASRecID(
        num_items=num_items,
        data_dir=data_dir,
        max_seq_len=int(m.get("max_seq_len", 50)),
        hidden_size=int(m.get("hidden_size", 128)),
        num_layers=int(m.get("num_layers", 2)),
        num_heads=int(m.get("num_heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
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


def instantiate_model_for_run(
    run_dir: Path,
    model_name: str,
    cfg: Dict[str, Any],
    data_dir: Path,
    num_items: int,
    group_num_classes: Dict[str, int],
    device: torch.device,
    checkpoint_name: str = "best_model.pt",
) -> torch.nn.Module:
    if model_name == "fare":
        model = build_fare_model(cfg, data_dir, num_items, group_num_classes, device)
    elif model_name == "sasrec_id":
        model = build_sasrec_model(cfg, num_items, device)
    elif model_name == "late_fusion_sasrec":
        model = build_late_fusion_model(cfg, data_dir, num_items, device)
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}. Supported: fare, sasrec_id, late_fusion_sasrec")

    state = load_checkpoint_state(run_dir, device, checkpoint_name=checkpoint_name)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] missing keys while loading {model_name}: {len(missing)}; example={missing[:6]}")
    if unexpected:
        print(f"[Info] unexpected keys while loading {model_name}: {len(unexpected)}; example={unexpected[:6]}")
    model.eval()
    return model


# ==========================================================
# Representation extraction
# ==========================================================


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.numpy().astype(np.float32, copy=False)


def load_raw_projected_mm_features(data_dir: Path, cfg: Dict[str, Any], num_items: int) -> np.ndarray:
    m = cfg.get("model", {})
    f = cfg.get("fare", {})

    def fair_value(key: str, default: Any) -> Any:
        return f.get(key, m.get(key, default))

    mm_score_dim = int(fair_value("mm_score_dim", 64))
    projection_seed = int(fair_value("projection_seed", cfg.get("seed", 2026)))
    feature_chunk_size = int(fair_value("feature_chunk_size", 32768))
    parts = []

    if bool(fair_value("use_text", True)) and (data_dir / "text_features.npy").exists():
        text = build_or_load_projected_features(
            data_dir=data_dir,
            feature_file="text_features.npy",
            mask_file="text_mask.npy",
            output_dim=mm_score_dim,
            seed=projection_seed + 11,
            cache_prefix="text_features",
            use_cache=True,
            chunk_size=feature_chunk_size,
        )
        parts.append(text)
    if bool(fair_value("use_vision", True)) and (data_dir / "vision_features.npy").exists():
        vision = build_or_load_projected_features(
            data_dir=data_dir,
            feature_file="vision_features.npy",
            mask_file="vision_mask.npy",
            output_dim=mm_score_dim,
            seed=projection_seed + 29,
            cache_prefix="vision_features",
            use_cache=True,
            chunk_size=feature_chunk_size,
        )
        parts.append(vision)

    if not parts:
        raise FileNotFoundError(f"No projected text/vision features available under {data_dir}")
    mm = torch.cat(parts, dim=1).float()
    if mm.shape[0] != num_items + 1:
        raise ValueError(f"Raw projected MM feature shape mismatch: {mm.shape[0]} vs {num_items + 1}")
    mm[0].zero_()
    return tensor_to_numpy(mm)


def extract_representations(
    model: torch.nn.Module,
    model_name: str,
    data_dir: Path,
    cfg: Dict[str, Any],
    num_items: int,
    requested: Sequence[str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    reps: Dict[str, np.ndarray] = {}
    requested = [r.strip() for r in requested if r.strip()]

    with torch.no_grad():
        if "id_embedding" in requested:
            if hasattr(model, "item_embedding"):
                reps["id_embedding"] = tensor_to_numpy(model.item_embedding.weight)
            else:
                print("[Warning] model has no item_embedding; skip id_embedding")

        if "mm_raw" in requested:
            if hasattr(model, "mm_item_features"):
                reps["mm_raw"] = tensor_to_numpy(model.mm_item_features)
            elif hasattr(model, "text_item_features") or hasattr(model, "vision_item_features"):
                parts = []
                if hasattr(model, "text_item_features") and model.text_item_features.numel() > 0:
                    parts.append(model.text_item_features.detach().float().cpu())
                if hasattr(model, "vision_item_features") and model.vision_item_features.numel() > 0:
                    parts.append(model.vision_item_features.detach().float().cpu())
                if parts:
                    mm = torch.cat(parts, dim=1)
                    mm[0].zero_()
                    reps["mm_raw"] = tensor_to_numpy(mm)
                else:
                    reps["mm_raw"] = load_raw_projected_mm_features(data_dir, cfg, num_items)
            else:
                reps["mm_raw"] = load_raw_projected_mm_features(data_dir, cfg, num_items)

        if "z_fair" in requested:
            if hasattr(model, "encode_all_fair_items"):
                reps["z_fair"] = tensor_to_numpy(model.encode_all_fair_items())
            else:
                print(f"[Warning] representation z_fair not available for model_name={model_name}; skipped")

    for name, arr in list(reps.items()):
        if arr.shape[0] != num_items + 1:
            raise ValueError(f"Representation {name} row count mismatch: {arr.shape[0]} vs {num_items + 1}")
    return reps


# ==========================================================
# Output helpers
# ==========================================================


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, model_name, run_id, representation), sub in ok.groupby(["dataset", "model_name", "run_id", "representation"]):
        rows.append({
            "dataset": dataset,
            "model_name": model_name,
            "run_id": run_id,
            "representation": representation,
            "num_groups": int(sub["group_name"].nunique()),
            "mean_accuracy": float(sub["accuracy"].mean()),
            "mean_balanced_accuracy": float(sub["balanced_accuracy"].mean()),
            "mean_macro_f1": float(sub["macro_f1"].mean()),
            "mean_weighted_f1": float(sub["weighted_f1"].mean()),
            "mean_relative_reduction_macro_f1": float(sub.get("relative_reduction/macro_f1", pd.Series(dtype=float)).mean()),
        })
    return pd.DataFrame(rows)


def append_csv(df: pd.DataFrame, path: Path, dedup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, df], ignore_index=True)
        if dedup:
            key_cols = [
                c for c in [
                    "dataset", "model_name", "run_id", "representation", "group_name",
                    "classifier", "test_size", "seed",
                ] if c in merged.columns
            ]
            if key_cols:
                merged = merged.drop_duplicates(subset=key_cols, keep="last")
        merged.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate representation leakage with linear probes")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory, e.g. results/Video_Games/fare/fare_Video_Games_sasrec_<RUN_TAG>")
    parser.add_argument("--datasets_config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--representations", type=str, default="mm_raw,z_fair,id_embedding")
    parser.add_argument("--groups", type=str, default="all", help="Comma-separated group names or all")
    parser.add_argument("--output", type=str, default=None, help="Local output CSV. Default: <run_dir>/leakage_probe.csv")
    parser.add_argument("--summary_output", type=str, default=None, help="Local summary CSV. Default: <run_dir>/leakage_probe_summary.csv")
    parser.add_argument("--global_csv", type=str, default="results/leakage.csv")
    parser.add_argument("--append_global", action="store_true")

    parser.add_argument("--reference_representation", type=str, default="mm_raw")
    parser.add_argument("--fair_representation", type=str, default="z_fair")
    parser.add_argument("--bias_representation", type=str, default="")

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_class_count", type=int, default=5)
    parser.add_argument("--max_items", type=int, default=None, help="Optional cap on total probed items for speed")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Optional cap per class for speed/balance")
    parser.add_argument("--classifier", type=str, default="logreg", choices=["logreg", "dummy"])
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"])
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict", action="store_true", help="Fail if a requested representation is unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    datasets_config = resolve_path(args.datasets_config)
    data_dir = resolve_data_dir(args.dataset, run_dir, datasets_config)
    num_items = infer_num_items(data_dir)
    model_name, run_id = infer_run_identity(run_dir)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    requested_reps = parse_csv_list(args.representations) or []
    requested_groups = parse_csv_list(args.groups) or ["all"]
    labels = load_fairness_group_labels(data_dir, requested_groups, num_items, ignore_index=args.ignore_index)

    cfg = load_config_resolved(run_dir)
    trained_group_classes = cfg.get("group_num_classes")
    if not trained_group_classes:
        # Needed only for FARE construction. Use selected labels as fallback.
        trained_group_classes = derive_group_num_classes(labels, ignore_index=args.ignore_index)
    trained_group_classes = {str(k): int(v) for k, v in trained_group_classes.items()}

    print("=" * 90)
    print(f"Dataset:       {args.dataset}")
    print(f"Run dir:       {run_dir}")
    print(f"Model name:    {model_name}")
    print(f"Run id:        {run_id}")
    print(f"Data dir:      {data_dir}")
    print(f"Num items:     {num_items}")
    print(f"Device:        {device}")
    print(f"Groups:        {list(labels)}")
    print(f"Representations requested: {requested_reps}")
    print("=" * 90)

    # Load model and extract representations.
    model = instantiate_model_for_run(
        run_dir=run_dir,
        model_name=model_name,
        cfg=cfg,
        data_dir=data_dir,
        num_items=num_items,
        group_num_classes=trained_group_classes,
        device=device,
        checkpoint_name=args.checkpoint,
    )
    reps = extract_representations(
        model=model,
        model_name=model_name,
        data_dir=data_dir,
        cfg=cfg,
        num_items=num_items,
        requested=requested_reps,
        device=device,
    )
    if args.strict:
        missing = sorted(set(requested_reps) - set(reps))
        if missing:
            raise ValueError(f"Requested representations not available: {missing}")
    if not reps:
        raise ValueError("No representations were extracted")

    print("Representations extracted:")
    for name, arr in reps.items():
        print(f"  {name}: shape={arr.shape}, finite={np.isfinite(arr).all()}")

    evaluator = LeakageProbeEvaluator(
        test_size=args.test_size,
        seed=args.seed,
        min_class_count=args.min_class_count,
        max_items=args.max_items,
        max_samples_per_class=args.max_samples_per_class,
        classifier=args.classifier,
        max_iter=args.max_iter,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        n_jobs=args.n_jobs,
        standardize=not args.no_standardize,
        ignore_index=args.ignore_index,
    )
    df = evaluator.evaluate_many(reps, labels)
    df.insert(0, "dataset", args.dataset)
    df.insert(1, "model_name", model_name)
    df.insert(2, "run_id", run_id)
    df.insert(3, "run_dir", relative_to_project(run_dir))
    df["classifier"] = args.classifier
    df["test_size"] = args.test_size
    df["seed"] = args.seed
    df["min_class_count_arg"] = args.min_class_count
    df["max_items_arg"] = args.max_items
    df["max_samples_per_class_arg"] = args.max_samples_per_class

    df = add_relative_leakage_columns(
        df,
        reference_representation=args.reference_representation,
        fair_representation=args.fair_representation,
        bias_representation=args.bias_representation,
    )

    output_path = resolve_path(args.output) if args.output else run_dir / "leakage_probe.csv"
    summary_path = resolve_path(args.summary_output) if args.summary_output else run_dir / "leakage_probe_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = build_summary(df)
    summary.to_csv(summary_path, index=False)

    meta = {
        "dataset": args.dataset,
        "model_name": model_name,
        "run_id": run_id,
        "run_dir": relative_to_project(run_dir),
        "data_dir": relative_to_project(data_dir),
        "num_items": num_items,
        "groups": list(labels),
        "representations": list(reps),
        "classifier": args.classifier,
        "test_size": args.test_size,
        "seed": args.seed,
        "reference_representation": args.reference_representation,
        "fair_representation": args.fair_representation,
        "bias_representation": args.bias_representation,
    }
    save_json(meta, run_dir / "leakage_probe_meta.json")

    if args.append_global:
        global_path = resolve_path(args.global_csv)
        append_csv(df, global_path, dedup=True)
        print(f"Global leakage CSV updated: {global_path}")

    print("Leakage probe saved:")
    print(f"  Detail:  {output_path}")
    print(f"  Summary: {summary_path}")
    print("\nPreview:")
    cols = [
        "dataset", "model_name", "run_id", "representation", "group_name", "status",
        "accuracy", "balanced_accuracy", "macro_f1", "majority_baseline_accuracy",
        "relative_reduction/macro_f1",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, max_rows=80))


if __name__ == "__main__":
    main()
