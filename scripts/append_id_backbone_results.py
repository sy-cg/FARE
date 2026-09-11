# -*- coding: utf-8 -*-
"""
scripts/append_id_backbone_results.py

Collect metrics_summary.json files under results/ and build an overall accuracy table.

This script is intentionally method-name aware. Current naming convention:
    FARE = final exposure-aware multimodal fairness framework.

Typical usage:

    python scripts/append_id_backbone_results.py \
      --results_root results \
      --output results/overall.csv \
      --split test \
      --k 20 \
      --dedup \
      --backup

Append to existing output:

    python scripts/append_id_backbone_results.py \
      --results_root results \
      --output results/overall.csv \
      --split test \
      --k 20 \
      --append \
      --dedup \
      --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


BAD_RUN_IDS = {
    "text_vision_freezesion_freeze",
}

# ==========================================================
# Method name normalization
# ==========================================================

BACKBONE_DISPLAY_MAP = {
    "sasrec": "SASRec",
    "sasrec_id": "SASRec",
    "gru4rec": "GRU4Rec",
    "gru4rec_id": "GRU4Rec",
    "bert4rec": "BERT4Rec",
    "bert4rec_id": "BERT4Rec",
}

ID_METHOD_MAP = {
    "sasrec_id": "SASRec-ID",
    "gru4rec_id": "GRU4Rec-ID",
    "bert4rec_id": "BERT4Rec-ID",
}

MULTIMODAL_METHOD_MAP = {
    "vbpr": "VBPR",
    "bm3": "BM3",
    "freedom": "FREEDOM",
    "lattice": "LATTICE",
    "findrec": "FindRec",
}

METHOD_NAME_MAP = {
    # -------------------------
    # ID-only backbones
    # -------------------------
    "sasrec_id": "SASRec-ID",
    "gru4rec_id": "GRU4Rec-ID",
    "bert4rec_id": "BERT4Rec-ID",

    # -------------------------
    # Late fusion baselines
    # -------------------------
    "text_only_freeze": "LF-Text",
    "vision_only_freeze": "LF-Vision",
    "text_vision_freeze": "LF-TextVision",
    "textvision_freeze": "LF-TextVision",

    # -------------------------
    # Multimodal baselines
    # -------------------------
    "vbpr": "VBPR",
    "bm3": "BM3",
    "freedom": "FREEDOM",
    "lattice": "LATTICE",
    "findrec": "FindRec",
    # FARE final model
    "fare": "FARE",
}


DISPLAY_METHODS = {
    # ID-only
    "SASRec-ID",
    "GRU4Rec-ID",
    "BERT4Rec-ID",

    # Late fusion
    "LF-Text",
    "LF-Vision",
    "LF-TextVision",
    "LateFusion-SASRec",
    "LateFusion-GRU4Rec",
    "LateFusion-BERT4Rec",

    # Multimodal
    "VBPR",
    "BM3",
    "FREEDOM",
    "LATTICE",
    "FindRec",

    # Fairness baselines
    "Adv-SASRec",
    "Adv-GRU4Rec",
    "Adv-BERT4Rec",
    "SASRec + PopReweight",
    "GRU4Rec + PopReweight",
    "BERT4Rec + PopReweight",
    "SASRec + FairRerank",
    "GRU4Rec + FairRerank",
    "BERT4Rec + FairRerank",

    # ModalityDebias
    "VBPR + ModalityDebias",
    "BM3 + ModalityDebias",
    "FREEDOM + ModalityDebias",
    "LATTICE + ModalityDebias",
    "LateFusion-SASRec + ModalityDebias",
    "LateFusion-GRU4Rec + ModalityDebias",
    "LateFusion-BERT4Rec + ModalityDebias",

    # Ours
    "FARE",
    "FARE-SASRec",
    "FARE-GRU4Rec",
    "FARE-BERT4Rec",
}


def _safe_str(x) -> str:
    """Convert a row value to clean string."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _row_get(row, key: str, default: str = "") -> str:
    """Support both pd.Series and dict-like objects."""
    try:
        return _safe_str(row.get(key, default))
    except Exception:
        return default


def _collect_row_text(row) -> str:
    keys = [
        "method",
        "model_name",
        "run_id",
        "run_name",
        "backbone",
        "backbone_type",
        "base_model",
        "source_model_name",
        "source_method",
        "source_run_id",
    ]
    return " ".join(_row_get(row, k) for k in keys if _row_get(row, k)).lower()


def _infer_backbone(row) -> str:
    """Infer canonical backbone display name from row fields."""
    candidates = [
        _row_get(row, "backbone"),
        _row_get(row, "backbone_type"),
        _row_get(row, "base_backbone"),
        _row_get(row, "model"),
        _row_get(row, "source_backbone"),
        _row_get(row, "source_model_name"),
        _row_get(row, "model_name"),
        _row_get(row, "run_name"),
        _row_get(row, "run_id"),
        _row_get(row, "base_model"),
    ]

    joined = " ".join(candidates).lower()

    # Important: BERT4Rec / GRU4Rec before SASRec to avoid accidental fallback.
    if "bert4rec" in joined or "bert4rec_id" in joined:
        return "BERT4Rec"
    if "gru4rec" in joined or "gru4rec_id" in joined:
        return "GRU4Rec"
    if "sasrec" in joined or "sasrec_id" in joined:
        return "SASRec"

    explicit = _row_get(row, "backbone").lower()
    if explicit in BACKBONE_DISPLAY_MAP:
        return BACKBONE_DISPLAY_MAP[explicit]

    return ""


def _infer_base_model(row) -> str:
    """Infer base model for ModalityDebias."""
    base = _row_get(row, "base_model").lower()
    text = _collect_row_text(row)

    if base in {"latefusion_sasrec", "late_fusion_sasrec"} or "latefusion_sasrec" in text:
        return "LateFusion-SASRec"
    if base in {"latefusion_gru4rec", "late_fusion_gru4rec"} or "latefusion_gru4rec" in text:
        return "LateFusion-GRU4Rec"
    if base in {"latefusion_bert4rec", "late_fusion_bert4rec"} or "latefusion_bert4rec" in text:
        return "LateFusion-BERT4Rec"

    if base in MULTIMODAL_METHOD_MAP:
        return MULTIMODAL_METHOD_MAP[base]

    if "lattice" in text:
        return "LATTICE"
    if "findrec" in text:
        return "FindRec"
    if "freedom" in text:
        return "FREEDOM"
    if "bm3" in text:
        return "BM3"
    if "vbpr" in text:
        return "VBPR"

    if "latefusion" in text or "late_fusion" in text:
        backbone = _infer_backbone(row)
        return f"LateFusion-{backbone}" if backbone else "LateFusion"

    backbone = _infer_backbone(row)
    return backbone if backbone else ""


def _format_with_backbone(prefix: str, row) -> str:
    backbone = _infer_backbone(row)
    return f"{prefix}-{backbone}" if backbone else prefix


def _format_plus_backbone(row, suffix: str) -> str:
    backbone = _infer_backbone(row)
    return f"{backbone} + {suffix}" if backbone else suffix


def normalize_method_name(row: pd.Series) -> str:
    """
    Map raw model/run identifiers to paper-friendly method names.

    Covered families:
    - SASRec / GRU4Rec / BERT4Rec
    - LateFusion baselines
    - VBPR / BM3 / FREEDOM / LATTICE
    - Adv-* fairness baselines
    - * + PopReweight
    - * + FairRerank
    - * + ModalityDebias
    - FARE final model
    """
    raw_method = _row_get(row, "method")
    model_name = _row_get(row, "model_name")
    run_id = _row_get(row, "run_id")
    run_name = _row_get(row, "run_name")
    base_model = _row_get(row, "base_model")

    low_method = raw_method.lower()
    low_model = model_name.lower()
    low_run = run_id.lower()
    low_run_name = run_name.lower()
    low_base = base_model.lower()
    text = _collect_row_text(row)

    # Already normalized display name.
    if raw_method in DISPLAY_METHODS:
        return raw_method

    # ======================================================
    # 1. ModalityDebias variants
    # ======================================================
    if (
        "modality_debias" in text
        or "modalitydebias" in text
        or low_method in {"modalitydebias", "modality debias", "modality_debias"}
    ):
        base = _infer_base_model(row)
        return f"{base} + ModalityDebias" if base else "ModalityDebias"

    # ======================================================
    # 2. FairRerank variants
    # ======================================================
    if "fair_rerank" in text or "fairrerank" in text or low_method == "fairrerank":
        return _format_plus_backbone(row, "FairRerank")

    # ======================================================
    # 3. PopReweight variants
    # ======================================================
    if (
        "pop_reweight" in text
        or "popreweight" in text
        or "popularity_reweight" in text
        or low_method == "popreweight"
    ):
        return _format_plus_backbone(row, "PopReweight")

    # ======================================================
    # 4. Adversarial debiasing variants
    # ======================================================
    if (
        "adv_" in text
        or "_adv" in text
        or "adversarial" in text
        or low_method in {"adv", "adversarialdebias", "adversarial debias"}
    ):
        return _format_with_backbone("Adv", row)
    # ======================================================
    # 5. Exact known raw ids
    # ======================================================
    for key in [run_id, model_name, run_name, base_model, raw_method]:
        low_key = _safe_str(key).lower()
        if low_key in METHOD_NAME_MAP:
            mapped = METHOD_NAME_MAP[low_key]
            if mapped == "FARE":
                backbone = _infer_backbone(row)
                return f"FARE-{backbone}" if backbone else "FARE"
            return mapped

    # Partial exact map for run ids.
    for key, value in sorted(METHOD_NAME_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key and key in text:
            if value == "FARE":
                backbone = _infer_backbone(row)
                return f"FARE-{backbone}" if backbone else "FARE"
            return value

    # ======================================================
    # 7. FARE final model
    # ======================================================
    if "fare" in text:
        backbone = _infer_backbone(row)
        return f"FARE-{backbone}" if backbone else "FARE"

    # ======================================================
    # 8. Late fusion baselines
    # ======================================================
    if "text_only" in text:
        return "LF-Text"
    if "vision_only" in text:
        return "LF-Vision"
    if "text_vision" in text or "textvision" in text:
        return "LF-TextVision"

    if "latefusion_sasrec" in text or "late_fusion_sasrec" in text:
        return "LateFusion-SASRec"
    if "latefusion_gru4rec" in text or "late_fusion_gru4rec" in text:
        return "LateFusion-GRU4Rec"
    if "latefusion_bert4rec" in text or "late_fusion_bert4rec" in text:
        return "LateFusion-BERT4Rec"

    if "latefusion" in text or "late_fusion" in text:
        backbone = _infer_backbone(row)
        return f"LateFusion-{backbone}" if backbone else "LateFusion"

    # ======================================================
    # 9. Multimodal baselines
    # ======================================================
    if "lattice" in text:
        return "LATTICE"
    if "freedom" in text:
        return "FREEDOM"
    if "bm3" in text:
        return "BM3"
    if "vbpr" in text:
        return "VBPR"

    # ======================================================
    # 10. ID-only backbones
    # ======================================================
    if "bert4rec_id" in text or (("bert4rec" in text) and "modality" not in text):
        return "BERT4Rec-ID"
    if "gru4rec_id" in text or (("gru4rec" in text) and "modality" not in text):
        return "GRU4Rec-ID"
    if "sasrec_id" in text or (("sasrec" in text) and "modality" not in text):
        return "SASRec-ID"

    # ======================================================
    # Fallback
    # ======================================================
    return run_id if run_id else (model_name if model_name else raw_method)


METHOD_ORDER = {
    "SASRec-ID": 0,
    "GRU4Rec-ID": 1,
    "BERT4Rec-ID": 2,

    "LF-Text": 10,
    "LF-Vision": 11,
    "LF-TextVision": 12,
    "LateFusion-SASRec": 13,
    "LateFusion-GRU4Rec": 14,
    "LateFusion-BERT4Rec": 15,

    "VBPR": 30,
    "BM3": 31,
    "FREEDOM": 32,
    "LATTICE": 33,
    "FindRec": 34,

    "Adv-SASRec": 50,
    "Adv-GRU4Rec": 51,
    "Adv-BERT4Rec": 52,

    "SASRec + PopReweight": 60,
    "GRU4Rec + PopReweight": 61,
    "BERT4Rec + PopReweight": 62,

    "SASRec + FairRerank": 70,
    "GRU4Rec + FairRerank": 71,
    "BERT4Rec + FairRerank": 72,

    "LateFusion-SASRec + ModalityDebias": 80,
    "LateFusion-GRU4Rec + ModalityDebias": 81,
    "LateFusion-BERT4Rec + ModalityDebias": 82,
    "VBPR + ModalityDebias": 83,
    "BM3 + ModalityDebias": 84,
    "FREEDOM + ModalityDebias": 85,
    "LATTICE + ModalityDebias": 86,

    "FARE-SASRec": 120,
    "FARE-GRU4Rec": 121,
    "FARE-BERT4Rec": 122,
    "FARE": 123,
}

# ==========================================================
# Basic utilities
# ==========================================================


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be a dict: {path}")
    return obj


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def first_existing(flat: Dict[str, Any], candidates: Sequence[str]) -> float:
    lower_map = {str(k).lower(): k for k in flat.keys()}
    for c in candidates:
        if c in flat:
            return safe_float(flat[c])
        lk = c.lower()
        if lk in lower_map:
            return safe_float(flat[lower_map[lk]])
    return float("nan")


def metric_value(split_metrics: Dict[str, Any], name: str, k: int) -> float:
    flat = flatten_dict(split_metrics)
    candidates = [
        f"{name}@{k}",
        f"{name}_{k}",
        f"{name}{k}",
        name,
        f"metrics/{name}@{k}",
        f"eval/{name}@{k}",
    ]

    # common aliases
    if name == "recall":
        candidates += [f"recall_at_{k}", f"Recall@{k}"]
    elif name == "ndcg":
        candidates += [f"ndcg_at_{k}", f"NDCG@{k}", f"nDCG@{k}"]
    elif name == "mrr":
        candidates += [f"mrr_at_{k}", f"MRR@{k}"]
    elif name == "hit":
        candidates += [f"hit@{k}", f"hit_rate@{k}", f"hr@{k}", f"HR@{k}"]

    return first_existing(flat, candidates)


def infer_dataset_from_path(metrics_path: Path) -> Optional[str]:
    # results/<dataset>/<model_name>/<run_id>/metrics_summary.json
    try:
        rel = metrics_path.resolve().relative_to(PROJECT_ROOT.resolve())
        parts = rel.parts
    except Exception:
        parts = metrics_path.parts

    if "results" in parts:
        idx = parts.index("results")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return None


def infer_model_run_from_path(metrics_path: Path) -> tuple[Optional[str], Optional[str]]:
    # parent = run_id, grandparent = model_name
    run_id = metrics_path.parent.name if metrics_path.parent else None
    model_name = metrics_path.parent.parent.name if metrics_path.parent and metrics_path.parent.parent else None
    return model_name, run_id


# ==========================================================
# Naming
# ==========================================================


def infer_backbone(row: Dict[str, Any]) -> str:
    explicit = str(row.get("backbone", "") or row.get("backbone_type", "")).strip().lower()
    if explicit in {"sasrec", "gru4rec", "bert4rec"}:
        return explicit

    text = " ".join(
        str(row.get(k, ""))
        for k in ["method", "model_name", "run_id", "config", "init_backbone_checkpoint", "init_sasrec_checkpoint"]
    ).lower()

    if "gru4rec" in text or "gru" in text:
        return "gru4rec"
    if "bert4rec" in text or "bert" in text:
        return "bert4rec"
    if "sasrec" in text or "sas" in text:
        return "sasrec"

    return ""


def pretty_backbone(backbone: str) -> str:
    b = str(backbone).lower()
    if b == "sasrec":
        return "SASRec"
    if b == "gru4rec":
        return "GRU4Rec"
    if b == "bert4rec":
        return "BERT4Rec"
    return backbone



# ==========================================================
# Collection
# ==========================================================


def find_metric_files(results_root: Path) -> List[Path]:
    return sorted(results_root.glob("**/metrics_summary.json"))


def extract_rows_from_summary(path: Path, k: int, wanted_splits: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    summary = read_json(path)

    path_model_name, path_run_id = infer_model_run_from_path(path)
    dataset = str(summary.get("dataset") or infer_dataset_from_path(path) or "")
    model_name = str(summary.get("model_name") or path_model_name or "")
    run_id = str(summary.get("run_id") or path_run_id or "")

    if run_id in BAD_RUN_IDS:
        return []

    base: Dict[str, Any] = {
        "dataset": dataset,
        "model_name": model_name,
        "run_id": run_id,
        "path": str(path),
        "run_dir": str(path.parent),
        "best_metric": safe_float(summary.get("best_metric")),
        "best_epoch": safe_float(summary.get("best_epoch")),
        "best_metric_name": summary.get("best_metric_name", ""),
        "method_raw": summary.get("method") or summary.get("method_name") or "",
        "backbone": summary.get("backbone") or summary.get("backbone_type") or "",
    }

    base["backbone"] = infer_backbone({**summary, **base})
    base["method"] = normalize_method_name({**summary, **base})

    rows: List[Dict[str, Any]] = []

    split_candidates = []
    for split in ["train", "val", "valid", "validation", "test"]:
        if isinstance(summary.get(split), dict):
            split_candidates.append(split)

    split_alias = {
        "valid": "val",
        "validation": "val",
    }

    wanted = None
    if wanted_splits:
        wanted = {split_alias.get(s, s) for s in wanted_splits}

    for split_key in split_candidates:
        canonical_split = split_alias.get(split_key, split_key)
        if wanted is not None and canonical_split not in wanted:
            continue

        metrics = summary.get(split_key, {})
        if not isinstance(metrics, dict):
            continue

        row = dict(base)
        row["split"] = canonical_split
        row["recall"] = metric_value(metrics, "recall", k)
        row["ndcg"] = metric_value(metrics, "ndcg", k)
        row["mrr"] = metric_value(metrics, "mrr", k)
        row["hit"] = metric_value(metrics, "hit", k)

        # Keep all split metrics with prefix for debugging.
        flat = flatten_dict(metrics)
        for key, value in flat.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[f"metric/{key}"] = safe_float(value)

        rows.append(row)

    return rows


def remove_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_id" in out.columns:
        out = out[~out["run_id"].astype(str).isin(BAD_RUN_IDS)]
    return out.reset_index(drop=True)


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__method_order"] = out["method"].map(METHOD_ORDER).fillna(999).astype(int)
    out = out.sort_values(
        ["dataset", "split", "__method_order", "method", "backbone", "model_name", "run_id"],
        kind="mergesort",
    )
    return out.drop(columns=["__method_order"]).reset_index(drop=True)


def deduplicate(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    key_cols = [c for c in ["dataset", "split", "model_name", "run_id", "method", "backbone"] if c in df.columns]
    return df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build overall accuracy table from metrics_summary.json files")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/overall.csv")
    parser.add_argument("--dataset", type=str, default=None, help="Comma-separated datasets to keep")
    parser.add_argument("--split", type=str, default="test", help="Comma-separated splits or all")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--append", action="store_true", help="Append scanned rows to existing output")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate rows after building/appending")
    parser.add_argument("--dedup_keep", choices=["first", "last"], default="last")
    parser.add_argument("--backup", action="store_true", help="Backup existing output before writing")
    parser.add_argument("--preview_rows", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = resolve_path(args.results_root)
    output_path = resolve_path(args.output)

    datasets = parse_list_arg(args.dataset)
    splits = parse_list_arg(args.split)
    if splits and any(s.lower() == "all" for s in splits):
        splits = None

    metric_files = find_metric_files(results_root)
    rows: List[Dict[str, Any]] = []

    for path in metric_files:
        try:
            rows.extend(extract_rows_from_summary(path, k=args.k, wanted_splits=splits))
        except Exception as exc:
            print(f"[Warning] failed to parse {path}: {exc}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No metrics rows found under: {results_root}")

    if datasets:
        df = df[df["dataset"].astype(str).isin(datasets)]

    df = remove_bad_rows(df)

    if args.append and output_path.exists():
        old = pd.read_csv(output_path)
        df = pd.concat([old, df], ignore_index=True, sort=False)

    if args.dedup:
        df = deduplicate(df, keep=args.dedup_keep)

    df = sort_rows(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backup and output_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_suffix(output_path.suffix + f".bak_{ts}")
        shutil.copy2(output_path, backup_path)
        print(f"Backup written: {backup_path}")

    df.to_csv(output_path, index=False)

    print("Overall results saved:")
    print(f"  {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    preview_cols = [
        "dataset",
        "split",
        "method",
        "backbone",
        "model_name",
        "run_id",
        "recall",
        "ndcg",
        "mrr",
        "hit",
        "best_metric",
        "best_epoch",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].tail(min(args.preview_rows, len(df))).to_string(index=False))


if __name__ == "__main__":
    main()
