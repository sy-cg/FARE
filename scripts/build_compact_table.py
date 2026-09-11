# -*- coding: utf-8 -*-
"""
scripts/build_compact_table.py

Build compact fairness/diversity comparison tables from results/fairness.csv.

Current naming convention:
    FARE = final exposure-aware multimodal fairness framework.

Typical usage:

    python scripts/build_compact_table.py \
      --input results/fairness.csv \
      --output results/compact_comparison.csv \
      --markdown results/compact_comparison.md \
      --split test \
      --k 20 \
      --add_delta

Specific dataset:

    python scripts/build_compact_table.py \
      --input results/fairness.csv \
      --output results/compact_video.csv \
      --dataset Video_Games \
      --split test \
      --k 20 \
      --add_delta
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_GROUPS = [
    "popularity_group",
    "text_quality_group",
    "vision_quality_group",
    "category_proxy_group",
    "brand_store_proxy_group",
    "multimodal_cluster_proxy_group",
]


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


DISPLAY_COLUMN_MAP = {
    "dataset": "Dataset",
    "split": "Split",
    "method": "Method",
    "backbone": "Backbone",
    "model_name": "Model",
    "run_id": "Run",
    "coverage": "Coverage",
    "entropy": "Entropy",
    "rec_gini": "RecGini",
    "avg_pop": "AvgPop",
    "tail_ratio": "TailRatio",
    "pop_gap": "Pop-U-Gap",
    "pop_l1": "Pop-U-L1",
    "text_gap": "TextQ-U-Gap",
    "text_l1": "TextQ-U-L1",
    "vision_gap": "VisionQ-U-Gap",
    "vision_l1": "VisionQ-U-L1",
    "cat_gap": "Cat-U-Gap",
    "cat_l1": "Cat-U-L1",
    "brand_gap": "Brand-U-Gap",
    "brand_l1": "Brand-U-L1",
    "cluster_gap": "Cluster-U-Gap",
    "cluster_l1": "Cluster-U-L1",
}


# ==========================================================
# Utilities
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


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_first_col(df: pd.DataFrame, candidates: Sequence[str], default: float = np.nan) -> pd.Series:
    lower_to_col = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col in df.columns:
            return safe_numeric(df[col])
        low = col.lower()
        if low in lower_to_col:
            return safe_numeric(df[lower_to_col[low]])
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def short_group_prefix(group_name: str) -> str:
    mapping = {
        "popularity_group": "pop",
        "text_quality_group": "text",
        "vision_quality_group": "vision",
        "category_proxy_group": "cat",
        "brand_store_proxy_group": "brand",
        "multimodal_cluster_proxy_group": "cluster",
        "modality_availability_group": "modality",
    }
    return mapping.get(group_name, re.sub(r"_group$", "", group_name))


def infer_backbone_from_row(row: pd.Series) -> str:
    explicit = str(row.get("backbone", "") or row.get("backbone_type", "")).strip().lower()
    if explicit in {"sasrec", "gru4rec", "bert4rec"}:
        return explicit

    text = " ".join(str(row.get(c, "")) for c in ["method", "model_name", "run_id"]).lower()
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
# Table construction
# ==========================================================


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "split", "model_name", "run_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    return df


def remove_bad_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_id" in out.columns:
        out = out[~out["run_id"].astype(str).isin(BAD_RUN_IDS)]
    return out.reset_index(drop=True)


def filter_rows(
    df: pd.DataFrame,
    datasets: Optional[Sequence[str]],
    split: Optional[str],
    methods: Optional[Sequence[str]],
) -> pd.DataFrame:
    out = df.copy()

    if datasets:
        out = out[out["dataset"].astype(str).isin(datasets)]

    if split and split.lower() != "all":
        out = out[out["split"].astype(str) == split]

    out["method"] = out.apply(normalize_method_name, axis=1)

    if "backbone" not in out.columns:
        out["backbone"] = out.apply(infer_backbone_from_row, axis=1)
    else:
        out["backbone"] = out.apply(infer_backbone_from_row, axis=1)

    if methods:
        out = out[out["method"].astype(str).isin(methods)]

    return out.reset_index(drop=True)


def deduplicate_runs(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    key_cols = [c for c in ["dataset", "split", "model_name", "run_id"] if c in df.columns]
    return df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)


def build_compact_table(df: pd.DataFrame, k: int, groups: Sequence[str], include_backbone: bool = True) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset"] = df["dataset"].astype(str)
    out["split"] = df["split"].astype(str)
    out["method"] = df["method"].astype(str)

    if include_backbone and "backbone" in df.columns:
        out["backbone"] = df["backbone"].astype(str)

    out["model_name"] = df["model_name"].astype(str)
    out["run_id"] = df["run_id"].astype(str)

    out["coverage"] = get_first_col(df, [
        f"summary@{k}/catalog_coverage",
        f"summary@{k}/coverage",
        f"summary@{k}/item_coverage",
    ])
    out["entropy"] = get_first_col(df, [
        f"summary@{k}/recommendation_entropy",
        f"summary@{k}/entropy",
    ])
    out["rec_gini"] = get_first_col(df, [
        f"summary@{k}/recommendation_gini",
        f"summary@{k}/rec_gini",
        f"summary@{k}/gini",
    ])
    out["avg_pop"] = get_first_col(df, [
        f"summary@{k}/average_train_popularity",
        f"summary@{k}/avg_pop",
        f"summary@{k}/average_popularity",
    ])
    out["tail_ratio"] = get_first_col(df, [
        f"summary@{k}/tail_item_ratio",
        f"summary@{k}/tail_ratio",
    ])

    for group in groups:
        prefix = short_group_prefix(group)
        out[f"{prefix}_gap"] = get_first_col(df, [
            f"{group}@{k}/utility_aware_gap",
            f"{group}@{k}/gap",
        ])
        out[f"{prefix}_l1"] = get_first_col(df, [
            f"{group}@{k}/utility_aware_l1",
            f"{group}@{k}/l1",
        ])

    return out


def sort_table(table: pd.DataFrame) -> pd.DataFrame:
    method_order = {
        # ==================================================
        # ID-only backbones
        # ==================================================
        "SASRec-ID": 0,
        "GRU4Rec-ID": 1,
        "BERT4Rec-ID": 2,

        # ==================================================
        # Simple late fusion
        # ==================================================
        "LF-Text": 10,
        "LF-Vision": 11,
        "LF-TextVision": 12,
        "LateFusion-SASRec": 13,
        "LateFusion-GRU4Rec": 14,
        "LateFusion-BERT4Rec": 15,

        # ==================================================
        # Multimodal recommendation baselines
        # ==================================================
        "VBPR": 30,
        "BM3": 31,
        "FREEDOM": 32,
        "LATTICE": 33,
        "FindRec": 34,

        # ==================================================
        # Adversarial fairness baselines
        # ==================================================
        "Adv-SASRec": 50,
        "Adv-GRU4Rec": 51,
        "Adv-BERT4Rec": 52,

        # ==================================================
        # PopReweight fairness baselines
        # ==================================================
        "SASRec + PopReweight": 60,
        "GRU4Rec + PopReweight": 61,
        "BERT4Rec + PopReweight": 62,

        # ==================================================
        # FairRerank fairness baselines
        # ==================================================
        "SASRec + FairRerank": 70,
        "GRU4Rec + FairRerank": 71,
        "BERT4Rec + FairRerank": 72,

        # ==================================================
        # ModalityDebias baselines
        # ==================================================
        "LateFusion-SASRec + ModalityDebias": 80,
        "LateFusion-GRU4Rec + ModalityDebias": 81,
        "LateFusion-BERT4Rec + ModalityDebias": 82,
        "VBPR + ModalityDebias": 83,
        "BM3 + ModalityDebias": 84,
        "FREEDOM + ModalityDebias": 85,
        "LATTICE + ModalityDebias": 86,

        # ==================================================
        # Ours
        # ==================================================
        "FARE-SASRec": 120,
        "FARE-GRU4Rec": 121,
        "FARE-BERT4Rec": 122,
        "FARE": 123,
    }

    out = table.copy()
    out["__method_order"] = out["method"].map(method_order).fillna(999).astype(int)

    sort_cols = []
    for c in ["dataset", "split", "__method_order", "method", "run_id"]:
        if c in out.columns:
            sort_cols.append(c)

    out = (
        out.sort_values(sort_cols, kind="mergesort")
        .drop(columns=["__method_order"])
        .reset_index(drop=True)
    )
    return out


def _canonical_backbone_name(value: object) -> str:
    text = str(value or "").strip().lower()
    if "bert4rec" in text or text in {"bert", "bert4rec_id"}:
        return "bert4rec"
    if "gru4rec" in text or text in {"gru", "gru4rec_id"}:
        return "gru4rec"
    if "sasrec" in text or text in {"sas", "sasrec_id"}:
        return "sasrec"
    return ""


def _infer_backbone_for_delta(row: pd.Series) -> str:
    """
    Infer backbone for delta baseline selection.

    Prefer the explicit backbone column, then fallback to method/model/run text.
    """
    explicit = _canonical_backbone_name(row.get("backbone", ""))
    if explicit:
        return explicit

    text = " ".join(
        str(row.get(c, ""))
        for c in ["method", "model_name", "run_id"]
    ).lower()

    return _canonical_backbone_name(text)


def _baseline_method_for_row(
    row: pd.Series,
    default_baseline_method: str = "SASRec-ID",
    delta_by_backbone: bool = False,
) -> str:
    """
    Choose the baseline method for one row.

    If delta_by_backbone=False:
        every row compares against default_baseline_method.

    If delta_by_backbone=True:
        SASRec-family rows compare against SASRec-ID,
        GRU4Rec-family rows compare against GRU4Rec-ID,
        BERT4Rec-family rows compare against BERT4Rec-ID.

    Rows with no inferable backbone, e.g. VBPR/BM3/FREEDOM/LATTICE,
    fallback to default_baseline_method.
    """
    if not delta_by_backbone:
        return default_baseline_method

    backbone = _infer_backbone_for_delta(row)

    if backbone == "sasrec":
        return "SASRec-ID"
    if backbone == "gru4rec":
        return "GRU4Rec-ID"
    if backbone == "bert4rec":
        return "BERT4Rec-ID"

    return default_baseline_method


def add_baseline_deltas(
    table: pd.DataFrame,
    baseline_method: str = "SASRec-ID",
    delta_by_backbone: bool = False,
) -> pd.DataFrame:
    """
    Add deltas against an ID baseline.

    Normal mode:
        every method compares against baseline_method, e.g. SASRec-ID.

    Backbone-aware mode:
        SASRec-family methods   compare against SASRec-ID
        GRU4Rec-family methods  compare against GRU4Rec-ID
        BERT4Rec-family methods compare against BERT4Rec-ID

    Non-backbone multimodal methods, e.g. VBPR/BM3/FREEDOM/LATTICE,
    fallback to baseline_method.
    """
    out = table.copy()

    metric_cols = [
        c for c in out.columns
        if c not in {"dataset", "split", "method", "backbone", "model_name", "run_id"}
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    if not metric_cols:
        print("[Warning] No numeric metric columns found; deltas skipped.")
        return out

    key_cols = ["dataset", "split"]
    missing_key_cols = [c for c in key_cols if c not in out.columns]
    if missing_key_cols:
        print(f"[Warning] Missing key columns {missing_key_cols}; deltas skipped.")
        return out

    # Build lookup:
    #   (dataset, split, method) -> baseline metric row
    baseline_rows = out[out["method"].astype(str).str.endswith("-ID")].copy()

    # Also include custom baseline_method if it does not end with -ID.
    custom_baseline = out[out["method"] == baseline_method].copy()
    if not custom_baseline.empty:
        baseline_rows = pd.concat([baseline_rows, custom_baseline], ignore_index=True, sort=False)

    if baseline_rows.empty:
        print("[Warning] No ID baseline rows found; deltas skipped.")
        return out

    baseline_rows = baseline_rows.drop_duplicates(
        subset=["dataset", "split", "method"],
        keep="last",
    )
    baseline_rows = baseline_rows.set_index(["dataset", "split", "method"])

    # Record which baseline each row uses. Useful for debugging / paper tables.
    used_baselines = []

    for col in metric_cols:
        values = []

        for _, row in out.iterrows():
            row_baseline_method = _baseline_method_for_row(
                row,
                default_baseline_method=baseline_method,
                delta_by_backbone=delta_by_backbone,
            )

            key = (row["dataset"], row["split"], row_baseline_method)

            if col == metric_cols[0]:
                used_baselines.append(row_baseline_method)

            if key not in baseline_rows.index or col not in baseline_rows.columns:
                values.append(np.nan)
            else:
                values.append(row[col] - baseline_rows.loc[key, col])

        out[f"delta/{col}"] = values

    out["delta/baseline_method"] = used_baselines
    return out


def rename_for_display(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns={c: DISPLAY_COLUMN_MAP.get(c, c) for c in table.columns})


def round_numeric(table: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = table.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def save_markdown(table: pd.DataFrame, path: Path, decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = round_numeric(rename_for_display(table), decimals)
    with open(path, "w", encoding="utf-8") as f:
        f.write(display.to_markdown(index=False))
        f.write("\n")


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact fairness comparison table")
    parser.add_argument("--input", type=str, default="results/fairness.csv")
    parser.add_argument("--output", type=str, default="results/compact_comparison.csv")
    parser.add_argument("--markdown", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--add_delta", action="store_true")
    parser.add_argument("--baseline_method", type=str, default="SASRec-ID")
    parser.add_argument("--delta_by_backbone", action="store_true")
    parser.add_argument("--dedup", choices=["first", "last", "none"], default="last")
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--display_names", action="store_true")
    parser.add_argument("--no_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    markdown_path = resolve_path(args.markdown) if args.markdown else None

    datasets = parse_list_arg(args.dataset)
    methods = parse_list_arg(args.methods)
    groups = parse_list_arg(args.groups) or DEFAULT_GROUPS

    df = load_input(input_path)
    df = remove_bad_runs(df)
    df = filter_rows(df, datasets=datasets, split=args.split, methods=methods)

    if df.empty:
        raise ValueError("No rows left after filtering. Check --dataset, --split, and --methods.")

    if args.dedup != "none":
        df = deduplicate_runs(df, keep=args.dedup)

    table = build_compact_table(
        df,
        k=args.k,
        groups=groups,
        include_backbone=not args.no_backbone,
    )
    table = sort_table(table)

    if args.add_delta:
        table = add_baseline_deltas(
            table,
            baseline_method=args.baseline_method,
            delta_by_backbone=args.delta_by_backbone,
        )

    table_to_save = round_numeric(table, args.decimals)
    if args.display_names:
        table_to_save = rename_for_display(table_to_save)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_to_save.to_csv(output_path, index=False)

    if markdown_path is not None:
        save_markdown(table, markdown_path, decimals=args.decimals)

    print("Compact table saved:")
    print(f"  CSV: {output_path}")
    if markdown_path is not None:
        print(f"  Markdown: {markdown_path}")
    print(f"Rows: {len(table)}")
    print(f"Columns: {len(table.columns)}")
    print(table_to_save.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
