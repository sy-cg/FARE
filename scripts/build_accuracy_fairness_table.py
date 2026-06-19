# -*- coding: utf-8 -*-
"""
scripts/build_accuracy_fairness_table.py

Merge accuracy metrics from results/overall.csv with fairness/diversity metrics
from results/fairness.csv.

Current naming convention:
    FARE = final exposure-aware multimodal fairness framework.

Typical usage:

    python scripts/build_accuracy_fairness_table.py \
      --overall results/overall.csv \
      --fairness results/fairness.csv \
      --output results/accuracy_fairness_table.csv \
      --split test \
      --k 20 \
      --add_delta \
      --delta_by_backbone
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
# Method normalization
# ==========================================================


BACKBONE_DISPLAY_MAP = {
    "sasrec": "SASRec",
    "sasrec_id": "SASRec",
    "gru4rec": "GRU4Rec",
    "gru4rec_id": "GRU4Rec",
    "bert4rec": "BERT4Rec",
    "bert4rec_id": "BERT4Rec",
}


METHOD_NAME_MAP = {
    # ID-only backbones
    "sasrec_id": "SASRec-ID",
    "gru4rec_id": "GRU4Rec-ID",
    "bert4rec_id": "BERT4Rec-ID",

    # Late fusion baselines
    "text_only_freeze": "LF-Text",
    "vision_only_freeze": "LF-Vision",
    "text_vision_freeze": "LF-TextVision",
    "textvision_freeze": "LF-TextVision",

    # Multimodal baselines
    "vbpr": "VBPR",
    "bm3": "BM3",
    "freedom": "FREEDOM",
    "lattice": "LATTICE",
    # FARE final model
    "fare": "FARE",
}


DISPLAY_METHODS = {
    "SASRec-ID",
    "GRU4Rec-ID",
    "BERT4Rec-ID",
    "LF-Text",
    "LF-Vision",
    "LF-TextVision",
    "LateFusion-SASRec",
    "LateFusion-GRU4Rec",
    "LateFusion-BERT4Rec",
    "VBPR",
    "BM3",
    "FREEDOM",
    "LATTICE",
    "Adv-SASRec",
    "Adv-GRU4Rec",
    "Adv-BERT4Rec",
    "SASRec + PopReweight",
    "GRU4Rec + PopReweight",
    "BERT4Rec + PopReweight",
    "SASRec + FairRerank",
    "GRU4Rec + FairRerank",
    "BERT4Rec + FairRerank",
    "VBPR + ModalityDebias",
    "BM3 + ModalityDebias",
    "FREEDOM + ModalityDebias",
    "LATTICE + ModalityDebias",
    "LateFusion-SASRec + ModalityDebias",
    "LateFusion-GRU4Rec + ModalityDebias",
    "LateFusion-BERT4Rec + ModalityDebias",
    "FARE",
    "FARE-SASRec",
    "FARE-GRU4Rec",
    "FARE-BERT4Rec",
}


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


DISPLAY_COLUMN_MAP = {
    "dataset": "Dataset",
    "split": "Split",
    "method": "Method",
    "backbone": "Backbone",
    "model_name": "Model",
    "run_id": "Run",
    "recall": "Recall",
    "ndcg": "NDCG",
    "mrr": "MRR",
    "hit": "Hit",
    "coverage": "Coverage",
    "entropy": "Entropy",
    "rec_gini": "RecGini",
    "avg_pop": "AvgPop",
    "tail_ratio": "TailRatio",
    "pop_gap": "Pop-U-Gap",
    "text_gap": "TextQ-U-Gap",
    "vision_gap": "VisionQ-U-Gap",
    "cat_gap": "Cat-U-Gap",
    "brand_gap": "Brand-U-Gap",
    "cluster_gap": "Cluster-U-Gap",
    "pop_l1": "Pop-U-L1",
    "text_l1": "TextQ-U-L1",
    "vision_l1": "VisionQ-U-L1",
    "cat_l1": "Cat-U-L1",
    "brand_l1": "Brand-U-L1",
    "cluster_l1": "Cluster-U-L1",
    "best_metric": "BestMetric",
    "best_epoch": "BestEpoch",
    "delta/baseline_method": "DeltaBaseline",
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


def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _row_get(row, key: str, default: str = "") -> str:
    try:
        return _safe_str(row.get(key, default))
    except Exception:
        return default


def _collect_row_text(row) -> str:
    keys = [
        "method",
        "method_name",
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
    return " ".join(_row_get(row, k) for k in keys if _row_get(row, k)).replace("-", "_").lower()


def _canonical_backbone_name(value: object) -> str:
    text = str(value or "").replace("-", "_").strip().lower()
    if "bert4rec" in text or text in {"bert", "bert4rec_id"}:
        return "bert4rec"
    if "gru4rec" in text or text in {"gru", "gru4rec_id"}:
        return "gru4rec"
    if "sasrec" in text or text in {"sas", "sasrec_id"}:
        return "sasrec"
    return ""


def infer_backbone_from_row(row: pd.Series) -> str:
    for c in ["backbone", "backbone_type", "base_backbone", "source_backbone"]:
        b = _canonical_backbone_name(row.get(c, ""))
        if b:
            return b

    text = _collect_row_text(row)
    return _canonical_backbone_name(text)


def pretty_backbone(backbone: str) -> str:
    b = str(backbone).lower()
    if b == "sasrec":
        return "SASRec"
    if b == "gru4rec":
        return "GRU4Rec"
    if b == "bert4rec":
        return "BERT4Rec"
    return backbone


def _infer_base_model(row) -> str:
    base = _row_get(row, "base_model").replace("-", "_").lower()
    text = _collect_row_text(row)

    if base in {"latefusion_sasrec", "late_fusion_sasrec"} or "latefusion_sasrec" in text or "late_fusion_sasrec" in text:
        return "LateFusion-SASRec"
    if base in {"latefusion_gru4rec", "late_fusion_gru4rec"} or "latefusion_gru4rec" in text or "late_fusion_gru4rec" in text:
        return "LateFusion-GRU4Rec"
    if base in {"latefusion_bert4rec", "late_fusion_bert4rec"} or "latefusion_bert4rec" in text or "late_fusion_bert4rec" in text:
        return "LateFusion-BERT4Rec"

    if base == "lattice" or "lattice" in text:
        return "LATTICE"
    if base == "freedom" or "freedom" in text:
        return "FREEDOM"
    if base == "bm3" or "bm3" in text:
        return "BM3"
    if base == "vbpr" or "vbpr" in text:
        return "VBPR"

    if "latefusion" in text or "late_fusion" in text:
        backbone = pretty_backbone(infer_backbone_from_row(row))
        return f"LateFusion-{backbone}" if backbone else "LateFusion"

    backbone = pretty_backbone(infer_backbone_from_row(row))
    return backbone if backbone else ""


def _format_with_backbone(prefix: str, row) -> str:
    backbone = pretty_backbone(infer_backbone_from_row(row))
    return f"{prefix}-{backbone}" if backbone else prefix


def _format_plus_backbone(row, suffix: str) -> str:
    backbone = pretty_backbone(infer_backbone_from_row(row))
    return f"{backbone} + {suffix}" if backbone else suffix


def normalize_method_name(row: pd.Series) -> str:
    run_id = _row_get(row, "run_id")
    model_name = _row_get(row, "model_name")
    raw_method = _row_get(row, "method") or _row_get(row, "method_name")

    if raw_method in DISPLAY_METHODS:
        return raw_method

    text = _collect_row_text(row)

    if run_id in BAD_RUN_IDS:
        return "LF-TextVision"

    # ModalityDebias variants first.
    if "modality_debias" in text or "modalitydebias" in text or "mdebias" in text:
        base = _infer_base_model(row)
        return f"{base} + ModalityDebias" if base else "ModalityDebias"

    # Post-processing / training fairness baselines.
    if "fair_rerank" in text or "fairrerank" in text:
        return _format_plus_backbone(row, "FairRerank")

    if "pop_reweight" in text or "popreweight" in text or "popularity_reweight" in text:
        return _format_plus_backbone(row, "PopReweight")

    if (
        text.startswith("adv_")
        or " adv_" in text
        or "_adv" in text
        or "adversarial" in text
        or raw_method.lower() in {"adv", "adversarialdebias", "adversarial debias"}
    ):
        return _format_with_backbone("Adv", row)
    # Exact keys.
    for key in [run_id, model_name, raw_method, run_id.lower(), model_name.lower(), raw_method.lower()]:
        key_norm = _safe_str(key).replace("-", "_").lower()
        if key_norm in METHOD_NAME_MAP:
            mapped = METHOD_NAME_MAP[key_norm]
            if mapped == "FARE":
                backbone = pretty_backbone(infer_backbone_from_row(row))
                return f"FARE-{backbone}" if backbone else "FARE"
            return mapped

    # Partial keys.
    for key, value in sorted(METHOD_NAME_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key and key in text:
            if value == "FARE":
                backbone = pretty_backbone(infer_backbone_from_row(row))
                return f"FARE-{backbone}" if backbone else "FARE"
            return value

    if "fare" in text:
        backbone = pretty_backbone(infer_backbone_from_row(row))
        return f"FARE-{backbone}" if backbone else "FARE"

    # Late fusion.
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
        backbone = pretty_backbone(infer_backbone_from_row(row))
        return f"LateFusion-{backbone}" if backbone else "LateFusion"

    # Multimodal baselines.
    if "lattice" in text:
        return "LATTICE"
    if "freedom" in text:
        return "FREEDOM"
    if "bm3" in text:
        return "BM3"
    if "vbpr" in text:
        return "VBPR"

    # ID-only backbones.
    if "bert4rec" in text:
        return "BERT4Rec-ID"
    if "gru4rec" in text:
        return "GRU4Rec-ID"
    if "sasrec" in text:
        return "SASRec-ID"

    return raw_method or run_id or model_name


# ==========================================================
# Loading and table construction
# ==========================================================


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    return pd.read_csv(path)


def remove_bad_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_id" in out.columns:
        out = out[~out["run_id"].astype(str).isin(BAD_RUN_IDS)]
    return out.reset_index(drop=True)


def prepare_overall(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = remove_bad_runs(df)

    required = {"dataset", "split", "model_name", "run_id"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"overall CSV missing required columns: {sorted(missing)}")

    out["method"] = out.apply(normalize_method_name, axis=1)
    out["backbone"] = out.apply(infer_backbone_from_row, axis=1)

    out["recall"] = get_first_col(out, ["recall", f"recall@{k}", f"metric/recall@{k}"])
    out["ndcg"] = get_first_col(out, ["ndcg", f"ndcg@{k}", f"metric/ndcg@{k}"])
    out["mrr"] = get_first_col(out, ["mrr", f"mrr@{k}", f"metric/mrr@{k}"])
    out["hit"] = get_first_col(out, ["hit", f"hit@{k}", f"hr@{k}", f"metric/hit@{k}", f"metric/hr@{k}"])

    out["best_metric"] = get_first_col(out, ["best_metric"])
    out["best_epoch"] = get_first_col(out, ["best_epoch"])

    return out


def build_fairness_compact(df: pd.DataFrame, k: int, groups: Sequence[str]) -> pd.DataFrame:
    f = remove_bad_runs(df)

    required = {"dataset", "split", "model_name", "run_id"}
    missing = required - set(f.columns)
    if missing:
        raise ValueError(f"fairness CSV missing required columns: {sorted(missing)}")

    f["method"] = f.apply(normalize_method_name, axis=1)
    f["backbone"] = f.apply(infer_backbone_from_row, axis=1)

    out = pd.DataFrame()
    out["dataset"] = f["dataset"].astype(str)
    out["split"] = f["split"].astype(str)
    out["model_name"] = f["model_name"].astype(str)
    out["run_id"] = f["run_id"].astype(str)
    out["method_fairness"] = f["method"].astype(str)
    out["backbone_fairness"] = f["backbone"].astype(str)

    out["coverage"] = get_first_col(f, [
        f"summary@{k}/catalog_coverage",
        f"summary@{k}/coverage",
        f"summary@{k}/item_coverage",
    ])
    out["entropy"] = get_first_col(f, [
        f"summary@{k}/recommendation_entropy",
        f"summary@{k}/entropy",
    ])
    out["rec_gini"] = get_first_col(f, [
        f"summary@{k}/recommendation_gini",
        f"summary@{k}/rec_gini",
        f"summary@{k}/gini",
    ])
    out["avg_pop"] = get_first_col(f, [
        f"summary@{k}/average_train_popularity",
        f"summary@{k}/avg_pop",
        f"summary@{k}/average_popularity",
    ])
    out["tail_ratio"] = get_first_col(f, [
        f"summary@{k}/tail_item_ratio",
        f"summary@{k}/tail_ratio",
    ])

    for group in groups:
        prefix = short_group_prefix(group)
        out[f"{prefix}_gap"] = get_first_col(f, [
            f"{group}@{k}/utility_aware_gap",
            f"{group}@{k}/gap",
        ])
        out[f"{prefix}_l1"] = get_first_col(f, [
            f"{group}@{k}/utility_aware_l1",
            f"{group}@{k}/l1",
        ])

    return out


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

    if methods:
        out = out[out["method"].astype(str).isin(methods)]

    return out.reset_index(drop=True)


def deduplicate(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    key_cols = [c for c in ["dataset", "split", "model_name", "run_id"] if c in df.columns]
    return df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)


def merge_accuracy_fairness(overall: pd.DataFrame, fairness: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["dataset", "split", "model_name", "run_id"]

    merged = overall.merge(
        fairness,
        on=merge_keys,
        how="left",
        validate="m:1",
    )

    # Prefer overall method/backbone, because metrics_summary.json is authoritative for model identity.
    if "method" not in merged.columns:
        merged["method"] = merged.get("method_fairness", "")
    if "backbone" not in merged.columns:
        merged["backbone"] = merged.get("backbone_fairness", "")

    return merged


# ==========================================================
# Delta logic
# ==========================================================


def _infer_backbone_for_delta(row: pd.Series) -> str:
    explicit = _canonical_backbone_name(row.get("backbone", ""))
    if explicit:
        return explicit
    text = " ".join(str(row.get(c, "")) for c in ["method", "model_name", "run_id", "base_model"])
    return _canonical_backbone_name(text)


def _baseline_method_for_row(
    row: pd.Series,
    default_baseline_method: str = "SASRec-ID",
    delta_by_backbone: bool = False,
) -> str:
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


def add_deltas(
    table: pd.DataFrame,
    baseline_method: str = "SASRec-ID",
    metric_cols: Optional[Sequence[str]] = None,
    delta_by_backbone: bool = False,
) -> pd.DataFrame:
    out = table.copy()

    if metric_cols is None:
        metric_cols = [
            "recall",
            "ndcg",
            "mrr",
            "hit",
            "coverage",
            "rec_gini",
            "avg_pop",
            "tail_ratio",
            "pop_gap",
            "text_gap",
            "vision_gap",
            "cat_gap",
            "brand_gap",
            "cluster_gap",
        ]
        metric_cols = [c for c in metric_cols if c in out.columns]

    if not metric_cols:
        print("[Warning] No metric columns found; deltas skipped.")
        return out

    baseline_rows = out[out["method"].astype(str).str.endswith("-ID")].copy()

    custom_baseline = out[out["method"] == baseline_method].copy()
    if not custom_baseline.empty:
        baseline_rows = pd.concat([baseline_rows, custom_baseline], ignore_index=True, sort=False)

    if baseline_rows.empty:
        print("[Warning] No ID baseline rows found; deltas skipped.")
        return out

    baseline_rows = baseline_rows.drop_duplicates(
        subset=["dataset", "split", "method"],
        keep="last",
    ).set_index(["dataset", "split", "method"])

    used_baselines: List[str] = []

    for col_i, col in enumerate(metric_cols):
        values = []
        for _, row in out.iterrows():
            row_baseline_method = _baseline_method_for_row(
                row,
                default_baseline_method=baseline_method,
                delta_by_backbone=delta_by_backbone,
            )
            key = (row["dataset"], row["split"], row_baseline_method)

            if col_i == 0:
                used_baselines.append(row_baseline_method)

            if key not in baseline_rows.index or col not in baseline_rows.columns:
                values.append(np.nan)
            else:
                values.append(row[col] - baseline_rows.loc[key, col])

        out[f"delta/{col}"] = values

    out["delta/baseline_method"] = used_baselines
    return out


# ==========================================================
# Output utilities
# ==========================================================


def sort_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["__method_order"] = out["method"].map(METHOD_ORDER).fillna(999).astype(int)

    sort_cols = ["dataset", "split", "__method_order", "method"]
    if "backbone" in out.columns:
        sort_cols.append("backbone")
    sort_cols.append("run_id")

    out = out.sort_values(sort_cols, kind="mergesort")
    return out.drop(columns=["__method_order"]).reset_index(drop=True)


def round_numeric(table: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = table.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def rename_for_display(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns={c: DISPLAY_COLUMN_MAP.get(c, c) for c in table.columns})


def save_markdown(table: pd.DataFrame, path: Path, decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = round_numeric(rename_for_display(table), decimals)
    with open(path, "w", encoding="utf-8") as f:
        f.write(display.to_markdown(index=False))
        f.write("\n")


def select_output_columns(df: pd.DataFrame, include_backbone: bool = True) -> pd.DataFrame:
    cols = ["dataset", "split", "method"]

    if include_backbone and "backbone" in df.columns:
        cols.append("backbone")

    cols += [
        "model_name",
        "run_id",
        "recall",
        "ndcg",
        "mrr",
        "hit",
        "coverage",
        "entropy",
        "rec_gini",
        "avg_pop",
        "tail_ratio",
        "pop_gap",
        "text_gap",
        "vision_gap",
        "cat_gap",
        "brand_gap",
        "cluster_gap",
        "pop_l1",
        "text_l1",
        "vision_l1",
        "cat_l1",
        "brand_l1",
        "cluster_l1",
        "best_metric",
        "best_epoch",
    ]

    existing = [c for c in cols if c in df.columns]
    delta_cols = [c for c in df.columns if c.startswith("delta/")]

    return df[existing + delta_cols]


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accuracy + fairness comparison table")
    parser.add_argument("--overall", type=str, default="results/overall.csv")
    parser.add_argument("--fairness", type=str, default="results/fairness.csv")
    parser.add_argument("--output", type=str, default="results/accuracy_fairness_table.csv")
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

    overall_path = resolve_path(args.overall)
    fairness_path = resolve_path(args.fairness)
    output_path = resolve_path(args.output)
    markdown_path = resolve_path(args.markdown) if args.markdown else None

    datasets = parse_list_arg(args.dataset)
    methods = parse_list_arg(args.methods)
    groups = parse_list_arg(args.groups) or DEFAULT_GROUPS

    overall_raw = load_csv(overall_path, "overall")
    fairness_raw = load_csv(fairness_path, "fairness")

    overall = prepare_overall(overall_raw, k=args.k)
    fairness = build_fairness_compact(fairness_raw, k=args.k, groups=groups)

    if args.dedup != "none":
        overall = deduplicate(overall, keep=args.dedup)
        fairness = deduplicate(fairness, keep=args.dedup)

    merged = merge_accuracy_fairness(overall, fairness)
    merged = filter_rows(merged, datasets=datasets, split=args.split, methods=methods)

    if merged.empty:
        raise ValueError("No rows left after filtering. Check --dataset, --split, and --methods.")

    if args.add_delta:
        merged = add_deltas(
            merged,
            baseline_method=args.baseline_method,
            delta_by_backbone=args.delta_by_backbone,
        )

    merged = sort_table(merged)
    table = select_output_columns(merged, include_backbone=not args.no_backbone)
    table = round_numeric(table, args.decimals)

    table_to_save = rename_for_display(table) if args.display_names else table

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_to_save.to_csv(output_path, index=False)

    if markdown_path is not None:
        save_markdown(table, markdown_path, decimals=args.decimals)

    print("Accuracy + fairness table saved:")
    print(f"  CSV: {output_path}")
    if markdown_path is not None:
        print(f"  Markdown: {markdown_path}")
    print(f"Rows: {len(table)}")
    print(f"Columns: {len(table.columns)}")
    print(table_to_save.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
