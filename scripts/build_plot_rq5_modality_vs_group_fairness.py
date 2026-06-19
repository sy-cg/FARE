#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build RQ5 results:
Is modality-level debiasing sufficient for group-aware fairness?

Main-text figure:
    Method-level 2x2 grouped bar chart:
        (a) NDCG@10
        (b) PopGap@10
        (c) BrandGap@10
        (d) ClusterGap@10

    X-axis:
        VBPR, BM3, FREEDOM, LATTICE

    Bars:
        Base vs +ModalityDebias

    Dashed reference line:
        FARE + SASRec average across datasets

Pairwise delta table:
    Dataset × multimodal family comparison:
        Base model vs Base + ModalityDebias

Input:
    results/tables/accuracy_fairness_table_k10.csv

Outputs:
    results/tables/rq5_modality_debias_pairwise_delta_k10.csv
    results/tables/rq5_modality_debias_pairwise_delta_k10.tex
    results/tables/rq5_method_level_summary_k10.csv

    results/figures/rq5_method_level_grouped_bars_k10.png
    results/figures/rq5_method_level_grouped_bars_k10.pdf

Usage:
    python scripts/build_plot_rq5_modality_vs_group_fairness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================

DATASETS = [
    "Baby_Products",
    "Musical_Instruments",
    "Video_Games",
]

DATASET_DISPLAY = {
    "Baby_Products": "Baby Products",
    "Musical_Instruments": "Musical Instruments",
    "Video_Games": "Video Games",
}

FAMILIES = [
    "VBPR",
    "BM3",
    "FREEDOM",
    "LATTICE",
]

BASE_METHODS = {
    "VBPR": "VBPR",
    "BM3": "BM3",
    "FREEDOM": "FREEDOM",
    "LATTICE": "LATTICE",
}

MD_METHODS = {
    "VBPR": "VBPR + ModalityDebias",
    "BM3": "BM3 + ModalityDebias",
    "FREEDOM": "FREEDOM + ModalityDebias",
    "LATTICE": "LATTICE + ModalityDebias",
}

FIG_METRICS = [
    ("ndcg", "NDCG@10 ↑"),
    ("pop_gap", "PopGap@10 ↓"),
    ("brand_gap", "BrandGap@10 ↓"),
    ("cluster_gap", "ClusterGap@10 ↓"),
]

PAIRWISE_COLUMNS = [
    "dataset",
    "dataset_display",
    "base_model",
    "md_model",
    "base_ndcg",
    "md_ndcg",
    "delta_ndcg_%",
    "base_hr",
    "md_hr",
    "delta_hr_%",
    "base_pop_gap",
    "md_pop_gap",
    "delta_pop_gap",
    "base_brand_gap",
    "md_brand_gap",
    "delta_brand_gap",
    "base_cluster_gap",
    "md_cluster_gap",
    "delta_cluster_gap",
    "gap_wins",
]


# =========================================================
# Utilities
# =========================================================

def get_hr_col(df: pd.DataFrame) -> str:
    if "hit" in df.columns:
        return "hit"
    if "recall" in df.columns:
        return "recall"
    raise ValueError("Cannot find HR column. Expected `hit` or `recall`.")


def pct_delta(new: float, old: float) -> float:
    if old == 0 or pd.isna(old):
        return np.nan
    return (new - old) / old * 100.0


def latex_escape(text: str) -> str:
    text = str(text)
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_signed(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):+.{digits}f}"


# =========================================================
# Method normalization
# =========================================================

def clean_method(row: pd.Series) -> Optional[str]:
    method = str(row.get("method", ""))
    run_id = str(row.get("run_id", ""))
    r = run_id.lower()

    # -----------------------------------------------------
    # Multimodal baselines.
    # -----------------------------------------------------
    if method in set(BASE_METHODS.values()):
        return method

    # -----------------------------------------------------
    # ModalityDebias variants.
    # -----------------------------------------------------
    if method in set(MD_METHODS.values()):
        return method

    if "vbpr_modality_debias" in r:
        return "VBPR + ModalityDebias"
    if "bm3_modality_debias" in r:
        return "BM3 + ModalityDebias"
    if "freedom_modality_debias" in r:
        return "FREEDOM + ModalityDebias"
    if "lattice_modality_debias" in r:
        return "LATTICE + ModalityDebias"

    # -----------------------------------------------------
    # FARE + SASRec reference.
    # -----------------------------------------------------
    if method in {"FARE-SASRec", "FARE + SASRec"}:
        return "FARE + SASRec"

    if method == "FARE":
        if r.startswith("fare_") and "sasrec" in r:
            return "FARE + SASRec"

    return None


def select_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    If duplicated dataset × method rows exist, keep the one with largest best_metric.
    This handles duplicated ModalityDebias runs or multiple FARE main sweeps.
    """
    metric_col = "best_metric" if "best_metric" in df.columns else "ndcg"

    selected_indices = []
    for (dataset, method_clean), group in df.groupby(["dataset", "method_clean"]):
        idx = group[metric_col].astype(float).idxmax()
        selected_indices.append(idx)

    return df.loc[selected_indices].copy()


def get_row(df: pd.DataFrame, dataset: str, method: str) -> Optional[pd.Series]:
    sub = df[(df["dataset"] == dataset) & (df["method_clean"] == method)]
    if sub.empty:
        return None
    return sub.iloc[0]


# =========================================================
# Build pairwise delta table
# =========================================================

def build_pairwise_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        for family in FAMILIES:
            base_method = BASE_METHODS[family]
            md_method = MD_METHODS[family]

            base = get_row(df, dataset, base_method)
            md = get_row(df, dataset, md_method)

            if base is None or md is None:
                print(f"[WARN] Missing pair: dataset={dataset}, family={family}")
                continue

            # Utility: positive means MD improves utility.
            delta_ndcg_pct = pct_delta(md["ndcg"], base["ndcg"])
            delta_hr_pct = pct_delta(md["hr"], base["hr"])

            # Gaps: lower is better, so positive means MD reduces the gap.
            delta_pop_gap = base["pop_gap"] - md["pop_gap"]
            delta_brand_gap = base["brand_gap"] - md["brand_gap"]
            delta_cluster_gap = base["cluster_gap"] - md["cluster_gap"]

            gap_wins = int(delta_pop_gap > 0)
            gap_wins += int(delta_brand_gap > 0)
            gap_wins += int(delta_cluster_gap > 0)

            rows.append({
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY.get(dataset, dataset.replace("_", " ")),
                "base_model": base_method,
                "md_model": md_method,

                "base_ndcg": float(base["ndcg"]),
                "md_ndcg": float(md["ndcg"]),
                "delta_ndcg_%": float(delta_ndcg_pct),

                "base_hr": float(base["hr"]),
                "md_hr": float(md["hr"]),
                "delta_hr_%": float(delta_hr_pct),

                "base_pop_gap": float(base["pop_gap"]),
                "md_pop_gap": float(md["pop_gap"]),
                "delta_pop_gap": float(delta_pop_gap),

                "base_brand_gap": float(base["brand_gap"]),
                "md_brand_gap": float(md["brand_gap"]),
                "delta_brand_gap": float(delta_brand_gap),

                "base_cluster_gap": float(base["cluster_gap"]),
                "md_cluster_gap": float(md["cluster_gap"]),
                "delta_cluster_gap": float(delta_cluster_gap),

                "gap_wins": int(gap_wins),
            })

    out = pd.DataFrame(rows)
    return out[PAIRWISE_COLUMNS]


# =========================================================
# Build method-level summary for the figure
# =========================================================

def build_method_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average each method over datasets.
    This is method-level, not category-level:
        VBPR
        VBPR + MD
        BM3
        BM3 + MD
        ...
        FARE + SASRec
    """
    rows = []

    for family in FAMILIES:
        for variant, method_name in [
            ("Base", BASE_METHODS[family]),
            ("+ModalityDebias", MD_METHODS[family]),
        ]:
            sub = df[df["method_clean"] == method_name].copy()
            if sub.empty:
                print(f"[WARN] Missing method for summary: {method_name}")
                continue

            rows.append({
                "family": family,
                "variant": variant,
                "method": method_name,
                "ndcg": sub["ndcg"].astype(float).mean(),
                "hr": sub["hr"].astype(float).mean(),
                "pop_gap": sub["pop_gap"].astype(float).mean(),
                "brand_gap": sub["brand_gap"].astype(float).mean(),
                "cluster_gap": sub["cluster_gap"].astype(float).mean(),
            })

    fair = df[df["method_clean"] == "FARE + SASRec"].copy()
    if fair.empty:
        print("[WARN] Missing FARE + SASRec reference.")
    else:
        rows.append({
            "family": "Reference",
            "variant": "FARE + SASRec",
            "method": "FARE + SASRec",
            "ndcg": fair["ndcg"].astype(float).mean(),
            "hr": fair["hr"].astype(float).mean(),
            "pop_gap": fair["pop_gap"].astype(float).mean(),
            "brand_gap": fair["brand_gap"].astype(float).mean(),
            "cluster_gap": fair["cluster_gap"].astype(float).mean(),
        })

    out = pd.DataFrame(rows)
    return out


# =========================================================
# Plot method-level grouped bars
# =========================================================

def add_bar_labels(ax, bars, values, metric: str) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    offset = span * 0.015

    for bar, value in zip(bars, values):
        if metric in {"brand_gap", "cluster_gap"}:
            label = f"{value:.2f}"
        else:
            label = f"{value:.4f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )


def plot_method_level_grouped_bars(summary: pd.DataFrame, fig_dir: Path) -> None:
    base_summary = summary[summary["variant"].isin(["Base", "+ModalityDebias"])].copy()
    fair_ref = summary[summary["method"] == "FARE + SASRec"].copy()

    if fair_ref.empty:
        fair_values = {}
    else:
        fair_row = fair_ref.iloc[0]
        fair_values = {
            metric: float(fair_row[metric])
            for metric, _ in FIG_METRICS
        }

    x = np.arange(len(FAMILIES))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, FIG_METRICS):
        base_values = []
        md_values = []

        for family in FAMILIES:
            base_row = base_summary[
                (base_summary["family"] == family)
                & (base_summary["variant"] == "Base")
            ]
            md_row = base_summary[
                (base_summary["family"] == family)
                & (base_summary["variant"] == "+ModalityDebias")
            ]

            base_values.append(float(base_row.iloc[0][metric]) if not base_row.empty else np.nan)
            md_values.append(float(md_row.iloc[0][metric]) if not md_row.empty else np.nan)

        bars_base = ax.bar(
            x - width / 2,
            base_values,
            width,
            label="Base",
        )
        bars_md = ax.bar(
            x + width / 2,
            md_values,
            width,
            label="+ModalityDebias",
        )

        # FARE + SASRec reference line.
        if metric in fair_values:
            ax.axhline(
                fair_values[metric],
                linestyle="--",
                linewidth=1.4,
                label="FARE + SASRec",
            )

        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES, rotation=0, fontsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

        values_all = [v for v in base_values + md_values if not pd.isna(v)]
        if metric in fair_values:
            values_all.append(fair_values[metric])

        ymin = min(values_all)
        ymax = max(values_all)
        span = ymax - ymin
        margin = max(span * 0.20, 0.001 if metric == "ndcg" else 0.05)

        lower = max(0.0, ymin - margin)
        upper = ymax + margin
        ax.set_ylim(lower, upper)

        add_bar_labels(ax, bars_base, base_values, metric)
        add_bar_labels(ax, bars_md, md_values, metric)

        if metric in fair_values:
            ref_value = fair_values[metric]
            label = (
                f"FARE: {ref_value:.4f}"
                if metric not in {"brand_gap", "cluster_gap"}
                else f"FARE: {ref_value:.2f}"
            )
            ax.text(
                0.98,
                (ref_value - lower) / (upper - lower) + 0.015,
                label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove duplicates while preserving order.
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.10,
        wspace=0.25,
        hspace=0.38,
    )

    png_path = fig_dir / "rq5_method_level_grouped_bars_k10.png"
    pdf_path = fig_dir / "rq5_method_level_grouped_bars_k10.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\n========== RQ5 figure saved ==========")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


# =========================================================
# LaTeX pairwise delta table
# =========================================================

def write_pairwise_latex(pairwise: pd.DataFrame, out_path: Path) -> None:
    lines = []

    lines.append("% Required packages:")
    lines.append("% \\usepackage{booktabs,multirow,graphicx}")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("    \\centering")
    lines.append("    \\caption{Pairwise effect of ModalityDebias on multimodal recommendation baselines at $K=10$. "
                 "Positive $\\Delta$NDCG and $\\Delta$HR indicate utility improvements. "
                 "Positive $\\Delta$Gap values indicate that ModalityDebias reduces the corresponding group-aware gap.}")
    lines.append("    \\label{tab:rq5_modality_debias_pairwise}")
    lines.append("    \\scriptsize")
    lines.append("    \\setlength{\\tabcolsep}{4.2pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.95}")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{llrrrrrr}")
    lines.append("        \\toprule")
    lines.append("        Dataset & Base model & $\\Delta$NDCG(\\%) & $\\Delta$HR(\\%) & "
                 "$\\Delta$PopGap & $\\Delta$BrandGap & $\\Delta$ClusterGap & Gap wins / 3 \\\\")
    lines.append("        \\midrule")

    for d_i, dataset in enumerate(DATASETS):
        sub = pairwise[pairwise["dataset"] == dataset].copy()
        if sub.empty:
            continue

        dataset_name = DATASET_DISPLAY.get(dataset, dataset.replace("_", " "))
        n_rows = len(sub)

        for r_i, (_, row) in enumerate(sub.iterrows()):
            dataset_cell = f"\\multirow{{{n_rows}}}{{*}}{{{dataset_name}}}" if r_i == 0 else ""

            cells = [
                dataset_cell,
                latex_escape(row["base_model"]),
                fmt_signed(row["delta_ndcg_%"], 2),
                fmt_signed(row["delta_hr_%"], 2),
                fmt_signed(row["delta_pop_gap"], 4),
                fmt_signed(row["delta_brand_gap"], 3),
                fmt_signed(row["delta_cluster_gap"], 3),
                f"{int(row['gap_wins'])}/3",
            ]
            lines.append("        " + " & ".join(cells) + " \\\\")

        if d_i != len(DATASETS) - 1:
            lines.append("        \\midrule")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("    \\vspace{0.3em}")
    lines.append("    \\footnotesize{Gap wins / 3 counts the number of improved group-aware gaps among PopGap@10, BrandGap@10, and ClusterGap@10.}")
    lines.append("\\end{table*}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== RQ5 LaTeX table saved ==========")
    print(f"LaTeX: {out_path}")


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RQ5 method-level ModalityDebias vs group-aware fairness results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/tables/accuracy_fairness_table_k10.csv",
        help="Input accuracy/fairness table at K=10.",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="results/tables",
        help="Directory to save tables.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    fig_dir = Path(args.fig_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_path}")

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    hr_col = get_hr_col(raw)

    raw = raw.copy()
    raw["hr"] = raw[hr_col]
    raw["method_clean"] = raw.apply(clean_method, axis=1)
    raw = raw[raw["method_clean"].notna()].copy()

    selected = select_best_rows(raw)

    print("========== Selected method counts ==========")
    print(pd.crosstab(selected["dataset"], selected["method_clean"]).to_string())

    pairwise = build_pairwise_delta_table(selected)
    summary = build_method_level_summary(selected)

    pairwise_csv = table_dir / "rq5_modality_debias_pairwise_delta_k10.csv"
    pairwise_tex = table_dir / "rq5_modality_debias_pairwise_delta_k10.tex"
    summary_csv = table_dir / "rq5_method_level_summary_k10.csv"

    pairwise.to_csv(pairwise_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("\n========== RQ5 CSV tables saved ==========")
    print(f"Pairwise delta CSV: {pairwise_csv}")
    print(f"Method-level summary CSV: {summary_csv}")

    print("\n========== Pairwise delta preview ==========")
    print(
        pairwise[
            [
                "dataset",
                "base_model",
                "delta_ndcg_%",
                "delta_hr_%",
                "delta_pop_gap",
                "delta_brand_gap",
                "delta_cluster_gap",
                "gap_wins",
            ]
        ].to_string(index=False)
    )

    print("\n========== Method-level summary preview ==========")
    print(summary.to_string(index=False))

    write_pairwise_latex(pairwise, pairwise_tex)
    plot_method_level_grouped_bars(summary, fig_dir)

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()
