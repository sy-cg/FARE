#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot RQ2 backbone-level average bar charts.

Input:
    results/tables/rq2_backbone_generalization_k10.csv

Expected columns:
    dataset
    backbone
    delta_ndcg_%
    delta_hr_%
    delta_coverage
    delta_rec_gini
    delta_pop_gap
    delta_brand_gap
    delta_cluster_gap
    fairness_wins

Output:
    results/figures/rq2_backbone_average_bars_k10.png
    results/figures/rq2_backbone_average_bars_k10.pdf

Usage:
    python scripts/plot_rq2_backbone_average_bars.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BACKBONE_ORDER = ["SASRec", "GRU4Rec", "BERT4Rec"]


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_and_summarize(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_cols = [
        "dataset",
        "backbone",
        "delta_ndcg_%",
        "delta_hr_%",
        "delta_coverage",
        "delta_rec_gini",
        "delta_pop_gap",
        "delta_brand_gap",
        "delta_cluster_gap",
        "fairness_wins",
    ]
    validate_columns(df, required_cols)

    summary = (
        df.groupby("backbone", as_index=False)
        .agg(
            mean_delta_ndcg_pct=("delta_ndcg_%", "mean"),
            mean_delta_hr_pct=("delta_hr_%", "mean"),
            mean_delta_coverage=("delta_coverage", "mean"),
            mean_delta_rec_gini=("delta_rec_gini", "mean"),
            mean_delta_pop_gap=("delta_pop_gap", "mean"),
            mean_delta_brand_gap=("delta_brand_gap", "mean"),
            mean_delta_cluster_gap=("delta_cluster_gap", "mean"),
            mean_fairness_wins=("fairness_wins", "mean"),
        )
    )

    order_map = {b: i for i, b in enumerate(BACKBONE_ORDER)}
    summary["order"] = summary["backbone"].map(lambda x: order_map.get(x, 999))
    summary = summary.sort_values("order").reset_index(drop=True)

    return summary


def add_bar_labels(ax, bars, fmt="{:+.2f}", dy_ratio=0.02) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    dy = span * dy_ratio

    for bar in bars:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2

        if height >= 0:
            y = height + dy
            va = "bottom"
        else:
            y = height - dy
            va = "top"

        ax.text(
            x,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=9,
        )


def plot_metric_bar(
    ax,
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    value_fmt: str,
    positive_is_good: bool = True,
) -> None:
    x_labels = summary["backbone"].tolist()
    values = summary[value_col].astype(float).tolist()
    x = list(range(len(x_labels)))

    bars = ax.bar(x, values, width=0.58)

    ax.axhline(0, linewidth=1.0, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(y_label, fontsize=11)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.45,
    )

    # Dynamic y-limits.
    vmin = min(values + [0])
    vmax = max(values + [0])
    margin = max((vmax - vmin) * 0.22, 0.05 if "Coverage" in title else 1.0)
    ax.set_ylim(vmin - margin, vmax + margin)

    add_bar_labels(ax, bars, fmt=value_fmt)

    if positive_is_good:
        note = "higher is better"
    else:
        note = "lower is better"

    ax.text(
        0.98,
        0.06,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/tables/rq2_backbone_generalization_k10.csv",
        help="Input RQ2 detailed comparison CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures",
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rq2_backbone_average_bars_k10",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please run scripts/build_rq2_backbone_table.py first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_and_summarize(input_path)

    print("========== RQ2 backbone-level summary ==========")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

    plot_metric_bar(
        ax=axes[0, 0],
        summary=summary,
        value_col="mean_delta_ndcg_pct",
        title="(a) Mean ΔNDCG@10",
        y_label="ΔNDCG@10 (%)",
        value_fmt="{:+.2f}",
        positive_is_good=True,
    )

    plot_metric_bar(
        ax=axes[0, 1],
        summary=summary,
        value_col="mean_delta_hr_pct",
        title="(b) Mean ΔHR@10",
        y_label="ΔHR@10 (%)",
        value_fmt="{:+.2f}",
        positive_is_good=True,
    )

    plot_metric_bar(
        ax=axes[1, 0],
        summary=summary,
        value_col="mean_delta_coverage",
        title="(c) Mean ΔCoverage@10",
        y_label="ΔCoverage@10",
        value_fmt="{:+.3f}",
        positive_is_good=True,
    )

    plot_metric_bar(
        ax=axes[1, 1],
        summary=summary,
        value_col="mean_fairness_wins",
        title="(d) Mean fairness wins",
        y_label="Improved fairness metrics / 5",
        value_fmt="{:.2f}",
        positive_is_good=True,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.28,
        hspace=0.35,
    )

    png_path = out_dir / f"{args.prefix}.png"
    pdf_path = out_dir / f"{args.prefix}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("========== Saved ==========")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()