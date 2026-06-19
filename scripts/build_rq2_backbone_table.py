#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("results/tables/accuracy_fairness_table_k10.csv")
OUT_DETAIL = Path("results/tables/rq2_backbone_generalization_k10.csv")
OUT_SUMMARY = Path("results/tables/rq2_backbone_generalization_summary_k10.csv")

PAIRS = {
    "sasrec": {
        "display": "SASRec",
        "base": "SASRec-ID",
        "ours": "FARE-SASRec",
    },
    "gru4rec": {
        "display": "GRU4Rec",
        "base": "GRU4Rec-ID",
        "ours": "FARE-GRU4Rec",
    },
    "bert4rec": {
        "display": "BERT4Rec",
        "base": "BERT4Rec-ID",
        "ours": "FARE-BERT4Rec",
    },
}

DATASETS = [
    "Baby_Products",
    "Musical_Instruments",
    "Video_Games",
]

def infer_fare_method(row):
    method = str(row["method"])
    run_id = str(row.get("run_id", ""))
    text = f"{method} {run_id}".lower()

    # Case 1: already clean method names
    aliases = {
        "FARE-SASRec": "FARE-SASRec",
        "FARE-GRU4Rec": "FARE-GRU4Rec",
        "FARE-BERT4Rec": "FARE-BERT4Rec",
        "FARE + SASRec": "FARE-SASRec",
        "FARE + GRU4Rec": "FARE-GRU4Rec",
        "FARE + BERT4Rec": "FARE-BERT4Rec",
        "FARE-F + SASRec": "FARE-SASRec",
        "FARE-F + GRU4Rec": "FARE-GRU4Rec",
        "FARE-F + BERT4Rec": "FARE-BERT4Rec",
    }
    if method in aliases:
        return aliases[method]

    # Case 2: raw table uses method = FARE
    if method == "FARE":
        is_final_fare = (
            run_id.startswith("fare_main_")
            or "fare_exposure" in text
            or "fair_sdr_exposure" in text
            or "fairfirst" in text
        )
        if not is_final_fare:
            return None

        if "sasrec" in text:
            return "FARE-SASRec"
        if "gru4rec" in text:
            return "FARE-GRU4Rec"
        if "bert4rec" in text:
            return "FARE-BERT4Rec"

    return method

def get_hr_column(df):
    if "hit" in df.columns:
        return "hit"
    if "recall" in df.columns:
        return "recall"
    raise ValueError("Cannot find HR column. Expected `hit` or `recall`.")

def select_best_rows(df):
    """
    If there are duplicated dataset × method rows, keep the row with largest best_metric.
    """
    metric_col = "best_metric" if "best_metric" in df.columns else "ndcg"
    idx = df.groupby(["dataset", "method_clean"])[metric_col].idxmax()
    return df.loc[idx].copy()

def pct_delta(ours, base):
    if base == 0 or pd.isna(base):
        return np.nan
    return (ours - base) / base * 100.0

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)
    hr_col = get_hr_column(df)

    df = df.copy()
    df["method_clean"] = df.apply(infer_fare_method, axis=1)
    df = df[df["method_clean"].notna()].copy()

    needed_methods = set()
    for cfg in PAIRS.values():
        needed_methods.add(cfg["base"])
        needed_methods.add(cfg["ours"])

    df = df[df["method_clean"].isin(needed_methods)].copy()
    df = select_best_rows(df)

    rows = []

    for dataset in DATASETS:
        for backbone_key, cfg in PAIRS.items():
            base_method = cfg["base"]
            ours_method = cfg["ours"]

            base_rows = df[(df["dataset"] == dataset) & (df["method_clean"] == base_method)]
            ours_rows = df[(df["dataset"] == dataset) & (df["method_clean"] == ours_method)]

            if base_rows.empty or ours_rows.empty:
                print(f"[WARN] Missing pair: dataset={dataset}, backbone={cfg['display']}")
                continue

            b = base_rows.iloc[0]
            o = ours_rows.iloc[0]

            delta_coverage = o["coverage"] - b["coverage"]
            delta_rec_gini = o["rec_gini"] - b["rec_gini"]
            delta_pop_gap = o["pop_gap"] - b["pop_gap"]
            delta_brand_gap = o["brand_gap"] - b["brand_gap"]
            delta_cluster_gap = o["cluster_gap"] - b["cluster_gap"]

            fairness_wins = 0
            fairness_wins += int(delta_coverage > 0)
            fairness_wins += int(delta_rec_gini < 0)
            fairness_wins += int(delta_pop_gap < 0)
            fairness_wins += int(delta_brand_gap < 0)
            fairness_wins += int(delta_cluster_gap < 0)

            rows.append({
                "dataset": dataset,
                "backbone": cfg["display"],
                "base_method": base_method,
                "ours_method": ours_method,

                "base_ndcg": b["ndcg"],
                "ours_ndcg": o["ndcg"],
                "delta_ndcg": o["ndcg"] - b["ndcg"],
                "delta_ndcg_%": pct_delta(o["ndcg"], b["ndcg"]),

                "base_hr": b[hr_col],
                "ours_hr": o[hr_col],
                "delta_hr": o[hr_col] - b[hr_col],
                "delta_hr_%": pct_delta(o[hr_col], b[hr_col]),

                "base_coverage": b["coverage"],
                "ours_coverage": o["coverage"],
                "delta_coverage": delta_coverage,

                "base_rec_gini": b["rec_gini"],
                "ours_rec_gini": o["rec_gini"],
                "delta_rec_gini": delta_rec_gini,

                "base_pop_gap": b["pop_gap"],
                "ours_pop_gap": o["pop_gap"],
                "delta_pop_gap": delta_pop_gap,

                "base_brand_gap": b["brand_gap"],
                "ours_brand_gap": o["brand_gap"],
                "delta_brand_gap": delta_brand_gap,

                "base_cluster_gap": b["cluster_gap"],
                "ours_cluster_gap": o["cluster_gap"],
                "delta_cluster_gap": delta_cluster_gap,

                "fairness_wins": fairness_wins,
            })

    out = pd.DataFrame(rows)
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DETAIL, index=False)

    summary = (
        out.groupby("backbone")
        .agg(
            mean_delta_ndcg_pct=("delta_ndcg_%", "mean"),
            mean_delta_hr_pct=("delta_hr_%", "mean"),
            mean_delta_coverage=("delta_coverage", "mean"),
            mean_delta_rec_gini=("delta_rec_gini", "mean"),
            mean_delta_pop_gap=("delta_pop_gap", "mean"),
            mean_delta_brand_gap=("delta_brand_gap", "mean"),
            mean_delta_cluster_gap=("delta_cluster_gap", "mean"),
            mean_fairness_wins=("fairness_wins", "mean"),
            total_fairness_wins=("fairness_wins", "sum"),
            ndcg_win_count=("delta_ndcg", lambda x: int((x > 0).sum())),
            hr_win_count=("delta_hr", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved detail:  {OUT_DETAIL}")
    print(f"Saved summary: {OUT_SUMMARY}")
    print("\nDetail:")
    print(out[[
        "dataset", "backbone",
        "delta_ndcg_%", "delta_hr_%",
        "delta_coverage", "delta_rec_gini",
        "delta_pop_gap", "delta_brand_gap", "delta_cluster_gap",
        "fairness_wins"
    ]].to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
