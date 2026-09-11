#!/usr/bin/env python3
"""Select revision configurations using validation metrics only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_statistics import policy_sensitivity, select_on_validation
from src.revision_protocol import load_protocol, seed_tiers


def _float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Long-form run manifest CSV")
    parser.add_argument("--selected-out", required=True)
    parser.add_argument("--sensitivity-out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alphas", default="0.25,0.4,0.5,0.6,0.75")
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--required-seeds", default=None)
    parser.add_argument("--group-cols", default="dataset,backbone")
    parser.add_argument("--candidate-col", default="config_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    frame = pd.read_csv(input_path)
    group_cols = [part.strip() for part in args.group_cols.split(",") if part.strip()]
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = PROJECT_ROOT / protocol_path
    breadth_seeds, _ = seed_tiers(load_protocol(protocol_path))
    required_seeds = _int_list(args.required_seeds) if args.required_seeds else breadth_seeds
    alphas = _float_list(args.alphas)

    selected = select_on_validation(
        frame=frame,
        alpha=args.alpha,
        group_cols=group_cols,
        candidate_col=args.candidate_col,
        required_seeds=required_seeds,
    )
    sensitivity = policy_sensitivity(
        frame=frame,
        alphas=alphas,
        group_cols=group_cols,
        candidate_col=args.candidate_col,
        required_seeds=required_seeds,
    )

    selected_path = Path(args.selected_out)
    sensitivity_path = Path(args.sensitivity_out)
    audit_path = Path(args.audit_out)
    for path in (selected_path, sensitivity_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(selected_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    split_values = frame["split"].astype(str).str.lower()
    audit = {
        "input": str(input_path),
        "selection_split": "val",
        "test_rows_ignored": int(split_values.eq("test").sum()),
        "validation_rows_used": int(split_values.eq("val").sum()),
        "group_columns": group_cols,
        "candidate_column": args.candidate_col,
        "required_seeds": required_seeds,
        "seed_tier": "explicit" if args.required_seeds else "breadth",
        "protocol": str(protocol_path),
        "primary_alpha": float(args.alpha),
        "sensitivity_alphas": alphas,
        "selected_config_ids": selected[args.candidate_col].astype(str).tolist(),
        "selection_metric_columns": [
            column
            for column in (
                "ndcg",
                "hr",
                "coverage",
                "rec_gini",
                "pop_gap",
                "brand_gap",
                "cluster_gap",
            )
            if column in frame.columns
        ],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Selected runs: {selected_path}")
    print(f"Policy sensitivity: {sensitivity_path}")
    print(f"Selection audit: {audit_path}")


if __name__ == "__main__":
    main()
