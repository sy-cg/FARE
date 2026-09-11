#!/usr/bin/env python3
"""Summarize FindRec optimization and recommendation-collapse diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_diagnostics import learning_curve_summary, ranking_collapse_summary


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _ranking_path(run_dir: Path) -> Path | None:
    for name in ("test_ranking.npz", "topk_test.npz"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def audit_run(run_dir: Path, coverage_threshold: float, k: int) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics_summary.json")
    train_log = run_dir / "train_log.csv"
    ranking_path = _ranking_path(run_dir)
    missing = []
    if not metrics:
        missing.append("metrics_summary.json")
    if not train_log.exists():
        missing.append("train_log.csv")
    if ranking_path is None:
        missing.append("test ranking archive")

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset": metrics.get("dataset", ""),
        "backbone": metrics.get("backbone", metrics.get("model_name", "FindRec")),
        "missing_artifacts": ", ".join(missing),
    }
    if missing:
        row["diagnosis"] = "adaptation_failure"
        return row

    curve_frame = pd.read_csv(train_log)
    curve = learning_curve_summary(curve_frame)
    with np.load(ranking_path, allow_pickle=False) as archive:
        topk = np.asarray(archive["topk_items"], dtype=np.int64)[:, : int(k)]
    num_items = int(metrics.get("num_items", int(topk.max(initial=0))))
    collapse = ranking_collapse_summary(topk, num_items, coverage_threshold)
    row.update({f"curve/{key}": value for key, value in curve.items()})
    row.update({f"ranking/{key}": value for key, value in collapse.items()})
    if collapse["coverage_collapse"]:
        diagnosis = "ranking_collapse"
    elif not curve["train_loss_decreased"] or curve["validation_improvement"] <= 0:
        diagnosis = "optimization_failure"
    else:
        diagnosis = "weak_performance_or_valid_run"
    row["diagnosis"] = diagnosis
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dirs", nargs="+")
    source.add_argument("--registry", help="CSV containing a run_dir column")
    parser.add_argument("--coverage-threshold", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.registry:
        registry = pd.read_csv(args.registry)
        if "run_dir" not in registry.columns:
            raise ValueError("FindRec registry requires a run_dir column")
        run_dirs = registry["run_dir"].astype(str).tolist()
    else:
        run_dirs = args.run_dirs
    rows = [audit_run(Path(path), args.coverage_threshold, args.k) for path in run_dirs]
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FindRec diagnostics: {output_csv}")


if __name__ == "__main__":
    main()
