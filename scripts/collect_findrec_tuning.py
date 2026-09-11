#!/usr/bin/env python3
"""Collect validation-only FindRec sweep evidence and select one trial per seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "run_id",
    "status",
    "metric_key",
    "metric",
    "metric_file",
    "arg/dataset",
    "arg/seed",
    "arg/skip_test_after_training",
}


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-results", nargs="+", required=True)
    parser.add_argument("--trials-out", required=True)
    parser.add_argument("--selected-out", required=True)
    parser.add_argument("--audit-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    leakage = []
    for source in args.sweep_results:
        path = Path(source)
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing tuning evidence columns: {sorted(missing)}")
        frame["source_sweep"] = str(path)
        frames.append(frame)
    trials = pd.concat(frames, ignore_index=True)
    trials["dataset"] = trials["arg/dataset"].astype(str)
    trials["seed"] = trials["arg/seed"].astype(int)
    trials["validation_metric"] = pd.to_numeric(trials["metric"], errors="coerce")
    trials["run_dir"] = trials["metric_file"].map(lambda value: str(Path(str(value)).parent))

    successful = trials[trials["status"].astype(str).str.lower().eq("ok")].copy()
    if successful.empty:
        raise ValueError("No successful FindRec tuning trials")

    for index, row in successful.iterrows():
        if not _is_true(row["arg/skip_test_after_training"]):
            leakage.append({"row": int(index), "reason": "test evaluation was not disabled"})
            continue
        metrics_path = Path(str(row["metric_file"]))
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        forbidden = [
            name
            for name in ("test_ranking.npz", "topk_test.npz")
            if (run_dir / name).exists()
        ]
        if "test_final" in metrics or forbidden:
            leakage.append(
                {
                    "row": int(index),
                    "reason": "test artifacts found in tuning run",
                    "artifacts": forbidden,
                }
            )
    if leakage:
        raise ValueError(f"FindRec tuning leakage detected: {leakage}")

    selected = (
        successful.sort_values(
            ["dataset", "seed", "validation_metric", "run_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(["dataset", "seed"], as_index=False, sort=True)
        .head(1)
    )

    trials_out = Path(args.trials_out)
    selected_out = Path(args.selected_out)
    audit_out = Path(args.audit_out)
    for path in (trials_out, selected_out, audit_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(trials_out, index=False)
    selected.to_csv(selected_out, index=False)
    audit = {
        "validation_only": True,
        "metric_source": "best validation checkpoint metric",
        "total_trials": int(len(trials)),
        "successful_trials": int(len(successful)),
        "failed_trials": int((~trials["status"].astype(str).str.lower().eq("ok")).sum()),
        "selected_trials": int(len(selected)),
        "datasets": sorted(trials["dataset"].unique().tolist()),
        "seeds": sorted(int(value) for value in trials["seed"].unique()),
        "source_sweeps": [str(Path(value)) for value in args.sweep_results],
        "test_artifacts_checked": ["test_final", "test_ranking.npz", "topk_test.npz"],
    }
    audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
