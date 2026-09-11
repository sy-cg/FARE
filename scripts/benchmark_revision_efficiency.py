"""Aggregate comparable efficiency fields from completed revision runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.efficiency_audit import epoch_time_summary


REQUIRED_MANIFEST = {"dataset", "backbone", "method", "seed", "run_dir"}
REQUIRED_EFFICIENCY = {
    "total_parameters",
    "trainable_parameters",
    "peak_cuda_memory_mb",
    "test_inference_sec",
    "test_user_count",
    "test_ms_per_user",
    "test_users_per_sec",
    "torch_version",
    "cuda_version",
    "gpu_name",
}
EXPOSURE_METHODS = {"fare", "exposurereweight-id", "fare-independentprior"}
OUTPUT_COLUMNS = [
    "dataset",
    "backbone",
    "method",
    "seed",
    "run_dir",
    "epoch_count",
    "mean_epoch_sec",
    "median_epoch_sec",
    "total_train_sec",
    *sorted(REQUIRED_EFFICIENCY),
    "exposure_weight_build_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED_MANIFEST - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    output_rows = []
    issues = []
    for row in manifest.to_dict("records"):
        run_dir = Path(str(row["run_dir"]))
        metrics_path = run_dir / "metrics_summary.json"
        log_path = run_dir / "train_log.csv"
        if not metrics_path.exists() or not log_path.exists():
            issues.append({"run_dir": str(run_dir), "reason": "missing metrics_summary.json or train_log.csv"})
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        efficiency = metrics.get("efficiency", {})
        missing_fields = REQUIRED_EFFICIENCY - set(efficiency)
        method_key = str(row["method"]).strip().lower()
        if method_key in EXPOSURE_METHODS and "exposure_weight_build_sec" not in efficiency:
            missing_fields.add("exposure_weight_build_sec")
        if missing_fields:
            issues.append({"run_dir": str(run_dir), "reason": f"missing efficiency fields: {sorted(missing_fields)}"})
            continue

        log = pd.read_csv(log_path)
        if "epoch_time_sec" not in log.columns:
            issues.append({"run_dir": str(run_dir), "reason": "missing epoch_time_sec"})
            continue
        train_stats = epoch_time_summary(log["epoch_time_sec"].astype(float).tolist())
        output_rows.append(
            {
                "dataset": row["dataset"],
                "backbone": row["backbone"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "run_dir": str(run_dir),
                **train_stats,
                **{key: efficiency[key] for key in sorted(REQUIRED_EFFICIENCY)},
                "exposure_weight_build_sec": efficiency.get("exposure_weight_build_sec"),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).to_csv(output, index=False)
    audit = {
        "manifest_runs": int(len(manifest)),
        "complete_runs": int(len(output_rows)),
        "excluded_runs": int(len(issues)),
        "issues": issues,
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
