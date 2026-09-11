"""Collect validation-only utility and fairness metrics for policy selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import fairness_metric_contract, load_protocol


REQUIRED_COLUMNS = {"dataset", "backbone", "seed", "config_id", "run_dir"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    args = parser.parse_args()

    registry = pd.read_csv(args.registry)
    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"Run registry missing columns: {sorted(missing)}")

    k = int(args.k)
    contract = fairness_metric_contract(load_protocol(args.protocol))["metrics"]
    output_rows = []
    for source in registry.to_dict("records"):
        run_dir = Path(str(source["run_dir"]))
        summary_path = run_dir / "metrics_summary.json"
        fairness_path = run_dir / "val_fairness_flat_metrics.json"
        if not summary_path.exists() or not fairness_path.exists():
            raise FileNotFoundError(
                f"Validation artifacts missing for {run_dir}: "
                "metrics_summary.json and val_fairness_flat_metrics.json are required"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fairness = json.loads(fairness_path.read_text(encoding="utf-8"))
        val = summary.get("val")
        if not isinstance(val, dict):
            raise ValueError(f"Run {run_dir} has no final validation metrics")

        def fair(key: str):
            value = fairness.get(key)
            return float(value) if value is not None else float("nan")

        row = {
            "dataset": source["dataset"],
            "backbone": source["backbone"],
            "seed": int(source["seed"]),
            "config_id": source["config_id"],
            "split": "val",
            "run_dir": str(run_dir),
            "ndcg": float(val[f"ndcg@{k}"]),
            "hr": float(val[f"hit@{k}"]),
        }
        for metric_name, spec in contract.items():
            row[metric_name] = fair(str(spec["artifact_key"]).format(k=k))
        output_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output, index=False)
    audit_path = Path(args.audit) if args.audit else output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "selection_split": "val",
                "k": k,
                "runs": len(output_rows),
                "input_registry": str(args.registry),
                "output": str(output),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Validation selection manifest: {output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
