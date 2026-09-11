"""Select ID checkpoints by validation metric and build the sweep registry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    datasets_cfg = yaml.safe_load(Path(args.datasets_config).read_text(encoding="utf-8")) or {}
    dataset_entries = datasets_cfg.get("datasets", {})
    candidates = []
    for config_path in results_root.rglob("config_resolved.json"):
        run_dir = config_path.parent
        checkpoint = run_dir / "best_model.pt"
        reference_topk = run_dir / "topk_val.npz"
        metrics_path = run_dir / "metrics_summary.json"
        if not checkpoint.exists() or not reference_topk.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        backbone = config.get("model_arg")
        if str(backbone).lower() not in {"sasrec", "gru4rec", "bert4rec"}:
            continue
        expected_run_name = f"{str(backbone).lower()}_id"
        if str(config.get("run_name", "")).lower() != expected_run_name:
            continue
        if config.get("method_name") not in (None, "", "ID"):
            continue
        if str(config.get("eval", {}).get("split_for_best", "val")).lower() != "val":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = metrics.get("best_metric")
        if value is None:
            continue
        dataset = str(config.get("dataset"))
        seed = int(config.get("seed", -1))
        if not dataset or seed < 0:
            continue
        platform = str(dataset_entries.get(dataset, {}).get("platform", "amazon"))
        candidates.append(
            {
                "dataset": dataset,
                "platform": platform,
                "backbone": str(backbone).lower(),
                "seed": seed,
                "init_backbone_checkpoint": str(checkpoint.resolve()),
                "reference_topk_path": str(reference_topk.resolve()),
                "validation_metric_name": str(metrics.get("best_metric_name", "unknown")),
                "validation_metric": float(value),
                "run_dir": str(run_dir.resolve()),
            }
        )

    selected = {}
    for row in candidates:
        key = (row["dataset"], row["backbone"], row["seed"])
        current = selected.get(key)
        score_key = (float(row["validation_metric"]), str(row["run_dir"]))
        current_key = (
            (float(current["validation_metric"]), str(current["run_dir"]))
            if current is not None
            else (-float("inf"), "")
        )
        if score_key > current_key:
            selected[key] = row

    output_rows = [selected[key] for key in sorted(selected)]
    if not output_rows:
        raise ValueError(f"No validation-selected ID checkpoints found under {results_root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"selected_checkpoints": len(output_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
