"""Convert generic sweep result tables into a validation run registry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = {
    "status",
    "metric_file",
    "arg/dataset",
    "arg/backbone",
    "arg/seed",
    "param/fair_rec_reweight_weight",
    "param/fair_rec_reweight_groups",
}


def recover_metric_file(metric_file: Path, source: dict[str, str]) -> Path:
    if metric_file.exists():
        return metric_file

    run_id = str(source.get("run_id", "")).strip()
    dataset = str(source.get("arg/dataset", "")).strip()
    if not run_id or len(metric_file.parents) < 4:
        raise FileNotFoundError(metric_file)

    results_root = metric_file.parents[3]
    search_roots = []
    if dataset:
        dataset_root = results_root / dataset
        if dataset_root.exists():
            search_roots.append(dataset_root)
    if results_root.exists():
        search_roots.append(results_root)

    candidates = []
    seen = set()
    for root in search_roots:
        for path in root.rglob("metrics_summary.json"):
            if path.parent.name != run_id:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{metric_file} is missing and run_id={run_id!r} matched multiple metric files: "
            f"{[str(path) for path in candidates[:5]]}"
        )
    raise FileNotFoundError(metric_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps-root", default="sweeps/revision")
    parser.add_argument("--sweep-results", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    source_files = (
        sorted(Path(value) for value in args.sweep_results)
        if args.sweep_results
        else sorted(Path(args.sweeps_root).rglob("sweep_results.csv"))
    )
    for source_path in source_files:
        with source_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{source_path} missing columns: {sorted(missing)}")
            for source in reader:
                if source["status"].lower() != "ok":
                    continue
                metric_file = recover_metric_file(Path(source["metric_file"]), source)
                gamma = source["param/fair_rec_reweight_weight"]
                group = source["param/fair_rec_reweight_groups"]
                rows.append(
                    {
                        "dataset": source["arg/dataset"],
                        "backbone": source["arg/backbone"],
                        "seed": int(source["arg/seed"]),
                        "config_id": f"gamma={gamma}|group={group}",
                        "split": "val",
                        "run_dir": str(metric_file.parent),
                        "sweep_results": str(source_path),
                    }
                )

    if not rows:
        raise ValueError(f"No successful sweep runs found under {args.sweeps_root}")
    rows.sort(key=lambda row: (row["dataset"], row["backbone"], row["seed"], row["config_id"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "sources": len(source_files), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
