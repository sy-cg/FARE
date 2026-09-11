"""Build executable generic FARE sweep YAMLs from a checkpoint registry."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import (
    fare_config_for_dataset,
    fare_run_name_for_dataset,
    model_overrides_for_backbone,
    seed_tiers,
    tagged_run_id,
)

REQUIRED_COLUMNS = {
    "dataset",
    "platform",
    "backbone",
    "seed",
    "init_backbone_checkpoint",
    "reference_topk_path",
}


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_yaml(protocol_path if protocol_path.is_absolute() else ROOT / protocol_path)
    registry_path = Path(args.registry)
    with registry_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Registry missing columns: {sorted(missing)}")
        rows = list(reader)

    row_map = {
        (row["dataset"], row["backbone"], int(row["seed"])): row
        for row in rows
    }
    datasets = [
        *protocol.get("datasets", {}).get("main", []),
        *protocol.get("datasets", {}).get("external", []),
    ]
    backbones = list(protocol.get("backbones", []))
    seeds, _ = seed_tiers(protocol)
    expected = set(product(datasets, backbones, seeds))
    missing_rows = sorted(expected - set(row_map))
    if missing_rows:
        raise ValueError(f"Checkpoint registry is incomplete; missing {len(missing_rows)} rows: {missing_rows[:5]}")

    exposure_cfg = protocol.get("exposure_reweighting", {})
    gamma_grid = [float(x) for x in exposure_cfg.get("gamma_grid", [])]
    groups = list(exposure_cfg.get("groups", []))
    metric_name = str(protocol.get("model_selection", {}).get("metric", "ndcg@10"))
    selection_split = str(protocol.get("model_selection", {}).get("split", ""))
    if selection_split != "val":
        raise ValueError("Sweep generation is allowed only for validation selection.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for dataset, backbone, seed in sorted(expected):
        row = row_map[(dataset, backbone, seed)]
        checkpoint = Path(row["init_backbone_checkpoint"])
        resolved_checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {resolved_checkpoint}")
        reference_topk = Path(row["reference_topk_path"])
        resolved_reference_topk = reference_topk if reference_topk.is_absolute() else ROOT / reference_topk
        if not resolved_reference_topk.exists():
            raise FileNotFoundError(f"Validation reference Top-K not found: {resolved_reference_topk}")

        slug = tagged_run_id(protocol, backbone, f"fare_gamma_{dataset}_{backbone}_s{seed}")
        run_name = fare_run_name_for_dataset(protocol, dataset, ROOT)
        sweep = {
            "name": slug,
            "script": "scripts/run_fare.py",
            "sampler": "grid",
            "seed": seed,
            "result_root": "results",
            "static_args": {
                "config": fare_config_for_dataset(protocol, dataset),
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "init_backbone_checkpoint": str(resolved_checkpoint),
                "fair_rec_exposure_source": "reference_topk",
                "fair_rec_exposure_topk_path": str(resolved_reference_topk),
                "skip_test_after_training": True,
                **model_overrides_for_backbone(protocol, backbone),
            },
            "search_space": {
                "fair_rec_reweight_weight": gamma_grid,
                "fair_rec_reweight_groups": groups,
            },
            "run_id": {
                "prefix": slug,
                "include_params": ["fair_rec_reweight_weight", "fair_rec_reweight_groups"],
            },
            "metric": {
                "key": f"val.{metric_name}",
                "mode": "max",
                "file_template": f"results/{{dataset}}/{run_name}/{{run_id}}/metrics_summary.json",
            },
        }
        path = output_dir / f"{slug}.yaml"
        path.write_text(yaml.safe_dump(sweep, sort_keys=False), encoding="utf-8")
        created.append(str(path))

    print(json.dumps({"created": len(created), "files": created}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
