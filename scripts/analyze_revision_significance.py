#!/usr/bin/env python3
"""Paired bootstrap/significance analysis for revision Top-K archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_statistics import (
    align_paired_user_values,
    group_utility_aware_gap,
    hierarchical_paired_bootstrap,
    hierarchical_topk_metric_bootstrap,
    per_user_ranking_metric,
    recommendation_coverage,
    recommendation_gini,
    seed_level_paired_bootstrap,
)
from src.revision_protocol import fairness_metric_contract, load_protocol, seed_tiers


def _load_archive(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ranking archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        user_key = "users" if "users" in keys else "user_ids"
        required = {user_key, "targets", "topk_items"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"ranking archive {path} is missing keys: {missing}")
        return {
            "users": np.asarray(archive[user_key], dtype=np.int64).reshape(-1),
            "targets": np.asarray(archive["targets"], dtype=np.int64).reshape(-1),
            "topk_items": np.asarray(archive["topk_items"], dtype=np.int64),
        }


def _align_archive_rows(
    base: dict[str, np.ndarray],
    method: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    users = base["users"]
    _, aligned_method_targets = align_paired_user_values(
        users,
        base["targets"],
        method["users"],
        method["targets"],
    )
    if not np.array_equal(base["targets"], aligned_method_targets.astype(np.int64)):
        raise ValueError("paired archives have different targets for the same users")
    method_index = {int(user): idx for idx, user in enumerate(method["users"])}
    order = np.asarray([method_index[int(user)] for user in users], dtype=np.int64)
    aligned_method = {
        "users": users.copy(),
        "targets": method["targets"][order],
        "topk_items": method["topk_items"][order],
    }
    return base, aligned_method


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_group_labels(data_dir: str | Path, group_name: str) -> np.ndarray:
    path = Path(data_dir) / "fairness_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"fairness group file not found: {path}")
    groups = json.loads(path.read_text(encoding="utf-8"))
    mapping = groups.get(group_name)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"group {group_name!r} is missing from {path}")
    max_item = max(int(item) for item in mapping)
    labels = np.full(max_item + 1, -1, dtype=np.int64)
    for item, group in mapping.items():
        labels[int(item)] = int(group)
    return labels


def _load_group_utility(data_dir: str | Path, group_name: str) -> dict[str, float]:
    path = Path(data_dir) / "group_utility_train.json"
    if not path.exists():
        raise FileNotFoundError(f"group utility file not found: {path}")
    utility = json.loads(path.read_text(encoding="utf-8")).get(group_name)
    if not isinstance(utility, dict) or not utility:
        raise ValueError(f"group {group_name!r} is missing from {path}")
    return {str(key): float(value) for key, value in utility.items()}


def _infer_num_items(data_dir: str | Path) -> int:
    path = Path(data_dir) / "item_popularity_train.npy"
    if not path.exists():
        raise FileNotFoundError(f"item popularity file not found: {path}")
    popularity = np.load(path, allow_pickle=False)
    if popularity.ndim != 1 or popularity.size <= 1:
        raise ValueError(f"invalid item popularity vector: {path}")
    return int(popularity.size - 1)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--ci-mode",
        choices=("seed", "hierarchical"),
        default="seed",
        help=(
            "CI bootstrap mode. 'seed' treats paired random seeds as the "
            "independent unit and is fast/conservative; 'hierarchical' also "
            "resamples users within seeds and is much slower for aggregate Top-K metrics."
        ),
    )
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--required-seeds", default=None)
    parser.add_argument(
        "--exposure-groups",
        default=None,
        help="Optional comma-separated group override; defaults to protocol group metrics available per dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    frame = pd.read_csv(manifest_path)
    required_columns = {"dataset", "backbone", "seed", "baseline_npz", "method_npz"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"significance manifest is missing columns: {missing}")

    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = PROJECT_ROOT / protocol_path
    protocol = load_protocol(protocol_path)
    _, confirmatory_seeds = seed_tiers(protocol)
    required_seeds = _parse_ints(args.required_seeds) if args.required_seeds else confirmatory_seeds
    metric_contract = fairness_metric_contract(protocol)
    group_specs = {
        name: spec
        for name, spec in metric_contract["metrics"].items()
        if "group" in spec
    }
    requested_groups = _parse_strings(args.exposure_groups) if args.exposure_groups else None
    if "data_dir" not in frame.columns:
        raise ValueError("significance manifest requires data_dir for the reported fairness metrics")

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_audit = []

    def write_outputs(status: str) -> None:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        audit = {
            "manifest": str(manifest_path),
            "status": status,
            "required_seeds": required_seeds,
            "k": int(args.k),
            "ci_mode": str(args.ci_mode),
            "bootstrap_repetitions": int(args.repetitions),
            "bootstrap_seed": int(args.bootstrap_seed),
            "positive_difference_means": {
                "utility": "method minus baseline",
                "coverage": "method minus baseline",
                "rec_gini_and_group_gaps": "baseline minus method",
            },
            "fairness_metric_contract": metric_contract,
            "requested_groups": requested_groups,
            "pairs": pair_audit,
            "results": rows,
        }
        output_json.write_text(
            json.dumps(_json_safe(audit), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    for (dataset, backbone), group in frame.groupby(["dataset", "backbone"], sort=True):
        print(f"[significance] start dataset={dataset} backbone={backbone}", flush=True)
        group = group.sort_values("seed")
        observed_seeds = [int(seed) for seed in group["seed"].tolist()]
        if observed_seeds != sorted(required_seeds):
            raise ValueError(
                f"{dataset}/{backbone} has incomplete seed pairs: "
                f"expected={sorted(required_seeds)} observed={observed_seeds}"
            )

        paired_archives = []
        for _, pair in group.iterrows():
            base, method = _align_archive_rows(
                _load_archive(pair["baseline_npz"]),
                _load_archive(pair["method_npz"]),
            )
            paired_archives.append((int(pair["seed"]), base, method))
            pair_audit.append(
                {
                    "dataset": str(dataset),
                    "backbone": str(backbone),
                    "seed": int(pair["seed"]),
                    "num_users": int(base["users"].size),
                    "baseline_npz": str(pair["baseline_npz"]),
                    "method_npz": str(pair["method_npz"]),
                }
            )

        for metric in ("hr", "ndcg"):
            differences = []
            for _, base, method in paired_archives:
                base_values = per_user_ranking_metric(
                    base["topk_items"], base["targets"], args.k, metric
                )
                method_values = per_user_ranking_metric(
                    method["topk_items"], method["targets"], args.k, metric
                )
                differences.append(method_values - base_values)
            if args.ci_mode == "hierarchical":
                result = hierarchical_paired_bootstrap(
                    differences,
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            else:
                result = seed_level_paired_bootstrap(
                    [float(values.mean()) for values in differences],
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "metric": f"{metric}@{args.k}",
                    **result,
                }
            )
            print(f"[significance] done dataset={dataset} backbone={backbone} metric={metric}@{args.k}", flush=True)
            write_outputs("partial")

        data_dirs = {str(value) for value in group["data_dir"].dropna().tolist()}
        if len(data_dirs) != 1:
            raise ValueError(f"{dataset}/{backbone} must use exactly one data_dir")
        data_dir = next(iter(data_dirs))
        base_topk = [base["topk_items"] for _, base, _ in paired_archives]
        method_topk = [method["topk_items"] for _, _, method in paired_archives]
        num_items = _infer_num_items(data_dir)
        aggregate_metrics = [
            (
                "coverage",
                lambda items, n=num_items: recommendation_coverage(items, n, args.k),
                True,
            ),
            (
                "rec_gini",
                lambda items, n=num_items: recommendation_gini(items, n, args.k),
                False,
            ),
        ]
        for metric_name, metric_fn, higher_is_better in aggregate_metrics:
            if args.ci_mode == "hierarchical":
                result = hierarchical_topk_metric_bootstrap(
                    base_topk,
                    method_topk,
                    metric_fn=metric_fn,
                    higher_is_better=higher_is_better,
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            else:
                direction = 1.0 if higher_is_better else -1.0
                result = seed_level_paired_bootstrap(
                    [
                        direction * (float(metric_fn(method_items)) - float(metric_fn(base_items)))
                        for base_items, method_items in zip(base_topk, method_topk)
                    ],
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "metric": f"{metric_name}@{args.k}",
                    "positive_difference": "method_minus_baseline" if higher_is_better else "baseline_minus_method",
                    **result,
                }
            )
            print(f"[significance] done dataset={dataset} backbone={backbone} metric={metric_name}@{args.k}", flush=True)
            write_outputs("partial")

        available_groups = json.loads(
            (Path(data_dir) / "fairness_groups.json").read_text(encoding="utf-8")
        )
        for metric_name, spec in group_specs.items():
            group_name = str(spec["group"])
            if requested_groups is not None and group_name not in requested_groups:
                continue
            if group_name not in available_groups:
                continue
            labels = _load_group_labels(data_dir, group_name)
            utility = _load_group_utility(data_dir, group_name)
            metric_fn = lambda items, l=labels, u=utility: group_utility_aware_gap(
                items, l, u, args.k
            )
            if args.ci_mode == "hierarchical":
                result = hierarchical_topk_metric_bootstrap(
                    base_topk,
                    method_topk,
                    metric_fn=metric_fn,
                    higher_is_better=False,
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            else:
                result = seed_level_paired_bootstrap(
                    [
                        float(metric_fn(base_items)) - float(metric_fn(method_items))
                        for base_items, method_items in zip(base_topk, method_topk)
                    ],
                    repetitions=args.repetitions,
                    seed=args.bootstrap_seed,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "metric": f"{metric_name}@{args.k}",
                    "group_name": group_name,
                    "positive_difference": "baseline_minus_method",
                    **result,
                }
            )
            print(f"[significance] done dataset={dataset} backbone={backbone} metric={metric_name}@{args.k}", flush=True)
            write_outputs("partial")

    write_outputs("complete")
    print(f"Statistics CSV: {output_csv}")
    print(f"Statistics audit: {output_json}")


if __name__ == "__main__":
    main()
