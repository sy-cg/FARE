#!/usr/bin/env python3
"""Recompute comparable test metrics for all revision methods at one Top-K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import fairness_metric_contract, load_protocol, seed_tiers
from src.revision_statistics import (
    group_utility_aware_gap,
    per_user_ranking_metric,
    recommendation_coverage,
    recommendation_gini,
)


JOB_METHODS = {
    "FairRR",
    "ExposureReweight-ID",
    "FARE-IndependentPrior",
    "FARE+ModalityDebias",
}


def _read_csv(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    return pd.read_csv(path).to_dict("records")


def _resolve(path: str | Path) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else (ROOT / value).resolve()


def _ranking_path(run_dir: Path) -> Path | None:
    for name in ("topk_test.npz", "test_ranking.npz"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _load_archive(path: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = {"targets", "topk_items"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        targets = np.asarray(archive["targets"], dtype=np.int64).reshape(-1)
        topk = np.asarray(archive["topk_items"], dtype=np.int64)
    if topk.ndim != 2 or topk.shape[0] != targets.size:
        raise ValueError(f"unaligned targets/topk_items in {path}")
    if topk.shape[1] < k:
        raise ValueError(f"{path} has width={topk.shape[1]} but k={k}")
    return targets, topk[:, :k]


def _load_group_labels(data_dir: Path, group_name: str) -> np.ndarray:
    groups = json.loads((data_dir / "fairness_groups.json").read_text(encoding="utf-8"))
    mapping = groups[group_name]
    labels = np.full(max(int(item) for item in mapping) + 1, -1, dtype=np.int64)
    for item, value in mapping.items():
        labels[int(item)] = int(value)
    return labels


def _load_group_utility(data_dir: Path, group_name: str) -> dict[str, float]:
    utility = json.loads((data_dir / "group_utility_train.json").read_text(encoding="utf-8"))[
        group_name
    ]
    return {str(key): float(value) for key, value in utility.items()}


def _dataset_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    datasets_path = _resolve(protocol.get("datasets_config", "configs/datasets.yaml"))
    datasets = (yaml.safe_load(datasets_path.read_text(encoding="utf-8")) or {}).get(
        "datasets", {}
    )
    return {name: _resolve(spec["path"]) for name, spec in datasets.items()}


def _source_manifest(args: argparse.Namespace, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    breadth, _ = seed_tiers(protocol)
    breadth_set = set(breadth)
    data_paths = _dataset_paths(protocol)
    rows: list[dict[str, Any]] = []

    def add(dataset: str, backbone: str, method: str, seed: int, run_dir: Any) -> None:
        if int(seed) not in breadth_set:
            return
        rows.append(
            {
                "dataset": str(dataset),
                "backbone": str(backbone).lower(),
                "method": str(method),
                "seed": int(seed),
                "run_dir": str(run_dir),
                "data_dir": str(data_paths[str(dataset)]),
            }
        )

    for row in _read_csv(args.id_registry):
        add(row["dataset"], row["backbone"], "ID", row["seed"], row["run_dir"])
    for row in _read_csv(args.fare_registry):
        add(row["dataset"], row["backbone"], "FARE", row["seed"], row["run_dir"])
    for row in _read_csv(args.findrec_registry):
        add(row["dataset"], "findrec", "FindRec", row["seed"], row["run_dir"])
    for row in _read_csv(args.jobs_manifest):
        method = str(row.get("method", ""))
        status = str(row.get("status", "")).lower()
        if method not in JOB_METHODS or status not in {"ready", "complete_existing"}:
            continue
        run_dir = row.get("expected_run_dir")
        if pd.isna(run_dir) or not str(run_dir).strip():
            raise ValueError(f"{method} job is missing expected_run_dir: {row}")
        add(row["dataset"], row["backbone"], method, row["seed"], run_dir)

    if not rows:
        raise ValueError("No result sources were supplied")
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "backbone", "method", "seed"], keep="last"
    )
    return frame.sort_values(["dataset", "backbone", "method", "seed"]).to_dict("records")


def _optional_fairrr_audit(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics_summary.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    change = summary.get("test_change_audit", {}) or {}
    pool = summary.get("test_candidate_pool_audit", {}) or {}
    selection = summary.get("lambda_selection_audit", {}) or {}
    constraints = summary.get("fairrr_selection_constraints", {}) or selection.get(
        "constraints", {}
    ) or {}
    return {
        "fairrr_lambda": summary.get("lambda_fair", selection.get("selected_lambda_fair")),
        "fairrr_rank_changed_user_rate": change.get("rank_changed_user_rate"),
        "fairrr_set_changed_user_rate": change.get("set_changed_user_rate"),
        "fairrr_effective_candidate_k": pool.get("effective_candidate_k"),
        "fairrr_selection_reason": selection.get("selection_reason"),
        "fairrr_validation_utility_retention": selection.get(
            "selected_utility_retention"
        ),
        "fairrr_validation_fairness_improvement": selection.get(
            "selected_fairness_improvement"
        ),
        "fairrr_max_relative_utility_loss": constraints.get(
            "max_relative_utility_loss"
        ),
        "fairrr_min_rank_changed_user_rate": constraints.get(
            "min_rank_changed_user_rate"
        ),
        "fairrr_min_set_changed_user_rate": constraints.get(
            "min_set_changed_user_rate"
        ),
        "fairrr_min_fairness_improvement": constraints.get(
            "min_fairness_improvement"
        ),
    }

def _aggregate(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["dataset", "backbone", "method"], sort=True):
        row = dict(zip(("dataset", "backbone", "method"), keys))
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--id-registry")
    parser.add_argument("--fare-registry")
    parser.add_argument("--jobs-manifest")
    parser.add_argument("--findrec-registry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    protocol = load_protocol(_resolve(args.protocol))
    sources = _source_manifest(args, protocol)
    contract = fairness_metric_contract(protocol)["metrics"]
    group_metrics = {
        name: str(spec["group"])
        for name, spec in contract.items()
        if "group" in spec
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sources).to_csv(output_dir / "revision_result_manifest.csv", index=False)

    rows = []
    issues = []
    for source in sources:
        run_dir = Path(source["run_dir"])
        ranking_path = _ranking_path(run_dir)
        if ranking_path is None:
            issues.append({**source, "reason": "missing topk_test.npz or test_ranking.npz"})
            continue
        try:
            targets, topk = _load_archive(ranking_path, args.k)
            data_dir = Path(source["data_dir"])
            popularity = np.load(data_dir / "item_popularity_train.npy", allow_pickle=False)
            num_items = int(popularity.size - 1)
            row = {
                **source,
                "ranking_path": str(ranking_path),
                "split": "test",
                "k": int(args.k),
                "num_users": int(targets.size),
                "topk_width": int(topk.shape[1]),
                f"hr@{args.k}": float(
                    per_user_ranking_metric(topk, targets, args.k, "hr").mean()
                ),
                f"ndcg@{args.k}": float(
                    per_user_ranking_metric(topk, targets, args.k, "ndcg").mean()
                ),
                f"coverage@{args.k}": recommendation_coverage(topk, num_items, args.k),
                f"rec_gini@{args.k}": recommendation_gini(topk, num_items, args.k),
            }
            available = json.loads(
                (data_dir / "fairness_groups.json").read_text(encoding="utf-8")
            )
            for metric_name, group_name in group_metrics.items():
                if group_name not in available:
                    continue
                row[f"{metric_name}@{args.k}"] = group_utility_aware_gap(
                    topk,
                    _load_group_labels(data_dir, group_name),
                    _load_group_utility(data_dir, group_name),
                    args.k,
                )
            if source["method"] == "FairRR":
                row.update(_optional_fairrr_audit(run_dir))
            rows.append(row)
        except Exception as exc:
            issues.append({**source, "reason": str(exc)})

    per_run = pd.DataFrame(rows)
    per_run.to_csv(output_dir / "revision_results_per_run.csv", index=False)
    base_metrics = [
        f"hr@{args.k}",
        f"ndcg@{args.k}",
        f"coverage@{args.k}",
        f"rec_gini@{args.k}",
    ]
    metric_columns = [
        *base_metrics,
        *[f"{name}@{args.k}" for name in group_metrics if f"{name}@{args.k}" in per_run],
    ]
    aggregate = _aggregate(per_run, metric_columns) if len(per_run) else pd.DataFrame()
    aggregate.to_csv(output_dir / "revision_results_aggregate.csv", index=False)
    audit = {
        "split": "test",
        "k": int(args.k),
        "manifest_runs": len(sources),
        "complete_runs": len(rows),
        "missing_runs": len(issues),
        "method_counts": per_run["method"].value_counts().to_dict() if len(per_run) else {},
        "issues": issues,
    }
    (output_dir / "revision_results_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if issues and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
