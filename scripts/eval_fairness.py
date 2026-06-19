# -*- coding: utf-8 -*-
"""
scripts/eval_fairness.py

Re-run fairness evaluation from saved RankingResult .npz files.

This script is useful when:
- fairness metrics are updated;
- group_names need to be changed;
- you want to evaluate optional proxy groups;
- you do not want to re-train or re-run model scoring.

Run from project root:
    python scripts/eval_fairness.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/sasrec_id/20260515_143012 \
        --split test

Evaluate both val and test:
    python scripts/eval_fairness.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/sasrec_id/20260515_143012 \
        --split all

Evaluate all available fairness groups, including proxy groups:
    python scripts/eval_fairness.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/sasrec_id/20260515_143012 \
        --split test \
        --groups all

Evaluate selected groups:
    python scripts/eval_fairness.py \
        --dataset Video_Games \
        --run_dir results/Video_Games/sasrec_id/20260515_143012 \
        --split test \
        --groups popularity_group,modality_availability_group,text_quality_group,vision_quality_group

Expected saved ranking file names in run_dir:
    val_ranking.npz
    test_ranking.npz

Outputs by default:
    <run_dir>/<split>_fairness_group_ranking.csv
    <run_dir>/<split>_fairness_exposure.csv
    <run_dir>/<split>_fairness_summary.csv
    <run_dir>/<split>_fairness_flat_metrics.json

It can also append compact aggregate rows into:
    results/fairness.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

# Ensure project root is on sys.path when running `python scripts/eval_fairness.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_fairness import FairnessEvaluator, save_eval_tables
from src.evaluator_ranking import RankingResult, load_ranking_result_npz


# ==========================================================
# Basic utilities
# ==========================================================


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_dataset_dir(project_root: Path, datasets_cfg_path: str | Path, dataset_name: str) -> Path:
    datasets_cfg_path = Path(datasets_cfg_path)
    if not datasets_cfg_path.is_absolute():
        datasets_cfg_path = project_root / datasets_cfg_path
    datasets_cfg = load_yaml(datasets_cfg_path)
    datasets = datasets_cfg.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Dataset {dataset_name!r} not found in {datasets_cfg_path}. Available: {list(datasets)}")

    data_path = Path(datasets[dataset_name]["path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")
    return data_path


def parse_groups_arg(groups_arg: str, data_dir: Path) -> Optional[List[str]]:
    """Parse --groups.

    Returns:
        None for mandatory default groups used by FairnessEvaluator.
        Explicit list for all/custom groups.
    """
    groups_arg = groups_arg.strip()
    if groups_arg.lower() in {"mandatory", "default", "required"}:
        return None

    fairness_groups_path = data_dir / "fairness_groups.json"
    if not fairness_groups_path.exists():
        raise FileNotFoundError(f"Missing fairness_groups.json: {fairness_groups_path}")
    available = list(load_json(fairness_groups_path).keys())

    if groups_arg.lower() == "all":
        return available

    groups = [g.strip() for g in groups_arg.split(",") if g.strip()]
    if not groups:
        raise ValueError("--groups is empty. Use mandatory, all, or comma-separated group names.")

    missing = [g for g in groups if g not in available]
    if missing:
        raise KeyError(f"Unknown fairness groups: {missing}. Available groups: {available}")
    return groups


def get_splits(split_arg: str) -> List[str]:
    split_arg = split_arg.strip().lower()
    if split_arg == "all":
        return ["val", "test"]
    splits = [s.strip() for s in split_arg.split(",") if s.strip()]
    valid = {"val", "test", "train"}
    unknown = [s for s in splits if s not in valid]
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Valid: val, test, train, all")
    return splits


def infer_run_metadata(run_dir: Path) -> Dict[str, str]:
    """Infer model/run metadata from results/<Dataset>/<model>/<run_id>."""
    parts = run_dir.parts
    meta = {
        "model_name": "unknown_model",
        "run_id": run_dir.name,
    }
    # Typical: results/Video_Games/sasrec_id/20260515_143012
    if len(parts) >= 3:
        meta["model_name"] = parts[-2]
    return meta


# ==========================================================
# Flattening / global summary
# ==========================================================


def flatten_fairness_tables(
    dataset: str,
    split: str,
    model_name: str,
    run_id: str,
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Create a compact flat metrics dictionary for easy logging.

    This does not replace the detailed CSV files. It only extracts common aggregate
    metrics used for quick comparison across models/runs.
    """
    flat: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "run_id": run_id,
    }

    summary = tables.get("summary", pd.DataFrame())
    if not summary.empty:
        for _, row in summary.iterrows():
            k = int(row["k"])
            for col in summary.columns:
                if col == "k":
                    continue
                val = row[col]
                if pd.notna(val):
                    flat[f"summary@{k}/{col}"] = float(val) if isinstance(val, (int, float)) else val

    exposure = tables.get("exposure", pd.DataFrame())
    if not exposure.empty and "group_value" in exposure.columns:
        agg = exposure[exposure["group_value"].astype(str) == "__aggregate__"].copy()
        for _, row in agg.iterrows():
            k = int(row["k"])
            group_name = str(row["group_name"])
            for col in [
                "exposure_share_gap",
                "exposure_share_gini",
                "utility_aware_gap",
                "utility_aware_l1",
            ]:
                if col in row and pd.notna(row[col]):
                    flat[f"{group_name}@{k}/{col}"] = float(row[col])

    group_ranking = tables.get("group_ranking", pd.DataFrame())
    if not group_ranking.empty:
        # Useful compact gaps for each group type and metric.
        for group_name, df_g in group_ranking.groupby("group_name"):
            for metric in ["recall@20", "ndcg@20", "mrr@20"]:
                if metric in df_g.columns and len(df_g[metric].dropna()) > 0:
                    vals = df_g[metric].astype(float).to_numpy()
                    flat[f"{group_name}/{metric}_gap"] = float(vals.max() - vals.min())
                    flat[f"{group_name}/{metric}_min"] = float(vals.min())
                    flat[f"{group_name}/{metric}_max"] = float(vals.max())

    return flat


def append_global_fairness_csv(flat_rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Append flat metrics to global results/fairness.csv.

    Uses union of existing/new columns. This is convenient for quick tracking,
    while detailed per-run CSV files remain the authoritative records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(flat_rows)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        combined = pd.concat([old_df, new_df], axis=0, ignore_index=True, sort=False)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)


# ==========================================================
# Main evaluation logic
# ==========================================================


def evaluate_one_split(
    dataset: str,
    data_dir: Path,
    run_dir: Path,
    split: str,
    groups: Optional[Sequence[str]],
    ks: Sequence[int],
    output_dir: Path,
    prefix: Optional[str],
    save_flat_json: bool,
) -> Dict[str, Any]:
    candidate_ranking_paths = [
        run_dir / f"{split}_ranking.npz",
        run_dir / f"topk_{split}.npz",
    ]

    ranking_path = None
    for p in candidate_ranking_paths:
        if p.exists():
            ranking_path = p
            break

    if ranking_path is None:
        raise FileNotFoundError(
            "Saved ranking result not found. Tried: "
            + ", ".join(str(p) for p in candidate_ranking_paths)
        )

    print(f"Loading ranking result: {ranking_path}")
    ranking_result = load_ranking_result_npz(str(ranking_path))

    evaluator = FairnessEvaluator(
        data_dir=str(data_dir),
        ks=ks,
        group_names=groups,
    )
    tables = evaluator.evaluate(ranking_result)

    if prefix is None:
        # Match training script naming:
        # test_fairness_group_ranking.csv, test_fairness_exposure.csv, ...
        prefix = f"{split}_fairness"

    print(f"Saving fairness tables to: {output_dir}")
    save_eval_tables(tables, output_dir=str(output_dir), prefix=prefix)

    meta = infer_run_metadata(run_dir)
    flat = flatten_fairness_tables(
        dataset=dataset,
        split=split,
        model_name=meta["model_name"],
        run_id=meta["run_id"],
        tables=tables,
    )

    if save_flat_json:
        save_json(flat, output_dir / f"{prefix}_flat_metrics.json")

    print(f"Finished split={split}. Key files:")
    print(f"  {output_dir / (prefix + '_group_ranking.csv')}")
    print(f"  {output_dir / (prefix + '_exposure.csv')}")
    print(f"  {output_dir / (prefix + '_summary.csv')}")
    return flat


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run fairness evaluation from saved ranking .npz files")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name in configs/datasets.yaml")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing <split>_ranking.npz")
    parser.add_argument("--datasets_config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--split", type=str, default="test", help="test, val, train, comma-separated, or all")
    parser.add_argument("--groups", type=str, default="mandatory", help="mandatory, all, or comma-separated group names")
    parser.add_argument("--ks", type=str, default="5,10,20", help="Comma-separated K values")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: run_dir")
    parser.add_argument("--prefix", type=str, default=None, help="Default: <split>_fairness")
    parser.add_argument("--no_flat_json", action="store_true", help="Do not save <prefix>_flat_metrics.json")
    parser.add_argument("--append_global", action="store_true", help="Append flat metrics to results/fairness.csv")
    parser.add_argument("--global_csv", type=str, default="results/fairness.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = args.dataset
    data_dir = resolve_dataset_dir(PROJECT_ROOT, args.datasets_config, dataset)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = get_splits(args.split)
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    if not ks:
        raise ValueError("--ks cannot be empty")

    groups = parse_groups_arg(args.groups, data_dir)

    print("=" * 90)
    print("Fairness evaluation")
    print(f"Dataset:    {dataset}")
    print(f"Data dir:   {data_dir}")
    print(f"Run dir:    {run_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Splits:     {splits}")
    print(f"Ks:         {ks}")
    print(f"Groups:     {groups if groups is not None else 'mandatory defaults'}")
    print("=" * 90)

    flat_rows: List[Dict[str, Any]] = []
    for split in splits:
        # If multiple splits are requested, avoid using a single fixed prefix.
        prefix = args.prefix
        if prefix is not None and len(splits) > 1:
            prefix = f"{split}_{prefix}"

        flat = evaluate_one_split(
            dataset=dataset,
            data_dir=data_dir,
            run_dir=run_dir,
            split=split,
            groups=groups,
            ks=ks,
            output_dir=output_dir,
            prefix=prefix,
            save_flat_json=not args.no_flat_json,
        )
        flat_rows.append(flat)

    if args.append_global:
        global_csv = Path(args.global_csv)
        if not global_csv.is_absolute():
            global_csv = PROJECT_ROOT / global_csv
        append_global_fairness_csv(flat_rows, global_csv)
        print(f"Appended flat metrics to: {global_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
