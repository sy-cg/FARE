# -*- coding: utf-8 -*-
"""
Prepare RecBole-format assets for the released FindRec code.

This is optional. The main project-native comparison entrypoint is
scripts/run_findrec.py because it uses the same full-sort ranking and fairness
evaluation pipeline as FARE. Use this script when you need to run the original
RecBole-based FindRec code for a utility-only cross-check.

Example:
    python scripts/prepare_findrec_recbole.py --dataset Video_Games
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.findrec_adapter import prepare_findrec_assets


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def resolve_dataset_dir(dataset: str, datasets_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg = load_yaml(PROJECT_ROOT / datasets_config)
    datasets = cfg.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Dataset {dataset!r} not found. Available: {list(datasets)}")
    info = datasets[dataset]
    path = Path(info["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RecBole assets for FindRec")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datasets_config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--findrec_dir", type=str, default="FindRec-main")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, info = resolve_dataset_dir(args.dataset, args.datasets_config)
    max_seq_len = int(args.max_seq_len if args.max_seq_len is not None else info.get("max_seq_len", 50))
    manifest = prepare_findrec_assets(
        project_data_dir=data_dir,
        findrec_dir=PROJECT_ROOT / args.findrec_dir,
        dataset_name=args.dataset,
        seed=int(args.seed),
        max_seq_len=max_seq_len,
        epochs=int(args.epochs),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
