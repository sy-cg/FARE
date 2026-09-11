#!/usr/bin/env python3
"""Audit whether a FairRR run had a valid pool and changed its rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_diagnostics import (
    candidate_pool_audit,
    fairrr_change_summary,
    validate_formal_fairrr,
)


def _topk(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "topk_items" not in archive.files:
            raise ValueError(f"{path} does not contain topk_items")
        return np.asarray(archive["topk_items"], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reranked", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _topk(args.source)
    reranked = _topk(args.reranked)
    pool = candidate_pool_audit(source.shape[1], args.candidate_k, args.top_k)
    change = fairrr_change_summary(source, reranked, args.top_k)
    audit = {
        "source": str(args.source),
        "reranked": str(args.reranked),
        "candidate_pool": pool,
        "change": change,
        "formal_required": bool(args.formal),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.formal:
        validate_formal_fairrr(change, pool)
    print(f"FairRR audit: {output}")


if __name__ == "__main__":
    main()

