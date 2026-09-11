#!/usr/bin/env python3
"""Replace one backbone in a validation-run registry with corrected reruns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["dataset", "backbone", "seed", "config_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--replace-backbone", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    replacement = pd.read_csv(args.replacement)
    required = set(KEYS) | {"run_dir", "split"}
    for name, frame in (("base", base), ("replacement", replacement)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} registry missing columns: {sorted(missing)}")
    backbone = args.replace_backbone.lower()
    if set(replacement["backbone"].astype(str).str.lower()) != {backbone}:
        raise ValueError("replacement registry must contain only the requested backbone")
    retained = base[base["backbone"].astype(str).str.lower() != backbone]
    merged = pd.concat([retained, replacement], ignore_index=True, sort=False)
    if merged.duplicated(KEYS).any():
        duplicates = merged.loc[merged.duplicated(KEYS, keep=False), KEYS]
        raise ValueError(f"merged registry has duplicate keys: {duplicates.head().to_dict('records')}")
    merged = merged.sort_values(KEYS, kind="mergesort")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged validation runs: {len(merged)} -> {output}")


if __name__ == "__main__":
    main()
