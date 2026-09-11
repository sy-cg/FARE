"""Join validation-selected configurations back to their seed-level run dirs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import load_protocol, seed_tiers


KEYS = ["dataset", "backbone", "config_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--run-registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--required-seeds", default=None)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected)
    runs = pd.read_csv(args.run_registry)
    for name, frame, required in [
        ("selected", selected, set(KEYS)),
        ("run registry", runs, set(KEYS) | {"seed", "run_dir"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if selected.duplicated(KEYS).any():
        raise ValueError("selected table contains duplicate dataset/backbone/config_id rows")

    joined = runs.merge(selected[KEYS], on=KEYS, how="inner", validate="many_to_one")
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    breadth_seeds, _ = seed_tiers(load_protocol(protocol_path))
    expected = (
        {int(value) for value in args.required_seeds.split(",") if value.strip()}
        if args.required_seeds
        else set(breadth_seeds)
    )
    for key, group in joined.groupby(KEYS, dropna=False):
        observed = {int(value) for value in group["seed"]}
        if observed != expected:
            raise ValueError(f"Selected configuration {key} has seed set {sorted(observed)}, expected {sorted(expected)}")
    if len(joined.groupby(KEYS)) != len(selected):
        raise ValueError("At least one selected configuration has no matching seed runs")
    joined = joined.sort_values(["dataset", "backbone", "seed"], kind="mergesort")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    print(f"Selected seed-level runs: {len(joined)} -> {output}")


if __name__ == "__main__":
    main()
