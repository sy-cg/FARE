#!/usr/bin/env python3
"""Execute ready rows from a generated revision jobs.csv without a shell."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict]) -> None:
    fields = [
        "phase",
        "method",
        "dataset",
        "backbone",
        "seed",
        "execution_status",
        "returncode",
        "argv_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(name, "")) for name in ("phase", "method", "dataset", "backbone", "seed"))


def _upsert_status_row(rows: list[dict], new_row: dict) -> None:
    key = _key(new_row)
    for idx, row in enumerate(rows):
        if _key(row) == key:
            rows[idx] = new_row
            return
    rows.append(new_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--backbone", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--status-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as f:
        jobs = list(csv.DictReader(f))
    jobs = [row for row in jobs if str(row.get("status", "")).lower() == "ready"]
    if args.phase:
        jobs = [row for row in jobs if row.get("phase") in set(args.phase)]
    if args.method:
        jobs = [row for row in jobs if row.get("method") in set(args.method)]
    if args.dataset:
        jobs = [row for row in jobs if row.get("dataset") in set(args.dataset)]
    if args.backbone:
        selected = {value.lower() for value in args.backbone}
        jobs = [row for row in jobs if str(row.get("backbone", "")).lower() in selected]
    if args.seed:
        jobs = [row for row in jobs if int(row.get("seed", -1)) in set(args.seed)]

    status_path = Path(args.status_out)
    status_rows = []
    if args.resume and status_path.exists():
        with status_path.open(encoding="utf-8", newline="") as f:
            status_rows = list(csv.DictReader(f))
    completed = {
        _key(row)
        for row in status_rows
        if row.get("execution_status") == "ok"
    }

    for job in jobs:
        if _key(job) in completed:
            continue
        argv = json.loads(job["argv_json"])
        if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
            raise ValueError(f"Invalid argv_json for job {_key(job)}")
        print(json.dumps({"job": _key(job), "argv": argv}, ensure_ascii=False))
        if args.dry_run:
            returncode = 0
            execution_status = "dry_run"
        else:
            returncode = int(subprocess.run(argv, cwd=ROOT, check=False).returncode)
            execution_status = "ok" if returncode == 0 else "failed"
        _upsert_status_row(
            status_rows,
            {
                **job,
                "execution_status": execution_status,
                "returncode": returncode,
            },
        )
        _write(status_path, status_rows)
        if returncode != 0 and not args.continue_on_error:
            raise SystemExit(returncode)
    if not status_path.exists():
        _write(status_path, status_rows)


if __name__ == "__main__":
    main()

