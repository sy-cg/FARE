# -*- coding: utf-8 -*-
"""
scripts/run_hparam_sweep.py

Generic hyperparameter sweep runner for this project.

It launches existing training/evaluation scripts with different CLI arguments,
records metrics_summary.json, and writes sweep_results.csv.

Supported target scripts:
    scripts/run_fare.py
    scripts/run_adv_backbone.py
    scripts/run_pop_reweight.py
    scripts/run_reranker_fair.py
    scripts/run_baselines.py
    scripts/run_id_backbone.py

Example:

    python scripts/run_hparam_sweep.py \
      --config configs/sweeps/fare_video_sasrec.yaml \
      --gpu 0

Dry run:

    python scripts/run_hparam_sweep.py \
      --config configs/sweeps/fare_video_sasrec.yaml \
      --dry_run

Resume:

    python scripts/run_hparam_sweep.py \
      --config configs/sweeps/fare_video_sasrec.yaml \
      --resume
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================================
# Basic utilities
# ==========================================================


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def save_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_slug(text: Any, max_len: int = 80) -> str:
    s = str(text)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "x"


def short_float(value: float) -> str:
    s = f"{value:g}"
    s = s.replace(".", "p").replace("-", "m")
    return s


def value_to_slug(value: Any) -> str:
    if isinstance(value, float):
        return short_float(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "-".join(value_to_slug(x) for x in value)
    if isinstance(value, dict):
        return "-".join(f"{sanitize_slug(k)}={value_to_slug(v)}" for k, v in value.items())
    return sanitize_slug(value)


def format_template(template: str, context: Dict[str, Any]) -> str:
    flat: Dict[str, Any] = dict(context)

    params = context.get("params", {})
    if isinstance(params, dict):
        for k, v in params.items():
            flat[f"param_{k}"] = v
            flat[k] = v

    return template.format_map(SafeDict(flat))


def nested_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        if part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ==========================================================
# CLI argument construction
# ==========================================================


def cli_key(name: str) -> str:
    return "--" + str(name).strip().replace("_", "_")


def cli_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    if isinstance(value, dict):
        return ",".join(f"{k}:{v}" for k, v in value.items())
    return str(value)


def build_cli_args(arg_map: Dict[str, Any]) -> List[str]:
    args: List[str] = []

    for key, value in arg_map.items():
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                args.append(cli_key(key))
            continue

        args.append(cli_key(key))
        v = cli_value(value)
        if v is not None:
            args.append(v)

    return args


# ==========================================================
# Trial generation
# ==========================================================


def normalize_search_space(space: Dict[str, Any]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for key, values in space.items():
        if isinstance(values, list):
            out[key] = values
        else:
            out[key] = [values]
    return out


def grid_trials(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    trials = []
    for combo in itertools.product(*values):
        trials.append({k: v for k, v in zip(keys, combo)})
    return trials


def sampled_trials(
    search_space: Dict[str, List[Any]],
    sampler: str,
    seed: int,
    max_trials: Optional[int],
) -> List[Dict[str, Any]]:
    trials = grid_trials(search_space)
    rng = random.Random(seed)

    if sampler == "grid":
        return trials[:max_trials] if max_trials else trials

    if sampler == "random":
        rng.shuffle(trials)
        return trials[:max_trials] if max_trials else trials

    raise ValueError(f"Unknown sampler={sampler!r}. Expected grid or random.")


def make_run_id(
    cfg: Dict[str, Any],
    trial_index: int,
    params: Dict[str, Any],
    merged_args: Dict[str, Any],
) -> str:
    run_id_cfg = cfg.get("run_id", {}) or {}
    if isinstance(run_id_cfg, str):
        run_id_cfg = {"template": run_id_cfg}

    context = {
        "trial": trial_index,
        "trial_id": f"t{trial_index:03d}",
        "dataset": merged_args.get("dataset", cfg.get("dataset", cfg.get("default_dataset", "dataset"))),
        "backbone": merged_args.get("backbone", merged_args.get("model", "")),
        "params": params,
    }

    template = run_id_cfg.get("template")
    if template:
        return sanitize_slug(format_template(str(template), context), max_len=180)

    prefix = str(run_id_cfg.get("prefix", cfg.get("name", "sweep")))
    include_params = run_id_cfg.get("include_params", list(params.keys()))

    pieces = [sanitize_slug(prefix), f"t{trial_index:03d}"]
    for key in include_params:
        if key in params:
            pieces.append(f"{sanitize_slug(key)}{value_to_slug(params[key])}")

    return sanitize_slug("_".join(pieces), max_len=180)


def build_trials(cfg: Dict[str, Any], cli_max_trials: Optional[int]) -> List[Dict[str, Any]]:
    search_space = normalize_search_space(cfg.get("search_space", {}) or {})

    sampler = str(cfg.get("sampler", "grid")).lower()
    seed = int(cfg.get("seed", 2026))

    max_trials_cfg = cfg.get("max_trials")
    max_trials = cli_max_trials if cli_max_trials is not None else max_trials_cfg
    max_trials = int(max_trials) if max_trials is not None else None

    params_list = sampled_trials(
        search_space=search_space,
        sampler=sampler,
        seed=seed,
        max_trials=max_trials,
    )

    base_args = dict(cfg.get("static_args", {}) or cfg.get("base_args", {}) or {})

    trials = []
    for idx, params in enumerate(params_list, start=1):
        merged_args = dict(base_args)
        merged_args.update(params)

        if "run_id" not in merged_args or not merged_args["run_id"]:
            merged_args["run_id"] = make_run_id(cfg, idx, params, merged_args)

        trials.append(
            {
                "trial_index": idx,
                "params": params,
                "args": merged_args,
                "run_id": merged_args["run_id"],
            }
        )

    return trials


# ==========================================================
# Metric extraction
# ==========================================================


def find_metric_file(cfg: Dict[str, Any], trial: Dict[str, Any]) -> Optional[Path]:
    metric_cfg = cfg.get("metric", {}) or {}

    context = {
        "trial": trial["trial_index"],
        "trial_id": f"t{trial['trial_index']:03d}",
        "run_id": trial["run_id"],
        "params": trial["params"],
        **trial["args"],
    }

    template = metric_cfg.get("file_template") or cfg.get("metric_file_template")
    if template:
        p = resolve_path(format_template(str(template), context))
        if p.exists():
            return p
        return p

    # Fallback: search results/**/<run_id>/metrics_summary.json
    run_id = str(trial["run_id"])
    result_root = resolve_path(str(cfg.get("result_root", "results")))
    candidates = list(result_root.glob(f"**/{run_id}/metrics_summary.json"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    return None


def read_metric(metric_file: Path, metric_key: str) -> Optional[float]:
    if not metric_file.exists():
        return None

    with open(metric_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    value = nested_get(obj, metric_key, default=None)

    if value is None:
        # Common fallback.
        if metric_key == "best_metric":
            value = obj.get("best_metric")
        elif metric_key.startswith("test/"):
            value = nested_get(obj, metric_key.replace("/", "."), default=None)
        elif metric_key.startswith("val/"):
            value = nested_get(obj, metric_key.replace("/", "."), default=None)

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


# ==========================================================
# Running
# ==========================================================


def resolve_sweep_script(script: str | Path) -> Path:
    script_path = resolve_path(script).resolve()
    scripts_dir = (PROJECT_ROOT / "scripts").resolve()

    if script_path.suffix.lower() != ".py":
        raise ValueError(f"Sweep script must be a Python file under scripts/: {script}")

    try:
        script_path.relative_to(scripts_dir)
    except ValueError as exc:
        raise ValueError(f"Sweep script must stay under {scripts_dir}: {script}") from exc

    if not script_path.exists():
        raise FileNotFoundError(f"Sweep script not found: {script_path}")

    return script_path


def build_command(cfg: Dict[str, Any], trial: Dict[str, Any]) -> List[str]:
    python_exec = str(cfg.get("python", sys.executable))
    script = cfg.get("script")
    if not script:
        raise ValueError("Sweep config must define script: scripts/xxx.py")

    script_path = str(resolve_sweep_script(script))
    args = build_cli_args(trial["args"])

    return [python_exec, script_path] + args


def run_command(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = " ".join(shlex.quote(x) for x in cmd)
    print(printable)

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(printable + "\n\n")
        log_f.flush()

        if dry_run:
            return 0

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        return int(proc.wait())


def trial_signature(trial: Dict[str, Any]) -> str:
    payload = {
        "run_id": trial["run_id"],
        "params": trial["params"],
        "args": trial["args"],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def already_done(done_rows: List[Dict[str, Any]], trial: Dict[str, Any]) -> bool:
    sig = trial_signature(trial)
    for row in done_rows:
        if row.get("signature") == sig and row.get("status") == "ok":
            return True
        if row.get("run_id") == trial.get("run_id") and row.get("status") == "ok":
            return True
    return False


def pick_best(rows: List[Dict[str, Any]], mode: str) -> Optional[Dict[str, Any]]:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("metric") is not None]
    if not ok_rows:
        return None

    reverse = mode == "max"
    return sorted(ok_rows, key=lambda r: float(r["metric"]), reverse=reverse)[0]


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic hyperparameter sweep runner")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--start", type=int, default=1, help="1-based trial index to start")
    parser.add_argument("--end", type=int, default=None, help="1-based trial index to end")
    parser.add_argument("--sweep_dir", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)

    name = str(cfg.get("name", cfg_path.stem))
    sweep_dir = resolve_path(args.sweep_dir) if args.sweep_dir else PROJECT_ROOT / "sweeps" / sanitize_slug(name)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    log_dir = sweep_dir / "logs"
    result_jsonl = sweep_dir / "sweep_results.jsonl"
    result_csv = sweep_dir / "sweep_results.csv"

    metric_cfg = cfg.get("metric", {}) or {}
    metric_key = str(metric_cfg.get("key", "best_metric"))
    metric_mode = str(metric_cfg.get("mode", "max")).lower()
    if metric_mode not in {"max", "min"}:
        raise ValueError("metric.mode must be max or min")

    trials = build_trials(cfg, cli_max_trials=args.max_trials)

    start = max(int(args.start), 1)
    end = int(args.end) if args.end is not None else len(trials)
    trials = [t for t in trials if start <= int(t["trial_index"]) <= end]

    existing_rows = load_jsonl(result_jsonl) if args.resume and not args.no_resume else []

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("========== Hyperparameter Sweep ==========")
    print(f"config:       {cfg_path}")
    print(f"name:         {name}")
    print(f"sweep_dir:    {sweep_dir}")
    print(f"num_trials:   {len(trials)}")
    print(f"metric_key:   {metric_key}")
    print(f"metric_mode:  {metric_mode}")
    print(f"gpu:          {args.gpu}")
    print(f"dry_run:      {args.dry_run}")
    print(f"resume:       {args.resume and not args.no_resume}")

    all_rows = list(existing_rows)

    for trial in trials:
        idx = int(trial["trial_index"])
        run_id = str(trial["run_id"])

        if args.resume and not args.no_resume and already_done(existing_rows, trial):
            print(f"[skip] trial={idx} run_id={run_id} already completed")
            continue

        print("\n" + "=" * 80)
        print(f"Trial {idx} | run_id={run_id}")
        print(json.dumps(trial["params"], indent=2, ensure_ascii=False, default=str))

        cmd = build_command(cfg, trial)
        log_path = log_dir / f"trial_{idx:03d}_{sanitize_slug(run_id, 80)}.log"

        start_time = time.time()
        returncode = run_command(
            cmd=cmd,
            log_path=log_path,
            env=env,
            dry_run=args.dry_run,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start_time

        status = "ok" if returncode == 0 else "failed"

        metric_file = find_metric_file(cfg, trial)
        metric_value = None
        if status == "ok" and metric_file is not None:
            metric_value = read_metric(metric_file, metric_key)

        row: Dict[str, Any] = {
            "trial": idx,
            "run_id": run_id,
            "status": status,
            "returncode": returncode,
            "metric_key": metric_key,
            "metric": metric_value,
            "metric_file": str(metric_file) if metric_file else "",
            "elapsed_sec": round(elapsed, 3),
            "log_path": str(log_path),
            "signature": trial_signature(trial),
        }

        for k, v in trial["params"].items():
            row[f"param/{k}"] = v

        for k, v in trial["args"].items():
            if k not in trial["params"]:
                row[f"arg/{k}"] = v

        save_jsonl_row(result_jsonl, row)
        all_rows.append(row)
        write_csv(result_csv, all_rows)

        print(f"[trial done] status={status} metric={metric_value} elapsed={elapsed:.1f}s")
        print(f"log: {log_path}")

        if returncode != 0 and not args.continue_on_error:
            print("[stop] trial failed. Use --continue_on_error to continue.")
            break

    best = pick_best(all_rows, metric_mode)
    print("\n========== Sweep Summary ==========")
    print(f"results_csv: {result_csv}")
    print(f"results_jsonl: {result_jsonl}")

    if best:
        print("Best trial:")
        print(json.dumps(best, indent=2, ensure_ascii=False, default=str))
        with open(sweep_dir / "best_trial.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False, default=str)
    else:
        print("No successful trial with readable metric yet.")


if __name__ == "__main__":
    main()
