"""Backfill missing efficiency fields for old ID-backbone revision runs.

This script does not retrain. It loads each selected ID run's best checkpoint,
re-runs test full-sort evaluation once, and writes the missing efficiency
metadata into that run's metrics_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.benchmark_revision_efficiency import REQUIRED_EFFICIENCY, REQUIRED_MANIFEST


@dataclass(frozen=True)
class BackfillCandidate:
    dataset: str
    backbone: str
    seed: str
    run_dir: str
    missing_fields: List[str]


@dataclass(frozen=True)
class SkippedRun:
    run_dir: str
    reason: str


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_rows(manifest_path: str | Path) -> List[Dict]:
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return manifest.to_dict("records")


def select_backfill_runs(manifest_path: str | Path) -> Tuple[List[BackfillCandidate], List[SkippedRun]]:
    """Select only ID manifest rows whose metrics_summary lacks required efficiency fields."""
    selected: List[BackfillCandidate] = []
    skipped: List[SkippedRun] = []

    for row in _manifest_rows(manifest_path):
        run_dir = Path(str(row["run_dir"]))
        if str(row["method"]).strip().lower() != "id":
            skipped.append(SkippedRun(run_dir=str(run_dir), reason="method is not ID"))
            continue

        metrics_path = run_dir / "metrics_summary.json"
        if not metrics_path.exists():
            skipped.append(SkippedRun(run_dir=str(run_dir), reason="missing metrics_summary.json"))
            continue

        metrics = _read_json(metrics_path)
        efficiency = metrics.get("efficiency", {})
        if not isinstance(efficiency, dict):
            efficiency = {}
        missing_fields = sorted(REQUIRED_EFFICIENCY - set(efficiency))
        if not missing_fields:
            skipped.append(SkippedRun(run_dir=str(run_dir), reason="already complete"))
            continue

        selected.append(
            BackfillCandidate(
                dataset=str(row["dataset"]),
                backbone=str(row.get("backbone", "")),
                seed=str(row["seed"]),
                run_dir=str(run_dir),
                missing_fields=missing_fields,
            )
        )

    return selected, skipped


def _canonical_backbone(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text.endswith("_id"):
            text = text[: -len("_id")]
        if text in {"sasrec", "gru4rec", "bert4rec"}:
            return text
    raise ValueError(f"Could not infer ID backbone from values: {values!r}")


def _load_config_from_run(run_dir: Path, checkpoint_obj: Optional[Dict]) -> Dict:
    config_path = run_dir / "config_resolved.json"
    if config_path.exists():
        cfg = _read_json(config_path)
        if isinstance(cfg, dict):
            return cfg
        raise ValueError(f"config_resolved.json root must be a mapping: {config_path}")
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("config"), dict):
        return dict(checkpoint_obj["config"])
    raise FileNotFoundError(f"Missing config_resolved.json and checkpoint config in: {run_dir}")


def _resolve_data_dir(cfg: Dict, dataset: str):
    from run_id_backbone import get_dataset_dir, load_yaml, resolve_path

    paths_cfg = cfg.get("paths", {})
    datasets_config = resolve_path(paths_cfg.get("datasets_config", "configs/datasets.yaml"))
    datasets_cfg = load_yaml(datasets_config) if datasets_config.exists() else {}

    if cfg.get("data_dir"):
        saved_data_dir = resolve_path(str(cfg["data_dir"]))
        if saved_data_dir and saved_data_dir.exists():
            return saved_data_dir
        fallback_dir = get_dataset_dir(dataset, datasets_cfg)
        if fallback_dir.exists():
            print(
                f"[warning] saved data_dir does not exist: {saved_data_dir}; "
                f"using current dataset directory: {fallback_dir}",
                flush=True,
            )
            return fallback_dir
        return saved_data_dir

    return get_dataset_dir(dataset, datasets_cfg)


def measure_id_efficiency(
    candidate: BackfillCandidate,
    device_arg: Optional[str] = None,
    eval_batch_size_arg: Optional[int] = None,
    num_workers_arg: Optional[int] = None,
) -> Dict[str, object]:
    """Load an ID checkpoint and measure comparable test inference efficiency."""
    import torch
    from torch.utils.data import DataLoader

    from run_id_backbone import (
        EvalNextItemDataset,
        build_model,
        build_seen_items,
        evaluate_full_sort,
        infer_num_items,
        make_collate_fn,
        read_user_sequences,
        safe_torch_load,
        set_seed,
        unwrap_state_dict,
    )
    from src.efficiency_audit import parameter_counts

    run_dir = Path(candidate.run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt: {checkpoint_path}")

    checkpoint_obj = safe_torch_load(checkpoint_path, map_location="cpu")
    cfg = _load_config_from_run(run_dir, checkpoint_obj if isinstance(checkpoint_obj, dict) else None)
    metrics_path = run_dir / "metrics_summary.json"
    metrics_summary = _read_json(metrics_path)
    for key in ["dataset", "data_dir", "num_items", "model_arg", "backbone"]:
        if key not in cfg and key in metrics_summary:
            cfg[key] = metrics_summary[key]

    dataset = str(cfg.get("dataset") or candidate.dataset)
    backbone = _canonical_backbone(
        cfg.get("model_arg"),
        cfg.get("backbone"),
        candidate.backbone,
        run_dir.parent.name,
    )

    seed = int(cfg.get("seed", candidate.seed))
    set_seed(seed)

    requested_device = str(device_arg or cfg.get("device_resolved") or cfg.get("device") or "cuda")
    device = torch.device(
        requested_device
        if requested_device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    eval_batch_size = int(eval_batch_size_arg or eval_cfg.get("batch_size", 512))
    num_workers = int(num_workers_arg if num_workers_arg is not None else train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    ks = [int(x) for x in eval_cfg.get("ks", [5, 10, 20])]
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))

    data_dir = _resolve_data_dir(cfg, dataset)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {data_dir}")

    train_rows = read_user_sequences(data_dir / "train.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")
    num_items = int(cfg.get("num_items") or infer_num_items(data_dir))
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=make_collate_fn(max_seq_len),
        drop_last=False,
    )

    model = build_model(backbone, num_items=num_items, cfg=cfg).to(device)
    state = unwrap_state_dict(checkpoint_obj)
    model.load_state_dict(state, strict=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    evaluate_full_sort(
        model=model,
        loader=test_loader,
        device=device,
        ks=ks,
        num_items=num_items,
        mask_seen_items=mask_seen_items,
        seen_items=build_seen_items(train_rows),
        save_topk_path=None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_inference_sec = time.perf_counter() - start

    counts = parameter_counts((p.numel(), p.requires_grad) for p in model.parameters())
    return {
        **counts,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
        "test_inference_sec": float(test_inference_sec),
        "test_user_count": int(len(test_ds)),
        "test_ms_per_user": float(1000.0 * test_inference_sec / max(len(test_ds), 1)),
        "test_users_per_sec": float(len(test_ds) / max(test_inference_sec, 1.0e-12)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def backfill_metrics_summary(candidate: BackfillCandidate, efficiency: Dict[str, object]) -> List[str]:
    run_dir = Path(candidate.run_dir)
    metrics_path = run_dir / "metrics_summary.json"
    summary = _read_json(metrics_path)
    current = summary.get("efficiency", {})
    if not isinstance(current, dict):
        current = {}
    current.update(efficiency)
    current["backfilled_efficiency"] = True
    summary["efficiency"] = current
    _write_json(metrics_path, summary)
    return sorted(REQUIRED_EFFICIENCY & set(efficiency))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="revision efficiency_manifest.csv")
    parser.add_argument("--audit", required=True, help="JSON audit path for this backfill run")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only report selected runs; do not load checkpoints")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _write_audit(path: str | Path, audit: Dict) -> None:
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(audit_path, audit)


def main() -> None:
    args = parse_args()
    selected, skipped = select_backfill_runs(args.manifest)
    audit: Dict[str, object] = {
        "manifest_runs": int(len(selected) + len(skipped)),
        "selected_runs": int(len(selected)),
        "skipped_runs": int(len(skipped)),
        "backfilled_runs": 0,
        "failed_runs": 0,
        "selected": [asdict(row) for row in selected],
        "skipped": [asdict(row) for row in skipped],
        "backfilled": [],
        "failed": [],
    }

    if args.dry_run:
        _write_audit(args.audit, audit)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    backfilled: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    for candidate in selected:
        try:
            efficiency = measure_id_efficiency(
                candidate,
                device_arg=args.device,
                eval_batch_size_arg=args.eval_batch_size,
                num_workers_arg=args.num_workers,
            )
            filled_fields = backfill_metrics_summary(candidate, efficiency)
            backfilled.append(
                {
                    "run_dir": candidate.run_dir,
                    "filled_fields": filled_fields,
                    "previously_missing_fields": candidate.missing_fields,
                }
            )
            print(f"[backfilled] {candidate.run_dir}", flush=True)
        except Exception as exc:
            failed.append({"run_dir": candidate.run_dir, "reason": str(exc)})
            print(f"[failed] {candidate.run_dir}: {exc}", flush=True)
            if not args.continue_on_error:
                audit["backfilled_runs"] = int(len(backfilled))
                audit["failed_runs"] = int(len(failed))
                audit["backfilled"] = backfilled
                audit["failed"] = failed
                _write_audit(args.audit, audit)
                raise

    audit["backfilled_runs"] = int(len(backfilled))
    audit["failed_runs"] = int(len(failed))
    audit["backfilled"] = backfilled
    audit["failed"] = failed
    _write_audit(args.audit, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
