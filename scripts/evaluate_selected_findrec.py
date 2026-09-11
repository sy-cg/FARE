#!/usr/bin/env python3
"""One-shot test evaluation for a validation-selected FindRec checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.revision_artifacts import file_sha256
from run_findrec import FindRec, evaluate_split, load_checkpoint, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config_resolved.json"
    checkpoint_path = run_dir / "best_model.pt"
    summary_path = run_dir / "metrics_summary.json"
    test_path = run_dir / "test_ranking.npz"
    audit_path = run_dir / "test_evaluation_audit.json"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Selected FindRec run requires config_resolved.json and best_model.pt")
    if not args.force and (test_path.exists() or audit_path.exists()):
        existing = test_path if test_path.exists() else audit_path
        raise FileExistsError(f"One-shot test artifact already exists: {existing}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(cfg.get("eval", {}).get("run_test_after_training", True)):
        raise ValueError(
            "Selected FindRec tuning run is invalid: run_test_after_training must be false"
        )
    dataset = str(cfg.get("dataset_name", cfg.get("dataset", "")))
    if not dataset:
        raise ValueError("Selected FindRec config has no dataset identity")
    data_dir = Path(str(cfg["data_dir"]))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    num_items = int(cfg["num_items"])
    device = torch.device(
        "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.setdefault("eval", {})
    eval_cfg["eval"]["run_fairness_eval"] = True
    eval_cfg["eval"]["save_topk_npz"] = True
    model = FindRec(num_items=num_items, data_dir=data_dir, config=eval_cfg).to(device)
    checkpoint_hash_before = file_sha256(checkpoint_path)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    val_start = time.perf_counter()
    val_out = evaluate_split(
        model, data_dir, "val", eval_cfg, num_items, device, run_dir, save_outputs=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_sec = time.perf_counter() - val_start

    test_start = time.perf_counter()
    test_out = evaluate_split(
        model, data_dir, "test", eval_cfg, num_items, device, run_dir, save_outputs=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_sec = time.perf_counter() - test_start
    with np.load(test_path, allow_pickle=False) as archive:
        user_key = "users" if "users" in archive.files else "user_ids"
        test_user_count = int(np.asarray(archive[user_key]).size)

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["val_final"] = val_out["ranking_metrics"]
    summary["test_final"] = test_out["ranking_metrics"]
    summary["efficiency"] = {
        "total_parameters": int(model.num_parameters),
        "trainable_parameters": int(model.num_trainable_parameters),
        "val_inference_sec": float(val_sec),
        "test_inference_sec": float(test_sec),
        "test_user_count": test_user_count,
        "test_ms_per_user": float(1000.0 * test_sec / max(test_user_count, 1)),
        "test_users_per_sec": float(test_user_count / max(test_sec, 1.0e-12)),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    save_json(summary, summary_path)

    checkpoint_hash_after = file_sha256(checkpoint_path)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("Frozen selected FindRec checkpoint changed during evaluation")
    audit = {
        "status": "ok",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "val",
        "test_evaluations": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_after,
        "config_sha256": file_sha256(config_path),
        "test_ranking": str(test_path),
        "validation_seconds": float(val_sec),
        "test_seconds": float(test_sec),
        "test_user_count": test_user_count,
    }
    save_json(audit, audit_path)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
