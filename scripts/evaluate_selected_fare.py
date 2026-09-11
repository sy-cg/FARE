"""One-shot test evaluation of a validation-selected frozen FARE checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.revision_artifacts import file_sha256
from run_fare import (
    FULL_GROUPS,
    build_fare_model,
    load_training_group_labels,
    parse_list_arg,
    run_saved_topk_fairness_eval,
)
from run_id_backbone import (
    EvalNextItemDataset,
    build_seen_items,
    evaluate_full_sort,
    infer_num_items,
    make_collate_fn,
    read_user_sequences,
    safe_torch_load,
    save_json,
    unwrap_state_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config_resolved.json"
    checkpoint_path = run_dir / "best_model.pt"
    audit_path = run_dir / "test_evaluation_audit.json"
    test_topk_path = run_dir / "topk_test.npz"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Selected run requires config_resolved.json and best_model.pt")
    if not args.force and (audit_path.exists() or test_topk_path.exists()):
        existing = audit_path if audit_path.exists() else test_topk_path
        raise FileExistsError(f"One-shot test artifact already exists: {existing}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    eval_cfg = cfg.get("eval", {})
    if str(eval_cfg.get("split_for_best", "")).lower() != "val":
        raise ValueError("Selected run is invalid: eval.split_for_best must be val")

    dataset = str(cfg["dataset"])
    data_dir = Path(str(cfg["data_dir"]))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    num_items = infer_num_items(data_dir)
    model_cfg = cfg.get("model", {})
    fare_cfg = cfg.get("fare", {})
    requested_groups = fare_cfg.get("groups") or FULL_GROUPS
    if isinstance(requested_groups, str):
        requested_groups = parse_list_arg(requested_groups) or FULL_GROUPS
    group_labels, group_num_classes, _ = load_training_group_labels(
        data_dir=data_dir,
        requested_groups=list(requested_groups),
        num_items=num_items,
        min_group_count=int(fare_cfg.get("min_group_count", 5)),
        drop_rare_classes=bool(fare_cfg.get("drop_rare_classes", True)),
    )

    device = torch.device(
        "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = build_fare_model(
        num_items=num_items,
        data_dir=data_dir,
        group_num_classes=group_num_classes,
        cfg=cfg,
    ).to(device)
    checkpoint_hash_before = file_sha256(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    model.eval()

    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    batch_size = int(eval_cfg.get("batch_size", 512))
    num_workers = int(cfg.get("train", {}).get("num_workers", 4))
    pin_memory = device.type == "cuda" and bool(cfg.get("train", {}).get("pin_memory", True))
    train_rows = read_user_sequences(data_dir / "train.txt")
    val_rows = read_user_sequences(data_dir / "val.txt")
    test_rows = read_user_sequences(data_dir / "test.txt")
    val_ds = EvalNextItemDataset(val_rows, max_seq_len=max_seq_len)
    test_ds = EvalNextItemDataset(test_rows, max_seq_len=max_seq_len)
    collate = make_collate_fn(max_seq_len)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    seen_train = build_seen_items(train_rows)
    ks = [int(value) for value in eval_cfg.get("ks", [5, 10, 20, 100])]
    mask_seen = bool(eval_cfg.get("mask_seen_items", True))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    val_start = time.perf_counter()
    val_metrics = evaluate_full_sort(
        model,
        val_loader,
        device,
        ks,
        num_items,
        mask_seen_items=mask_seen,
        seen_items=seen_train,
        save_topk_path=run_dir / "topk_val.npz",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_sec = time.perf_counter() - val_start

    test_start = time.perf_counter()
    test_metrics = evaluate_full_sort(
        model,
        test_loader,
        device,
        ks,
        num_items,
        mask_seen_items=mask_seen,
        seen_items=seen_train,
        save_topk_path=test_topk_path,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_sec = time.perf_counter() - test_start

    summary_path = run_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["val"] = val_metrics
    summary["test"] = test_metrics
    efficiency = summary.setdefault("efficiency", {})
    efficiency.update(
        {
            "val_inference_sec": float(val_sec),
            "val_user_count": len(val_ds),
            "val_ms_per_user": float(1000.0 * val_sec / max(len(val_ds), 1)),
            "test_inference_sec": float(test_sec),
            "test_user_count": len(test_ds),
            "test_ms_per_user": float(1000.0 * test_sec / max(len(test_ds), 1)),
            "test_users_per_sec": float(len(test_ds) / max(test_sec, 1.0e-12)),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }
    )
    if bool(eval_cfg.get("run_fairness_eval", True)):
        summary["fairness_eval"] = run_saved_topk_fairness_eval(
            data_dir=data_dir,
            run_dir=run_dir,
            dataset=dataset,
            run_name=str(cfg.get("run_name", "fare")),
            run_id=str(cfg.get("run_id", run_dir.name)),
            splits=["val", "test"],
            ks=ks,
            group_spec=eval_cfg.get("fairness_groups", "all"),
        )
    save_json(summary, summary_path)

    checkpoint_hash_after = file_sha256(checkpoint_path)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("Frozen selected checkpoint changed during evaluation")
    audit = {
        "status": "ok",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "val",
        "test_evaluations": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_after,
        "config_sha256": file_sha256(config_path),
        "topk_test": str(test_topk_path),
        "test_user_count": len(test_ds),
    }
    save_json(audit, audit_path)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
