#!/usr/bin/env python3
"""Re-evaluate an existing ID-backbone checkpoint and refresh saved Top-K files.

This script does not train. It loads ``best_model.pt`` from an existing
SASRec/GRU4Rec/BERT4Rec ID run directory, evaluates validation/test splits, and
writes ``topk_val.npz`` / ``topk_test.npz`` with internal item ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from src.revision_artifacts import file_sha256


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_run_config(run_dir: Path, checkpoint_obj: Any) -> Dict[str, Any]:
    config_path = run_dir / "config_resolved.json"
    if config_path.exists():
        cfg = _read_json(config_path)
        if not isinstance(cfg, dict):
            raise ValueError(f"config_resolved.json root must be a mapping: {config_path}")
        return cfg
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("config"), dict):
        return dict(checkpoint_obj["config"])
    raise FileNotFoundError(f"Missing config_resolved.json and checkpoint config in: {run_dir}")


def _canonical_backbone(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text.endswith("_id"):
            text = text[: -len("_id")]
        if text in {"sasrec", "gru4rec", "bert4rec"}:
            return text
        if text == "gru":
            return "gru4rec"
        if text == "bert":
            return "bert4rec"
        if text == "sas":
            return "sasrec"
    raise ValueError(f"Could not infer ID backbone from values: {values!r}")


def _metric_ks_with_topk_width(metric_ks: Iterable[int], topk_width: int) -> List[int]:
    values = {int(k) for k in metric_ks if int(k) > 0}
    values.add(int(topk_width))
    return sorted(values)


def _split_names(split: str) -> List[str]:
    split = str(split).strip().lower()
    if split == "all":
        return ["val", "test"]
    if split in {"val", "test"}:
        return [split]
    raise ValueError(f"Unsupported split={split!r}; expected val/test/all")


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _resolve_data_dir(cfg: Dict[str, Any], dataset: str) -> Path:
    from run_id_backbone import get_dataset_dir, load_yaml  # type: ignore

    if cfg.get("data_dir"):
        saved = _resolve_path(str(cfg["data_dir"]))
        if saved.exists():
            return saved

    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    datasets_config = _resolve_path(str(paths_cfg.get("datasets_config", "configs/datasets.yaml")))
    datasets_cfg: Dict[str, Any] = load_yaml(datasets_config) if datasets_config.exists() else {}
    fallback = get_dataset_dir(dataset, datasets_cfg)

    if cfg.get("data_dir"):
        saved = _resolve_path(str(cfg["data_dir"]))
        if not saved.exists() and fallback.exists():
            print(
                f"[warning] saved data_dir does not exist: {saved}; using {fallback}",
                flush=True,
            )
            return fallback
        return saved

    return fallback


def _dataset_from_context(cfg: Dict[str, Any], summary: Dict[str, Any], run_dir: Path) -> str:
    for value in (cfg.get("dataset"), summary.get("dataset")):
        if value:
            return str(value)
    if run_dir.parent.parent.name:
        return run_dir.parent.parent.name
    raise ValueError("Could not infer dataset from config, metrics_summary.json, or run path")


def _load_existing_summary(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return {}
    obj = _read_json(metrics_path)
    if not isinstance(obj, dict):
        raise ValueError(f"metrics_summary.json root must be a mapping: {metrics_path}")
    return obj


def validate_topk_archive(path: str | Path, num_items: int) -> Dict[str, Any]:
    """Validate that a saved top-k archive uses the internal item-id space."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "topk_items" not in archive.files:
            raise KeyError(f"topk archive missing key 'topk_items': {path}")
        raw = archive["topk_items"]
        if raw.ndim != 2:
            raise ValueError(f"topk_items must be 2-D in {path}, got shape={raw.shape}")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"topk_items must be an integer array in {path}, got dtype={raw.dtype}")
        items = np.asarray(raw, dtype=np.int64)

    invalid = items[(items < 1) | (items > int(num_items))]
    if invalid.size:
        examples = invalid[:10].astype(np.int64).tolist()
        raise ValueError(
            f"{path} contains {int(invalid.size)} out-of-range item ids for valid "
            f"internal item range 1..{int(num_items)}; examples={examples}. "
            "Regenerate this archive from the ID checkpoint before running FARE/FairRR."
        )

    return {
        "path": str(path),
        "shape": [int(items.shape[0]), int(items.shape[1])],
        "min_item_id": int(items.min()) if items.size else None,
        "max_item_id": int(items.max()) if items.size else None,
        "out_of_range_count": 0,
    }


def _build_eval_loader(
    data_dir: Path,
    split: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[Any, int]:
    from torch.utils.data import DataLoader
    from run_id_backbone import EvalNextItemDataset, make_collate_fn, read_user_sequences  # type: ignore

    rows = read_user_sequences(data_dir / f"{split}.txt")
    dataset = EvalNextItemDataset(rows, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=make_collate_fn(max_seq_len),
        drop_last=False,
    )
    return loader, len(dataset)


def evaluate_checkpoint(
    run_dir: Path,
    split_names: Sequence[str],
    device_arg: Optional[str],
    eval_batch_size_arg: Optional[int],
    num_workers_arg: Optional[int],
    topk_width: int,
    force: bool,
) -> Dict[str, Any]:
    import torch

    from run_id_backbone import (  # type: ignore
        build_model,
        build_seen_items,
        evaluate_full_sort,
        infer_num_items,
        read_user_sequences,
        safe_torch_load,
        set_seed,
        unwrap_state_dict,
    )
    from src.efficiency_audit import parameter_counts

    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt: {checkpoint_path}")

    for split in split_names:
        output_path = run_dir / f"topk_{split}.npz"
        if output_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing {output_path}; pass --force")

    checkpoint_obj = safe_torch_load(checkpoint_path, map_location="cpu")
    cfg = _load_run_config(run_dir, checkpoint_obj)
    summary = _load_existing_summary(run_dir)

    dataset = _dataset_from_context(cfg, summary, run_dir)
    backbone = _canonical_backbone(
        cfg.get("model_arg"),
        cfg.get("backbone"),
        summary.get("backbone"),
        run_dir.parent.name,
    )
    seed = int(cfg.get("seed", summary.get("seed", 2026)))
    set_seed(seed)

    data_dir = _resolve_data_dir(cfg, dataset)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {data_dir}")

    num_items = int(cfg.get("num_items") or summary.get("num_items") or infer_num_items(data_dir))
    if int(topk_width) > num_items:
        raise ValueError(f"--topk-width {topk_width} exceeds num_items={num_items}")

    requested_device = str(device_arg or cfg.get("device_resolved") or cfg.get("device") or "cuda")
    device = torch.device(
        requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    max_seq_len = int(model_cfg.get("max_seq_len", 50))
    eval_batch_size = int(eval_batch_size_arg or eval_cfg.get("batch_size", 512))
    num_workers = int(num_workers_arg if num_workers_arg is not None else train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    metric_ks = eval_cfg.get("ks", [5, 10, 20])
    ks = _metric_ks_with_topk_width(metric_ks, int(topk_width))
    mask_seen_items = bool(eval_cfg.get("mask_seen_items", True))

    train_rows = read_user_sequences(data_dir / "train.txt")
    seen_train = build_seen_items(train_rows)
    model = build_model(backbone, num_items=num_items, cfg=cfg).to(device)
    model.load_state_dict(unwrap_state_dict(checkpoint_obj), strict=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    split_audits: Dict[str, Any] = {}
    inference_times: Dict[str, float] = {}
    user_counts: Dict[str, int] = {}
    split_metrics: Dict[str, Dict[str, float]] = {}

    for split in split_names:
        loader, user_count = _build_eval_loader(
            data_dir=data_dir,
            split=split,
            max_seq_len=max_seq_len,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        output_path = run_dir / f"topk_{split}.npz"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        metrics = evaluate_full_sort(
            model=model,
            loader=loader,
            device=device,
            ks=ks,
            num_items=num_items,
            mask_seen_items=mask_seen_items,
            seen_items=seen_train,
            save_topk_path=output_path,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        split_metrics[split] = metrics
        inference_times[split] = float(elapsed)
        user_counts[split] = int(user_count)
        split_audits[split] = validate_topk_archive(output_path, num_items=num_items)
        print(f"[{split}] wrote {output_path} elapsed={elapsed:.2f}s", flush=True)

    counts = parameter_counts((p.numel(), p.requires_grad) for p in model.parameters())
    efficiency = summary.get("efficiency", {})
    if not isinstance(efficiency, dict):
        efficiency = {}
    efficiency.update(counts)
    efficiency.update(
        {
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }
    )
    for split, elapsed in inference_times.items():
        count = user_counts[split]
        efficiency[f"{split}_inference_sec"] = float(elapsed)
        efficiency[f"{split}_user_count"] = int(count)
        efficiency[f"{split}_ms_per_user"] = float(1000.0 * elapsed / max(count, 1))
        if split == "test":
            efficiency["test_inference_sec"] = float(elapsed)
            efficiency["test_user_count"] = int(count)
            efficiency["test_ms_per_user"] = float(1000.0 * elapsed / max(count, 1))
            efficiency["test_users_per_sec"] = float(count / max(elapsed, 1.0e-12))

    summary.update(
        {
            "dataset": dataset,
            "model_name": f"{backbone}_id",
            "run_id": run_dir.name,
            "backbone": backbone,
            "num_items": int(num_items),
            "efficiency": efficiency,
        }
    )
    for split, metrics in split_metrics.items():
        summary[split] = metrics
        summary[f"num_{split}_samples"] = int(user_counts[split])

    audit = {
        "script": "scripts/evaluate_id_checkpoint.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "dataset": dataset,
        "backbone": backbone,
        "seed": int(seed),
        "data_dir": str(data_dir),
        "device": str(device),
        "splits": list(split_names),
        "ks": ks,
        "topk_width": int(topk_width),
        "force": bool(force),
        "topk_audit": split_audits,
    }
    summary["id_checkpoint_evaluation_audit"] = audit
    _write_json(run_dir / "metrics_summary.json", summary)
    _write_json(run_dir / "id_checkpoint_evaluation_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing ID run directory containing best_model.pt")
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--topk-width", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Overwrite existing topk_<split>.npz files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = evaluate_checkpoint(
        run_dir=Path(args.run_dir),
        split_names=_split_names(args.split),
        device_arg=args.device,
        eval_batch_size_arg=args.eval_batch_size,
        num_workers_arg=args.num_workers,
        topk_width=int(args.topk_width),
        force=bool(args.force),
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()