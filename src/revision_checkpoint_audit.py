"""Torch-independent checkpoint identity checks for composed revision methods."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fare_checkpoint_source(
    checkpoint: str | Path,
    *,
    expected_dataset: str,
    expected_backbone: str,
    expected_num_items: int,
    expected_model: dict[str, Any] | None = None,
    expected_fare: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that a checkpoint is a FARE run for the requested experiment."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"FARE checkpoint not found: {checkpoint}")
    config_path = checkpoint.parent / "config_resolved.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"FARE checkpoint audit requires sibling config_resolved.json: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = str(config.get("method", config.get("method_name", ""))).strip()
    dataset = str(config.get("dataset", ""))
    backbone = str(
        config.get("backbone", config.get("model", {}).get("backbone", ""))
    ).lower()
    num_items = int(config.get("num_items", -1))
    errors = []
    if method != "FARE":
        errors.append(f"method={method!r}, expected 'FARE'")
    if dataset != str(expected_dataset):
        errors.append(f"dataset={dataset!r}, expected {expected_dataset!r}")
    if backbone != str(expected_backbone).lower():
        errors.append(f"backbone={backbone!r}, expected {str(expected_backbone).lower()!r}")
    if num_items != int(expected_num_items):
        errors.append(f"num_items={num_items}, expected {int(expected_num_items)}")

    architecture = {}
    for section_name, expected in (("model", expected_model), ("fare", expected_fare)):
        source = config.get(section_name, {})
        for key, expected_value in (expected or {}).items():
            if key not in source:
                continue
            source_value = source[key]
            architecture[f"{section_name}.{key}"] = source_value
            if source_value != expected_value:
                errors.append(
                    f"{section_name}.{key}={source_value!r}, expected {expected_value!r}"
                )
    if errors:
        raise ValueError("FARE checkpoint identity mismatch: " + "; ".join(errors))
    return {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "source_config": str(config_path),
        "dataset": dataset,
        "method": method,
        "backbone": backbone,
        "num_items": num_items,
        "architecture": architecture,
    }
