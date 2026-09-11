"""Exposure-prior utilities for FARE revision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ExposureWeightResult:
    class_weights: np.ndarray
    catalog_counts: np.ndarray
    exposure_counts: np.ndarray
    target_distribution: np.ndarray
    exposure_distribution: np.ndarray


def resolve_exposure_item_prior(
    data_dir: str | Path,
    source: str,
    explicit_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Load an item-level prior for a named independent exposure source."""
    data_dir = Path(data_dir)
    source = str(source).strip().lower()

    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = data_dir / path
    elif source == "train_popularity":
        path = data_dir / "item_popularity_train.npy"
    elif source == "platform_views":
        path = data_dir / "platform_views.npy"
    else:
        raise ValueError(
            f"Exposure source {source!r} requires an explicit item-prior path."
        )

    if not path.exists():
        raise FileNotFoundError(f"Exposure item-prior file not found: {path}")

    prior = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(prior, 0.0, None)


def build_exposure_class_weights(
    labels: np.ndarray,
    gamma: float,
    target_type: str = "catalog",
    source: str = "reference_topk",
    topk_items: Optional[np.ndarray] = None,
    item_prior: Optional[np.ndarray] = None,
    exposure_k: int = 10,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> ExposureWeightResult:
    """Build group-class weights from a model or independent exposure source."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size <= 1:
        raise ValueError("labels must contain padding plus at least one item")

    catalog_labels = labels[1:]
    catalog_labels = catalog_labels[catalog_labels >= 0]
    if catalog_labels.size == 0:
        raise ValueError("labels contain no valid non-padding item groups")

    num_classes = int(catalog_labels.max()) + 1
    catalog_counts = np.bincount(catalog_labels, minlength=num_classes).astype(np.float64)
    valid_classes = catalog_counts > 0
    if int(valid_classes.sum()) <= 1:
        raise ValueError("at least two non-empty exposure groups are required")

    source = str(source).strip().lower()
    if source == "reference_topk":
        if topk_items is None:
            raise ValueError("topk_items is required for source='reference_topk'")
        topk = np.asarray(topk_items, dtype=np.int64)
        if topk.ndim != 2:
            raise ValueError(f"topk_items must be 2-D, got shape={topk.shape}")
        if int(exposure_k) > 0:
            topk = topk[:, : min(int(exposure_k), topk.shape[1])]
        item_ids = topk.reshape(-1)
        item_ids = item_ids[(item_ids > 0) & (item_ids < labels.size)]
        exposed_labels = labels[item_ids]
        exposed_labels = exposed_labels[exposed_labels >= 0]
        exposure_counts = np.bincount(
            exposed_labels,
            minlength=num_classes,
        ).astype(np.float64)
    else:
        if item_prior is None:
            raise ValueError(f"item_prior is required for exposure source {source!r}")
        prior = np.asarray(item_prior, dtype=np.float64).reshape(-1)
        if prior.size != labels.size:
            raise ValueError(
                "item_prior and labels must have the same length, "
                f"got {prior.size} and {labels.size}"
            )
        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        prior = np.clip(prior, 0.0, None)
        exposure_counts = np.zeros(num_classes, dtype=np.float64)
        for class_id in range(num_classes):
            exposure_counts[class_id] = float(prior[labels == class_id].sum())

    if float(exposure_counts.sum()) <= 0:
        raise ValueError("exposure source contains no positive mass on valid items")

    target_type = str(target_type).strip().lower()
    if target_type == "catalog":
        target = catalog_counts / catalog_counts.sum()
    elif target_type == "uniform":
        target = np.zeros(num_classes, dtype=np.float64)
        target[valid_classes] = 1.0 / float(valid_classes.sum())
    else:
        raise ValueError(f"Unsupported target_type={target_type!r}")

    exposure = exposure_counts / exposure_counts.sum()
    imbalance = np.zeros(num_classes, dtype=np.float64)
    imbalance[valid_classes] = (
        target[valid_classes] - exposure[valid_classes]
    ) / np.maximum(target[valid_classes], 1e-8)

    low = float(min(min_weight, max_weight))
    high = float(max(min_weight, max_weight))
    class_weights = np.ones(num_classes, dtype=np.float64)
    class_weights[valid_classes] = 1.0 + float(gamma) * imbalance[valid_classes]
    class_weights = np.clip(class_weights, low, high)
    class_weights[valid_classes] /= max(float(class_weights[valid_classes].mean()), 1e-8)
    class_weights = np.nan_to_num(class_weights, nan=1.0, posinf=high, neginf=low)

    return ExposureWeightResult(
        class_weights=class_weights.astype(np.float32),
        catalog_counts=catalog_counts,
        exposure_counts=exposure_counts,
        target_distribution=target,
        exposure_distribution=exposure,
    )

