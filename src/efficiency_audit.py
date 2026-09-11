"""Small, dependency-light helpers for revision efficiency reporting."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def latency_summary(samples_ms: Sequence[float], warmup_count: int = 0) -> dict:
    values = np.asarray(samples_ms, dtype=float)
    if warmup_count < 0 or warmup_count >= len(values):
        raise ValueError("warmup_count must leave at least one measured sample")
    measured = values[int(warmup_count) :]
    if not np.all(np.isfinite(measured)) or np.any(measured < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "measurement_count": int(len(measured)),
        "mean_ms": float(np.mean(measured)),
        "median_ms": float(np.median(measured)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
    }


def epoch_time_summary(epoch_seconds: Sequence[float]) -> dict:
    values = np.asarray(epoch_seconds, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("epoch times must be non-empty, finite, and non-negative")
    return {
        "epoch_count": int(len(values)),
        "mean_epoch_sec": float(np.mean(values)),
        "median_epoch_sec": float(np.median(values)),
        "total_train_sec": float(np.sum(values)),
    }


def parameter_counts(parameters: Iterable[Tuple[int, bool]]) -> dict:
    rows = [(int(numel), bool(trainable)) for numel, trainable in parameters]
    if any(numel < 0 for numel, _ in rows):
        raise ValueError("parameter sizes must be non-negative")
    return {
        "total_parameters": int(sum(numel for numel, _ in rows)),
        "trainable_parameters": int(sum(numel for numel, trainable in rows if trainable)),
    }
