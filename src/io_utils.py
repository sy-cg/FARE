# -*- coding: utf-8 -*-
"""I/O helpers shared by training and evaluation scripts."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def safe_torch_load(path: str | Path, map_location: Any = None) -> Any:
    """Load PyTorch files with weights-only deserialization when available.

    PyTorch's historical ``torch.load`` default can execute pickle payloads.
    Project checkpoints and graph caches only need tensors plus primitive
    containers, so weights-only loading is the right default.
    """
    kwargs = {"map_location": map_location}

    try:
        sig = inspect.signature(torch.load)
    except (TypeError, ValueError):
        sig = None

    if sig is not None and "weights_only" in sig.parameters:
        kwargs["weights_only"] = True

    try:
        return torch.load(path, **kwargs)
    except TypeError as exc:
        if "weights_only" in kwargs and "weights_only" in str(exc):
            kwargs.pop("weights_only", None)
            return torch.load(path, **kwargs)
        raise
    except Exception as exc:
        if kwargs.get("weights_only"):
            raise RuntimeError(
                f"Failed to safely load PyTorch file with weights_only=True: {path}. "
                "Regenerate the checkpoint/cache with this project if it was produced "
                "by an older or external tool."
            ) from exc
        raise
