# -*- coding: utf-8 -*-
"""
src/losses.py

Common loss utilities for SDR experiments.

This file includes a popularity-weighted cross entropy loss used by the
PopReweight fairness baseline:

    loss_i = CE(logits_i, target_i) * w[target_i]

where w[target_item] is usually inverse to item popularity estimated from the
training split only.

Design principles:
- Padding item id is 0 and its weight is always 0 by default.
- Popularity weights are finite, clipped, and optionally normalized.
- The weighted CE can normalize batch weights to avoid changing the effective
  learning rate too much.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ==========================================================
# Standard losses
# ==========================================================


def full_sort_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard full-sort cross entropy loss.

    Parameters
    ----------
    logits:
        Tensor with shape [batch_size, num_items + 1].
    targets:
        Tensor with shape [batch_size]. Target item ids.
    label_smoothing:
        Passed to torch.nn.functional.cross_entropy.
    reduction:
        "none", "mean", or "sum".
    """
    return F.cross_entropy(
        logits,
        targets.long(),
        reduction=reduction,
        label_smoothing=float(label_smoothing),
    )


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    loss = -F.logsigmoid(pos_scores - neg_scores)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


# ==========================================================
# Popularity weights
# ==========================================================


def sanitize_popularity_counts(pop_counts: np.ndarray | torch.Tensor, num_items: Optional[int] = None) -> torch.Tensor:
    """Convert item popularity counts into a clean float tensor.

    The expected shape is [num_items + 1], where index 0 is padding.
    """
    if isinstance(pop_counts, np.ndarray):
        counts = torch.from_numpy(pop_counts)
    else:
        counts = pop_counts.detach().cpu()

    counts = counts.to(dtype=torch.float32).flatten()
    counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    counts = counts.clamp(min=0.0)

    if num_items is not None:
        expected = int(num_items) + 1
        if counts.numel() < expected:
            padded = torch.zeros(expected, dtype=torch.float32)
            padded[: counts.numel()] = counts
            counts = padded
        elif counts.numel() > expected:
            counts = counts[:expected]

    if counts.numel() > 0:
        counts[0] = 0.0
    return counts


def build_popularity_weights(
    pop_counts: np.ndarray | torch.Tensor,
    num_items: Optional[int] = None,
    scheme: str = "inverse_log",
    alpha: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    normalize_to_mean: bool = True,
    zero_padding: bool = True,
) -> torch.Tensor:
    """Build item-level training weights from item popularity counts.

    Parameters
    ----------
    pop_counts:
        Item popularity counts from training data only. Shape [num_items + 1]
        is preferred. Index 0 is padding.
    num_items:
        Number of non-padding items. Used to pad/truncate pop_counts.
    scheme:
        Weighting scheme:
        - "inverse_log": 1 / log(2 + pop)
        - "inverse_sqrt": 1 / sqrt(1 + pop)
        - "inverse_power": (mean_pop / (pop + 1)) ** alpha
        - "group_inverse": inverse mean popularity per quantile-like group is
          not implemented here because it requires external group ids.
    alpha:
        Exponent used by "inverse_power".
    min_weight, max_weight:
        Clipping range after computing raw weights.
    normalize_to_mean:
        If True, normalize non-padding item weights so their mean is close to 1.
        This makes the weighted objective less sensitive to learning-rate shifts.
    zero_padding:
        If True, weight[0] = 0.

    Returns
    -------
    torch.Tensor
        Float tensor of shape [num_items + 1].
    """
    counts = sanitize_popularity_counts(pop_counts, num_items=num_items)
    if counts.numel() <= 1:
        raise ValueError("pop_counts must include at least one non-padding item")

    scheme = scheme.lower().strip()
    nonpad = counts[1:]
    positive = nonpad[nonpad > 0]
    mean_pop = positive.mean().clamp(min=1.0) if positive.numel() > 0 else torch.tensor(1.0)

    if scheme == "inverse_log":
        # Stable and mild: tail items get larger weights, but not explosively.
        weights = 1.0 / torch.log2(counts + 2.0)
    elif scheme == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(counts + 1.0)
    elif scheme == "inverse_power":
        weights = torch.pow(mean_pop / (counts + 1.0), float(alpha))
    else:
        raise ValueError(
            f"Unsupported popularity weight scheme: {scheme}. "
            "Choose inverse_log, inverse_sqrt, or inverse_power."
        )

    weights = torch.nan_to_num(weights, nan=1.0, posinf=float(max_weight), neginf=float(min_weight))
    weights = weights.clamp(min=float(min_weight), max=float(max_weight))

    if normalize_to_mean:
        valid = weights[1:]
        mean_w = valid.mean().clamp(min=1e-8)
        weights = weights / mean_w
        weights = weights.clamp(min=float(min_weight), max=float(max_weight))

    if zero_padding and weights.numel() > 0:
        weights[0] = 0.0

    return weights.to(dtype=torch.float32)


def load_item_popularity_counts(
    data_dir: str | Path,
    num_items: Optional[int] = None,
    filename: str = "item_popularity_train.npy",
) -> torch.Tensor:
    """Load item popularity counts from processed dataset directory."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Popularity file not found: {path}")
    arr = np.load(path)
    return sanitize_popularity_counts(arr, num_items=num_items)


def describe_popularity_weights(weights: torch.Tensor, pop_counts: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Return simple diagnostics for logging."""
    w = weights.detach().cpu().float().flatten()
    valid = w[1:] if w.numel() > 1 else w
    valid = valid[torch.isfinite(valid)]
    out: Dict[str, float] = {
        "weight_min": float(valid.min().item()) if valid.numel() else float("nan"),
        "weight_max": float(valid.max().item()) if valid.numel() else float("nan"),
        "weight_mean": float(valid.mean().item()) if valid.numel() else float("nan"),
        "weight_std": float(valid.std(unbiased=False).item()) if valid.numel() else float("nan"),
    }
    if pop_counts is not None:
        c = pop_counts.detach().cpu().float().flatten()
        c = c[1:] if c.numel() > 1 else c
        c = c[torch.isfinite(c)]
        out.update({
            "pop_min": float(c.min().item()) if c.numel() else float("nan"),
            "pop_max": float(c.max().item()) if c.numel() else float("nan"),
            "pop_mean": float(c.mean().item()) if c.numel() else float("nan"),
            "pop_median": float(c.median().item()) if c.numel() else float("nan"),
        })
    return out


# ==========================================================
# Popularity-weighted CE
# ==========================================================


def popularity_weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    item_weights: torch.Tensor,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    normalize_batch_weight: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Popularity-weighted full-sort cross entropy.

    Parameters
    ----------
    logits:
        [B, num_items + 1] full-sort logits.
    targets:
        [B] target item ids.
    item_weights:
        [num_items + 1] item-level weights. Usually built with
        build_popularity_weights().
    label_smoothing:
        Optional label smoothing for CE.
    reduction:
        "none", "mean", or "sum".
    normalize_batch_weight:
        If True, divide target weights by their batch mean so the average weight
        per batch is approximately 1.
    eps:
        Numerical stability constant.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, N], got {tuple(logits.shape)}")
    if targets.dim() != 1:
        targets = targets.view(-1)
    targets = targets.long()

    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )

    weights = item_weights.to(device=logits.device, dtype=logits.dtype)
    if weights.dim() != 1:
        weights = weights.view(-1)
    if weights.numel() < logits.size(1):
        raise ValueError(
            f"item_weights has length {weights.numel()}, but logits has {logits.size(1)} classes"
        )

    target_weights = weights.gather(0, targets.clamp(min=0, max=weights.numel() - 1))
    target_weights = torch.nan_to_num(target_weights, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=0.0)

    if normalize_batch_weight:
        denom = target_weights.mean().clamp(min=float(eps))
        target_weights = target_weights / denom

    loss = ce * target_weights

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def popularity_weighted_ce_loss_with_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    item_weights: torch.Tensor,
    label_smoothing: float = 0.0,
    normalize_batch_weight: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted CE plus lightweight diagnostics for training logs."""
    if targets.dim() != 1:
        targets = targets.view(-1)
    targets = targets.long()

    weights = item_weights.to(device=logits.device, dtype=logits.dtype)
    target_weights = weights.gather(0, targets.clamp(min=0, max=weights.numel() - 1))
    raw_mean = target_weights.mean().detach()

    loss = popularity_weighted_ce_loss(
        logits=logits,
        targets=targets,
        item_weights=item_weights,
        label_smoothing=label_smoothing,
        reduction="mean",
        normalize_batch_weight=normalize_batch_weight,
    )

    stats = {
        "target_weight_mean_raw": float(raw_mean.detach().cpu().item()),
        "target_weight_min_raw": float(target_weights.min().detach().cpu().item()),
        "target_weight_max_raw": float(target_weights.max().detach().cpu().item()),
    }
    return loss, stats
