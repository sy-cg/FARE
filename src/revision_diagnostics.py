"""Pure NumPy/pandas diagnostics for revision baseline audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def candidate_pool_audit(source_width: int, candidate_k: int, top_k: int) -> dict[str, Any]:
    source_width = int(source_width)
    candidate_k = int(candidate_k)
    top_k = int(top_k)
    effective_width = min(source_width, candidate_k)
    return {
        "source_width": source_width,
        "requested_candidate_k": candidate_k,
        "effective_candidate_k": effective_width,
        "top_k": top_k,
        "candidate_pool_wider_than_topk": bool(effective_width > top_k),
        "requested_pool_fully_available": bool(source_width >= candidate_k),
        "formal_pool_valid": bool(effective_width > top_k and source_width >= candidate_k),
    }


def fairrr_change_summary(
    source_topk: np.ndarray,
    reranked_topk: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    source = np.asarray(source_topk, dtype=np.int64)
    reranked = np.asarray(reranked_topk, dtype=np.int64)
    if source.ndim != 2 or reranked.ndim != 2 or source.shape[0] != reranked.shape[0]:
        raise ValueError("source and reranked Top-K arrays must be aligned 2-D matrices")
    top_k = min(int(top_k), source.shape[1], reranked.shape[1])
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    source = source[:, :top_k]
    reranked = reranked[:, :top_k]

    rank_changed = np.any(source != reranked, axis=1)
    set_changed = np.asarray(
        [set(left.tolist()) != set(right.tolist()) for left, right in zip(source, reranked)],
        dtype=bool,
    )
    changed_positions = (source != reranked).sum(axis=1)
    return {
        "num_users": int(source.shape[0]),
        "top_k": int(top_k),
        "rank_changed_users": int(rank_changed.sum()),
        "rank_changed_user_rate": float(rank_changed.mean()),
        "set_changed_users": int(set_changed.sum()),
        "set_changed_user_rate": float(set_changed.mean()),
        "mean_changed_positions": float(changed_positions.mean()),
        "identical_output": bool(not rank_changed.any()),
    }


def fairrr_candidate_group_diversity(
    topk_items: np.ndarray,
    group_matrix: np.ndarray,
    candidate_k: int,
    sample_limit: int = 1000,
) -> dict[str, Any]:
    topk = np.asarray(topk_items, dtype=np.int64)
    groups = np.asarray(group_matrix)
    if topk.ndim != 2 or groups.ndim != 2:
        raise ValueError("topk_items and group_matrix must be 2-D")
    width = min(int(candidate_k), topk.shape[1])
    sampled = topk[: max(int(sample_limit), 0), :width]
    eligible = 0
    distinct_counts = []
    for row in sampled:
        valid = row[(row > 0) & (row < groups.shape[0])]
        if valid.size == 0:
            distinct_counts.append(0)
            continue
        distinct = int(np.unique(groups[valid], axis=0).shape[0])
        distinct_counts.append(distinct)
        eligible += int(distinct > 1)
    num_sampled = int(sampled.shape[0])
    return {
        "candidate_k": int(width),
        "sample_limit": int(sample_limit),
        "sampled_users": num_sampled,
        "users_with_group_choice": int(eligible),
        "users_with_group_choice_rate": float(eligible / max(num_sampled, 1)),
        "mean_distinct_group_signatures": float(np.mean(distinct_counts)) if distinct_counts else 0.0,
    }


def validate_formal_fairrr(
    change_summary: dict[str, Any],
    pool_audit: dict[str, Any],
    min_rank_changed_user_rate: float = 0.0,
    min_set_changed_user_rate: float = 0.0,
) -> None:
    if not bool(pool_audit.get("formal_pool_valid", False)):
        raise ValueError(
            "FairRR formal candidate pool is invalid: source candidates must cover "
            "candidate_k and the effective candidate pool must be wider than final Top-K."
        )
    if bool(change_summary.get("identical_output", True)):
        raise ValueError("FairRR did not change any recommendation list")

    rank_rate = float(change_summary.get("rank_changed_user_rate", 0.0))
    set_rate = float(change_summary.get("set_changed_user_rate", 0.0))
    min_rank = float(min_rank_changed_user_rate)
    min_set = float(min_set_changed_user_rate)
    if rank_rate + 1e-12 < min_rank or set_rate + 1e-12 < min_set:
        raise ValueError(
            "FairRR change rate is below required minimum: "
            f"rank={rank_rate:.6g} (required {min_rank:.6g}), "
            f"set={set_rate:.6g} (required {min_set:.6g})"
        )

def validate_backbone_checkpoint_load(
    missing: list[str],
    unexpected: list[str],
    checkpoint_path: str,
) -> None:
    if missing or unexpected:
        raise RuntimeError(
            "incomplete backbone checkpoint load; refusing to train or freeze a "
            f"partially initialized backbone from {checkpoint_path}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or float(values.sum()) <= 0:
        return 0.0
    values = np.sort(np.clip(values, 0.0, None))
    index = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, values) / values.sum() - values.size - 1.0) / values.size)


def ranking_collapse_summary(
    topk_items: np.ndarray,
    num_items: int,
    coverage_threshold: float = 0.01,
) -> dict[str, Any]:
    topk = np.asarray(topk_items, dtype=np.int64)
    if topk.ndim != 2:
        raise ValueError("topk_items must be 2-D")
    num_items = int(num_items)
    valid = topk[(topk > 0) & (topk <= num_items)]
    counts = np.bincount(valid, minlength=num_items + 1)[1:].astype(np.float64)
    distinct = int(np.count_nonzero(counts))
    coverage = float(distinct / max(num_items, 1))
    unique_lists = int(np.unique(topk, axis=0).shape[0]) if topk.size else 0
    dominant_share = float(counts.max() / counts.sum()) if float(counts.sum()) > 0 else 0.0
    return {
        "num_users": int(topk.shape[0]),
        "topk_width": int(topk.shape[1]),
        "num_items": num_items,
        "distinct_items": distinct,
        "catalog_coverage": coverage,
        "recommendation_gini": _gini(counts),
        "unique_list_count": unique_lists,
        "unique_list_rate": float(unique_lists / max(topk.shape[0], 1)),
        "dominant_item_share": dominant_share,
        "coverage_threshold": float(coverage_threshold),
        "coverage_collapse": bool(coverage < float(coverage_threshold)),
    }


def learning_curve_summary(
    frame: pd.DataFrame,
    epoch_col: str = "epoch",
    train_loss_col: str = "train_loss",
    val_metric_col: str = "val_metric",
) -> dict[str, Any]:
    required = {epoch_col, train_loss_col, val_metric_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"learning curve is missing columns: {missing}")
    clean = frame[[epoch_col, train_loss_col, val_metric_col]].copy()
    clean = clean.apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("learning curve contains no finite complete epochs")
    best_index = clean[val_metric_col].idxmax()
    first = clean.iloc[0]
    last = clean.iloc[-1]
    best = clean.loc[best_index]
    return {
        "num_logged_epochs": int(len(clean)),
        "first_epoch": int(first[epoch_col]),
        "last_epoch": int(last[epoch_col]),
        "best_epoch": int(best[epoch_col]),
        "first_validation_metric": float(first[val_metric_col]),
        "best_validation_metric": float(best[val_metric_col]),
        "validation_improvement": float(best[val_metric_col] - first[val_metric_col]),
        "first_train_loss": float(first[train_loss_col]),
        "last_train_loss": float(last[train_loss_col]),
        "train_loss_decreased": bool(last[train_loss_col] < first[train_loss_col]),
    }
