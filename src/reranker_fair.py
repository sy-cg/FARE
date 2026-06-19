# -*- coding: utf-8 -*-
"""
src/reranker_fair.py

FairRerank: post-processing fairness baseline for recommendation lists.

This module reranks an existing top-K candidate list without retraining the
underlying recommender. It is intended as a fairness baseline for experiments
such as:

    SASRec-ID + FairRerank
    LateFusion + FairRerank
    GRU4Rec-ID + FairRerank
    BERT4Rec-ID + FairRerank

Core idea
---------
Given a candidate list already sorted by a base model, greedily build a new list
that balances:

    1) base relevance, approximated from original rank or provided scores;
    2) fairness exposure penalty over item group columns;
    3) optional popularity penalty.

At each rank position t, select the item with the highest objective:

    score = alpha * relevance
            - lambda_fair * exposure_penalty_if_selected
            - lambda_pop * popularity_penalty

The group information comes from item_group_matrix.npy with shape
[num_items + 1, num_group_columns]. Padding item id 0 is ignored.

Important limitation
--------------------
If the input topk file only contains top-20 candidates, this reranker can only
change the order of those 20 items. It cannot promote an item that was not in
the candidate pool. For stronger reranking, export a larger candidate pool
(e.g. top-100) from the base model and rerank to top-20.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


DEFAULT_RERANK_GROUPS = [
    "popularity_group",
    "text_quality_group",
    "vision_quality_group",
    "category_proxy_group",
    "brand_store_proxy_group",
    "multimodal_cluster_proxy_group",
]


@dataclass
class FairRerankConfig:
    top_k: int = 20
    candidate_k: int = 20
    alpha_relevance: float = 1.0
    lambda_fair: float = 0.2
    lambda_popularity: float = 0.0
    target_distribution: str = "uniform"  # uniform, empirical
    penalty: str = "l2"  # l1, l2, max_gap
    relevance_mode: str = "log_rank"  # log_rank, reciprocal_rank, linear_rank, score
    discount: str = "log"  # log, reciprocal, none
    groups: Tuple[str, ...] = tuple(DEFAULT_RERANK_GROUPS)
    normalize_group_rows: bool = True
    random_tie_break: bool = False
    seed: int = 2026


# ==========================================================
# Loading helpers
# ==========================================================


def load_npz_topk(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TopK npz not found: {path}")
    obj = np.load(path, allow_pickle=False)
    data = {k: obj[k] for k in obj.files}
    if "topk_items" not in data:
        raise KeyError(f"{path} must contain key 'topk_items'")
    if "targets" not in data:
        raise KeyError(f"{path} must contain key 'targets'")
    if data["topk_items"].ndim != 2:
        raise ValueError(f"topk_items must be [N, K], got {data['topk_items'].shape}")
    return data


def load_item_group_matrix(data_dir: str | Path, filename: str = "item_group_matrix.npy") -> np.ndarray:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"item group matrix not found: {path}")
    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"item_group_matrix must be 2D, got shape={mat.shape}")
    mat = mat.astype(np.float32, copy=False)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    if mat.shape[0] > 0:
        mat[0, :] = 0.0
    return mat


def load_popularity(data_dir: str | Path, filename: str = "item_popularity_train.npy") -> Optional[np.ndarray]:
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32, copy=False).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    if arr.shape[0] > 0:
        arr[0] = 0.0
    return arr


def find_group_metadata_file(data_dir: str | Path, explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = Path(data_dir) / p
        return p if p.exists() else None

    candidates = [
        "fairness_groups.json",
        "fairness_metadata.json",
        "group_metadata.json",
        "item_group_metadata.json",
        "item_group_info.json",
        "fairness_assets.json",
    ]
    data_dir = Path(data_dir)
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None


def load_group_metadata(data_dir: str | Path, explicit_path: Optional[str | Path] = None) -> Optional[Dict]:
    path = find_group_metadata_file(data_dir, explicit_path)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ==========================================================
# Group column selection
# ==========================================================


def _columns_from_spec(spec) -> Optional[List[int]]:
    """Parse flexible metadata column specs."""
    if spec is None:
        return None

    if isinstance(spec, list):
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            start, end = int(spec[0]), int(spec[1])
            # Interpret [start, end] as Python slice if end > start + 1.
            if end > start:
                return list(range(start, end))
        return [int(x) for x in spec]

    if isinstance(spec, dict):
        for key in ["columns", "cols", "column_indices", "indices"]:
            if key in spec:
                return _columns_from_spec(spec[key])
        if "start" in spec and "end" in spec:
            return list(range(int(spec["start"]), int(spec["end"])))
        if "offset" in spec and "dim" in spec:
            s = int(spec["offset"])
            return list(range(s, s + int(spec["dim"])))
        if "start_idx" in spec and "num_classes" in spec:
            s = int(spec["start_idx"])
            return list(range(s, s + int(spec["num_classes"])))

    return None


def infer_group_columns_from_metadata(metadata: Optional[Dict], group_names: Sequence[str]) -> Dict[str, List[int]]:
    """Infer group_name -> item_group_matrix columns from flexible metadata."""
    if not metadata:
        return {}

    result: Dict[str, List[int]] = {}

    # Common direct dict layouts.
    for top_key in [
        "group_slices",
        "group_columns",
        "group_name_to_columns",
        "group_column_indices",
        "item_group_slices",
    ]:
        block = metadata.get(top_key)
        if isinstance(block, dict):
            for g in group_names:
                cols = _columns_from_spec(block.get(g))
                if cols is not None:
                    result[g] = cols

    # List-of-dict layouts.
    for top_key in ["groups", "group_info", "fairness_groups", "group_metadata"]:
        block = metadata.get(top_key)
        if isinstance(block, list):
            for entry in block:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("group_name") or entry.get("key")
                if name in group_names:
                    cols = _columns_from_spec(entry)
                    if cols is not None:
                        result[str(name)] = cols
        elif isinstance(block, dict):
            for g in group_names:
                cols = _columns_from_spec(block.get(g))
                if cols is not None:
                    result[g] = cols

    return result


def select_group_columns(
    group_matrix: np.ndarray,
    groups: Sequence[str] | str = "all",
    metadata: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Select group columns for reranking.

    If metadata is unavailable, selected columns fall back to all non-zero group
    columns. This keeps the reranker usable across preprocessing versions.
    """
    num_cols = group_matrix.shape[1]
    if isinstance(groups, str):
        if groups.lower().strip() == "all":
            group_names = ["all"]
        else:
            group_names = [g.strip() for g in groups.split(",") if g.strip()]
    else:
        group_names = [str(g).strip() for g in groups if str(g).strip()]

    if not group_names or group_names == ["all"]:
        # Avoid columns that are globally all zero.
        active_cols = np.where(group_matrix[1:].sum(axis=0) > 0)[0].astype(int).tolist()
        if not active_cols:
            active_cols = list(range(num_cols))
        return group_matrix[:, active_cols], active_cols, ["all"]

    col_map = infer_group_columns_from_metadata(metadata, group_names)
    cols: List[int] = []
    used_groups: List[str] = []
    for g in group_names:
        if g in col_map:
            valid = [c for c in col_map[g] if 0 <= int(c) < num_cols]
            if valid:
                cols.extend(valid)
                used_groups.append(g)

    # Fallback: if metadata cannot locate requested names, use all active cols.
    if not cols:
        active_cols = np.where(group_matrix[1:].sum(axis=0) > 0)[0].astype(int).tolist()
        if not active_cols:
            active_cols = list(range(num_cols))
        return group_matrix[:, active_cols], active_cols, ["metadata_missing_fallback_all"]

    # Preserve order but remove duplicates.
    seen = set()
    unique_cols = []
    for c in cols:
        c = int(c)
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    return group_matrix[:, unique_cols], unique_cols, used_groups


# ==========================================================
# Reranking primitives
# ==========================================================


def rank_discounts(k: int, mode: str = "log") -> np.ndarray:
    pos = np.arange(k, dtype=np.float32) + 1.0
    mode = mode.lower()
    if mode == "log":
        return (1.0 / np.log2(pos + 1.0)).astype(np.float32)
    if mode == "reciprocal":
        return (1.0 / pos).astype(np.float32)
    if mode == "none":
        return np.ones(k, dtype=np.float32)
    raise ValueError(f"Unsupported discount mode: {mode}")


def base_relevance(candidate_items: np.ndarray, row_data: Dict[str, np.ndarray], row_idx: int, mode: str) -> np.ndarray:
    """Get base relevance for one candidate row."""
    n = candidate_items.shape[0]
    mode = mode.lower()
    if mode == "score" and "topk_scores" in row_data:
        scores = row_data["topk_scores"][row_idx, :n].astype(np.float32, copy=False)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        std = float(scores.std())
        if std > EPS:
            return ((scores - scores.mean()) / std).astype(np.float32)
        return scores.astype(np.float32)

    ranks = np.arange(n, dtype=np.float32) + 1.0
    if mode == "log_rank" or mode == "score":
        return (1.0 / np.log2(ranks + 1.0)).astype(np.float32)
    if mode == "reciprocal_rank":
        return (1.0 / ranks).astype(np.float32)
    if mode == "linear_rank":
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        return (1.0 - (ranks - 1.0) / float(n - 1)).astype(np.float32)
    raise ValueError(f"Unsupported relevance mode: {mode}")


def normalize_group_rows(groups: np.ndarray) -> np.ndarray:
    out = groups.astype(np.float32, copy=True)
    row_sum = out.sum(axis=1, keepdims=True)
    mask = row_sum.squeeze(-1) > EPS
    out[mask] = out[mask] / row_sum[mask]
    return out


def compute_target_distribution(group_matrix: np.ndarray, mode: str = "uniform") -> np.ndarray:
    """Compute target exposure distribution over selected group columns."""
    g = group_matrix[1:].astype(np.float64, copy=False)
    col_mass = g.sum(axis=0)
    active = col_mass > 0
    if not np.any(active):
        return np.ones(group_matrix.shape[1], dtype=np.float32) / max(group_matrix.shape[1], 1)

    mode = mode.lower()
    target = np.zeros(group_matrix.shape[1], dtype=np.float64)
    if mode == "uniform":
        target[active] = 1.0 / float(active.sum())
    elif mode == "empirical":
        target[active] = col_mass[active] / max(float(col_mass[active].sum()), EPS)
    else:
        raise ValueError(f"Unsupported target_distribution: {mode}")
    return target.astype(np.float32)


def exposure_penalty(proposed_exposure: np.ndarray, target: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Vectorized penalty for proposed exposures.

    proposed_exposure: [R, G]
    target: [G]
    returns: [R]
    """
    row_sum = proposed_exposure.sum(axis=1, keepdims=True)
    normalized = np.divide(
        proposed_exposure,
        np.maximum(row_sum, EPS),
        out=np.zeros_like(proposed_exposure, dtype=np.float32),
        where=row_sum > EPS,
    )
    diff = normalized - target.reshape(1, -1)
    mode = mode.lower()
    if mode == "l2":
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
    if mode == "l1":
        return np.sum(np.abs(diff), axis=1).astype(np.float32)
    if mode == "max_gap":
        return np.max(np.abs(diff), axis=1).astype(np.float32)
    raise ValueError(f"Unsupported penalty mode: {mode}")


def build_popularity_penalty(popularity: Optional[np.ndarray], num_items: int) -> Optional[np.ndarray]:
    if popularity is None:
        return None
    pop = popularity.astype(np.float32, copy=False).reshape(-1)
    if pop.shape[0] < num_items + 1:
        padded = np.zeros(num_items + 1, dtype=np.float32)
        padded[: pop.shape[0]] = pop
        pop = padded
    elif pop.shape[0] > num_items + 1:
        pop = pop[: num_items + 1]
    pop = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)
    pop = np.maximum(pop, 0.0)
    logp = np.log1p(pop)
    if logp[1:].std() > EPS:
        logp = (logp - logp[1:].mean()) / max(logp[1:].std(), EPS)
    else:
        logp = np.zeros_like(logp)
    logp[0] = 0.0
    return logp.astype(np.float32)


def rerank_one_list(
    candidate_items: np.ndarray,
    relevance: np.ndarray,
    group_matrix: np.ndarray,
    target_distribution: np.ndarray,
    discounts: np.ndarray,
    cfg: FairRerankConfig,
    popularity_penalty: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Greedy fairness-aware reranking for one user list."""
    candidates = candidate_items.astype(np.int64, copy=True)
    valid_mask = candidates > 0
    candidates = candidates[valid_mask]
    relevance = relevance[valid_mask].astype(np.float32, copy=False)

    if candidates.size == 0:
        return np.zeros(cfg.top_k, dtype=np.int64)

    # Remove duplicated candidate items while preserving base order.
    seen = set()
    unique_items: List[int] = []
    unique_rel: List[float] = []
    for item, rel in zip(candidates.tolist(), relevance.tolist()):
        if item not in seen:
            seen.add(int(item))
            unique_items.append(int(item))
            unique_rel.append(float(rel))
    candidates = np.asarray(unique_items, dtype=np.int64)
    relevance = np.asarray(unique_rel, dtype=np.float32)

    top_k = min(cfg.top_k, candidates.shape[0])
    selected: List[int] = []
    exposure = np.zeros(group_matrix.shape[1], dtype=np.float32)
    remaining = np.arange(candidates.shape[0], dtype=np.int64)

    for pos in range(top_k):
        rem_items = candidates[remaining]
        rem_rel = relevance[remaining]
        rem_groups = group_matrix[rem_items]
        proposed = exposure.reshape(1, -1) + float(discounts[pos]) * rem_groups
        penalties = exposure_penalty(proposed, target_distribution, mode=cfg.penalty)

        objective = cfg.alpha_relevance * rem_rel - cfg.lambda_fair * penalties
        if popularity_penalty is not None and cfg.lambda_popularity > 0:
            objective = objective - cfg.lambda_popularity * popularity_penalty[rem_items]

        if cfg.random_tie_break and rng is not None:
            objective = objective + rng.normal(0.0, 1e-8, size=objective.shape).astype(np.float32)

        best_local = int(np.argmax(objective))
        chosen_global_idx = int(remaining[best_local])
        chosen_item = int(candidates[chosen_global_idx])
        selected.append(chosen_item)
        exposure += float(discounts[pos]) * group_matrix[chosen_item]
        remaining = np.delete(remaining, best_local)
        if remaining.size == 0:
            break

    if len(selected) < cfg.top_k:
        # Pad with zeros only if candidate pool is smaller than top_k.
        selected.extend([0] * (cfg.top_k - len(selected)))
    return np.asarray(selected[: cfg.top_k], dtype=np.int64)


# ==========================================================
# Metrics and batch processing
# ==========================================================


def compute_ranks_from_topk(topk_items: np.ndarray, targets: np.ndarray, fallback_ranks: Optional[np.ndarray] = None) -> np.ndarray:
    n, k = topk_items.shape
    ranks = np.full(n, k + 1, dtype=np.int64)
    if fallback_ranks is not None:
        fallback = fallback_ranks.astype(np.int64, copy=False).reshape(-1)
        if fallback.shape[0] == n:
            ranks[:] = fallback
    for i in range(n):
        pos = np.where(topk_items[i] == int(targets[i]))[0]
        if pos.size > 0:
            ranks[i] = int(pos[0]) + 1
    return ranks


def ranking_metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    ranks = ranks.astype(np.int64, copy=False).reshape(-1)
    out: Dict[str, float] = {}
    if ranks.size == 0:
        for k in ks:
            for m in ["hit", "recall", "ndcg", "mrr"]:
                out[f"{m}@{k}"] = float("nan")
        return out
    for k in ks:
        hit = ranks <= int(k)
        out[f"hit@{k}"] = float(hit.mean())
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
        out[f"mrr@{k}"] = float(np.where(hit, 1.0 / ranks, 0.0).mean())
    return out


def rerank_topk_data(
    topk_data: Dict[str, np.ndarray],
    group_matrix: np.ndarray,
    cfg: FairRerankConfig,
    popularity_penalty: Optional[np.ndarray] = None,
    ks: Sequence[int] = (5, 10, 20),
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Rerank loaded topk data and return new npz-compatible data + metrics."""
    topk_items = topk_data["topk_items"].astype(np.int64, copy=False)
    targets = topk_data["targets"].astype(np.int64, copy=False).reshape(-1)
    user_ids = topk_data.get("user_ids", np.arange(topk_items.shape[0], dtype=np.int64)).astype(np.int64, copy=False)

    if topk_items.shape[0] != targets.shape[0]:
        raise ValueError(f"topk_items rows {topk_items.shape[0]} != targets length {targets.shape[0]}")

    candidate_k = min(int(cfg.candidate_k), topk_items.shape[1])
    top_k = min(int(cfg.top_k), candidate_k)
    cfg = FairRerankConfig(**{**asdict(cfg), "top_k": top_k, "candidate_k": candidate_k})

    if cfg.normalize_group_rows:
        selected_group_matrix = normalize_group_rows(group_matrix)
    else:
        selected_group_matrix = group_matrix.astype(np.float32, copy=False)

    target_distribution = compute_target_distribution(selected_group_matrix, mode=cfg.target_distribution)
    discounts = rank_discounts(cfg.top_k, mode=cfg.discount)
    rng = np.random.default_rng(cfg.seed)

    reranked = np.zeros((topk_items.shape[0], cfg.top_k), dtype=np.int64)
    for i in range(topk_items.shape[0]):
        candidates = topk_items[i, :candidate_k]
        rel = base_relevance(candidates, topk_data, i, mode=cfg.relevance_mode)
        reranked[i] = rerank_one_list(
            candidate_items=candidates,
            relevance=rel,
            group_matrix=selected_group_matrix,
            target_distribution=target_distribution,
            discounts=discounts,
            cfg=cfg,
            popularity_penalty=popularity_penalty,
            rng=rng,
        )

    fallback_ranks = topk_data.get("ranks")
    ranks = compute_ranks_from_topk(reranked, targets, fallback_ranks=fallback_ranks)
    metrics = ranking_metrics_from_ranks(ranks, ks=ks)

    out = {
        "user_ids": user_ids,
        "targets": targets,
        "topk_items": reranked,
        "ranks": ranks,
    }
    # Preserve optional arrays for traceability if dimensions match.
    if "original_ranks" in topk_data:
        out["original_ranks"] = topk_data["original_ranks"]
    elif "ranks" in topk_data:
        out["original_ranks"] = topk_data["ranks"]
    out["source_topk_items"] = topk_items[:, :candidate_k]
    return out, metrics


def save_topk_npz(data: Dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def rerank_topk_file(
    input_npz: str | Path,
    output_npz: str | Path,
    group_matrix: np.ndarray,
    cfg: FairRerankConfig,
    popularity_penalty: Optional[np.ndarray] = None,
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    topk_data = load_npz_topk(input_npz)
    reranked, metrics = rerank_topk_data(
        topk_data=topk_data,
        group_matrix=group_matrix,
        cfg=cfg,
        popularity_penalty=popularity_penalty,
        ks=ks,
    )
    save_topk_npz(reranked, output_npz)
    return metrics
