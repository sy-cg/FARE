# -*- coding: utf-8 -*-
"""
src/evaluator_fairness.py

Item-side and modality-side fairness evaluator for SDR / FARE experiments.

This module consumes RankingResult objects produced by src/evaluator_ranking.py.
It computes:
- group-wise ranking metrics by target-item group;
- position-discounted item and group exposure;
- utility-aware exposure fairness using train-only group utility;
- recommendation distribution metrics: coverage, entropy, Gini, average popularity;
- optional CSV saving helpers and smoke-test helper.

Expected processed dataset files:
- fairness_groups.json
- group_schema.json
- group_utility_train.json
- group_item_counts.json
- item_popularity_train.npy
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from .evaluator_ranking import (
        RankingResult,
        RankingEvaluator,
        PopularityDummyModel,
        infer_num_items_from_processed_dir,
        score_fn_from_model_method,
        save_ranking_result_npz,
    )
except ImportError:  # Allows running this file directly from src/ during quick tests.
    from evaluator_ranking import (  # type: ignore
        RankingResult,
        RankingEvaluator,
        PopularityDummyModel,
        infer_num_items_from_processed_dir,
        score_fn_from_model_method,
        save_ranking_result_npz,
    )


# ==========================================================
# Basic utilities
# ==========================================================


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(numer: float, denom: float, eps: float = 1e-12) -> float:
    return float(numer) / float(denom + eps)


def entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / (total + eps)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs + eps)).sum())


def gini_from_values(values: np.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient for non-negative values."""
    x = np.asarray(values, dtype=np.float64).flatten()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / (cumx[-1] + eps)) / n)


def save_eval_tables(tables: Dict[str, pd.DataFrame], output_dir: str, prefix: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(output_dir, f"{prefix}_{name}.csv"), index=False)


# ==========================================================
# Fairness evaluator
# ==========================================================


class FairnessEvaluator:
    """Fairness evaluator for item-side and modality-side groups.

    It supports two analysis views:
    1. Group-wise ranking metrics: grouped by the target item of each test case.
    2. Exposure fairness: grouped by recommended items in the Top-K list.

    Default groups are the four mandatory fairness groups:
    - popularity_group
    - modality_availability_group
    - text_quality_group
    - vision_quality_group

    Optional proxy groups can be enabled through group_names:
    - category_proxy_group
    - brand_store_proxy_group
    - multimodal_cluster_proxy_group
    """

    DEFAULT_GROUPS = (
        "popularity_group",
        "modality_availability_group",
        "text_quality_group",
        "vision_quality_group",
    )

    def __init__(
        self,
        data_dir: str,
        ks: Sequence[int] = (5, 10, 20),
        group_names: Optional[Sequence[str]] = None,
        eps: float = 1e-12,
    ) -> None:
        self.data_dir = data_dir
        self.ks = tuple(sorted(set(int(k) for k in ks)))
        if not self.ks or min(self.ks) <= 0:
            raise ValueError(f"Invalid ks: {ks}")
        self.max_k = max(self.ks)
        self.eps = float(eps)

        self.num_items = infer_num_items_from_processed_dir(data_dir)
        self.popularity = self._load_popularity()

        self.fairness_groups = self._load_required_json("fairness_groups.json")
        self.group_schema = self._load_required_json("group_schema.json")
        self.group_utility = self._load_required_json("group_utility_train.json")
        self.group_item_counts = self._load_required_json("group_item_counts.json")

        available = list(self.fairness_groups.keys())
        if group_names is None:
            self.group_names = [g for g in self.DEFAULT_GROUPS if g in self.fairness_groups]
        else:
            missing = [g for g in group_names if g not in self.fairness_groups]
            if missing:
                raise KeyError(f"Unknown fairness groups: {missing}. Available groups: {available}")
            self.group_names = list(group_names)

        if not self.group_names:
            raise ValueError(f"No valid fairness groups found in {data_dir}")

        self.group_arrays = {
            group_name: self._group_dict_to_array(self.fairness_groups[group_name])
            for group_name in self.group_names
        }
        self.group_value_names = self.group_schema.get("group_value_names", {})

    def _load_required_json(self, filename: str) -> Any:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required fairness file missing: {path}")
        return load_json(path)

    def _load_popularity(self) -> np.ndarray:
        path = os.path.join(self.data_dir, "item_popularity_train.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing item_popularity_train.npy: {path}")
        popularity = np.load(path).astype(np.float64)
        expected = self.num_items + 1
        if popularity.shape[0] != expected:
            raise ValueError(f"Popularity shape mismatch: got {popularity.shape[0]}, expected {expected}")
        return popularity

    def _group_dict_to_array(self, group_dict: Dict[str, int]) -> np.ndarray:
        arr = np.full(self.num_items + 1, -1, dtype=np.int64)
        for item_str, group_value in group_dict.items():
            item = int(item_str)
            if 0 <= item <= self.num_items:
                arr[item] = int(group_value)
        return arr

    def _group_label(self, group_name: str, group_value: int | str) -> str:
        if isinstance(group_value, str):
            return group_value
        names = self.group_value_names.get(group_name, {})
        return str(names.get(str(int(group_value)), str(int(group_value))))

    @staticmethod
    def _discounts(k: int) -> np.ndarray:
        ranks = np.arange(1, k + 1, dtype=np.float64)
        return 1.0 / np.log2(ranks + 1.0)

    def evaluate(self, ranking_result: RankingResult) -> Dict[str, pd.DataFrame]:
        """Return all fairness tables for a RankingResult."""
        if ranking_result.topk_items.ndim != 2:
            raise ValueError(f"topk_items must be 2-D, got {ranking_result.topk_items.shape}")
        if ranking_result.topk_items.shape[1] < self.max_k:
            raise ValueError(
                f"RankingResult top-K={ranking_result.topk_items.shape[1]} smaller than required max_k={self.max_k}"
            )

        return {
            "group_ranking": self.groupwise_ranking_metrics(ranking_result),
            "exposure": self.exposure_metrics(ranking_result.topk_items),
            "summary": self.summary_metrics(ranking_result.topk_items),
        }

    # ------------------------------------------------------
    # Group-wise target ranking metrics
    # ------------------------------------------------------

    def groupwise_ranking_metrics(self, ranking_result: RankingResult) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        targets = ranking_result.targets.astype(np.int64)
        hit_ranks = ranking_result.hit_ranks.astype(np.int64)

        if len(targets) != len(hit_ranks):
            raise ValueError("targets and hit_ranks have different lengths")

        for group_name, group_arr in self.group_arrays.items():
            target_groups = group_arr[targets]
            valid_values = sorted(int(v) for v in np.unique(target_groups) if int(v) >= 0)

            for group_value in valid_values:
                mask = target_groups == group_value
                num_targets = int(mask.sum())
                if num_targets == 0:
                    continue

                group_ranks = hit_ranks[mask]
                row: Dict[str, Any] = {
                    "group_name": group_name,
                    "group_value": int(group_value),
                    "group_label": self._group_label(group_name, group_value),
                    "num_targets": num_targets,
                }
                row.update(RankingEvaluator.compute_single_target_metrics(group_ranks, self.ks))
                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------
    # Exposure fairness metrics
    # ------------------------------------------------------

    def exposure_metrics(self, topk_items: np.ndarray) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        topk_items = np.asarray(topk_items, dtype=np.int64)

        for k in self.ks:
            recs_k = topk_items[:, :k]
            item_exposure = self._compute_item_exposure(recs_k, k)
            total_exposure = float(item_exposure.sum())
            total_train_utility = float(self.popularity[1:].sum())

            for group_name, group_arr in self.group_arrays.items():
                valid_values = sorted(int(v) for v in np.unique(group_arr[1:]) if int(v) >= 0)

                group_exposure_shares: List[float] = []
                group_utility_ratios: List[float] = []

                for group_value in valid_values:
                    item_mask = group_arr == group_value
                    item_mask[0] = False
                    item_count = int(item_mask.sum())
                    exposure_g = float(item_exposure[item_mask].sum())
                    exposure_share = safe_div(exposure_g, total_exposure, self.eps)

                    utility_g = float(self.group_utility.get(group_name, {}).get(str(group_value), 0.0))
                    utility_share = safe_div(utility_g, total_train_utility, self.eps)
                    exposure_per_utility = safe_div(exposure_share, utility_share, self.eps)
                    exposure_per_item = safe_div(exposure_g, item_count, self.eps)

                    group_exposure_shares.append(exposure_share)
                    group_utility_ratios.append(exposure_per_utility)

                    rows.append({
                        "k": int(k),
                        "group_name": group_name,
                        "group_value": int(group_value),
                        "group_label": self._group_label(group_name, group_value),
                        "num_items": item_count,
                        "utility_train": utility_g,
                        "utility_share": utility_share,
                        "exposure": exposure_g,
                        "exposure_share": exposure_share,
                        "exposure_per_item": exposure_per_item,
                        "exposure_per_utility": exposure_per_utility,
                    })

                if group_exposure_shares:
                    exp_arr = np.asarray(group_exposure_shares, dtype=np.float64)
                    ratio_arr = np.asarray(group_utility_ratios, dtype=np.float64)
                    finite_ratio = ratio_arr[np.isfinite(ratio_arr)]
                    if len(finite_ratio) == 0:
                        finite_ratio = np.asarray([0.0])

                    rows.append({
                        "k": int(k),
                        "group_name": group_name,
                        "group_value": "__aggregate__",
                        "group_label": "__aggregate__",
                        "num_items": int(np.sum(group_arr[1:] >= 0)),
                        "utility_train": float(self.group_utility.get(group_name, {}).get("__total__", 0.0)),
                        "utility_share": 1.0,
                        "exposure": total_exposure,
                        "exposure_share": 1.0,
                        "exposure_per_item": np.nan,
                        "exposure_per_utility": np.nan,
                        "exposure_share_gap": float(exp_arr.max() - exp_arr.min()),
                        "exposure_share_gini": gini_from_values(exp_arr),
                        "utility_aware_gap": float(finite_ratio.max() - finite_ratio.min()),
                        "utility_aware_l1": float(np.mean(np.abs(finite_ratio - 1.0))),
                    })

        return pd.DataFrame(rows)

    def _compute_item_exposure(self, recs_k: np.ndarray, k: int) -> np.ndarray:
        item_exposure = np.zeros(self.num_items + 1, dtype=np.float64)
        discounts = self._discounts(k)

        for rank_idx in range(k):
            items = recs_k[:, rank_idx]
            valid = (items > 0) & (items <= self.num_items)
            if np.any(valid):
                np.add.at(item_exposure, items[valid], discounts[rank_idx])
        return item_exposure

    # ------------------------------------------------------
    # Overall recommendation distribution metrics
    # ------------------------------------------------------

    def summary_metrics(self, topk_items: np.ndarray) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        topk_items = np.asarray(topk_items, dtype=np.int64)
        popularity_group = self.group_arrays.get("popularity_group")

        for k in self.ks:
            recs_k = topk_items[:, :k]
            valid = (recs_k > 0) & (recs_k <= self.num_items)
            recommended = recs_k[valid]

            if len(recommended) == 0:
                rows.append({
                    "k": int(k),
                    "catalog_coverage": 0.0,
                    "recommendation_entropy": 0.0,
                    "recommendation_gini": 0.0,
                    "average_train_popularity": 0.0,
                    "num_distinct_recommended_items": 0,
                    "tail_item_ratio": 0.0,
                })
                continue

            item_counts = np.bincount(recommended, minlength=self.num_items + 1).astype(np.float64)
            distinct_items = int(np.count_nonzero(item_counts[1:] > 0))
            catalog_coverage = safe_div(distinct_items, self.num_items, self.eps)
            entropy = entropy_from_counts(item_counts[1:], self.eps)
            gini = gini_from_values(item_counts[1:], self.eps)
            avg_pop = float(np.mean(self.popularity[recommended]))

            row: Dict[str, Any] = {
                "k": int(k),
                "catalog_coverage": catalog_coverage,
                "recommendation_entropy": entropy,
                "recommendation_gini": gini,
                "average_train_popularity": avg_pop,
                "num_distinct_recommended_items": distinct_items,
            }

            if popularity_group is not None:
                tail_mask = popularity_group[recommended] == 0
                row["tail_item_ratio"] = float(np.mean(tail_mask))
            else:
                row["tail_item_ratio"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)


# ==========================================================
# Smoke test
# ==========================================================


def smoke_test_evaluators(
    data_dir: str,
    split: str = "test",
    batch_size: int = 256,
    max_seq_len: int = 50,
    output_dir: Optional[str] = None,
) -> Tuple[RankingResult, Dict[str, pd.DataFrame]]:
    """Run end-to-end smoke test using a deterministic popularity dummy model."""
    num_items = infer_num_items_from_processed_dir(data_dir)
    popularity = np.load(os.path.join(data_dir, "item_popularity_train.npy"))
    model = PopularityDummyModel(popularity)

    ranking_evaluator = RankingEvaluator(
        num_items=num_items,
        ks=(5, 10, 20),
        max_seq_len=max_seq_len,
    )

    split_path = os.path.join(data_dir, f"{split}.txt")
    result = ranking_evaluator.evaluate_from_file(
        model=model,
        split_path=split_path,
        score_fn=score_fn_from_model_method,
        batch_size=batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        split_name=split,
        mask_seen_items=True,
    )

    print("Ranking metrics:")
    print(json.dumps(result.metrics, indent=2))

    fairness_evaluator = FairnessEvaluator(data_dir=data_dir, ks=(5, 10, 20))
    tables = fairness_evaluator.evaluate(result)

    for name, df in tables.items():
        print(f"\n[{name}] head:")
        print(df.head())

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_ranking_result_npz(result, os.path.join(output_dir, f"{split}_popularity_dummy_ranking.npz"))
        save_eval_tables(tables, output_dir, prefix=f"{split}_popularity_dummy")

    return result, tables


if __name__ == "__main__":
    # Run from project root:
    #   python -m src.evaluator_fairness
    data_dir = os.path.join("data", "Processed_All_Beauty")
    if os.path.exists(data_dir):
        smoke_test_evaluators(
            data_dir=data_dir,
            split="test",
            batch_size=128,
            max_seq_len=50,
            output_dir=os.path.join("results", "smoke_test", "All_Beauty"),
        )
    else:
        print(f"Data dir not found: {data_dir}")
