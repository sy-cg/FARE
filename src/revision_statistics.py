"""Torch-independent selection and statistical utilities for the IPM revision."""

from __future__ import annotations

import itertools
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


UTILITY_METRICS = ("ndcg", "hr")
EXPOSURE_HIGH_METRICS = ("coverage",)
EXPOSURE_LOW_METRICS = ("rec_gini", "pop_gap", "brand_gap", "cluster_gap")


def _normalized(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = series.astype(float)
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    out = (values - low) / (high - low)
    return out if higher_is_better else 1.0 - out


def _validate_selection_frame(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    candidate_col: str,
    required_seeds: Optional[Iterable[int]],
) -> pd.DataFrame:
    required = set(group_cols) | {candidate_col, "split", "seed", *UTILITY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"selection manifest is missing columns: {missing}")

    val = frame[frame["split"].astype(str).str.lower().eq("val")].copy()
    if val.empty:
        raise ValueError("selection manifest contains no validation rows")

    if required_seeds is not None:
        expected = {int(seed) for seed in required_seeds}
        keys = list(group_cols) + [candidate_col]
        for key, rows in val.groupby(keys, dropna=False):
            observed = {int(seed) for seed in rows["seed"].tolist()}
            if observed != expected:
                raise ValueError(
                    f"candidate {key!r} has incomplete seed set: "
                    f"expected={sorted(expected)} observed={sorted(observed)}"
                )
    return val


def select_on_validation(
    frame: pd.DataFrame,
    alpha: float = 0.5,
    group_cols: Sequence[str] = ("dataset", "backbone"),
    candidate_col: str = "config_id",
    required_seeds: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Select one candidate per group using validation metrics only."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    val = _validate_selection_frame(
        frame=frame,
        group_cols=group_cols,
        candidate_col=candidate_col,
        required_seeds=required_seeds,
    )

    metric_cols = [
        metric
        for metric in (*UTILITY_METRICS, *EXPOSURE_HIGH_METRICS, *EXPOSURE_LOW_METRICS)
        if metric in val.columns
    ]
    means = (
        val.groupby([*group_cols, candidate_col], as_index=False, dropna=False)[metric_cols]
        .mean(numeric_only=True)
    )

    selections = []
    group_key = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for _, group in means.groupby(group_key, sort=True, dropna=False):
        group = group.copy()
        utility_components = [
            _normalized(group[metric], higher_is_better=True)
            for metric in UTILITY_METRICS
            if metric in group.columns
        ]
        exposure_components = [
            _normalized(group[metric], higher_is_better=True)
            for metric in EXPOSURE_HIGH_METRICS
            if metric in group.columns
        ]
        exposure_components.extend(
            _normalized(group[metric], higher_is_better=False)
            for metric in EXPOSURE_LOW_METRICS
            if metric in group.columns
        )
        if not utility_components or not exposure_components:
            raise ValueError("selection requires utility and exposure metrics")

        group["utility_score"] = pd.concat(utility_components, axis=1).mean(axis=1)
        group["exposure_score"] = pd.concat(exposure_components, axis=1).mean(axis=1)
        group["tradeoff_score"] = (
            alpha * group["utility_score"] + (1.0 - alpha) * group["exposure_score"]
        )
        group["alpha"] = alpha
        group = group.sort_values(
            ["tradeoff_score", "ndcg", candidate_col],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selections.append(group.iloc[[0]])

    return pd.concat(selections, ignore_index=True)


def policy_sensitivity(
    frame: pd.DataFrame,
    alphas: Sequence[float],
    group_cols: Sequence[str] = ("dataset", "backbone"),
    candidate_col: str = "config_id",
    required_seeds: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Run validation-only selection for every requested utility weight."""
    outputs = [
        select_on_validation(
            frame=frame,
            alpha=float(alpha),
            group_cols=group_cols,
            candidate_col=candidate_col,
            required_seeds=required_seeds,
        )
        for alpha in alphas
    ]
    if not outputs:
        raise ValueError("alphas cannot be empty")
    return pd.concat(outputs, ignore_index=True)


def per_user_ranking_metric(
    topk_items: np.ndarray,
    targets: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    """Return one HR or NDCG contribution per evaluation record."""
    topk = np.asarray(topk_items, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if topk.ndim != 2 or topk.shape[0] != targets.size:
        raise ValueError("topk_items must be 2-D and aligned with targets")
    k = min(max(int(k), 1), topk.shape[1])
    matches = topk[:, :k] == targets[:, None]
    hits = matches.any(axis=1)

    metric = str(metric).strip().lower()
    if metric in {"hr", "hit", "recall"}:
        return hits.astype(np.float64)
    if metric != "ndcg":
        raise ValueError(f"Unsupported per-user ranking metric={metric!r}")

    out = np.zeros(targets.size, dtype=np.float64)
    if hits.any():
        ranks = matches[hits].argmax(axis=1) + 1
        out[hits] = 1.0 / np.log2(ranks.astype(np.float64) + 1.0)
    return out


def align_paired_user_values(
    left_users: np.ndarray,
    left_values: np.ndarray,
    right_users: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align paired values to the left user order and reject invalid pairs."""
    left_users = np.asarray(left_users, dtype=np.int64).reshape(-1)
    right_users = np.asarray(right_users, dtype=np.int64).reshape(-1)
    left_values = np.asarray(left_values, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right_values, dtype=np.float64).reshape(-1)
    if left_users.size != left_values.size or right_users.size != right_values.size:
        raise ValueError("user ids and values must have matching lengths")
    if np.unique(left_users).size != left_users.size or np.unique(right_users).size != right_users.size:
        raise ValueError("paired archives must contain unique user ids")
    if set(left_users.tolist()) != set(right_users.tolist()):
        raise ValueError("paired archives must contain the same user set")

    right_map = {int(user): float(value) for user, value in zip(right_users, right_values)}
    aligned_right = np.asarray([right_map[int(user)] for user in left_users], dtype=np.float64)
    return left_values, aligned_right


def exact_sign_flip_test(seed_differences: Sequence[float]) -> dict[str, float | int]:
    """Exact two-sided paired randomization test over model-seed deltas."""
    values = np.asarray(seed_differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed_differences must be non-empty and finite")
    observed = abs(float(values.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistics.append(abs(float((values * np.asarray(signs)).mean())))
    statistics_arr = np.asarray(statistics, dtype=np.float64)
    p_value = float(np.mean(statistics_arr >= observed - 1e-12))
    return {
        "estimate": float(values.mean()),
        "p_value_two_sided": p_value,
        "num_permutations": int(statistics_arr.size),
    }


def _paired_effect(seed_differences: np.ndarray) -> float:
    if seed_differences.size <= 1:
        return float("nan")
    scale = float(seed_differences.std(ddof=1))
    mean = float(seed_differences.mean())
    if np.isclose(scale, 0.0):
        if np.isclose(mean, 0.0):
            return 0.0
        return float(np.copysign(np.inf, mean))
    return mean / scale


def seed_level_paired_bootstrap(
    seed_differences: Sequence[float],
    repetitions: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int | str | list[float]]:
    """Bootstrap paired seed-level differences.

    This treats each matched random seed as the independent experimental unit.
    It is much cheaper than user-level resampling and is the more conservative
    default for revision claims about robustness across model initializations.
    """
    values = np.asarray(seed_differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed_differences must be non-empty and finite")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rng = np.random.default_rng(int(seed))
    num_seeds = int(values.size)
    selected = rng.integers(0, num_seeds, size=(repetitions, num_seeds))
    samples = values[selected].mean(axis=1)
    sign_flip = exact_sign_flip_test(values)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "probability_improvement": float(np.mean(samples > 0.0)),
        "paired_standardized_effect": _paired_effect(values),
        "seed_differences": values.tolist(),
        "sign_flip_p_value": float(sign_flip["p_value_two_sided"]),
        "num_seeds": num_seeds,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": int(seed),
        "bootstrap_mode": "seed",
    }


def hierarchical_paired_bootstrap(
    user_differences_by_seed: Sequence[np.ndarray],
    repetitions: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int | list[float]]:
    """Bootstrap paired user differences while preserving model-seed variation."""
    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for values in user_differences_by_seed]
    if not arrays or any(values.size == 0 for values in arrays):
        raise ValueError("every seed must contain at least one paired user difference")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("paired user differences must be finite")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    seed_means = np.asarray([values.mean() for values in arrays], dtype=np.float64)
    estimate = float(seed_means.mean())
    rng = np.random.default_rng(int(seed))
    samples = np.empty(repetitions, dtype=np.float64)
    num_seeds = len(arrays)
    for rep in range(repetitions):
        selected_seeds = rng.integers(0, num_seeds, size=num_seeds)
        sampled_seed_means = []
        for seed_index in selected_seeds:
            values = arrays[int(seed_index)]
            indices = rng.integers(0, values.size, size=values.size)
            sampled_seed_means.append(float(values[indices].mean()))
        samples[rep] = float(np.mean(sampled_seed_means))

    sign_flip = exact_sign_flip_test(seed_means)
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "probability_improvement": float(np.mean(samples > 0.0)),
        "paired_standardized_effect": _paired_effect(seed_means),
        "seed_differences": seed_means.tolist(),
        "sign_flip_p_value": float(sign_flip["p_value_two_sided"]),
        "num_seeds": int(num_seeds),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": int(seed),
        "bootstrap_mode": "hierarchical",
    }


def group_exposure_gap(
    topk_items: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Position-discounted max-minus-min group exposure-share gap."""
    topk = np.asarray(topk_items, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if topk.ndim != 2:
        raise ValueError("topk_items must be 2-D")
    k = min(max(int(k), 1), topk.shape[1])
    catalog_labels = labels[1:]
    catalog_labels = catalog_labels[catalog_labels >= 0]
    if catalog_labels.size == 0:
        raise ValueError("labels contain no valid catalog groups")
    num_groups = int(catalog_labels.max()) + 1
    exposure = np.zeros(num_groups, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))

    for rank in range(k):
        item_ids = topk[:, rank]
        valid = (item_ids > 0) & (item_ids < labels.size)
        group_ids = labels[item_ids[valid]]
        group_ids = group_ids[group_ids >= 0]
        if group_ids.size:
            exposure += np.bincount(group_ids, minlength=num_groups) * discounts[rank]
    total = float(exposure.sum())
    if total <= 0:
        raise ValueError("recommendations contain no valid grouped exposure")
    shares = exposure / total
    return float(shares.max() - shares.min())


def recommendation_coverage(topk_items: np.ndarray, num_items: int, k: int) -> float:
    """Catalog coverage using the same definition as ``FairnessEvaluator``."""
    topk = np.asarray(topk_items, dtype=np.int64)
    if topk.ndim != 2:
        raise ValueError("topk_items must be 2-D")
    num_items = int(num_items)
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    k = min(max(int(k), 1), topk.shape[1])
    values = topk[:, :k]
    valid = values[(values > 0) & (values <= num_items)]
    return float(np.unique(valid).size / num_items) if valid.size else 0.0


def recommendation_gini(topk_items: np.ndarray, num_items: int, k: int) -> float:
    """Gini of recommendation counts over the complete item catalog."""
    topk = np.asarray(topk_items, dtype=np.int64)
    if topk.ndim != 2:
        raise ValueError("topk_items must be 2-D")
    num_items = int(num_items)
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    k = min(max(int(k), 1), topk.shape[1])
    values = topk[:, :k]
    valid = values[(values > 0) & (values <= num_items)]
    counts = np.bincount(valid, minlength=num_items + 1)[1:].astype(np.float64)
    if not counts.any():
        return 0.0
    ordered = np.sort(counts)
    cumulative = np.cumsum(ordered)
    n = ordered.size
    return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)


def group_utility_aware_gap(
    topk_items: np.ndarray,
    labels: np.ndarray,
    group_utility: dict[int | str, float],
    k: int,
) -> float:
    """Max-minus-min exposure/utility ratio used in manuscript fairness tables."""
    topk = np.asarray(topk_items, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if topk.ndim != 2:
        raise ValueError("topk_items must be 2-D")
    k = min(max(int(k), 1), topk.shape[1])
    groups = sorted(int(value) for value in np.unique(labels[1:]) if int(value) >= 0)
    if not groups:
        raise ValueError("labels contain no valid catalog groups")
    utility = {
        group: float(group_utility.get(group, group_utility.get(str(group), 0.0)))
        for group in groups
    }
    total_utility = float(
        group_utility.get("__total__", sum(utility.values()))
    )
    if total_utility <= 0 or any(value <= 0 for value in utility.values()):
        raise ValueError("every catalog group must have positive train utility")

    exposure = {group: 0.0 for group in groups}
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    for rank in range(k):
        item_ids = topk[:, rank]
        valid = (item_ids > 0) & (item_ids < labels.size)
        for group in labels[item_ids[valid]]:
            if int(group) in exposure:
                exposure[int(group)] += float(discounts[rank])
    total_exposure = float(sum(exposure.values()))
    if total_exposure <= 0:
        raise ValueError("recommendations contain no valid grouped exposure")
    ratios = [
        (exposure[group] / total_exposure) / (utility[group] / total_utility)
        for group in groups
    ]
    return float(max(ratios) - min(ratios))


def hierarchical_topk_metric_bootstrap(
    base_topk_by_seed: Sequence[np.ndarray],
    method_topk_by_seed: Sequence[np.ndarray],
    metric_fn,
    higher_is_better: bool,
    repetitions: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int | list[float]]:
    """Hierarchical paired bootstrap for aggregate Top-K fairness metrics.

    The returned estimate is always oriented so a positive value means that the
    method improved over the baseline.
    """
    base = [np.asarray(items, dtype=np.int64) for items in base_topk_by_seed]
    method = [np.asarray(items, dtype=np.int64) for items in method_topk_by_seed]
    if not base or len(base) != len(method):
        raise ValueError("base and method must contain the same non-zero number of seeds")
    for base_items, method_items in zip(base, method):
        if base_items.shape != method_items.shape or base_items.ndim != 2:
            raise ValueError("paired Top-K archives must have identical 2-D shapes")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    direction = 1.0 if higher_is_better else -1.0
    seed_differences = np.asarray(
        [direction * (float(metric_fn(m)) - float(metric_fn(b))) for b, m in zip(base, method)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(repetitions, dtype=np.float64)
    num_seeds = len(base)
    for rep in range(repetitions):
        selected_seeds = rng.integers(0, num_seeds, size=num_seeds)
        differences = []
        for seed_index in selected_seeds:
            base_items = base[int(seed_index)]
            method_items = method[int(seed_index)]
            rows = rng.integers(0, base_items.shape[0], size=base_items.shape[0])
            differences.append(
                direction
                * (float(metric_fn(method_items[rows])) - float(metric_fn(base_items[rows])))
            )
        samples[rep] = float(np.mean(differences))

    sign_flip = exact_sign_flip_test(seed_differences)
    return {
        "estimate": float(seed_differences.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "probability_improvement": float(np.mean(samples > 0.0)),
        "paired_standardized_effect": _paired_effect(seed_differences),
        "seed_differences": seed_differences.tolist(),
        "sign_flip_p_value": float(sign_flip["p_value_two_sided"]),
        "num_seeds": int(num_seeds),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": int(seed),
        "bootstrap_mode": "hierarchical",
    }


def hierarchical_exposure_gap_bootstrap(
    base_topk_by_seed: Sequence[np.ndarray],
    method_topk_by_seed: Sequence[np.ndarray],
    labels: np.ndarray,
    k: int,
    repetitions: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int | list[float]]:
    """Bootstrap positive exposure-gap reductions for paired recommendation lists."""
    base = [np.asarray(items, dtype=np.int64) for items in base_topk_by_seed]
    method = [np.asarray(items, dtype=np.int64) for items in method_topk_by_seed]
    if not base or len(base) != len(method):
        raise ValueError("base and method must contain the same non-zero number of seeds")
    for base_items, method_items in zip(base, method):
        if base_items.shape != method_items.shape or base_items.ndim != 2:
            raise ValueError("paired exposure archives must have identical 2-D shapes")

    seed_differences = np.asarray(
        [
            group_exposure_gap(base_items, labels, k)
            - group_exposure_gap(method_items, labels, k)
            for base_items, method_items in zip(base, method)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    repetitions = int(repetitions)
    samples = np.empty(repetitions, dtype=np.float64)
    num_seeds = len(base)
    for rep in range(repetitions):
        selected_seeds = rng.integers(0, num_seeds, size=num_seeds)
        differences = []
        for seed_index in selected_seeds:
            base_items = base[int(seed_index)]
            method_items = method[int(seed_index)]
            row_indices = rng.integers(0, base_items.shape[0], size=base_items.shape[0])
            differences.append(
                group_exposure_gap(base_items[row_indices], labels, k)
                - group_exposure_gap(method_items[row_indices], labels, k)
            )
        samples[rep] = float(np.mean(differences))

    sign_flip = exact_sign_flip_test(seed_differences)
    return {
        "estimate": float(seed_differences.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "probability_improvement": float(np.mean(samples > 0.0)),
        "paired_standardized_effect": _paired_effect(seed_differences),
        "seed_differences": seed_differences.tolist(),
        "sign_flip_p_value": float(sign_flip["p_value_two_sided"]),
        "num_seeds": int(num_seeds),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": int(seed),
        "bootstrap_mode": "hierarchical",
    }
