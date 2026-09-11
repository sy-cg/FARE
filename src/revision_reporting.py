"""Evidence summaries used by the IPM revision report generator."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _direction_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "outcomes": int(len(frame)),
        "positive_estimates": int((frame["estimate"] > 0).sum()),
        "negative_estimates": int((frame["estimate"] < 0).sum()),
        "positive_cis": int((frame["ci_low"] > 0).sum()),
        "negative_cis": int((frame["ci_high"] < 0).sum()),
        "cis_crossing_zero": int(((frame["ci_low"] <= 0) & (frame["ci_high"] >= 0)).sum()),
    }


def significance_claim_audit(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"dataset", "backbone", "metric", "estimate", "ci_low", "ci_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Significance table missing columns: {sorted(missing)}")
    utility_mask = frame["metric"].astype(str).str.startswith(("hr@", "ndcg@"))
    utility = _direction_counts(frame[utility_mask])
    fairness = _direction_counts(frame[~utility_mask])
    return {
        "confirmatory_pairs": int(frame[["dataset", "backbone"]].drop_duplicates().shape[0]),
        "utility": utility,
        "fairness": fairness,
        "supports_uniform_no_utility_sacrifice": bool(
            utility["outcomes"] > 0
            and utility["negative_estimates"] == 0
            and utility["positive_cis"] == utility["outcomes"]
        ),
        "supports_uniform_fairness_improvement": bool(
            fairness["outcomes"] > 0
            and fairness["negative_estimates"] == 0
            and fairness["positive_cis"] == fairness["outcomes"]
        ),
    }
