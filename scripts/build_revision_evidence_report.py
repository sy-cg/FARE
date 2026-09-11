#!/usr/bin/env python3
"""Build paper tables, a claim audit, and an experiment point-to-point draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import confirmatory_pairs, load_protocol, seed_tiers
from src.revision_reporting import significance_claim_audit


COMMON_METHODS = {
    "ID",
    "FARE",
    "FairRR",
    "ExposureReweight-ID",
    "FARE-IndependentPrior",
    "FARE+ModalityDebias",
}

EFFICIENCY_METHODS = {
    "ID",
    "FARE",
    "ExposureReweight-ID",
    "FARE-IndependentPrior",
    "FARE+ModalityDebias",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _display_results(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    metrics = [f"hr@{k}", f"ndcg@{k}", f"coverage@{k}", f"rec_gini@{k}"]
    metrics.extend(
        metric
        for metric in (f"pop_gap@{k}", f"brand_gap@{k}", f"cluster_gap@{k}")
        if f"{metric}_mean" in frame.columns
    )
    rows = []
    for row in frame.to_dict("records"):
        out = {
            "Dataset": row["dataset"],
            "Backbone": row["backbone"],
            "Method": row["method"],
            "Seeds": int(row["seed_count"]),
        }
        for metric in metrics:
            mean = float(row.get(f"{metric}_mean", float("nan")))
            std = float(row.get(f"{metric}_std", float("nan")))
            out[metric] = f"{mean:.6f} +/- {std:.6f}"
        rows.append(out)
    return pd.DataFrame(rows)


def _latex_table(frame: pd.DataFrame) -> str:
    def esc(value: Any) -> str:
        return str(value).replace("_", "\\_").replace("+", "\\texttt{+}")

    columns = list(frame.columns)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(esc(value) for value in columns) + " \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(esc(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _completeness_issues(
    protocol: dict[str, Any],
    aggregate: pd.DataFrame,
    significance: pd.DataFrame,
    efficiency: pd.DataFrame,
    findrec: pd.DataFrame,
    k: int,
) -> list[str]:
    issues = []
    datasets = [
        *protocol.get("datasets", {}).get("main", []),
        *protocol.get("datasets", {}).get("external", []),
    ]
    backbones = [str(value).lower() for value in protocol.get("backbones", [])]
    breadth, confirmatory = seed_tiers(protocol)
    expected_common = {
        (dataset, backbone, method)
        for dataset in datasets
        for backbone in backbones
        for method in COMMON_METHODS
    }
    observed_common = {
        (str(row.dataset), str(row.backbone), str(row.method))
        for row in aggregate.itertuples()
        if str(row.method) in COMMON_METHODS and int(row.seed_count) >= len(breadth)
    }
    missing_common = sorted(expected_common - observed_common)
    if missing_common:
        issues.append(f"missing breadth result groups: {missing_common[:10]} ({len(missing_common)} total)")
    expected_findrec = set(datasets)
    observed_findrec = set(
        aggregate.loc[
            (aggregate["method"] == "FindRec") & (aggregate["seed_count"] >= len(breadth)),
            "dataset",
        ].astype(str)
    )
    if expected_findrec - observed_findrec:
        issues.append(f"missing FindRec datasets: {sorted(expected_findrec - observed_findrec)}")

    expected_sig = set(confirmatory_pairs(protocol))
    observed_sig = set(
        zip(significance["dataset"].astype(str), significance["backbone"].astype(str))
    )
    if expected_sig != observed_sig:
        issues.append(
            f"significance scope mismatch: missing={sorted(expected_sig - observed_sig)}, "
            f"unexpected={sorted(observed_sig - expected_sig)}"
        )
    if len(significance) and not (significance["num_seeds"].astype(int) == len(confirmatory)).all():
        issues.append("significance rows do not all use the confirmatory seed count")
    required_efficiency_columns = {"dataset", "backbone", "method", "seed"}
    missing_efficiency_columns = required_efficiency_columns - set(efficiency.columns)
    if missing_efficiency_columns:
        issues.append(
            "efficiency.csv missing columns: " + str(sorted(missing_efficiency_columns))
        )
    else:
        expected_efficiency = {
            (dataset, backbone, method)
            for dataset in datasets
            for backbone in backbones
            for method in EFFICIENCY_METHODS
        }
        observed_efficiency = set()
        for keys, rows in efficiency.groupby(["dataset", "backbone", "method"], sort=True):
            dataset, backbone, method = (str(keys[0]), str(keys[1]), str(keys[2]))
            if method in EFFICIENCY_METHODS and rows["seed"].nunique() >= len(breadth):
                observed_efficiency.add((dataset, backbone, method))
        missing_efficiency = sorted(expected_efficiency - observed_efficiency)
        if missing_efficiency:
            issues.append(
                f"missing efficiency result groups: {missing_efficiency[:10]} "
                f"({len(missing_efficiency)} total)"
            )
    expected_findrec_runs = len(datasets) * len(breadth)
    if len(findrec) != expected_findrec_runs:
        issues.append(f"FindRec diagnostics have {len(findrec)} rows; expected {expected_findrec_runs}")
    width_col = "ranking/topk_width"
    if width_col in findrec and len(findrec) and not (findrec[width_col].astype(int) == k).all():
        issues.append(f"FindRec diagnostics are not uniformly Top-{k}")
    return issues


def _fairrr_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _fairrr_formal_pass_mask(fairrr: pd.DataFrame) -> pd.Series:
    rank_rate = _fairrr_numeric(fairrr, "fairrr_rank_changed_user_rate")
    set_rate = _fairrr_numeric(fairrr, "fairrr_set_changed_user_rate")
    min_rank = _fairrr_numeric(fairrr, "fairrr_min_rank_changed_user_rate")
    min_set = _fairrr_numeric(fairrr, "fairrr_min_set_changed_user_rate")
    utility_retention = _fairrr_numeric(
        fairrr, "fairrr_validation_utility_retention"
    )
    max_utility_loss = _fairrr_numeric(
        fairrr, "fairrr_max_relative_utility_loss"
    )
    fairness_improvement = _fairrr_numeric(
        fairrr, "fairrr_validation_fairness_improvement"
    )
    min_fairness_improvement = _fairrr_numeric(
        fairrr, "fairrr_min_fairness_improvement"
    )
    if "fairrr_selection_reason" in fairrr.columns:
        valid_reason = fairrr["fairrr_selection_reason"].astype(str).eq(
            "best_validation_fairness_under_constraints"
        )
    else:
        valid_reason = pd.Series(False, index=fairrr.index, dtype=bool)
    tolerance = 1e-12
    return (
        valid_reason
        & rank_rate.notna()
        & set_rate.notna()
        & min_rank.notna()
        & min_set.notna()
        & (rank_rate + tolerance >= min_rank)
        & (set_rate + tolerance >= min_set)
        & utility_retention.notna()
        & max_utility_loss.between(0.0, 1.0, inclusive="both")
        & (utility_retention + tolerance >= 1.0 - max_utility_loss)
        & fairness_improvement.notna()
        & min_fairness_improvement.notna()
        & (fairness_improvement > 0.0)
        & (fairness_improvement + tolerance >= min_fairness_improvement)
    )


def _fairrr_examples(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row.get('dataset', '?')}/{row.get('backbone', '?')}/s{row.get('seed', '?')}"
        for row in frame.head(10).to_dict("records")
    ]


def _fairrr_change_issues(per_run: pd.DataFrame) -> list[str]:
    if "method" not in per_run.columns:
        return ["per-run results missing method column"]
    fairrr = per_run[per_run["method"].astype(str).eq("FairRR")].copy()
    if fairrr.empty:
        return []

    issues = []
    rank_rate = _fairrr_numeric(fairrr, "fairrr_rank_changed_user_rate")
    nonpositive = fairrr.loc[rank_rate.isna() | (rank_rate <= 0.0)]
    if len(nonpositive):
        issues.append(
            "FairRR missing/nonpositive test rank-change audit: "
            f"{_fairrr_examples(nonpositive)} ({len(nonpositive)} total)"
        )

    set_rate = _fairrr_numeric(fairrr, "fairrr_set_changed_user_rate")
    min_rank = _fairrr_numeric(fairrr, "fairrr_min_rank_changed_user_rate")
    min_set = _fairrr_numeric(fairrr, "fairrr_min_set_changed_user_rate")
    tolerance = 1e-12
    below_change_threshold = fairrr.loc[
        rank_rate.isna()
        | set_rate.isna()
        | min_rank.isna()
        | min_set.isna()
        | (rank_rate + tolerance < min_rank)
        | (set_rate + tolerance < min_set)
    ]
    if len(below_change_threshold):
        issues.append(
            "FairRR test intervention is below formal change threshold or lacks its audit: "
            f"{_fairrr_examples(below_change_threshold)} "
            f"({len(below_change_threshold)} total)"
        )

    selection_columns = [
        "fairrr_selection_reason",
        "fairrr_validation_utility_retention",
        "fairrr_max_relative_utility_loss",
        "fairrr_validation_fairness_improvement",
        "fairrr_min_fairness_improvement",
    ]
    selection_missing = any(column not in fairrr.columns for column in selection_columns)
    if selection_missing:
        invalid_selection = fairrr
    else:
        reason = fairrr["fairrr_selection_reason"].astype(str)
        utility_retention = _fairrr_numeric(
            fairrr, "fairrr_validation_utility_retention"
        )
        max_utility_loss = _fairrr_numeric(
            fairrr, "fairrr_max_relative_utility_loss"
        )
        fairness_improvement = _fairrr_numeric(
            fairrr, "fairrr_validation_fairness_improvement"
        )
        min_fairness_improvement = _fairrr_numeric(
            fairrr, "fairrr_min_fairness_improvement"
        )
        invalid_selection = fairrr.loc[
            ~reason.eq("best_validation_fairness_under_constraints")
            | utility_retention.isna()
            | ~max_utility_loss.between(0.0, 1.0, inclusive="both")
            | (utility_retention + tolerance < 1.0 - max_utility_loss)
            | fairness_improvement.isna()
            | min_fairness_improvement.isna()
            | (fairness_improvement <= 0.0)
            | (fairness_improvement + tolerance < min_fairness_improvement)
        ]
    if len(invalid_selection):
        issues.append(
            "FairRR validation selection audit is missing or violates its utility/exposure constraints: "
            f"{_fairrr_examples(invalid_selection)} ({len(invalid_selection)} total)"
        )
    return issues

def _point_to_point(
    audit: dict[str, Any],
    fairrr_changed: int,
    fairrr_total: int,
    findrec: pd.DataFrame,
    efficiency: pd.DataFrame,
    status: str,
) -> str:
    utility = audit["utility"]
    fairness = audit["fairness"]
    diagnoses = findrec["diagnosis"].value_counts().to_dict() if len(findrec) else {}
    return f"""# Experimental comments: point-to-point response draft

Evidence status: **{status}**. This draft is generated from test-only Top-10 summaries and paired-seed statistics. It should be submitted only when the status is `complete`.

## Reviewer 5, Comment 1: wider exposure hyperparameter analysis

Response: We expanded the exposure-weight grid to include 0.0, 0.02, 0.05, 0.1, 0.2, 0.4, and 0.8 and report the validation-only policy-sensitivity analysis. The test split is evaluated only after the policy is frozen.

## Reviewer 5, Comment 2: reproducibility details

Response: We now report backbone-specific architecture overrides, including the one-layer GRU4Rec configuration, the validation-selected exposure group and weight for each dataset-backbone pair, the breadth/confirmatory seed tiers, and checkpoint coverage audits.

## Reviewer 6, Comment 1 and Reviewer 3, Comment 1: significance and confidence intervals

Response: We added paired six-seed inference for {audit['confirmatory_pairs']} dataset-backbone pairs. Among {utility['outcomes']} utility outcomes, {utility['positive_estimates']} have positive mean differences and {utility['positive_cis']} have 95% CIs strictly above zero. Among {fairness['outcomes']} exposure outcomes (with all signs oriented so that positive means improvement), {fairness['positive_estimates']} have positive means and {fairness['positive_cis']} have CIs strictly above zero. We therefore report setting-specific effects rather than a universal improvement claim.

## Reviewer 6, Comment 2 and Reviewer 3, Comment 3: FindRec validity

Response: We added validation-only learning-rate/weight-decay tuning, learning-curve diagnostics, and a frozen-checkpoint final test. The Top-10 diagnostic counts are {json.dumps(diagnoses, ensure_ascii=False)}; the paper reports these diagnostics and no longer uses Top-100 coverage to characterize FindRec@10.

## Reviewer 6, Comments 3-4: selection policy and leakage

Response: The primary policy is fixed at equal utility/exposure weighting (alpha=0.5), with alternative alpha values reported as validation-only sensitivity checks. Gamma and exposure-control group selection use validation artifacts aggregated across breadth seeds; the selected policy is frozen before one-shot test evaluation.

## Reviewer 6, Comment 5: ID reweighting control

Response: We added ExposureReweight-ID, which applies the same validation-selected exposure weights directly to a trainable ID backbone while disabling the multimodal residual. Its Top-10 utility and exposure results are reported beside ID and FARE.

## Reviewer 6, Comment 6: independent exposure prior

Response: We added FARE-IndependentPrior, using train popularity for Amazon and platform-view exposure for MicroLens. This separates the contribution of the exposure source from the residual architecture.

## Reviewer 6, Comment 7 and Reviewer 5, Comment 3: cross-platform evidence

Response: We added MicroLens-100K, which differs from Amazon in platform, interaction behavior, and exposure structure, and evaluate SASRec, GRU4Rec, and BERT4Rec under the same text-and-image protocol. The conclusion is restricted to the three tested Amazon categories and one non-Amazon platform, MicroLens-100K, rather than broad universal generalization.

## Reviewer 6, Comment 8: FARE plus ModalityDebias

Response: We added the FARE+ModalityDebias composition with strict source-checkpoint auditing and report its Top-10 utility and exposure results as a composition study, without presuming that the two methods are automatically complementary.

## Reviewer 3, Comment 2: FairRR identical outputs

Response: FairRR now uses a verified Top-100 candidate pool and selects lambda only on validation by minimizing discounted exposure deviation subject to a shared utility-retention constraint and material rank/set-change thresholds. {fairrr_changed} of {fairrr_total} completed runs satisfy the full validation-selection and test-intervention audit; infeasible runs fail rather than being reported as ID-identical results.

## Reviewer 3, Comments 4-5: strength of the core claim

Response: We replaced the blanket phrase "improves exposure fairness without sacrificing utility" with the evidence-bounded conclusion that FARE provides a validation-selected utility-exposure trade-off whose direction and statistical certainty vary by dataset, backbone, and exposure endpoint.

## Reviewer 3, Comment 7: efficiency

Response: We added parameter counts, train time, inference latency/throughput, peak CUDA memory, software/hardware versions, and exposure-weight construction time. The efficiency table contains {len(efficiency)} run-level measurements and retains dataset, backbone, method, and seed identifiers.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--results-aggregate", required=True)
    parser.add_argument("--results-per-run", required=True)
    parser.add_argument("--significance", required=True)
    parser.add_argument("--efficiency", required=True)
    parser.add_argument("--findrec-diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    aggregate = pd.read_csv(args.results_aggregate)
    per_run = pd.read_csv(args.results_per_run)
    significance = pd.read_csv(args.significance)
    efficiency = pd.read_csv(args.efficiency)
    findrec = pd.read_csv(args.findrec_diagnostics)
    issues = _completeness_issues(
        protocol, aggregate, significance, efficiency, findrec, args.k
    )
    issues.extend(_fairrr_change_issues(per_run))
    status = "complete" if not issues else "incomplete"
    claim_audit = significance_claim_audit(significance)
    claim_audit.update({"status": status, "issues": issues})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    display = _display_results(aggregate, args.k)
    (output / "revision_main_table.md").write_text(
        "# Revision test results\n\n" + _markdown_table(display) + "\n",
        encoding="utf-8",
    )
    (output / "revision_main_table.tex").write_text(
        _latex_table(display), encoding="utf-8"
    )

    efficiency_grouped = (
        efficiency.groupby(["dataset", "backbone", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            total_parameters=("total_parameters", "mean"),
            trainable_parameters=("trainable_parameters", "mean"),
            total_train_sec=("total_train_sec", "mean"),
            test_ms_per_user=("test_ms_per_user", "mean"),
            peak_cuda_memory_mb=("peak_cuda_memory_mb", "mean"),
            exposure_weight_build_sec=("exposure_weight_build_sec", "mean"),
        )
        if "backbone" in efficiency.columns
        else pd.DataFrame()
    )
    (output / "revision_efficiency_table.md").write_text(
        "# Revision efficiency results\n\n"
        + (_markdown_table(efficiency_grouped.round(4)) if len(efficiency_grouped) else "Incomplete.")
        + "\n",
        encoding="utf-8",
    )
    (output / "claim_audit.json").write_text(
        json.dumps(claim_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fairrr = per_run[per_run["method"].astype(str).eq("FairRR")].copy()
    changed = int(_fairrr_formal_pass_mask(fairrr).sum())
    total = int(len(fairrr))
    (output / "experimental_comments_point_by_point_response.md").write_text(
        _point_to_point(claim_audit, changed, total, findrec, efficiency, status),
        encoding="utf-8",
    )
    print(json.dumps(claim_audit, indent=2, ensure_ascii=False))
    if issues and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
