"""Validate revision protocol semantics and local dataset readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import confirmatory_pairs, fairness_metric_contract, seed_tiers


REQUIRED_DATA_FILES = [
    "train.txt",
    "val.txt",
    "test.txt",
    "item2id.json",
    "text_features.npy",
    "vision_features.npy",
    "text_mask.npy",
    "vision_mask.npy",
    "fairness_groups.json",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def validate(config_path: Path) -> dict:
    cfg = load_yaml(config_path)
    errors = []
    warnings = []

    breadth_seeds, confirmatory_seeds = seed_tiers(cfg)
    if len(set(breadth_seeds)) < 3:
        errors.append("The breadth tier requires at least three distinct seeds.")
    if len(set(confirmatory_seeds)) < 6:
        errors.append("The confirmatory tier requires at least six distinct paired seeds.")
    if not set(breadth_seeds).issubset(confirmatory_seeds):
        errors.append("Breadth seeds must be a subset of confirmatory seeds.")
    scoped_pairs = confirmatory_pairs(cfg)
    if not scoped_pairs:
        errors.append("confirmatory_scope cannot be empty.")
    try:
        metric_contract = fairness_metric_contract(cfg)
    except ValueError as exc:
        errors.append(str(exc))
        metric_contract = {}
    selection_split = str(cfg.get("model_selection", {}).get("split", "")).lower()
    if selection_split != "val":
        errors.append("Model and policy selection must use split=val.")
    if not bool(cfg.get("final_evaluation", {}).get("test_once_after_selection", False)):
        errors.append("Final test evaluation must be explicitly one-shot after selection.")

    gamma_grid = [float(x) for x in cfg.get("exposure_reweighting", {}).get("gamma_grid", [])]
    if 0.0 not in gamma_grid or not gamma_grid or max(gamma_grid) < 0.8:
        errors.append("Gamma grid must include the zero control and extend to at least 0.8.")

    datasets_cfg_path = resolve(cfg.get("datasets_config", "configs/datasets.yaml"))
    datasets_cfg = load_yaml(datasets_cfg_path).get("datasets", {})
    protocol_datasets = [
        *cfg.get("datasets", {}).get("main", []),
        *cfg.get("datasets", {}).get("external", []),
    ]
    dataset_reports = {}
    for name in protocol_datasets:
        entry = datasets_cfg.get(name)
        if not isinstance(entry, dict):
            errors.append(f"Dataset {name} is not registered in {datasets_cfg_path}.")
            dataset_reports[name] = {"complete": False, "missing": ["dataset registration"]}
            continue
        data_dir = resolve(entry.get("path", f"data/Processed_{name}"))
        missing = [filename for filename in REQUIRED_DATA_FILES if not (data_dir / filename).exists()]
        exposure_prior = entry.get("exposure_prior")
        if exposure_prior and not (data_dir / str(exposure_prior)).exists():
            missing.append(str(exposure_prior))
        complete = not missing
        dataset_reports[name] = {
            "path": str(data_dir),
            "platform": entry.get("platform", "amazon"),
            "complete": complete,
            "missing": missing,
        }
        if missing:
            errors.append(f"Dataset {name} is missing required files: {missing}")

    external = cfg.get("datasets", {}).get("external", [])
    if not external:
        errors.append("At least one external non-Amazon dataset is required.")
    elif not any(dataset_reports.get(name, {}).get("platform") != "amazon" for name in external):
        warnings.append("External datasets do not declare a non-Amazon platform.")

    base_jobs = len(protocol_datasets) * len(cfg.get("backbones", [])) * len(breadth_seeds)
    reweight_groups = cfg.get("exposure_reweighting", {}).get("groups", [])
    expected_job_counts = {
        "id_backbones": int(base_jobs),
        "fare_gamma_sweep": int(base_jobs * len(gamma_grid) * len(reweight_groups)),
        "id_reweight_control": int(base_jobs),
        "independent_prior_control": int(base_jobs),
        "fare_modality_debias": int(base_jobs),
        "fairrr_evaluations": int(base_jobs),
        "findrec_tuning_trials": int(
            len(protocol_datasets)
            * len(breadth_seeds)
            * len(cfg.get("findrec_tuning", {}).get("learning_rates", []))
            * len(cfg.get("findrec_tuning", {}).get("weight_decays", []))
        ),
        "selected_fare_final_evaluations": int(base_jobs),
        "selected_findrec_final_evaluations": int(len(protocol_datasets) * len(breadth_seeds)),
        "confirmatory_pairs_per_method": int(len(scoped_pairs) * len(confirmatory_seeds)),
        "confirmatory_extra_seed_pairs_per_method": int(
            len(scoped_pairs) * len(set(confirmatory_seeds) - set(breadth_seeds))
        ),
    }
    breadth_training_jobs = sum(
        expected_job_counts[key]
        for key in (
            "id_backbones",
            "fare_gamma_sweep",
            "id_reweight_control",
            "independent_prior_control",
            "fare_modality_debias",
        )
    )
    confirmatory_extra = expected_job_counts["confirmatory_extra_seed_pairs_per_method"]
    expected_job_counts["total_training_jobs"] = int(breadth_training_jobs + 2 * confirmatory_extra)
    expected_job_counts["total_scheduled_jobs"] = int(
        expected_job_counts["total_training_jobs"]
        + expected_job_counts["fairrr_evaluations"]
        + expected_job_counts["findrec_tuning_trials"]
        + expected_job_counts["selected_fare_final_evaluations"]
        + expected_job_counts["selected_findrec_final_evaluations"]
    )

    return {
        "status": "ok" if not errors else "error",
        "protocol_version": cfg.get("protocol_version"),
        "model_selection_split": selection_split,
        "seeds": {
            "breadth": breadth_seeds,
            "confirmatory": confirmatory_seeds,
        },
        "confirmatory_scope": scoped_pairs,
        "fairness_metric_contract": metric_contract,
        "gamma_grid": gamma_grid,
        "datasets": dataset_reports,
        "expected_job_counts": expected_job_counts,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = validate(resolve(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
