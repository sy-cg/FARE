#!/usr/bin/env python3
"""Generate revision training jobs plus significance and efficiency manifests."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from itertools import product
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import (
    confirmatory_pairs,
    fare_config_for_dataset,
    fare_run_name_for_dataset,
    load_protocol,
    model_overrides_for_backbone,
    seed_tiers,
    tagged_run_id,
)


EFFICIENCY_JOB_METHODS = {
    "ExposureReweight-ID",
    "FARE-IndependentPrior",
    "FARE+ModalityDebias",
}


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _require_csv(path: Path, label: str, hint: str) -> Path:
    if path.exists():
        return path
    raise SystemExit(
        f"{label} not found: {path}\n"
        f"Generate it first, then rerun build_revision_jobs.py. Suggested command:\n"
        f"{hint}"
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _argv(*parts) -> str:
    return json.dumps([str(part) for part in parts], ensure_ascii=False)


def _selected_policy(row: dict[str, str]) -> tuple[float, str]:
    if row.get("fair_rec_reweight_weight") and row.get("fair_rec_reweight_groups"):
        return float(row["fair_rec_reweight_weight"]), row["fair_rec_reweight_groups"]
    values = {}
    for part in str(row.get("config_id", "")).split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key.strip()] = value.strip()
    gamma = values.get("gamma", values.get("fair_rec_reweight_weight"))
    group = values.get("group", values.get("fair_rec_reweight_groups"))
    if gamma is None or not group:
        raise ValueError(
            "Selected FARE rows require config_id='gamma=<x>|group=<name>' or explicit policy columns"
        )
    return float(gamma), str(group)


def _model_override_argv(protocol: dict, backbone: str) -> tuple[object, ...]:
    parts: list[object] = []
    for key, value in model_overrides_for_backbone(protocol, backbone).items():
        parts.extend((f"--{key}", value))
    return tuple(parts)


def _primary_sweep(
    output: Path,
    protocol: dict,
    row: dict[str, str],
) -> Path:
    dataset = row["dataset"]
    backbone = row["backbone"].lower()
    seed = int(row["seed"])
    slug = tagged_run_id(protocol, backbone, f"fare_gamma_{dataset}_{backbone}_s{seed}")
    exposure = protocol.get("exposure_reweighting", {})
    metric = str(protocol.get("model_selection", {}).get("metric", "ndcg@10"))
    run_name = fare_run_name_for_dataset(protocol, dataset, ROOT)
    sweep = {
        "name": slug,
        "script": "scripts/run_fare.py",
        "sampler": "grid",
        "seed": seed,
        "static_args": {
            "config": fare_config_for_dataset(protocol, dataset),
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "init_backbone_checkpoint": str(_resolve(row["init_backbone_checkpoint"])),
            "fair_rec_exposure_source": "reference_topk",
            "fair_rec_exposure_topk_path": str(_resolve(row["reference_topk_path"])),
            "skip_test_after_training": True,
            **model_overrides_for_backbone(protocol, backbone),
        },
        "search_space": {
            "fair_rec_reweight_weight": [float(value) for value in exposure.get("gamma_grid", [])],
            "fair_rec_reweight_groups": list(exposure.get("groups", [])),
        },
        "run_id": {
            "prefix": slug,
            "include_params": ["fair_rec_reweight_weight", "fair_rec_reweight_groups"],
        },
        "metric": {
            "key": f"val.{metric}",
            "mode": "max",
            "file_template": f"results/{{dataset}}/{run_name}/{{run_id}}/metrics_summary.json",
        },
    }
    path = output / "sweeps" / f"{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(sweep, sort_keys=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/revision/revision_protocol.yaml")
    parser.add_argument("--id-registry", required=True)
    parser.add_argument("--selected-fare-registry", default=None)
    parser.add_argument("--selected-findrec-registry", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = load_protocol(protocol_path)
    breadth_seeds, confirmatory_seeds = seed_tiers(protocol)
    datasets = [
        *protocol.get("datasets", {}).get("main", []),
        *protocol.get("datasets", {}).get("external", []),
    ]
    backbones = [str(value).lower() for value in protocol.get("backbones", [])]
    method_run_tags = protocol.get("run_tag_by_method", {}) or {}
    if not isinstance(method_run_tags, dict):
        raise ValueError("run_tag_by_method must be a mapping")
    fairrr_run_tag = str(method_run_tags.get("FairRR", "")).strip()
    scoped_pairs = set(confirmatory_pairs(protocol))
    output = _resolve(args.output_dir, Path.cwd())
    output.mkdir(parents=True, exist_ok=True)

    id_registry_path = _resolve(args.id_registry, Path.cwd())
    id_rows = _read_csv(
        _require_csv(
            id_registry_path,
            "ID registry",
            "python scripts/collect_id_checkpoint_registry.py "
            "--results-root /root/autodl-tmp/results_sdr "
            "--datasets-config configs/datasets.yaml "
            "--output revision_outputs/id_checkpoints_all_seeds_corrected.csv",
        )
    )
    id_map = {
        (row["dataset"], row["backbone"].lower(), int(row["seed"])): row
        for row in id_rows
    }
    selected_rows = (
        _read_csv(
            _require_csv(
                _resolve(args.selected_fare_registry, Path.cwd()),
                "Selected FARE registry",
                "python scripts/materialize_selected_runs.py "
                "--selected revision_outputs/selected_configs_corrected.csv "
                "--run-registry revision_outputs/fare_validation_runs_corrected.csv "
                "--output revision_outputs/selected_seed_runs_corrected.csv",
            )
        )
        if args.selected_fare_registry
        else []
    )
    fare_map = {
        (row["dataset"], row["backbone"].lower(), int(row["seed"])): row
        for row in selected_rows
    }
    selected_findrec_rows = (
        _read_csv(
            _require_csv(
                _resolve(args.selected_findrec_registry, Path.cwd()),
                "Selected FindRec registry",
                "python scripts/collect_findrec_tuning.py "
                "--sweep-results sweeps/revision/findrec/*/sweep_results.csv "
                "--trials-out revision_outputs/findrec_tuning_trials.csv "
                "--selected-out revision_outputs/findrec_selected.csv "
                "--audit-out revision_outputs/findrec_tuning_audit.json",
            )
        )
        if args.selected_findrec_registry
        else []
    )

    jobs = []
    expected_breadth = set(product(datasets, backbones, breadth_seeds))
    for key in sorted(expected_breadth & set(id_map)):
        dataset, backbone, seed = key
        id_row = id_map[key]
        fairrr_run_id = f"revision_fairrr_{dataset}_{backbone}_s{seed}"
        if fairrr_run_tag:
            fairrr_run_id = f"{fairrr_run_id}_{fairrr_run_tag}"
        sweep_path = _primary_sweep(output, protocol, id_row)
        jobs.append(
            {
                "phase": "breadth_selection",
                "method": "FARE-Sweep",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": "ready",
                "argv_json": _argv("python", "scripts/run_hparam_sweep.py", "--config", sweep_path, "--resume", "--continue_on_error"),
                "expected_run_dir": "",
            }
        )
        jobs.append(
            {
                "phase": "breadth_control",
                "method": "FairRR",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": "ready",
                "argv_json": _argv(
                    "python",
                    "scripts/run_reranker_fair.py",
                    "--config",
                    "configs/reranker_fair_3090.yaml",
                    "--dataset",
                    dataset,
                    "--backbone",
                    backbone,
                    "--source_run_dir",
                    id_row["run_dir"],
                    "--seed",
                    seed,
                    "--split",
                    "all",
                    "--run_id",
                    fairrr_run_id,
                ),
                "expected_run_dir": str(
                    (
                        ROOT
                        / "results"
                        / dataset
                        / f"{backbone}_fair_rerank"
                        / fairrr_run_id
                    ).resolve()
                ),
            }
        )

    findrec_template_path = ROOT / "configs" / "revision" / "findrec_tuning_3090.yaml"
    findrec_template = yaml.safe_load(findrec_template_path.read_text(encoding="utf-8"))
    findrec_protocol = protocol.get("findrec_tuning", {})
    if findrec_protocol:
        findrec_template["search_space"]["lr"] = [
            float(value) for value in findrec_protocol.get("learning_rates", [])
        ]
        findrec_template["search_space"]["weight_decay"] = [
            float(value) for value in findrec_protocol.get("weight_decays", [])
        ]
        if not bool(findrec_protocol.get("validation_only", False)):
            raise ValueError("FindRec tuning must be validation_only")
    for dataset, seed in sorted(product(datasets, breadth_seeds)):
        slug = f"findrec_tuning_{dataset}_s{seed}"
        sweep = copy.deepcopy(findrec_template)
        sweep["name"] = slug
        sweep["seed"] = int(seed)
        sweep["static_args"]["dataset"] = dataset
        sweep["static_args"]["seed"] = int(seed)
        sweep["run_id"]["prefix"] = slug
        path = output / "sweeps" / f"{slug}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(sweep, sort_keys=False), encoding="utf-8")
        jobs.append(
            {
                "phase": "breadth_diagnostic_tuning",
                "method": "FindRec-Tuning",
                "dataset": dataset,
                "backbone": "n/a",
                "seed": seed,
                "status": "ready",
                "argv_json": _argv(
                    "python",
                    "scripts/run_hparam_sweep.py",
                    "--config",
                    path,
                    "--continue_on_error",
                ),
                "expected_run_dir": "",
            }
        )

    for row in selected_findrec_rows:
        dataset = str(row["dataset"])
        seed = int(row["seed"])
        run_dir = _resolve(row["run_dir"])
        test_ranking = run_dir / "test_ranking.npz"
        test_audit = run_dir / "test_evaluation_audit.json"
        if test_ranking.exists() and test_audit.exists():
            status = "complete_existing"
        elif test_ranking.exists():
            status = "blocked_existing_unaudited_test"
        elif (run_dir / "best_model.pt").exists() and (run_dir / "config_resolved.json").exists():
            status = "ready"
        else:
            status = "blocked_missing_selected_checkpoint"
        jobs.append(
            {
                "phase": "breadth_final_test",
                "method": "FindRec-Selected-Test",
                "dataset": dataset,
                "backbone": "findrec",
                "seed": seed,
                "status": status,
                "argv_json": _argv(
                    "python",
                    "scripts/evaluate_selected_findrec.py",
                    "--run-dir",
                    run_dir,
                    "--device",
                    "cuda",
                ),
                "expected_run_dir": str(run_dir),
            }
        )

    extra_confirmatory = set(confirmatory_seeds) - set(breadth_seeds)
    for dataset, backbone in sorted(scoped_pairs):
        for seed in sorted(extra_confirmatory):
            key = (dataset, backbone, seed)
            if key in id_map:
                continue
            jobs.append(
                {
                    "phase": "confirmatory_dependency",
                    "method": "ID",
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": seed,
                    "status": "ready",
                    "argv_json": _argv(
                        "python",
                        "scripts/run_id_backbone.py",
                        "--model",
                        backbone,
                        "--config",
                        f"configs/{backbone}_3090.yaml",
                        "--dataset",
                        dataset,
                        "--seed",
                        seed,
                        "--run_id",
                        f"revision_confirmatory_{dataset}_{backbone}_s{seed}",
                    ),
                    "expected_run_dir": "",
                }
            )

    source_by_platform = protocol.get("exposure_reweighting", {}).get("source_by_platform", {})
    for key in sorted(expected_breadth & set(id_map) & set(fare_map)):
        dataset, backbone, seed = key
        id_row = id_map[key]
        fare_row = fare_map[key]
        gamma, group_name = _selected_policy(fare_row)
        checkpoint = _resolve(id_row["init_backbone_checkpoint"])
        reference_topk = _resolve(id_row["reference_topk_path"])
        fare_run_dir = _resolve(fare_row["run_dir"])
        fare_checkpoint = fare_run_dir / "best_model.pt"
        fare_config = fare_run_dir / "config_resolved.json"
        fare_test = fare_run_dir / "topk_test.npz"
        fare_test_audit = fare_run_dir / "test_evaluation_audit.json"
        if fare_test.exists() and fare_test_audit.exists():
            selected_test_status = "complete_existing"
        elif fare_test.exists():
            selected_test_status = "blocked_existing_unaudited_test"
        elif fare_checkpoint.exists() and fare_config.exists():
            selected_test_status = "ready"
        else:
            selected_test_status = "blocked_missing_selected_checkpoint"
        jobs.append(
            {
                "phase": "breadth_final_test",
                "method": "FARE-Selected-Test",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": selected_test_status,
                "argv_json": _argv(
                    "python",
                    "scripts/evaluate_selected_fare.py",
                    "--run-dir",
                    fare_run_dir,
                    "--device",
                    "cuda",
                ),
                "expected_run_dir": str(fare_run_dir),
            }
        )
        idrw_run_id = tagged_run_id(
            protocol, backbone, f"revision_idrw_{dataset}_{backbone}_s{seed}"
        )
        prior_run_id = tagged_run_id(
            protocol, backbone, f"revision_prior_{dataset}_{backbone}_s{seed}"
        )
        fare_md_run_id = tagged_run_id(
            protocol, backbone, f"revision_fare_md_{dataset}_{backbone}_s{seed}"
        )
        common = (
            "--dataset", dataset,
            "--backbone", backbone,
            "--seed", seed,
            "--init_backbone_checkpoint", checkpoint,
            "--fair_rec_reweight_weight", gamma,
            "--fair_rec_reweight_groups", group_name,
            *_model_override_argv(protocol, backbone),
        )
        jobs.append(
            {
                "phase": "breadth_control",
                "method": "ExposureReweight-ID",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": "ready",
                "argv_json": _argv(
                    "python", "scripts/run_fare.py", "--config",
                    "configs/revision/fare_id_exposure_reweight_3090.yaml",
                    *common,
                    "--fair_rec_exposure_source", "reference_topk",
                    "--fair_rec_exposure_topk_path", reference_topk,
                    "--run_id", idrw_run_id,
                ),
                "expected_run_dir": str((ROOT / "results" / dataset / "fare_id_exposure_reweight" / idrw_run_id).resolve()),
            }
        )
        prior_source = source_by_platform.get(id_row.get("platform", "amazon"), "train_popularity")
        jobs.append(
            {
                "phase": "breadth_control",
                "method": "FARE-IndependentPrior",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": "ready",
                "argv_json": _argv(
                    "python", "scripts/run_fare.py", "--config",
                    "configs/revision/exposure_prior_control_3090.yaml",
                    *common,
                    "--fair_rec_exposure_source", prior_source,
                    "--run_id", prior_run_id,
                ),
                "expected_run_dir": str((ROOT / "results" / dataset / "fare_train_prior" / prior_run_id).resolve()),
            }
        )
        jobs.append(
            {
                "phase": "breadth_composition",
                "method": "FARE+ModalityDebias",
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "status": "ready" if fare_checkpoint.exists() else "blocked_missing_fare_checkpoint",
                "argv_json": _argv(
                    "python", "scripts/run_modality_debias.py", "--config",
                    "configs/revision/fare_md_3090.yaml",
                    "--dataset", dataset,
                    "--base_model", f"fare_{backbone}",
                    "--seed", seed,
                    "--init_base_checkpoint", fare_checkpoint,
                    "--freeze_base",
                    *_model_override_argv(protocol, backbone),
                    "--run_id", fare_md_run_id,
                ),
                "expected_run_dir": str((ROOT / "results" / dataset / "fare_modality_debias" / fare_md_run_id).resolve()),
            }
        )

    generated_confirmatory_fare = {}
    for dataset, backbone in sorted(scoped_pairs):
        policy_rows = [
            row
            for (row_dataset, row_backbone, _), row in fare_map.items()
            if row_dataset == dataset and row_backbone == backbone
        ]
        if not policy_rows:
            continue
        policies = {_selected_policy(row) for row in policy_rows}
        if len(policies) != 1:
            raise ValueError(
                f"Selected FARE policy is not frozen for {dataset}/{backbone}: {sorted(policies)}"
            )
        gamma, group_name = next(iter(policies))
        config_path = fare_config_for_dataset(protocol, dataset)
        run_name = fare_run_name_for_dataset(protocol, dataset, ROOT)
        for seed in sorted(extra_confirmatory):
            key = (dataset, backbone, seed)
            if key not in id_map or key in fare_map:
                continue
            id_row = id_map[key]
            run_id = tagged_run_id(
                protocol,
                backbone,
                f"revision_confirmatory_fare_{dataset}_{backbone}_s{seed}",
            )
            expected_run = (ROOT / "results" / dataset / run_name / run_id).resolve()
            checkpoint = _resolve(id_row["init_backbone_checkpoint"])
            reference_topk = _resolve(id_row["reference_topk_path"])
            status = "ready" if checkpoint.exists() and reference_topk.exists() else "blocked_missing_id_artifacts"
            jobs.append(
                {
                    "phase": "confirmatory",
                    "method": "FARE-Confirmatory",
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": seed,
                    "status": status,
                    "argv_json": _argv(
                        "python", "scripts/run_fare.py",
                        "--config", config_path,
                        "--dataset", dataset,
                        "--backbone", backbone,
                        "--seed", seed,
                        "--init_backbone_checkpoint", checkpoint,
                        "--fair_rec_exposure_source", "reference_topk",
                        "--fair_rec_exposure_topk_path", reference_topk,
                        "--fair_rec_reweight_weight", gamma,
                        "--fair_rec_reweight_groups", group_name,
                        *_model_override_argv(protocol, backbone),
                        "--run_id", run_id,
                    ),
                    "expected_run_dir": str(expected_run),
                }
            )
            generated_confirmatory_fare[key] = {
                "dataset": dataset,
                "backbone": backbone,
                "seed": str(seed),
                "config_id": policy_rows[0].get("config_id", ""),
                "run_dir": str(expected_run),
            }
    dataset_cfg_path = _resolve(protocol.get("datasets_config", "configs/datasets.yaml"))
    dataset_cfg = yaml.safe_load(dataset_cfg_path.read_text(encoding="utf-8"))["datasets"]
    significance = []
    efficiency = []
    analysis_fare_map = dict(fare_map)
    analysis_fare_map.update(generated_confirmatory_fare)
    for key in sorted(set(id_map) & set(analysis_fare_map)):
        dataset, backbone, seed = key
        id_run = _resolve(id_map[key]["run_dir"])
        fare_run = _resolve(analysis_fare_map[key]["run_dir"])
        for method, run in (("ID", id_run), ("FARE", fare_run)):
            efficiency.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run),
                    "ready": (run / "metrics_summary.json").exists(),
                }
            )
        if (dataset, backbone) not in scoped_pairs or seed not in confirmatory_seeds:
            continue
        base_npz = id_run / "topk_test.npz"
        method_npz = fare_run / "topk_test.npz"
        data_path = _resolve(dataset_cfg[dataset]["path"])
        significance.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "baseline_npz": str(base_npz),
                "method_npz": str(method_npz),
                "data_dir": str(data_path),
                "ready": base_npz.exists() and method_npz.exists(),
            }
        )

    for job in jobs:
        method = str(job.get("method", ""))
        status = str(job.get("status", "")).lower()
        if method not in EFFICIENCY_JOB_METHODS or status not in {"ready", "complete_existing"}:
            continue
        run_dir = str(job.get("expected_run_dir", "")).strip()
        if not run_dir:
            continue
        run = _resolve(run_dir)
        efficiency.append(
            {
                "dataset": job["dataset"],
                "backbone": job["backbone"],
                "method": method,
                "seed": job["seed"],
                "run_dir": str(run),
                "ready": (run / "metrics_summary.json").exists(),
            }
        )

    job_fields = ["phase", "method", "dataset", "backbone", "seed", "status", "argv_json", "expected_run_dir"]
    sig_fields = ["dataset", "backbone", "seed", "baseline_npz", "method_npz", "data_dir", "ready"]
    eff_fields = ["dataset", "backbone", "method", "seed", "run_dir", "ready"]
    _write_csv(output / "jobs.csv", jobs, job_fields)
    _write_csv(output / "significance_manifest.csv", significance, sig_fields)
    _write_csv(output / "efficiency_manifest.csv", efficiency, eff_fields)

    audit = {
        "protocol": str(protocol_path),
        "seed_tiers": {"breadth": breadth_seeds, "confirmatory": confirmatory_seeds},
        "confirmatory_scope": sorted([list(pair) for pair in scoped_pairs]),
        "breadth_id_registry_missing": [list(key) for key in sorted(expected_breadth - set(id_map))],
        "jobs": len(jobs),
        "significance_pairs": len(significance),
        "significance_ready_pairs": sum(bool(row["ready"]) for row in significance),
        "efficiency_runs": len(efficiency),
        "selected_fare_registry_supplied": bool(args.selected_fare_registry),
        "selected_findrec_registry_supplied": bool(args.selected_findrec_registry),
        "selected_findrec_final_jobs": len(selected_findrec_rows),
        "confirmatory_fare_jobs": len(generated_confirmatory_fare),
    }
    (output / "generation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
