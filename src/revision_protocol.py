"""Shared protocol helpers for IPM revision scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROTOCOL_PATH = Path("configs/revision/revision_protocol.yaml")

DEFAULT_FAIRNESS_METRICS = {
    "coverage": {
        "artifact_key": "summary@{k}/catalog_coverage",
        "direction": "higher",
    },
    "rec_gini": {
        "artifact_key": "summary@{k}/recommendation_gini",
        "direction": "lower",
    },
    "pop_gap": {
        "group": "popularity_group",
        "direction": "lower",
    },
    "brand_gap": {
        "group": "brand_store_proxy_group",
        "direction": "lower",
    },
    "cluster_gap": {
        "group": "multimodal_cluster_proxy_group",
        "direction": "lower",
    },
}


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Protocol root must be a mapping: {path}")
    return value


def seed_tiers(protocol: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Return breadth and confirmatory seeds, accepting the legacy list form."""
    configured = protocol.get("seeds", [])
    if isinstance(configured, dict):
        breadth = [int(value) for value in configured.get("breadth", [])]
        confirmatory = [int(value) for value in configured.get("confirmatory", [])]
    else:
        breadth = [int(value) for value in configured]
        confirmatory = list(breadth)
    return breadth, confirmatory


def confirmatory_pairs(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    scope = protocol.get("confirmatory_scope", {})
    if not isinstance(scope, dict):
        raise ValueError("confirmatory_scope must map datasets to backbone lists")
    return sorted(
        (str(dataset), str(backbone).lower())
        for dataset, backbones in scope.items()
        for backbone in (backbones or [])
    )


def model_overrides_for_backbone(
    protocol: dict[str, Any],
    backbone: str,
) -> dict[str, Any]:
    configured = protocol.get("model_overrides_by_backbone", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("model_overrides_by_backbone must be a mapping")
    overrides = configured.get(str(backbone).lower(), {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Model overrides for {backbone} must be a mapping")
    return dict(overrides)


def tagged_run_id(protocol: dict[str, Any], backbone: str, base: str) -> str:
    configured = protocol.get("run_tag_by_backbone", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("run_tag_by_backbone must be a mapping")
    tag = str(configured.get(str(backbone).lower(), "")).strip()
    return f"{base}_{tag}" if tag else base


def fairness_metric_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    configured = protocol.get("fairness_metrics", {})
    aggregate = str(configured.get("group_gap_aggregate", "utility_aware_gap"))
    if aggregate != "utility_aware_gap":
        raise ValueError("The revision protocol requires group_gap_aggregate=utility_aware_gap")
    metrics = {name: dict(spec) for name, spec in DEFAULT_FAIRNESS_METRICS.items()}
    for name, spec in (configured.get("metrics", {}) or {}).items():
        metrics[str(name)] = dict(spec)
    for spec in metrics.values():
        if "group" in spec:
            spec["artifact_key"] = f"{spec['group']}@{{k}}/{aggregate}"
    return {"group_gap_aggregate": aggregate, "metrics": metrics}


def fare_config_for_dataset(protocol: dict[str, Any], dataset: str) -> str:
    configs = protocol.get("runner_configs", {}) or {}
    overrides = configs.get("fare_by_dataset", {}) or {}
    if dataset in overrides:
        return str(overrides[dataset])
    if dataset == "MicroLens_100K":
        return "configs/revision/microlens_fare_3090.yaml"
    return str(configs.get("fare_default", "configs/fare_3090.yaml"))


def fare_run_name_for_dataset(
    protocol: dict[str, Any],
    dataset: str,
    base_dir: str | Path = ".",
) -> str:
    config_path = Path(fare_config_for_dataset(protocol, dataset))
    if not config_path.is_absolute():
        config_path = Path(base_dir) / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"FARE config root must be a mapping: {config_path}")
    return str(cfg.get("run_name", "fare"))
