# -*- coding: utf-8 -*-
"""
scripts/run_reranker_fair.py

Run FairRerank post-processing baseline on an existing model's top-K files.

This is a post-processing fairness baseline. It does not train a model.

Input:
    results/<Dataset>/<source_model>/<source_run>/topk_test.npz
    results/<Dataset>/<source_model>/<source_run>/topk_val.npz optional
    data/Processed_<Dataset>/item_group_matrix.npy

Output naming:
    --backbone sasrec
        results/<Dataset>/sasrec_fair_rerank/<run_id>
        method = SASRec + FairRerank

    --backbone gru4rec
        results/<Dataset>/gru4rec_fair_rerank/<run_id>
        method = GRU4Rec + FairRerank

    --backbone bert4rec
        results/<Dataset>/bert4rec_fair_rerank/<run_id>
        method = BERT4Rec + FairRerank

Config support:
    CLI arguments override YAML config.
    YAML config overrides hard-coded defaults.

Typical usage:

    python scripts/run_reranker_fair.py \
      --config configs/reranker_fair_3090.yaml \
      --dataset Video_Games \
      --backbone sasrec \
      --source_run_dir results/Video_Games/sasrec_id/20260515_194126 \
      --run_id sasrec_fair_rerank_video

Or mostly from YAML:

    python scripts/run_reranker_fair.py \
      --config configs/reranker_fair_3090.yaml \
      --dataset Video_Games \
      --backbone sasrec \
      --source_run_dir results/Video_Games/sasrec_id/20260515_194126

Important:
    FairRerank is most meaningful when source topk files contain candidate_k > top_k,
    e.g. top-100 candidates reranked into top-20 final recommendations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from reranker_fair import (  # type: ignore
        DEFAULT_RERANK_GROUPS,
        FairRerankConfig,
        aggregate_exposure_penalty,
        build_popularity_penalty,
        load_npz_topk,
        load_group_metadata,
        load_item_group_matrix,
        load_popularity,
        rerank_topk_data,
        rerank_topk_file,
        select_group_columns,
    )
    from revision_diagnostics import (  # type: ignore
        candidate_pool_audit,
        fairrr_candidate_group_diversity,
        fairrr_change_summary,
        validate_formal_fairrr,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import reranker or revision diagnostics. Make sure "
        "src/reranker_fair.py and src/revision_diagnostics.py pass py_compile."
    ) from exc


# ==========================================================
# Utilities
# ==========================================================


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def get_nested(cfg: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def first_not_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_list_value(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        parsed = parse_list_arg(value)
        return parsed if parsed is not None else list(default or [])
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return list(default or [])


def normalize_ks(value: Any, default: Sequence[int] = (5, 10, 20)) -> List[int]:
    if value is None:
        return [int(x) for x in default]
    if isinstance(value, str):
        parsed = parse_list_arg(value)
        if not parsed:
            return [int(x) for x in default]
        return [int(x) for x in parsed]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def normalize_float_grid(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        raw = list(default)
    elif isinstance(value, str):
        raw = [float(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw = [float(part) for part in value]
    else:
        raw = [float(value)]

    out: List[float] = []
    for item in raw:
        value = float(item)
        if value not in out:
            out.append(value)
    return out


def select_effective_lambda_fair(
    topk_data: Dict[str, np.ndarray],
    group_matrix: np.ndarray,
    cfg: FairRerankConfig,
    lambda_candidates: Sequence[float],
    popularity_penalty: Optional[np.ndarray],
    ks: Sequence[int],
    metric_for_best: Optional[str] = None,
    max_relative_utility_loss: float = 0.05,
    min_rank_changed_user_rate: float = 0.0,
    min_set_changed_user_rate: float = 0.0,
    min_fairness_improvement: float = 0.0,
) -> tuple[FairRerankConfig, Dict[str, Any]]:
    """Select lambda on validation by fairness gain under explicit constraints."""
    max_relative_utility_loss = float(max_relative_utility_loss)
    min_rank_changed_user_rate = float(min_rank_changed_user_rate)
    min_set_changed_user_rate = float(min_set_changed_user_rate)
    min_fairness_improvement = float(min_fairness_improvement)
    if not 0.0 <= max_relative_utility_loss <= 1.0:
        raise ValueError("max_relative_utility_loss must be in [0, 1]")
    for name, value in (
        ("min_rank_changed_user_rate", min_rank_changed_user_rate),
        ("min_set_changed_user_rate", min_set_changed_user_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if min_fairness_improvement < 0.0:
        raise ValueError("min_fairness_improvement must be non-negative")

    source_topk = np.asarray(topk_data["topk_items"], dtype=np.int64)
    diversity = fairrr_candidate_group_diversity(
        source_topk,
        group_matrix,
        candidate_k=cfg.candidate_k,
    )
    candidates = normalize_float_grid(lambda_candidates, default=[cfg.lambda_fair])
    if float(cfg.lambda_fair) not in candidates:
        candidates.insert(0, float(cfg.lambda_fair))

    baseline_cfg = FairRerankConfig(**{**cfg.__dict__, "lambda_fair": 0.0})
    baseline, baseline_metrics = rerank_topk_data(
        topk_data=topk_data,
        group_matrix=group_matrix,
        cfg=baseline_cfg,
        popularity_penalty=popularity_penalty,
        ks=ks,
    )
    if metric_for_best is None:
        metric_for_best = f"ndcg@{max(int(k) for k in ks)}"
    if metric_for_best not in baseline_metrics:
        raise ValueError(
            f"validation utility metric {metric_for_best!r} is unavailable; "
            f"available={sorted(baseline_metrics)}"
        )
    baseline_utility = float(baseline_metrics[metric_for_best])
    baseline_fairness = aggregate_exposure_penalty(
        baseline["topk_items"],
        group_matrix,
        baseline_cfg,
    )
    utility_floor = baseline_utility * (1.0 - max_relative_utility_loss)

    trials: List[Dict[str, Any]] = []
    feasible: List[tuple[float, float, float, FairRerankConfig, Dict[str, Any]]] = []
    for candidate in candidates:
        trial_cfg = FairRerankConfig(**{**cfg.__dict__, "lambda_fair": float(candidate)})
        reranked, metrics = rerank_topk_data(
            topk_data=topk_data,
            group_matrix=group_matrix,
            cfg=trial_cfg,
            popularity_penalty=popularity_penalty,
            ks=ks,
        )
        change = fairrr_change_summary(
            source_topk=source_topk,
            reranked_topk=np.asarray(reranked["topk_items"], dtype=np.int64),
            top_k=trial_cfg.top_k,
        )
        utility = float(metrics[metric_for_best])
        utility_retention = utility / baseline_utility if baseline_utility > 0.0 else 1.0
        fairness = aggregate_exposure_penalty(
            reranked["topk_items"],
            group_matrix,
            trial_cfg,
        )
        fairness_improvement = baseline_fairness - fairness
        constraints = {
            "utility": bool(utility >= utility_floor - 1e-12),
            "rank_change": bool(
                change["rank_changed_user_rate"] >= min_rank_changed_user_rate - 1e-12
            ),
            "set_change": bool(
                change["set_changed_user_rate"] >= min_set_changed_user_rate - 1e-12
            ),
            "fairness_improvement": bool(
                fairness_improvement >= min_fairness_improvement - 1e-12
            ),
        }
        trial = {
            "lambda_fair": float(candidate),
            **change,
            "utility_metric": metric_for_best,
            "utility": utility,
            "utility_retention": utility_retention,
            "fairness_penalty": fairness,
            "fairness_improvement": fairness_improvement,
            "constraints": constraints,
            "feasible": bool(all(constraints.values())),
        }
        trials.append(trial)
        if trial["feasible"]:
            feasible.append((fairness, -utility, float(candidate), trial_cfg, trial))

    constraints_audit = {
        "utility_metric": metric_for_best,
        "max_relative_utility_loss": max_relative_utility_loss,
        "utility_floor": utility_floor,
        "min_rank_changed_user_rate": min_rank_changed_user_rate,
        "min_set_changed_user_rate": min_set_changed_user_rate,
        "min_fairness_improvement": min_fairness_improvement,
    }
    if not feasible:
        return cfg, {
            "selection_split": "val",
            "selection_reason": "no_feasible_validation_candidate",
            "selected_lambda_fair": float(cfg.lambda_fair),
            "candidate_group_diversity": diversity,
            "baseline_utility": baseline_utility,
            "baseline_fairness_penalty": baseline_fairness,
            "constraints": constraints_audit,
            "trials": trials,
        }

    _, _, _, selected_cfg, selected = min(feasible, key=lambda row: row[:3])
    return selected_cfg, {
        "selection_split": "val",
        "selection_reason": "best_validation_fairness_under_constraints",
        "selected_lambda_fair": float(selected_cfg.lambda_fair),
        "selected_utility_retention": float(selected["utility_retention"]),
        "selected_fairness_improvement": float(selected["fairness_improvement"]),
        "candidate_group_diversity": diversity,
        "baseline_utility": baseline_utility,
        "baseline_fairness_penalty": baseline_fairness,
        "constraints": constraints_audit,
        "trials": trials,
    }

def get_dataset_dir(dataset: str, datasets_cfg: Dict[str, Any]) -> Path:
    candidates = []

    block = datasets_cfg.get("datasets")
    if isinstance(block, dict) and isinstance(block.get(dataset), dict):
        d = block[dataset]
        candidates.extend([d.get("data_dir"), d.get("path"), d.get("processed_dir"), d.get("dir")])

    if isinstance(datasets_cfg.get(dataset), dict):
        d = datasets_cfg[dataset]
        candidates.extend([d.get("data_dir"), d.get("path"), d.get("processed_dir"), d.get("dir")])

    for c in candidates:
        if c:
            return resolve_path(str(c))

    return PROJECT_ROOT / "data" / f"Processed_{dataset}"


def safe_slug(text: str) -> str:
    text = str(text).strip().replace("\\", "/")
    text = text.rstrip("/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120] if text else "run"


def infer_source_identity(source_run_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {
        "source_model_name": source_run_dir.parent.name,
        "source_run_id": source_run_dir.name,
    }

    metrics_path = source_run_dir / "metrics_summary.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                out["source_model_name"] = str(obj.get("model_name", out["source_model_name"]))
                out["source_run_id"] = str(obj.get("run_id", out["source_run_id"]))
                if obj.get("method") is not None:
                    out["source_method"] = str(obj.get("method"))
                if obj.get("backbone") is not None:
                    out["source_backbone"] = str(obj.get("backbone"))
        except Exception as exc:
            print(f"[Warning] failed to read source metrics_summary.json: {exc}")

    config_path = source_run_dir / "config_resolved.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                if "source_backbone" not in out and obj.get("backbone") is not None:
                    out["source_backbone"] = str(obj.get("backbone"))
                if "source_method" not in out and obj.get("method") is not None:
                    out["source_method"] = str(obj.get("method"))
        except Exception:
            pass

    return out


def infer_backbone(
    source_info: Dict[str, str],
    source_run_dir: Path,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return explicit.lower().strip()

    if source_info.get("source_backbone"):
        b = str(source_info["source_backbone"]).lower().strip()
        if b in {"sasrec", "gru4rec", "bert4rec"}:
            return b

    text = " ".join(
        [
            str(source_info.get("source_method", "")),
            str(source_info.get("source_model_name", "")),
            str(source_info.get("source_run_id", "")),
            str(source_run_dir),
        ]
    ).lower()

    if "gru4rec" in text or "gru" in text:
        return "gru4rec"
    if "bert4rec" in text or "bert" in text:
        return "bert4rec"
    if "sasrec" in text or "sas" in text:
        return "sasrec"

    print("[Warning] Could not infer backbone from source_run_dir; defaulting to sasrec.")
    return "sasrec"


def method_name_from_backbone(backbone: str) -> str:
    backbone = str(backbone).lower().strip()
    if backbone == "sasrec":
        return "SASRec + FairRerank"
    if backbone == "gru4rec":
        return "GRU4Rec + FairRerank"
    if backbone == "bert4rec":
        return "BERT4Rec + FairRerank"
    return f"{backbone} + FairRerank"


def run_name_from_backbone(backbone: str) -> str:
    backbone = str(backbone).lower().strip()
    return f"{backbone}_fair_rerank"


def inspect_topk_width(npz_path: Path) -> Optional[int]:
    if not npz_path.exists():
        return None

    try:
        z = np.load(npz_path)
        for key in ["items", "item_ids", "topk_items", "indices", "recommendations"]:
            if key in z.files:
                arr = z[key]
                if arr.ndim >= 2:
                    return int(arr.shape[1])
        for key in z.files:
            arr = z[key]
            if arr.ndim >= 2:
                return int(arr.shape[1])
    except Exception:
        return None

    return None


# ==========================================================
# CLI
# ==========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FairRerank post-processing baseline")

    parser.add_argument("--config", type=str, default="configs/reranker_fair_3090.yaml")

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["sasrec", "gru4rec", "bert4rec"])

    parser.add_argument(
        "--source_run_dir",
        type=str,
        default=None,
        help="Directory containing source topk_val.npz/topk_test.npz",
    )

    parser.add_argument("--data_dir", type=str, default=None, help="Processed dataset directory")
    parser.add_argument("--datasets_config", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output dir")
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--split", type=str, default=None, choices=["val", "test", "all"])
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--ks", type=str, default=None)
    parser.add_argument("--metric_for_best", type=str, default=None)

    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names or all",
    )
    parser.add_argument("--group_matrix_file", type=str, default=None)
    parser.add_argument("--group_metadata", type=str, default=None)

    parser.add_argument("--alpha_relevance", type=float, default=None)
    parser.add_argument("--lambda_fair", type=float, default=None)
    parser.add_argument("--lambda_popularity", type=float, default=None)
    parser.add_argument("--target_distribution", type=str, default=None, choices=["uniform", "empirical"])
    parser.add_argument("--penalty", type=str, default=None, choices=["l1", "l2", "max_gap"])
    parser.add_argument("--relevance_mode", type=str, default=None, choices=["log_rank", "reciprocal_rank", "linear_rank", "score"])
    parser.add_argument("--discount", type=str, default=None, choices=["log", "reciprocal", "none"])
    parser.add_argument("--no_normalize_group_rows", action="store_true")
    parser.add_argument("--random_tie_break", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--popularity_file", type=str, default=None)
    parser.add_argument(
        "--require_effective_change",
        action="store_true",
        help="Fail formal runs when the candidate pool or validation constraints are invalid.",
    )
    parser.add_argument("--max_relative_utility_loss", type=float, default=None)
    parser.add_argument("--min_rank_changed_user_rate", type=float, default=None)
    parser.add_argument("--min_set_changed_user_rate", type=float, default=None)
    parser.add_argument("--min_fairness_improvement", type=float, default=None)

    return parser.parse_args()


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config) if args.config else None
    file_cfg = load_yaml(cfg_path) if cfg_path is not None else {}

    dataset = first_not_none(
        args.dataset,
        file_cfg.get("default_dataset"),
        default=None,
    )
    if not dataset:
        raise ValueError("Dataset must be provided by --dataset or default_dataset in config.")

    dataset = str(dataset)

    datasets_config = first_not_none(
        args.datasets_config,
        get_nested(file_cfg, ["paths", "datasets_config"]),
        default="configs/datasets.yaml",
    )

    result_root = first_not_none(
        args.result_root,
        get_nested(file_cfg, ["paths", "result_root"]),
        default="results",
    )

    source_run_dir_raw = first_not_none(
        args.source_run_dir,
        get_nested(file_cfg, ["source", "source_run_dir"]),
        default=None,
    )
    if not source_run_dir_raw:
        raise ValueError("source_run_dir must be provided by --source_run_dir or source.source_run_dir in config.")

    source_run_dir = resolve_path(str(source_run_dir_raw))
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source_run_dir not found: {source_run_dir}")

    data_dir_raw = first_not_none(
        args.data_dir,
        get_nested(file_cfg, ["data", "data_dir"]),
        default=None,
    )
    if data_dir_raw:
        data_dir = resolve_path(str(data_dir_raw))
    else:
        datasets_cfg = load_yaml(resolve_path(str(datasets_config)))
        data_dir = get_dataset_dir(dataset, datasets_cfg)

    if not data_dir.exists():
        raise FileNotFoundError(f"processed data_dir not found: {data_dir}")

    seed = int(first_not_none(args.seed, file_cfg.get("seed"), default=2026))
    np.random.seed(seed)

    source_info = infer_source_identity(source_run_dir)

    explicit_backbone = first_not_none(
        args.backbone,
        get_nested(file_cfg, ["source", "backbone"]),
        default=None,
    )
    backbone = infer_backbone(
        source_info=source_info,
        source_run_dir=source_run_dir,
        explicit=explicit_backbone,
    )
    if backbone not in {"sasrec", "gru4rec", "bert4rec"}:
        raise ValueError(f"Unsupported backbone={backbone!r}")

    method_name = method_name_from_backbone(backbone)
    model_name = run_name_from_backbone(backbone)

    split = str(
        first_not_none(
            args.split,
            get_nested(file_cfg, ["rerank", "split"]),
            default="test",
        )
    )
    if split not in {"val", "test", "all"}:
        raise ValueError(f"Invalid split={split!r}. Expected val/test/all.")

    top_k = int(
        first_not_none(
            args.top_k,
            get_nested(file_cfg, ["rerank", "top_k"]),
            default=20,
        )
    )

    candidate_k = int(
        first_not_none(
            args.candidate_k,
            get_nested(file_cfg, ["rerank", "candidate_k"]),
            default=100,
        )
    )

    ks_value = first_not_none(
        args.ks,
        get_nested(file_cfg, ["rerank", "ks"]),
        default=[5, 10, 20],
    )
    ks = normalize_ks(ks_value, default=(5, 10, 20))

    metric_for_best = str(
        first_not_none(
            args.metric_for_best,
            get_nested(file_cfg, ["rerank", "metric_for_best"]),
            default="ndcg@10",
        )
    ).lower()

    groups_value = first_not_none(
        args.groups,
        get_nested(file_cfg, ["rerank", "groups"]),
        default=",".join(DEFAULT_RERANK_GROUPS),
    )

    group_matrix_file = str(
        first_not_none(
            args.group_matrix_file,
            get_nested(file_cfg, ["data", "group_matrix_file"]),
            default="item_group_matrix.npy",
        )
    )

    group_metadata = first_not_none(
        args.group_metadata,
        get_nested(file_cfg, ["data", "group_metadata"]),
        default=None,
    )

    popularity_file = str(
        first_not_none(
            args.popularity_file,
            get_nested(file_cfg, ["data", "popularity_file"]),
            default="item_popularity_train.npy",
        )
    )

    alpha_relevance = float(
        first_not_none(
            args.alpha_relevance,
            get_nested(file_cfg, ["rerank", "alpha_relevance"]),
            default=1.0,
        )
    )

    lambda_fair = float(
        first_not_none(
            args.lambda_fair,
            get_nested(file_cfg, ["rerank", "lambda_fair"]),
            default=0.2,
        )
    )
    lambda_fair_candidates = normalize_float_grid(
        get_nested(file_cfg, ["rerank", "lambda_fair_candidates"]),
        default=[lambda_fair],
    )
    if lambda_fair not in lambda_fair_candidates:
        lambda_fair_candidates.insert(0, lambda_fair)

    lambda_popularity = float(
        first_not_none(
            args.lambda_popularity,
            get_nested(file_cfg, ["rerank", "lambda_popularity"]),
            default=0.0,
        )
    )

    target_distribution = str(
        first_not_none(
            args.target_distribution,
            get_nested(file_cfg, ["rerank", "target_distribution"]),
            default="uniform",
        )
    )

    penalty = str(
        first_not_none(
            args.penalty,
            get_nested(file_cfg, ["rerank", "penalty"]),
            default="l2",
        )
    )

    relevance_mode = str(
        first_not_none(
            args.relevance_mode,
            get_nested(file_cfg, ["rerank", "relevance_mode"]),
            default="log_rank",
        )
    )

    discount = str(
        first_not_none(
            args.discount,
            get_nested(file_cfg, ["rerank", "discount"]),
            default="log",
        )
    )

    cfg_normalize_group_rows = bool(
        get_nested(file_cfg, ["rerank", "normalize_group_rows"], default=True)
    )
    normalize_group_rows = False if args.no_normalize_group_rows else cfg_normalize_group_rows

    random_tie_break = bool(
        args.random_tie_break
        or get_nested(file_cfg, ["rerank", "random_tie_break"], default=False)
    )
    require_effective_change = bool(
        args.require_effective_change
        or get_nested(file_cfg, ["rerank", "require_effective_change"], default=False)
    )
    max_relative_utility_loss = float(
        first_not_none(
            args.max_relative_utility_loss,
            get_nested(file_cfg, ["rerank", "max_relative_utility_loss"]),
            default=0.05,
        )
    )
    min_rank_changed_user_rate = float(
        first_not_none(
            args.min_rank_changed_user_rate,
            get_nested(file_cfg, ["rerank", "min_rank_changed_user_rate"]),
            default=0.01,
        )
    )
    min_set_changed_user_rate = float(
        first_not_none(
            args.min_set_changed_user_rate,
            get_nested(file_cfg, ["rerank", "min_set_changed_user_rate"]),
            default=0.01,
        )
    )
    min_fairness_improvement = float(
        first_not_none(
            args.min_fairness_improvement,
            get_nested(file_cfg, ["rerank", "min_fairness_improvement"]),
            default=1e-6,
        )
    )

    run_id = args.run_id
    if not run_id:
        source_slug = safe_slug(source_info.get("source_run_id", source_run_dir.name))
        run_id = f"{model_name}_{source_slug}_lf{lambda_fair:g}_lp{lambda_popularity:g}"

    output_dir_raw = first_not_none(
        args.output_dir,
        get_nested(file_cfg, ["output", "output_dir"]),
        default=None,
    )
    if output_dir_raw:
        output_dir = resolve_path(str(output_dir_raw))
    else:
        output_dir = resolve_path(str(result_root)) / dataset / model_name / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["val", "test"] if split == "all" else [split]

    full_group_matrix = load_item_group_matrix(data_dir, filename=group_matrix_file)
    metadata = load_group_metadata(data_dir, explicit_path=group_metadata)

    if isinstance(groups_value, str) and groups_value.lower().strip() == "all":
        requested_groups: Sequence[str] | str = "all"
    else:
        requested_groups = normalize_list_value(groups_value, default=list(DEFAULT_RERANK_GROUPS))

    group_matrix, selected_cols, used_groups = select_group_columns(
        full_group_matrix,
        groups=requested_groups,
        metadata=metadata,
    )

    selected_cols = [int(x) for x in selected_cols]

    popularity = load_popularity(data_dir, filename=popularity_file)
    popularity_penalty = build_popularity_penalty(
        popularity,
        num_items=full_group_matrix.shape[0] - 1,
    )

    if lambda_popularity <= 0:
        popularity_penalty = None

    rerank_cfg = FairRerankConfig(
        top_k=int(top_k),
        candidate_k=int(candidate_k),
        alpha_relevance=float(alpha_relevance),
        lambda_fair=float(lambda_fair),
        lambda_popularity=float(lambda_popularity),
        target_distribution=str(target_distribution),
        penalty=str(penalty),
        relevance_mode=str(relevance_mode),
        discount=str(discount),
        groups=tuple(used_groups),
        normalize_group_rows=bool(normalize_group_rows),
        random_tie_break=bool(random_tie_break),
        seed=int(seed),
    )

    lambda_selection_audit = None
    if require_effective_change and "val" in splits and len(lambda_fair_candidates) > 1:
        val_npz = source_run_dir / "topk_val.npz"
        if val_npz.exists():
            rerank_cfg, lambda_selection_audit = select_effective_lambda_fair(
                topk_data=load_npz_topk(val_npz),
                group_matrix=group_matrix,
                cfg=rerank_cfg,
                lambda_candidates=lambda_fair_candidates,
                popularity_penalty=popularity_penalty,
                ks=ks,
                metric_for_best=metric_for_best,
                max_relative_utility_loss=max_relative_utility_loss,
                min_rank_changed_user_rate=min_rank_changed_user_rate,
                min_set_changed_user_rate=min_set_changed_user_rate,
                min_fairness_improvement=min_fairness_improvement,
            )
        else:
            lambda_selection_audit = {
                "selection_split": "val",
                "selection_reason": "missing_validation_topk",
                "selected_lambda_fair": float(rerank_cfg.lambda_fair),
                "trials": [],
            }

    resolved = {
        "config": str(cfg_path) if cfg_path is not None else None,
        "dataset": dataset,
        "method": method_name,
        "model_name": model_name,
        "run_id": run_id,
        "backbone": backbone,
        "source_run_dir": str(source_run_dir),
        **source_info,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "splits": splits,
        "ks": ks,
        "metric_for_best": metric_for_best,
        "rerank_config": rerank_cfg.__dict__,
        "lambda_fair_candidates": [float(value) for value in lambda_fair_candidates],
        "lambda_selection_audit": lambda_selection_audit,
        "requested_groups": requested_groups if isinstance(requested_groups, str) else list(requested_groups),
        "used_groups": list(used_groups),
        "selected_group_columns": selected_cols,
        "num_selected_group_columns": len(selected_cols),
        "group_metadata_found": metadata is not None,
        "group_matrix_shape": list(full_group_matrix.shape),
        "datasets_config": str(datasets_config),
        "result_root": str(result_root),
        "group_matrix_file": group_matrix_file,
        "group_metadata": str(group_metadata) if group_metadata is not None else None,
        "popularity_file": popularity_file,
        "require_effective_change": require_effective_change,
        "fairrr_selection_constraints": {
            "max_relative_utility_loss": max_relative_utility_loss,
            "min_rank_changed_user_rate": min_rank_changed_user_rate,
            "min_set_changed_user_rate": min_set_changed_user_rate,
            "min_fairness_improvement": min_fairness_improvement,
        },
    }

    save_json(resolved, output_dir / "config_resolved.json")

    if require_effective_change and lambda_selection_audit is not None:
        reason = lambda_selection_audit.get("selection_reason")
        if reason in {"no_feasible_validation_candidate", "missing_validation_topk"}:
            diversity = lambda_selection_audit.get("candidate_group_diversity", {})
            raise ValueError(
                "FairRR validation-only lambda selection failed: "
                f"reason={reason}, constraints={lambda_selection_audit.get('constraints')}, "
                f"candidate_group_audit={diversity}. Inspect config_resolved.json."
            )
    print("========== FairRerank ==========")
    print(f"config:         {cfg_path}")
    print(f"dataset:        {dataset}")
    print(f"method:         {method_name}")
    print(f"backbone:       {backbone}")
    print(f"model_name:     {model_name}")
    print(f"run_id:         {run_id}")
    print(f"source_run_dir: {source_run_dir}")
    print(f"output_dir:     {output_dir}")
    print(f"data_dir:       {data_dir}")
    print(f"splits:         {splits}")
    print(f"ks:             {ks}")
    print(f"metric_best:    {metric_for_best}")
    print(f"top_k:          {rerank_cfg.top_k}")
    print(f"candidate_k:    {rerank_cfg.candidate_k}")
    print(f"lambda_fair:    {rerank_cfg.lambda_fair}")
    print(f"lambda_pop:     {rerank_cfg.lambda_popularity}")
    print(f"groups:         {used_groups}")
    print(f"group columns:  {len(selected_cols)} / {full_group_matrix.shape[1]}")

    if metadata is None:
        print("[Warning] group metadata not found; selected columns may fall back to all active columns.")

    if rerank_cfg.candidate_k <= rerank_cfg.top_k:
        print("[Note] candidate_k <= top_k. Reranker can only reorder the final top-k set, not promote new items.")

    for sp in splits:
        input_npz = source_run_dir / f"topk_{sp}.npz"
        width = inspect_topk_width(input_npz)
        if width is not None and width < rerank_cfg.candidate_k:
            print(
                f"[Warning] source {input_npz.name} appears to contain only {width} candidates, "
                f"but candidate_k={rerank_cfg.candidate_k}. Effective reranking pool may be limited."
            )

    summary: Dict[str, object] = {
        "dataset": dataset,
        "method": method_name,
        "model_name": model_name,
        "run_id": run_id,
        "backbone": backbone,
        "source_run_dir": str(source_run_dir),
        **source_info,
        "top_k": int(rerank_cfg.top_k),
        "candidate_k": int(rerank_cfg.candidate_k),
        "lambda_fair": float(rerank_cfg.lambda_fair),
        "lambda_popularity": float(rerank_cfg.lambda_popularity),
        "lambda_fair_candidates": [float(value) for value in lambda_fair_candidates],
        "lambda_selection_audit": lambda_selection_audit,
        "used_groups": list(used_groups),
        "selected_group_columns": selected_cols,
        "fairrr_selection_constraints": {
            "max_relative_utility_loss": max_relative_utility_loss,
            "min_rank_changed_user_rate": min_rank_changed_user_rate,
            "min_set_changed_user_rate": min_set_changed_user_rate,
            "min_fairness_improvement": min_fairness_improvement,
        },
    }

    for sp in splits:
        input_npz = source_run_dir / f"topk_{sp}.npz"
        if not input_npz.exists():
            raise FileNotFoundError(
                f"Missing source topk file: {input_npz}. "
                "Make sure the source model was trained with save_topk_npz=true."
            )

        output_npz = output_dir / f"topk_{sp}.npz"

        start = time.time()
        metrics = rerank_topk_file(
            input_npz=input_npz,
            output_npz=output_npz,
            group_matrix=group_matrix,
            cfg=rerank_cfg,
            popularity_penalty=popularity_penalty,
            ks=ks,
        )
        elapsed = time.time() - start

        with np.load(input_npz, allow_pickle=False) as source_archive:
            source_topk = np.asarray(source_archive["topk_items"], dtype=np.int64)
        with np.load(output_npz, allow_pickle=False) as reranked_archive:
            reranked_topk = np.asarray(reranked_archive["topk_items"], dtype=np.int64)
        pool_audit = candidate_pool_audit(
            source_width=source_topk.shape[1],
            candidate_k=rerank_cfg.candidate_k,
            top_k=rerank_cfg.top_k,
        )
        change_audit = fairrr_change_summary(
            source_topk=source_topk,
            reranked_topk=reranked_topk,
            top_k=rerank_cfg.top_k,
        )
        if require_effective_change:
            validate_formal_fairrr(
                change_audit,
                pool_audit,
                min_rank_changed_user_rate=min_rank_changed_user_rate,
                min_set_changed_user_rate=min_set_changed_user_rate,
            )

        summary[sp] = metrics
        summary[f"{sp}_candidate_pool_audit"] = pool_audit
        summary[f"{sp}_change_audit"] = change_audit
        summary[f"{sp}_topk_path"] = str(output_npz)
        summary[f"{sp}_elapsed_sec"] = float(elapsed)

        print(f"[{sp}] saved: {output_npz}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))
        print(f"[{sp}] elapsed: {elapsed:.2f}s")
        print(f"[{sp}] change audit: {json.dumps(change_audit, ensure_ascii=False)}")

    if "val" in summary and isinstance(summary["val"], dict) and metric_for_best in summary["val"]:
        best_metric = float(summary["val"][metric_for_best])  # type: ignore[index]
    elif "test" in summary and isinstance(summary["test"], dict) and metric_for_best in summary["test"]:
        best_metric = float(summary["test"][metric_for_best])  # type: ignore[index]
    else:
        best_metric = float("nan")

    summary["best_metric_name"] = metric_for_best
    summary["best_metric"] = best_metric
    summary["best_epoch"] = -1

    save_json(summary, output_dir / "metrics_summary.json")

    print("========== Finished ==========")
    print(f"Run dir: {output_dir}")


if __name__ == "__main__":
    main()
