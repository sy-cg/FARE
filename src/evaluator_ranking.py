# -*- coding: utf-8 -*-
"""
src/evaluator_ranking.py

Full-sort ranking evaluator for SDR / FARE experiments.

Compatible saved ranking formats:

1. Full format:
    split_name, users, targets, topk_items, topk_scores, hit_ranks/ranks, metrics

2. Lightweight top-k format produced by current training scripts:
    user_ids, targets, topk_items, ranks

The lightweight format is enough for fairness evaluation because fairness only
needs users, targets, top-k item ids, and hit ranks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


# ==========================================================
# Data structures
# ==========================================================


@dataclass(frozen=True)
class EvalRecord:
    user_id: int
    prefix: List[int]
    target: int


@dataclass
class RankingResult:
    split_name: str
    metrics: Dict[str, float]
    topk_items: np.ndarray        # [num_records, max_k]
    targets: np.ndarray           # [num_records]
    users: np.ndarray             # [num_records]
    hit_ranks: np.ndarray         # [num_records], 1-indexed rank if hit within max_k, 0 otherwise
    topk_scores: Optional[np.ndarray] = None

    @property
    def ranks(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.hit_ranks

    @property
    def user_ids(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.users


# score_fn(model, user_ids, sequences, lengths) -> full-sort scores [B, num_items + 1]
ScoreFn = Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ==========================================================
# Basic utilities
# ==========================================================


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_num_items_from_processed_dir(data_dir: str | Path) -> int:
    """Infer number of non-padding items from a processed dataset directory."""
    data_dir = str(data_dir)

    item2id_path = os.path.join(data_dir, "item2id.json")
    if os.path.exists(item2id_path):
        item2id = load_json(item2id_path)
        return len(item2id)

    pop_path = os.path.join(data_dir, "item_popularity_train.npy")
    if os.path.exists(pop_path):
        return int(np.load(pop_path, mmap_mode="r").shape[0] - 1)

    raise FileNotFoundError(
        f"Cannot infer num_items for {data_dir}: missing item2id.json and item_popularity_train.npy"
    )


def read_eval_records(path: str | Path) -> List[EvalRecord]:
    """Read evaluation records from train/val/test style sequence files.

    Expected line:
        user_id item_1 item_2 ... item_T

    Evaluation:
        prefix = item_1 ... item_{T-1}
        target = item_T
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

    records: List[EvalRecord] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts or len(parts) < 3:
                continue

            try:
                ids = [int(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {path}:{line_no}: {line[:120]}") from exc

            user_id = ids[0]
            seq = ids[1:]
            prefix = [x for x in seq[:-1] if x > 0]
            target = int(seq[-1])

            if user_id <= 0 or target <= 0 or len(prefix) == 0:
                continue

            records.append(EvalRecord(user_id=user_id, prefix=prefix, target=target))

    return records


def pad_prefixes(
    prefixes: Sequence[Sequence[int]],
    max_len: Optional[int] = None,
    pad_id: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length prefixes.

    Right padding is consistent with the current project models, which use
    lengths to gather the final valid hidden state.
    """
    if len(prefixes) == 0:
        raise ValueError("Cannot pad an empty prefix batch")

    if max_len is None:
        max_len = max(1, max(len(p) for p in prefixes))
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    batch_size = len(prefixes)
    arr = np.full((batch_size, max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for row, seq in enumerate(prefixes):
        clean = [int(x) for x in seq if int(x) > 0]
        if not clean:
            continue
        clean = clean[-max_len:]
        arr[row, : len(clean)] = np.asarray(clean, dtype=np.int64)
        lengths[row] = len(clean)

    seq_tensor = torch.from_numpy(arr)
    len_tensor = torch.from_numpy(lengths)

    if device is not None:
        seq_tensor = seq_tensor.to(device, non_blocking=True)
        len_tensor = len_tensor.to(device, non_blocking=True)

    return seq_tensor, len_tensor


def _infer_split_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("topk_"):
        return stem.replace("topk_", "", 1)
    if stem.endswith("_ranking"):
        return stem.replace("_ranking", "")
    return "unknown"


def _decode_np_scalar(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def save_ranking_result_npz(result: RankingResult, output_path: str | Path) -> None:
    """Save a RankingResult as compressed npz.

    Saves both old and new aliases:
        users and user_ids
        hit_ranks and ranks
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "split_name": result.split_name,
        "topk_items": np.asarray(result.topk_items),
        "targets": np.asarray(result.targets),
        "users": np.asarray(result.users),
        "user_ids": np.asarray(result.users),
        "hit_ranks": np.asarray(result.hit_ranks),
        "ranks": np.asarray(result.hit_ranks),
        "metrics": json.dumps(result.metrics),
    }

    if result.topk_scores is not None:
        payload["topk_scores"] = np.asarray(result.topk_scores)

    np.savez_compressed(output_path, **payload)


def load_ranking_result_npz(path: str | Path) -> RankingResult:
    """
    Load saved ranking result.

    Supports both formats:

    1. Full ranking format:
        users, targets, topk_items, topk_scores, hit_ranks/ranks, metrics, split_name

    2. Lightweight top-k format produced by current training scripts:
        user_ids, targets, topk_items, ranks

    In lightweight format:
        users       <- user_ids
        topk_scores <- None
        metrics     <- {}
        split_name  <- inferred from filename, e.g. topk_test.npz -> test
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)

        def has(name: str) -> bool:
            return name in files

        def get_first(names: Sequence[str], required: bool = True, default: Any = None) -> Any:
            for name in names:
                if name in files:
                    return data[name]
            if required:
                raise KeyError(
                    f"None of {list(names)} found in archive {path}. "
                    f"Available keys: {sorted(files)}"
                )
            return default

        users = np.asarray(get_first(["users", "user_ids", "uids", "user"]), dtype=np.int64)
        targets = np.asarray(
            get_first(["targets", "target_items", "target", "labels", "ground_truth"]),
            dtype=np.int64,
        )
        topk_items = np.asarray(
            get_first(["topk_items", "items", "recommendations", "ranking_items"]),
            dtype=np.int64,
        )

        topk_scores = get_first(
            ["topk_scores", "scores", "ranking_scores"],
            required=False,
            default=None,
        )
        if topk_scores is not None:
            topk_scores = np.asarray(topk_scores, dtype=np.float32)

        hit_ranks = get_first(
            ["hit_ranks", "ranks", "rank"],
            required=False,
            default=None,
        )

        if hit_ranks is not None:
            hit_ranks = np.asarray(hit_ranks, dtype=np.int64)
        else:
            target_col = targets.reshape(-1, 1)
            hit_pos = np.where(topk_items == target_col)
            hit_ranks = np.zeros((targets.shape[0],), dtype=np.int64)
            if len(hit_pos[0]) > 0:
                for row_idx, col_idx in zip(hit_pos[0], hit_pos[1]):
                    if hit_ranks[row_idx] == 0:
                        hit_ranks[row_idx] = int(col_idx) + 1

        if has("metrics"):
            raw_metrics = _decode_np_scalar(data["metrics"])
            try:
                if isinstance(raw_metrics, dict):
                    metrics = raw_metrics
                else:
                    metrics = json.loads(str(raw_metrics))
                if not isinstance(metrics, dict):
                    metrics = {}
            except Exception:
                metrics = {}
        else:
            metrics = {}

        if has("split_name"):
            raw_split_name = _decode_np_scalar(data["split_name"])
            split_name = str(raw_split_name)
        else:
            split_name = _infer_split_name_from_path(path)

    return RankingResult(
        split_name=split_name,
        metrics=metrics,
        topk_items=topk_items,
        targets=targets,
        users=users,
        hit_ranks=hit_ranks,
        topk_scores=topk_scores,
    )


def score_fn_from_model_method(
    model: Any,
    user_ids: torch.Tensor,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Adapter for models exposing full_sort_scores(user_ids, sequences, lengths)."""
    if not hasattr(model, "full_sort_scores"):
        raise AttributeError("Model must implement full_sort_scores(user_ids, sequences, lengths)")
    return model.full_sort_scores(user_ids, sequences, lengths)


# ==========================================================
# Ranking evaluator
# ==========================================================


class RankingEvaluator:
    """Model-agnostic full-sort next-item ranking evaluator."""

    def __init__(
        self,
        num_items: int,
        ks: Sequence[int] = (5, 10, 20),
        max_seq_len: int = 50,
        pad_id: int = 0,
        mask_padding_item: bool = True,
    ) -> None:
        self.num_items = int(num_items)
        self.ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
        if not self.ks:
            raise ValueError("ks cannot be empty")
        self.max_k = max(self.ks)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.mask_padding_item = bool(mask_padding_item)

        if self.max_k > self.num_items:
            self.max_k = self.num_items
            self.ks = tuple(k for k in self.ks if k <= self.max_k)

    @staticmethod
    def compute_single_target_metrics(hit_ranks: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
        """
        Backward-compatible public helper used by evaluator_fairness.py.

        Args:
            hit_ranks:
                1-indexed hit rank for each example. 0 means the target item
                is not found within the saved top-k list.
            ks:
                Evaluation cutoffs, e.g. (5, 10, 20).

        Returns:
            Metrics for single-target next-item recommendation:
                hit@K, recall@K, ndcg@K, mrr@K
        """
        ranks = np.asarray(hit_ranks, dtype=np.int64)
        n = max(int(ranks.shape[0]), 1)

        metrics: Dict[str, float] = {}
        for k in ks:
            k = int(k)
            hit = (ranks > 0) & (ranks <= k)
            hit_float = hit.astype(np.float64)

            # In single-target next-item recommendation, HitRate@K == Recall@K.
            metrics[f"hit@{k}"] = float(hit_float.mean())
            metrics[f"recall@{k}"] = float(hit_float.mean())

            ndcg = np.zeros_like(hit_float, dtype=np.float64)
            if hit.any():
                ndcg[hit] = 1.0 / np.log2(ranks[hit].astype(np.float64) + 1.0)
            metrics[f"ndcg@{k}"] = float(ndcg.sum() / n)

            mrr = np.zeros_like(hit_float, dtype=np.float64)
            if hit.any():
                mrr[hit] = 1.0 / ranks[hit].astype(np.float64)
            metrics[f"mrr@{k}"] = float(mrr.sum() / n)

        return metrics

    @staticmethod
    def _metrics_from_hit_ranks(hit_ranks: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
        ranks = np.asarray(hit_ranks, dtype=np.int64)
        n = max(int(ranks.shape[0]), 1)

        metrics: Dict[str, float] = {}

        for k in ks:
            hit = (ranks > 0) & (ranks <= int(k))
            hit_float = hit.astype(np.float64)

            # Single-target next-item setting: HitRate@K == Recall@K.
            metrics[f"hit@{k}"] = float(hit_float.mean())
            metrics[f"recall@{k}"] = float(hit_float.mean())

            ndcg = np.zeros_like(hit_float, dtype=np.float64)
            if hit.any():
                ndcg[hit] = 1.0 / np.log2(ranks[hit].astype(np.float64) + 1.0)
            metrics[f"ndcg@{k}"] = float(ndcg.sum() / n)

            mrr = np.zeros_like(hit_float, dtype=np.float64)
            if hit.any():
                mrr[hit] = 1.0 / ranks[hit].astype(np.float64)
            metrics[f"mrr@{k}"] = float(mrr.sum() / n)

        return metrics

    def _mask_seen_items(
        self,
        scores: torch.Tensor,
        records: Sequence[EvalRecord],
        seen_items: Optional[Dict[int, Sequence[int]]] = None,
    ) -> torch.Tensor:
        """Mask training/prefix items while preserving the target item."""
        for row_idx, rec in enumerate(records):
            if seen_items is not None and rec.user_id in seen_items:
                items = seen_items[rec.user_id]
            else:
                items = rec.prefix

            for item in items:
                item = int(item)
                if item <= 0 or item > self.num_items or item == rec.target:
                    continue
                scores[row_idx, item] = -1.0e9

        return scores

    @torch.no_grad()
    def evaluate_records(
        self,
        model: Any,
        records: Sequence[EvalRecord],
        score_fn: ScoreFn = score_fn_from_model_method,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
        split_name: str = "test",
        mask_seen_items: bool = False,
        seen_items: Optional[Dict[int, Sequence[int]]] = None,
        save_path: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> RankingResult:
        if not records:
            raise ValueError("No evaluation records")

        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(model, "eval"):
            model.eval()

        all_users: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_topk_items: List[np.ndarray] = []
        all_topk_scores: List[np.ndarray] = []
        all_hit_ranks: List[np.ndarray] = []

        iterator = range(0, len(records), int(batch_size))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {split_name}", leave=False)

        for start in iterator:
            batch_records = list(records[start : start + int(batch_size)])

            users_np = np.asarray([r.user_id for r in batch_records], dtype=np.int64)
            targets_np = np.asarray([r.target for r in batch_records], dtype=np.int64)

            sequences, lengths = pad_prefixes(
                [r.prefix for r in batch_records],
                max_len=self.max_seq_len,
                pad_id=self.pad_id,
                device=device,
            )
            user_ids = torch.from_numpy(users_np).to(device, non_blocking=True)

            scores = score_fn(model, user_ids, sequences, lengths)
            if not isinstance(scores, torch.Tensor):
                raise TypeError("score_fn must return a torch.Tensor")
            if scores.ndim != 2:
                raise ValueError(f"score_fn returned invalid score shape: {tuple(scores.shape)}")
            if scores.size(1) < self.num_items + 1:
                raise ValueError(
                    f"score_fn returned only {scores.size(1)} item scores, "
                    f"expected at least {self.num_items + 1}"
                )

            scores = scores[:, : self.num_items + 1]
            scores = torch.nan_to_num(scores, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)

            if self.mask_padding_item:
                scores[:, 0] = -1.0e9

            if mask_seen_items:
                scores = self._mask_seen_items(scores, batch_records, seen_items=seen_items)

            top_scores, top_items = torch.topk(scores, k=self.max_k, dim=1)

            top_items_np = top_items.detach().cpu().numpy().astype(np.int64)
            top_scores_np = top_scores.detach().cpu().numpy().astype(np.float32)

            hit_ranks = np.zeros((len(batch_records),), dtype=np.int64)
            for i, target in enumerate(targets_np):
                pos = np.where(top_items_np[i] == int(target))[0]
                if len(pos) > 0:
                    hit_ranks[i] = int(pos[0]) + 1

            all_users.append(users_np)
            all_targets.append(targets_np)
            all_topk_items.append(top_items_np)
            all_topk_scores.append(top_scores_np)
            all_hit_ranks.append(hit_ranks)

        users = np.concatenate(all_users, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        topk_items = np.concatenate(all_topk_items, axis=0)
        topk_scores = np.concatenate(all_topk_scores, axis=0)
        hit_ranks = np.concatenate(all_hit_ranks, axis=0)

        metrics = self._metrics_from_hit_ranks(hit_ranks, self.ks)

        result = RankingResult(
            split_name=split_name,
            metrics=metrics,
            topk_items=topk_items,
            topk_scores=topk_scores,
            targets=targets,
            users=users,
            hit_ranks=hit_ranks,
        )

        if save_path is not None:
            save_ranking_result_npz(result, save_path)

        return result

    def evaluate_from_file(
        self,
        model: Any,
        split_path: str | Path,
        score_fn: ScoreFn = score_fn_from_model_method,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
        split_name: str = "test",
        mask_seen_items: bool = False,
        seen_items: Optional[Dict[int, Sequence[int]]] = None,
        save_path: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> RankingResult:
        records = read_eval_records(split_path)
        return self.evaluate_records(
            model=model,
            records=records,
            score_fn=score_fn,
            batch_size=batch_size,
            device=device,
            split_name=split_name,
            mask_seen_items=mask_seen_items,
            seen_items=seen_items,
            save_path=save_path,
            show_progress=show_progress,
        )

    # Backward-compatible method name.
    def evaluate(self, *args, **kwargs) -> RankingResult:
        if args and isinstance(args[0], (str, Path)):
            return self.evaluate_from_file(*args, **kwargs)
        return self.evaluate_records(*args, **kwargs)


# ==========================================================
# Smoke-test model
# ==========================================================


class PopularityDummyModel:
    """Simple popularity scorer for evaluator smoke tests.

    This is not a paper baseline. It is only for checking that the evaluator
    and fairness pipeline run end-to-end.
    """

    def __init__(self, popularity: np.ndarray):
        if popularity.ndim != 1:
            raise ValueError("popularity must be 1-D")
        self.popularity = torch.tensor(popularity.astype(np.float32))

    def eval(self) -> "PopularityDummyModel":
        return self

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del user_ids, lengths
        scores = self.popularity.to(sequences.device).unsqueeze(0).repeat(sequences.shape[0], 1)
        return scores
