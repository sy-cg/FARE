# -*- coding: utf-8 -*-
"""
src/evaluator_leakage.py

Representation leakage evaluator for item-side fairness groups.

Given an item representation matrix X and group labels y, this module trains a
simple probe classifier to predict the group label from X. If a representation is
fair/debiased, the probe should perform closer to a majority/random baseline.
Main metrics:
- accuracy
- balanced_accuracy
- macro_f1
- weighted_f1
- majority_baseline_accuracy
- majority_baseline_macro_f1

Typical use:
    evaluator = LeakageProbeEvaluator(test_size=0.3, seed=2026)
    result = evaluator.evaluate_one(X, labels, group_name="popularity_group", representation="z_fair")

This module is intentionally model-agnostic. Model loading and representation
extraction are handled by scripts/eval_leakage.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier


@dataclass
class LeakageProbeResult:
    representation: str
    group_name: str
    status: str
    message: str
    n_items: int
    n_valid: int
    n_train: int
    n_test: int
    n_classes: int
    min_class_count: int
    max_class_count: int
    majority_class: int
    majority_baseline_accuracy: float
    majority_baseline_balanced_accuracy: float
    majority_baseline_macro_f1: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    log_loss: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LeakageProbeEvaluator:
    """Train linear probes for representation leakage evaluation."""

    def __init__(
        self,
        test_size: float = 0.3,
        seed: int = 2026,
        min_class_count: int = 5,
        max_items: Optional[int] = None,
        max_samples_per_class: Optional[int] = None,
        classifier: str = "logreg",
        max_iter: int = 500,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = 4,
        standardize: bool = True,
        ignore_index: int = -1,
    ) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.min_class_count = int(min_class_count)
        self.max_items = None if max_items is None else int(max_items)
        self.max_samples_per_class = None if max_samples_per_class is None else int(max_samples_per_class)
        self.classifier = str(classifier).lower()
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.n_jobs = int(n_jobs)
        self.standardize = bool(standardize)
        self.ignore_index = int(ignore_index)

    @staticmethod
    def _empty_result(
        representation: str,
        group_name: str,
        status: str,
        message: str,
        n_items: int,
        n_valid: int = 0,
        n_classes: int = 0,
    ) -> LeakageProbeResult:
        return LeakageProbeResult(
            representation=representation,
            group_name=group_name,
            status=status,
            message=message,
            n_items=int(n_items),
            n_valid=int(n_valid),
            n_train=0,
            n_test=0,
            n_classes=int(n_classes),
            min_class_count=0,
            max_class_count=0,
            majority_class=-1,
            majority_baseline_accuracy=np.nan,
            majority_baseline_balanced_accuracy=np.nan,
            majority_baseline_macro_f1=np.nan,
            accuracy=np.nan,
            balanced_accuracy=np.nan,
            macro_f1=np.nan,
            weighted_f1=np.nan,
            micro_f1=np.nan,
            log_loss=np.nan,
        )

    def _sanitize_X(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape={X.shape}")
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _make_classifier(self):
        if self.classifier == "logreg":
            clf = LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                solver="lbfgs",
                multi_class="auto",
                n_jobs=max(1, self.n_jobs),
                random_state=self.seed,
            )
        elif self.classifier == "dummy":
            clf = DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError(f"Unsupported classifier={self.classifier!r}; supported: logreg, dummy")

        if self.standardize and self.classifier != "dummy":
            return make_pipeline(StandardScaler(), clf)
        return clf

    def _subsample_by_class(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optional deterministic subsampling for speed and class balance."""
        rng = np.random.default_rng(self.seed)
        indices: List[int] = []

        if self.max_samples_per_class is not None and self.max_samples_per_class > 0:
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                if len(cls_idx) > self.max_samples_per_class:
                    cls_idx = rng.choice(cls_idx, size=self.max_samples_per_class, replace=False)
                indices.extend(cls_idx.tolist())
            indices = sorted(indices)
            X = X[indices]
            y = y[indices]

        if self.max_items is not None and self.max_items > 0 and len(y) > self.max_items:
            chosen = rng.choice(np.arange(len(y)), size=self.max_items, replace=False)
            chosen = np.sort(chosen)
            X = X[chosen]
            y = y[chosen]

        return X, y

    def evaluate_one(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        group_name: str,
        representation: str,
        item_ids: Optional[np.ndarray] = None,
    ) -> LeakageProbeResult:
        """Evaluate group-label leakage from one representation.

        Args:
            X: item representation matrix, shape [num_items+1, dim]. Row 0 is padding.
            labels: group labels, shape [num_items+1]. Unknown labels should be ignore_index.
            group_name: fairness group name.
            representation: representation name, e.g. mm_raw or z_fair.
            item_ids: optional item ids to include. If omitted, uses all non-padding rows.
        """
        X = self._sanitize_X(X)
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape={labels.shape}")
        if len(labels) != X.shape[0]:
            raise ValueError(f"Label length {len(labels)} does not match X rows {X.shape[0]}")

        n_items = X.shape[0] - 1
        if item_ids is None:
            item_ids = np.arange(1, X.shape[0], dtype=np.int64)
        else:
            item_ids = np.asarray(item_ids, dtype=np.int64)
            item_ids = item_ids[(item_ids > 0) & (item_ids < X.shape[0])]

        valid_mask = labels[item_ids] != self.ignore_index
        item_ids = item_ids[valid_mask]
        if len(item_ids) == 0:
            return self._empty_result(representation, group_name, "skipped", "no valid labels", n_items, 0, 0)

        Xv = X[item_ids]
        y = labels[item_ids]
        finite_rows = np.isfinite(Xv).all(axis=1)
        Xv = Xv[finite_rows]
        y = y[finite_rows]

        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if n_classes < 2:
            return self._empty_result(
                representation, group_name, "skipped", "fewer than 2 classes", n_items, len(y), n_classes
            )
        if counts.min() < self.min_class_count:
            return self._empty_result(
                representation,
                group_name,
                "skipped",
                f"min class count {counts.min()} < min_class_count {self.min_class_count}",
                n_items,
                len(y),
                n_classes,
            )

        Xv, y = self._subsample_by_class(Xv, y)
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if n_classes < 2 or counts.min() < 2:
            return self._empty_result(
                representation, group_name, "skipped", "insufficient classes after subsampling", n_items, len(y), n_classes
            )

        stratify = y if counts.min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                Xv,
                y,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=stratify,
            )
        except ValueError as exc:
            return self._empty_result(
                representation, group_name, "skipped", f"train_test_split failed: {exc}", n_items, len(y), n_classes
            )

        clf = self._make_classifier()
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        except Exception as exc:
            return self._empty_result(
                representation, group_name, "failed", f"probe training failed: {exc}", n_items, len(y), n_classes
            )

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        ll = np.nan
        if hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X_test)
                ll = float(log_loss(y_test, proba, labels=np.unique(y_train)))
            except Exception:
                ll = np.nan

        majority_class = int(pd.Series(y_train).mode().iloc[0])
        return LeakageProbeResult(
            representation=representation,
            group_name=group_name,
            status="ok",
            message="",
            n_items=int(n_items),
            n_valid=int(len(y)),
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            n_classes=int(n_classes),
            min_class_count=int(counts.min()),
            max_class_count=int(counts.max()),
            majority_class=majority_class,
            majority_baseline_accuracy=float(accuracy_score(y_test, y_dummy)),
            majority_baseline_balanced_accuracy=float(balanced_accuracy_score(y_test, y_dummy)),
            majority_baseline_macro_f1=float(f1_score(y_test, y_dummy, average="macro", zero_division=0)),
            accuracy=float(accuracy_score(y_test, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
            macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            weighted_f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            micro_f1=float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
            log_loss=ll,
        )

    def evaluate_many(
        self,
        representations: Dict[str, np.ndarray],
        group_labels: Dict[str, np.ndarray],
        item_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for rep_name, X in representations.items():
            for group_name, labels in group_labels.items():
                result = self.evaluate_one(
                    X=X,
                    labels=labels,
                    group_name=group_name,
                    representation=rep_name,
                    item_ids=item_ids,
                )
                rows.append(result.to_dict())
        return pd.DataFrame(rows)


def add_relative_leakage_columns(
    df: pd.DataFrame,
    reference_representation: str = "mm_raw",
    fair_representation: str = "z_fair",
    bias_representation: str = "",
) -> pd.DataFrame:
    """Add reference-relative leakage columns.

    For every group, uses reference_representation as baseline and computes:
        reduction = (reference_metric - metric) / reference_metric

    Positive reduction means the current representation leaks less than the reference.
    If bias_representation is provided and exists, also computes bias_minus_fair.
    """
    out = df.copy()
    key_cols = [c for c in ["dataset", "model_name", "run_id", "group_name"] if c in out.columns]
    if not key_cols:
        key_cols = ["group_name"]

    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        out[f"reference_{metric}"] = np.nan
        out[f"relative_reduction/{metric}"] = np.nan
        out[f"bias_minus_fair/{metric}"] = np.nan

    grouped = out.groupby(key_cols, dropna=False)
    for _, idx in grouped.groups.items():
        sub = out.loc[idx]
        ref = sub[sub["representation"] == reference_representation]
        fair = sub[sub["representation"] == fair_representation]
        bias = sub[sub["representation"] == bias_representation] if bias_representation else sub.iloc[0:0]

        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            if len(ref) > 0:
                ref_val = float(ref.iloc[0].get(metric, np.nan))
                out.loc[idx, f"reference_{metric}"] = ref_val
                if np.isfinite(ref_val) and abs(ref_val) > 1e-12:
                    out.loc[idx, f"relative_reduction/{metric}"] = (ref_val - out.loc[idx, metric].astype(float)) / ref_val
            if len(fair) > 0 and len(bias) > 0:
                fair_val = float(fair.iloc[0].get(metric, np.nan))
                bias_val = float(bias.iloc[0].get(metric, np.nan))
                if np.isfinite(fair_val) and np.isfinite(bias_val):
                    out.loc[idx, f"bias_minus_fair/{metric}"] = bias_val - fair_val

    return out
