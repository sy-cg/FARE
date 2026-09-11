# -*- coding: utf-8 -*-
"""
Prepare MicroLens-100K for the FARE experiment layout.

The script reads the official MicroLens-100K files and writes:
    data/Processed_MicroLens_100K/train.txt
    data/Processed_MicroLens_100K/val.txt
    data/Processed_MicroLens_100K/test.txt
    text_features.npy / vision_features.npy with row 0 reserved for padding
    fairness group files consumed by scripts/run_fare.py and evaluator_fairness.py

Only text and image modalities are used. Video frame and audio assets are audited
when present, but they are not required for this processed output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_USERS = 100_000
EXPECTED_ITEMS = 19_738
EXPECTED_INTERACTIONS = 719_405


def split_leave_one_out(sequence: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    """Return cumulative train/validation/test sequences."""
    seq = [int(x) for x in sequence]
    if len(seq) < 3:
        raise ValueError(f"Need at least 3 interactions for leave-one-out, got {len(seq)}")
    return seq[:-2], seq[:-1], seq


def with_padding_row(
    features: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepend zero padding row and return item-valid mask."""
    arr = np.asarray(features)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape={arr.shape}")

    out = np.zeros((arr.shape[0] + 1, arr.shape[1]), dtype=arr.dtype)
    out[1:] = arr

    if valid_mask is None:
        mask = np.ones(arr.shape[0] + 1, dtype=bool)
        mask[0] = False
    else:
        item_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if item_mask.shape[0] != arr.shape[0]:
            raise ValueError(f"valid_mask length mismatch: {item_mask.shape[0]} vs {arr.shape[0]}")
        mask = np.zeros(arr.shape[0] + 1, dtype=bool)
        mask[1:] = item_mask
        out[~mask] = 0

    return out, mask


def tertile_labels_from_counts(
    counts: np.ndarray,
    q1: Optional[float] = None,
    q2: Optional[float] = None,
) -> np.ndarray:
    """Map item counts to tail/middle/head labels; padding stays -1."""
    arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("counts must include padding plus at least one item")

    positive = arr[1:][arr[1:] > 0]
    if positive.size == 0:
        raise ValueError("Cannot build tertile labels from all-zero counts")

    if q1 is None or q2 is None:
        q1, q2 = np.quantile(positive, [1.0 / 3.0, 2.0 / 3.0])

    labels = np.full(arr.shape[0], -1, dtype=np.int64)
    item_values = arr[1:]
    labels[1:][item_values <= float(q1)] = 0
    labels[1:][(item_values > float(q1)) & (item_values <= float(q2))] = 1
    labels[1:][item_values > float(q2)] = 2
    return labels


def quantile_labels_from_values(values: np.ndarray, num_groups: int = 3) -> Tuple[np.ndarray, List[float]]:
    """Return quantile labels for positive item-side values."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("values must include padding plus at least one item")
    if num_groups < 2:
        raise ValueError(f"num_groups must be >=2, got {num_groups}")

    observed = arr[1:]
    finite = observed[np.isfinite(observed)]
    if finite.size == 0:
        raise ValueError("No finite values for quantile grouping")

    cuts = [float(x) for x in np.quantile(finite, [i / num_groups for i in range(1, num_groups)])]
    labels = np.full(arr.shape[0], -1, dtype=np.int64)
    labels[1:] = np.searchsorted(np.asarray(cuts, dtype=np.float64), observed, side="right")
    return labels, cuts


def build_group_matrix(
    groups: Mapping[str, Mapping[str, int]],
    group_value_names: Mapping[str, Mapping[str, str]],
    num_items: int,
) -> Tuple[np.ndarray, List[Dict[str, int | str]]]:
    """Build one-hot item-group matrix and column metadata."""
    columns: List[Dict[str, int | str]] = []
    for group_name, item_to_label in groups.items():
        named_values = {int(x) for x in group_value_names.get(group_name, {}).keys()}
        observed_values = {int(v) for v in item_to_label.values() if int(v) >= 0}
        for value in sorted(named_values | observed_values):
            if value < 0:
                continue
            columns.append(
                {
                    "column": len(columns),
                    "group_name": str(group_name),
                    "group_value": int(value),
                }
            )

    matrix = np.zeros((num_items + 1, len(columns)), dtype=np.float32)
    column_lookup = {
        (str(entry["group_name"]), int(entry["group_value"])): int(entry["column"])
        for entry in columns
    }
    for group_name, item_to_label in groups.items():
        for item_str, label in item_to_label.items():
            item_id = int(item_str)
            if 1 <= item_id <= num_items and int(label) >= 0:
                col = column_lookup[(str(group_name), int(label))]
                matrix[item_id, col] = 1.0
    return matrix, columns


def compute_group_item_counts(groups: Mapping[str, Mapping[str, int]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for group_name, item_to_label in groups.items():
        counts = Counter(int(v) for v in item_to_label.values() if int(v) >= 0)
        out[group_name] = {str(k): int(counts[k]) for k in sorted(counts)}
    return out


def compute_group_utility(
    groups: Mapping[str, Mapping[str, int]],
    popularity: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    pop = np.asarray(popularity, dtype=np.float64).reshape(-1)
    out: Dict[str, Dict[str, float]] = {}
    for group_name, item_to_label in groups.items():
        totals: Dict[int, float] = {}
        for item_str, label in item_to_label.items():
            if int(label) < 0:
                continue
            item_id = int(item_str)
            if 0 <= item_id < pop.shape[0]:
                totals[int(label)] = totals.get(int(label), 0.0) + float(pop[item_id])
        group_total = float(sum(totals.values()))
        out[group_name] = {str(k): float(totals[k]) for k in sorted(totals)}
        out[group_name]["__total__"] = group_total
    return out


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_sequences_tsv(path: Path) -> Tuple[List[Tuple[int, List[int]]], Dict[str, Any]]:
    rows: List[Tuple[int, List[int]]] = []
    users: set[int] = set()
    items: set[int] = set()
    user_item_pairs: set[Tuple[int, int]] = set()
    duplicate_pairs = 0
    seq_lengths: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row at {path}:{line_no}: expected 2 columns")
            user_id = int(parts[0])
            sequence = [int(x) for x in parts[1].split() if x]
            if len(sequence) < 3:
                raise ValueError(f"User {user_id} has too few interactions: {len(sequence)}")
            users.add(user_id)
            seq_lengths.append(len(sequence))
            for item_id in sequence:
                items.add(item_id)
                key = (user_id, item_id)
                if key in user_item_pairs:
                    duplicate_pairs += 1
                user_item_pairs.add(key)
            rows.append((user_id, sequence))

    stats = {
        "num_users": len(users),
        "num_items": len(items),
        "num_interactions": int(sum(seq_lengths)),
        "duplicate_user_item_pairs": duplicate_pairs,
        "user_min": min(users) if users else None,
        "user_max": max(users) if users else None,
        "item_min": min(items) if items else None,
        "item_max": max(items) if items else None,
        "seq_len_min": min(seq_lengths) if seq_lengths else None,
        "seq_len_median": float(np.median(seq_lengths)) if seq_lengths else None,
        "seq_len_mean": float(np.mean(seq_lengths)) if seq_lengths else None,
        "seq_len_max": max(seq_lengths) if seq_lengths else None,
    }
    return rows, stats


def write_sequence_file(rows: Sequence[Tuple[int, Sequence[int]]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for user_id, sequence in rows:
            seq_text = " ".join(str(int(x)) for x in sequence)
            f.write(f"{int(user_id)} {seq_text}\n")


def read_title_csv(path: Path, num_items: int) -> Tuple[Dict[int, str], np.ndarray]:
    titles: Dict[int, str] = {}
    mask = np.zeros(num_items + 1, dtype=bool)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            item_id = int(row[0].strip())
            title = ",".join(row[1:]).strip() if len(row) > 1 else ""
            titles[item_id] = title
            if title:
                mask[item_id] = True
    return titles, mask


def read_tag_csv(path: Path, num_items: int) -> Dict[int, str]:
    tags: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            item_id = int(row[0].strip())
            tag = ",".join(row[1:]).strip() if len(row) > 1 else ""
            tags[item_id] = tag or "Unknown"
    for item_id in range(1, num_items + 1):
        tags.setdefault(item_id, "Unknown")
    return tags


def read_likes_views(path: Path, num_items: int) -> Tuple[np.ndarray, np.ndarray]:
    likes = np.zeros(num_items + 1, dtype=np.float64)
    views = np.zeros(num_items + 1, dtype=np.float64)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"Invalid likes/views row at {path}:{line_no}")
            item_id = int(parts[0])
            if 1 <= item_id <= num_items:
                likes[item_id] = float(parts[1])
                views[item_id] = float(parts[2])
    return likes, views


def zip_status(path: Path, test_crc: bool = True) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    status: Dict[str, Any] = {"exists": True, "size_bytes": path.stat().st_size}
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            status["entries"] = len(infos)
            status["first_entries"] = [info.filename for info in infos[:5]]
            status["last_entries"] = [info.filename for info in infos[-5:]]
            status["bad_entry"] = zf.testzip() if test_crc else None
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def load_feature_from_zip(zip_path: Path, member_name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name, "r") as src:
            payload = src.read()
    arr = np.load(io.BytesIO(payload), allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{member_name} must be 2-D, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{member_name} contains NaN or Inf")
    return np.asarray(arr, dtype=np.float32)


def cover_file_sizes(zip_path: Path, num_items: int) -> np.ndarray:
    sizes = np.zeros(num_items + 1, dtype=np.float64)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            stem = Path(info.filename).stem
            try:
                item_id = int(stem)
            except ValueError:
                continue
            if 1 <= item_id <= num_items:
                sizes[item_id] = float(info.file_size)
    return sizes


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_to_json_dict(labels: np.ndarray, num_items: int) -> Dict[str, int]:
    return {str(item_id): int(labels[item_id]) for item_id in range(1, num_items + 1)}


def make_top_value_labels(
    item_values: Mapping[int, str],
    num_items: int,
    top_n: int,
    min_count: int,
) -> Tuple[np.ndarray, Dict[str, str], List[Tuple[str, int]]]:
    counts = Counter(item_values.get(item_id, "Unknown") for item_id in range(1, num_items + 1))
    top_values = [
        value
        for value, count in counts.most_common()
        if value != "Unknown" and count >= int(min_count)
    ][: int(top_n)]

    value_to_label = {value: idx + 2 for idx, value in enumerate(top_values)}
    labels = np.full(num_items + 1, -1, dtype=np.int64)
    labels[1:] = 1
    for item_id in range(1, num_items + 1):
        value = item_values.get(item_id, "Unknown")
        labels[item_id] = value_to_label.get(value, 0 if value == "Unknown" else 1)

    names = {"0": "unknown", "1": "other"}
    for value, label in value_to_label.items():
        names[str(label)] = value
    return labels, names, counts.most_common()


def build_multimodal_clusters(
    text_features: np.ndarray,
    vision_features: np.ndarray,
    text_mask: np.ndarray,
    vision_mask: np.ndarray,
    num_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, str], Dict[str, Any]]:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import normalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for multimodal cluster groups") from exc

    valid = text_mask | vision_mask
    valid[0] = False
    if int(valid.sum()) < int(num_clusters):
        raise ValueError(f"Not enough valid multimodal items for {num_clusters} clusters")

    text_norm = normalize(text_features[valid].astype(np.float32), norm="l2", axis=1)
    vision_norm = normalize(vision_features[valid].astype(np.float32), norm="l2", axis=1)
    features = np.concatenate([text_norm, vision_norm], axis=1)

    model = MiniBatchKMeans(
        n_clusters=int(num_clusters),
        random_state=int(seed),
        batch_size=4096,
        n_init=10,
        reassignment_ratio=0.01,
    )
    cluster_values = model.fit_predict(features)

    labels = np.full(text_features.shape[0], -1, dtype=np.int64)
    labels[np.where(valid)[0]] = cluster_values.astype(np.int64)

    names = {str(i): f"mm_cluster_{i}" for i in range(int(num_clusters))}
    counts = Counter(int(x) for x in cluster_values)
    info = {
        "num_clusters": int(num_clusters),
        "valid_items": int(valid.sum()),
        "inertia": float(model.inertia_),
        "cluster_counts": {str(k): int(counts[k]) for k in sorted(counts)},
        "min_cluster_count": int(min(counts.values())) if counts else 0,
    }
    return labels, names, info


def make_item_text(
    titles: Mapping[int, str],
    tags: Mapping[int, str],
    num_items: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item_id in range(1, num_items + 1):
        title = titles.get(item_id, "").strip()
        tag = tags.get(item_id, "Unknown").strip() or "Unknown"
        if title:
            out[str(item_id)] = f"Tag: {tag}. Title: {title}."
        else:
            out[str(item_id)] = f"Tag: {tag}."
    return out


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def ensure_required_files(raw_dir: Path) -> Dict[str, Path]:
    required = {
        "pairs_tsv": raw_dir / "MicroLens-100k_pairs.tsv",
        "pairs_csv": raw_dir / "MicroLens-100k_pairs.csv",
        "titles": raw_dir / "MicroLens-100k_title_en.csv",
        "tags": raw_dir / "tags_to_summary.csv",
        "likes_views": raw_dir / "MicroLens-100k_likes_and_views.txt",
        "features_zip": raw_dir / "extracted_modality_features.zip",
        "covers_zip": raw_dir / "MicroLens-100k_covers.zip",
        "readme": raw_dir / "readme.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required MicroLens files:\n" + "\n".join(missing))
    return required


def validate_counts(name: str, actual: int, expected: int) -> Dict[str, Any]:
    return {"actual": int(actual), "expected": int(expected), "ok": int(actual) == int(expected)}


def prepare_microlens100k(
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    cluster_count: int = 20,
    top_categories: int = 20,
    min_proxy_group_size: int = 50,
    seed: int = 2026,
    hash_outputs: bool = False,
) -> Dict[str, Any]:
    paths = ensure_required_files(raw_dir)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite to regenerate files in place."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, interaction_stats = read_sequences_tsv(paths["pairs_tsv"])
    num_items = int(interaction_stats["item_max"])
    if num_items != EXPECTED_ITEMS:
        raise ValueError(f"Expected {EXPECTED_ITEMS} MicroLens items, got {num_items}")

    train_rows: List[Tuple[int, List[int]]] = []
    val_rows: List[Tuple[int, List[int]]] = []
    test_rows: List[Tuple[int, List[int]]] = []
    item_popularity = np.zeros(num_items + 1, dtype=np.int64)
    val_targets = np.zeros(num_items + 1, dtype=np.int64)
    test_targets = np.zeros(num_items + 1, dtype=np.int64)

    for user_id, sequence in sequences:
        train_seq, val_seq, test_seq = split_leave_one_out(sequence)
        train_rows.append((user_id, train_seq))
        val_rows.append((user_id, val_seq))
        test_rows.append((user_id, test_seq))
        np.add.at(item_popularity, np.asarray(train_seq, dtype=np.int64), 1)
        val_targets[val_seq[-1]] += 1
        test_targets[test_seq[-1]] += 1

    write_sequence_file(train_rows, output_dir / "train.txt")
    write_sequence_file(val_rows, output_dir / "val.txt")
    write_sequence_file(test_rows, output_dir / "test.txt")
    np.save(output_dir / "item_popularity_train.npy", item_popularity)

    titles, title_mask = read_title_csv(paths["titles"], num_items=num_items)
    tags = read_tag_csv(paths["tags"], num_items=num_items)
    likes, views = read_likes_views(paths["likes_views"], num_items=num_items)
    cover_sizes = cover_file_sizes(paths["covers_zip"], num_items=num_items)

    image_raw = load_feature_from_zip(paths["features_zip"], "MicroLens-100k_image_features_CLIPRN50.npy")
    text_raw = load_feature_from_zip(paths["features_zip"], "MicroLens-100k_title_en_text_features_BgeM3.npy")
    if image_raw.shape[0] != num_items or text_raw.shape[0] != num_items:
        raise ValueError(f"Feature row mismatch: image={image_raw.shape}, text={text_raw.shape}, items={num_items}")

    text_features, text_mask = with_padding_row(text_raw, valid_mask=title_mask[1:])
    vision_mask_raw = (cover_sizes[1:] > 0) & np.isfinite(image_raw).all(axis=1)
    vision_features, vision_mask = with_padding_row(image_raw, valid_mask=vision_mask_raw)
    np.save(output_dir / "text_features.npy", text_features)
    np.save(output_dir / "vision_features.npy", vision_features)
    np.save(output_dir / "text_mask.npy", text_mask)
    np.save(output_dir / "vision_mask.npy", vision_mask)

    item2id = {str(item_id): item_id for item_id in range(1, num_items + 1)}
    user2id = {str(user_id): user_id for user_id, _ in sequences}
    id2item = {str(item_id): str(item_id) for item_id in range(1, num_items + 1)}
    id2user = {str(user_id): str(user_id) for user_id, _ in sequences}
    save_json(item2id, output_dir / "item2id.json")
    save_json(user2id, output_dir / "user2id.json")
    save_json(id2item, output_dir / "id2item.json")
    save_json(id2user, output_dir / "id2user.json")
    save_json(make_item_text(titles, tags, num_items), output_dir / "item_text.json")

    item_metadata = {}
    for item_id in range(1, num_items + 1):
        item_metadata[str(item_id)] = {
            "video_id": item_id,
            "title_en": titles.get(item_id, ""),
            "tag_summary": tags.get(item_id, "Unknown"),
            "title_len": len(titles.get(item_id, "")),
            "has_title": bool(title_mask[item_id]),
            "has_cover": bool(cover_sizes[item_id] > 0),
            "cover_size_bytes": int(cover_sizes[item_id]),
            "likes": int(likes[item_id]),
            "views": int(views[item_id]),
            "train_popularity": int(item_popularity[item_id]),
        }
    save_json(item_metadata, output_dir / "item_metadata_compact.json")

    pop_positive = item_popularity[1:][item_popularity[1:] > 0]
    pop_q1, pop_q2 = [float(x) for x in np.quantile(pop_positive, [1.0 / 3.0, 2.0 / 3.0])]
    popularity_labels = tertile_labels_from_counts(item_popularity, pop_q1, pop_q2)

    view_labels, view_cuts = quantile_labels_from_values(views, num_groups=3)
    like_labels, like_cuts = quantile_labels_from_values(likes, num_groups=3)

    modality_labels = np.full(num_items + 1, -1, dtype=np.int64)
    for item_id in range(1, num_items + 1):
        has_text = bool(text_mask[item_id])
        has_vision = bool(vision_mask[item_id])
        modality_labels[item_id] = 2 if has_text and has_vision else 1 if has_text or has_vision else 0

    title_lengths = np.zeros(num_items + 1, dtype=np.float64)
    for item_id in range(1, num_items + 1):
        title_lengths[item_id] = float(len(titles.get(item_id, "")))
    text_quality_labels, title_len_cuts = quantile_labels_from_values(title_lengths, num_groups=3)

    vision_quality_labels, cover_size_cuts = quantile_labels_from_values(cover_sizes, num_groups=3)

    category_labels, category_names, category_counts = make_top_value_labels(
        tags,
        num_items=num_items,
        top_n=top_categories,
        min_count=min_proxy_group_size,
    )

    cluster_labels, cluster_names, cluster_info = build_multimodal_clusters(
        text_features=text_features,
        vision_features=vision_features,
        text_mask=text_mask,
        vision_mask=vision_mask,
        num_clusters=cluster_count,
        seed=seed,
    )

    label_arrays: Dict[str, np.ndarray] = {
        "popularity_group": popularity_labels,
        "platform_view_group": view_labels,
        "platform_like_group": like_labels,
        "modality_availability_group": modality_labels,
        "text_quality_group": text_quality_labels,
        "vision_quality_group": vision_quality_labels,
        "category_proxy_group": category_labels,
        "multimodal_cluster_proxy_group": cluster_labels,
    }
    fairness_groups = {
        group_name: labels_to_json_dict(labels, num_items)
        for group_name, labels in label_arrays.items()
    }

    group_value_names: Dict[str, Dict[str, str]] = {
        "popularity_group": {
            "0": "unseen_or_tail_train_popularity",
            "1": "middle_train_popularity",
            "2": "head_train_popularity",
        },
        "platform_view_group": {
            "0": "low_platform_views",
            "1": "middle_platform_views",
            "2": "high_platform_views",
        },
        "platform_like_group": {
            "0": "low_platform_likes",
            "1": "middle_platform_likes",
            "2": "high_platform_likes",
        },
        "modality_availability_group": {
            "0": "no_text_no_image",
            "1": "single_modality_available",
            "2": "text_and_image_available",
        },
        "text_quality_group": {
            "0": "short_or_empty_title",
            "1": "middle_title_length",
            "2": "long_title",
        },
        "vision_quality_group": {
            "0": "low_cover_size",
            "1": "middle_cover_size",
            "2": "high_cover_size",
        },
        "category_proxy_group": category_names,
        "multimodal_cluster_proxy_group": cluster_names,
    }

    item_group_matrix, matrix_columns = build_group_matrix(
        fairness_groups,
        group_value_names,
        num_items=num_items,
    )
    np.save(output_dir / "item_group_matrix.npy", item_group_matrix)
    np.savez_compressed(output_dir / "item_group_labels.npz", **label_arrays)

    group_item_counts = compute_group_item_counts(fairness_groups)
    group_utility = compute_group_utility(fairness_groups, item_popularity)

    group_schema = {
        "dataset": "MicroLens_100K",
        "important_note": (
            "These are item-side and content/platform proxy groups. They are not user sensitive "
            "attributes and they are not true impression logs. Popularity utility is computed "
            "from train.txt only."
        ),
        "group_value_names": group_value_names,
        "item_group_matrix_columns": matrix_columns,
        "minimum_group_size_for_proxy_groups": int(min_proxy_group_size),
        "top_categories": int(top_categories),
        "cluster_count": int(cluster_count),
        "group_sources": {
            "popularity_group": "train.txt item interaction counts only",
            "platform_view_group": "MicroLens-100k_likes_and_views.txt views tertiles",
            "platform_like_group": "MicroLens-100k_likes_and_views.txt likes tertiles",
            "modality_availability_group": "title non-empty mask and cover availability",
            "text_quality_group": "English title length tertiles",
            "vision_quality_group": "cover JPEG file-size tertiles",
            "category_proxy_group": "tags_to_summary.csv top categories with other bucket",
            "multimodal_cluster_proxy_group": "MiniBatchKMeans over normalized text and image features",
        },
        "cut_points": {
            "train_popularity": [pop_q1, pop_q2],
            "platform_views": view_cuts,
            "platform_likes": like_cuts,
            "title_length": title_len_cuts,
            "cover_size_bytes": cover_size_cuts,
        },
    }
    save_json(fairness_groups, output_dir / "fairness_groups.json")
    save_json(group_schema, output_dir / "group_schema.json")
    save_json(group_utility, output_dir / "group_utility_train.json")
    save_json(group_item_counts, output_dir / "group_item_counts.json")

    id_signature = hashlib.sha1(json.dumps(item2id, sort_keys=True).encode("utf-8")).hexdigest()
    text_manifest = {
        "dataset": "MicroLens_100K",
        "source_file": "extracted_modality_features.zip::MicroLens-100k_title_en_text_features_BgeM3.npy",
        "model": "BGE-M3 official MicroLens release",
        "raw_shape": list(text_raw.shape),
        "output_shape": list(text_features.shape),
        "dtype": str(text_features.dtype),
        "num_items": num_items,
        "padding_id": 0,
        "valid_items": int(text_mask[1:].sum()),
        "coverage": float(text_mask[1:].mean()),
        "empty_title_items": int((~title_mask[1:]).sum()),
        "item2id_signature": id_signature,
    }
    vision_manifest = {
        "dataset": "MicroLens_100K",
        "source_file": "extracted_modality_features.zip::MicroLens-100k_image_features_CLIPRN50.npy",
        "model": "CLIP-RN50 official MicroLens release",
        "raw_shape": list(image_raw.shape),
        "output_shape": list(vision_features.shape),
        "dtype": str(vision_features.dtype),
        "num_items": num_items,
        "padding_id": 0,
        "valid_items": int(vision_mask[1:].sum()),
        "coverage": float(vision_mask[1:].mean()),
        "item2id_signature": id_signature,
    }
    save_json(text_manifest, output_dir / "text_feature_manifest.json")
    save_json(vision_manifest, output_dir / "vision_feature_manifest.json")

    dataset_stats = {
        "dataset": "MicroLens_100K",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "num_users": int(interaction_stats["num_users"]),
        "num_items": num_items,
        "num_interactions_final": int(interaction_stats["num_interactions"]),
        "avg_seq_len": float(interaction_stats["seq_len_mean"]),
        "median_seq_len": float(interaction_stats["seq_len_median"]),
        "min_seq_len": int(interaction_stats["seq_len_min"]),
        "max_seq_len": int(interaction_stats["seq_len_max"]),
        "sparsity": float(1.0 - interaction_stats["num_interactions"] / (interaction_stats["num_users"] * num_items)),
        "split_protocol": "leave-one-out: train=seq[:-2], val=seq[:-1], test=seq",
        "train_interactions": int(item_popularity.sum()),
        "val_records": len(val_rows),
        "test_records": len(test_rows),
        "train_seen_items": int((item_popularity[1:] > 0).sum()),
        "train_unseen_items": int((item_popularity[1:] == 0).sum()),
        "val_cold_target_records": int(sum(1 for _, seq in val_rows if item_popularity[seq[-1]] == 0)),
        "test_cold_target_records": int(sum(1 for _, seq in test_rows if item_popularity[seq[-1]] == 0)),
        "text_model": text_manifest["model"],
        "text_dim": int(text_features.shape[1]),
        "text_coverage": text_manifest["coverage"],
        "vision_model": vision_manifest["model"],
        "vision_dim": int(vision_features.shape[1]),
        "vision_coverage": vision_manifest["coverage"],
        "fairness_assets": {
            "fairness_groups": True,
            "item_group_matrix_shape": list(item_group_matrix.shape),
            "group_names": list(fairness_groups.keys()),
            "utility_source": "train.txt item popularity only",
        },
        "item2id_signature": id_signature,
    }
    save_json(dataset_stats, output_dir / "dataset_stats.json")

    np.save(output_dir / "platform_likes.npy", likes)
    np.save(output_dir / "platform_views.npy", views)
    np.save(output_dir / "cover_size_bytes.npy", cover_sizes)

    output_files = sorted(p for p in output_dir.iterdir() if p.is_file())
    output_file_report = {
        p.name: {
            "size_bytes": int(p.stat().st_size),
            **({"sha256": sha256_file(p)} if hash_outputs else {}),
        }
        for p in output_files
    }

    report: Dict[str, Any] = {
        "dataset": "MicroLens_100K",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "raw_file_audit": {
            "features_zip": zip_status(paths["features_zip"], test_crc=True),
            "covers_zip": zip_status(paths["covers_zip"], test_crc=True),
            "frames_zip": zip_status(raw_dir / "MicroLens-100k_frames_interval_1_number_5.zip", test_crc=False),
            "comments_file_exists": (raw_dir / "MicroLens-100k_comment_en.txt").exists(),
        },
        "official_count_checks": {
            "users": validate_counts("users", interaction_stats["num_users"], EXPECTED_USERS),
            "items": validate_counts("items", interaction_stats["num_items"], EXPECTED_ITEMS),
            "interactions": validate_counts(
                "interactions",
                interaction_stats["num_interactions"],
                EXPECTED_INTERACTIONS,
            ),
        },
        "interaction_stats": interaction_stats,
        "split_stats": {
            "train_users": len(train_rows),
            "val_users": len(val_rows),
            "test_users": len(test_rows),
            "train_interactions": int(item_popularity.sum()),
            "val_interactions": int(sum(len(seq) for _, seq in val_rows)),
            "test_interactions": int(sum(len(seq) for _, seq in test_rows)),
            "train_seen_items": int((item_popularity[1:] > 0).sum()),
            "train_unseen_items": int((item_popularity[1:] == 0).sum()),
            "val_items_zero_in_train": int(((val_targets > 0) & (item_popularity == 0))[1:].sum()),
            "test_items_zero_in_train": int(((test_targets > 0) & (item_popularity == 0))[1:].sum()),
            "val_cold_target_records": dataset_stats["val_cold_target_records"],
            "test_cold_target_records": dataset_stats["test_cold_target_records"],
            "train_popularity_quantiles_nonzero": [
                float(x) for x in np.quantile(pop_positive, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
            ],
        },
        "feature_stats": {
            "text_raw_shape": list(text_raw.shape),
            "vision_raw_shape": list(image_raw.shape),
            "text_output_shape": list(text_features.shape),
            "vision_output_shape": list(vision_features.shape),
            "text_finite": bool(np.isfinite(text_features).all()),
            "vision_finite": bool(np.isfinite(vision_features).all()),
            "text_mask_true": int(text_mask.sum()),
            "vision_mask_true": int(vision_mask.sum()),
            "text_empty_titles": int((~title_mask[1:]).sum()),
            "vision_missing_covers": int((cover_sizes[1:] <= 0).sum()),
        },
        "group_stats": {
            "group_item_counts": group_item_counts,
            "group_utility_train": group_utility,
            "item_group_matrix_shape": list(item_group_matrix.shape),
            "cluster_info": cluster_info,
            "top_category_counts": [[k, int(v)] for k, v in category_counts[: int(top_categories)]],
        },
        "output_files": output_file_report,
        "warnings": [],
    }

    if report["raw_file_audit"]["frames_zip"].get("error"):
        report["warnings"].append(
            "Video frame zip is not readable as a standalone zip; ignored because only text and image are used."
        )
    if not report["raw_file_audit"]["comments_file_exists"]:
        report["warnings"].append(
            "MicroLens-100k_comment_en.txt is missing; ignored because title text features are used."
        )
    if dataset_stats["test_cold_target_records"] > 0:
        report["warnings"].append(
            "Some validation/test targets are unseen in train under leave-one-out; they are assigned to tail popularity."
        )
    if int(cluster_info["min_cluster_count"]) < int(min_proxy_group_size):
        report["warnings"].append(
            "At least one multimodal cluster is smaller than min_proxy_group_size; avoid over-interpreting it."
        )

    validation_path = output_dir / "preprocess_validation_report.json"
    save_json(report, validation_path)
    report["output_files"][validation_path.name] = {
        "size_bytes": int(validation_path.stat().st_size),
        **({"sha256": sha256_file(validation_path)} if hash_outputs else {}),
    }
    return report


def print_report(report: Mapping[str, Any]) -> None:
    checks = report["official_count_checks"]
    split = report["split_stats"]
    features = report["feature_stats"]
    group_stats = report["group_stats"]

    print("MicroLens_100K preprocessing validation")
    print(f"raw_dir: {report['raw_dir']}")
    print(f"output_dir: {report['output_dir']}")
    print("official_counts:")
    for key in ["users", "items", "interactions"]:
        row = checks[key]
        print(f"  {key}: {row['actual']} / expected {row['expected']} / ok={row['ok']}")
    print("splits:")
    print(f"  train/val/test users: {split['train_users']} / {split['val_users']} / {split['test_users']}")
    print(
        "  interactions train/val/test: "
        f"{split['train_interactions']} / {split['val_interactions']} / {split['test_interactions']}"
    )
    print(
        "  train seen/unseen items: "
        f"{split['train_seen_items']} / {split['train_unseen_items']}"
    )
    print(
        "  cold target records val/test: "
        f"{split['val_cold_target_records']} / {split['test_cold_target_records']}"
    )
    print("features:")
    print(f"  text raw/output: {features['text_raw_shape']} -> {features['text_output_shape']}")
    print(f"  vision raw/output: {features['vision_raw_shape']} -> {features['vision_output_shape']}")
    print(f"  text finite / vision finite: {features['text_finite']} / {features['vision_finite']}")
    print(f"  text valid items: {features['text_mask_true']} ({features['text_empty_titles']} empty titles)")
    print(f"  vision valid items: {features['vision_mask_true']} ({features['vision_missing_covers']} missing covers)")
    print("groups:")
    print(f"  item_group_matrix_shape: {group_stats['item_group_matrix_shape']}")
    for group_name, counts in group_stats["group_item_counts"].items():
        print(f"  {group_name}: {counts}")
    print("warnings:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"  - {warning}")
    else:
        print("  none")
    print("validation_report: preprocess_validation_report.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MicroLens-100K for FARE experiments")
    parser.add_argument("--raw-dir", default="data/MicroLens-100K")
    parser.add_argument("--output-dir", default="data/Processed_MicroLens_100K")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cluster-count", type=int, default=20)
    parser.add_argument("--top-categories", type=int, default=20)
    parser.add_argument("--min-proxy-group-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hash-outputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = prepare_microlens100k(
        raw_dir=resolve_path(args.raw_dir),
        output_dir=resolve_path(args.output_dir),
        overwrite=bool(args.overwrite),
        cluster_count=int(args.cluster_count),
        top_categories=int(args.top_categories),
        min_proxy_group_size=int(args.min_proxy_group_size),
        seed=int(args.seed),
        hash_outputs=bool(args.hash_outputs),
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
