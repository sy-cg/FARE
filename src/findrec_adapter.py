# -*- coding: utf-8 -*-
"""
Utilities for adapting this project's processed datasets to FindRec/RecBole.

The project stores one line per user in train/val/test text files:
    user_id item_1 item_2 ... item_T

FindRec's released code expects RecBole-style interaction files and item
feature matrices. These helpers keep the conversion explicit and reproducible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SequenceRow = Tuple[int, List[int]]


def slugify_dataset_name(dataset_name: str) -> str:
    """Return the RecBole dataset name used for exported FindRec assets."""
    base = re.sub(r"[^A-Za-z0-9]+", "_", str(dataset_name).strip()).strip("_").lower()
    if not base:
        raise ValueError("dataset_name cannot be empty")
    return f"fare_{base}_findrec"


def read_sequence_rows(path: str | Path, min_items: int = 1) -> List[SequenceRow]:
    """Read a project sequence file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")

    rows: List[SequenceRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                ids = [int(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {path}:{line_no}: {line[:120]}") from exc

            user_id = int(ids[0])
            items = [int(x) for x in ids[1:] if int(x) > 0]
            if user_id <= 0 or len(items) < int(min_items):
                continue
            rows.append((user_id, items))
    return rows


def iter_recbole_interactions(rows: Iterable[SequenceRow]) -> Iterable[Tuple[int, int, int]]:
    """Yield RecBole interaction records with deterministic per-user timestamps."""
    for user_id, items in rows:
        for pos, item_id in enumerate(items, start=1):
            yield int(user_id), int(item_id), int(pos)


def write_recbole_inter_from_sequence_file(split_path: str | Path, output_path: str | Path) -> Dict[str, int]:
    """Write one RecBole .inter file from a project sequence split file.

    For FindRec comparison we export test.txt because it contains the complete
    train-prefix + validation target + test target sequence under the current
    leave-one-out preprocessing protocol. RecBole's leave-one-out split can then
    reconstruct the same chronological validation/test endpoints.
    """
    rows = read_sequence_rows(split_path, min_items=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_interactions = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("user_id:token,item_id:token,timestamp:float\n")
        for user_id, item_id, timestamp in iter_recbole_interactions(rows):
            f.write(f"{user_id},{item_id},{timestamp}\n")
            num_interactions += 1

    return {
        "num_users": len(rows),
        "num_interactions": num_interactions,
    }


def render_findrec_config(
    *,
    dataset_name: str,
    recbole_dataset_name: str,
    image_feature_path: str,
    text_feature_path: str,
    image_feature_dim: int,
    text_feature_dim: int,
    seed: int = 2026,
    max_seq_len: int = 50,
    epochs: int = 100,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
) -> str:
    """Render a FindRec config aligned with this project's exported assets."""
    return f"""# Auto-generated FindRec config for {dataset_name}.
# Generated from this project's processed train/val/test split files.

gpu_id: '0'
device: cuda
seed: {int(seed)}
reproducibility: true
log_wandb: false
checkpoint_dir: saved
show_progress: true

id_embedding_dim: 128
num_attention_heads: 8
hidden_size: 128
dropout_prob: 0.2
loss_type: CE

bottleneck:
  dim: 256
  beta: 1.0
  weight: 0.1
  use_stein: true
  kernel_type: rbf
  bandwidth_factor: 1.0
  entropy_weight: 0.01
  adaptive_bandwidth: true
  min_bandwidth: 0.1
  max_bandwidth: 10.0
  alignment_dropout: 0.1
  layer_norm_eps: 1.0e-12
  score_hidden_ratio: 2

stein:
  kernel_type: rbf
  adaptive_bandwidth: true
  min_bandwidth: 0.1
  max_bandwidth: 10.0

mamba:
  d_state: 16
  d_conv: 4
  expand: 2
  norm_eps: 1.0e-5
  hidden_dim: 128
  num_layers: 2

feed_forward:
  d_ff: 512
  dropout: 0.2
  layer_norm_eps: 1.0e-12

attention:
  num_heads: 8
  dropout: 0.2
  use_bias: true

multimodal:
  hidden_size: 256
  projection_dropout: 0.1
  fusion_dropout: 0.1
  feature_dropout: 0.1
  alignment_hidden: 256
  alignment_dropout: 0.2
  use_bottleneck: true

expert:
  num_experts: 4
  hidden_size: 256
  output_dim: 128
  dropout: 0.2

router:
  hidden_size: 256
  dropout: 0.1

text:
  feature_dim: {int(text_feature_dim)}
  projection_dim: 256
  layer_norm_eps: 1.0e-12

image:
  feature_dim: {int(image_feature_dim)}
  projection_dim: 256
  layer_norm_eps: 1.0e-12

feature_fusion:
  hidden_size: 256
  dropout: 0.2
  activation: gelu
  use_residual: true

feature_paths:
  image_feature_path: "{image_feature_path}"
  text_feature_path: "{text_feature_path}"

dataset: {recbole_dataset_name}
data_path: dataset
MAX_ITEM_LIST_LENGTH: {int(max_seq_len)}
field_separator: ","
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
load_col:
  inter: [user_id, item_id, timestamp]
user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"

epochs: {int(epochs)}
train_batch_size: {int(train_batch_size)}
eval_batch_size: {int(eval_batch_size)}
gradient_clip_norm: 2.0
learner: adamW
learning_rate: 0.001
weight_decay: 0.01
scheduler: cosine
warmup_steps: 2000
gradient_accumulation_steps: 1
eval_step: 1
stopping_step: 5
train_neg_sample_args: ~

eval_args:
  split:
    LS: valid_and_test
  order: TO
  group_by: user
  mode: full
metrics: [Hit, NDCG, MRR]
valid_metric: NDCG@10
topk: [5, 10, 20, 100]

memory_optimize: true
cache_size: 3
clear_cache_step: 100
num_workers: 4
prefetch_factor: 2
pin_memory: true
"""


def _copy_feature_without_padding(src_path: Path, dst_path: Path) -> int:
    """Copy an item feature matrix after dropping padding row 0.

    The released FindRec code indexes feature row as int(item_token) - 1.
    This project stores features as [num_items + 1, dim] with row 0 reserved for
    padding, so the exported matrix must drop row 0.
    """
    import numpy as np

    arr = np.load(src_path).astype("float32")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix at {src_path}, got {arr.shape}")
    if arr.shape[0] <= 1:
        raise ValueError(f"Feature matrix must include padding row plus items: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, arr[1:])
    return int(arr.shape[1])


def prepare_findrec_assets(
    *,
    project_data_dir: str | Path,
    findrec_dir: str | Path,
    dataset_name: str,
    seed: int = 2026,
    max_seq_len: int = 50,
    epochs: int = 100,
) -> Dict[str, Any]:
    """Export RecBole interaction/features/config files for FindRec."""
    project_data_dir = Path(project_data_dir)
    findrec_dir = Path(findrec_dir)
    if not project_data_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {project_data_dir}")
    if not findrec_dir.exists():
        raise FileNotFoundError(f"FindRec directory not found: {findrec_dir}")

    recbole_dataset_name = slugify_dataset_name(dataset_name)
    dataset_dir = findrec_dir / "dataset"
    emb_dir = findrec_dir / "emb"
    inter_path = dataset_dir / f"{recbole_dataset_name}.inter"
    image_path = emb_dir / f"{recbole_dataset_name}_image_features.npy"
    text_path = emb_dir / f"{recbole_dataset_name}_text_features.npy"
    config_path = findrec_dir / f"config_{recbole_dataset_name}.yaml"

    inter_stats = write_recbole_inter_from_sequence_file(project_data_dir / "test.txt", inter_path)
    image_dim = _copy_feature_without_padding(project_data_dir / "vision_features.npy", image_path)
    text_dim = _copy_feature_without_padding(project_data_dir / "text_features.npy", text_path)

    config_text = render_findrec_config(
        dataset_name=dataset_name,
        recbole_dataset_name=recbole_dataset_name,
        image_feature_path=str(image_path.relative_to(findrec_dir)).replace("\\", "/"),
        text_feature_path=str(text_path.relative_to(findrec_dir)).replace("\\", "/"),
        image_feature_dim=image_dim,
        text_feature_dim=text_dim,
        seed=seed,
        max_seq_len=max_seq_len,
        epochs=epochs,
    )
    config_path.write_text(config_text, encoding="utf-8", newline="\n")

    manifest = {
        "dataset_name": dataset_name,
        "recbole_dataset_name": recbole_dataset_name,
        "source_data_dir": str(project_data_dir),
        "findrec_dir": str(findrec_dir),
        "inter_path": str(inter_path),
        "image_feature_path": str(image_path),
        "text_feature_path": str(text_path),
        "config_path": str(config_path),
        "image_feature_dim": image_dim,
        "text_feature_dim": text_dim,
        **inter_stats,
    }
    manifest_path = findrec_dir / f"manifest_{recbole_dataset_name}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
