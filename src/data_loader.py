# -*- coding: utf-8 -*-
"""
src/data_loader.py

Data loading utilities for sequential recommendation experiments.

Expected processed split files:
    train.txt / val.txt / test.txt

Each line format:
    user_id item_1 item_2 ... item_T

For SASRec training, train.txt is expanded into prefix -> next-item examples:
    prefix = [item_1, ..., item_{t-1}], target = item_t

The collate function left-pads prefixes to max_seq_len. Item id 0 is reserved for padding.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class UserSequence:
    user_id: int
    items: List[int]


@dataclass(frozen=True)
class TrainExampleIndex:
    seq_idx: int
    target_pos: int


def read_sequence_file(path: str, min_items: int = 1) -> List[UserSequence]:
    """Read sequence file into UserSequence objects.

    Args:
        path: Split path.
        min_items: Minimum number of items after user_id. Lines below this are skipped.

    Returns:
        List of UserSequence.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence file not found: {path}")

    sequences: List[UserSequence] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                ids = [int(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {path}:{line_no}: {line[:120]}") from exc
            if len(ids) < 1 + min_items:
                continue
            user_id = ids[0]
            items = [x for x in ids[1:] if x > 0]
            if user_id <= 0 or len(items) < min_items:
                continue
            sequences.append(UserSequence(user_id=user_id, items=items))
    return sequences


def left_pad_sequence(seq: Sequence[int], max_len: int, pad_id: int = 0) -> np.ndarray:
    """Right-pad one sequence after keeping the most recent max_len items.

    Historical note: the function name is kept for backward compatibility, but
    the implementation intentionally uses right padding. With PyTorch's
    TransformerEncoder, left padding combined with a causal mask and a key padding
    mask can create fully-masked attention rows for padding queries, which may
    produce NaNs. Right padding avoids this failure mode, and the model gathers
    the final valid hidden state using lengths.
    """
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    arr = np.full(max_len, pad_id, dtype=np.int64)
    clean = [int(x) for x in seq if int(x) > 0]

    if not clean:
        return arr

    clean = clean[-max_len:]
    arr[: len(clean)] = np.asarray(clean, dtype=np.int64)
    return arr


class SASRecTrainDataset(Dataset):
    """Prefix-to-next-item training dataset for SASRec.

    For each user sequence [i1, i2, ..., iT] in train.txt, examples are:
        [i1] -> i2
        [i1, i2] -> i3
        ...
        [i1, ..., i_{T-1}] -> iT

    This is a single-target full-softmax training setup.
    """

    def __init__(
        self,
        train_path: str,
        max_seq_len: int = 50,
        min_prefix_len: int = 1,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if min_prefix_len <= 0:
            raise ValueError(f"min_prefix_len must be positive, got {min_prefix_len}")

        self.train_path = train_path
        self.max_seq_len = int(max_seq_len)
        self.min_prefix_len = int(min_prefix_len)
        self.sequences = read_sequence_file(train_path, min_items=min_prefix_len + 1)

        self.examples: List[TrainExampleIndex] = []
        for seq_idx, user_seq in enumerate(self.sequences):
            # target_pos indexes into user_seq.items.
            for target_pos in range(self.min_prefix_len, len(user_seq.items)):
                self.examples.append(TrainExampleIndex(seq_idx=seq_idx, target_pos=target_pos))

        if not self.examples:
            raise ValueError(f"No training examples built from {train_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        user_seq = self.sequences[ex.seq_idx]
        prefix = user_seq.items[: ex.target_pos]
        target = user_seq.items[ex.target_pos]
        seq_arr = left_pad_sequence(prefix, self.max_seq_len, pad_id=0)
        length = min(len(prefix), self.max_seq_len)
        return {
            "user_id": user_seq.user_id,
            "sequence": seq_arr,
            "length": length,
            "target": target,
        }


def sasrec_collate_fn(batch: List[dict]) -> dict:
    """Collate SASRec training batch."""
    user_ids = torch.tensor([b["user_id"] for b in batch], dtype=torch.long)
    sequences = torch.tensor(np.stack([b["sequence"] for b in batch], axis=0), dtype=torch.long)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    return {
        "user_ids": user_ids,
        "sequences": sequences,
        "lengths": lengths,
        "targets": targets,
    }


def seed_worker(worker_id: int) -> None:
    """Deterministic dataloader worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_sasrec_train_loader(
    train_path: str,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 2026,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[SASRecTrainDataset, DataLoader]:
    """Build SASRec training Dataset and DataLoader."""
    dataset = SASRecTrainDataset(train_path=train_path, max_seq_len=max_seq_len)
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sasrec_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=bool(num_workers > 0),
    )
    return dataset, loader


def count_train_interactions(train_path: str) -> int:
    """Count non-padding item occurrences in train.txt."""
    total = 0
    for seq in read_sequence_file(train_path, min_items=1):
        total += len(seq.items)
    return total
