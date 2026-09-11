# -*- coding: utf-8 -*-
"""
src/modality_debias.py

Generic Modality Debiasing module for multimodal recommendation baselines.

Based on:
    Improving Item-side Fairness of Multimodal Recommendation via Modality Debiasing,
    WWW 2024.

Core idea:
    Training:
        L = L_multi + L_vision + L_text

    Inference:
        Add fairness-aware counterfactual inference:
            debiased_score = average(TIE_vision, TIE_text)

This implementation is intentionally model-agnostic:
    - base_model only needs full_sort_scores(user_ids, sequences, lengths)
    - unimodal text / vision branches are attached externally
    - can wrap VBPR, BM3, FREEDOM, LATTICE, LateFusion-* models

For stability in this project, two inference modes are provided:

1. paper_product:
    Faithful approximation of the paper equations:
        real = y * sigmoid(y_v) * sigmoid(y_t)
        TIE_v = real - s_v * y_cf * sigmoid(y_v) * sigmoid(y_t_cf)
        TIE_t = real - s_t * y_cf * sigmoid(y_v_cf) * sigmoid(y_t)
        final = average(TIE_v, TIE_t)

2. rank_subtract:
    More numerically stable ranking variant:
        final = y - lambda * average(s_m * sigmoid(y_m))

Default config can choose either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Feature / popularity utilities
# ==========================================================


def load_item_feature_matrix(
    data_dir: str | Path,
    feature_file: str,
    num_items: int,
    mask_file: Optional[str] = None,
    normalize: bool = True,
    zero_missing: bool = True,
) -> torch.Tensor:
    data_dir = Path(data_dir)
    feature_path = data_dir / feature_file
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    arr = np.load(feature_path)
    if arr.ndim != 2:
        raise ValueError(f"Feature file must be 2-D, got {arr.shape}: {feature_path}")

    arr = arr.astype(np.float32, copy=False)

    if arr.shape[0] == num_items:
        arr = np.concatenate([np.zeros((1, arr.shape[1]), dtype=np.float32), arr], axis=0)
    elif arr.shape[0] > num_items + 1:
        arr = arr[: num_items + 1]
    elif arr.shape[0] < num_items + 1:
        raise ValueError(
            f"Feature rows mismatch: got={arr.shape[0]}, expected={num_items + 1} or {num_items}. "
            f"path={feature_path}"
        )

    arr[0] = 0.0

    if mask_file:
        mask_path = data_dir / mask_file
        if mask_path.exists():
            mask = np.load(mask_path).reshape(-1)
            if mask.shape[0] == num_items:
                mask = np.concatenate([np.zeros((1,), dtype=mask.dtype), mask], axis=0)
            elif mask.shape[0] > num_items + 1:
                mask = mask[: num_items + 1]
            elif mask.shape[0] < num_items + 1:
                raise ValueError(
                    f"Mask rows mismatch: got={mask.shape[0]}, expected={num_items + 1} or {num_items}. "
                    f"path={mask_path}"
                )
            valid = mask.astype(bool)
            valid[0] = False
            if zero_missing:
                arr[~valid] = 0.0

    x = torch.from_numpy(arr).float()
    if normalize:
        x = F.normalize(x, p=2, dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x[0] = 0.0
    return x


def compute_item_popularity_from_train(
    train_rows,
    num_items: int,
) -> torch.Tensor:
    counts = torch.zeros(num_items + 1, dtype=torch.float32)
    for _, seq in train_rows:
        for item in seq:
            item = int(item)
            if 0 < item <= num_items:
                counts[item] += 1.0
    counts[0] = 0.0
    return counts


def load_optional_labels(data_dir: str | Path, candidates) -> Optional[np.ndarray]:
    data_dir = Path(data_dir)
    for name in candidates:
        p = data_dir / name
        if p.exists():
            arr = np.load(p)
            return np.asarray(arr).reshape(-1)
    return None


def build_frequency_rank(
    data_dir: str | Path,
    train_rows,
    num_items: int,
    modality: str,
    rank_normalize: bool = True,
) -> torch.Tensor:
    """
    Build modality content frequency ranking.

    Preferred:
        use modality group labels if available:
            vision_quality_group.npy / text_quality_group.npy / ...
        group frequency = total train interactions of items in group

    Fallback:
        use item popularity ranking.

    Rank convention:
        high frequency -> rank 0
        low frequency  -> larger rank
    """
    data_dir = Path(data_dir)

    if modality == "vision":
        label_candidates = [
            "vision_content_group.npy",
            "vision_cluster_group.npy",
            "vision_quality_group.npy",
            "vision_group.npy",
        ]
    elif modality == "text":
        label_candidates = [
            "text_content_group.npy",
            "text_cluster_group.npy",
            "text_quality_group.npy",
            "text_group.npy",
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    labels = load_optional_labels(data_dir, label_candidates)
    pop = compute_item_popularity_from_train(train_rows, num_items=num_items).numpy()

    if labels is not None:
        labels = labels.astype(np.int64)
        if labels.shape[0] == num_items:
            labels = np.concatenate([np.array([-1], dtype=np.int64), labels], axis=0)
        elif labels.shape[0] > num_items + 1:
            labels = labels[: num_items + 1]
        elif labels.shape[0] < num_items + 1:
            labels = None

    if labels is not None:
        group_freq: Dict[int, float] = {}
        for item in range(1, num_items + 1):
            g = int(labels[item])
            if g < 0:
                continue
            group_freq[g] = group_freq.get(g, 0.0) + float(pop[item])

        sorted_groups = sorted(group_freq.keys(), key=lambda g: group_freq[g], reverse=True)
        group_rank = {g: r for r, g in enumerate(sorted_groups)}

        ranks = np.full(num_items + 1, fill_value=len(sorted_groups), dtype=np.float32)
        for item in range(1, num_items + 1):
            g = int(labels[item])
            ranks[item] = float(group_rank.get(g, len(sorted_groups)))
    else:
        # fallback: item popularity rank
        order = np.argsort(-pop)
        ranks = np.empty(num_items + 1, dtype=np.float32)
        ranks[order] = np.arange(num_items + 1, dtype=np.float32)
        ranks[0] = num_items

    if rank_normalize:
        denom = float(max(np.nanmax(ranks[1:]), 1.0))
        ranks = ranks / denom

    ranks[0] = float(np.nanmax(ranks[1:]) if num_items > 0 else 1.0)
    return torch.from_numpy(ranks.astype(np.float32))


# ==========================================================
# Loss helpers
# ==========================================================


def pairwise_bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def gather_pair_scores(
    full_scores: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = full_scores.gather(1, pos_items.long().view(-1, 1)).squeeze(1)
    neg = full_scores.gather(1, neg_items.long().view(-1, 1)).squeeze(1)
    return pos, neg


# ==========================================================
# Generic late fusion sequential base
# ==========================================================


class LateFusionSequentialBase(nn.Module):
    """
    Generic LateFusion wrapper for ID sequential backbones.

    It assumes id_backbone provides:
        full_sort_scores(user_ids, sequences, lengths)
        encode_sequence(sequences, lengths)

    If encode_sequence is unavailable, the wrapper falls back to ID-only scores.
    """

    def __init__(
        self,
        id_backbone: nn.Module,
        num_items: int,
        data_dir: str | Path,
        hidden_size: int,
        mm_dim: int = 64,
        use_text: bool = True,
        use_vision: bool = True,
        text_feature_file: str = "text_features.npy",
        vision_feature_file: str = "vision_features.npy",
        text_mask_file: str = "text_mask.npy",
        vision_mask_file: str = "vision_mask.npy",
        normalize_features: bool = True,
        zero_missing_features: bool = True,
        text_weight: float = 0.05,
        vision_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.id_backbone = id_backbone
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.mm_dim = int(mm_dim)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)
        self.text_weight = float(text_weight)
        self.vision_weight = float(vision_weight)

        if self.use_text:
            text = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=text_feature_file,
                num_items=num_items,
                mask_file=text_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.text_feature = nn.Embedding.from_pretrained(text, freeze=True, padding_idx=0)
            self.text_proj = nn.Linear(text.shape[1], mm_dim)
            self.text_query = nn.Linear(hidden_size, mm_dim, bias=False)
        else:
            self.text_feature = None
            self.text_proj = None
            self.text_query = None

        if self.use_vision:
            vision = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=vision_feature_file,
                num_items=num_items,
                mask_file=vision_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.vision_feature = nn.Embedding.from_pretrained(vision, freeze=True, padding_idx=0)
            self.vision_proj = nn.Linear(vision.shape[1], mm_dim)
            self.vision_query = nn.Linear(hidden_size, mm_dim, bias=False)
        else:
            self.vision_feature = None
            self.vision_proj = None
            self.vision_query = None

    def _encode_sequence(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if hasattr(self.id_backbone, "encode_sequence"):
            return self.id_backbone.encode_sequence(sequences, lengths)
        return None

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.id_backbone.full_sort_scores(user_ids, sequences, lengths)
        h = self._encode_sequence(sequences, lengths)
        if h is None:
            scores[:, 0] = -1.0e9
            return scores

        if self.use_text and self.text_feature is not None:
            q = F.normalize(self.text_query(h), p=2, dim=-1)
            z = F.normalize(self.text_proj(self.text_feature.weight), p=2, dim=-1)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z[0] = 0.0
            scores = scores + self.text_weight * torch.matmul(q, z.t())

        if self.use_vision and self.vision_feature is not None:
            q = F.normalize(self.vision_query(h), p=2, dim=-1)
            z = F.normalize(self.vision_proj(self.vision_feature.weight), p=2, dim=-1)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z[0] = 0.0
            scores = scores + self.vision_weight * torch.matmul(q, z.t())

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = -1.0e9
        return scores


# ==========================================================
# Modality Debiasing wrapper
# ==========================================================


class ModalityDebiasWrapper(nn.Module):
    """
    Generic Modality Debiasing wrapper.

    base_model:
        Any model with full_sort_scores(user_ids, sequences, lengths).

    Unimodal branches:
        text_score(u, i)   = <p_u^t, q_i^t>
        vision_score(u, i) = <p_u^v, q_i^v>
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_users: int,
        num_items: int,
        data_dir: str | Path,
        train_rows,
        branch_dim: int = 64,
        use_text: bool = True,
        use_vision: bool = True,
        text_feature_file: str = "text_features.npy",
        vision_feature_file: str = "vision_features.npy",
        text_mask_file: str = "text_mask.npy",
        vision_mask_file: str = "vision_mask.npy",
        normalize_features: bool = True,
        zero_missing_features: bool = True,
        branch_dropout: float = 0.0,
        branch_loss_weight: float = 1.0,
        alpha: float = 5.0,
        debias_lambda: float = 1.0,
        counterfactual_mode: str = "paper_product",
        base_counterfactual: str = "user_mean",
        cf_prob: float = 0.5,
        rank_normalize: bool = True,
        debias_min: float = 0.0,
        debias_max: float = 1.0,
    ) -> None:
        super().__init__()

        if not use_text and not use_vision:
            raise ValueError("At least one modality must be enabled.")

        self.base_model = base_model
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.branch_dim = int(branch_dim)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)
        self.branch_loss_weight = float(branch_loss_weight)
        self.alpha = float(alpha)
        self.debias_lambda = float(debias_lambda)
        self.counterfactual_mode = str(counterfactual_mode)
        self.base_counterfactual = str(base_counterfactual)
        self.cf_prob = float(cf_prob)
        self.rank_normalize = bool(rank_normalize)
        self.debias_min = float(debias_min)
        self.debias_max = float(debias_max)

        self.dropout = nn.Dropout(float(branch_dropout))

        if self.use_text:
            text = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=text_feature_file,
                num_items=num_items,
                mask_file=text_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.text_item_feature = nn.Embedding.from_pretrained(text, freeze=True, padding_idx=0)
            self.text_item_proj = nn.Linear(text.shape[1], branch_dim)
            self.text_user = nn.Embedding(num_users + 1, branch_dim, padding_idx=0)
            self.text_item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
            text_rank = build_frequency_rank(
                data_dir=data_dir,
                train_rows=train_rows,
                num_items=num_items,
                modality="text",
                rank_normalize=rank_normalize,
            )
            self.register_buffer("text_freq_rank", text_rank, persistent=False)
        else:
            self.text_item_feature = None
            self.text_item_proj = None
            self.text_user = None
            self.text_item_bias = None
            self.register_buffer("text_freq_rank", torch.zeros(num_items + 1), persistent=False)

        if self.use_vision:
            vision = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=vision_feature_file,
                num_items=num_items,
                mask_file=vision_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.vision_item_feature = nn.Embedding.from_pretrained(vision, freeze=True, padding_idx=0)
            self.vision_item_proj = nn.Linear(vision.shape[1], branch_dim)
            self.vision_user = nn.Embedding(num_users + 1, branch_dim, padding_idx=0)
            self.vision_item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
            vision_rank = build_frequency_rank(
                data_dir=data_dir,
                train_rows=train_rows,
                num_items=num_items,
                modality="vision",
                rank_normalize=rank_normalize,
            )
            self.register_buffer("vision_freq_rank", vision_rank, persistent=False)
        else:
            self.vision_item_feature = None
            self.vision_item_proj = None
            self.vision_user = None
            self.vision_item_bias = None
            self.register_buffer("vision_freq_rank", torch.zeros(num_items + 1), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.text_user is not None:
            nn.init.xavier_uniform_(self.text_user.weight)
            nn.init.zeros_(self.text_item_bias.weight)
            self.text_user.weight.data[0].zero_()
            self.text_item_bias.weight.data[0].zero_()

        if self.vision_user is not None:
            nn.init.xavier_uniform_(self.vision_user.weight)
            nn.init.zeros_(self.vision_item_bias.weight)
            self.vision_user.weight.data[0].zero_()
            self.vision_item_bias.weight.data[0].zero_()

        if self.text_item_proj is not None:
            nn.init.xavier_uniform_(self.text_item_proj.weight)
            nn.init.zeros_(self.text_item_proj.bias)

        if self.vision_item_proj is not None:
            nn.init.xavier_uniform_(self.vision_item_proj.weight)
            nn.init.zeros_(self.vision_item_proj.bias)

    def _branch_item_embeddings(self, modality: str) -> torch.Tensor:
        if modality == "text":
            assert self.text_item_feature is not None and self.text_item_proj is not None
            z = self.text_item_proj(self.dropout(self.text_item_feature.weight))
        elif modality == "vision":
            assert self.vision_item_feature is not None and self.vision_item_proj is not None
            z = self.vision_item_proj(self.dropout(self.vision_item_feature.weight))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        z = F.normalize(z, p=2, dim=-1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z[0] = 0.0
        return z

    def unimodal_full_scores(self, user_ids: torch.Tensor, modality: str) -> torch.Tensor:
        user_ids = user_ids.long()

        if modality == "text":
            if not self.use_text:
                return user_ids.new_zeros((user_ids.numel(), self.num_items + 1), dtype=torch.float32)
            assert self.text_user is not None and self.text_item_bias is not None
            u = F.normalize(self.text_user(user_ids), p=2, dim=-1)
            z = self._branch_item_embeddings("text")
            scores = torch.matmul(u, z.t()) + self.text_item_bias.weight.squeeze(-1).unsqueeze(0)
        elif modality == "vision":
            if not self.use_vision:
                return user_ids.new_zeros((user_ids.numel(), self.num_items + 1), dtype=torch.float32)
            assert self.vision_user is not None and self.vision_item_bias is not None
            u = F.normalize(self.vision_user(user_ids), p=2, dim=-1)
            z = self._branch_item_embeddings("vision")
            scores = torch.matmul(u, z.t()) + self.vision_item_bias.weight.squeeze(-1).unsqueeze(0)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = -1.0e9
        return scores

    def base_full_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = self.base_model.full_sort_scores(user_ids, sequences, lengths)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = -1.0e9
        return scores

    def _score_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert scores to ranking values.

        high score -> rank 0
        low score  -> larger rank
        """
        order = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.argsort(order, dim=1).to(dtype=torch.float32)

        if self.rank_normalize:
            ranks = ranks / float(max(scores.size(1) - 1, 1))
        return ranks

    def _debias_strength(self, modality_scores: torch.Tensor, modality: str) -> torch.Tensor:
        score_rank = self._score_rank(modality_scores)

        if modality == "text":
            freq_rank = self.text_freq_rank.to(device=modality_scores.device, dtype=modality_scores.dtype)
        elif modality == "vision":
            freq_rank = self.vision_freq_rank.to(device=modality_scores.device, dtype=modality_scores.dtype)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        gap = torch.abs(score_rank - freq_rank.unsqueeze(0))
        strength = torch.exp(-self.alpha * gap)
        strength = strength.clamp(min=self.debias_min, max=self.debias_max)
        strength[:, 0] = 0.0
        return strength

    def _base_cf(self, base_scores: torch.Tensor) -> torch.Tensor:
        if self.base_counterfactual == "zero":
            return torch.zeros((base_scores.size(0), 1), device=base_scores.device, dtype=base_scores.dtype)
        if self.base_counterfactual == "user_mean":
            masked = base_scores.clone()
            masked[:, 0] = 0.0
            return masked.mean(dim=1, keepdim=True)
        if self.base_counterfactual == "user_median":
            masked = base_scores.clone()
            masked[:, 0] = 0.0
            return masked.median(dim=1, keepdim=True).values
        raise ValueError(f"Unknown base_counterfactual={self.base_counterfactual!r}")

    def debiased_full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        base = self.base_full_scores(user_ids, sequences, lengths)
        text = self.unimodal_full_scores(user_ids, "text") if self.use_text else torch.zeros_like(base)
        vision = self.unimodal_full_scores(user_ids, "vision") if self.use_vision else torch.zeros_like(base)

        if self.counterfactual_mode == "rank_subtract":
            components = []
            if self.use_vision:
                s_v = self._debias_strength(vision, "vision")
                components.append(s_v * torch.sigmoid(vision))
            if self.use_text:
                s_t = self._debias_strength(text, "text")
                components.append(s_t * torch.sigmoid(text))

            if components:
                penalty = torch.stack(components, dim=0).mean(dim=0)
                out = base - self.debias_lambda * penalty
            else:
                out = base

        elif self.counterfactual_mode == "paper_product":
            p_v = torch.sigmoid(vision) if self.use_vision else torch.ones_like(base)
            p_t = torch.sigmoid(text) if self.use_text else torch.ones_like(base)

            real = base * p_v * p_t
            base_cf = self._base_cf(base)
            cf_prob = base.new_tensor(float(self.cf_prob))

            ties = []

            if self.use_vision:
                s_v = self._debias_strength(vision, "vision")
                tie_v = real - self.debias_lambda * s_v * base_cf * p_v * cf_prob
                ties.append(tie_v)

            if self.use_text:
                s_t = self._debias_strength(text, "text")
                tie_t = real - self.debias_lambda * s_t * base_cf * cf_prob * p_t
                ties.append(tie_t)

            out = torch.stack(ties, dim=0).mean(dim=0) if ties else real

        else:
            raise ValueError(f"Unknown counterfactual_mode={self.counterfactual_mode!r}")

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, 0] = -1.0e9
        return out

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.debiased_full_sort_scores(user_ids, sequences, lengths)

    def training_loss(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        train_on_fused: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_scores = self.base_full_scores(user_ids, sequences, lengths)

        if train_on_fused:
            text_scores = self.unimodal_full_scores(user_ids, "text") if self.use_text else torch.zeros_like(base_scores)
            vision_scores = self.unimodal_full_scores(user_ids, "vision") if self.use_vision else torch.zeros_like(base_scores)
            main_scores = base_scores
            if self.use_vision:
                main_scores = main_scores * torch.sigmoid(vision_scores)
            if self.use_text:
                main_scores = main_scores * torch.sigmoid(text_scores)
        else:
            main_scores = base_scores

        pos_main, neg_main = gather_pair_scores(main_scores, pos_items, neg_items)
        loss_main = pairwise_bpr_loss(pos_main, neg_main)

        loss_text = base_scores.new_tensor(0.0)
        loss_vision = base_scores.new_tensor(0.0)

        if self.use_text:
            text_scores = self.unimodal_full_scores(user_ids, "text")
            pos_t, neg_t = gather_pair_scores(text_scores, pos_items, neg_items)
            loss_text = pairwise_bpr_loss(pos_t, neg_t)

        if self.use_vision:
            vision_scores = self.unimodal_full_scores(user_ids, "vision")
            pos_v, neg_v = gather_pair_scores(vision_scores, pos_items, neg_items)
            loss_vision = pairwise_bpr_loss(pos_v, neg_v)

        loss_modal = loss_text + loss_vision
        total = loss_main + self.branch_loss_weight * loss_modal
        total = torch.nan_to_num(total, nan=0.0, posinf=1.0e4, neginf=1.0e4)

        stats = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_main": float(loss_main.detach().cpu().item()),
            "loss_text": float(loss_text.detach().cpu().item()),
            "loss_vision": float(loss_vision.detach().cpu().item()),
            "loss_modal": float(loss_modal.detach().cpu().item()),
        }
        return total, stats
