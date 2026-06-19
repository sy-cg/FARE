# -*- coding: utf-8 -*-
"""
src/model_late_fusion.py

Late-fusion multimodal SASRec baseline.

Core idea:
- Keep SASRec ID backbone unchanged.
- Text/image features are NOT used as sequence inputs.
- Text/image only contribute candidate-side residual scores:

    score(u, j) = score_id(u, j)
                + w_t * <Q_t h_u, c_j^t>
                + w_v * <Q_v h_u, c_j^v>

where c_j^t and c_j^v are fixed projected text/vision item features.

This is intended as a clean baseline for diagnosing whether multimodal candidate
scores change group-wise ranking and exposure fairness.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_sasrec import SASRecID
except ImportError:
    from model_sasrec import SASRecID  # type: ignore


# ==========================================================
# Feature projection utilities
# ==========================================================


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def _random_projection_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Gaussian random projection. Scaling keeps variance approximately stable.
    mat = rng.normal(loc=0.0, scale=1.0 / math.sqrt(output_dim), size=(input_dim, output_dim)).astype(np.float32)
    return mat


def build_or_load_projected_features(
    data_dir: str | Path,
    feature_file: str,
    mask_file: str,
    output_dim: int,
    seed: int,
    cache_prefix: str,
    use_cache: bool = True,
    chunk_size: int = 32768,
) -> torch.Tensor:
    """Load raw item features and return fixed projected features [num_items+1, output_dim].

    The projection is deterministic and cached under data_dir. This avoids repeatedly
    projecting 768/512-d raw features in every training run.
    """
    data_dir = Path(data_dir)
    feature_path = data_dir / feature_file
    mask_path = data_dir / mask_file
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    cache_name = f"{cache_prefix}_rp{output_dim}_seed{seed}.npy"
    cache_path = data_dir / cache_name
    if use_cache and cache_path.exists():
        arr = np.load(cache_path).astype(np.float32)
        return torch.from_numpy(arr)

    raw = np.load(feature_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix at {feature_path}, got {raw.shape}")
    if mask.shape[0] != raw.shape[0]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs raw feature {raw.shape}")

    num_items_plus_one, input_dim = raw.shape
    proj = _random_projection_matrix(input_dim=input_dim, output_dim=output_dim, seed=seed)
    out = np.zeros((num_items_plus_one, output_dim), dtype=np.float32)

    for start in range(0, num_items_plus_one, chunk_size):
        end = min(num_items_plus_one, start + chunk_size)
        x = raw[start:end]
        x = _l2_normalize_np(x)
        y = x @ proj
        y = _l2_normalize_np(y)
        y *= mask[start:end, None]
        out[start:end] = y.astype(np.float32)

    out[0] = 0.0
    if use_cache:
        np.save(cache_path, out)
        manifest = {
            "source_feature": feature_file,
            "source_mask": mask_file,
            "cache_file": cache_name,
            "projection": "gaussian_random_projection",
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "seed": int(seed),
        }
        with open(data_dir / f"{cache_prefix}_rp{output_dim}_seed{seed}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    return torch.from_numpy(out)


def _init_logit_from_weight(weight: float, max_weight: float) -> float:
    weight = float(weight)
    max_weight = float(max_weight)
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    ratio = min(max(weight / max_weight, 1e-6), 1.0 - 1e-6)
    return float(math.log(ratio / (1.0 - ratio)))


# ==========================================================
# Model
# ==========================================================


class LateFusionSASRecID(SASRecID):
    """SASRec-ID backbone with late multimodal candidate score residuals."""

    def __init__(
        self,
        num_items: int,
        data_dir: str | Path,
        max_seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        tie_output_embedding: bool = True,
        mm_score_dim: int = 64,
        projection_seed: int = 2026,
        use_text: bool = True,
        use_vision: bool = True,
        cache_projected_features: bool = True,
        feature_chunk_size: int = 32768,
        text_weight_init: float = 0.05,
        vision_weight_init: float = 0.05,
        max_modal_weight: float = 0.5,
        learnable_modal_weights: bool = True,
        normalize_queries: bool = True,
        modal_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            tie_output_embedding=tie_output_embedding,
        )
        if mm_score_dim <= 0:
            raise ValueError(f"mm_score_dim must be positive, got {mm_score_dim}")
        if not use_text and not use_vision:
            raise ValueError("At least one modality must be enabled for LateFusionSASRecID")

        self.data_dir = str(data_dir)
        self.mm_score_dim = int(mm_score_dim)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)
        self.max_modal_weight = float(max_modal_weight)
        self.learnable_modal_weights = bool(learnable_modal_weights)
        self.normalize_queries = bool(normalize_queries)
        self.modal_dropout = nn.Dropout(float(modal_dropout))

        if self.use_text:
            text_item = build_or_load_projected_features(
                data_dir=data_dir,
                feature_file="text_features.npy",
                mask_file="text_mask.npy",
                output_dim=mm_score_dim,
                seed=projection_seed + 11,
                cache_prefix="text_features",
                use_cache=cache_projected_features,
                chunk_size=feature_chunk_size,
            )
            if text_item.shape[0] != num_items + 1:
                raise ValueError(f"Text feature item count mismatch: {text_item.shape[0]} vs {num_items + 1}")
            self.register_buffer("text_item_features", text_item, persistent=False)
            self.text_query = nn.Linear(hidden_size, mm_score_dim, bias=False)
        else:
            self.register_buffer("text_item_features", torch.empty(0), persistent=False)
            self.text_query = None

        if self.use_vision:
            vision_item = build_or_load_projected_features(
                data_dir=data_dir,
                feature_file="vision_features.npy",
                mask_file="vision_mask.npy",
                output_dim=mm_score_dim,
                seed=projection_seed + 29,
                cache_prefix="vision_features",
                use_cache=cache_projected_features,
                chunk_size=feature_chunk_size,
            )
            if vision_item.shape[0] != num_items + 1:
                raise ValueError(f"Vision feature item count mismatch: {vision_item.shape[0]} vs {num_items + 1}")
            self.register_buffer("vision_item_features", vision_item, persistent=False)
            self.vision_query = nn.Linear(hidden_size, mm_score_dim, bias=False)
        else:
            self.register_buffer("vision_item_features", torch.empty(0), persistent=False)
            self.vision_query = None

        # Modal weights are bounded into (0, max_modal_weight) with sigmoid.
        if self.learnable_modal_weights:
            self.raw_text_weight = nn.Parameter(torch.tensor(_init_logit_from_weight(text_weight_init, max_modal_weight)))
            self.raw_vision_weight = nn.Parameter(torch.tensor(_init_logit_from_weight(vision_weight_init, max_modal_weight)))
        else:
            self.register_buffer("raw_text_weight", torch.tensor(float(text_weight_init)), persistent=True)
            self.register_buffer("raw_vision_weight", torch.tensor(float(vision_weight_init)), persistent=True)

        self._reset_late_fusion_parameters()

    def _reset_late_fusion_parameters(self) -> None:
        if self.text_query is not None:
            nn.init.xavier_uniform_(self.text_query.weight)
        if self.vision_query is not None:
            nn.init.xavier_uniform_(self.vision_query.weight)

    def _modal_weight(self, raw_weight: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(raw_weight).all():
            raise FloatingPointError(f"Non-finite raw modal weight detected: {raw_weight}")
    
        if self.learnable_modal_weights:
            raw_weight = raw_weight.clamp(min=-20.0, max=20.0)
            return self.max_modal_weight * torch.sigmoid(raw_weight)
    
        return raw_weight.clamp(min=0.0, max=self.max_modal_weight)

    def get_modal_weights(self) -> Tuple[float, float]:
        with torch.no_grad():
            text_w = self._modal_weight(self.raw_text_weight).item() if self.use_text else 0.0
            vision_w = self._modal_weight(self.raw_vision_weight).item() if self.use_vision else 0.0
        return float(text_w), float(vision_w)

    def _modal_scores(self, final_hidden: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(final_hidden.shape[0], self.num_items + 1, device=final_hidden.device, dtype=final_hidden.dtype)

        h = self.modal_dropout(final_hidden)

        if self.use_text and self.text_query is not None:
            q = self.text_query(h)
            if self.normalize_queries:
                q = F.normalize(q, p=2, dim=-1)
            item = self.text_item_features.to(device=q.device, dtype=q.dtype)
            text_scores = torch.matmul(q, item.t())
            scores = scores + self._modal_weight(self.raw_text_weight).to(q.dtype) * text_scores

        if self.use_vision and self.vision_query is not None:
            q = self.vision_query(h)
            if self.normalize_queries:
                q = F.normalize(q, p=2, dim=-1)
            item = self.vision_item_features.to(device=q.device, dtype=q.dtype)
            vision_scores = torch.matmul(q, item.t())
            scores = scores + self._modal_weight(self.raw_vision_weight).to(q.dtype) * vision_scores

        scores[:, 0] = 0.0
        return scores

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del user_ids
        final_hidden = self.encode_sequence(sequences, lengths)
        item_weight = self.item_embedding.weight if self.tie_output_embedding else self.output_embedding.weight
        id_scores = torch.matmul(final_hidden, item_weight.t()) + self.output_bias
        mm_scores = self._modal_scores(final_hidden)
        return id_scores + mm_scores
