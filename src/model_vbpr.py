# -*- coding: utf-8 -*-
"""
src/model_vbpr.py

VBPR baseline adapted to this project's evaluation pipeline.

Reference:
    VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback.
    This implementation follows the common PyTorch structure used in
    aaossa/VBPR-PyTorch:
        - user latent factor gamma_u
        - item latent factor gamma_i
        - user visual factor theta_u
        - fixed pretrained visual features f_i
        - visual projection E
        - item bias beta_i
        - visual bias beta_visual^T f_i
        - BPR pairwise objective

This file additionally provides full_sort_scores(user_ids, sequences, lengths)
so the model can be evaluated by the project's existing full-sort evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_numpy_feature(
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
        raise FileNotFoundError(f"VBPR feature file not found: {feature_path}")

    arr = np.load(feature_path)
    if arr.ndim != 2:
        raise ValueError(f"Feature file must be 2-D, got shape={arr.shape}: {feature_path}")

    arr = arr.astype(np.float32, copy=False)

    # Expected project format is usually num_items + 1, where row 0 is padding.
    # If features are num_items, prepend padding row.
    if arr.shape[0] == num_items:
        pad = np.zeros((1, arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif arr.shape[0] > num_items + 1:
        arr = arr[: num_items + 1]
    elif arr.shape[0] < num_items + 1:
        raise ValueError(
            f"Feature item count mismatch: feature rows={arr.shape[0]}, "
            f"expected num_items+1={num_items + 1}. path={feature_path}"
        )

    arr[0] = 0.0

    if mask_file:
        mask_path = data_dir / mask_file
        if mask_path.exists():
            mask = np.load(mask_path)
            mask = np.asarray(mask).reshape(-1)
            if mask.shape[0] == num_items:
                mask = np.concatenate([np.zeros((1,), dtype=mask.dtype), mask], axis=0)
            elif mask.shape[0] > num_items + 1:
                mask = mask[: num_items + 1]
            elif mask.shape[0] < num_items + 1:
                raise ValueError(
                    f"Mask item count mismatch: mask rows={mask.shape[0]}, "
                    f"expected num_items+1={num_items + 1}. path={mask_path}"
                )

            valid = mask.astype(bool)
            valid[0] = False
            if zero_missing:
                arr[~valid] = 0.0

    x = torch.from_numpy(arr).float()

    if normalize:
        # Normalize only non-padding rows. Zero rows remain zero after nan_to_num.
        x = F.normalize(x, p=2, dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x[0] = 0.0

    return x


class VBPR(nn.Module):
    """
    Visual Bayesian Personalized Ranking baseline.

    Score:
        s(u, i) =
            beta_i
            + <gamma_u, gamma_i>
            + <theta_u, E f_i>
            + beta_visual^T f_i

    Pairwise BPR:
        x_uij = s(u, i_pos) - s(u, i_neg)
        L = -log sigmoid(x_uij)

    Parameters
    ----------
    num_users:
        Maximum real user id. Internally the embedding size is num_users + 1
        to support 1-based ids and padding/user 0.
    num_items:
        Maximum real item id. Internally the embedding size is num_items + 1
        to support padding item 0.
    data_dir:
        Processed dataset directory containing vision_features.npy.
    feature_file:
        Visual feature file. Default: vision_features.npy.
    mask_file:
        Optional visual feature mask. Default: vision_mask.npy.
    latent_dim:
        Dimension of gamma user/item latent factors.
    visual_dim:
        Dimension of projected visual preference space.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        data_dir: str | Path,
        feature_file: str = "vision_features.npy",
        mask_file: Optional[str] = "vision_mask.npy",
        latent_dim: int = 64,
        visual_dim: int = 64,
        normalize_features: bool = True,
        zero_missing_features: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()

        if num_users <= 0:
            raise ValueError(f"num_users must be positive, got {num_users}")
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if visual_dim <= 0:
            raise ValueError(f"visual_dim must be positive, got {visual_dim}")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.latent_dim = int(latent_dim)
        self.visual_dim = int(visual_dim)
        self.dropout = float(dropout)

        features = _load_numpy_feature(
            data_dir=data_dir,
            feature_file=feature_file,
            num_items=num_items,
            mask_file=mask_file,
            normalize=normalize_features,
            zero_missing=zero_missing_features,
        )
        self.feature_dim = int(features.shape[1])
        self.register_buffer("visual_features", features, persistent=False)

        # Latent factors.
        self.gamma_users = nn.Embedding(num_users + 1, latent_dim, padding_idx=0)
        self.gamma_items = nn.Embedding(num_items + 1, latent_dim, padding_idx=0)

        # Visual factors.
        self.theta_users = nn.Embedding(num_users + 1, visual_dim, padding_idx=0)
        self.visual_projection = nn.Linear(self.feature_dim, visual_dim, bias=False)

        # Bias terms.
        self.item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.visual_bias = nn.Linear(self.feature_dim, 1, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, init_std: float = 0.01) -> None:
        # Small normal init is generally stable for BPR.
        nn.init.normal_(self.gamma_users.weight, std=init_std)
        nn.init.normal_(self.gamma_items.weight, std=init_std)
        nn.init.normal_(self.theta_users.weight, std=init_std)
        nn.init.normal_(self.visual_projection.weight, std=init_std)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.visual_bias.weight, std=init_std)

        with torch.no_grad():
            self.gamma_users.weight[0].zero_()
            self.gamma_items.weight[0].zero_()
            self.theta_users.weight[0].zero_()
            self.item_bias.weight[0].zero_()

    def encode_visual_items(self, item_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return projected visual factors and visual bias.

        If item_ids is None:
            visual_emb:  [num_items + 1, visual_dim]
            visual_bias: [num_items + 1]
        Else:
            visual_emb:  item_ids.shape + [visual_dim]
            visual_bias: item_ids.shape
        """
        if item_ids is None:
            f = self.visual_features
        else:
            f = self.visual_features[item_ids]

        f = f.to(dtype=self.visual_projection.weight.dtype)
        if self.dropout > 0 and self.training:
            f = self.dropout_layer(f)

        visual_emb = self.visual_projection(f)
        visual_bias = self.visual_bias(f).squeeze(-1)

        visual_emb = torch.nan_to_num(visual_emb, nan=0.0, posinf=0.0, neginf=0.0)
        visual_bias = torch.nan_to_num(visual_bias, nan=0.0, posinf=0.0, neginf=0.0)

        if item_ids is None:
            visual_emb[0] = 0.0
            visual_bias[0] = 0.0

        return visual_emb, visual_bias

    def score_items(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Score specific user-item pairs.

        user_ids and item_ids must be broadcast-compatible tensors with same shape.
        Returns tensor with shape equal to broadcasted user/item shape.
        """
        user_ids = user_ids.long()
        item_ids = item_ids.long()

        gamma_u = self.gamma_users(user_ids)
        theta_u = self.theta_users(user_ids)
        gamma_i = self.gamma_items(item_ids)

        visual_i, visual_b = self.encode_visual_items(item_ids)

        beta_i = self.item_bias(item_ids).squeeze(-1)
        latent_score = (gamma_u * gamma_i).sum(dim=-1)
        visual_score = (theta_u * visual_i).sum(dim=-1)

        score = beta_i + latent_score + visual_score + visual_b
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return pairwise margin x_uij = s(u, pos) - s(u, neg).
        """
        pos_score = self.score_items(user_ids, pos_items)
        neg_score = self.score_items(user_ids, neg_items)
        return pos_score - neg_score

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        margin = self.forward(user_ids, pos_items, neg_items)
        loss = -F.logsigmoid(margin)

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction={reduction!r}")

    @torch.no_grad()
    def generate_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Precompute all item components for full-sort inference.
        """
        visual_emb, visual_b = self.encode_visual_items(item_ids=None)
        gamma_i = self.gamma_items.weight
        beta_i = self.item_bias.weight.squeeze(-1)

        gamma_i = torch.nan_to_num(gamma_i, nan=0.0, posinf=0.0, neginf=0.0)
        beta_i = torch.nan_to_num(beta_i, nan=0.0, posinf=0.0, neginf=0.0)

        gamma_i[0] = 0.0
        beta_i[0] = 0.0
        visual_emb[0] = 0.0
        visual_b[0] = 0.0

        return gamma_i, visual_emb, beta_i + visual_b

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full-sort scores for project evaluator.

        sequences and lengths are accepted for API compatibility but ignored,
        because VBPR is not sequential.
        """
        del sequences, lengths

        user_ids = user_ids.long()
        gamma_u = self.gamma_users(user_ids)
        theta_u = self.theta_users(user_ids)

        gamma_i, visual_i, item_total_bias = self.generate_cache()
        gamma_i = gamma_i.to(device=gamma_u.device, dtype=gamma_u.dtype)
        visual_i = visual_i.to(device=theta_u.device, dtype=theta_u.dtype)
        item_total_bias = item_total_bias.to(device=gamma_u.device, dtype=gamma_u.dtype)

        latent_scores = torch.matmul(gamma_u, gamma_i.t())
        visual_scores = torch.matmul(theta_u, visual_i.t())

        scores = latent_scores + visual_scores + item_total_bias.unsqueeze(0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Never recommend padding item.
        scores[:, 0] = -1.0e9
        return scores

    def recommend(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.full_sort_scores(user_ids=user_ids)