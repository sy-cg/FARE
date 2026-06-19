# -*- coding: utf-8 -*-
"""
src/model_lattice.py

LATTICE baseline adapted to this project's evaluation pipeline.

Reference:
    LATTICE: Mining Latent Structures for Multimedia Recommendation,
    ACM Multimedia 2021.

Original key ideas:
    - Build image/text item-item similarity graphs.
    - Keep original modality graphs as cached structures.
    - Learn modality-specific item graphs from projected modality features.
    - Fuse image/text graphs with learnable modality weights.
    - Interpolate learned graph with original graph using lambda_coeff.
    - Propagate item ID embeddings on the fused item graph.
    - Combine item graph representation with CF graph representation.
    - Train with BPR loss.

This implementation:
    - Uses train.txt interactions to build user-item graph.
    - Uses text_features.npy and vision_features.npy as multimodal features.
    - Supports 1-based user/item ids with row 0 reserved for padding.
    - Provides full_sort_scores(user_ids, sequences, lengths) for existing evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .io_utils import safe_torch_load
except ImportError:
    from io_utils import safe_torch_load  # type: ignore


# ==========================================================
# Feature loading
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
        raise ValueError(f"Feature file must be 2-D, got shape={arr.shape}: {feature_path}")

    arr = arr.astype(np.float32, copy=False)

    if arr.shape[0] == num_items:
        pad = np.zeros((1, arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif arr.shape[0] > num_items + 1:
        arr = arr[: num_items + 1]
    elif arr.shape[0] < num_items + 1:
        raise ValueError(
            f"Feature rows mismatch for {feature_path}: got {arr.shape[0]}, "
            f"expected {num_items + 1} or {num_items}."
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
                    f"Mask rows mismatch for {mask_path}: got {mask.shape[0]}, "
                    f"expected {num_items + 1} or {num_items}."
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


# ==========================================================
# Sparse graph utilities
# ==========================================================


def normalize_bipartite_values(
    edge_indices: torch.Tensor,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    device = edge_indices.device
    users = edge_indices[0].long()
    items = edge_indices[1].long()

    user_degree = torch.zeros(num_users + 1, dtype=torch.float32, device=device)
    item_degree = torch.zeros(num_items + 1, dtype=torch.float32, device=device)

    one = torch.ones_like(users, dtype=torch.float32, device=device)
    user_degree.scatter_add_(0, users, one)
    item_degree.scatter_add_(0, items, one)

    user_degree = user_degree.clamp_min(1.0e-7)
    item_degree = item_degree.clamp_min(1.0e-7)

    return torch.pow(user_degree[users], -0.5) * torch.pow(item_degree[items], -0.5)


def build_user_item_norm_adj(
    train_rows: Sequence[Tuple[int, Sequence[int]]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    item_offset = num_users + 1
    n_nodes = (num_users + 1) + (num_items + 1)

    users: List[int] = []
    items: List[int] = []

    for uid, seq in train_rows:
        uid = int(uid)
        if uid <= 0 or uid > num_users:
            continue

        unique_items = set()
        for item in seq:
            item = int(item)
            if 0 < item <= num_items:
                unique_items.add(item)

        for item in unique_items:
            users.append(uid)
            items.append(item)

    if not users:
        raise ValueError("No valid user-item edges found for LATTICE graph.")

    local_edges = torch.tensor([users, items], dtype=torch.long)
    values = normalize_bipartite_values(local_edges, num_users=num_users, num_items=num_items)

    u = local_edges[0]
    i = local_edges[1] + item_offset

    forward = torch.stack([u, i], dim=0)
    backward = torch.stack([i, u], dim=0)

    indices = torch.cat([forward, backward], dim=1)
    all_values = torch.cat([values, values], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        all_values,
        size=(n_nodes, n_nodes),
    ).coalesce()


def build_dynamic_knn_adj(
    features: torch.Tensor,
    topk: int = 10,
    block_size: int = 1024,
    clamp_min: float = 0.0,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Build differentiable sparse top-k item graph from feature tensor.

    features:
        [num_items + 1, dim]

    Returns:
        sparse normalized item-item graph [num_items + 1, num_items + 1].
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.shape}")

    device = features.device
    n_items = int(features.shape[0])

    if n_items <= 1:
        raise ValueError("Need at least one real item to build kNN graph.")

    k = min(int(topk), n_items - 1)
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    x = F.normalize(features, p=2, dim=-1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clone()
    x[0] = 0.0

    rows_all: List[torch.Tensor] = []
    cols_all: List[torch.Tensor] = []
    vals_all: List[torch.Tensor] = []

    for start in range(1, n_items, int(block_size)):
        end = min(start + int(block_size), n_items)
        row_ids = torch.arange(start, end, dtype=torch.long, device=device)

        sim = torch.matmul(x[start:end], x.t())
        sim[:, 0] = -float("inf")

        if not include_self:
            local = torch.arange(end - start, dtype=torch.long, device=device)
            sim[local, row_ids] = -float("inf")

        top_vals, top_idx = torch.topk(sim, k=k, dim=-1)

        if clamp_min is not None:
            top_vals = top_vals.clamp_min(float(clamp_min))

        rows = row_ids.view(-1, 1).expand(-1, k).reshape(-1)
        cols = top_idx.reshape(-1).long()
        vals = top_vals.reshape(-1).float()

        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)

    row = torch.cat(rows_all, dim=0)
    col = torch.cat(cols_all, dim=0)
    vals = torch.cat(vals_all, dim=0)

    degree = torch.zeros(n_items, dtype=vals.dtype, device=device)
    degree.scatter_add_(0, row, vals)
    degree = degree.clamp_min(1.0e-7)

    norm_vals = vals / torch.sqrt(degree[row] * degree[col].clamp_min(1.0e-7))
    norm_vals = torch.nan_to_num(norm_vals, nan=0.0, posinf=0.0, neginf=0.0)

    indices = torch.stack([row, col], dim=0).long()

    return torch.sparse_coo_tensor(
        indices,
        norm_vals,
        size=(n_items, n_items),
        device=device,
    ).coalesce()


def build_static_knn_adj(
    features: torch.Tensor,
    topk: int = 10,
    block_size: int = 1024,
    clamp_min: float = 0.0,
    include_self: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        return build_dynamic_knn_adj(
            features=features.detach().float(),
            topk=topk,
            block_size=block_size,
            clamp_min=clamp_min,
            include_self=include_self,
        ).detach().cpu()


# ==========================================================
# LATTICE model
# ==========================================================


class LATTICE(nn.Module):
    """
    LATTICE multimodal recommendation model.

    This implementation supports:
        cf_model = "lightgcn", "mf", or "ngcf"

    Default is LightGCN because the original repo states its structure is largely
    based on LightGCN.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        data_dir: str | Path,
        embedding_dim: int = 64,
        feat_embed_dim: int = 64,
        weight_size: Optional[Sequence[int]] = None,
        mess_dropout: Optional[Sequence[float]] = None,
        n_item_layers: int = 1,
        topk: int = 10,
        lambda_coeff: float = 0.9,
        cf_model: str = "lightgcn",
        decay: float = 1.0e-4,
        text_feature_file: Optional[str] = "text_features.npy",
        vision_feature_file: Optional[str] = "vision_features.npy",
        text_mask_file: Optional[str] = "text_mask.npy",
        vision_mask_file: Optional[str] = "vision_mask.npy",
        use_text: bool = True,
        use_vision: bool = True,
        normalize_features: bool = True,
        zero_missing_features: bool = True,
        freeze_features: bool = False,
        cache_item_graph: bool = True,
        item_graph_cache_prefix: str = "lattice",
        knn_block_size: int = 1024,
        dynamic_graph_max_items: int = 20000,
        clamp_sim_min: float = 0.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()

        if num_users <= 0:
            raise ValueError(f"num_users must be positive, got {num_users}")
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if not use_text and not use_vision:
            raise ValueError("At least one of use_text/use_vision must be enabled.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.feat_embed_dim = int(feat_embed_dim)
        self.n_item_layers = int(n_item_layers)
        self.topk = int(topk)
        self.lambda_coeff = float(lambda_coeff)
        self.cf_model = str(cf_model).lower()
        self.decay = float(decay)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)
        self.knn_block_size = int(knn_block_size)
        self.dynamic_graph_max_items = int(dynamic_graph_max_items)
        self.clamp_sim_min = float(clamp_sim_min)

        self.user_node_count = self.num_users + 1
        self.item_node_count = self.num_items + 1
        self.n_nodes = self.user_node_count + self.item_node_count

        expected_shape = torch.Size([self.n_nodes, self.n_nodes])
        if norm_adj.shape != expected_shape:
            raise ValueError(f"norm_adj shape mismatch: got {norm_adj.shape}, expected {expected_shape}")

        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)

        if weight_size is None:
            weight_size = [embedding_dim]
        if mess_dropout is None:
            mess_dropout = [0.0] * len(weight_size)

        self.weight_size = [embedding_dim] + [int(x) for x in weight_size]
        self.n_ui_layers = len(self.weight_size) - 1

        if len(mess_dropout) < self.n_ui_layers:
            mess_dropout = list(mess_dropout) + [float(mess_dropout[-1] if mess_dropout else 0.0)] * (
                self.n_ui_layers - len(mess_dropout)
            )

        self.user_embedding = nn.Embedding(self.num_users + 1, embedding_dim, padding_idx=0)
        self.item_id_embedding = nn.Embedding(self.num_items + 1, embedding_dim, padding_idx=0)

        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        if self.cf_model == "ngcf":
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(float(mess_dropout[i])))

        data_dir = Path(data_dir)

        self.image_embedding: Optional[nn.Embedding] = None
        self.image_trs: Optional[nn.Linear] = None
        self.text_embedding: Optional[nn.Embedding] = None
        self.text_trs: Optional[nn.Linear] = None

        image_feat: Optional[torch.Tensor] = None
        text_feat: Optional[torch.Tensor] = None

        if self.use_vision:
            image_feat = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=str(vision_feature_file),
                num_items=num_items,
                mask_file=vision_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.image_embedding = nn.Embedding.from_pretrained(image_feat, freeze=freeze_features, padding_idx=0)
            self.image_trs = nn.Linear(image_feat.shape[1], feat_embed_dim)

        if self.use_text:
            text_feat = load_item_feature_matrix(
                data_dir=data_dir,
                feature_file=str(text_feature_file),
                num_items=num_items,
                mask_file=text_mask_file,
                normalize=normalize_features,
                zero_missing=zero_missing_features,
            )
            self.text_embedding = nn.Embedding.from_pretrained(text_feat, freeze=freeze_features, padding_idx=0)
            self.text_trs = nn.Linear(text_feat.shape[1], feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.image_original_adj: Optional[torch.Tensor] = None
        self.text_original_adj: Optional[torch.Tensor] = None

        if image_feat is not None:
            image_cache = data_dir / f"{item_graph_cache_prefix}_image_adj_k{self.topk}.pt"
            if cache_item_graph and image_cache.exists():
                self.image_original_adj = safe_torch_load(image_cache, map_location="cpu").coalesce()
            else:
                self.image_original_adj = build_static_knn_adj(
                    image_feat,
                    topk=self.topk,
                    block_size=self.knn_block_size,
                    clamp_min=self.clamp_sim_min,
                    include_self=True,
                )
                if cache_item_graph:
                    image_cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.image_original_adj.cpu(), image_cache)

        if text_feat is not None:
            text_cache = data_dir / f"{item_graph_cache_prefix}_text_adj_k{self.topk}.pt"
            if cache_item_graph and text_cache.exists():
                self.text_original_adj = safe_torch_load(text_cache, map_location="cpu").coalesce()
            else:
                self.text_original_adj = build_static_knn_adj(
                    text_feat,
                    topk=self.topk,
                    block_size=self.knn_block_size,
                    clamp_min=self.clamp_sim_min,
                    include_self=True,
                )
                if cache_item_graph:
                    text_cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.text_original_adj.cpu(), text_cache)

        self.item_adj: Optional[torch.Tensor] = None

        self._cached_user_embeddings: Optional[torch.Tensor] = None
        self._cached_item_embeddings: Optional[torch.Tensor] = None

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, init_std: float = 0.01) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.image_trs is not None:
            nn.init.xavier_uniform_(self.image_trs.weight)
            nn.init.zeros_(self.image_trs.bias)

        if self.text_trs is not None:
            nn.init.xavier_uniform_(self.text_trs.weight)
            nn.init.zeros_(self.text_trs.bias)

        for layer in self.GC_Linear_list:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in self.Bi_Linear_list:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_id_embedding.weight[0].zero_()
            if self.image_embedding is not None:
                self.image_embedding.weight[0].zero_()
            if self.text_embedding is not None:
                self.text_embedding.weight[0].zero_()

    def clear_eval_cache(self) -> None:
        self._cached_user_embeddings = None
        self._cached_item_embeddings = None

    def _get_modality_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_vision and self.use_text:
            weight = self.softmax(self.modal_weight)
            return weight[0], weight[1]
        if self.use_vision:
            one = self.modal_weight.new_tensor(1.0)
            zero = self.modal_weight.new_tensor(0.0)
            return one, zero
        one = self.modal_weight.new_tensor(1.0)
        zero = self.modal_weight.new_tensor(0.0)
        return zero, one

    def _original_item_adj(self, device: torch.device) -> torch.Tensor:
        image_w, text_w = self._get_modality_weights()

        pieces: List[torch.Tensor] = []

        if self.image_original_adj is not None:
            pieces.append(image_w * self.image_original_adj.to(device))

        if self.text_original_adj is not None:
            pieces.append(text_w * self.text_original_adj.to(device))

        if not pieces:
            raise RuntimeError("No original item adjacency is available.")

        out = pieces[0]
        for p in pieces[1:]:
            out = out + p

        return out.coalesce()

    def _learned_item_adj(self) -> torch.Tensor:
        device = self.user_embedding.weight.device
        n_items = self.num_items + 1

        if n_items > self.dynamic_graph_max_items:
            # For very large item sets, rebuilding learned dense/block top-k graph
            # every epoch can be too expensive. Fall back to original graph.
            return self._original_item_adj(device)

        image_w, text_w = self._get_modality_weights()
        pieces: List[torch.Tensor] = []

        if self.image_embedding is not None and self.image_trs is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = torch.nan_to_num(image_feats, nan=0.0, posinf=0.0, neginf=0.0)
            image_feats[0] = 0.0
            image_adj = build_dynamic_knn_adj(
                image_feats,
                topk=self.topk,
                block_size=self.knn_block_size,
                clamp_min=self.clamp_sim_min,
                include_self=True,
            )
            pieces.append(image_w * image_adj)

        if self.text_embedding is not None and self.text_trs is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = torch.nan_to_num(text_feats, nan=0.0, posinf=0.0, neginf=0.0)
            text_feats[0] = 0.0
            text_adj = build_dynamic_knn_adj(
                text_feats,
                topk=self.topk,
                block_size=self.knn_block_size,
                clamp_min=self.clamp_sim_min,
                include_self=True,
            )
            pieces.append(text_w * text_adj)

        if not pieces:
            raise RuntimeError("No learned item adjacency can be built.")

        out = pieces[0]
        for p in pieces[1:]:
            out = out + p

        return out.coalesce()

    def build_item_graph(self, learn_graph: bool = True) -> torch.Tensor:
        device = self.user_embedding.weight.device
        original_adj = self._original_item_adj(device)

        if learn_graph:
            learned_adj = self._learned_item_adj()
            item_adj = (1.0 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
            self.item_adj = item_adj.coalesce()
        else:
            if self.item_adj is None:
                self.item_adj = original_adj.coalesce()
            else:
                self.item_adj = self.item_adj.detach().coalesce()

        return self.item_adj

    def graph_forward(self, adj: Optional[torch.Tensor] = None, build_item_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.clear_eval_cache()

        device = self.user_embedding.weight.device

        if adj is None:
            adj = self.norm_adj
        if adj.device != device:
            adj = adj.to(device)

        item_adj = self.build_item_graph(learn_graph=build_item_graph)
        if item_adj.device != device:
            item_adj = item_adj.to(device)

        h = self.item_id_embedding.weight
        for _ in range(self.n_item_layers):
            h = torch.sparse.mm(item_adj, h)

        if self.cf_model == "mf":
            user_g = self.user_embedding.weight
            item_g = self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)
            user_g[0] = 0.0
            item_g[0] = 0.0
            return user_g, item_g

        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego]

        if self.cf_model == "lightgcn":
            x = ego
            for _ in range(self.n_ui_layers):
                x = torch.sparse.mm(adj, x)
                all_embeddings.append(x)

            all_embeddings_t = torch.stack(all_embeddings, dim=1).mean(dim=1)
            user_g, item_g = torch.split(all_embeddings_t, [self.num_users + 1, self.num_items + 1], dim=0)
            item_g = item_g + F.normalize(h, p=2, dim=1)

        elif self.cf_model == "ngcf":
            x = ego
            all_embeddings = [x]

            for layer_idx in range(self.n_ui_layers):
                side = torch.sparse.mm(adj, x)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[layer_idx](side))
                bi_embeddings = torch.mul(x, side)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[layer_idx](bi_embeddings))
                x = sum_embeddings + bi_embeddings
                x = self.dropout_list[layer_idx](x)
                x = F.normalize(x, p=2, dim=1)
                all_embeddings.append(x)

            all_embeddings_t = torch.stack(all_embeddings, dim=1).mean(dim=1)
            user_g, item_g = torch.split(all_embeddings_t, [self.num_users + 1, self.num_items + 1], dim=0)
            item_g = item_g + F.normalize(h, p=2, dim=1)

        else:
            raise ValueError(f"Unsupported cf_model={self.cf_model!r}; use lightgcn, ngcf, or mf.")

        user_g = torch.nan_to_num(user_g, nan=0.0, posinf=0.0, neginf=0.0)
        item_g = torch.nan_to_num(item_g, nan=0.0, posinf=0.0, neginf=0.0)
        user_g[0] = 0.0
        item_g[0] = 0.0
        return user_g, item_g

    @staticmethod
    def bpr_loss(
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def calculate_loss(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        build_item_graph: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        user_ids = user_ids.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        user_all, item_all = self.graph_forward(self.norm_adj, build_item_graph=build_item_graph)

        u = user_all[user_ids]
        pos_i = item_all[pos_items]
        neg_i = item_all[neg_items]

        mf_loss = self.bpr_loss(u, pos_i, neg_i)

        emb_loss = (
            self.user_embedding(user_ids).pow(2).sum()
            + self.item_id_embedding(pos_items).pow(2).sum()
            + self.item_id_embedding(neg_items).pow(2).sum()
        ) / max(int(user_ids.numel()), 1)

        reg_loss = self.decay * emb_loss
        total = mf_loss + reg_loss
        total = torch.nan_to_num(total, nan=0.0, posinf=1.0e4, neginf=1.0e4)

        stats = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_mf": float(mf_loss.detach().cpu().item()),
            "loss_emb": float(emb_loss.detach().cpu().item()),
            "loss_reg": float(reg_loss.detach().cpu().item()),
        }

        return total, stats

    def _get_eval_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_user_embeddings is not None and self._cached_item_embeddings is not None:
            return self._cached_user_embeddings, self._cached_item_embeddings

        user_g, item_g = self.graph_forward(self.norm_adj, build_item_graph=True)

        self._cached_user_embeddings = user_g.detach()
        self._cached_item_embeddings = item_g.detach()

        return self._cached_user_embeddings, self._cached_item_embeddings

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full-sort prediction for project evaluator.

        LATTICE is graph-based and non-sequential, so sequences/lengths are ignored.
        """
        del sequences, lengths

        user_ids = user_ids.long()

        if self.training:
            user_all, item_all = self.graph_forward(self.norm_adj, build_item_graph=False)
        else:
            user_all, item_all = self._get_eval_embeddings()

        scores = torch.matmul(user_all[user_ids], item_all.t())
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = -1.0e9
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        build_item_graph: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.calculate_loss(user_ids, pos_items, neg_items, build_item_graph=build_item_graph)
