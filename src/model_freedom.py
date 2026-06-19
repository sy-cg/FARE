# -*- coding: utf-8 -*-
"""
src/model_freedom.py

FREEDOM baseline adapted to this project's evaluation pipeline.

Reference:
    FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures
    for Multimodal Recommendation, ACM MM 2023.

Original key ideas:
    - Freeze multimodal item-item kNN graph.
    - Denoise user-item graph via degree-sensitive edge pruning.
    - Propagate item ID embeddings on the frozen multimodal graph.
    - Propagate user/item ID embeddings on the pruned user-item graph.
    - Train with BPR loss, plus modality BPR auxiliary losses.

This implementation:
    - Uses train.txt interactions to build the user-item graph.
    - Uses text_features.npy and vision_features.npy to build frozen item-item graph.
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


def _coalesce_sparse(indices: torch.Tensor, values: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()


def _symmetric_normalize_from_edges(
    row: np.ndarray,
    col: np.ndarray,
    n_nodes: int,
) -> torch.Tensor:
    degree = np.bincount(row, minlength=n_nodes).astype(np.float32)
    degree = np.maximum(degree, 1.0)

    values = 1.0 / np.sqrt(degree[row] * degree[col])
    values = values.astype(np.float32)

    indices = torch.from_numpy(np.stack([row, col], axis=0)).long()
    values_t = torch.from_numpy(values).float()
    return _coalesce_sparse(indices, values_t, size=(n_nodes, n_nodes))


def build_user_item_norm_adj(
    train_rows: Sequence[Tuple[int, Sequence[int]]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """
    Build normalized bipartite user-item graph.

    Node layout:
        users: 0 ... num_users
        items: num_users + 1 ... num_users + num_items
        padding item 0 maps to num_users + 1 + 0 but should be isolated.
    """
    item_offset = num_users + 1
    n_nodes = (num_users + 1) + (num_items + 1)

    rows: List[int] = []
    cols: List[int] = []

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
            u = uid
            i = item_offset + item
            rows.extend([u, i])
            cols.extend([i, u])

    if not rows:
        raise ValueError("No valid user-item edges found for FREEDOM graph.")

    row_np = np.asarray(rows, dtype=np.int64)
    col_np = np.asarray(cols, dtype=np.int64)
    return _symmetric_normalize_from_edges(row_np, col_np, n_nodes=n_nodes)


def build_edge_info(
    train_rows: Sequence[Tuple[int, Sequence[int]]],
    num_users: int,
    num_items: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return bipartite edge coordinates [2, E] in user/item local coordinates:
        edge_indices[0] = user id
        edge_indices[1] = item id

    edge_values are degree-normalized values used as sampling probabilities
    for degree-sensitive pruning.
    """
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
        raise ValueError("No valid bipartite edges found for FREEDOM edge_info.")

    edge_indices = torch.tensor([users, items], dtype=torch.long)
    edge_values = normalize_bipartite_values(
        edge_indices=edge_indices,
        num_users=num_users,
        num_items=num_items,
    )
    return edge_indices, edge_values


def normalize_bipartite_values(
    edge_indices: torch.Tensor,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """
    Compute D_u^-1/2 D_i^-1/2 values for user-item local edge indices.
    """
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

    values = torch.pow(user_degree[users], -0.5) * torch.pow(item_degree[items], -0.5)
    return values.float()


def build_pruned_user_item_adj(
    keep_edges: torch.Tensor,
    keep_values: torch.Tensor,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """
    Convert local user-item edges to full symmetric user-item adjacency.
    """
    item_offset = num_users + 1
    n_nodes = (num_users + 1) + (num_items + 1)

    users = keep_edges[0].long()
    items = keep_edges[1].long() + item_offset

    forward = torch.stack([users, items], dim=0)
    backward = torch.stack([items, users], dim=0)

    all_indices = torch.cat([forward, backward], dim=1)
    all_values = torch.cat([keep_values, keep_values], dim=0)

    return torch.sparse_coo_tensor(
        all_indices,
        all_values,
        size=(n_nodes, n_nodes),
        device=keep_edges.device,
    ).coalesce()


def build_knn_item_adj(
    features: torch.Tensor,
    knn_k: int = 10,
    block_size: int = 1024,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Build frozen item-item kNN graph from item features.

    Shape:
        features: [num_items + 1, feat_dim]
        return: sparse [num_items + 1, num_items + 1]

    Padding row 0 is isolated.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.shape}")

    n_items_plus_pad = int(features.shape[0])
    if n_items_plus_pad <= 1:
        raise ValueError("Need at least one real item to build kNN graph.")

    k = int(min(knn_k, n_items_plus_pad - 1))
    if k <= 0:
        raise ValueError(f"knn_k must be positive, got {knn_k}")

    x = features.detach().float().cpu()
    x = F.normalize(x, p=2, dim=-1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[0] = 0.0

    rows_all: List[torch.Tensor] = []
    cols_all: List[torch.Tensor] = []

    real_item_ids = torch.arange(1, n_items_plus_pad, dtype=torch.long)

    for start in range(1, n_items_plus_pad, int(block_size)):
        end = min(start + int(block_size), n_items_plus_pad)
        row_ids = torch.arange(start, end, dtype=torch.long)

        sim = torch.matmul(x[start:end], x.t())
        sim[:, 0] = -float("inf")

        if not include_self:
            local = torch.arange(end - start, dtype=torch.long)
            sim[local, row_ids] = -float("inf")

        _, topk = torch.topk(sim, k=k, dim=-1)

        rows = row_ids.view(-1, 1).expand(-1, k).reshape(-1)
        cols = topk.reshape(-1).long()

        rows_all.append(rows)
        cols_all.append(cols)

    row = torch.cat(rows_all, dim=0)
    col = torch.cat(cols_all, dim=0)

    # Unweighted graph, then symmetric normalization by out-degree/in-degree.
    values = torch.ones_like(row, dtype=torch.float32)

    row_np = row.numpy()
    col_np = col.numpy()
    degree = np.bincount(row_np, minlength=n_items_plus_pad).astype(np.float32)
    degree = np.maximum(degree, 1.0)

    norm_values = 1.0 / np.sqrt(degree[row_np] * degree[col_np])
    norm_values = norm_values.astype(np.float32)

    indices = torch.stack([row, col], dim=0).long()
    values_t = torch.from_numpy(norm_values).float()

    adj = torch.sparse_coo_tensor(
        indices,
        values_t,
        size=(n_items_plus_pad, n_items_plus_pad),
    ).coalesce()

    return adj


def combine_sparse_adjs(
    image_adj: Optional[torch.Tensor],
    text_adj: Optional[torch.Tensor],
    image_weight: float,
) -> torch.Tensor:
    if image_adj is None and text_adj is None:
        raise ValueError("At least one item-item adjacency is required.")

    if image_adj is not None and text_adj is not None:
        return (float(image_weight) * image_adj + (1.0 - float(image_weight)) * text_adj).coalesce()

    if image_adj is not None:
        return image_adj.coalesce()

    assert text_adj is not None
    return text_adj.coalesce()


# ==========================================================
# FREEDOM model
# ==========================================================


class FREEDOM(nn.Module):
    """
    FREEDOM multimodal graph recommendation model.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_values: torch.Tensor,
        data_dir: str | Path,
        embedding_dim: int = 64,
        feat_embed_dim: int = 64,
        n_mm_layers: int = 1,
        n_ui_layers: int = 2,
        knn_k: int = 10,
        mm_image_weight: float = 0.1,
        edge_dropout: float = 0.8,
        degree_ratio: float = 1.0,
        reg_weight: float = 1.0e-4,
        l2_weight: float = 0.0,
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
        item_graph_cache_file: Optional[str] = None,
        knn_block_size: int = 1024,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()

        if num_users <= 0:
            raise ValueError(f"num_users must be positive, got {num_users}")
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if feat_embed_dim <= 0:
            raise ValueError(f"feat_embed_dim must be positive, got {feat_embed_dim}")
        if not use_text and not use_vision:
            raise ValueError("At least one of use_text/use_vision must be enabled.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.feat_embed_dim = int(feat_embed_dim)
        self.n_mm_layers = int(n_mm_layers)
        self.n_ui_layers = int(n_ui_layers)
        self.knn_k = int(knn_k)
        self.mm_image_weight = float(mm_image_weight)
        self.edge_dropout = float(edge_dropout)
        self.degree_ratio = float(degree_ratio)
        self.reg_weight = float(reg_weight)
        self.l2_weight = float(l2_weight)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)

        self.user_node_count = self.num_users + 1
        self.item_node_count = self.num_items + 1
        self.n_nodes = self.user_node_count + self.item_node_count

        expected_shape = torch.Size([self.n_nodes, self.n_nodes])
        if norm_adj.shape != expected_shape:
            raise ValueError(f"norm_adj shape mismatch: got {norm_adj.shape}, expected {expected_shape}")

        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)
        self.register_buffer("edge_indices", edge_indices.long(), persistent=False)
        self.register_buffer("edge_values", edge_values.float(), persistent=False)

        self.masked_adj: Optional[torch.Tensor] = None

        self.user_embedding = nn.Embedding(self.num_users + 1, embedding_dim, padding_idx=0)
        self.item_id_embedding = nn.Embedding(self.num_items + 1, embedding_dim, padding_idx=0)

        self.image_embedding: Optional[nn.Embedding] = None
        self.image_trs: Optional[nn.Linear] = None
        self.text_embedding: Optional[nn.Embedding] = None
        self.text_trs: Optional[nn.Linear] = None

        data_dir = Path(data_dir)

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

        if item_graph_cache_file:
            cache_file = data_dir / item_graph_cache_file
        else:
            cache_file = data_dir / f"mm_adj_freedom_k{self.knn_k}_w{int(10 * self.mm_image_weight)}.pt"

        if cache_item_graph and cache_file.exists():
            mm_adj = safe_torch_load(cache_file, map_location="cpu")
            if not mm_adj.is_sparse:
                raise ValueError(f"Cached item graph must be sparse tensor: {cache_file}")
        else:
            image_adj = None
            text_adj = None

            if image_feat is not None:
                image_adj = build_knn_item_adj(
                    image_feat,
                    knn_k=self.knn_k,
                    block_size=knn_block_size,
                    include_self=True,
                )

            if text_feat is not None:
                text_adj = build_knn_item_adj(
                    text_feat,
                    knn_k=self.knn_k,
                    block_size=knn_block_size,
                    include_self=True,
                )

            mm_adj = combine_sparse_adjs(
                image_adj=image_adj,
                text_adj=text_adj,
                image_weight=self.mm_image_weight,
            )

            if cache_item_graph:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(mm_adj.cpu(), cache_file)

        if mm_adj.shape != torch.Size([self.num_items + 1, self.num_items + 1]):
            raise ValueError(f"mm_adj shape mismatch: got {mm_adj.shape}, expected {(self.num_items + 1, self.num_items + 1)}")

        self.register_buffer("mm_adj", mm_adj.coalesce(), persistent=False)

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

        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_id_embedding.weight[0].zero_()
            if self.image_embedding is not None:
                self.image_embedding.weight[0].zero_()
            if self.text_embedding is not None:
                self.text_embedding.weight[0].zero_()

    def pre_epoch_processing(self) -> None:
        """
        Degree-sensitive edge pruning for user-item graph.

        Original FREEDOM samples retained user-item edges according to
        normalized edge values. edge_dropout=0.8 means keeping about 20% edges.
        """
        if self.edge_dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return

        edge_values = self.edge_values
        edge_indices = self.edge_indices

        if edge_values.device != self.user_embedding.weight.device:
            edge_values = edge_values.to(self.user_embedding.weight.device)
            edge_indices = edge_indices.to(self.user_embedding.weight.device)

        num_edges = int(edge_values.numel())
        keep_len = int(num_edges * (1.0 - self.edge_dropout))
        keep_len = max(1, min(keep_len, num_edges))

        probs = edge_values.clamp_min(1.0e-12)
        if self.degree_ratio != 1.0:
            probs = probs.pow(float(self.degree_ratio))
        probs = probs / probs.sum().clamp_min(1.0e-12)

        sampled = torch.multinomial(probs, keep_len, replacement=False)
        keep_edges = edge_indices[:, sampled]

        keep_values = normalize_bipartite_values(
            edge_indices=keep_edges,
            num_users=self.num_users,
            num_items=self.num_items,
        )

        self.masked_adj = build_pruned_user_item_adj(
            keep_edges=keep_edges,
            keep_values=keep_values,
            num_users=self.num_users,
            num_items=self.num_items,
        )

    def graph_forward(self, adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if adj is None:
            adj = self.masked_adj if self.masked_adj is not None else self.norm_adj

        device = self.user_embedding.weight.device
        if adj.device != device:
            adj = adj.to(device)

        mm_adj = self.mm_adj
        if mm_adj.device != device:
            mm_adj = mm_adj.to(device)

        # Frozen multimodal item-item graph propagation.
        h_item_mm = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            h_item_mm = torch.sparse.mm(mm_adj, h_item_mm)

        # User-item graph propagation.
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego]

        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            all_embeddings.append(x)

        out = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_g, item_g = torch.split(out, [self.num_users + 1, self.num_items + 1], dim=0)

        item_g = item_g + h_item_mm

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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        user_ids = user_ids.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        user_all, item_all = self.graph_forward(self.masked_adj)

        u = user_all[user_ids]
        pos_i = item_all[pos_items]
        neg_i = item_all[neg_items]

        main_loss = self.bpr_loss(u, pos_i, neg_i)

        text_loss = u.new_tensor(0.0)
        image_loss = u.new_tensor(0.0)

        if self.text_embedding is not None and self.text_trs is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = torch.nan_to_num(text_feats, nan=0.0, posinf=0.0, neginf=0.0)
            text_loss = self.bpr_loss(u, text_feats[pos_items], text_feats[neg_items])

        if self.image_embedding is not None and self.image_trs is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = torch.nan_to_num(image_feats, nan=0.0, posinf=0.0, neginf=0.0)
            image_loss = self.bpr_loss(u, image_feats[pos_items], image_feats[neg_items])

        l2 = u.new_tensor(0.0)
        if self.l2_weight > 0:
            l2 = (
                self.user_embedding(user_ids).pow(2).mean()
                + self.item_id_embedding(pos_items).pow(2).mean()
                + self.item_id_embedding(neg_items).pow(2).mean()
            )

        total = main_loss + self.reg_weight * (text_loss + image_loss) + self.l2_weight * l2
        total = torch.nan_to_num(total, nan=0.0, posinf=1.0e4, neginf=1.0e4)

        stats = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_main": float(main_loss.detach().cpu().item()),
            "loss_text": float(text_loss.detach().cpu().item()),
            "loss_image": float(image_loss.detach().cpu().item()),
            "loss_l2": float(l2.detach().cpu().item()),
        }

        return total, stats

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full-sort prediction for project evaluator.

        FREEDOM is graph-based and non-sequential, so sequences/lengths are ignored.
        """
        del sequences, lengths

        user_ids = user_ids.long()
        user_all, item_all = self.graph_forward(self.norm_adj)

        scores = torch.matmul(user_all[user_ids], item_all.t())
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = -1.0e9
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.calculate_loss(user_ids, pos_items, neg_items)
