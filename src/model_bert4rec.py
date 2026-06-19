# -*- coding: utf-8 -*-
"""
src/model_bert4rec.py

BERT4Rec-style ID backbone for sequential recommendation.

Important note:
- The original BERT4Rec trains with a masked-item prediction objective.
- This implementation is a bidirectional Transformer encoder adapted to the
  existing next-item full-sort CE pipeline used in this project.
- Because the input sequence is only the observed prefix, bidirectional attention
  over the prefix does not leak the target item. It is therefore a practical
  BERT4Rec-style next-item baseline that reuses the existing data loader,
  evaluator, and loss code.

API compatibility:
    logits = model.full_sort_scores(user_ids, sequences, lengths)

Padding item id is 0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class BERT4RecID(nn.Module):
    """Bidirectional Transformer encoder for next-item full-sort scoring.

    Parameters
    ----------
    num_items:
        Number of non-padding items. Item id 0 is padding.
    max_seq_len:
        Maximum sequence length used by the data loader.
    hidden_size:
        Transformer hidden size and item embedding dimension.
    num_layers:
        Number of Transformer encoder layers.
    num_heads:
        Number of self-attention heads.
    dropout:
        Dropout used in embeddings and Transformer layers.
    activation:
        Feed-forward activation, e.g. "gelu" or "relu".
    layer_norm_eps:
        LayerNorm epsilon.
    tie_output_embedding:
        Whether to share output item weights with input item embeddings.
    use_cls_token:
        If True, prepend a learned [CLS]-like token and use its final state as
        the sequence representation. If False, gather the final non-padding
        item position. Default False is usually closer to next-item prediction.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        tie_output_embedding: bool = True,
        use_cls_token: bool = False,
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.activation = str(activation)
        self.layer_norm_eps = float(layer_norm_eps)
        self.tie_output_embedding = bool(tie_output_embedding)
        self.use_cls_token = bool(use_cls_token)
        self.embedding_init_std = float(embedding_init_std)

        # +1 for padding item id 0.
        self.item_embedding = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=0)

        # If use_cls_token=True, position ids include one extra token position.
        pos_len = self.max_seq_len + (1 if self.use_cls_token else 0)
        self.position_embedding = nn.Embedding(pos_len, self.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(self.dropout)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
        )

        if self.tie_output_embedding:
            self.output_embedding = None
        else:
            self.output_embedding = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=0)

        self.output_bias = nn.Parameter(torch.zeros(self.num_items + 1))
        self.reset_parameters()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.embedding_init_std)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=self.embedding_init_std)

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=self.embedding_init_std)

        if self.output_embedding is not None:
            nn.init.normal_(self.output_embedding.weight, mean=0.0, std=self.embedding_init_std)
            with torch.no_grad():
                self.output_embedding.weight[0].zero_()

        nn.init.ones_(self.embedding_layer_norm.weight)
        nn.init.zeros_(self.embedding_layer_norm.bias)
        nn.init.zeros_(self.output_bias)

        # Keep PyTorch TransformerEncoderLayer default Linear initialization;
        # it is stable for this setup. Explicitly zero padding row after any
        # possible future initialization changes.
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            if self.output_embedding is not None:
                self.output_embedding.weight[0].zero_()

    def _infer_lengths(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if lengths is not None:
            return lengths.to(device=sequences.device, dtype=torch.long).clamp(min=1, max=sequences.size(1))
        inferred = (sequences != 0).sum(dim=1).to(dtype=torch.long)
        return inferred.clamp(min=1, max=sequences.size(1))

    def _build_embeddings(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build item + position embeddings and padding mask.

        Returns
        -------
        x:
            [B, L'] hidden states before Transformer.
        padding_mask:
            [B, L'] bool mask where True means padding/ignored.
        """
        if sequences.dim() != 2:
            raise ValueError(f"sequences must be [B, L], got shape={tuple(sequences.shape)}")

        batch_size, seq_len = sequences.shape
        if seq_len > self.max_seq_len:
            # Data loader should truncate, but keep a safe fallback.
            sequences = sequences[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        item_emb = self.item_embedding(sequences)
        padding_mask = sequences.eq(0)

        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, 1, self.hidden_size)
            item_emb = torch.cat([cls, item_emb], dim=1)
            cls_mask = torch.zeros(batch_size, 1, device=sequences.device, dtype=torch.bool)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            seq_len = seq_len + 1

        position_ids = torch.arange(seq_len, device=sequences.device, dtype=torch.long).unsqueeze(0)
        pos_emb = self.position_embedding(position_ids).expand(batch_size, -1, -1)
        x = item_emb + pos_emb
        x = self.embedding_layer_norm(x)
        x = self.embedding_dropout(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, padding_mask

    def encode_sequence(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode prefix sequence into a single user state."""
        lengths = self._infer_lengths(sequences, lengths)
        x, padding_mask = self._build_embeddings(sequences)

        hidden = self.encoder(x, src_key_padding_mask=padding_mask)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_cls_token:
            final_hidden = hidden[:, 0, :]
        else:
            # If no CLS token, gather the final non-padding observed item state.
            # hidden has the same sequence length as `sequences` when CLS=False.
            gather_idx = (lengths - 1).clamp(min=0, max=hidden.size(1) - 1)
            final_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), gather_idx]

        return torch.nan_to_num(final_hidden, nan=0.0, posinf=0.0, neginf=0.0)

    def get_output_item_weight(self) -> torch.Tensor:
        if self.tie_output_embedding:
            return self.item_embedding.weight
        assert self.output_embedding is not None
        return self.output_embedding.weight

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del user_ids
        final_hidden = self.encode_sequence(sequences, lengths)
        item_weight = self.get_output_item_weight()
        scores = torch.matmul(final_hidden, item_weight.t()) + self.output_bias
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[:, 0] = 0.0
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.full_sort_scores(user_ids, sequences, lengths)
