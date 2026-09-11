# -*- coding: utf-8 -*-
"""
src/model_sasrec.py

A clean SASRec-ID baseline implementation.

Design choices:
- Item id 0 is padding.
- Input sequences are right-padded and left-truncated by the data loader.
- The final hidden state at the last position is used for next-item prediction.
- full_sort_scores returns logits over all items, including padding item 0.
  The evaluator masks item 0; the loss masks item 0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SASRecID(nn.Module):
    """SASRec with ID embeddings only."""

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
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(hidden_size)
        self.tie_output_embedding = bool(tie_output_embedding)

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.input_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        if not self.tie_output_embedding:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        else:
            self.output_embedding = None

        self.output_bias = nn.Parameter(torch.zeros(num_items + 1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.output_embedding is not None:
            nn.init.normal_(self.output_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.output_embedding.weight[0].fill_(0.0)
        nn.init.zeros_(self.output_bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return bool upper-triangular causal mask for TransformerEncoder."""
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def encode_sequence(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor] = None   ) -> torch.Tensor:
        """Encode right-padded sequences and return the final valid hidden state.

        Args:
            sequences: LongTensor [B, L], right-padded with 0.
            lengths: LongTensor [B], number of valid non-padding items.

        Returns:
            final_hidden: FloatTensor [B, hidden_size]
        """
        if sequences.ndim != 2:
            raise ValueError(f"sequences must be [B, L], got {tuple(sequences.shape)}")

        batch_size, seq_len = sequences.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"input seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = sequences.device

        if lengths is None:
            lengths = sequences.ne(0).sum(dim=1)
        lengths = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=seq_len)

        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.item_embedding(sequences) + self.position_embedding(positions)
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        padding_mask = sequences.eq(0)
        causal_mask = self._causal_mask(seq_len, device)

        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        encoded = self.final_layer_norm(encoded)

        final_index = (lengths - 1).clamp(min=0, max=seq_len - 1)
        batch_index = torch.arange(batch_size, device=device)
        final_hidden = encoded[batch_index, final_index, :]

        if not torch.isfinite(final_hidden).all():
            raise FloatingPointError(
                "Non-finite final_hidden detected in SASRecID.encode_sequence. "
                "Check padding/mask logic and input sequences."
            )

        return final_hidden

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return full-sort logits over all item ids [0, num_items]."""
        del user_ids  # SASRec-ID does not use user ids directly.
        final_hidden = self.encode_sequence(sequences, lengths)
        item_weight = self.item_embedding.weight if self.tie_output_embedding else self.output_embedding.weight
        scores = torch.matmul(final_hidden, item_weight.t()) + self.output_bias
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.full_sort_scores(user_ids, sequences, lengths)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
