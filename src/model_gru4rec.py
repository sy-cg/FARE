# -*- coding: utf-8 -*-
"""
src/model_gru4rec.py

GRU4Rec-ID backbone for sequential recommendation.

Design goals:
- API-compatible with the existing SASRec-ID training/evaluation pipeline.
- Uses the same full-sort next-item classification interface:
      logits = model.full_sort_scores(user_ids, sequences, lengths)
- Padding item id is 0.
- Candidate scores are produced over item ids [0, num_items].
- The padding score is forced to zero before the evaluator masks it.

This implementation is an ID-only recurrent sequential backbone. It is intended as
an additional backbone for testing whether FARE-style modules are backbone
agnostic. A FARE extension can subclass this model in the same way that the
current FARE subclasses SASRecID.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GRU4RecID(nn.Module):
    """GRU4Rec with full-sort next-item scoring.

    Parameters
    ----------
    num_items:
        Number of non-padding items. The actual embedding table has
        num_items + 1 rows because item id 0 is reserved for padding.
    max_seq_len:
        Kept for API compatibility with SASRecID and run scripts. GRU itself
        does not require positional embeddings, but sequence tensors are still
        truncated/padded by the data loader to this length.
    hidden_size:
        Item embedding and GRU hidden size.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout between GRU layers when num_layers > 1 and on input embeddings.
    tie_output_embedding:
        If True, output item weights share the input item embedding matrix.
        If False, a separate output embedding matrix is used.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        tie_output_embedding: bool = True,
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.tie_output_embedding = bool(tie_output_embedding)
        self.embedding_init_std = float(embedding_init_std)

        self.item_embedding = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=0)
        self.input_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

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

        if self.output_embedding is not None:
            nn.init.normal_(self.output_embedding.weight, mean=0.0, std=self.embedding_init_std)
            with torch.no_grad():
                self.output_embedding.weight[0].zero_()

        # PyTorch's default GRU init is usable, but explicit Xavier/orthogonal
        # makes experiments more reproducible across environments.
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        nn.init.zeros_(self.output_bias)

    def _infer_lengths(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if lengths is not None:
            return lengths.to(device=sequences.device, dtype=torch.long).clamp(min=1, max=sequences.size(1))
        inferred = (sequences != 0).sum(dim=1).to(dtype=torch.long)
        return inferred.clamp(min=1, max=sequences.size(1))

    def encode_sequence(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a padded item sequence into a single user state.

        Returns
        -------
        torch.Tensor
            Shape [batch_size, hidden_size].
        """
        if sequences.dim() != 2:
            raise ValueError(f"sequences must be [B, L], got shape={tuple(sequences.shape)}")

        lengths = self._infer_lengths(sequences, lengths)
        emb = self.item_embedding(sequences)
        emb = self.input_dropout(emb)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Packing avoids learning from padded timesteps and is robust for users
        # with shorter histories.
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        final_hidden = h_n[-1]
        final_hidden = self.output_norm(final_hidden)
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
        """Return full-sort scores over all items including padding row."""
        del user_ids  # Kept for API compatibility.
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
