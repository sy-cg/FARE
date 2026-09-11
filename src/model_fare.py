# -*- coding: utf-8 -*-
"""
src/model_fare.py

FARE: fairness-aware multimodal residual representation for sequential recommendation.

Current design
--------------
FARE is a generic multimodal fairness representation framework built on top
of an ID-based sequential recommendation backbone.

Supported ID backbones:
    - SASRec
    - GRU4Rec
    - BERT4Rec

The backbone models user behavior from item IDs. FARE adds a multimodal
residual branch based on text/vision item features:

    score(u, j) = score_id(u, j) + w_fair * <Q h_u, z_fair_j>

where:
    h_u       : user sequence representation from ID backbone
    z_fair_j  : preference-relevant / debiased multimodal item representation
    Q         : user-side projection into fair residual space
    w_fair    : bounded learnable residual weight

Engineering convention
----------------------
    scripts/run_fare.py
        FARE final Exposure trainer

This model file intentionally does not implement exposure loss internally.
Exposure/diversity/fairness metrics are evaluated by evaluator_fairness.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_sasrec import SASRecID
    from .model_late_fusion import build_or_load_projected_features, _init_logit_from_weight
except ImportError:
    from model_sasrec import SASRecID  # type: ignore
    from model_late_fusion import build_or_load_projected_features, _init_logit_from_weight  # type: ignore


# Optional backbone imports.
# These try/except blocks make the file usable even before every backbone is implemented.
try:
    from .model_gru4rec import GRU4RecID  # type: ignore
except Exception:
    try:
        from model_gru4rec import GRU4RecID  # type: ignore
    except Exception:
        GRU4RecID = None  # type: ignore

try:
    from .model_bert4rec import BERT4RecID  # type: ignore
except Exception:
    try:
        from model_bert4rec import BERT4RecID  # type: ignore
    except Exception:
        BERT4RecID = None  # type: ignore


# ==========================================================
# Small modules
# ==========================================================


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    layers = []
    if use_layer_norm:
        layers.append(nn.LayerNorm(input_dim))
    layers.extend(
        [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ]
    )
    return nn.Sequential(*layers)


# ==========================================================
# Backbone utilities
# ==========================================================


def _has_var_kwargs(cls: Type[nn.Module]) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False

    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _adapt_constructor_kwargs(cls: Type[nn.Module], common: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instantiate backbones even if their constructor uses slightly different names.

    Preferred constructor style in this project:
        num_items, max_seq_len, hidden_size, num_layers, num_heads, dropout, ...

    Supported common aliases:
        item_num / n_items / num_item
        max_len / seq_len
        embedding_dim / embed_dim / hidden_dim
        n_layers / num_blocks
        dropout_rate
        tie_embeddings
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(common)

    if _has_var_kwargs(cls):
        return dict(common)

    params = set(sig.parameters.keys())
    out: Dict[str, Any] = {}

    alias_map = {
        "num_items": ["num_items", "item_num", "n_items", "num_item", "item_count"],
        "max_seq_len": ["max_seq_len", "max_len", "seq_len", "sequence_length"],
        "hidden_size": ["hidden_size", "hidden_dim", "embedding_dim", "embed_dim", "d_model"],
        "num_layers": ["num_layers", "n_layers", "num_blocks", "n_blocks"],
        "num_heads": ["num_heads", "n_heads"],
        "dropout": ["dropout", "dropout_rate"],
        "activation": ["activation"],
        "layer_norm_eps": ["layer_norm_eps"],
        "tie_output_embedding": ["tie_output_embedding", "tie_embeddings"],
    }

    used_common_keys = set()

    for common_key, aliases in alias_map.items():
        if common_key not in common:
            continue
        for alias in aliases:
            if alias in params:
                out[alias] = common[common_key]
                used_common_keys.add(common_key)
                break

    for key, value in common.items():
        if key in params and key not in out:
            out[key] = value
            used_common_keys.add(key)

    return out


def _build_backbone(
    backbone_type: str,
    num_items: int,
    max_seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    activation: str,
    layer_norm_eps: float,
    tie_output_embedding: bool,
) -> nn.Module:
    backbone_type = str(backbone_type).lower().strip()

    backbone_cls: Optional[Type[nn.Module]]

    if backbone_type == "sasrec":
        backbone_cls = SASRecID
    elif backbone_type == "gru4rec":
        backbone_cls = GRU4RecID
    elif backbone_type == "bert4rec":
        backbone_cls = BERT4RecID
    else:
        raise ValueError(
            f"Unknown backbone_type={backbone_type!r}. "
            "Supported: sasrec, gru4rec, bert4rec."
        )

    if backbone_cls is None:
        raise ImportError(
            f"Backbone {backbone_type!r} is not available. "
            f"Please check src/model_{backbone_type}.py and its class name."
        )

    common_kwargs = {
        "num_items": int(num_items),
        "max_seq_len": int(max_seq_len),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dropout": float(dropout),
        "activation": str(activation),
        "layer_norm_eps": float(layer_norm_eps),
        "tie_output_embedding": bool(tie_output_embedding),
    }

    kwargs = _adapt_constructor_kwargs(backbone_cls, common_kwargs)
    return backbone_cls(**kwargs)


def _strip_prefix_from_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)

    out: Dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[plen:]] = value
        else:
            out[key] = value
    return out


def _only_prefixed_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[plen:]] = value
    return out


# ==========================================================
# FARE model
# ==========================================================


class FARE(nn.Module):
    """
    Generic FARE model.

    It wraps an ID backbone and adds a multimodal fair residual branch.

    Required backbone methods:
        encode_sequence(sequences, lengths=None) -> Tensor[B, H]

    Preferred backbone method:
        full_sort_scores(user_ids, sequences, lengths=None) -> Tensor[B, num_items + 1]

    If full_sort_scores is not available, this class falls back to:
        final_hidden @ item_embedding.weight.T + output_bias
    """

    def __init__(
        self,
        num_items: int,
        data_dir: str | Path,
        group_num_classes: Optional[Dict[str, int]] = None,
        max_seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        tie_output_embedding: bool = True,
        backbone_type: str = "sasrec",
        mm_score_dim: int = 64,
        fair_dim: int = 128,
        projection_seed: int = 2026,
        use_text: bool = True,
        use_vision: bool = True,
        cache_projected_features: bool = True,
        feature_chunk_size: int = 32768,
        encoder_hidden_dim: int = 128,
        encoder_dropout: float = 0.1,
        fair_weight_init: float = 0.01,
        max_fair_weight: float = 0.1,
        residual_score_weight: float = 1.0,
        learnable_fair_weight: bool = True,
        normalize_representations: bool = True,
    ) -> None:
        super().__init__()

        if not use_text and not use_vision:
            raise ValueError("At least one modality must be enabled.")

        self.num_items = int(num_items)
        self.data_dir = str(data_dir)
        self.backbone_type = str(backbone_type).lower().strip()

        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(hidden_size)
        self.tie_output_embedding = bool(tie_output_embedding)

        self.mm_score_dim = int(mm_score_dim)
        self.fair_dim = int(fair_dim)
        self.use_text = bool(use_text)
        self.use_vision = bool(use_vision)
        fair_weight_init = float(fair_weight_init)
        self.max_fair_weight = float(max_fair_weight)
        if self.max_fair_weight < 0.0:
            raise ValueError(f"max_fair_weight must be non-negative, got {self.max_fair_weight}")
        if self.max_fair_weight == 0.0 and fair_weight_init != 0.0:
            raise ValueError("fair_weight_init must be 0.0 when max_fair_weight is 0.0")
        self.residual_score_weight = float(residual_score_weight)
        self.learnable_fair_weight = bool(learnable_fair_weight)
        self.normalize_representations = bool(normalize_representations)
        self.group_num_classes = dict(group_num_classes or {})

        # -------------------------
        # ID backbone
        # -------------------------
        self.backbone = _build_backbone(
            backbone_type=self.backbone_type,
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

        # -------------------------
        # Multimodal item features
        # -------------------------
        parts = []

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
                raise ValueError(
                    f"Text feature item count mismatch: {text_item.shape[0]} vs {num_items + 1}"
                )
            parts.append(text_item)

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
                raise ValueError(
                    f"Vision feature item count mismatch: {vision_item.shape[0]} vs {num_items + 1}"
                )
            parts.append(vision_item)

        mm_item = torch.cat(parts, dim=1).float()
        mm_item[0].zero_()
        self.mm_input_dim = int(mm_item.shape[1])
        self.register_buffer("mm_item_features", mm_item, persistent=False)

        # -------------------------
        # Fair item encoder
        # -------------------------
        self.fair_item_encoder = make_mlp(
            input_dim=self.mm_input_dim,
            hidden_dim=int(encoder_hidden_dim),
            output_dim=fair_dim,
            dropout=encoder_dropout,
            use_layer_norm=True,
        )

        self.user_fair_query = nn.Linear(hidden_size, fair_dim, bias=False)

        # -------------------------
        # Bounded residual weight
        # -------------------------
        if self.learnable_fair_weight:
            raw_fair_weight = 0.0
            if self.max_fair_weight > 0.0:
                raw_fair_weight = _init_logit_from_weight(fair_weight_init, self.max_fair_weight)
            self.raw_fair_weight = nn.Parameter(
                torch.tensor(raw_fair_weight)
            )
        else:
            self.register_buffer(
                "raw_fair_weight",
                torch.tensor(float(fair_weight_init)),
                persistent=True,
            )

        self._reset_fare_parameters()

    # ======================================================
    # Compatibility properties
    # ======================================================

    @property
    def item_embedding(self) -> nn.Module:
        if hasattr(self.backbone, "item_embedding"):
            return getattr(self.backbone, "item_embedding")
        raise AttributeError(f"{self.backbone_type} backbone has no item_embedding attribute.")

    @property
    def output_embedding(self) -> nn.Module:
        if hasattr(self.backbone, "output_embedding"):
            return getattr(self.backbone, "output_embedding")
        if hasattr(self.backbone, "item_embedding"):
            return getattr(self.backbone, "item_embedding")
        raise AttributeError(f"{self.backbone_type} backbone has no output_embedding/item_embedding attribute.")

    @property
    def output_bias(self) -> torch.Tensor:
        if hasattr(self.backbone, "output_bias"):
            return getattr(self.backbone, "output_bias")

        # Fallback zero bias on the correct device.
        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = self.raw_fair_weight.device
        return torch.zeros(self.num_items + 1, device=device)

    # ======================================================
    # Initialization / checkpoint / freezing
    # ======================================================

    def _reset_fare_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_fair_query.weight)

    def load_backbone_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ):
        """
        Load an ID-backbone checkpoint into self.backbone.

        Supports:
            - plain backbone checkpoints:
                item_embedding.weight, ...
            - wrapped checkpoints:
                backbone.item_embedding.weight, ...
                id_backbone.item_embedding.weight, ...
                model.item_embedding.weight, ...
                module.item_embedding.weight, ...
            - full FARE checkpoints:
                backbone.xxx plus fair branch keys

        Returns:
            missing_keys, unexpected_keys
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")

        # If a full FARE checkpoint is passed, prefer the explicit backbone.* subset.
        explicit_prefixes = [
            "backbone.",
            "id_backbone.",
            "base_model.",
            "model.backbone.",
            "module.backbone.",
            "module.id_backbone.",
        ]

        candidate_states = []

        for prefix in explicit_prefixes:
            sub = _only_prefixed_state_dict(state_dict, prefix)
            if sub:
                candidate_states.append(sub)

        # Also try common stripping for checkpoints saved under module/model.
        candidate_states.append(_strip_prefix_from_state_dict(state_dict, "module."))
        candidate_states.append(_strip_prefix_from_state_dict(state_dict, "model."))
        candidate_states.append(dict(state_dict))

        last_result = None
        last_error: Optional[Exception] = None

        for cand in candidate_states:
            try:
                result = self.backbone.load_state_dict(cand, strict=strict)
                last_result = result

                # Prefer the first candidate with at least some matched keys.
                missing = list(result.missing_keys)
                unexpected = list(result.unexpected_keys)
                loaded_key_count = len(cand) - len(unexpected)

                if loaded_key_count > 0:
                    return missing, unexpected
            except Exception as exc:
                last_error = exc

        if last_result is not None:
            return list(last_result.missing_keys), list(last_result.unexpected_keys)

        if last_error is not None:
            raise last_error

        return [], list(state_dict.keys())

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ======================================================
    # Core encoding / scoring
    # ======================================================

    def _bounded_fair_weight(self) -> torch.Tensor:
        raw = self.raw_fair_weight
        if not torch.isfinite(raw).all():
            raise FloatingPointError(f"Non-finite raw_fair_weight: {raw}")

        if self.max_fair_weight == 0.0:
            return raw.new_zeros(())

        if self.learnable_fair_weight:
            return self.max_fair_weight * torch.sigmoid(raw.clamp(min=-20.0, max=20.0))

        return raw.clamp(min=0.0, max=self.max_fair_weight)

    def get_fair_weight(self) -> float:
        with torch.no_grad():
            return float(self._bounded_fair_weight().item())

    def encode_sequence(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return final user representation h_u from the ID backbone.

        All ID backbones used with FARE should implement encode_sequence().
        """
        if hasattr(self.backbone, "encode_sequence"):
            return self.backbone.encode_sequence(sequences, lengths)  # type: ignore[attr-defined]

        raise AttributeError(
            f"{self.backbone_type} backbone does not implement encode_sequence(sequences, lengths). "
            "Please add encode_sequence() to the backbone model."
        )

    def _id_full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor],
        final_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ID-backbone full-sort scores.

        Preferred path:
            backbone.full_sort_scores()

        Fallback path:
            encode_sequence() @ output_embedding.T + output_bias
        """
        if hasattr(self.backbone, "full_sort_scores"):
            scores = self.backbone.full_sort_scores(user_ids, sequences, lengths)  # type: ignore[attr-defined]
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            if scores.size(1) == self.num_items + 1:
                scores[:, 0] = 0.0
            return scores

        if final_hidden is None:
            final_hidden = self.encode_sequence(sequences, lengths)

        if hasattr(self.backbone, "tie_output_embedding"):
            tie = bool(getattr(self.backbone, "tie_output_embedding"))
        else:
            tie = self.tie_output_embedding

        if tie and hasattr(self.backbone, "item_embedding"):
            item_weight = getattr(self.backbone, "item_embedding").weight
        elif hasattr(self.backbone, "output_embedding"):
            item_weight = getattr(self.backbone, "output_embedding").weight
        elif hasattr(self.backbone, "item_embedding"):
            item_weight = getattr(self.backbone, "item_embedding").weight
        else:
            raise AttributeError(
                f"{self.backbone_type} backbone has neither full_sort_scores() nor item/output embeddings."
            )

        output_bias = self.output_bias
        scores = torch.matmul(final_hidden, item_weight.t()) + output_bias
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if scores.size(1) == self.num_items + 1:
            scores[:, 0] = 0.0

        return scores

    # ======================================================
    # Multimodal item encoders
    # ======================================================

    def encode_all_fair_items(self) -> torch.Tensor:
        mm = self.mm_item_features
        z = self.fair_item_encoder(mm)

        if self.normalize_representations:
            z = F.normalize(z, p=2, dim=-1)

        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = z.clone()
        z[0].zero_()
        return z

    # ======================================================
    # Recommendation scores
    # ======================================================

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        final_hidden = self.encode_sequence(sequences, lengths)

        id_scores = self._id_full_sort_scores(
            user_ids=user_ids,
            sequences=sequences,
            lengths=lengths,
            final_hidden=final_hidden,
        )

        q = self.user_fair_query(final_hidden)

        if self.normalize_representations:
            q = F.normalize(q, p=2, dim=-1)

        z_items = self.encode_all_fair_items().to(device=q.device, dtype=q.dtype)
        fair_scores = torch.matmul(q, z_items.t())
        fair_scores = torch.nan_to_num(fair_scores, nan=0.0, posinf=0.0, neginf=0.0)

        residual_weight = self.residual_score_weight * self._bounded_fair_weight().to(q.dtype)
        scores = id_scores + residual_weight * fair_scores
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if scores.size(1) == self.num_items + 1:
            scores[:, 0] = 0.0

        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.full_sort_scores(user_ids, sequences, lengths)
