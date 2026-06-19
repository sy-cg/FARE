# -*- coding: utf-8 -*-
"""
src/model_adv_backbone.py

Adversarial ID-backbone baseline for item-side fairness.

Supported methods:
    Adv-SASRec
    Adv-GRU4Rec
    Adv-BERT4Rec

Design
------
This model wraps an ID-based sequential recommender and adds adversarial group
classifiers on the final user representation h_u.

Recommendation:
    score(u, j) = score_id(u, j)

Adversarial auxiliary task:
    GRL(h_u) -> group(label of target item)

Training objective:
    L = L_rec + lambda_adv * L_adv

Because of Gradient Reversal Layer, the group classifiers learn to predict item
fairness/proxy groups from h_u, while the ID backbone learns to make h_u less
predictive of those groups.

This is a pure ID fairness baseline. It does not use text/vision features.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_sasrec import SASRecID
except ImportError:
    from model_sasrec import SASRecID  # type: ignore

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
# Gradient reversal
# ==========================================================


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, lambd)


# ==========================================================
# Small modules
# ==========================================================


class GroupClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(num_classes) <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================================
# Backbone construction helpers
# ==========================================================


def _has_var_kwargs(cls: Type[nn.Module]) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _adapt_constructor_kwargs(cls: Type[nn.Module], common: Dict[str, Any]) -> Dict[str, Any]:
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

    for common_key, aliases in alias_map.items():
        if common_key not in common:
            continue
        for alias in aliases:
            if alias in params:
                out[alias] = common[common_key]
                break

    for key, value in common.items():
        if key in params and key not in out:
            out[key] = value

    return out


def build_id_backbone(
    backbone_type: str,
    num_items: int,
    max_seq_len: int = 50,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
    activation: str = "gelu",
    layer_norm_eps: float = 1e-12,
    tie_output_embedding: bool = True,
) -> nn.Module:
    backbone_type = str(backbone_type).lower().strip()

    if backbone_type == "sasrec":
        cls = SASRecID
    elif backbone_type == "gru4rec":
        cls = GRU4RecID
    elif backbone_type == "bert4rec":
        cls = BERT4RecID
    else:
        raise ValueError(
            f"Unknown backbone_type={backbone_type!r}. "
            "Supported: sasrec, gru4rec, bert4rec."
        )

    if cls is None:
        raise ImportError(
            f"Backbone {backbone_type!r} is not available. "
            f"Please check src/model_{backbone_type}.py."
        )

    common = {
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

    kwargs = _adapt_constructor_kwargs(cls, common)
    return cls(**kwargs)


def _only_prefixed_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    n = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[n:]] = v
    return out


def _strip_prefix_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    n = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[n:]] = v
        else:
            out[k] = v
    return out


# ==========================================================
# AdvIDBackbone
# ==========================================================


class AdvIDBackbone(nn.Module):
    """
    Generic adversarial fairness baseline over ID backbone.

    Required backbone interface:
        encode_sequence(sequences, lengths=None)

    Preferred:
        full_sort_scores(user_ids, sequences, lengths=None)

    Fallback:
        h_u @ item_embedding.weight.T + output_bias
    """

    def __init__(
        self,
        num_items: int,
        group_num_classes: Dict[str, int],
        backbone_type: str = "sasrec",
        max_seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        tie_output_embedding: bool = True,
        classifier_hidden_dim: int = 128,
        aux_dropout: float = 0.1,
        normalize_user_repr: bool = False,
    ) -> None:
        super().__init__()

        if not group_num_classes:
            raise ValueError("group_num_classes cannot be empty.")

        self.num_items = int(num_items)
        self.backbone_type = str(backbone_type).lower().strip()
        self.hidden_size = int(hidden_size)
        self.tie_output_embedding = bool(tie_output_embedding)
        self.group_num_classes = dict(group_num_classes)
        self.normalize_user_repr = bool(normalize_user_repr)

        self.backbone = build_id_backbone(
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

        self.adv_classifiers = nn.ModuleDict()
        for group_name, n_classes in self.group_num_classes.items():
            self.adv_classifiers[group_name] = GroupClassifier(
                input_dim=hidden_size,
                num_classes=int(n_classes),
                hidden_dim=int(classifier_hidden_dim),
                dropout=float(aux_dropout),
            )

    # ------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------

    @property
    def item_embedding(self) -> nn.Module:
        if hasattr(self.backbone, "item_embedding"):
            return getattr(self.backbone, "item_embedding")
        raise AttributeError(f"{self.backbone_type} backbone has no item_embedding.")

    @property
    def output_embedding(self) -> nn.Module:
        if hasattr(self.backbone, "output_embedding"):
            return getattr(self.backbone, "output_embedding")
        if hasattr(self.backbone, "item_embedding"):
            return getattr(self.backbone, "item_embedding")
        raise AttributeError(f"{self.backbone_type} backbone has no output_embedding/item_embedding.")

    @property
    def output_bias(self) -> torch.Tensor:
        if hasattr(self.backbone, "output_bias"):
            return getattr(self.backbone, "output_bias")
        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = next(self.parameters()).device
        return torch.zeros(self.num_items + 1, device=device)

    # ------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------

    def load_backbone_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ):
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be dict, got {type(state_dict)}")

        candidate_states = []

        for prefix in [
            "backbone.",
            "id_backbone.",
            "base_model.",
            "model.backbone.",
            "module.backbone.",
            "module.id_backbone.",
        ]:
            sub = _only_prefixed_state_dict(state_dict, prefix)
            if sub:
                candidate_states.append(sub)

        candidate_states.append(_strip_prefix_state_dict(state_dict, "module."))
        candidate_states.append(_strip_prefix_state_dict(state_dict, "model."))
        candidate_states.append(dict(state_dict))

        last_result = None
        last_error = None

        for cand in candidate_states:
            try:
                result = self.backbone.load_state_dict(cand, strict=strict)
                last_result = result

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
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------
    # Core methods
    # ------------------------------------------------------

    def encode_sequence(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "encode_sequence"):
            h = self.backbone.encode_sequence(sequences, lengths)  # type: ignore[attr-defined]
        else:
            raise AttributeError(
                f"{self.backbone_type} backbone does not implement encode_sequence()."
            )

        if self.normalize_user_repr:
            h = F.normalize(h, p=2, dim=-1)

        return h

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "full_sort_scores"):
            scores = self.backbone.full_sort_scores(user_ids, sequences, lengths)  # type: ignore[attr-defined]
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            if scores.size(1) == self.num_items + 1:
                scores[:, 0] = 0.0
            return scores

        h = self.encode_sequence(sequences, lengths)

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
                f"{self.backbone_type} backbone has neither full_sort_scores() nor output embedding."
            )

        scores = torch.matmul(h, item_weight.t()) + self.output_bias
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if scores.size(1) == self.num_items + 1:
            scores[:, 0] = 0.0

        return scores

    def group_logits(
        self,
        user_repr: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        z = grad_reverse(user_repr, lambd=grl_lambda)
        return {
            group_name: clf(z)
            for group_name, clf in self.adv_classifiers.items()
        }

    def forward(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.full_sort_scores(user_ids, sequences, lengths)