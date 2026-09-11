# -*- coding: utf-8 -*-
"""
FindRec baseline adapted to this project's full-sort evaluation pipeline.

This implementation follows the released FindRec/Fluid-MMRec code structure:
- Mamba sequence encoder for ID embeddings;
- Stein-guided multi-view entropy bottleneck for image/text alignment;
- cross-modal multi-head attention;
- expert routing over multimodal head features;
- final ID + multimodal fusion for next-item prediction.

The adaptation removes the RecBole dependency and uses this project's processed
feature matrices directly. Item id 0 remains padding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - depends on training environment
            raise ImportError(
                "FindRec requires the optional package 'mamba-ssm'. "
                "Install it in the experiment environment before running scripts/run_findrec.py."
            ) from exc

        mamba_config = config["mamba"]
        self.num_layers = int(mamba_config["num_layers"])
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=int(mamba_config["hidden_dim"]),
                    d_state=int(mamba_config["d_state"]),
                    d_conv=int(mamba_config["d_conv"]),
                    expand=int(mamba_config["expand"]),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(int(mamba_config["hidden_dim"]), eps=float(mamba_config["norm_eps"]))
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(float(config["dropout_prob"]))
        self.layer_scales = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, 1, int(mamba_config["hidden_dim"])) * 0.1)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            residual = x
            y = self.norms[i](x)
            y = self.mamba_layers[i](y)
            y = self.dropout(y)
            x = residual + self.layer_scales[i] * y
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.num_heads = int(config["num_attention_heads"])
        self.hidden_size = int(config["multimodal"]["hidden_size"])
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("multimodal.hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads

        self.image_projection = nn.Sequential(
            nn.Linear(int(config["image"]["projection_dim"]), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(config["dropout_prob"])),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(int(config["text"]["projection_dim"]), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(config["dropout_prob"])),
        )
        self.image_to_text_heads = nn.ModuleList(
            [nn.MultiheadAttention(self.head_dim, 1, dropout=float(config["dropout_prob"])) for _ in range(self.num_heads)]
        )
        self.text_to_image_heads = nn.ModuleList(
            [nn.MultiheadAttention(self.head_dim, 1, dropout=float(config["dropout_prob"])) for _ in range(self.num_heads)]
        )
        self.head_norms = nn.ModuleList([nn.LayerNorm(self.head_dim * 2) for _ in range(self.num_heads)])
        self.scaling = float(self.head_dim) ** -0.5

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = image_features.size(0), image_features.size(1)
        image_proj = F.normalize(self.image_projection(image_features), p=2, dim=-1)
        text_proj = F.normalize(self.text_projection(text_features), p=2, dim=-1)

        image_heads = image_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        text_heads = text_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_padding_mask = None if attention_mask is None else (~attention_mask).to(image_features.device).bool()

        outputs = []
        for head_idx in range(self.num_heads):
            curr_image = image_heads[:, head_idx] * self.scaling
            curr_text = text_heads[:, head_idx] * self.scaling
            curr_image_t = curr_image.transpose(0, 1)
            curr_text_t = curr_text.transpose(0, 1)

            img2text, _ = self.image_to_text_heads[head_idx](
                query=curr_image_t,
                key=curr_text_t,
                value=curr_text_t,
                key_padding_mask=key_padding_mask,
            )
            text2img, _ = self.text_to_image_heads[head_idx](
                query=curr_text_t,
                key=curr_image_t,
                value=curr_image_t,
                key_padding_mask=key_padding_mask,
            )
            combined = torch.cat(
                [curr_image + img2text.transpose(0, 1), curr_text + text2img.transpose(0, 1)],
                dim=-1,
            )
            outputs.append(self.head_norms[head_idx](combined))

        return torch.stack(outputs, dim=1)


class ExpertRouter(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.num_experts = int(config["expert"]["num_experts"])
        head_dim = int(config["multimodal"]["hidden_size"]) // int(config["num_attention_heads"])
        input_dim = head_dim * 2
        hidden_size = int(config["router"]["hidden_size"])

        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(float(config["router"]["dropout"])),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.num_experts),
        )
        self.temperature = 0.1
        self.balance_coefficient = 0.001
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_heads, seq_len, dim = x.size()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = F.normalize(x.reshape(-1, dim), dim=-1).view(batch_size, num_heads, seq_len, dim)
        logits = self.router(x.view(batch_size * num_heads, seq_len, dim))
        log_gates = F.log_softmax(logits / self.temperature, dim=-1)

        if training:
            gates = F.gumbel_softmax(log_gates + torch.rand_like(log_gates) * 0.05, tau=self.temperature, hard=True, dim=-1)
        else:
            gates = torch.exp(log_gates)

        gates = gates.view(batch_size, num_heads, seq_len, -1).clamp(min=1e-6, max=1.0)
        if not training:
            return gates, None

        expert_usage = gates.mean(dim=(0, 1, 2)).clamp(min=1e-6)
        balance_loss = -torch.sum(expert_usage * torch.log(expert_usage))
        return gates, self.balance_coefficient * balance_loss.clamp(max=1.0)


class SteinKernel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        bottleneck_cfg = config["bottleneck"]
        self.adaptive_bandwidth = bool(bottleneck_cfg["adaptive_bandwidth"])
        self.min_bandwidth = float(bottleneck_cfg["min_bandwidth"])
        self.max_bandwidth = float(bottleneck_cfg["max_bandwidth"])
        self.bandwidth_factor = float(bottleneck_cfg["bandwidth_factor"])
        self.register_buffer("bandwidth", torch.tensor(1.0))

    def update_bandwidth(self, x: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            sample_size = min(512, x.size(0))
            diff = x[:sample_size].unsqueeze(1) - y[:sample_size].unsqueeze(0)
            dist_sq = torch.sum(diff * diff, dim=-1)
            bandwidth_sq = torch.median(dist_sq.reshape(-1))
            bandwidth = torch.sqrt(bandwidth_sq / 2.0).clamp(min=1e-6) * self.bandwidth_factor
            self.bandwidth = torch.clamp(bandwidth, min=self.min_bandwidth, max=self.max_bandwidth)

    def paired_rbf_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.update_bandwidth(x, y)
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-dist_sq / (2 * self.bandwidth**2))


class MultiViewEntropyBottleneck(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["multimodal"]["hidden_size"])
        self.output_dim = int(config["bottleneck"]["dim"])
        self.beta = float(config["bottleneck"]["beta"])

        self.image_encoder = nn.Sequential(
            nn.Linear(int(config["image"]["projection_dim"]), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(float(config["multimodal"]["projection_dropout"])),
            nn.GELU(),
            nn.Linear(hidden_size, self.output_dim * 2),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(int(config["text"]["projection_dim"]), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(float(config["multimodal"]["projection_dropout"])),
            nn.GELU(),
            nn.Linear(hidden_size, self.output_dim * 2),
        )
        self.stein_kernel = SteinKernel(config)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def compute_kl_loss(
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)
        kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)
        if mask is not None:
            sequence_lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            batch_indices = torch.arange(mask.size(0), device=mask.device)
            kl1 = kl1[batch_indices, sequence_lengths]
            kl2 = kl2[batch_indices, sequence_lengths]
        return (kl1.mean() + kl2.mean()) / 2

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        image_mu, image_logvar = torch.chunk(self.image_encoder(image_features), 2, dim=-1)
        text_mu, text_logvar = torch.chunk(self.text_encoder(text_features), 2, dim=-1)
        image_logvar = image_logvar.clamp(min=-10.0, max=10.0)
        text_logvar = text_logvar.clamp(min=-10.0, max=10.0)
        image_z = F.normalize(self.reparameterize(image_mu, image_logvar), p=2, dim=-1)
        text_z = F.normalize(self.reparameterize(text_mu, text_logvar), p=2, dim=-1)

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1).long().clamp(min=1) - 1
            batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
            image_z_last = image_z[batch_indices, sequence_lengths]
            text_z_last = text_z[batch_indices, sequence_lengths]
        else:
            image_z_last = image_z[:, -1]
            text_z_last = text_z[:, -1]

        alignment_similarity = self.stein_kernel.paired_rbf_similarity(image_z_last, text_z_last).mean()
        # The overall objective is minimized, so maximizing the paper's paired
        # RBF similarity is equivalent to minimizing this bounded distance.
        alignment_loss = 1.0 - alignment_similarity
        kl_loss = self.compute_kl_loss(image_mu, image_logvar, text_mu, text_logvar, attention_mask)
        return {
            "image_repr": image_z,
            "text_repr": text_z,
            "alignment_loss": alignment_loss,
            "kl_loss": kl_loss,
            "total_loss": kl_loss + self.beta * alignment_loss,
        }


def _load_feature_tensor(data_dir: str | Path, feature_file: str, mask_file: Optional[str] = None) -> torch.Tensor:
    data_dir = Path(data_dir)
    arr = np.load(data_dir / feature_file).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix: {data_dir / feature_file}, got {arr.shape}")
    if mask_file:
        mask = np.load(data_dir / mask_file).astype(np.float32)
        if mask.shape[0] != arr.shape[0]:
            raise ValueError(f"Mask shape mismatch for {feature_file}: {mask.shape} vs {arr.shape}")
        arr = arr * mask[:, None]
    arr[0] = 0.0
    return torch.from_numpy(arr)


class FindRec(nn.Module):
    """Project-native FindRec baseline with full_sort_scores interface."""

    def __init__(
        self,
        num_items: int,
        data_dir: str | Path,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.config = config
        self.id_embedding_dim = int(config["id_embedding_dim"])
        self.modal_hidden_size = int(config["multimodal"]["hidden_size"])
        self.dropout_prob = float(config["dropout_prob"])
        self.loss_type = str(config.get("loss_type", "CE")).upper()
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = self.modal_hidden_size // self.num_heads
        if self.id_embedding_dim != int(config["mamba"]["hidden_dim"]):
            raise ValueError("id_embedding_dim must match mamba.hidden_dim for FindRec.")

        model_cfg = config.get("model", {})
        self.text_features = _load_feature_tensor(
            data_dir,
            str(model_cfg.get("text_feature_file", "text_features.npy")),
            str(model_cfg.get("text_mask_file", "text_mask.npy")),
        )
        self.image_features = _load_feature_tensor(
            data_dir,
            str(model_cfg.get("vision_feature_file", "vision_features.npy")),
            str(model_cfg.get("vision_mask_file", "vision_mask.npy")),
        )
        if self.text_features.shape[0] != self.num_items + 1:
            raise ValueError(f"Text feature item count mismatch: {self.text_features.shape[0]} vs {self.num_items + 1}")
        if self.image_features.shape[0] != self.num_items + 1:
            raise ValueError(f"Image feature item count mismatch: {self.image_features.shape[0]} vs {self.num_items + 1}")

        config = dict(config)
        config["text"] = dict(config["text"])
        config["image"] = dict(config["image"])
        config["text"]["feature_dim"] = int(self.text_features.shape[1])
        config["image"]["feature_dim"] = int(self.image_features.shape[1])
        self.config = config

        self.register_buffer("text_feature_buffer", self.text_features, persistent=False)
        self.register_buffer("image_feature_buffer", self.image_features, persistent=False)

        self.item_embedding = nn.Embedding(self.num_items + 1, self.id_embedding_dim, padding_idx=0)
        self.image_projection = nn.Linear(int(config["image"]["feature_dim"]), int(config["image"]["projection_dim"]))
        self.text_projection = nn.Linear(int(config["text"]["feature_dim"]), int(config["text"]["projection_dim"]))
        self.id_mamba = MambaLayer(config)
        self.cross_attention = MultiHeadCrossAttention(config)
        self.router = ExpertRouter(config)
        self.stein_mveb = MultiViewEntropyBottleneck(config)
        self.bottleneck_weight = float(config["bottleneck"]["weight"])

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.head_dim * 2, self.id_embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                )
                for _ in range(int(config["expert"]["num_experts"]))
            ]
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * self.num_heads, self.id_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.id_embedding_dim),
            nn.Dropout(float(config["multimodal"]["fusion_dropout"])),
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * 2, self.id_embedding_dim),
            nn.LayerNorm(self.id_embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )
        self.final_norm = nn.LayerNorm(self.id_embedding_dim)
        self.last_mveb_loss: Optional[torch.Tensor] = None
        self.last_mveb_parts: Dict[str, torch.Tensor] = {}
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_multimodal_features(self, item_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_seq = item_seq.clamp(min=0, max=self.num_items)
        attention_mask = safe_seq.ne(0)
        image_features = self.image_feature_buffer.to(item_seq.device)[safe_seq]
        text_features = self.text_feature_buffer.to(item_seq.device)[safe_seq]
        return image_features, text_features, attention_mask

    @staticmethod
    def gather_indexes(output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.size(-1))
        return output.gather(dim=1, index=gather_index).squeeze(1)

    def encode_sequence(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sequences.ndim != 2:
            raise ValueError(f"sequences must be [B, L], got {tuple(sequences.shape)}")
        if lengths is None:
            lengths = sequences.ne(0).sum(dim=1)
        lengths = lengths.to(device=sequences.device, dtype=torch.long).clamp(min=1, max=sequences.size(1))

        item_emb = self.item_embedding(sequences)
        id_seq_output = self.id_mamba(item_emb)
        image_features, text_features, mask = self.get_multimodal_features(sequences)
        image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)

        image_proj = F.normalize(self.image_projection(image_features), p=2, dim=-1)
        text_proj = F.normalize(self.text_projection(text_features), p=2, dim=-1)
        mveb_output = self.stein_mveb(image_proj, text_proj, mask)
        self.last_mveb_loss = mveb_output["total_loss"]
        self.last_mveb_parts = {
            "alignment_loss": mveb_output["alignment_loss"],
            "kl_loss": mveb_output["kl_loss"],
        }

        head_features = torch.nan_to_num(
            self.cross_attention(mveb_output["image_repr"], mveb_output["text_repr"], mask),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        gates, _ = self.router(head_features, training=self.training)

        expert_outputs = []
        batch_size, seq_len = sequences.size(0), sequences.size(1)
        for head_idx in range(self.num_heads):
            head_output = head_features[:, head_idx]
            head_gates = gates[:, head_idx]
            head_expert_output = torch.zeros(batch_size, seq_len, self.id_embedding_dim, device=sequences.device)
            for expert_idx, expert in enumerate(self.experts):
                expert_out = F.normalize(expert(head_output), p=2, dim=-1)
                head_expert_output = head_expert_output + expert_out * head_gates[:, :, expert_idx].unsqueeze(-1)
            expert_outputs.append(head_expert_output)

        modal_features = F.normalize(self.feature_fusion(torch.cat(expert_outputs, dim=-1)), p=2, dim=-1)
        fused_features = self.final_fusion(torch.cat([id_seq_output, modal_features], dim=-1))
        final_output = torch.clamp(self.final_norm(fused_features), min=-10.0, max=10.0)
        return self.gather_indexes(final_output, lengths - 1)

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del user_ids
        seq_output = self.encode_sequence(sequences, lengths)
        scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        scores[:, 0] = -1.0e9
        return scores

    def training_loss(
        self,
        user_ids: torch.Tensor,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.full_sort_scores(user_ids, sequences, lengths)
        rec_loss = F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))
        mveb_loss = self.last_mveb_loss if self.last_mveb_loss is not None else rec_loss.new_tensor(0.0)
        loss = rec_loss + self.bottleneck_weight * mveb_loss
        return loss, {
            "rec_loss": float(rec_loss.detach().cpu().item()),
            "mveb_loss": float(mveb_loss.detach().cpu().item()),
            "alignment_loss": float(
                self.last_mveb_parts.get("alignment_loss", rec_loss.new_tensor(0.0)).detach().cpu().item()
            ),
            "kl_loss": float(self.last_mveb_parts.get("kl_loss", rec_loss.new_tensor(0.0)).detach().cpu().item()),
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
