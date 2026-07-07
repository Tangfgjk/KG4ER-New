"""Semantic and pedagogical feature-aware ConvE model.

V7 keeps ID embeddings as the stable anchor and fuses side information with a
feature-token attention block. This avoids directly adding heterogeneous
features at full strength while still allowing the model to use semantic and
pedagogical evidence when it is helpful.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID, SemanticFeatureBundle


VALID_MODEL_ABLATIONS = {
    "full",
    "id_only",
    "direct_sum_fusion",
    "no_concept_text",
    "no_exercise_text",
    "no_text_semantic",
    "no_pedagogical",
    "no_relation_aware",
    "no_type_aware_scoring",
    "no_mastery",
    "no_forgetting",
    "no_seq",
    # Backward-compatible aliases used by older command files and tests.
    "no_content_entity",
    "no_semantic",
    "no_concept_semantic",
    "no_exercise_semantic",
    "no_exercise_irt",
    "no_learner_irt",
    "no_cluster",
    "no_relation_strength",
    "discrete_relation",
    "hybrid_relation",
    "concept_extra",
    "concept_name_only",
}

FEATURE_RESIDUAL_INITIAL_VALUE = 0.05


def _logit(value: float = FEATURE_RESIDUAL_INITIAL_VALUE) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class FeatureAttentionFusion(nn.Module):
    """Fuse projected feature tokens with self-attention and a small MLP."""

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            num_heads = 1
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError("tokens must be shaped [batch, token_count, embedding_dim]")
        if active_mask.shape != tokens.shape[:2]:
            raise ValueError("active_mask must be shaped [batch, token_count]")

        active_mask = active_mask.bool()
        # MultiheadAttention cannot handle rows where every key is masked. Type
        # tokens are normally active, but this fallback keeps ablations safe.
        safe_mask = active_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
            tokens = tokens.clone()
            tokens[empty_rows, 0] = 0.0

        normalized_tokens = self.token_norm(tokens)
        attended, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        weights = safe_mask.unsqueeze(-1).to(attended.dtype)
        pooled = (attended * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        return self.output_norm(self.output(pooled))


class SemanticConvE(nn.Module):
    """ConvE with attention-fused semantic, pedagogical and relation features."""

    def __init__(
        self,
        nentity: int,
        nrelation: int,
        text_dim: int,
        numeric_dim: int,
        relation_type_ids: torch.Tensor,
        relation_strengths: torch.Tensor,
        entity_type_ids: torch.Tensor,
        cluster_ids: torch.Tensor,
        text_features: torch.Tensor,
        numeric_features: torch.Tensor,
        semantic_quality: torch.Tensor,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        hidden_size: int = 9728,
        input_drop: float = 0.2,
        hidden_drop: float = 0.2,
        feat_drop: float = 0.3,
        use_bias: bool = True,
        freeze_text_features: bool = True,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        if ablation_mode not in VALID_MODEL_ABLATIONS:
            raise ValueError(f"Unknown SemanticConvE ablation mode: {ablation_mode}")
        if embedding_dim % embedding_shape1 != 0:
            raise ValueError("embedding_dim must be divisible by embedding_shape1")

        self.nentity = nentity
        self.nrelation = nrelation
        self.embedding_dim = embedding_dim
        self.emb_dim1 = embedding_shape1
        self.emb_dim2 = embedding_dim // embedding_shape1

        self.emb_e = nn.Embedding(nentity, embedding_dim)
        self.entity_type_emb = nn.Embedding(4, embedding_dim)
        self.cluster_emb = nn.Embedding(NO_CLUSTER_ID + 1, embedding_dim)
        self.relation_id_emb = nn.Embedding(nrelation, embedding_dim)
        self.relation_type_emb = nn.Embedding(len(RELATION_TYPE_TO_ID), embedding_dim)

        self.text_projector = nn.Linear(text_dim, embedding_dim, bias=False)
        self.numeric_projector = nn.Sequential(
            nn.Linear(numeric_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.relation_strength_projector = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.semantic_token_norm = nn.LayerNorm(embedding_dim)
        self.pedagogical_token_norm = nn.LayerNorm(embedding_dim)
        self.entity_type_token_norm = nn.LayerNorm(embedding_dim)
        self.cluster_token_norm = nn.LayerNorm(embedding_dim)
        self.relation_type_token_norm = nn.LayerNorm(embedding_dim)
        self.relation_strength_token_norm = nn.LayerNorm(embedding_dim)

        self.entity_fusion = FeatureAttentionFusion(embedding_dim, num_heads=4, dropout=0.0)
        self.relation_fusion = FeatureAttentionFusion(embedding_dim, num_heads=4, dropout=0.0)
        self.raw_entity_residual_scale = nn.Parameter(torch.tensor(_logit(), dtype=torch.float32))
        self.raw_relation_residual_scale = nn.Parameter(torch.tensor(_logit(), dtype=torch.float32))

        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.relation_norm = nn.LayerNorm(embedding_dim)

        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_map_drop = nn.Dropout2d(feat_drop)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.register_parameter("b", nn.Parameter(torch.zeros(nentity)))

        self.register_buffer("text_features", text_features.detach().clone())
        self.register_buffer("numeric_features", numeric_features.detach().clone())
        self.register_buffer("entity_type_ids", entity_type_ids.detach().clone())
        self.register_buffer("cluster_ids", cluster_ids.detach().clone())
        self.register_buffer("semantic_quality", semantic_quality.detach().clone())
        self.register_buffer("relation_type_ids", relation_type_ids.detach().clone())
        self.register_buffer("relation_strengths", relation_strengths.detach().clone())
        self.freeze_text_features = freeze_text_features
        self.ablation_mode = ablation_mode
        self.init()

    @classmethod
    def from_feature_bundle(cls, bundle: SemanticFeatureBundle, **kwargs) -> "SemanticConvE":
        return cls(
            nentity=bundle.nentity,
            nrelation=bundle.nrelation,
            text_dim=bundle.text_dim,
            numeric_dim=bundle.numeric_dim,
            relation_type_ids=bundle.relation_type_ids,
            relation_strengths=bundle.relation_strengths,
            entity_type_ids=bundle.entity_type_ids,
            cluster_ids=bundle.cluster_ids,
            text_features=bundle.text_features,
            numeric_features=bundle.numeric_features,
            semantic_quality=bundle.semantic_quality,
            **kwargs,
        )

    def init(self) -> None:
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.entity_type_emb.weight.data)
        xavier_normal_(self.cluster_emb.weight.data)
        xavier_normal_(self.relation_id_emb.weight.data)
        xavier_normal_(self.relation_type_emb.weight.data)

    def _semantic_is_disabled(self) -> bool:
        return self.ablation_mode in {"no_text_semantic", "no_semantic", "no_content_entity", "id_only"}

    def _pedagogical_is_disabled(self) -> bool:
        return self.ablation_mode in {"no_pedagogical", "no_content_entity", "id_only"}

    def _entity_feature_tokens(self, entity_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        type_ids = self.entity_type_ids[entity_ids]

        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        semantic_token = self.semantic_token_norm(self.text_projector(text_features))
        semantic_quality = self.semantic_quality[entity_ids].squeeze(-1).clamp(0.0, 1.0)
        semantic_active = semantic_quality > 0
        if self._semantic_is_disabled():
            semantic_active = torch.zeros_like(semantic_active)
        if self.ablation_mode in {"no_concept_text", "no_concept_semantic"}:
            semantic_active = semantic_active & (type_ids != ENTITY_TYPE_TO_ID["kc"])
        if self.ablation_mode in {"no_exercise_text", "no_exercise_semantic"}:
            semantic_active = semantic_active & (type_ids != ENTITY_TYPE_TO_ID["ex"])
        semantic_token = semantic_token * semantic_quality.unsqueeze(-1)

        pedagogical_token = self.pedagogical_token_norm(self.numeric_projector(self.numeric_features[entity_ids]))
        pedagogical_active = (type_ids == ENTITY_TYPE_TO_ID["uid"]) | (type_ids == ENTITY_TYPE_TO_ID["ex"])
        if self._pedagogical_is_disabled():
            pedagogical_active = torch.zeros_like(pedagogical_active)
        if self.ablation_mode == "no_exercise_irt":
            pedagogical_active = pedagogical_active & (type_ids != ENTITY_TYPE_TO_ID["ex"])
        if self.ablation_mode == "no_learner_irt":
            pedagogical_active = pedagogical_active & (type_ids != ENTITY_TYPE_TO_ID["uid"])

        type_token = self.entity_type_token_norm(self.entity_type_emb(type_ids))
        type_active = torch.ones_like(pedagogical_active)

        cluster_token = self.cluster_token_norm(
            self.cluster_emb(self.cluster_ids[entity_ids].clamp(min=0, max=NO_CLUSTER_ID))
        )
        cluster_active = type_ids == ENTITY_TYPE_TO_ID["uid"]
        if self.ablation_mode in {"no_pedagogical", "no_content_entity", "no_cluster", "id_only"}:
            cluster_active = torch.zeros_like(cluster_active)

        tokens = torch.stack([semantic_token, pedagogical_token, type_token, cluster_token], dim=1)
        mask = torch.stack([semantic_active, pedagogical_active, type_active, cluster_active], dim=1)
        return tokens, mask

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)

        tokens, mask = self._entity_feature_tokens(entity_ids)
        if self.ablation_mode == "direct_sum_fusion":
            extra = (tokens * mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1)
            return self.entity_norm(id_emb + extra)

        fused_extra = self.entity_fusion(tokens, mask)
        residual_scale = torch.sigmoid(self.raw_entity_residual_scale)
        return self.entity_norm(id_emb + residual_scale * fused_extra)

    def gate_values(self) -> dict[str, dict[str, float]]:
        names_by_id = {idx: name for name, idx in ENTITY_TYPE_TO_ID.items()}
        entity_scale = float(torch.sigmoid(self.raw_entity_residual_scale).detach().cpu().item())
        relation_scale = float(torch.sigmoid(self.raw_relation_residual_scale).detach().cpu().item())
        values = {
            names_by_id[idx]: {
                "feature_residual_scale": entity_scale,
                "attention_fusion": 0.0 if self.ablation_mode == "direct_sum_fusion" else 1.0,
            }
            for idx in sorted(names_by_id)
        }
        values["relation"] = {
            "feature_residual_scale": relation_scale,
            "attention_fusion": 0.0 if self.ablation_mode == "direct_sum_fusion" else 1.0,
            "relation_id_anchor": 1.0,
        }
        return values

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if self.ablation_mode in {"discrete_relation", "id_only", "no_relation_aware"}:
            return self.relation_norm(id_emb)

        type_token = self.relation_type_token_norm(self.relation_type_emb(self.relation_type_ids[relation_ids]))
        type_active = torch.ones((relation_ids.shape[0],), dtype=torch.bool, device=relation_ids.device)
        strength_token = self.relation_strength_token_norm(
            self.relation_strength_projector(self.relation_strengths[relation_ids])
        )
        strength_active = torch.ones_like(type_active)
        if self.ablation_mode == "no_relation_strength":
            strength_active = torch.zeros_like(strength_active)

        tokens = torch.stack([type_token, strength_token], dim=1)
        mask = torch.stack([type_active, strength_active], dim=1)
        if self.ablation_mode in {"direct_sum_fusion", "hybrid_relation"}:
            extra = (tokens * mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1)
            return self.relation_norm(id_emb + extra)

        fused_extra = self.relation_fusion(tokens, mask)
        residual_scale = torch.sigmoid(self.raw_relation_residual_scale)
        return self.relation_norm(id_emb + residual_scale * fused_extra)

    def conve_transform(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        h_2d = h_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_2d = r_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([h_2d, r_2d], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)
        x = self.conve_transform(h_emb, r_emb)
        score = torch.sum(x * t_emb, dim=1) + self.b[t]
        return torch.sigmoid(score)

    def score_tail_pairs_from_head_embeddings(
        self,
        h_emb: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if h_emb.dim() != 2:
            raise ValueError("h_emb must be a 2D tensor shaped [batch_size, embedding_dim]")
        if h_emb.shape[0] != r.shape[0] or h_emb.shape[0] != t.shape[0]:
            raise ValueError("h_emb, r, and t must contain the same number of samples")
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)
        x = self.conve_transform(h_emb, r_emb)
        score = torch.sum(x * t_emb, dim=1) + self.b[t]
        return torch.sigmoid(score)

    def score_tails(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        tail_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        x = self.conve_transform(h_emb, r_emb)
        if tail_ids is None:
            all_ids = torch.arange(self.nentity, device=h.device)
            tail_emb = self.entity_embedding(all_ids)
            bias = self.b
        else:
            tail_emb = self.entity_embedding(tail_ids)
            bias = self.b[tail_ids]
        scores = torch.mm(x, tail_emb.transpose(1, 0)) + bias.unsqueeze(0)
        return torch.sigmoid(scores)
