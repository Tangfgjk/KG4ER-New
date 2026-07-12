"""Semantic and pedagogical feature-aware ConvE model.

V11.1 treats the ID embedding as an explicit attention token together with
semantic and pedagogical feature tokens. The attended feature set is compressed
back to the ConvE embedding dimension, so the model can learn whether ID,
semantic content, or educational attributes should dominate each representation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID, SemanticFeatureBundle


VALID_MODEL_ABLATIONS = {
    "full",
    "full_state_hybrid",
    "irt_only_ped",
    "stat_only_ped",
    "no_irt",
    "no_stat_ped",
    "no_mastery",
    "no_forgetting",
    "no_seq",
    "id_head_reference",
    "no_content_entity",
    "no_relation_aware",
    "no_type_aware_scoring",
    "no_semantic",
    "no_text_semantic",
    "no_concept_semantic",
    "no_exercise_semantic",
    "no_pedagogical",
    "no_exercise_irt",
    "no_learner_irt",
    "no_cluster",
    "no_relation_strength",
    "discrete_relation",
    "hybrid_relation",
    "relation_id_only",
    "id_only",
}

def _numeric_projector(input_dim: int, embedding_dim: int) -> nn.Module:
    width = max(1, int(input_dim))
    return nn.Sequential(
        nn.Linear(width, embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim, embedding_dim),
        nn.LayerNorm(embedding_dim),
    )


class FeatureAttentionFusion(nn.Module):
    """Fuse a variable set of projected feature tokens.

    Each feature source is first projected into the ConvE embedding dimension.
    We then run a small self-attention block over the active tokens and mean-pool
    the attended tokens. Rows without any active token return zeros.
    """

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        heads = num_heads if embedding_dim % num_heads == 0 else 1
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
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
            raise ValueError("tokens must be shaped [batch, feature_count, embedding_dim]")
        if active_mask.dim() != 2:
            raise ValueError("active_mask must be shaped [batch, feature_count]")
        active = active_mask.bool()
        output = torch.zeros(tokens.shape[0], tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
        valid_rows = active.any(dim=1)
        if not valid_rows.any():
            return output
        valid_tokens = self.token_norm(tokens[valid_rows])
        valid_mask = active[valid_rows]
        attended, _ = self.attn(
            valid_tokens,
            valid_tokens,
            valid_tokens,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (attended * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        output[valid_rows] = self.output_norm(self.output(pooled))
        return output


class SemanticConvE(nn.Module):
    """ConvE with ID-token attention-based multi-source feature fusion."""

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
        state_features: Optional[torch.Tensor] = None,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        hidden_size: int = 9728,
        input_drop: float = 0.2,
        hidden_drop: float = 0.2,
        feat_drop: float = 0.3,
        use_bias: bool = True,
        freeze_text_features: bool = True,
        ablation_mode: str = "full",
        numeric_feature_slices: Optional[dict[str, tuple[int, int]]] = None,
        state_feature_slices: Optional[dict[str, tuple[int, int]]] = None,
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
        self.freeze_text_features = freeze_text_features
        self.ablation_mode = ablation_mode
        self.numeric_feature_slices = numeric_feature_slices or {"learner_irt": (0, 0), "exercise_irt": (0, 0)}
        self.state_feature_slices = state_feature_slices or {}

        learner_irt_width = self._slice_width("learner_irt")
        exercise_irt_width = self._slice_width("exercise_irt")

        self.emb_e = nn.Embedding(nentity, embedding_dim)
        self.relation_id_emb = nn.Embedding(nrelation, embedding_dim)
        self.cluster_emb = nn.Embedding(NO_CLUSTER_ID + 1, embedding_dim)
        self.relation_type_emb = nn.Embedding(len(RELATION_TYPE_TO_ID), embedding_dim)

        self.text_projector = nn.Sequential(
            nn.Linear(max(1, text_dim), embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )
        self.learner_irt_projector = _numeric_projector(learner_irt_width, embedding_dim)
        self.exercise_irt_projector = _numeric_projector(exercise_irt_width, embedding_dim)
        self.relation_strength_projector = _numeric_projector(1, embedding_dim)
        self.entity_fusion = FeatureAttentionFusion(embedding_dim)
        self.relation_fusion = FeatureAttentionFusion(embedding_dim)
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
        if state_features is None:
            state_features = torch.zeros((nentity, 1), dtype=torch.float32, device=numeric_features.device)
        self.register_buffer("state_features", state_features.detach().clone())
        self.register_buffer("entity_type_ids", entity_type_ids.detach().clone())
        self.register_buffer("cluster_ids", cluster_ids.detach().clone())
        self.register_buffer("semantic_quality", semantic_quality.detach().clone())
        self.register_buffer("relation_type_ids", relation_type_ids.detach().clone())
        self.register_buffer("relation_strengths", relation_strengths.detach().clone())
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
            state_features=bundle.state_features,
            semantic_quality=bundle.semantic_quality,
            numeric_feature_slices=bundle.numeric_feature_slices,
            state_feature_slices=bundle.state_feature_slices,
            **kwargs,
        )

    def init(self) -> None:
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.relation_id_emb.weight.data)
        xavier_normal_(self.cluster_emb.weight.data)
        xavier_normal_(self.relation_type_emb.weight.data)

    def _slice_width(self, name: str) -> int:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        return max(0, int(end - start))

    def _slice_numeric(self, entity_ids: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        if end <= start:
            return torch.zeros((entity_ids.numel(), 1), dtype=self.numeric_features.dtype, device=entity_ids.device)
        return self.numeric_features[entity_ids, start:end]

    def _semantic_active(self, type_ids: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
        active = (
            (type_ids == ENTITY_TYPE_TO_ID["kc"]) | (type_ids == ENTITY_TYPE_TO_ID["ex"])
        ) & (self.semantic_quality[entity_ids, 0] > 0)
        if self.ablation_mode in {"id_only", "no_content_entity", "no_semantic", "no_text_semantic"}:
            active = torch.zeros_like(active)
        elif self.ablation_mode == "no_concept_semantic":
            active = active & (type_ids != ENTITY_TYPE_TO_ID["kc"])
        elif self.ablation_mode == "no_exercise_semantic":
            active = active & (type_ids != ENTITY_TYPE_TO_ID["ex"])
        return active

    def _learner_irt_active(self, type_ids: torch.Tensor) -> torch.Tensor:
        active = type_ids == ENTITY_TYPE_TO_ID["uid"]
        if self.ablation_mode in {
            "id_only",
            "no_content_entity",
            "no_pedagogical",
            "no_irt",
            "stat_only_ped",
            "no_learner_irt",
        }:
            active = torch.zeros_like(active)
        return active

    def _exercise_irt_active(self, type_ids: torch.Tensor) -> torch.Tensor:
        active = type_ids == ENTITY_TYPE_TO_ID["ex"]
        if self.ablation_mode in {
            "id_only",
            "no_content_entity",
            "no_pedagogical",
            "no_irt",
            "stat_only_ped",
            "no_exercise_irt",
        }:
            active = torch.zeros_like(active)
        return active

    def _cluster_active(self, type_ids: torch.Tensor) -> torch.Tensor:
        active = type_ids == ENTITY_TYPE_TO_ID["uid"]
        if self.ablation_mode in {"id_only", "no_content_entity", "no_pedagogical", "no_cluster"}:
            active = torch.zeros_like(active)
        return active

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)

        type_ids = self.entity_type_ids[entity_ids]
        token_list: list[torch.Tensor] = [id_emb]
        mask_list: list[torch.Tensor] = [torch.ones_like(entity_ids, dtype=torch.bool)]

        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        if text_features.shape[1] == 0:
            text_features = torch.zeros((entity_ids.numel(), 1), dtype=id_emb.dtype, device=entity_ids.device)
        token_list.append(self.text_projector(text_features))
        mask_list.append(self._semantic_active(type_ids, entity_ids))

        learner_irt = self._slice_numeric(entity_ids, "learner_irt")
        token_list.append(self.learner_irt_projector(learner_irt))
        mask_list.append(self._learner_irt_active(type_ids))

        exercise_irt = self._slice_numeric(entity_ids, "exercise_irt")
        token_list.append(self.exercise_irt_projector(exercise_irt))
        mask_list.append(self._exercise_irt_active(type_ids))

        cluster_ids = self.cluster_ids[entity_ids].clamp(min=0, max=NO_CLUSTER_ID)
        token_list.append(self.cluster_emb(cluster_ids))
        mask_list.append(self._cluster_active(type_ids))

        tokens = torch.stack(token_list, dim=1)
        active_mask = torch.stack(mask_list, dim=1)
        fused_emb = self.entity_fusion(tokens, active_mask)
        return self.entity_norm(fused_emb)

    def gate_values(self) -> dict[str, object]:
        return {
            "fusion": "ID token + feature-token self-attention + MLP compression",
            "entity_features": {
                "uid": ["entity_id", "theta_mirt_norm", "cluster_id"],
                "ex": ["entity_id", "topic_v/text", "difficulty_mirt_norm", "discrimination_mirt_norm"],
                "kc": ["entity_id", "concept_semantic"],
            },
            "relation_features": ["relation_id", "relation_type", "relation_strength"],
            "ablation": self.ablation_mode,
        }

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if self.ablation_mode in {"id_only", "no_relation_aware", "relation_id_only", "discrete_relation"}:
            return self.relation_norm(id_emb)

        type_token = self.relation_type_emb(self.relation_type_ids[relation_ids])
        strength_token = self.relation_strength_projector(self.relation_strengths[relation_ids])
        type_active = torch.ones_like(relation_ids, dtype=torch.bool)
        strength_active = torch.ones_like(relation_ids, dtype=torch.bool)
        if self.ablation_mode == "no_relation_strength":
            strength_active = torch.zeros_like(strength_active)
        id_active = torch.ones_like(relation_ids, dtype=torch.bool)
        tokens = torch.stack([id_emb, type_token, strength_token], dim=1)
        active_mask = torch.stack([id_active, type_active, strength_active], dim=1)
        relation_emb = self.relation_fusion(tokens, active_mask)
        return self.relation_norm(relation_emb)

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
