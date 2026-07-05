"""Semantic and pedagogical feature-aware ConvE model."""

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
    "no_mastery",
    "no_forgetting",
    "no_seq",
    "no_semantic",
    "no_concept_semantic",
    "no_exercise_semantic",
    "no_pedagogical",
    "no_exercise_irt",
    "no_learner_irt",
    "no_cluster",
    "no_relation_strength",
    "discrete_relation",
    "hybrid_relation",
    "id_only",
}

GATE_INITIAL_VALUE = 0.1


def _gate_logit(value: float = GATE_INITIAL_VALUE) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class SemanticConvE(nn.Module):
    """ConvE with semantic, pedagogical and relation-aware representations.

    Entity representation:
        ID embedding + projected text embedding + projected numeric features
        + entity type embedding + learner cluster embedding, followed by LayerNorm.

    Relation representation:
        relation type embedding + projected continuous relation strength, followed
        by LayerNorm in the default setting. Fine-grained ablations can switch
        to discrete relation-id embeddings or a hybrid relation representation.
    """

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
        gate_init = torch.full((len(ENTITY_TYPE_TO_ID),), _gate_logit(), dtype=torch.float32)
        self.raw_semantic_gate = nn.Parameter(gate_init.clone())
        self.raw_pedagogical_gate = nn.Parameter(gate_init.clone())
        self.raw_cluster_gate = nn.Parameter(gate_init.clone())

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

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)
        type_ids = self.entity_type_ids[entity_ids]

        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        if self.ablation_mode in {"no_semantic", "id_only"}:
            semantic_emb = torch.zeros_like(id_emb)
        else:
            semantic_emb = self.text_projector(text_features)
            semantic_gate = torch.sigmoid(self.raw_semantic_gate[type_ids]).unsqueeze(-1)
            semantic_mask = torch.ones_like(semantic_gate)
            if self.ablation_mode == "no_concept_semantic":
                semantic_mask = (type_ids != ENTITY_TYPE_TO_ID["kc"]).float().unsqueeze(-1)
            elif self.ablation_mode == "no_exercise_semantic":
                semantic_mask = (type_ids != ENTITY_TYPE_TO_ID["ex"]).float().unsqueeze(-1)
            semantic_emb = semantic_mask * semantic_gate * self.semantic_quality[entity_ids] * semantic_emb

        if self.ablation_mode in {"no_pedagogical", "id_only"}:
            pedagogical_emb = torch.zeros_like(id_emb)
        else:
            pedagogical_emb = self.numeric_projector(self.numeric_features[entity_ids])
            pedagogical_gate = torch.sigmoid(self.raw_pedagogical_gate[type_ids]).unsqueeze(-1)
            pedagogical_mask = (
                (type_ids == ENTITY_TYPE_TO_ID["uid"]) | (type_ids == ENTITY_TYPE_TO_ID["ex"])
            ).float().unsqueeze(-1)
            if self.ablation_mode == "no_exercise_irt":
                pedagogical_mask = pedagogical_mask * (type_ids != ENTITY_TYPE_TO_ID["ex"]).float().unsqueeze(-1)
            elif self.ablation_mode == "no_learner_irt":
                pedagogical_mask = pedagogical_mask * (type_ids != ENTITY_TYPE_TO_ID["uid"]).float().unsqueeze(-1)
            pedagogical_emb = pedagogical_mask * pedagogical_gate * pedagogical_emb

        type_emb = self.entity_type_emb(type_ids)
        if self.ablation_mode in {"no_pedagogical", "no_cluster", "id_only"}:
            cluster_emb = torch.zeros_like(id_emb)
        else:
            cluster_emb = self.cluster_emb(self.cluster_ids[entity_ids].clamp(min=0, max=NO_CLUSTER_ID))
            cluster_gate = torch.sigmoid(self.raw_cluster_gate[type_ids]).unsqueeze(-1)
            cluster_mask = (type_ids == ENTITY_TYPE_TO_ID["uid"]).float().unsqueeze(-1)
            cluster_emb = cluster_mask * cluster_gate * cluster_emb
        return self.entity_norm(id_emb + semantic_emb + pedagogical_emb + type_emb + cluster_emb)

    def gate_values(self) -> dict[str, dict[str, float]]:
        names_by_id = {idx: name for name, idx in ENTITY_TYPE_TO_ID.items()}
        semantic = torch.sigmoid(self.raw_semantic_gate).detach().cpu().tolist()
        pedagogical = torch.sigmoid(self.raw_pedagogical_gate).detach().cpu().tolist()
        cluster = torch.sigmoid(self.raw_cluster_gate).detach().cpu().tolist()
        return {
            names_by_id[idx]: {
                "semantic": float(semantic[idx]),
                "pedagogical": float(pedagogical[idx]),
                "cluster": float(cluster[idx]),
            }
            for idx in sorted(names_by_id)
        }

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if self.ablation_mode in {"discrete_relation", "id_only"}:
            return self.relation_norm(id_emb)

        type_emb = self.relation_type_emb(self.relation_type_ids[relation_ids])
        if self.ablation_mode in {"no_relation_strength", "id_only"}:
            strength_emb = torch.zeros_like(type_emb)
        else:
            strength_emb = self.relation_strength_projector(self.relation_strengths[relation_ids])
        if self.ablation_mode == "hybrid_relation":
            return self.relation_norm(id_emb + type_emb + strength_emb)
        return self.relation_norm(type_emb + strength_emb)

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
