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

GATE_INITIAL_VALUE = 0.05


def _gate_logit(value: float = GATE_INITIAL_VALUE) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class SemanticConvE(nn.Module):
    """ConvE with semantic, pedagogical and relation-aware representations.

    Entity representation:
        ID embedding is the anchor. Projected text, numeric, type, and cluster
        features are gated residual supplements, followed by LayerNorm.

    Relation representation:
        relation ID embedding is the anchor. Relation type and continuous
        relation strength are gated residual supplements, followed by LayerNorm.
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

        self.emb_e = nn.Embedding(nentity, embedding_dim)
        self.entity_type_emb = nn.Embedding(4, embedding_dim)
        self.cluster_emb = nn.Embedding(NO_CLUSTER_ID + 1, embedding_dim)
        self.relation_id_emb = nn.Embedding(nrelation, embedding_dim)
        self.relation_type_emb = nn.Embedding(len(RELATION_TYPE_TO_ID), embedding_dim)
        gate_init = torch.full((len(ENTITY_TYPE_TO_ID),), _gate_logit(), dtype=torch.float32)
        self.raw_semantic_gate = nn.Parameter(gate_init.clone())
        self.raw_pedagogical_gate = nn.Parameter(gate_init.clone())
        self.raw_cluster_gate = nn.Parameter(gate_init.clone())
        self.raw_entity_type_gate = nn.Parameter(gate_init.clone())
        self.raw_relation_type_gate = nn.Parameter(torch.tensor(_gate_logit(), dtype=torch.float32))
        self.raw_relation_strength_gate = nn.Parameter(torch.tensor(_gate_logit(), dtype=torch.float32))

        self.text_projector = nn.Linear(text_dim, embedding_dim, bias=False)
        self.numeric_projector = nn.Sequential(
            nn.Linear(numeric_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        state_dim = int(state_features.shape[1]) if state_features is not None else max(1, numeric_dim)
        self.state_projector = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.relation_strength_projector = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.state_norm = nn.LayerNorm(embedding_dim)
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
            state_features = torch.zeros((nentity, state_dim), dtype=torch.float32, device=numeric_features.device)
        self.register_buffer("state_features", state_features.detach().clone())
        self.register_buffer("entity_type_ids", entity_type_ids.detach().clone())
        self.register_buffer("cluster_ids", cluster_ids.detach().clone())
        self.register_buffer("semantic_quality", semantic_quality.detach().clone())
        self.register_buffer("relation_type_ids", relation_type_ids.detach().clone())
        self.register_buffer("relation_strengths", relation_strengths.detach().clone())
        self.freeze_text_features = freeze_text_features
        self.ablation_mode = ablation_mode
        self.numeric_feature_slices = numeric_feature_slices or {"irt": (0, numeric_dim), "stat": (numeric_dim, numeric_dim)}
        self.state_feature_slices = state_feature_slices or {"learner_irt": (0, 0), "learner_stat": (0, 0)}
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
        xavier_normal_(self.entity_type_emb.weight.data)
        xavier_normal_(self.cluster_emb.weight.data)
        xavier_normal_(self.relation_id_emb.weight.data)
        xavier_normal_(self.relation_type_emb.weight.data)

    def _zero_feature_group(
        self,
        features: torch.Tensor,
        slices: dict[str, tuple[int, int]],
        names: set[str],
    ) -> torch.Tensor:
        if not names:
            return features
        output = features.clone()
        for name in names:
            start, end = slices.get(name, (0, 0))
            if end > start:
                output[:, start:end] = 0.0
        return output

    def _numeric_groups_to_zero(self) -> set[str]:
        if self.ablation_mode in {"no_pedagogical", "no_content_entity", "id_only"}:
            return set(self.numeric_feature_slices)
        if self.ablation_mode in {"no_irt", "stat_only_ped"}:
            return {"learner_irt", "exercise_irt", "irt"}
        if self.ablation_mode in {"no_stat_ped", "irt_only_ped"}:
            return {"learner_stat", "exercise_stat", "stat"}
        if self.ablation_mode == "no_exercise_irt":
            return {"exercise_irt"}
        if self.ablation_mode == "no_learner_irt":
            return {"learner_irt"}
        return set()

    def _state_groups_to_zero(self) -> set[str]:
        groups: set[str] = set()
        if self.ablation_mode in {"no_pedagogical", "no_content_entity", "id_only"}:
            groups.update({"learner_irt", "learner_stat", "cluster"})
        if self.ablation_mode in {"no_irt", "stat_only_ped", "no_learner_irt"}:
            groups.add("learner_irt")
        if self.ablation_mode in {"no_stat_ped", "irt_only_ped"}:
            groups.add("learner_stat")
        if self.ablation_mode == "no_cluster":
            groups.add("cluster")
        if self.ablation_mode == "no_mastery":
            groups.add("mastery")
        if self.ablation_mode == "no_seq":
            groups.add("sequence")
        if self.ablation_mode == "no_forgetting":
            groups.add("forgetting")
        return groups

    def state_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        state_features = self.state_features[entity_ids]
        state_features = self._zero_feature_group(state_features, self.state_feature_slices, self._state_groups_to_zero())
        return self.state_norm(self.state_projector(state_features))

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)
        type_ids = self.entity_type_ids[entity_ids]

        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        if self.ablation_mode in {"no_semantic", "no_content_entity", "id_only"}:
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

        if self.ablation_mode in {"no_pedagogical", "no_content_entity", "id_only"}:
            pedagogical_emb = torch.zeros_like(id_emb)
        else:
            numeric_features = self.numeric_features[entity_ids]
            numeric_features = self._zero_feature_group(numeric_features, self.numeric_feature_slices, self._numeric_groups_to_zero())
            pedagogical_emb = self.numeric_projector(numeric_features)
            pedagogical_gate = torch.sigmoid(self.raw_pedagogical_gate[type_ids]).unsqueeze(-1)
            pedagogical_mask = (
                (type_ids == ENTITY_TYPE_TO_ID["uid"]) | (type_ids == ENTITY_TYPE_TO_ID["ex"])
            ).float().unsqueeze(-1)
            if self.ablation_mode == "no_exercise_irt":
                pedagogical_mask = pedagogical_mask * (type_ids != ENTITY_TYPE_TO_ID["ex"]).float().unsqueeze(-1)
            elif self.ablation_mode == "no_learner_irt":
                pedagogical_mask = pedagogical_mask * (type_ids != ENTITY_TYPE_TO_ID["uid"]).float().unsqueeze(-1)
            pedagogical_emb = pedagogical_mask * pedagogical_gate * pedagogical_emb

        type_gate = torch.sigmoid(self.raw_entity_type_gate[type_ids]).unsqueeze(-1)
        type_emb = type_gate * self.entity_type_emb(type_ids)
        if self.ablation_mode in {"no_pedagogical", "no_content_entity", "no_cluster", "id_only"}:
            cluster_emb = torch.zeros_like(id_emb)
        else:
            cluster_emb = self.cluster_emb(self.cluster_ids[entity_ids].clamp(min=0, max=NO_CLUSTER_ID))
            cluster_gate = torch.sigmoid(self.raw_cluster_gate[type_ids]).unsqueeze(-1)
            cluster_mask = (type_ids == ENTITY_TYPE_TO_ID["uid"]).float().unsqueeze(-1)
            cluster_emb = cluster_mask * cluster_gate * cluster_emb
        entity_emb = self.entity_norm(id_emb + semantic_emb + pedagogical_emb + type_emb + cluster_emb)
        if self.ablation_mode == "id_head_reference":
            return entity_emb
        learner_mask = (type_ids == ENTITY_TYPE_TO_ID["uid"]).unsqueeze(-1)
        if learner_mask.any():
            state_emb = self.state_embedding(entity_ids)
            entity_emb = torch.where(learner_mask, state_emb, entity_emb)
        return entity_emb

    def gate_values(self) -> dict[str, dict[str, float]]:
        names_by_id = {idx: name for name, idx in ENTITY_TYPE_TO_ID.items()}
        semantic = torch.sigmoid(self.raw_semantic_gate).detach().cpu().tolist()
        pedagogical = torch.sigmoid(self.raw_pedagogical_gate).detach().cpu().tolist()
        cluster = torch.sigmoid(self.raw_cluster_gate).detach().cpu().tolist()
        entity_gates = {
            names_by_id[idx]: {
                "semantic": float(semantic[idx]),
                "pedagogical": float(pedagogical[idx]),
                "cluster": float(cluster[idx]),
                "type": float(torch.sigmoid(self.raw_entity_type_gate[idx]).detach().cpu().item()),
            }
            for idx in sorted(names_by_id)
        }
        entity_gates["relation"] = {
            "type": float(torch.sigmoid(self.raw_relation_type_gate).detach().cpu().item()),
            "strength": float(torch.sigmoid(self.raw_relation_strength_gate).detach().cpu().item()),
            "id": 0.0 if self.ablation_mode not in {"discrete_relation", "hybrid_relation", "id_only", "no_relation_aware", "relation_id_only"} else 1.0,
        }
        return entity_gates

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if self.ablation_mode in {"discrete_relation", "id_only", "no_relation_aware", "relation_id_only"}:
            return self.relation_norm(id_emb)

        type_gate = torch.sigmoid(self.raw_relation_type_gate)
        type_emb = type_gate * self.relation_type_emb(self.relation_type_ids[relation_ids])
        if self.ablation_mode in {"no_relation_strength", "id_only"}:
            strength_emb = torch.zeros_like(type_emb)
        else:
            strength_gate = torch.sigmoid(self.raw_relation_strength_gate)
            strength_emb = strength_gate * self.relation_strength_projector(self.relation_strengths[relation_ids])
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
