"""Semantic and pedagogical feature-aware ConvE model.

This V11.2 variant uses compact type-specific raw feature concatenation:

* learner: concat(ID_200, theta_1) -> MLP -> 200
* exercise: concat(ID_200, text_100, difficulty_1, discrimination_1) -> MLP -> 200
* knowledge concept: ID_200 -> MLP -> 200
* relation: concat(relation_ID_200, relation_type_16, strength_1) -> MLP -> 200

It also supports a progressive ID-removal diagnostic sequence:

* no_learner_id: learner uses theta only;
* no_learner_relation_id: learner uses theta only and relations use type plus strength only;
* feature_only: learner, exercise, and relation IDs are removed while KC IDs remain.

The final 200-dimensional entity/relation representations are then consumed by
the original ConvE scoring module.
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
    "compact_features",
    "no_theta",
    "no_text",
    "no_exercise_ped",
    "no_relation_features",
    "id_only",
    "no_learner_id",
    "no_learner_relation_id",
    "feature_only",
}


class RawConcatFusion(nn.Module):
    """Fuse raw feature vectors by concatenation and an MLP.

    Unlike the older token-mask implementation, this module does not create
    zero placeholder tokens. Each caller passes exactly the features that should
    be used by the current entity/relation type and ablation.
    """

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        hidden_dim = embedding_dim * 2
        self.output = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must be shaped [batch, feature_dim]")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected feature_dim={self.input_dim}, got {features.shape[1]}")
        return self.output_norm(self.output(self.input_norm(features)))


class SemanticConvE(nn.Module):
    """ConvE with type-specific raw concat-MLP feature fusion."""

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

        self.text_dim = max(1, int(text_dim))
        self.learner_irt_width = max(1, self._slice_width("learner_irt"))
        self.exercise_irt_width = max(1, self._slice_width("exercise_irt"))
        self.relation_type_dim = 16

        self.emb_e = nn.Embedding(nentity, embedding_dim)
        self.relation_id_emb = nn.Embedding(nrelation, embedding_dim)
        self.relation_type_emb = nn.Embedding(len(RELATION_TYPE_TO_ID), self.relation_type_dim)

        self.uid_fusion = RawConcatFusion(embedding_dim + self.learner_irt_width, embedding_dim)
        self.uid_id_fusion = RawConcatFusion(embedding_dim, embedding_dim)
        self.uid_feature_only_fusion = RawConcatFusion(self.learner_irt_width, embedding_dim)
        self.exercise_fusion = RawConcatFusion(
            embedding_dim + self.text_dim + self.exercise_irt_width,
            embedding_dim,
        )
        self.exercise_no_text_fusion = RawConcatFusion(embedding_dim + self.exercise_irt_width, embedding_dim)
        self.exercise_no_ped_fusion = RawConcatFusion(embedding_dim + self.text_dim, embedding_dim)
        self.exercise_id_fusion = RawConcatFusion(embedding_dim, embedding_dim)
        self.exercise_feature_only_fusion = RawConcatFusion(
            self.text_dim + self.exercise_irt_width,
            embedding_dim,
        )
        self.kc_fusion = RawConcatFusion(embedding_dim, embedding_dim)
        self.relation_fusion = RawConcatFusion(embedding_dim + self.relation_type_dim + 1, embedding_dim)
        self.relation_feature_only_fusion = RawConcatFusion(self.relation_type_dim + 1, embedding_dim)
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
        xavier_normal_(self.relation_type_emb.weight.data)

    def _slice_width(self, name: str) -> int:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        return max(0, int(end - start))

    def _slice_numeric(self, entity_ids: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        if end <= start:
            return torch.zeros((entity_ids.numel(), 1), dtype=self.numeric_features.dtype, device=entity_ids.device)
        return self.numeric_features[entity_ids, start:end]

    def _entity_text(self, entity_ids: torch.Tensor) -> torch.Tensor:
        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        if text_features.shape[1] == 0:
            return torch.zeros((entity_ids.numel(), self.text_dim), dtype=self.emb_e.weight.dtype, device=entity_ids.device)
        if text_features.shape[1] != self.text_dim:
            raise ValueError(f"expected text_dim={self.text_dim}, got {text_features.shape[1]}")
        return text_features

    def _uses_theta(self) -> bool:
        return self.ablation_mode not in {
            "id_only",
            "no_theta",
            "no_content_entity",
            "no_pedagogical",
            "no_irt",
            "stat_only_ped",
            "no_learner_irt",
        }

    def _uses_exercise_text(self) -> bool:
        return self.ablation_mode not in {
            "id_only",
            "no_text",
            "no_content_entity",
            "no_semantic",
            "no_text_semantic",
            "no_exercise_semantic",
        }

    def _uses_exercise_pedagogy(self) -> bool:
        return self.ablation_mode not in {
            "id_only",
            "no_exercise_ped",
            "no_content_entity",
            "no_pedagogical",
            "no_irt",
            "stat_only_ped",
            "no_exercise_irt",
        }

    def _uses_relation_features(self) -> bool:
        return self.ablation_mode not in {
            "id_only",
            "no_relation_features",
            "no_relation_aware",
            "relation_id_only",
            "discrete_relation",
        }

    def _removes_learner_id(self) -> bool:
        return self.ablation_mode in {
            "no_learner_id",
            "no_learner_relation_id",
            "feature_only",
        }

    def _removes_relation_id(self) -> bool:
        return self.ablation_mode in {
            "no_learner_relation_id",
            "feature_only",
        }

    def _removes_exercise_id(self) -> bool:
        return self.ablation_mode == "feature_only"

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)

        type_ids = self.entity_type_ids[entity_ids]
        fused_emb = torch.empty_like(id_emb)

        uid_mask = type_ids == ENTITY_TYPE_TO_ID["uid"]
        if uid_mask.any():
            uid_ids = entity_ids[uid_mask]
            uid_id_emb = id_emb[uid_mask]
            if self._removes_learner_id():
                theta = self._slice_numeric(uid_ids, "learner_irt")
                fused_emb[uid_mask] = self.uid_feature_only_fusion(theta)
            elif self._uses_theta():
                theta = self._slice_numeric(uid_ids, "learner_irt")
                fused_emb[uid_mask] = self.uid_fusion(torch.cat([uid_id_emb, theta], dim=1))
            else:
                fused_emb[uid_mask] = self.uid_id_fusion(uid_id_emb)

        ex_mask = type_ids == ENTITY_TYPE_TO_ID["ex"]
        if ex_mask.any():
            ex_ids = entity_ids[ex_mask]
            ex_id_emb = id_emb[ex_mask]
            use_text = self._uses_exercise_text()
            use_ped = self._uses_exercise_pedagogy()
            if self._removes_exercise_id():
                if not (use_text and use_ped):
                    raise ValueError("feature_only requires exercise text and MIRT pedagogical features")
                fused_emb[ex_mask] = self.exercise_feature_only_fusion(
                    torch.cat([self._entity_text(ex_ids), self._slice_numeric(ex_ids, "exercise_irt")], dim=1)
                )
            elif use_text and use_ped:
                fused_emb[ex_mask] = self.exercise_fusion(
                    torch.cat([ex_id_emb, self._entity_text(ex_ids), self._slice_numeric(ex_ids, "exercise_irt")], dim=1)
                )
            elif use_text:
                fused_emb[ex_mask] = self.exercise_no_ped_fusion(torch.cat([ex_id_emb, self._entity_text(ex_ids)], dim=1))
            elif use_ped:
                fused_emb[ex_mask] = self.exercise_no_text_fusion(
                    torch.cat([ex_id_emb, self._slice_numeric(ex_ids, "exercise_irt")], dim=1)
                )
            else:
                fused_emb[ex_mask] = self.exercise_id_fusion(ex_id_emb)

        kc_mask = type_ids == ENTITY_TYPE_TO_ID["kc"]
        if kc_mask.any():
            fused_emb[kc_mask] = self.kc_fusion(id_emb[kc_mask])

        other_mask = ~(uid_mask | ex_mask | kc_mask)
        if other_mask.any():
            fused_emb[other_mask] = id_emb[other_mask]

        return self.entity_norm(fused_emb)

    def gate_values(self) -> dict[str, object]:
        uid_features = ["theta_mirt_norm_1"] if self._removes_learner_id() else ["entity_id_200", "theta_mirt_norm_1"]
        exercise_features = (
            ["topic_v/text_100", "difficulty_mirt_norm_1", "discrimination_mirt_norm_1"]
            if self._removes_exercise_id()
            else ["entity_id_200", "topic_v/text_100", "difficulty_mirt_norm_1", "discrimination_mirt_norm_1"]
        )
        relation_features = (
            ["relation_type_16", "relation_strength_1"]
            if self._removes_relation_id()
            else ["relation_id_200", "relation_type_16", "relation_strength_1"]
        )
        return {
            "fusion": "type-specific raw feature concatenation + MLP compression",
            "entity_features": {
                "uid": uid_features,
                "ex": exercise_features,
                "kc": ["entity_id_200"],
            },
            "relation_features": relation_features,
            "ablation": self.ablation_mode,
        }

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if not self._uses_relation_features():
            return self.relation_norm(id_emb)

        type_emb = self.relation_type_emb(self.relation_type_ids[relation_ids])
        strength = self.relation_strengths[relation_ids]
        if self.ablation_mode == "no_relation_strength":
            strength = torch.zeros_like(strength)
        if self._removes_relation_id():
            relation_emb = self.relation_feature_only_fusion(torch.cat([type_emb, strength], dim=1))
        else:
            relation_emb = self.relation_fusion(torch.cat([id_emb, type_emb, strength], dim=1))
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
