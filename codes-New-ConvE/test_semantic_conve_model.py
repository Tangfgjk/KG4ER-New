"""Unit checks for SemanticConvE V7 attention fusion."""

from __future__ import annotations

import unittest

import torch

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_conve_model import FEATURE_RESIDUAL_INITIAL_VALUE, SemanticConvE


class SemanticConvEV7Test(unittest.TestCase):
    def build_model(
        self,
        entity_type_ids: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
        semantic_quality: torch.Tensor | None = None,
        ablation_mode: str = "full",
    ) -> SemanticConvE:
        torch.manual_seed(2024)
        nentity = 3
        nrelation = 3
        text_dim = 4
        numeric_dim = 2
        if entity_type_ids is None:
            entity_type_ids = torch.tensor(
                [ENTITY_TYPE_TO_ID["uid"], ENTITY_TYPE_TO_ID["kc"], ENTITY_TYPE_TO_ID["ex"]],
                dtype=torch.long,
            )
        if cluster_ids is None:
            cluster_ids = torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long)
        if numeric_features is None:
            numeric_features = torch.zeros((nentity, numeric_dim), dtype=torch.float32)
        if text_features is None:
            text_features = torch.zeros((nentity, text_dim), dtype=torch.float32)
        if semantic_quality is None:
            semantic_quality = torch.ones((nentity, 1), dtype=torch.float32)
        return SemanticConvE(
            nentity=nentity,
            nrelation=nrelation,
            text_dim=text_dim,
            numeric_dim=numeric_dim,
            relation_type_ids=torch.tensor(
                [
                    RELATION_TYPE_TO_ID["mlkc"],
                    RELATION_TYPE_TO_ID["mlkc"],
                    RELATION_TYPE_TO_ID["mlkc"],
                ],
                dtype=torch.long,
            ),
            relation_strengths=torch.tensor([[0.50], [0.50], [0.80]], dtype=torch.float32),
            entity_type_ids=entity_type_ids,
            cluster_ids=cluster_ids,
            text_features=text_features,
            numeric_features=numeric_features,
            semantic_quality=semantic_quality,
            embedding_dim=20,
            embedding_shape1=4,
            hidden_size=576,
            ablation_mode=ablation_mode,
        )

    def clone_model(self, **kwargs) -> tuple[SemanticConvE, SemanticConvE]:
        model_a = self.build_model(**kwargs)
        torch.manual_seed(2024)
        model_b = self.build_model(**kwargs)
        return model_a, model_b

    def test_full_entity_embedding_uses_concept_text(self) -> None:
        model_a, model_b = self.clone_model()
        with torch.no_grad():
            model_b.text_features[1, 0] = 5.0

        entity_ids = torch.tensor([1], dtype=torch.long)

        self.assertFalse(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_no_concept_text_masks_concept_text(self) -> None:
        model_a, model_b = self.clone_model(ablation_mode="no_concept_text")
        with torch.no_grad():
            model_b.text_features[1, 0] = 5.0

        entity_ids = torch.tensor([1], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_no_exercise_text_masks_exercise_text(self) -> None:
        model_a, model_b = self.clone_model(ablation_mode="no_exercise_text")
        with torch.no_grad():
            model_b.text_features[2, 0] = 5.0

        entity_ids = torch.tensor([2], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_no_text_semantic_masks_all_text(self) -> None:
        model_a, model_b = self.clone_model(ablation_mode="no_text_semantic")
        with torch.no_grad():
            model_b.text_features.fill_(8.0)

        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_no_pedagogical_masks_numeric_and_cluster(self) -> None:
        model_a, model_b = self.clone_model(ablation_mode="no_pedagogical")
        with torch.no_grad():
            model_b.numeric_features.fill_(7.0)
            model_b.cluster_ids.fill_(1)

        entity_ids = torch.tensor([0, 2], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_id_only_returns_normalized_id_embedding(self) -> None:
        model = self.build_model(ablation_mode="id_only")
        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        with torch.no_grad():
            expected = model.entity_norm(model.emb_e(entity_ids))

        self.assertTrue(torch.allclose(model.entity_embedding(entity_ids), expected, atol=1e-6))

    def test_direct_sum_fusion_differs_from_attention_residual(self) -> None:
        model_a = self.build_model(ablation_mode="full")
        torch.manual_seed(2024)
        model_b = self.build_model(ablation_mode="direct_sum_fusion")
        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertFalse(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids)))

    def test_relation_embedding_keeps_id_anchor_and_strength(self) -> None:
        model = self.build_model()
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertFalse(torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6))
        self.assertFalse(torch.allclose(relation_embeddings[0], relation_embeddings[2], atol=1e-6))

    def test_no_relation_aware_uses_relation_id_only(self) -> None:
        model = self.build_model(ablation_mode="no_relation_aware")
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        with torch.no_grad():
            expected = model.relation_norm(model.relation_id_emb(relation_ids))

        self.assertTrue(torch.allclose(model.relation_embedding(relation_ids), expected, atol=1e-6))

    def test_residual_scales_start_near_five_percent(self) -> None:
        model = self.build_model()
        values = model.gate_values()

        self.assertAlmostEqual(values["uid"]["feature_residual_scale"], FEATURE_RESIDUAL_INITIAL_VALUE, places=6)
        self.assertAlmostEqual(values["relation"]["feature_residual_scale"], FEATURE_RESIDUAL_INITIAL_VALUE, places=6)

    def test_can_score_tail_pairs_from_external_head_embeddings(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [ENTITY_TYPE_TO_ID["ex"], ENTITY_TYPE_TO_ID["ex"], ENTITY_TYPE_TO_ID["ex"]],
                dtype=torch.long,
            )
        )
        model.eval()
        tail_ids = torch.tensor([1, 2], dtype=torch.long)
        relation_ids = torch.tensor([0, 0], dtype=torch.long)
        base_head = model.entity_embedding(torch.tensor([0, 0], dtype=torch.long))
        shifted_head = base_head + 0.25

        base_scores = model.score_tail_pairs_from_head_embeddings(base_head, relation_ids, tail_ids)
        shifted_scores = model.score_tail_pairs_from_head_embeddings(shifted_head, relation_ids, tail_ids)

        self.assertEqual(tuple(base_scores.shape), (2,))
        self.assertFalse(torch.allclose(base_scores, shifted_scores, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
