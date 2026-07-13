"""Unit checks for compact raw-concat SemanticConvE representations."""

from __future__ import annotations

import unittest

import torch

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_conve_model import SemanticConvE


class SemanticConvECompactConcatTest(unittest.TestCase):
    def build_model(
        self,
        entity_type_ids: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
    ) -> SemanticConvE:
        torch.manual_seed(2024)
        nentity = 3
        nrelation = 3
        text_dim = 4
        numeric_dim = 3
        if entity_type_ids is None:
            entity_type_ids = torch.tensor(
                [ENTITY_TYPE_TO_ID["uid"], ENTITY_TYPE_TO_ID["kc"], ENTITY_TYPE_TO_ID["ex"]],
                dtype=torch.long,
            )
        if cluster_ids is None:
            cluster_ids = torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long)
        if numeric_features is None:
            numeric_features = torch.zeros((nentity, numeric_dim), dtype=torch.float32)
        return SemanticConvE(
            nentity=nentity,
            nrelation=nrelation,
            text_dim=text_dim,
            numeric_dim=numeric_dim,
            relation_type_ids=torch.tensor(
                [RELATION_TYPE_TO_ID["mlkc"], RELATION_TYPE_TO_ID["mlkc"], RELATION_TYPE_TO_ID["pkc"]],
                dtype=torch.long,
            ),
            relation_strengths=torch.tensor([[0.50], [0.80], [0.50]], dtype=torch.float32),
            entity_type_ids=entity_type_ids,
            cluster_ids=cluster_ids,
            text_features=torch.zeros((nentity, text_dim), dtype=torch.float32),
            numeric_features=numeric_features,
            semantic_quality=torch.ones((nentity, 1), dtype=torch.float32),
            embedding_dim=40,
            id_embedding_dim=20,
            embedding_shape1=8,
            hidden_size=1344,
            numeric_feature_slices={"learner_irt": (0, 1), "exercise_irt": (1, 3)},
        )

    def test_full_relation_keeps_relation_id_capacity(self) -> None:
        model = self.build_model()
        relation_ids = torch.tensor([0, 1], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "Full relation representation keeps relation ID information while adding type/strength features",
        )

    def test_no_relation_aware_uses_relation_id_only(self) -> None:
        model = self.build_model()
        model.ablation_mode = "no_relation_aware"
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        with torch.no_grad():
            expected = model.relation_norm(model.relation_id_fusion(model.relation_id_emb(relation_ids)))
        self.assertTrue(torch.allclose(relation_embeddings, expected, atol=1e-6))

    def test_id_only_entity_is_independent_from_side_features(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "id_only"
        model_b.ablation_mode = "id_only"
        with torch.no_grad():
            model_b.text_features.fill_(9.0)
            model_b.numeric_features.fill_(7.0)
            model_b.cluster_ids.fill_(1)

        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids), atol=1e-6))

    def test_no_theta_masks_learner_theta(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "no_theta"
        model_b.ablation_mode = "no_theta"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.numeric_features[0, 0] = 1.0

        embeddings_a = model_a.entity_embedding(torch.tensor([0], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([0], dtype=torch.long))

        self.assertTrue(torch.allclose(embeddings_a, embeddings_b, atol=1e-6))

    def test_no_text_masks_exercise_text(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "no_text"
        model_b.ablation_mode = "no_text"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.text_features[2, 0] = 1.0

        embeddings_a = model_a.entity_embedding(torch.tensor([2], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([2], dtype=torch.long))

        self.assertTrue(torch.allclose(embeddings_a, embeddings_b, atol=1e-6))

    def test_no_exercise_ped_masks_exercise_irt(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "no_exercise_ped"
        model_b.ablation_mode = "no_exercise_ped"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.numeric_features[2, 1:] = torch.tensor([0.3, 0.9])

        embeddings_a = model_a.entity_embedding(torch.tensor([2], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([2], dtype=torch.long))

        self.assertTrue(torch.allclose(embeddings_a, embeddings_b, atol=1e-6))

    def test_full_exercise_uses_text_and_pedagogical_features(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.text_features[2, 0] = 1.0
            model_b.numeric_features[2, 1:] = torch.tensor([0.3, 0.9])

        embeddings_a = model_a.entity_embedding(torch.tensor([2], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([2], dtype=torch.long))

        self.assertFalse(torch.allclose(embeddings_a, embeddings_b, atol=1e-6))

    def test_gate_values_report_raw_concat_padding_fusion(self) -> None:
        model = self.build_model()

        values = model.gate_values()

        self.assertEqual(values["fusion"], "type-specific raw feature concatenation + zero padding to ConvE dimension")
        self.assertIn("entity_id_200", values["entity_features"]["uid"])
        self.assertIn("relation_id_200", values["relation_features"])

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
