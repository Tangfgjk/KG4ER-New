"""Unit checks for SemanticConvE relation encoding."""

from __future__ import annotations

import unittest

import torch

from feature_loader import ENTITY_TYPE_TO_ID, RELATION_TYPE_TO_ID
from semantic_conve_model import SemanticConvE


class SemanticConvERelationEncodingTest(unittest.TestCase):
    def build_model(self) -> SemanticConvE:
        torch.manual_seed(2024)
        nentity = 3
        nrelation = 3
        text_dim = 4
        numeric_dim = 2
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
            entity_type_ids=torch.zeros(nentity, dtype=torch.long),
            cluster_ids=torch.zeros(nentity, dtype=torch.long),
            text_features=torch.zeros((nentity, text_dim), dtype=torch.float32),
            numeric_features=torch.zeros((nentity, numeric_dim), dtype=torch.float32),
            semantic_quality=torch.ones((nentity, 1), dtype=torch.float32),
            embedding_dim=20,
            embedding_shape1=4,
            hidden_size=144,
        )

    def test_relation_embedding_depends_on_type_and_strength_not_relation_id(self) -> None:
        model = self.build_model()
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertTrue(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "same relation type and same continuous strength should share one representation",
        )
        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[2], atol=1e-6),
            "different continuous strengths should produce different relation representations",
        )

    def test_no_relation_strength_ignores_continuous_strength(self) -> None:
        model = self.build_model()
        model.ablation_mode = "no_relation_strength"
        relation_ids = torch.tensor([0, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertTrue(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "no_relation_strength should keep relation type but ignore continuous strength",
        )

    def test_no_semantic_ignores_text_features(self) -> None:
        torch.manual_seed(2024)
        model_a = self.build_model()
        model_a.ablation_mode = "no_semantic"
        torch.manual_seed(2024)
        model_b = self.build_model()
        model_b.ablation_mode = "no_semantic"
        model_b.text_features.fill_(10.0)

        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(
            torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids), atol=1e-6),
            "no_semantic should make entity embeddings independent from text features",
        )

    def test_gate_values_are_type_level_and_start_at_half(self) -> None:
        model = self.build_model()

        gate_values = model.gate_values()

        self.assertEqual(set(gate_values), set(ENTITY_TYPE_TO_ID))
        for values in gate_values.values():
            self.assertAlmostEqual(values["semantic"], 0.5, places=6)
            self.assertAlmostEqual(values["pedagogical"], 0.5, places=6)
            self.assertAlmostEqual(values["cluster"], 0.5, places=6)

    def test_semantic_quality_scales_semantic_contribution(self) -> None:
        model = self.build_model()
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.cluster_emb.weight.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.text_projector.weight.zero_()
            model.text_projector.weight[0, 0] = 1.0
            model.text_projector.weight[1, 1] = 2.0
            model.text_features[0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
            model.text_features[1] = torch.tensor([1.0, 1.0, 0.0, 0.0])
            model.semantic_quality[0, 0] = 0.0
            model.semantic_quality[1, 0] = 1.0

        embeddings = model.entity_embedding(torch.tensor([0, 1], dtype=torch.long))

        self.assertFalse(
            torch.allclose(embeddings[0], embeddings[1], atol=1e-6),
            "semantic_quality should reduce or preserve the semantic contribution per entity",
        )


if __name__ == "__main__":
    unittest.main()
