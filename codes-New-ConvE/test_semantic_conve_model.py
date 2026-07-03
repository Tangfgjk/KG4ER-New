"""Unit checks for SemanticConvE relation encoding."""

from __future__ import annotations

import unittest

import torch

from feature_loader import RELATION_TYPE_TO_ID
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


if __name__ == "__main__":
    unittest.main()
