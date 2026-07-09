"""Unit checks for SemanticConvE relation encoding."""

from __future__ import annotations

import unittest

import torch

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_conve_model import SemanticConvE


class SemanticConvERelationEncodingTest(unittest.TestCase):
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
        numeric_dim = 2
        if entity_type_ids is None:
            entity_type_ids = torch.zeros(nentity, dtype=torch.long)
        if cluster_ids is None:
            cluster_ids = torch.zeros(nentity, dtype=torch.long)
        if numeric_features is None:
            numeric_features = torch.zeros((nentity, numeric_dim), dtype=torch.float32)
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
            text_features=torch.zeros((nentity, text_dim), dtype=torch.float32),
            numeric_features=numeric_features,
            semantic_quality=torch.ones((nentity, 1), dtype=torch.float32),
            embedding_dim=20,
            embedding_shape1=4,
            hidden_size=576,
        )

    def test_full_relation_embedding_uses_relation_type_and_strength_without_id(self) -> None:
        model = self.build_model()
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertTrue(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "V6 full relation encoding should ignore relation-id-specific capacity when type and strength match",
        )
        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[2], atol=1e-6),
            "V6 full relation encoding should react to continuous relation strength",
        )

    def test_can_score_tail_pairs_from_external_head_embeddings(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["ex"],
                    ENTITY_TYPE_TO_ID["ex"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
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
        self.assertFalse(
            torch.allclose(base_scores, shifted_scores, atol=1e-6),
            "explicit forgetting heads should be able to change tail scores",
        )

    def test_no_relation_strength_ignores_continuous_strength(self) -> None:
        model = self.build_model()
        model.ablation_mode = "no_relation_strength"
        relation_ids = torch.tensor([0, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertTrue(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "no_relation_strength should keep relation type while ignoring both relation ID and continuous strength",
        )

    def test_no_relation_aware_uses_relation_id_only(self) -> None:
        model = self.build_model()
        model.ablation_mode = "no_relation_aware"
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        with torch.no_grad():
            expected = model.relation_norm(model.relation_id_emb(relation_ids))
        self.assertTrue(torch.allclose(relation_embeddings, expected, atol=1e-6))

    def test_v9_full_masks_statistical_pedagogical_features(self) -> None:
        model = self.build_model()

        self.assertIn("stat", model._numeric_groups_to_zero())
        self.assertIn("learner_stat", model._state_groups_to_zero())
        self.assertNotIn("irt", model._numeric_groups_to_zero())
        self.assertNotIn("learner_irt", model._state_groups_to_zero())

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

    def test_no_concept_semantic_only_masks_concept_text(self) -> None:
        model_a = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            )
        )
        torch.manual_seed(2024)
        model_b = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            )
        )
        model_a.ablation_mode = "no_concept_semantic"
        model_b.ablation_mode = "no_concept_semantic"
        with torch.no_grad():
            for model in [model_a, model_b]:
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
                model.raw_semantic_gate.fill_(10.0)
            model_a.text_features.zero_()
            model_b.text_features.zero_()
            model_b.text_features[1, 0] = 1.0
            model_b.text_features[2, 0] = 1.0

        embeddings_a = model_a.entity_embedding(torch.tensor([1, 2], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([1, 2], dtype=torch.long))

        self.assertTrue(
            torch.allclose(embeddings_a[0], embeddings_b[0], atol=1e-6),
            "no_concept_semantic should remove text contribution for concept entities only",
        )
        self.assertFalse(
            torch.allclose(embeddings_a[1], embeddings_b[1], atol=1e-6),
            "no_concept_semantic should keep exercise semantic text active",
        )

    def test_no_exercise_semantic_only_masks_exercise_text(self) -> None:
        model_a = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            )
        )
        torch.manual_seed(2024)
        model_b = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            )
        )
        model_a.ablation_mode = "no_exercise_semantic"
        model_b.ablation_mode = "no_exercise_semantic"
        with torch.no_grad():
            for model in [model_a, model_b]:
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
                model.raw_semantic_gate.fill_(10.0)
            model_a.text_features.zero_()
            model_b.text_features.zero_()
            model_b.text_features[1, 0] = 1.0
            model_b.text_features[2, 0] = 1.0

        embeddings_a = model_a.entity_embedding(torch.tensor([1, 2], dtype=torch.long))
        embeddings_b = model_b.entity_embedding(torch.tensor([1, 2], dtype=torch.long))

        self.assertFalse(
            torch.allclose(embeddings_a[0], embeddings_b[0], atol=1e-6),
            "no_exercise_semantic should keep concept semantic text active",
        )
        self.assertTrue(
            torch.allclose(embeddings_a[1], embeddings_b[1], atol=1e-6),
            "no_exercise_semantic should remove text contribution for exercise entities only",
        )

    def test_gate_values_are_type_level_and_start_near_five_percent(self) -> None:
        model = self.build_model()

        gate_values = model.gate_values()

        entity_gate_values = {key: value for key, value in gate_values.items() if key != "relation"}
        self.assertEqual(set(entity_gate_values), set(ENTITY_TYPE_TO_ID))
        for values in entity_gate_values.values():
            self.assertAlmostEqual(values["semantic"], 0.05, places=6)
            self.assertAlmostEqual(values["pedagogical"], 0.05, places=6)
            self.assertAlmostEqual(values["cluster"], 0.05, places=6)
            self.assertAlmostEqual(values["type"], 0.05, places=6)
        relation_gates = gate_values["relation"]
        self.assertAlmostEqual(relation_gates["type"], 0.05, places=6)
        self.assertAlmostEqual(relation_gates["strength"], 0.05, places=6)

    def test_no_content_entity_keeps_id_and_type_but_masks_added_entity_features(self) -> None:
        model_a = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            numeric_features=torch.ones((3, 2), dtype=torch.float32),
            cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long),
        )
        torch.manual_seed(2024)
        model_b = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            numeric_features=torch.ones((3, 2), dtype=torch.float32),
            cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long),
        )
        model_a.ablation_mode = "no_content_entity"
        model_b.ablation_mode = "no_content_entity"
        with torch.no_grad():
            model_b.text_features.fill_(9.0)
            model_b.numeric_features.fill_(7.0)
            model_b.cluster_ids.fill_(1)

        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(
            torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids), atol=1e-6),
            "no_content_entity should ignore semantic, pedagogical, and cluster feature values",
        )

    def test_semantic_quality_scales_semantic_contribution(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            )
        )
        model.ablation_mode = "id_head_reference"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.cluster_emb.weight.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
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

    def test_cluster_embedding_only_affects_learners(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long),
        )
        model.ablation_mode = "id_head_reference"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.text_projector.weight.zero_()
            model.semantic_quality.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.cluster_emb.weight.zero_()
            model.cluster_emb.weight[0, 0] = 1.0
            model.cluster_emb.weight[NO_CLUSTER_ID, 0] = 1.0
            model.raw_cluster_gate.fill_(10.0)

        embeddings = model.entity_embedding(torch.tensor([0, 1, 2], dtype=torch.long))

        self.assertFalse(torch.allclose(embeddings[0], torch.zeros_like(embeddings[0]), atol=1e-6))
        self.assertTrue(
            torch.allclose(embeddings[1], torch.zeros_like(embeddings[1]), atol=1e-6),
            "concept entities should not receive learner cluster embeddings",
        )
        self.assertTrue(
            torch.allclose(embeddings[2], torch.zeros_like(embeddings[2]), atol=1e-6),
            "exercise entities should not receive learner cluster embeddings",
        )

    def test_pedagogical_projection_is_masked_for_concepts(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                    ENTITY_TYPE_TO_ID["uid"],
                ],
                dtype=torch.long,
            )
        )
        model.ablation_mode = "id_head_reference"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.text_projector.weight.zero_()
            model.semantic_quality.zero_()
            model.cluster_emb.weight.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.numeric_projector[-1].bias[0] = 1.0
            model.raw_pedagogical_gate.fill_(10.0)

        embeddings = model.entity_embedding(torch.tensor([0, 1, 2], dtype=torch.long))

        self.assertTrue(
            torch.allclose(embeddings[0], torch.zeros_like(embeddings[0]), atol=1e-6),
            "concept entities have no pedagogical numeric feature and should not receive projector bias",
        )
        self.assertFalse(torch.allclose(embeddings[1], torch.zeros_like(embeddings[1]), atol=1e-6))
        self.assertFalse(torch.allclose(embeddings[2], torch.zeros_like(embeddings[2]), atol=1e-6))

    def test_no_exercise_irt_only_masks_exercise_numeric_features(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            numeric_features=torch.ones((3, 2), dtype=torch.float32),
        )
        model.ablation_mode = "no_exercise_irt"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.text_projector.weight.zero_()
            model.semantic_quality.zero_()
            model.cluster_emb.weight.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.numeric_projector[-1].bias[0] = 1.0
            model.raw_pedagogical_gate.fill_(10.0)

        embeddings = model.entity_embedding(torch.tensor([0, 2], dtype=torch.long))

        self.assertTrue(
            torch.allclose(embeddings[1], torch.zeros_like(embeddings[1]), atol=1e-6),
            "no_exercise_irt should remove numeric features only for exercise entities",
        )

    def test_no_learner_irt_only_masks_learner_numeric_features(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            numeric_features=torch.ones((3, 2), dtype=torch.float32),
        )
        model.ablation_mode = "no_learner_irt"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.text_projector.weight.zero_()
            model.semantic_quality.zero_()
            model.cluster_emb.weight.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.numeric_projector[-1].bias[0] = 1.0
            model.raw_pedagogical_gate.fill_(10.0)

        embeddings = model.entity_embedding(torch.tensor([0, 2], dtype=torch.long))

        self.assertTrue(
            torch.allclose(embeddings[0], torch.zeros_like(embeddings[0]), atol=1e-6),
            "no_learner_irt should remove numeric features only for learner entities",
        )
        self.assertFalse(torch.allclose(embeddings[1], torch.zeros_like(embeddings[1]), atol=1e-6))

    def test_no_cluster_masks_learner_cluster_only(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [
                    ENTITY_TYPE_TO_ID["uid"],
                    ENTITY_TYPE_TO_ID["kc"],
                    ENTITY_TYPE_TO_ID["ex"],
                ],
                dtype=torch.long,
            ),
            cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long),
        )
        model.ablation_mode = "no_cluster"
        with torch.no_grad():
            model.emb_e.weight.zero_()
            model.entity_type_emb.weight.zero_()
            model.text_projector.weight.zero_()
            model.semantic_quality.zero_()
            for layer in model.numeric_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            for layer in model.state_projector:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
            model.cluster_emb.weight.zero_()
            model.cluster_emb.weight[0, 0] = 1.0
            model.raw_cluster_gate.fill_(10.0)

        embedding = model.entity_embedding(torch.tensor([0], dtype=torch.long))[0]

        self.assertTrue(
            torch.allclose(embedding, torch.zeros_like(embedding), atol=1e-6),
            "no_cluster should remove learner cluster embeddings without disabling other pedagogical features",
        )

    def test_discrete_relation_uses_relation_id_embedding(self) -> None:
        model = self.build_model()
        model.ablation_mode = "discrete_relation"
        relation_ids = torch.tensor([0, 1], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "discrete_relation should distinguish relations with different ids even when type and strength match",
        )

    def test_hybrid_relation_combines_relation_id_with_type_and_strength(self) -> None:
        model = self.build_model()
        model.ablation_mode = "hybrid_relation"
        relation_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        relation_embeddings = model.relation_embedding(relation_ids)

        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[1], atol=1e-6),
            "hybrid_relation should retain relation-id-specific capacity",
        )
        self.assertFalse(
            torch.allclose(relation_embeddings[0], relation_embeddings[2], atol=1e-6),
            "hybrid_relation should also react to continuous relation strength",
        )


if __name__ == "__main__":
    unittest.main()
