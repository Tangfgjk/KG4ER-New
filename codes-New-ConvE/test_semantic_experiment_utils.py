"""Tests for SemanticConvE experiment utilities."""

from __future__ import annotations

import unittest
import tempfile
from argparse import Namespace
from pathlib import Path

from run_semantic_experiments import command_test
from semantic_experiment_utils import (
    DEFAULT_ALL_ABLATIONS,
    MODEL_VERSION,
    ablation_model_dir,
    graph_path_for_dataset,
    parse_ablation_list,
)


class SemanticExperimentUtilsTest(unittest.TestCase):
    def test_graph_path_prefers_prepared_for_formal_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            prepared = data_root / "statics2011" / "prepared_for_kt"
            prepared.mkdir(parents=True)
            (prepared / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")

            graph_path = graph_path_for_dataset("statics2011", data_root)

            self.assertEqual(graph_path.name, "prepared_for_kt")
            self.assertTrue((graph_path / "entities.dict").exists())

    def test_graph_path_uses_eedi_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            eedi = data_root / "Eedi"
            eedi.mkdir(parents=True)
            (eedi / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")

            graph_path = graph_path_for_dataset("Eedi", data_root)

            self.assertEqual(graph_path.name, "Eedi")
            self.assertTrue((graph_path / "entities.dict").exists())

    def test_model_version_marks_continuous_relation_encoding(self) -> None:
        self.assertIn("v5", MODEL_VERSION)
        self.assertIn("id_anchored", MODEL_VERSION)

    def test_ablation_model_dir_keeps_full_backward_compatible(self) -> None:
        self.assertEqual(ablation_model_dir("full"), "SemanticConvE")
        self.assertEqual(ablation_model_dir("no_semantic"), "SemanticConvE_no_semantic")
        self.assertEqual(ablation_model_dir("no_forgetting"), "SemanticConvE_no_forgetting")

    def test_parse_ablation_list_expands_all_keyword(self) -> None:
        ablations = parse_ablation_list("all")

        self.assertEqual(ablations, DEFAULT_ALL_ABLATIONS)
        self.assertEqual(
            ablations,
            [
                "full",
                "no_content_entity",
                "no_relation_aware",
                "no_type_aware_scoring",
                "no_mastery",
                "no_forgetting",
                "no_seq",
            ],
        )

    def test_parse_ablation_list_keeps_explicit_order(self) -> None:
        self.assertEqual(parse_ablation_list("full,no_cluster"), ["full", "no_cluster"])

    def test_test_command_passes_forgetting_score_weight(self) -> None:
        args = Namespace(dataset="Eedi", cuda="auto", forgetting_score_weight=0.2, forgetting_exercise_batch_size=128)

        command = command_test(args, Path("graph"), Path("seed"), "full")

        self.assertIn("--forgetting-score-weight", command)
        self.assertIn("0.2", command)
        self.assertIn("--forgetting-exercise-batch-size", command)
        self.assertIn("128", command)


if __name__ == "__main__":
    unittest.main()
