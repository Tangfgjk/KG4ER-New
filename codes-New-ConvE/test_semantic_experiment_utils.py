"""Tests for SemanticConvE experiment utilities."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

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
        self.assertIn("continuous_relation", MODEL_VERSION)

    def test_ablation_model_dir_keeps_full_backward_compatible(self) -> None:
        self.assertEqual(ablation_model_dir("full"), "SemanticConvE")
        self.assertEqual(ablation_model_dir("no_semantic"), "SemanticConvE_no_semantic")
        self.assertEqual(ablation_model_dir("no_forgetting"), "SemanticConvE_no_forgetting")

    def test_parse_ablation_list_expands_all_keyword(self) -> None:
        ablations = parse_ablation_list("all")

        self.assertEqual(ablations, DEFAULT_ALL_ABLATIONS)
        self.assertIn("full", ablations)
        self.assertIn("no_concept_semantic", ablations)
        self.assertIn("no_exercise_irt", ablations)
        self.assertIn("hybrid_relation", ablations)

    def test_parse_ablation_list_keeps_explicit_order(self) -> None:
        self.assertEqual(parse_ablation_list("full,no_cluster"), ["full", "no_cluster"])


if __name__ == "__main__":
    unittest.main()
