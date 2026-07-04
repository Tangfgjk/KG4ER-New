"""Tests for SemanticConvE experiment utilities."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from semantic_experiment_utils import MODEL_VERSION, ablation_model_dir, graph_path_for_dataset


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


if __name__ == "__main__":
    unittest.main()
