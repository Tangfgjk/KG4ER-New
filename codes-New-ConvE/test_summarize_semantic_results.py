"""Unit checks for SemanticConvE summary helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_semantic_results import collect_gate_rows, collect_seed, summarize_gate_rows


class SemanticSummaryTest(unittest.TestCase):
    def write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_collect_seed_reads_ablation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            seed_dir = run_dir / "SemanticConvE_no_semantic" / "seed2024"
            self.write_json(seed_dir / "eval" / "metrics.json", {"ACC": {"10": {"mean": 0.7}}, "NOV": {"10": {"mean": 0.9}}})
            self.write_json(seed_dir / "metrics.json", {"training_seconds": 12.5})
            self.write_json(seed_dir / "semantic_conve_inference.json", {"inference_seconds": 1.25})

            row = collect_seed(run_dir, 2024, [10], "no_semantic")

            self.assertIsNotNone(row)
            self.assertEqual(row["ablation"], "no_semantic")
            self.assertEqual(row["ACC@10"], 0.7)
            self.assertEqual(row["NOV@10"], 0.9)
            self.assertEqual(row["training_seconds"], 12.5)

    def test_collects_gate_values_per_seed_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            seed2024 = run_dir / "SemanticConvE" / "seed2024"
            seed2025 = run_dir / "SemanticConvE" / "seed2025"
            self.write_json(
                seed2024 / "gate_values.json",
                {"gate_values": {"uid": {"semantic": 0.4, "pedagogical": 0.6, "cluster": 0.8}}},
            )
            self.write_json(
                seed2025 / "gate_values.json",
                {"gate_values": {"uid": {"semantic": 0.6, "pedagogical": 0.8, "cluster": 1.0}}},
            )

            rows = collect_gate_rows(run_dir, [2024, 2025], ["full"])
            summaries = summarize_gate_rows(rows)

            self.assertEqual(len(rows), 6)
            uid_semantic = next(
                item for item in summaries if item["entity_type"] == "uid" and item["feature_type"] == "semantic"
            )
            self.assertAlmostEqual(uid_semantic["mean"], 0.5)
            self.assertAlmostEqual(uid_semantic["std"], 0.14142135623730948)


if __name__ == "__main__":
    unittest.main()
