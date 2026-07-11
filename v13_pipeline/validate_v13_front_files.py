from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_DATASETS,
    load_matrix_json,
    matrix_stats,
    output_data_root,
    read_entity_dict,
    read_json,
    read_q_matrix,
    v13_dataset_dir,
    write_json,
)


def count_relations(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def triple_relation_stats(path: Path) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    value_one_counter: Counter[str] = Counter()
    total = 0
    if not path.exists():
        return {"exists": False, "total": 0, "by_type": {}, "value_one_by_type": {}}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            total += 1
            rel = parts[1]
            if rel == "rec":
                kind = "rec"
            elif rel.startswith("mlkc"):
                kind = "mlkc"
            elif rel.startswith("pkc"):
                kind = "pkc"
            elif rel.startswith("exfr"):
                kind = "exfr"
            else:
                kind = "other"
            counter[kind] += 1
            if rel.endswith("1.00"):
                value_one_counter[kind] += 1
    return {
        "exists": True,
        "total": int(total),
        "by_type": dict(counter),
        "value_one_by_type": dict(value_one_counter),
        "value_one_ratio_by_type": {
            kind: float(value_one_counter[kind] / count) if count else 0.0 for kind, count in counter.items()
        },
    }


def rounded_tie_ratio(full_scores: np.ndarray, decimals: int = 2) -> float:
    if full_scores.size == 0:
        return 0.0
    rounded = np.round(full_scores, decimals)
    ratios = []
    for row in rounded:
        counts = Counter(row.tolist())
        tied = sum(count for count in counts.values() if count > 1)
        ratios.append(tied / len(row) if len(row) else 0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def validate_dataset(dataset: str, data_root: Path) -> dict[str, Any]:
    graph_dir = v13_dataset_dir(dataset, data_root)
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "entities.dict",
        "relations.dict",
        "Q.txt",
        "stu2know_mastery.json",
        "stu2know_seq.json",
        "stu2know_forget.json",
        "stu2ex_forget.json",
        "stu2ex_recommend.json",
        "stu2ex_recommend_full_precision.json",
        "triples.txt",
        "test_triples.txt",
    ]
    for name in required:
        if not (graph_dir / name).exists():
            errors.append(f"missing {name}")

    report: dict[str, Any] = {"dataset": dataset, "graph_dir": graph_dir, "errors": errors, "warnings": warnings}
    if errors:
        report["status"] = "failed"
        return report

    entity2id = read_entity_dict(graph_dir / "entities.dict")
    q = read_q_matrix(graph_dir / "Q.txt")
    mastery = load_matrix_json(graph_dir / "stu2know_mastery.json")
    seq = load_matrix_json(graph_dir / "stu2know_seq.json")
    know_forget = load_matrix_json(graph_dir / "stu2know_forget.json")
    ex_forget = load_matrix_json(graph_dir / "stu2ex_forget.json")
    rec = load_matrix_json(graph_dir / "stu2ex_recommend.json")
    rec_full = load_matrix_json(graph_dir / "stu2ex_recommend_full_precision.json")

    expected_shape = (sum(1 for name in entity2id if name.startswith("uid")), q.shape[1])
    expected_ex_shape = (expected_shape[0], q.shape[0])
    for name, matrix in [("mastery", mastery), ("seq", seq), ("know_forget", know_forget)]:
        if tuple(matrix.shape) != expected_shape:
            errors.append(f"{name} shape {matrix.shape} != expected {expected_shape}")
        if matrix.size and (np.min(matrix) < -1e-8 or np.max(matrix) > 1 + 1e-8):
            errors.append(f"{name} values out of [0,1]")
    for name, matrix in [("ex_forget", ex_forget), ("rec", rec), ("rec_full", rec_full)]:
        if tuple(matrix.shape) != expected_ex_shape:
            errors.append(f"{name} shape {matrix.shape} != expected {expected_ex_shape}")
    if ex_forget.size and (np.min(ex_forget) < -1e-8 or np.max(ex_forget) > 1 + 1e-8):
        errors.append("stu2ex_forget values out of [0,1]")
    if seq.size:
        seq_std = float(np.std(seq))
        seq_unique_ratio = float(np.unique(np.round(seq, 8), axis=0).shape[0] / max(1, seq.shape[0]))
        if seq_std < 0.03:
            warnings.append(f"stu2know_seq std is small: {seq_std:.6f}")
        if seq_unique_ratio < 0.8:
            warnings.append(f"stu2know_seq unique-row ratio is low: {seq_unique_ratio:.6f}")
        if float(np.min(seq)) > 0.9:
            warnings.append(f"stu2know_seq min > 0.9: {float(np.min(seq)):.6f}")
    relation_count = count_relations(graph_dir / "relations.dict")
    if relation_count != 304:
        errors.append(f"relation_count {relation_count} != 304")

    feature_dir = graph_dir / "semantic_kg_features"
    feature_status = {
        "exists": feature_dir.exists(),
        "text_manifest": (feature_dir / "text_embeddings" / "text_embedding_manifest.json").exists(),
        "concept_semantics": (feature_dir / "entity_features" / "concept_semantics.json").exists(),
        "exercise_semantics": (feature_dir / "entity_features" / "exercise_semantics.json").exists(),
        "learner_pedagogy": (feature_dir / "entity_features" / "learner_pedagogy.json").exists(),
        "exercise_irt": (feature_dir / "irt_features" / "exercise_irt_features.json").exists(),
    }
    if not all(feature_status.values()):
        warnings.append(f"semantic feature directory is incomplete: {feature_status}")
    if feature_status["text_manifest"]:
        text_manifest = read_json(feature_dir / "text_embeddings" / "text_embedding_manifest.json")
        model_name = str(
            text_manifest.get("model")
            or text_manifest.get("model_name")
            or text_manifest.get("source")
            or ""
        ).lower()
        if "bge" in model_name or "sentence" in model_name:
            errors.append(
                "text embeddings are legacy BGE/SentenceTransformer features; "
                "strict v13 should use EKTM_mirt TopicRNNModel topic_v embeddings"
            )
        if "ektm" not in model_name and "topic" not in model_name:
            errors.append("text embedding manifest must identify EKTM_mirt TopicRNNModel topic_v source")

    manifest = read_json(graph_dir / "v13_graph_manifest.json") if (graph_dir / "v13_graph_manifest.json").exists() else {}
    report.update(
        {
            "status": "failed" if errors else "passed",
            "entity_count": len(entity2id),
            "learner_count": expected_shape[0],
            "exercise_count": q.shape[0],
            "concept_count": q.shape[1],
            "relation_count": relation_count,
            "matrices": {
                "stu2know_mastery": matrix_stats(mastery),
                "stu2know_seq": matrix_stats(seq),
                "stu2know_forget": matrix_stats(know_forget),
                "stu2ex_forget": matrix_stats(ex_forget),
                "stu2ex_recommend": matrix_stats(rec),
            },
            "recommendation": {
                "rounded_2_decimal_tie_ratio": rounded_tie_ratio(rec_full, decimals=2),
                "ranking_precision": manifest.get("recommendation", {}).get("ranking_precision"),
                "sequence_term": manifest.get("recommendation", {}).get("sequence_term"),
            },
            "triples": {
                "train": triple_relation_stats(graph_dir / "triples.txt"),
                "test": triple_relation_stats(graph_dir / "test_triples.txt"),
            },
            "semantic_features": feature_status,
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v13 front files after regeneration.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    reports = [validate_dataset(dataset, args.output_data_root) for dataset in datasets]
    status = "passed" if all(report["status"] == "passed" for report in reports) else "failed"
    payload = {"data_root": args.output_data_root, "reports": reports, "status": status}
    output_file = args.output_file or (args.output_data_root / "v13_validation_report.json")
    write_json(output_file, payload)
    print(json_dumps(payload))
    if status != "passed":
        raise SystemExit(1)


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()

