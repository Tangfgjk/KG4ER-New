from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import output_data_root, read_entity_dict, read_json, read_q_matrix, v13_dataset_dir, write_json


def load_embedding_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(path, delimiter=",")
    elif suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            if "embeddings" in payload:
                payload = payload["embeddings"]
            elif "exercise_embeddings" in payload:
                payload = payload["exercise_embeddings"]
            elif "topic_v" in payload:
                payload = payload["topic_v"]
        matrix = np.asarray(payload, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding file format: {path}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2-D: {path}, shape={matrix.shape}")
    if matrix.shape[1] <= 0:
        raise ValueError(f"Embedding dimension must be positive: {path}, shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Embedding matrix contains NaN or Inf: {path}")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import EKTM_mirt TopicRNNModel topic_v exercise embeddings into v13 semantic_kg_features."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--topic-embedding-file", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--output-feature-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="EKTM_mirt.TopicRNNModel")
    parser.add_argument("--source-name", default="topic_v")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    entity2id = read_entity_dict(graph_dir / "entities.dict")
    q_matrix = read_q_matrix(graph_dir / "Q.txt")
    exercise_ids = [f"ex{idx}" for idx in range(q_matrix.shape[0])]
    concept_ids = [f"kc{idx}" for idx in range(q_matrix.shape[1])]
    missing_ex = [entity_id for entity_id in exercise_ids if entity_id not in entity2id]
    missing_kc = [entity_id for entity_id in concept_ids if entity_id not in entity2id]
    if missing_ex or missing_kc:
        raise ValueError(
            "v13 graph entities do not align with Q.txt: "
            f"missing_ex={missing_ex[:5]}, missing_kc={missing_kc[:5]}"
        )

    exercise_embeddings = load_embedding_matrix(args.topic_embedding_file)
    if exercise_embeddings.shape[0] != len(exercise_ids):
        raise ValueError(
            "EKTM_mirt topic_v row count must match v13 exercise count: "
            f"rows={exercise_embeddings.shape[0]}, exercises={len(exercise_ids)}"
        )
    embedding_dim = int(exercise_embeddings.shape[1])
    concept_embeddings = np.zeros((len(concept_ids), embedding_dim), dtype=np.float32)

    feature_dir = args.output_feature_dir or (graph_dir / "semantic_kg_features")
    text_dir = feature_dir / "text_embeddings"
    if text_dir.exists() and any(text_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{text_dir} already exists. Use --force to overwrite.")
    text_dir.mkdir(parents=True, exist_ok=True)

    np.save(text_dir / "exercise_text_embeddings.npy", exercise_embeddings.astype(np.float32))
    np.save(text_dir / "concept_text_embeddings.npy", concept_embeddings)
    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model_name,
        "source": args.source_name,
        "embedding_dim": embedding_dim,
        "exercise_entity_ids": exercise_ids,
        "concept_entity_ids": concept_ids,
        "files": {
            "exercise_text_embeddings": "exercise_text_embeddings.npy",
            "concept_text_embeddings": "concept_text_embeddings.npy",
        },
        "exercise_embedding_source_file": str(args.topic_embedding_file),
        "concept_embedding_source": "zero_vector_not_used_by_EKTM_mirt_text_encoder",
        "notes": (
            "Strict v13 uses EKTM_mirt TopicRNNModel topic_v for exercise text embeddings. "
            "Concept text embeddings are zero vectors because EKTM_mirt does not encode concept definitions."
        ),
    }
    write_json(text_dir / "text_embedding_manifest.json", manifest)
    print(f"imported EKTM_mirt topic embeddings: {text_dir}")


if __name__ == "__main__":
    main()

