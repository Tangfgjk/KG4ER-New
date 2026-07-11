from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import (
    locate_source_feature_dir,
    locate_source_graph_dir,
    minmax,
    output_data_root,
    read_entity_dict,
    read_json,
    source_data_root,
    source_dataset_dir,
    v13_dataset_dir,
    write_json,
)


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return np.asarray(matrix, dtype=np.float64)


def _log_count(value: float) -> float:
    return min(1.0, math.log1p(max(0.0, float(value))) / 10.0)


def sequence_stats(input_dir: Path) -> tuple[dict[int, int], dict[int, float], dict[int, int], dict[int, float]]:
    all_path = input_dir / "all.csv"
    if not all_path.exists():
        return {}, {}, {}, {}
    df = pd.read_csv(all_path)
    item_counts = df.groupby("item_id").size().astype(int).to_dict()
    item_correct = df.groupby("item_id")["score"].mean().astype(float).to_dict()
    user_counts = df.groupby("user_id").size().astype(int).to_dict()
    user_correct = df.groupby("user_id")["score"].mean().astype(float).to_dict()
    return item_counts, item_correct, user_counts, user_correct


def build_exercise_features(
    a_param: np.ndarray,
    b_param: np.ndarray,
    exercise_count: int,
    interaction_counts: dict[int, int],
    correct_rates: dict[int, float],
) -> dict[str, dict[str, Any]]:
    difficulty = np.zeros(exercise_count, dtype=np.float64)
    discrimination = np.zeros(exercise_count, dtype=np.float64)
    observed = min(exercise_count, b_param.shape[0], a_param.shape[0])
    if observed:
        difficulty[:observed] = b_param.reshape(-1)[:observed]
        discrimination[:observed] = np.linalg.norm(a_param[:observed], axis=1)
    difficulty_norm = minmax(difficulty, default=0.5)
    discrimination_norm = minmax(discrimination, default=0.5)
    features: dict[str, dict[str, Any]] = {}
    for idx in range(exercise_count):
        correct_rate = float(correct_rates.get(idx, 0.0))
        count = int(interaction_counts.get(idx, 0))
        features[f"ex{idx}"] = {
            "entity_id": f"ex{idx}",
            "entity_type": "exercise",
            "difficulty_mirt": float(difficulty[idx]),
            "difficulty_mirt_norm": float(difficulty_norm[idx]),
            "discrimination_mirt": float(discrimination[idx]),
            "discrimination_mirt_norm": float(discrimination_norm[idx]),
            # The current feature_loader expects these canonical names.
            "difficulty": float(difficulty[idx]),
            "difficulty_norm": float(difficulty_norm[idx]),
            "discrimination": float(discrimination[idx]),
            "discrimination_norm": float(discrimination_norm[idx]),
            "correct_rate": correct_rate,
            "error_rate": float(1.0 - correct_rate),
            "interaction_count": count,
            "log_interaction_count": _log_count(count),
            "kc_discrimination_vector": [float(v) for v in a_param[idx].tolist()] if idx < a_param.shape[0] else [],
            "feature_source": "v13_graph_subset_EduCDM_modified_no_q_MIRT",
        }
    return features


def build_learner_features(
    theta_param: np.ndarray,
    mastery_matrix: np.ndarray,
    user_counts: dict[int, int],
    user_correct: dict[int, float],
    n_clusters: int,
) -> dict[str, dict[str, Any]]:
    learner_count = theta_param.shape[0]
    theta_mean = theta_param.mean(axis=1)
    theta_norm = minmax(theta_mean, default=0.5)
    mastery = np.asarray(mastery_matrix, dtype=np.float64)
    if mastery.ndim != 2 or mastery.shape[0] != learner_count:
        raise ValueError(
            "mastery matrix must align with v13 graph learners: "
            f"mastery_shape={mastery.shape}, learner_count={learner_count}"
        )
    cluster_input = np.column_stack([theta_norm, theta_mean, mastery.mean(axis=1), mastery.std(axis=1)])
    cluster_count = max(1, min(int(n_clusters), learner_count))
    clusters = np.zeros(learner_count, dtype=int)
    if cluster_count > 1:
        clusters = KMeans(n_clusters=cluster_count, random_state=2024, n_init=10).fit_predict(cluster_input)

    features: dict[str, dict[str, Any]] = {}
    for idx in range(learner_count):
        history_length = int(user_counts.get(idx, 0))
        mastery_mean = float(mastery[idx].mean()) if mastery.shape[1] else float(theta_norm[idx])
        mastery_std = float(mastery[idx].std()) if mastery.shape[1] else 0.0
        features[f"uid{idx}"] = {
            "entity_id": f"uid{idx}",
            "entity_type": "learner",
            "theta_mirt_mean": float(theta_mean[idx]),
            "theta_norm": float(theta_norm[idx]),
            "overall_mastery_irt": float(theta_norm[idx]),
            "overall_mastery_kt_mean": mastery_mean,
            "concept_mastery_mean": mastery_mean,
            "concept_mastery_std": mastery_std,
            "correct_rate": float(user_correct.get(idx, 0.0)),
            "history_length": history_length,
            "log_history_length": _log_count(history_length),
            "cluster_id": int(clusters[idx]),
            "theta_mirt_vector": [float(v) for v in theta_param[idx].tolist()],
            "feature_source": "v13_MIRT_theta_plus_v13_mastery",
        }
    return features


def copy_text_features(source_feature_dir: Path | None, output_feature_dir: Path) -> None:
    if source_feature_dir is None:
        raise FileNotFoundError(
            "No source semantic feature directory found. "
            "v13 can reuse concept/exercise text metadata, but text embeddings must be imported "
            "from EKTM_mirt TopicRNNModel topic_v."
        )
    for rel in [
        Path("entity_features") / "concept_semantics.json",
        Path("entity_features") / "exercise_semantics.json",
        Path("entity_features") / "entity_feature_index.json",
    ]:
        src = source_feature_dir / rel
        if src.exists():
            dst = output_feature_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def build_entity_feature_index(
    entity2id: dict[str, int],
    exercise_features: dict[str, dict[str, Any]],
    learner_features: dict[str, dict[str, Any]],
    output_feature_dir: Path,
) -> dict[str, Any]:
    text_dir = output_feature_dir / "text_embeddings"
    concept_text_path = text_dir / "concept_text_embeddings.npy"
    exercise_text_path = text_dir / "exercise_text_embeddings.npy"

    concept_text_shape = list(np.load(concept_text_path).shape) if concept_text_path.exists() else None
    exercise_text_shape = list(np.load(exercise_text_path).shape) if exercise_text_path.exists() else None

    concepts = sorted((name for name in entity2id if name.startswith("kc")), key=lambda x: entity2id[x])
    exercises = sorted((name for name in entity2id if name.startswith("ex")), key=lambda x: entity2id[x])
    learners = sorted((name for name in entity2id if name.startswith("uid")), key=lambda x: entity2id[x])

    return {
        "version": "v13_entity_feature_index",
        "entity_count": len(entity2id),
        "entity_groups": {
            "learners": {
                "count": len(learners),
                "entities": learners,
                "feature_file": "entity_features/learner_pedagogy.json",
                "feature_key": "learners",
                "feature_count": len(learner_features),
                "numeric_features": [
                    "theta_norm",
                    "overall_mastery_irt",
                    "overall_mastery_kt_mean",
                    "correct_rate",
                    "history_length",
                    "concept_mastery_mean",
                    "concept_mastery_std",
                    "cluster_id",
                ],
                "text_embedding": None,
            },
            "exercises": {
                "count": len(exercises),
                "entities": exercises,
                "semantic_file": "entity_features/exercise_semantics.json",
                "irt_feature_file": "irt_features/exercise_irt_features.json",
                "feature_key": "exercises",
                "feature_count": len(exercise_features),
                "numeric_features": [
                    "difficulty_mirt_norm",
                    "discrimination_mirt_norm",
                    "correct_rate",
                    "error_rate",
                    "interaction_count",
                ],
                "text_embedding": {
                    "file": "text_embeddings/exercise_text_embeddings.npy",
                    "shape": exercise_text_shape,
                    "source": "EKTM_mirt TopicRNNModel / Bi-GRU",
                },
            },
            "concepts": {
                "count": len(concepts),
                "entities": concepts,
                "semantic_file": "entity_features/concept_semantics.json",
                "feature_key": "concepts",
                "feature_count": len(concepts),
                "numeric_features": [],
                "text_embedding": {
                    "file": "text_embeddings/concept_text_embeddings.npy",
                    "shape": concept_text_shape,
                    "source": "same trained EKTM_mirt TopicRNNModel / Bi-GRU encoder",
                },
            },
        },
        "notes": (
            "This file is an audit index for SemanticConvE feature alignment. "
            "The runtime feature loader aligns features by entity ids from entities.dict; "
            "this index records which feature files correspond to uid/ex/kc entities."
        ),
    }


def require_ektm_text_embeddings(output_feature_dir: Path) -> None:
    manifest_path = output_feature_dir / "text_embeddings" / "text_embedding_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Strict v13 no longer accepts BGE text embeddings. "
            "Run v13_pipeline/import_ektm_topic_embeddings_v13.py first."
        )
    manifest = read_json(manifest_path)
    model_name = str(
        manifest.get("model")
        or manifest.get("model_name")
        or manifest.get("source")
        or ""
    ).lower()
    if "bge" in model_name or "sentence" in model_name:
        raise ValueError(
            f"Legacy BGE/SentenceTransformer text embeddings are not allowed in v13: {manifest_path}. "
            "Import EKTM_mirt TopicRNNModel topic_v embeddings instead."
        )
    if "ektm" not in model_name and "topic" not in model_name:
        raise ValueError(
            f"Text embedding manifest does not identify EKTM_mirt topic_v source: {manifest_path}. "
            "Expected model/source to contain EKTM or topic."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v13 MIRT-derived feature files for SemanticConvE.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--mirt-output-dir", type=Path, default=None)
    parser.add_argument("--mastery-file", type=Path, default=None)
    parser.add_argument("--output-feature-dir", type=Path, default=None)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    graph_dir = locate_source_graph_dir(source_dir)
    v13_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    input_dir = args.input_dir or (v13_dir / "mirt" / "inputs")
    mirt_output_dir = args.mirt_output_dir or (v13_dir / "mirt" / "outputs")
    output_feature_dir = args.output_feature_dir or (v13_dir / "semantic_kg_features")
    mastery_file = args.mastery_file or (v13_dir / "stu2know_mastery.json")
    if output_feature_dir.exists() and any(output_feature_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_feature_dir} already exists. Use --force to overwrite.")

    manifest = read_json(mirt_output_dir / "mirt_export_manifest.json")
    latent_dim = int(manifest["latent_dim"])
    a_param = load_matrix(mirt_output_dir / f"a_param_{latent_dim}.csv")
    b_param = load_matrix(mirt_output_dir / f"b_param_{latent_dim}.csv")
    theta_param = load_matrix(mirt_output_dir / f"theta_param_{latent_dim}.csv")
    entity2id = read_entity_dict(graph_dir / "entities.dict")
    exercise_count = sum(1 for name in entity2id if name.startswith("ex"))
    mastery = np.asarray(read_json(mastery_file), dtype=np.float64)
    item_counts, item_correct, user_counts, user_correct = sequence_stats(input_dir)
    exercise_features = build_exercise_features(a_param, b_param, exercise_count, item_counts, item_correct)
    learner_features = build_learner_features(theta_param, mastery, user_counts, user_correct, args.n_clusters)

    source_feature_dir = locate_source_feature_dir(args.dataset, source_dir, graph_dir)
    copy_text_features(source_feature_dir, output_feature_dir)
    require_ektm_text_embeddings(output_feature_dir)
    write_json(output_feature_dir / "entity_features" / "learner_pedagogy.json", {"dataset": args.dataset, "learners": learner_features})
    write_json(output_feature_dir / "irt_features" / "exercise_irt_features.json", {"dataset": args.dataset, "exercises": exercise_features})
    write_json(
        output_feature_dir / "entity_features" / "entity_feature_index.json",
        build_entity_feature_index(entity2id, exercise_features, learner_features, output_feature_dir),
    )
    write_json(
        output_feature_dir / "feature_generation_manifest.json",
        {
            "dataset": args.dataset,
            "feature_version": "v13_graph_subset_mirt_features",
            "mirt_output_dir": mirt_output_dir,
            "mastery_file": mastery_file,
            "source_feature_dir": source_feature_dir,
            "output_feature_dir": output_feature_dir,
            "latent_dim": latent_dim,
            "exercise_count": len(exercise_features),
            "learner_count": len(learner_features),
            "notes": (
                "Concept/exercise text metadata is reused. Learner/exercise pedagogical features are regenerated "
                "from v13 MIRT and v13 mastery. Text embeddings must come from EKTM_mirt TopicRNNModel topic_v; "
                "legacy BGE/SentenceTransformer embeddings are rejected."
            ),
        },
    )
    print(f"saved v13 semantic feature files: {output_feature_dir}")


if __name__ == "__main__":
    main()

