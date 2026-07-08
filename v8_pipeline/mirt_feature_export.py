from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import (
    copy_dir,
    default_data_root,
    graph_learner_fit_indices,
    locate_dataset_dir,
    locate_graph_dir,
    locate_semantic_feature_dir,
    minmax,
    read_entity_dict,
    read_json,
    write_json,
)


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return np.asarray(matrix, dtype=np.float64)


def _log_count(value: float) -> float:
    return min(1.0, math.log1p(max(0.0, float(value))) / 10.0)


def build_exercise_features(
    a_param: np.ndarray,
    b_param: np.ndarray,
    interaction_counts: dict[int, int] | None = None,
    correct_rates: dict[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    interaction_counts = interaction_counts or {}
    correct_rates = correct_rates or {}
    difficulty = b_param.reshape(-1)
    discrimination = np.linalg.norm(a_param, axis=1)
    difficulty_norm = minmax(difficulty, default=0.5)
    discrimination_norm = minmax(discrimination, default=0.5)
    features: dict[str, dict[str, Any]] = {}
    for idx in range(a_param.shape[0]):
        correct_rate = float(correct_rates.get(idx, 0.0))
        count = int(interaction_counts.get(idx, 0))
        features[f"ex{idx}"] = {
            "entity_id": f"ex{idx}",
            "entity_type": "exercise",
            "difficulty_mirt": float(difficulty[idx]),
            "difficulty_mirt_norm": float(difficulty_norm[idx]),
            "discrimination_mirt": float(discrimination[idx]),
            "discrimination_mirt_norm": float(discrimination_norm[idx]),
            "difficulty": float(difficulty[idx]),
            "difficulty_norm": float(difficulty_norm[idx]),
            "discrimination": float(discrimination[idx]),
            "discrimination_norm": float(discrimination_norm[idx]),
            "correct_rate": correct_rate,
            "error_rate": float(1.0 - correct_rate),
            "interaction_count": count,
            "log_interaction_count": _log_count(count),
            "kc_discrimination_vector": [float(v) for v in a_param[idx].tolist()],
            "feature_source": "EduCDM_modified_no_q_MIRT",
        }
    return features


def build_learner_features(
    theta_param: np.ndarray,
    mastery_matrix: list[list[float]],
    correct_rates: dict[int, float] | None = None,
    history_lengths: dict[int, int] | None = None,
    n_clusters: int = 5,
) -> dict[str, dict[str, Any]]:
    correct_rates = correct_rates or {}
    history_lengths = history_lengths or {}
    theta_mean = theta_param.mean(axis=1)
    theta_norm = minmax(theta_mean, default=0.5)
    n_learners = theta_param.shape[0]
    mastery_arr = np.zeros((n_learners, 0), dtype=np.float64)
    if mastery_matrix:
        mastery_arr = np.asarray(mastery_matrix, dtype=np.float64)
        if mastery_arr.ndim == 1:
            mastery_arr = mastery_arr.reshape(-1, 1)
        if mastery_arr.shape[0] < n_learners:
            padding = np.full((n_learners - mastery_arr.shape[0], mastery_arr.shape[1]), 0.5)
            mastery_arr = np.vstack([mastery_arr, padding])
        mastery_arr = mastery_arr[:n_learners]
    cluster_input = np.column_stack([theta_norm, theta_mean])
    if mastery_arr.size:
        cluster_input = np.column_stack([cluster_input, mastery_arr.mean(axis=1), mastery_arr.std(axis=1)])
    cluster_count = max(1, min(int(n_clusters), n_learners))
    if cluster_count == 1:
        clusters = np.zeros(n_learners, dtype=int)
    else:
        clusters = KMeans(n_clusters=cluster_count, random_state=2024, n_init=10).fit_predict(cluster_input)

    features: dict[str, dict[str, Any]] = {}
    for idx in range(n_learners):
        mastery_row = mastery_arr[idx] if mastery_arr.size else np.asarray([], dtype=np.float64)
        mastery_mean = float(mastery_row.mean()) if mastery_row.size else float(theta_norm[idx])
        mastery_std = float(mastery_row.std()) if mastery_row.size else 0.0
        history_length = int(history_lengths.get(idx, 0))
        features[f"uid{idx}"] = {
            "entity_id": f"uid{idx}",
            "entity_type": "learner",
            "theta_mirt_mean": float(theta_mean[idx]),
            "theta_norm": float(theta_norm[idx]),
            "overall_mastery_irt": float(theta_norm[idx]),
            "overall_mastery_kt_mean": mastery_mean,
            "concept_mastery_mean": mastery_mean,
            "concept_mastery_std": mastery_std,
            "correct_rate": float(correct_rates.get(idx, 0.0)),
            "history_length": history_length,
            "log_history_length": _log_count(history_length),
            "cluster_id": int(clusters[idx]),
            "theta_mirt_vector": [float(v) for v in theta_param[idx].tolist()],
            "feature_source": "EduCDM_modified_no_q_MIRT_and_V8_mastery",
        }
    return features


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


def align_mastery_for_graph_learners(mastery: list[list[float]], fit_indices: list[int], full_user_count: int) -> list[list[float]]:
    if not mastery:
        return []
    mastery_arr = np.asarray(mastery, dtype=np.float64)
    if mastery_arr.ndim == 1:
        mastery_arr = mastery_arr.reshape(-1, 1)
    if mastery_arr.shape[0] == len(fit_indices):
        return mastery_arr.tolist()
    if mastery_arr.shape[0] == full_user_count:
        return mastery_arr[fit_indices].tolist()
    raise ValueError(
        "V8 mastery row count is neither graph learner count nor full MIRT user count: "
        f"mastery_rows={mastery_arr.shape[0]}, graph_learners={len(fit_indices)}, mirt_users={full_user_count}"
    )


def ensure_text_feature_files(source_feature_dir: Path | None, output_feature_dir: Path, entity2id: dict[str, int]) -> None:
    entity_out = output_feature_dir / "entity_features"
    if source_feature_dir:
        source_entity = source_feature_dir / "entity_features"
        for file_name in ["concept_semantics.json", "exercise_semantics.json"]:
            src = source_entity / file_name
            if src.exists():
                entity_out.mkdir(parents=True, exist_ok=True)
                (entity_out / file_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        source_text = source_feature_dir / "text_embeddings"
        if source_text.exists():
            copy_dir(source_text, output_feature_dir / "text_embeddings")

    concept_path = entity_out / "concept_semantics.json"
    exercise_path = entity_out / "exercise_semantics.json"
    if not concept_path.exists():
        concepts = {name: {"entity_id": name, "entity_type": "knowledge_concept", "text_for_embedding": name, "definition_source": "missing_v8_fallback"} for name in entity2id if name.startswith("kc")}
        write_json(concept_path, {"concepts": concepts})
    if not exercise_path.exists():
        exercises = {name: {"entity_id": name, "entity_type": "exercise", "text_for_embedding": name, "text_source": "missing_v8_fallback"} for name in entity2id if name.startswith("ex")}
        write_json(exercise_path, {"exercises": exercises})

    text_dir = output_feature_dir / "text_embeddings"
    manifest_path = text_dir / "text_embedding_manifest.json"
    if not manifest_path.exists():
        concept_ids = sorted([name for name in entity2id if name.startswith("kc")], key=lambda x: int(x[2:]))
        exercise_ids = sorted([name for name in entity2id if name.startswith("ex")], key=lambda x: int(x[2:]))
        text_dir.mkdir(parents=True, exist_ok=True)
        np.save(text_dir / "concept_text_embeddings.npy", np.zeros((len(concept_ids), 1024), dtype=np.float32))
        np.save(text_dir / "exercise_text_embeddings.npy", np.zeros((len(exercise_ids), 1024), dtype=np.float32))
        write_json(
            manifest_path,
            {
                "embedding_dim": 1024,
                "model_name": "zero_fallback_v8_missing_text_embeddings",
                "concept_entity_ids": concept_ids,
                "exercise_entity_ids": exercise_ids,
                "files": {
                    "concept_text_embeddings": "concept_text_embeddings.npy",
                    "exercise_text_embeddings": "exercise_text_embeddings.npy",
                },
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V8 MIRT-derived feature JSON files for SemanticConvE.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--input-dir", type=Path, default=None, help="mirt_v8/inputs directory")
    parser.add_argument("--mirt-output-dir", type=Path, default=None)
    parser.add_argument("--mastery-file", type=Path, default=None)
    parser.add_argument("--output-feature-dir", type=Path, default=None)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = locate_dataset_dir(args.dataset, args.data_root)
    graph_dir = locate_graph_dir(dataset_dir)
    input_dir = args.input_dir or (dataset_dir / "mirt_v8" / "inputs")
    mirt_output_dir = args.mirt_output_dir or (dataset_dir / "mirt_v8" / "outputs")
    output_feature_dir = args.output_feature_dir or (dataset_dir / "semantic_kg_features_v8")
    if output_feature_dir.exists() and any(output_feature_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_feature_dir} already exists. Use --force to overwrite V8 feature files.")

    manifest = read_json(mirt_output_dir / "mirt_export_manifest.json")
    latent_dim = int(manifest["latent_dim"])
    a_param = load_matrix(mirt_output_dir / f"a_param_{latent_dim}.csv")
    b_param = load_matrix(mirt_output_dir / f"b_param_{latent_dim}.csv")
    theta_param = load_matrix(mirt_output_dir / f"theta_param_{latent_dim}.csv")
    mastery_file = args.mastery_file or (dataset_dir / "kt_exports_v8" / "stu2know_mastery.json")
    mastery = read_json(mastery_file) if mastery_file.exists() else []
    item_counts, item_correct, user_counts, user_correct = sequence_stats(input_dir)
    exercise_features = build_exercise_features(a_param, b_param, item_counts, item_correct)
    fit_indices, alignment = graph_learner_fit_indices(args.dataset, dataset_dir, graph_dir, input_dir)
    aligned_theta = theta_param[fit_indices]
    aligned_mastery = align_mastery_for_graph_learners(mastery, fit_indices, theta_param.shape[0])
    aligned_user_counts = {idx: int(user_counts.get(fit_uid, 0)) for idx, fit_uid in enumerate(fit_indices)}
    aligned_user_correct = {idx: float(user_correct.get(fit_uid, 0.0)) for idx, fit_uid in enumerate(fit_indices)}
    learner_features = build_learner_features(
        aligned_theta,
        aligned_mastery,
        aligned_user_correct,
        aligned_user_counts,
        n_clusters=args.n_clusters,
    )

    entity2id = read_entity_dict(graph_dir / "entities.dict")
    source_feature_dir = locate_semantic_feature_dir(dataset_dir, graph_dir)
    ensure_text_feature_files(source_feature_dir, output_feature_dir, entity2id)
    write_json(output_feature_dir / "entity_features" / "learner_pedagogy.json", {"dataset": args.dataset, "learners": learner_features})
    write_json(output_feature_dir / "irt_features" / "exercise_irt_features.json", {"dataset": args.dataset, "exercises": exercise_features})
    write_json(
        output_feature_dir / "feature_generation_manifest.json",
        {
            "dataset": args.dataset,
            "feature_version": "v8_noq_mirt",
            "mirt_output_dir": mirt_output_dir,
            "mastery_file": mastery_file,
            "source_feature_dir": source_feature_dir,
            "output_feature_dir": output_feature_dir,
            "latent_dim": latent_dim,
            "exercise_count": len(exercise_features),
            "learner_count": len(learner_features),
            "mirt_learner_count": int(theta_param.shape[0]),
            "learner_alignment": alignment,
            "notes": "Difficulty/discrimination are exported from modified EduCDM no-Q MIRT, not custom 2PL.",
        },
    )
    print(f"saved V8 semantic feature files: {output_feature_dir}")


if __name__ == "__main__":
    main()
