import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np


SUPPORTED_BASELINES = ["EB-CF", "SB-CF", "CBF", "KCP-ER"]


def load_json_matrix(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_q_matrix(path):
    q_matrix = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                q_matrix.append([int(value) for value in line.strip().split(",")])
    return q_matrix


def parse_int_list(text):
    if text is None:
        return []
    return [int(value) for value in str(text).split(",") if value and int(value) >= 0]


def load_sequence_interactions(sequence_file):
    interactions = {}
    if not sequence_file:
        return interactions
    path = Path(sequence_file)
    if not path.exists():
        return interactions

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            uid = f"uid{row['uid']}" if not str(row["uid"]).startswith("uid") else str(row["uid"])
            if "questions" in row and "responses" in row:
                questions = parse_int_list(row.get("questions"))
                responses = parse_int_list(row.get("responses"))
            else:
                questions = parse_int_list(row.get("question"))
                responses = parse_int_list(row.get("response"))
            pairs = []
            for question, response in zip(questions, responses):
                if response in (0, 1):
                    pairs.append((question, response))
            interactions.setdefault(uid, []).extend(pairs)
    return interactions


def _cosine(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _mean(values, default=0.0):
    values = [value for value in values if value is not None]
    if not values:
        return default
    return float(sum(values) / len(values))


def normalize_scores(scores):
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def exercise_concepts(q_matrix, exercise_idx):
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def content_based_scores(q_matrix, mastery, sequence=None, forgetting=None, sequence_weight=0.2, forgetting_weight=0.2):
    sequence = sequence or [0.0] * len(mastery)
    forgetting = forgetting or [0.0] * len(q_matrix)
    scores = []
    for exercise_idx in range(len(q_matrix)):
        concepts = exercise_concepts(q_matrix, exercise_idx)
        if not concepts:
            scores.append(0.0)
            continue
        weak_score = _mean([1.0 - float(mastery[kc]) for kc in concepts])
        progress_score = _mean([float(sequence[kc]) for kc in concepts])
        score = weak_score + sequence_weight * progress_score + forgetting_weight * float(forgetting[exercise_idx])
        scores.append(score)
    return normalize_scores(scores)


def kcp_er_scores(q_matrix, mastery, forgetting=None, target_mastery=0.8, forgetting_weight=0.2):
    forgetting = forgetting or [0.0] * len(q_matrix)
    scores = []
    for exercise_idx in range(len(q_matrix)):
        concepts = exercise_concepts(q_matrix, exercise_idx)
        if not concepts:
            scores.append(0.0)
            continue
        concept_mastery = _mean([float(mastery[kc]) for kc in concepts])
        target_fit = 1.0 - abs(float(target_mastery) - concept_mastery)
        score = target_fit + forgetting_weight * float(forgetting[exercise_idx])
        scores.append(score)
    return normalize_scores(scores)


def _exercise_popularity(interactions, exercise_count):
    positives = np.zeros(exercise_count, dtype=float)
    totals = np.zeros(exercise_count, dtype=float)
    for pairs in interactions.values():
        for exercise_idx, response in pairs:
            if 0 <= exercise_idx < exercise_count:
                totals[exercise_idx] += 1
                positives[exercise_idx] += 1 if response == 1 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        popularity = np.divide(positives, totals, out=np.zeros_like(positives), where=totals > 0)
    return popularity.tolist()


def exercise_based_cf_scores(q_matrix, interactions, user_ids=None):
    exercise_count = len(q_matrix)
    user_ids = user_ids or sorted(interactions.keys())
    popularity = _exercise_popularity(interactions, exercise_count)
    q_array = _row_normalize(q_matrix)
    popularity_array = np.asarray(popularity, dtype=float)
    uid_ex_scores = []
    for uid in user_ids:
        positives = [exercise_idx for exercise_idx, response in interactions.get(uid, []) if response == 1]
        positives = [idx for idx in positives if 0 <= idx < exercise_count]
        if positives:
            profile = np.mean(q_array[positives], axis=0)
            scores = 0.8 * np.matmul(q_array, profile) + 0.2 * popularity_array
        else:
            scores = popularity_array
        uid_ex_scores.append((uid, normalize_scores(scores.tolist())))
    return uid_ex_scores


def student_based_cf_scores(q_matrix, mastery, interactions, user_ids=None, neighbor_count=20):
    exercise_count = len(q_matrix)
    user_ids = user_ids or [f"uid{i}" for i in range(len(mastery))]
    mastery_by_uid = {uid: np.asarray(mastery[idx], dtype=float) for idx, uid in enumerate(user_ids)}
    popularity = _exercise_popularity(interactions, exercise_count)
    uid_ex_scores = []

    for uid in user_ids:
        source_mastery = mastery_by_uid[uid]
        neighbors = []
        for other_uid in user_ids:
            if other_uid == uid:
                continue
            sim = _cosine(source_mastery, mastery_by_uid[other_uid])
            neighbors.append((other_uid, sim))
        neighbors = sorted(neighbors, key=lambda item: item[1], reverse=True)[:neighbor_count]

        numerator = np.zeros(exercise_count, dtype=float)
        denominator = np.zeros(exercise_count, dtype=float)
        for other_uid, sim in neighbors:
            if sim <= 0:
                continue
            for exercise_idx, response in interactions.get(other_uid, []):
                if 0 <= exercise_idx < exercise_count:
                    numerator[exercise_idx] += sim * response
                    denominator[exercise_idx] += abs(sim)
        neighbor_scores = numerator
        scores = neighbor_scores + 0.2 * np.asarray(popularity)
        uid_ex_scores.append((uid, normalize_scores(scores.tolist())))
    return uid_ex_scores


def build_all_baseline_scores(
    q_matrix,
    mastery,
    sequence,
    forgetting,
    interactions=None,
    methods=None,
    user_ids=None,
):
    methods = methods or SUPPORTED_BASELINES
    interactions = interactions or {}
    user_ids = user_ids or [f"uid{i}" for i in range(len(mastery))]
    all_scores = {}

    for method in methods:
        if method == "EB-CF":
            all_scores[method] = exercise_based_cf_scores(q_matrix, interactions, user_ids=user_ids)
        elif method == "SB-CF":
            all_scores[method] = student_based_cf_scores(q_matrix, mastery, interactions, user_ids=user_ids)
        elif method == "CBF":
            all_scores[method] = [
                (
                    uid,
                    content_based_scores(
                        q_matrix,
                        mastery[idx],
                        sequence[idx] if sequence else None,
                        forgetting[idx] if forgetting else None,
                    ),
                )
                for idx, uid in enumerate(user_ids)
            ]
        elif method == "KCP-ER":
            all_scores[method] = [
                (
                    uid,
                    kcp_er_scores(
                        q_matrix,
                        mastery[idx],
                        forgetting[idx] if forgetting else None,
                    ),
                )
                for idx, uid in enumerate(user_ids)
            ]
        else:
            raise ValueError(f"Unsupported baseline: {method}")
    return all_scores


def save_uid_ex_scores(uid_ex_scores, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        with path.open("wb") as fp:
            pickle.dump(uid_ex_scores, fp)
    else:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(uid_ex_scores, fp, ensure_ascii=False)
