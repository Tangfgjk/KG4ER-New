import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mirt_inputs import build_mirt_frames, normalise_interactions
from mirt_feature_export import build_exercise_features, build_learner_features, minmax


def test_normalise_interactions_maps_raw_ids_to_dense_ids():
    df = pd.DataFrame(
        {
            "uid": ["stu_b", "stu_a", "stu_b"],
            "question": [20, 10, 10],
            "response": [1, 0, 1],
            "source_split": ["train", "test", "train"],
        }
    )

    normalised, maps = normalise_interactions(df)

    assert normalised[["user_id", "item_id", "score"]].to_dict("records") == [
        {"user_id": 1, "item_id": 1, "score": 1.0},
        {"user_id": 0, "item_id": 0, "score": 0.0},
        {"user_id": 1, "item_id": 0, "score": 1.0},
    ]
    assert maps["user_id_to_raw"] == {"0": "stu_a", "1": "stu_b"}
    assert maps["item_id_to_raw"] == {"0": "10", "1": "20"}


def test_build_mirt_frames_uses_all_strategy_for_train_when_requested():
    df = pd.DataFrame(
        {
            "uid": [0, 0, 1, 1],
            "question": [0, 1, 0, 1],
            "response": [1, 0, 1, 1],
            "source_split": ["train", "train", "test", "test"],
        }
    )

    normalised, _ = normalise_interactions(df)
    frames = build_mirt_frames(normalised, strategy="all")

    assert len(frames["train"]) == 4
    assert len(frames["valid"]) == 2
    assert len(frames["test"]) == 2


def test_minmax_returns_default_for_constant_values():
    values = minmax(np.array([3.0, 3.0, 3.0]), default=0.5)
    assert values.tolist() == [0.5, 0.5, 0.5]


def test_build_exercise_features_uses_b_as_difficulty_and_l2_a_as_discrimination():
    a = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=float)
    b = np.array([[1.0], [3.0]], dtype=float)

    features = build_exercise_features(a, b, interaction_counts={0: 12, 1: 3}, correct_rates={0: 0.25, 1: 0.75})

    assert set(features) == {"ex0", "ex1"}
    assert features["ex0"]["difficulty_mirt"] == 1.0
    assert features["ex1"]["difficulty_mirt"] == 3.0
    assert features["ex0"]["discrimination_mirt"] == 5.0
    assert features["ex1"]["discrimination_mirt"] == 2.0
    assert 0.0 <= features["ex0"]["difficulty_norm"] <= 1.0
    assert 0.0 <= features["ex0"]["discrimination_norm"] <= 1.0


def test_build_learner_features_adds_cluster_and_mastery_proxy():
    theta = np.array([[1.0, 2.0], [3.0, 5.0], [-1.0, -2.0]], dtype=float)
    mastery = [[0.7, 0.9], [0.2, 0.4], [0.5, 0.5]]

    features = build_learner_features(theta, mastery, correct_rates={0: 0.8, 1: 0.3, 2: 0.5}, history_lengths={0: 10, 1: 4, 2: 6}, n_clusters=2)

    assert set(features) == {"uid0", "uid1", "uid2"}
    assert all("cluster_id" in item for item in features.values())
    assert features["uid0"]["overall_mastery_kt_mean"] == 0.8
    assert 0.0 <= features["uid0"]["theta_norm"] <= 1.0
