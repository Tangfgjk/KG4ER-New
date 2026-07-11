from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v13_graph import exercise_forgetting_average, recommendation_distance
from prepare_mirt_inputs_v13 import normalise_filtered_interactions


def test_exercise_forgetting_uses_average_not_sum() -> None:
    know_forget = np.asarray([[0.2, 0.8, 1.0]], dtype=np.float64)
    q = np.asarray([[1, 1, 0], [0, 0, 1]], dtype=np.int64)
    ex_forget = exercise_forgetting_average(know_forget, q)
    assert ex_forget.tolist() == [[0.5, 1.0]]


def test_recommendation_sequence_term_prefers_high_cosine() -> None:
    mastery = np.asarray([[0.8, 0.8]], dtype=np.float64)
    sequence = np.asarray([[1.0, 0.0]], dtype=np.float64)
    ex_forget = np.asarray([[0.8, 0.8]], dtype=np.float64)
    q = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    scores = recommendation_distance(
        mastery,
        sequence,
        ex_forget,
        q,
        delta_1=0.8,
        delta_2=0.8,
        sequence_term="one_minus_cos_sq",
    )
    assert scores[0, 0] < scores[0, 1]


def test_mirt_input_keeps_graph_item_ids_uncompressed() -> None:
    df = pd.DataFrame(
        {
            "uid": [10, 10, 11],
            "question": [5, 9, 5],
            "response": [1, 0, 1],
        }
    )
    out, info = normalise_filtered_interactions(df, graph_uids=["10"], exercise_count=10)
    assert out["user_id"].tolist() == [0, 0]
    assert out["item_id"].tolist() == [5, 9]
    assert info["observed_item_count"] == 2
    assert info["missing_item_count"] == 8

