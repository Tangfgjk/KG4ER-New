from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    ensure_large_csv_field_limit,
    entity_count,
    locate_source_graph_dir,
    locate_test_sequences,
    output_data_root,
    read_entity_dict,
    read_q_matrix,
    source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
    write_matrix_json,
)
from prepare_mirt_inputs_v11 import eedi_valid_question_length
from build_v11_graph import exercise_forgetting_average


def cal_forget(t: float, t_prime: float, theta: float) -> float:
    return 1.0 - float(np.exp(-abs(t - t_prime) / abs(theta)))


def normalize_numeric_timestamp(value: float, unit: str) -> float:
    if unit == "milliseconds":
        return value / 1000.0
    if unit == "minutes":
        return value * 60.0
    if unit == "days":
        return value * 86400.0
    if unit == "seconds":
        return value
    if unit != "auto":
        raise ValueError(f"unknown timestamp unit: {unit}")
    abs_value = abs(value)
    # Unix timestamps in milliseconds are usually around 1e12+.
    # Seconds are around 1e9+. Smaller elapsed-time counters are kept as seconds.
    if abs_value >= 1e12:
        return value / 1000.0
    return value


def parse_timestamp_tokens(value: Any, unit: str = "auto") -> list[float]:
    timestamps: list[float] = []
    for token in str(value).split(","):
        if not token or token == "-1":
            continue
        try:
            timestamps.append(float(normalize_numeric_timestamp(float(token), unit)))
            continue
        except ValueError:
            pass
        parsed = pd.to_datetime(token, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"unrecognized timestamp token: {token}")
        timestamps.append(float(parsed.timestamp()))
    return timestamps


def parse_concept_steps(value: Any) -> list[list[int]]:
    steps: list[list[int]] = []
    for token in str(value).split(","):
        if not token or token == "-1":
            continue
        steps.append([int(i) for i in token.split("_") if i and i != "-1"])
    return steps


def load_aligned_test_rows(dataset: str, source_dir: Path, graph_dir: Path, learner_count: int) -> list[dict[str, Any]]:
    test_sequences = locate_test_sequences(dataset, source_dir, graph_dir)
    ensure_large_csv_field_limit()
    if dataset == "Eedi":
        rows: list[dict[str, Any]] = []
        with test_sequences.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                valid_len = eedi_valid_question_length(row)
                if 10 <= valid_len <= 198:
                    rows.append(dict(row))
        rows = rows[:learner_count]
    else:
        rows = pd.read_csv(test_sequences, low_memory=False).head(learner_count).to_dict("records")
    if len(rows) != learner_count:
        raise ValueError(f"{dataset} aligned test rows={len(rows)} != graph learners={learner_count}")
    return rows


def build_knowledge_forgetting(
    rows: list[dict[str, Any]],
    concept_count: int,
    theta: float,
    timestamp_unit: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    all_students: list[list[float]] = []
    positive_deltas: list[float] = []
    for row in rows:
        timestamps = parse_timestamp_tokens(row.get("timestamps", ""), unit=timestamp_unit)
        concept_steps = parse_concept_steps(row.get("concepts", ""))
        usable = min(len(timestamps), len(concept_steps))
        timestamps = timestamps[:usable]
        concept_steps = concept_steps[:usable]
        if usable == 0:
            all_students.append([1.0] * concept_count)
            continue
        last_timestamp = timestamps[-1]
        student_forget: list[float] = []
        for concept_id in range(concept_count):
            value = 1.0
            for j in range(usable - 1, -1, -1):
                if concept_id in concept_steps[j]:
                    delta = abs(float(last_timestamp) - float(timestamps[j]))
                    positive_deltas.append(delta)
                    value = cal_forget(last_timestamp, timestamps[j], theta)
                    break
            student_forget.append(round(float(max(0.0, min(1.0, value))), 6))
        all_students.append(student_forget)
    delta_arr = np.asarray(positive_deltas, dtype=np.float64)
    delta_stats = {
        "timestamp_unit_normalized_to": "seconds",
        "raw_timestamp_unit": timestamp_unit,
        "theta_seconds": float(theta),
        "positive_delta_count": int(delta_arr.size),
        "median_delta_seconds": float(np.median(delta_arr)) if delta_arr.size else None,
        "p90_delta_seconds": float(np.percentile(delta_arr, 90)) if delta_arr.size else None,
        "max_delta_seconds": float(np.max(delta_arr)) if delta_arr.size else None,
    }
    return np.asarray(all_students, dtype=np.float64), delta_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate V11 forgetting files with exercise-level average aggregation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--theta", type=float, default=10000000, help="Forgetting decay denominator in seconds.")
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "seconds", "milliseconds", "minutes", "days"],
        default="auto",
        help="Numeric timestamp unit. String dates are always parsed into Unix seconds.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    graph_dir = locate_source_graph_dir(source_dir)
    output_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    entity2id = read_entity_dict(graph_dir / "entities.dict")
    learner_count = entity_count(entity2id, "uid")
    q_matrix = read_q_matrix(graph_dir / "Q.txt")
    rows = load_aligned_test_rows(args.dataset, source_dir, graph_dir, learner_count)
    know_forget, delta_stats = build_knowledge_forgetting(
        rows,
        q_matrix.shape[1],
        theta=args.theta,
        timestamp_unit=args.timestamp_unit,
    )
    ex_forget = exercise_forgetting_average(know_forget, q_matrix)
    forget_dir = output_dir / "forgetting"
    write_matrix_json(forget_dir / "stu2know_forget.json", know_forget, decimals=6)
    write_matrix_json(forget_dir / "stu2ex_forget.json", ex_forget, decimals=6)
    (output_dir / "stu2know_forget.json").write_text((forget_dir / "stu2know_forget.json").read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "stu2ex_forget.json").write_text((forget_dir / "stu2ex_forget.json").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        forget_dir / "forgetting_manifest.json",
        {
            "dataset": args.dataset,
            "source_graph_dir": graph_dir,
            "theta": args.theta,
            "timestamp": delta_stats,
            "student_count": int(know_forget.shape[0]),
            "concept_count": int(know_forget.shape[1]),
            "exercise_count": int(ex_forget.shape[1]),
            "knowledge_forgetting": "time-decay value; unseen concepts default to 1.0",
            "exercise_forgetting": "average of linked concept forgetting values, not sum and not post-hoc clipping",
            "range": {
                "stu2know_forget_min": float(np.min(know_forget)),
                "stu2know_forget_max": float(np.max(know_forget)),
                "stu2ex_forget_min": float(np.min(ex_forget)),
                "stu2ex_forget_max": float(np.max(ex_forget)),
            },
        },
    )
    print(f"saved V11 forgetting files: {forget_dir}")


if __name__ == "__main__":
    main()
