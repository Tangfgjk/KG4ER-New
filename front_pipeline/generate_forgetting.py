from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import data_fin_root, front_dir, load_raw_dataset, matrix_stats, normalise_timestamp, write_json, write_matrix_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw-data knowledge and exercise forgetting features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--theta", type=float, default=10_000_000.0, help="Decay denominator after timestamps are normalised to seconds.")
    parser.add_argument("--timestamp-unit", choices=["auto", "seconds", "milliseconds", "minutes", "days"], default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root)
    output = front_dir(args.dataset, root) / "forgetting"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "stu2know_forget.json"
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} exists. Use --force to regenerate.")
    if args.theta <= 0:
        raise ValueError("--theta must be positive")

    know_forget = np.ones((raw.student_count, raw.concept_count), dtype=np.float64)
    all_deltas: list[float] = []
    for uid in range(raw.student_count):
        frame = raw.interactions[raw.interactions["uid"] == uid]
        if frame.empty:
            continue
        times = np.asarray([normalise_timestamp(value, args.timestamp_unit) for value in frame["timestamp"]], dtype=np.float64)
        last_time = float(times[-1])
        last_seen = np.full(raw.concept_count, np.nan, dtype=np.float64)
        for item_id, timestamp in zip(frame["question"].to_numpy(dtype=np.int64), times):
            last_seen[raw.q_matrix[item_id] > 0] = timestamp
        observed = np.isfinite(last_seen)
        deltas = np.abs(last_time - last_seen[observed])
        all_deltas.extend(deltas.tolist())
        know_forget[uid, observed] = 1.0 - np.exp(-deltas / float(args.theta))
    know_forget = np.clip(know_forget, 0.0, 1.0)
    q_counts = raw.q_matrix.sum(axis=1).clip(min=1.0)
    exercise_forget = (know_forget @ raw.q_matrix.T) / q_counts[None, :]
    exercise_forget = np.clip(exercise_forget, 0.0, 1.0)
    write_matrix_json(output / "stu2know_forget.json", know_forget)
    write_matrix_json(output / "stu2ex_forget.json", exercise_forget)
    write_matrix_json(front_dir(args.dataset, root) / "stu2know_forget.json", know_forget)
    write_matrix_json(front_dir(args.dataset, root) / "stu2ex_forget.json", exercise_forget)
    delta_arr = np.asarray(all_deltas, dtype=np.float64)
    write_json(
        output / "forgetting_manifest.json",
        {
            "dataset": raw.name,
            "formula": "1 - exp(-elapsed_time / theta)",
            "theta_seconds": args.theta,
            "timestamp_unit": args.timestamp_unit,
            "unseen_concept_value": 1.0,
            "exercise_aggregation": "mean forgetting over Q-linked concepts",
            "positive_delta_seconds": matrix_stats(delta_arr),
            "knowledge_forgetting": matrix_stats(know_forget),
            "exercise_forgetting": matrix_stats(exercise_forget),
        },
    )
    print(f"saved raw forgetting outputs: {output}")


if __name__ == "__main__":
    main()
