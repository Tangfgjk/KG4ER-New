"""Build statistical exercise pedagogical features for SemanticConvE.

The script estimates exercise difficulty and discrimination from historical
response logs. It writes:

    semantic_kg_features/stat_features/exercise_stat_features.json

The feature loader prefers this file over the older IRT exercise file when it
is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from semantic_experiment_utils import DEFAULT_DATASETS, default_data_root, graph_path_for_dataset, parse_csv_list


csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

USER_COLUMNS = ["uid", "user_id", "student_id", "student", "learner_id", "original_uid"]
EXERCISE_COLUMNS = ["question", "ex", "exercise", "exercise_id", "question_id", "item_id", "problem_id"]
CORRECT_COLUMNS = ["response", "correct", "is_correct", "answer", "label"]
SPLIT_COLUMNS = ["source_split", "split", "data_split"]


def read_dict(path: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split("\t")
            result[name] = int(idx)
    return result


def entity_kind(entity_name: str) -> str:
    if entity_name.startswith("uid"):
        return "uid"
    if entity_name.startswith("kc"):
        return "kc"
    if entity_name.startswith("ex"):
        return "ex"
    return "other"


def find_first_column(fieldnames: Iterable[str], candidates: List[str]) -> str | None:
    normalized = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def normalize_entity(prefix: str, value: Any) -> str:
    text = str(value).strip()
    if text.startswith(prefix):
        return text
    try:
        number = int(float(text))
        return f"{prefix}{number}"
    except Exception:
        return f"{prefix}{text}"


def parse_binary(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "correct", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "incorrect", "no"}:
        return 0
    try:
        number = float(text)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return 1 if number >= 0.5 else 0


def candidate_sequence_files(graph_path: Path) -> List[Path]:
    roots = []
    for root in [graph_path, graph_path.parent, graph_path.parent / "processed", graph_path.parent / "prepared_for_kt"]:
        if root not in roots:
            roots.append(root)
    candidates: List[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "sequence_interactions.csv",
                root / "interactions.csv",
                root / "responses.csv",
            ]
        )
    return [path for path in candidates if path.exists()]


def is_training_split(split_value: str | None) -> bool:
    if split_value is None:
        return True
    text = split_value.strip().lower()
    if not text:
        return True
    return "test" not in text


def iter_interactions(path: Path, train_only: bool) -> Iterator[Tuple[str, str, int]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return
        user_col = find_first_column(reader.fieldnames, USER_COLUMNS)
        exercise_col = find_first_column(reader.fieldnames, EXERCISE_COLUMNS)
        correct_col = find_first_column(reader.fieldnames, CORRECT_COLUMNS)
        split_col = find_first_column(reader.fieldnames, SPLIT_COLUMNS)
        if not user_col or not exercise_col or not correct_col:
            raise ValueError(
                f"Cannot parse {path}; need user/exercise/correct columns, got {reader.fieldnames}"
            )
        for row in reader:
            if train_only and split_col and not is_training_split(row.get(split_col)):
                continue
            correct = parse_binary(row.get(correct_col))
            if correct is None:
                continue
            uid = str(row.get(user_col, "")).strip()
            ex = normalize_entity("ex", row.get(exercise_col, ""))
            if not uid or not ex:
                continue
            yield uid, ex, correct


def collect_counts(path: Path, train_only: bool) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], int]:
    student_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    exercise_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    row_count = 0
    for uid, ex, correct in iter_interactions(path, train_only=train_only):
        student_counts[uid][0] += 1
        student_counts[uid][1] += correct
        exercise_counts[ex][0] += 1
        exercise_counts[ex][1] += correct
        row_count += 1
    return student_counts, exercise_counts, row_count


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def collect_group_counts(
    path: Path,
    train_only: bool,
    high_users: set[str],
    low_users: set[str],
) -> Dict[str, List[int]]:
    group_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for uid, ex, correct in iter_interactions(path, train_only=train_only):
        if uid in high_users:
            group_counts[ex][0] += 1
            group_counts[ex][1] += correct
        if uid in low_users:
            group_counts[ex][2] += 1
            group_counts[ex][3] += correct
    return group_counts


def build_dataset_features(
    dataset: str,
    data_root: Path,
    alpha: float,
    include_test_interactions: bool,
    force: bool,
) -> Dict[str, Any]:
    graph_path = graph_path_for_dataset(dataset, data_root)
    sequence_files = candidate_sequence_files(graph_path)
    if not sequence_files:
        raise FileNotFoundError(f"No sequence_interactions.csv found for {dataset} under {graph_path}")
    sequence_path = sequence_files[0]
    train_only = not include_test_interactions

    student_counts, exercise_counts, used_rows = collect_counts(sequence_path, train_only=train_only)
    if used_rows == 0 and train_only:
        train_only = False
        student_counts, exercise_counts, used_rows = collect_counts(sequence_path, train_only=False)
    if used_rows == 0:
        raise ValueError(f"No usable response rows found in {sequence_path}")

    student_rates = {
        uid: counts[1] / counts[0]
        for uid, counts in student_counts.items()
        if counts[0] > 0
    }
    low_threshold = quantile(list(student_rates.values()), 0.27)
    high_threshold = quantile(list(student_rates.values()), 0.73)
    low_users = {uid for uid, rate in student_rates.items() if rate <= low_threshold}
    high_users = {uid for uid, rate in student_rates.items() if rate >= high_threshold}
    group_counts = collect_group_counts(sequence_path, train_only=train_only, high_users=high_users, low_users=low_users)

    entity2id = read_dict(graph_path / "entities.dict")
    exercise_entities = sorted(name for name in entity2id if entity_kind(name) == "ex")
    global_correct = sum(correct for _, correct in exercise_counts.values())
    global_count = sum(count for count, _ in exercise_counts.values())
    global_correct_rate = global_correct / global_count if global_count else 0.0
    global_error_rate = 1.0 - global_correct_rate

    exercises: Dict[str, Dict[str, Any]] = {}
    missing_count = 0
    for ex in exercise_entities:
        count, correct = exercise_counts.get(ex, [0, 0])
        if count == 0:
            missing_count += 1
        wrong = count - correct
        correct_rate = correct / count if count else global_correct_rate
        error_rate = 1.0 - correct_rate
        difficulty_stat = (wrong + alpha * global_error_rate) / (count + alpha) if count + alpha > 0 else global_error_rate
        high_count, high_correct, low_count, low_correct = group_counts.get(ex, [0, 0, 0, 0])
        high_rate = high_correct / high_count if high_count else global_correct_rate
        low_rate = low_correct / low_count if low_count else global_correct_rate
        discrimination = max(0.0, min(1.0, high_rate - low_rate))
        exercises[ex] = {
            "entity_id": ex,
            "entity_type": "exercise",
            "difficulty_stat": difficulty_stat,
            "difficulty_stat_norm": max(0.0, min(1.0, difficulty_stat)),
            "discrimination_stat": discrimination,
            "discrimination_stat_norm": discrimination,
            "correct_rate": correct_rate,
            "error_rate": error_rate,
            "interaction_count": count,
            "high_group_correct_rate": high_rate,
            "high_group_count": high_count,
            "low_group_correct_rate": low_rate,
            "low_group_count": low_count,
            "feature_source": "statistical_response_features",
        }

    output_dir = graph_path / "semantic_kg_features" / "stat_features"
    output_path = output_dir / "exercise_stat_features.json"
    if output_path.exists() and not force:
        status = "skipped_existing"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset,
            "graph_path": str(graph_path),
            "sequence_interactions": str(sequence_path),
            "train_only": train_only,
            "alpha": alpha,
            "student_count": len(student_counts),
            "exercise_count": len(exercise_entities),
            "used_interaction_count": used_rows,
            "global_correct_rate": global_correct_rate,
            "global_error_rate": global_error_rate,
            "low_group_threshold": low_threshold,
            "high_group_threshold": high_threshold,
            "low_group_user_count": len(low_users),
            "high_group_user_count": len(high_users),
            "missing_exercise_interaction_count": missing_count,
            "exercises": exercises,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "written"

    return {
        "dataset": dataset,
        "graph_path": str(graph_path),
        "output_file": str(output_path),
        "status": status,
        "sequence_interactions": str(sequence_path),
        "train_only": train_only,
        "used_interaction_count": used_rows,
        "student_count": len(student_counts),
        "exercise_count": len(exercise_entities),
        "missing_exercise_interaction_count": missing_count,
        "global_correct_rate": global_correct_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build statistical exercise difficulty/discrimination features.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--alpha", type=float, default=10.0, help="Smoothing strength for exercise difficulty.")
    parser.add_argument("--include-test-interactions", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        build_dataset_features(
            dataset=dataset,
            data_root=args.data_root,
            alpha=args.alpha,
            include_test_interactions=args.include_test_interactions,
            force=args.force,
        )
        for dataset in parse_csv_list(args.datasets)
    ]
    summary = {"data_root": str(args.data_root), "reports": reports}
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
