from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    copy_file,
    cosine_similarity,
    load_matrix_json,
    locate_source_graph_dir,
    matrix_stats,
    output_data_root,
    read_entity_dict,
    read_q_matrix,
    read_triple_students,
    relation_label,
    source_data_root,
    source_dataset_dir,
    sorted_entities,
    v13_dataset_dir,
    write_fixed_relations,
    write_json,
    write_matrix_json,
)


def exercise_forgetting_average(stu2know_forget: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    if stu2know_forget.shape[1] != q_matrix.shape[1]:
        raise ValueError(
            "stu2know_forget and Q concept dimensions do not match: "
            f"{stu2know_forget.shape[1]} != {q_matrix.shape[1]}"
        )
    weights = q_matrix.astype(np.float64)
    counts = weights.sum(axis=1)
    sums = stu2know_forget @ weights.T
    out = np.zeros_like(sums, dtype=np.float64)
    np.divide(sums, counts[None, :], out=out, where=counts[None, :] > 0)
    return np.clip(out, 0.0, 1.0)


def mastery_product(mastery_row: np.ndarray, q_row: np.ndarray) -> float:
    concepts = np.where(q_row > 0)[0]
    if len(concepts) == 0:
        return 0.0
    return float(np.prod(mastery_row[concepts]))


def recommendation_distance(
    mastery: np.ndarray,
    sequence: np.ndarray,
    exercise_forget: np.ndarray,
    q_matrix: np.ndarray,
    delta_1: float,
    delta_2: float,
    sequence_term: str,
) -> np.ndarray:
    scores = np.zeros((mastery.shape[0], q_matrix.shape[0]), dtype=np.float64)
    for uid in range(mastery.shape[0]):
        seq_row = sequence[uid]
        for ex_id in range(q_matrix.shape[0]):
            q_row = q_matrix[ex_id].astype(np.float64)
            term_mastery = (float(delta_1) - mastery_product(mastery[uid], q_row)) ** 2
            cos = cosine_similarity(q_row, seq_row)
            if sequence_term == "one_minus_cos_sq":
                term_seq = (1.0 - cos) ** 2
            elif sequence_term == "legacy_cos_sq":
                term_seq = cos**2
            else:
                raise ValueError(f"unknown sequence_term: {sequence_term}")
            term_forget = (float(delta_2) - float(exercise_forget[uid, ex_id])) ** 2
            scores[uid, ex_id] = float(np.sqrt(term_mastery + term_seq + term_forget))
    return scores


def split_students_from_source(source_graph_dir: Path, learner_count: int, train_ratio: float, seed: int) -> tuple[list[int], list[int], str]:
    train_students = read_triple_students(source_graph_dir / "triples.txt")
    test_students = read_triple_students(source_graph_dir / "test_triples.txt")
    if train_students and test_students:
        return sorted(train_students), sorted(test_students), "source_triples"
    rng = np.random.default_rng(seed)
    ids = np.arange(learner_count)
    rng.shuffle(ids)
    n_train = int(learner_count * train_ratio)
    return sorted(ids[:n_train].tolist()), sorted(ids[n_train:].tolist()), "deterministic_random_split"


def write_triples(
    path: Path,
    students: list[int],
    mastery: np.ndarray,
    sequence: np.ndarray,
    exercise_forget: np.ndarray,
    q_matrix: np.ndarray,
    rec_scores: np.ndarray,
    include_rec: bool,
    top_k_rec: int,
) -> dict[str, int]:
    counts = {"mlkc": 0, "pkc": 0, "exfr": 0, "rec": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for uid in students:
            for kc in range(mastery.shape[1]):
                fp.write(f"kc{kc}\t{relation_label('mlkc', mastery[uid, kc])}\tuid{uid}\n")
                fp.write(f"kc{kc}\t{relation_label('pkc', sequence[uid, kc])}\tuid{uid}\n")
                counts["mlkc"] += 1
                counts["pkc"] += 1
            for ex_id in range(q_matrix.shape[0]):
                fp.write(f"ex{ex_id}\t{relation_label('exfr', exercise_forget[uid, ex_id])}\tuid{uid}\n")
                counts["exfr"] += 1
            if include_rec:
                ranked = sorted(range(q_matrix.shape[0]), key=lambda ex: (float(rec_scores[uid, ex]), ex))[:top_k_rec]
                for ex_id in ranked:
                    fp.write(f"uid{uid}\trec\tex{ex_id}\n")
                    counts["rec"] += 1
    return counts


def copy_optional_evaluation_files(source_graph_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for src in source_graph_dir.glob("*_uid_kc_response.txt"):
        copy_file(src, output_dir / src.name)
        copied.append(src.name)
    return copied


def require_or_copy_matrix(
    name: str,
    output_dir: Path,
    source_graph_dir: Path,
    strict_file: Path | None,
    allow_existing_state: bool,
) -> tuple[Path, str]:
    target = output_dir / name
    if strict_file is not None:
        copy_file(strict_file, target)
        return target, str(strict_file)
    if target.exists():
        return target, str(target)
    if allow_existing_state:
        src = source_graph_dir / name
        copy_file(src, target)
        return target, f"copied_existing_state:{src}"
    raise FileNotFoundError(
        f"Missing {target}. Provide --{name.replace('.json', '').replace('_', '-')}-file "
        f"or generate it under the v13 directory first. Use --allow-existing-state only for debugging."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v13 ER graph files from regenerated cognitive-state files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--mastery-file", type=Path, default=None)
    parser.add_argument("--seq-file", type=Path, default=None)
    parser.add_argument("--know-forget-file", type=Path, default=None)
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sequence-term", choices=["one_minus_cos_sq", "legacy_cos_sq"], default="one_minus_cos_sq")
    parser.add_argument("--allow-existing-state", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    source_graph_dir = locate_source_graph_dir(source_dir)
    output_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        # Permit incremental runs when only adding missing state files.
        pass
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_file(source_graph_dir / "entities.dict", output_dir / "entities.dict")
    copy_file(source_graph_dir / "Q.txt", output_dir / "Q.txt")
    copied_eval = copy_optional_evaluation_files(source_graph_dir, output_dir)
    write_fixed_relations(output_dir / "relations.dict")

    mastery_path, mastery_source = require_or_copy_matrix(
        "stu2know_mastery.json", output_dir, source_graph_dir, args.mastery_file, args.allow_existing_state
    )
    seq_path, seq_source = require_or_copy_matrix(
        "stu2know_seq.json", output_dir, source_graph_dir, args.seq_file, args.allow_existing_state
    )
    know_forget_path, forget_source = require_or_copy_matrix(
        "stu2know_forget.json", output_dir, source_graph_dir, args.know_forget_file, args.allow_existing_state
    )

    q_matrix = read_q_matrix(output_dir / "Q.txt")
    mastery = load_matrix_json(mastery_path)
    sequence = load_matrix_json(seq_path)
    know_forget = load_matrix_json(know_forget_path)
    if mastery.shape != sequence.shape or mastery.shape != know_forget.shape:
        raise ValueError(f"state matrix shapes mismatch: mastery={mastery.shape}, seq={sequence.shape}, forget={know_forget.shape}")
    if mastery.shape[1] != q_matrix.shape[1]:
        raise ValueError(f"state concept count {mastery.shape[1]} != Q concept count {q_matrix.shape[1]}")

    exercise_forget = exercise_forgetting_average(know_forget, q_matrix)
    write_matrix_json(output_dir / "stu2ex_forget.json", exercise_forget)
    rec_scores = recommendation_distance(
        mastery,
        sequence,
        exercise_forget,
        q_matrix,
        delta_1=args.delta_1,
        delta_2=args.delta_2,
        sequence_term=args.sequence_term,
    )
    write_json(output_dir / "stu2ex_recommend_full_precision.json", rec_scores.tolist())
    write_matrix_json(output_dir / "stu2ex_recommend.json", rec_scores, decimals=6)

    entity2id = read_entity_dict(output_dir / "entities.dict")
    learner_count = len(sorted_entities(entity2id, "uid"))
    train_students, test_students, split_source = split_students_from_source(
        source_graph_dir, learner_count, train_ratio=args.train_ratio, seed=args.seed
    )
    train_counts = write_triples(
        output_dir / "triples.txt",
        train_students,
        mastery,
        sequence,
        exercise_forget,
        q_matrix,
        rec_scores,
        include_rec=True,
        top_k_rec=args.top_k_rec,
    )
    test_counts = write_triples(
        output_dir / "test_triples.txt",
        test_students,
        mastery,
        sequence,
        exercise_forget,
        q_matrix,
        rec_scores,
        include_rec=False,
        top_k_rec=args.top_k_rec,
    )

    # Keep old-cos recommendation scores for controlled comparison without mixing them into the main v13 graph.
    old_cos_dir = output_dir.parent / "v13_rec_legacy_cos"
    if args.sequence_term == "one_minus_cos_sq":
        old_cos_dir.mkdir(parents=True, exist_ok=True)
        legacy_scores = recommendation_distance(
            mastery,
            sequence,
            exercise_forget,
            q_matrix,
            delta_1=args.delta_1,
            delta_2=args.delta_2,
            sequence_term="legacy_cos_sq",
        )
        write_json(old_cos_dir / "stu2ex_recommend_full_precision.json", legacy_scores.tolist())
        write_matrix_json(old_cos_dir / "stu2ex_recommend.json", legacy_scores, decimals=6)
        for file_name in [
            "entities.dict",
            "Q.txt",
            "relations.dict",
            "stu2know_mastery.json",
            "stu2know_seq.json",
            "stu2know_forget.json",
            "stu2ex_forget.json",
        ]:
            copy_file(output_dir / file_name, old_cos_dir / file_name)
        for src in output_dir.glob("*_uid_kc_response.txt"):
            copy_file(src, old_cos_dir / src.name)
        legacy_train_counts = write_triples(
            old_cos_dir / "triples.txt",
            train_students,
            mastery,
            sequence,
            exercise_forget,
            q_matrix,
            legacy_scores,
            include_rec=True,
            top_k_rec=args.top_k_rec,
        )
        legacy_test_counts = write_triples(
            old_cos_dir / "test_triples.txt",
            test_students,
            mastery,
            sequence,
            exercise_forget,
            q_matrix,
            legacy_scores,
            include_rec=False,
            top_k_rec=args.top_k_rec,
        )
        write_json(
            old_cos_dir / "v13_graph_manifest.json",
            {
                "dataset": args.dataset,
                "version": "v13_front_files_legacy_cos",
                "source_graph_dir": source_graph_dir,
                "output_dir": old_cos_dir,
                "base_v13_dir": output_dir,
                "state_sources": {
                    "stu2know_mastery": mastery_source,
                    "stu2know_seq": seq_source,
                    "stu2know_forget": forget_source,
                },
                "forgetting": {
                    "stu2ex_forget": "copied from v13 average exercise forgetting",
                    "range": matrix_stats(exercise_forget),
                },
                "recommendation": {
                    "sequence_term": "legacy_cos_sq",
                    "distance_formula": "sqrt((delta1 - mastery_product)^2 + cos(Q, seq)^2 + (delta2 - exercise_forget)^2)",
                    "ranking_precision": "full precision scores; stu2ex_recommend.json is rounded only for storage/readability",
                    "delta_1": args.delta_1,
                    "delta_2": args.delta_2,
                    "top_k_rec": args.top_k_rec,
                },
                "split": {
                    "source": split_source,
                    "train_students": len(train_students),
                    "test_students": len(test_students),
                },
                "relation_count": 304,
                "triple_counts": {"train": legacy_train_counts, "test": legacy_test_counts},
                "copied_evaluation_files": copied_eval,
            },
        )

    manifest = {
        "dataset": args.dataset,
        "version": "v13_front_files",
        "source_graph_dir": source_graph_dir,
        "output_dir": output_dir,
        "state_sources": {
            "stu2know_mastery": mastery_source,
            "stu2know_seq": seq_source,
            "stu2know_forget": forget_source,
        },
        "forgetting": {
            "stu2ex_forget": "average of linked knowledge-concept forgetting rates from stu2know_forget and Q.txt",
            "range": matrix_stats(exercise_forget),
        },
        "recommendation": {
            "sequence_term": args.sequence_term,
            "distance_formula": "sqrt((delta1 - mastery_product)^2 + sequence_term + (delta2 - exercise_forget)^2)",
            "ranking_precision": "full precision scores; stu2ex_recommend.json is rounded only for storage/readability",
            "delta_1": args.delta_1,
            "delta_2": args.delta_2,
            "top_k_rec": args.top_k_rec,
        },
        "split": {
            "source": split_source,
            "train_students": len(train_students),
            "test_students": len(test_students),
        },
        "relation_count": 304,
        "triple_counts": {"train": train_counts, "test": test_counts},
        "copied_evaluation_files": copied_eval,
    }
    write_json(output_dir / "v13_graph_manifest.json", manifest)
    print(f"created v13 graph files: {output_dir}")
    print(f"train_triples={sum(train_counts.values())} test_triples={sum(test_counts.values())} sequence_term={args.sequence_term}")


if __name__ == "__main__":
    main()

