from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import (
    copy_file,
    data_fin_root,
    front_dir,
    graph_dir,
    load_raw_dataset,
    matrix_stats,
    read_matrix_json,
    relation_label,
    write_json,
    write_matrix_json,
    write_relations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KG4ER graph files from Data_Fin/raw and regenerated front features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="raw")
    parser.add_argument(
        "--graph-name",
        default="er_graph",
        help="Output graph folder name under Data_Fin/<dataset>. Use er_graph_with_seq for KGE baselines with pkc edges.",
    )
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        help="Include pkc sequence/progress relations and the sequence term in recommendation-distance construction.",
    )
    parser.add_argument(
        "--sequence-term",
        choices=["one_minus_cos_sq", "cos_sq"],
        default="one_minus_cos_sq",
        help="Sequence distance term used only with --include-sequence.",
    )
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mastery_product(mastery: np.ndarray, q_row: np.ndarray) -> float:
    concepts = np.flatnonzero(q_row > 0)
    return float(np.prod(mastery[concepts])) if len(concepts) else 0.0


def sequence_distance(q_row: np.ndarray, sequence_row: np.ndarray, sequence_term: str) -> float:
    denominator = float(np.linalg.norm(q_row) * np.linalg.norm(sequence_row))
    cosine = 0.0 if denominator <= 1e-12 else float(np.dot(q_row, sequence_row.T) / denominator)
    if sequence_term == "cos_sq":
        return cosine**2
    return (1.0 - cosine) ** 2


def recommendation_distances(
    mastery: np.ndarray,
    exercise_forget: np.ndarray,
    q: np.ndarray,
    delta_1: float,
    delta_2: float,
    sequence: np.ndarray | None = None,
    sequence_term: str = "one_minus_cos_sq",
) -> np.ndarray:
    result = np.zeros((mastery.shape[0], q.shape[0]), dtype=np.float64)
    for uid in range(mastery.shape[0]):
        for ex in range(q.shape[0]):
            term_mastery = (float(delta_1) - mastery_product(mastery[uid], q[ex])) ** 2
            term_forget = (float(delta_2) - float(exercise_forget[uid, ex])) ** 2
            term_sequence = sequence_distance(q[ex], sequence[uid], sequence_term) if sequence is not None else 0.0
            result[uid, ex] = np.sqrt(term_mastery + term_sequence + term_forget)
    return result


def write_entities(path: Path, student_count: int, concept_count: int, exercise_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [*(f"uid{uid}" for uid in range(student_count)), *(f"kc{kc}" for kc in range(concept_count)), *(f"ex{ex}" for ex in range(exercise_count))]
    with path.open("w", encoding="utf-8") as fp:
        for index, name in enumerate(names):
            fp.write(f"{index}\t{name}\n")


def write_triples(
    path: Path,
    users: list[int],
    mastery: np.ndarray,
    ex_forget: np.ndarray,
    q: np.ndarray,
    distances: np.ndarray,
    top_k_rec: int,
    include_rec: bool,
    sequence: np.ndarray | None = None,
) -> dict[str, int]:
    counts = {"mlkc": 0, "pkc": 0, "exfr": 0, "rec": 0}
    with path.open("w", encoding="utf-8") as fp:
        for uid in users:
            for kc in range(mastery.shape[1]):
                fp.write(f"kc{kc}\t{relation_label('mlkc', mastery[uid, kc])}\tuid{uid}\n")
                counts["mlkc"] += 1
                if sequence is not None:
                    fp.write(f"kc{kc}\t{relation_label('pkc', sequence[uid, kc])}\tuid{uid}\n")
                    counts["pkc"] += 1
            for ex in range(q.shape[0]):
                fp.write(f"ex{ex}\t{relation_label('exfr', ex_forget[uid, ex])}\tuid{uid}\n")
                counts["exfr"] += 1
            if include_rec:
                ranked = np.lexsort((np.arange(q.shape[0]), distances[uid]))[:top_k_rec]
                for ex in ranked.tolist():
                    fp.write(f"uid{uid}\trec\tex{ex}\n")
                    counts["rec"] += 1
    return counts


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    front = front_dir(args.dataset, root)
    output = graph_dir(args.dataset, root) if args.graph_name == "er_graph" else root / args.dataset / args.graph_name
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"{output} already contains files. Use --force to rebuild the graph.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    mastery = read_matrix_json(front / "stu2know_mastery.json")
    know_forget = read_matrix_json(front / "stu2know_forget.json")
    expected = (raw.student_count, raw.concept_count)
    for name, value in [("mastery", mastery), ("knowledge_forgetting", know_forget)]:
        if value.shape != expected:
            raise ValueError(f"{name} shape={value.shape}, expected={expected}")
    q_counts = raw.q_matrix.sum(axis=1).clip(min=1.0)
    ex_forget = np.clip((know_forget @ raw.q_matrix.T) / q_counts[None, :], 0.0, 1.0)
    sequence = None
    if args.include_sequence:
        sequence_path = front / "stu2know_seq.json"
        if not sequence_path.exists():
            raise FileNotFoundError(f"--include-sequence requires {sequence_path}")
        sequence = read_matrix_json(sequence_path)
        if sequence.shape != expected:
            raise ValueError(f"sequence shape={sequence.shape}, expected={expected}")
        sequence = np.clip(sequence, 0.0, 1.0)
    distances = recommendation_distances(
        mastery,
        ex_forget,
        raw.q_matrix,
        args.delta_1,
        args.delta_2,
        sequence=sequence,
        sequence_term=args.sequence_term,
    )

    write_entities(output / "entities.dict", raw.student_count, raw.concept_count, raw.exercise_count)
    write_relations(output / "relations.dict", include_sequence=args.include_sequence)
    copy_file(raw.root / "Q.txt", output / "Q.txt")
    copy_file(raw.root / "evaluation_uid_kc_response.txt", output / f"{raw.name}_uid_kc_response.txt")
    for name, value in [
        ("stu2know_mastery.json", mastery),
        ("stu2know_forget.json", know_forget),
        ("stu2ex_forget.json", ex_forget),
    ]:
        write_matrix_json(output / name, value, decimals=6)
    if sequence is not None:
        write_matrix_json(output / "stu2know_seq.json", sequence, decimals=6)
    # Ranking always uses this in-memory distance matrix. Persist it without rounding so
    # later inspection/re-ranking cannot introduce artificial ties.
    write_json(output / "stu2ex_recommend_full_precision.json", distances)
    write_matrix_json(output / "stu2ex_recommend.json", distances, decimals=6)
    train_counts = write_triples(
        output / "triples.txt",
        raw.train_uids,
        mastery,
        ex_forget,
        raw.q_matrix,
        distances,
        args.top_k_rec,
        True,
        sequence=sequence,
    )
    test_counts = write_triples(
        output / "test_triples.txt",
        raw.test_uids,
        mastery,
        ex_forget,
        raw.q_matrix,
        distances,
        args.top_k_rec,
        False,
        sequence=sequence,
    )
    source_features = front / "semantic_kg_features"
    if not source_features.exists():
        raise FileNotFoundError("Run front_pipeline/build_semantic_features.py before graph construction")
    shutil.copytree(source_features, output / "semantic_kg_features")
    write_json(
        output / "er_graph_manifest.json",
        {
            "dataset": raw.name,
            "version": "raw_front_er_graph_with_sequence_v1" if args.include_sequence else "raw_front_er_graph_no_sequence_v2",
            "graph_name": args.graph_name,
            "include_sequence": bool(args.include_sequence),
            "raw_dir": raw.root,
            "front_dir": front,
            "split_source": raw.root / "student_split.csv",
            "student_counts": {"train": len(raw.train_uids), "test": len(raw.test_uids)},
            "state_shapes": {"mastery": list(mastery.shape), "forgetting": list(know_forget.shape)},
            "recommendation": {
                "formula": "sqrt((delta_1 - mastery_product)^2 + (delta_2 - mean_q_forgetting)^2)",
                "formula_with_sequence": (
                    f"sqrt((delta_1 - mastery_product)^2 + {args.sequence_term} + "
                    "(delta_2 - mean_q_forgetting)^2)"
                    if args.include_sequence
                    else None
                ),
                "ranking_precision": "full precision; rounded JSON is storage-only",
                "delta_1": args.delta_1,
                "delta_2": args.delta_2,
                "top_k_rec": args.top_k_rec,
            },
            "relations": 304 if args.include_sequence else 203,
            "triple_counts": {"train": train_counts, "test": test_counts},
            "ranges": {
                "exercise_forgetting": matrix_stats(ex_forget),
                "sequence": matrix_stats(sequence) if sequence is not None else None,
                "recommendation_distance": matrix_stats(distances),
            },
        },
    )
    print(f"created raw-front ER graph: {output}")
    print(f"train_triples={sum(train_counts.values())} test_triples={sum(test_counts.values())}")


if __name__ == "__main__":
    main()
