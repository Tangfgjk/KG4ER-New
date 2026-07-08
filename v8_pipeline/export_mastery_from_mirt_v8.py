from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import default_data_root, graph_learner_fit_indices, locate_dataset_dir, locate_graph_dir, locate_q_file, read_q_matrix, sigmoid, write_json


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return np.asarray(matrix, dtype=np.float64)


def concept_difficulty_from_items(b_param: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    b = b_param.reshape(-1)
    concept_count = q_matrix.shape[1]
    result = np.zeros(concept_count, dtype=np.float64)
    global_mean = float(np.mean(b)) if b.size else 0.0
    for kc in range(concept_count):
        item_ids = np.where(q_matrix[:, kc] > 0)[0]
        result[kc] = float(np.mean(b[item_ids])) if len(item_ids) else global_mean
    return result


def observed_concept_mastery(input_dir: Path, q_matrix: np.ndarray, user_num: int) -> tuple[np.ndarray, np.ndarray]:
    all_path = input_dir / "all.csv"
    concept_count = q_matrix.shape[1]
    correct = np.zeros((user_num, concept_count), dtype=np.float64)
    total = np.zeros((user_num, concept_count), dtype=np.float64)
    if not all_path.exists():
        return correct, total
    df = pd.read_csv(all_path)
    for row in df.itertuples(index=False):
        uid = int(row.user_id)
        item = int(row.item_id)
        if uid < 0 or uid >= user_num or item < 0 or item >= q_matrix.shape[0]:
            continue
        concepts = np.where(q_matrix[item] > 0)[0]
        if len(concepts) == 0:
            continue
        score = float(row.score)
        correct[uid, concepts] += score
        total[uid, concepts] += 1.0
    return correct, total


def build_mastery_proxy(theta_param: np.ndarray, b_param: np.ndarray, q_matrix: np.ndarray, input_dir: Path, history_weight: float = 0.5) -> list[list[float]]:
    user_num = theta_param.shape[0]
    theta_scalar = theta_param.mean(axis=1)
    theta_scalar = (theta_scalar - float(np.mean(theta_scalar))) / (float(np.std(theta_scalar)) + 1e-8)
    concept_difficulty = concept_difficulty_from_items(b_param, q_matrix)
    concept_difficulty = (concept_difficulty - float(np.mean(concept_difficulty))) / (float(np.std(concept_difficulty)) + 1e-8)
    mirt_mastery = sigmoid(theta_scalar[:, None] - concept_difficulty[None, :])
    correct, total = observed_concept_mastery(input_dir, q_matrix, user_num)
    observed = np.divide(correct, total, out=np.full_like(correct, 0.5), where=total > 0)
    weights = np.clip(total / 5.0, 0.0, 1.0) * max(0.0, min(1.0, history_weight))
    mastery = (1.0 - weights) * mirt_mastery + weights * observed
    return np.clip(mastery, 0.0, 1.0).round(6).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V8 stu2know_mastery.json from modified no-Q MIRT parameters.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--mirt-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--history-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = locate_dataset_dir(args.dataset, args.data_root)
    graph_dir = locate_graph_dir(dataset_dir)
    input_dir = args.input_dir or (dataset_dir / "mirt_v8" / "inputs")
    mirt_output_dir = args.mirt_output_dir or (dataset_dir / "mirt_v8" / "outputs")
    output_dir = args.output_dir or (dataset_dir / "kt_exports_v8")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite V8 KT exports.")
    manifest = __import__("json").loads((mirt_output_dir / "mirt_export_manifest.json").read_text(encoding="utf-8"))
    latent_dim = int(manifest["latent_dim"])
    b_param = load_matrix(mirt_output_dir / f"b_param_{latent_dim}.csv")
    theta_param = load_matrix(mirt_output_dir / f"theta_param_{latent_dim}.csv")
    q_matrix = read_q_matrix(locate_q_file(dataset_dir))
    full_mastery = build_mastery_proxy(theta_param, b_param, q_matrix, input_dir, history_weight=args.history_weight)
    fit_indices, alignment = graph_learner_fit_indices(args.dataset, dataset_dir, graph_dir, input_dir)
    mastery = [full_mastery[fit_uid] for fit_uid in fit_indices]
    write_json(output_dir / "stu2know_mastery.json", mastery)
    write_json(
        output_dir / "mastery_export_manifest.json",
        {
            "dataset": args.dataset,
            "source": "mirt_proxy",
            "mirt_output_dir": mirt_output_dir,
            "input_dir": input_dir,
            "output_file": output_dir / "stu2know_mastery.json",
            "student_count": len(mastery),
            "mirt_student_count": len(full_mastery),
            "concept_count": len(mastery[0]) if mastery else 0,
            "alignment": alignment,
            "history_weight": args.history_weight,
            "notes": "This proxy uses no-Q MIRT theta, item b-derived concept difficulty, and observed concept response history. Replace with EKTM_mirt checkpoint export when a compatible checkpoint is available.",
        },
    )
    print(f"saved V8 mastery: {output_dir / 'stu2know_mastery.json'}")


if __name__ == "__main__":
    main()
