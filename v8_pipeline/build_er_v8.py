from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    copy_dir,
    copy_file,
    default_data_root,
    er_root,
    locate_dataset_dir,
    locate_graph_dir,
    write_json,
)


STATIC_GRAPH_FILES = ["entities.dict", "Q.txt", "stu2know_seq.json", "stu2know_forget.json", "stu2ex_forget.json"]


def copy_optional_evaluation_files(dataset: str, dataset_dir: Path, source_graph_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    candidates = []
    for root in [source_graph_dir, dataset_dir]:
        candidates.extend(root.glob("*_uid_kc_response.txt"))
        candidates.append(root / f"{dataset}_uid_kc_response.txt")
    seen = set()
    for src in candidates:
        if src in seen or not src.exists() or not src.is_file():
            continue
        seen.add(src)
        copy_file(src, output_dir / src.name)
        copied.append(src.name)
    return sorted(set(copied))


def run(command: list[str], cwd: Path) -> None:
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated ER v8 graph directory from V8 mastery and feature files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--kt-export-dir", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = locate_dataset_dir(args.dataset, args.data_root).resolve()
    source_graph_dir = locate_graph_dir(dataset_dir).resolve()
    kt_export_dir = (args.kt_export_dir or (dataset_dir / "kt_exports_v8")).resolve()
    feature_dir = (args.feature_dir or (dataset_dir / "semantic_kg_features_v8")).resolve()
    output_dir = (args.output_dir or (dataset_dir / "er_v8")).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite V8 ER graph files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in STATIC_GRAPH_FILES:
        copy_file(source_graph_dir / file_name, output_dir / file_name)
    copy_file(kt_export_dir / "stu2know_mastery.json", output_dir / "stu2know_mastery.json")
    copy_dir(feature_dir, output_dir / "semantic_kg_features")
    copied_eval_files = copy_optional_evaluation_files(args.dataset, dataset_dir, source_graph_dir, output_dir)

    data_scripts_dir = er_root() / "KG4ER" / "data"
    run(
        [
            sys.executable,
            str(data_scripts_dir / "step1_cal_recommend.py"),
            "--data-dir",
            str(output_dir),
            "--output-file",
            "stu2ex_recommend.json",
            "--delta-1",
            str(args.delta_1),
            "--delta-2",
            str(args.delta_2),
        ],
        cwd=data_scripts_dir,
    )
    run(
        [
            sys.executable,
            str(data_scripts_dir / "step2_createtriples.py"),
            "--data-dir",
            str(output_dir),
            "--train-ratio",
            str(args.train_ratio),
            "--seed",
            str(args.seed),
            "--top-k-rec",
            str(args.top_k_rec),
            "--relation-min",
            "0.0",
            "--relation-max",
            "1.0",
        ],
        cwd=data_scripts_dir,
    )
    run(
        [
            sys.executable,
            str(data_scripts_dir / "create_relations_dict.py"),
            "--data-dir",
            str(output_dir),
            "--output-file",
            "relations.dict",
            "--fixed-kg4er-relations",
        ],
        cwd=data_scripts_dir,
    )
    write_json(
        output_dir / "er_v8_manifest.json",
        {
            "dataset": args.dataset,
            "source_graph_dir": source_graph_dir,
            "kt_export_dir": kt_export_dir,
            "feature_dir": feature_dir,
            "output_dir": output_dir,
            "top_k_rec": args.top_k_rec,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "delta_1": args.delta_1,
            "delta_2": args.delta_2,
            "relation_count": 304,
            "copied_evaluation_files": copied_eval_files,
        },
    )
    print(f"created V8 ER graph: {output_dir}")


if __name__ == "__main__":
    main()
