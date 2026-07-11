from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import output_data_root, read_json, v11_dataset_dir, write_json, write_matrix_json


def load_mastery_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".csv":
        matrix = np.loadtxt(path, delimiter=",")
    else:
        matrix = np.asarray(read_json(path), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"mastery matrix must be 2-D: {path}, shape={matrix.shape}")
    if np.min(matrix) < -1e-8 or np.max(matrix) > 1 + 1e-8:
        raise ValueError(f"mastery values must be in [0,1]: min={np.min(matrix)} max={np.max(matrix)}")
    return np.asarray(matrix, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import strict EKTM_mirt know_output as V11 stu2know_mastery.json.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--know-output-file", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--expected-students", type=int, default=None)
    parser.add_argument("--expected-concepts", type=int, default=None)
    parser.add_argument("--source-note", default="EKTM_mirt EKTSeqModel_cdm know_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_mastery_matrix(args.know_output_file)
    if args.expected_students is not None and matrix.shape[0] != args.expected_students:
        raise ValueError(f"students={matrix.shape[0]} expected={args.expected_students}")
    if args.expected_concepts is not None and matrix.shape[1] != args.expected_concepts:
        raise ValueError(f"concepts={matrix.shape[1]} expected={args.expected_concepts}")
    output_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    target = output_dir / "ektm_mirt" / "stu2know_mastery.json"
    write_matrix_json(target, matrix, decimals=6)
    (output_dir / "stu2know_mastery.json").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        output_dir / "ektm_mirt" / "mastery_export_manifest.json",
        {
            "dataset": args.dataset,
            "source": "strict_ektm_mirt_know_output",
            "source_note": args.source_note,
            "know_output_file": args.know_output_file,
            "output_file": target,
            "student_count": int(matrix.shape[0]),
            "concept_count": int(matrix.shape[1]),
            "range": {
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix)),
                "std": float(np.std(matrix)),
            },
        },
    )
    print(f"imported V11 EKTM_mirt mastery: {target}")


if __name__ == "__main__":
    main()
