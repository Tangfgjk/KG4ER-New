from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import output_data_root, v11_dataset_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V11 front-file recommendation distances before SemanticConvE training."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--top-ks", default="10,15,20,30,50,75,100")
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--nov-alpha", type=float, default=1.0)
    parser.add_argument("--ep-top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    if not graph_dir.exists():
        raise FileNotFoundError(graph_dir)

    distances = np.asarray(json.loads((graph_dir / "stu2ex_recommend_full_precision.json").read_text(encoding="utf-8")), dtype=float)
    # evaluate_recommendations ranks larger scores first; V11 front files store smaller-is-better distances.
    uid_ex_scores = [[f"uid{idx}", (-distances[idx]).tolist()] for idx in range(distances.shape[0])]
    score_file = graph_dir / "front_oracle_uid_ex_scores.json"
    write_json(score_file, uid_ex_scores)

    code_dir = Path(__file__).resolve().parents[1] / "codes-New-ConvE"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import evaluate_recommendations

    out_dir = graph_dir / "front_oracle_eval"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_recommendations.py",
            "--data-dir",
            str(graph_dir),
            "--scores-file",
            str(score_file),
            "--output-dir",
            str(out_dir),
            "--dataset-name",
            args.dataset,
            "--model-name",
            "V11_front_oracle",
            "--top-ks",
            args.top_ks,
            "--target-mastery",
            str(args.target_mastery),
            "--nov-alpha",
            str(args.nov_alpha),
            "--ep-top-k",
            str(args.ep_top_k),
        ]
        evaluate_recommendations.main()
    finally:
        sys.argv = old_argv
    print(f"saved V11 front oracle scores: {score_file}")
    print(f"saved V11 front oracle metrics: {out_dir}")


if __name__ == "__main__":
    main()
