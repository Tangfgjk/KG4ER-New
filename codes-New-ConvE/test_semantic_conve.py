"""Export recommendation scores from a trained SemanticConvE model."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import torch

from feature_loader import load_semantic_feature_bundle
from semantic_conve_model import SemanticConvE, VALID_MODEL_ABLATIONS


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_device(cuda: str) -> torch.device:
    if cuda == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda.lower() in {"true", "1", "yes", "cuda"}:
        return torch.device("cuda")
    return torch.device("cpu")


def read_q_count(data_path: Path) -> int:
    with (data_path / "Q.txt").open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def test_users_from_triples(data_path: Path) -> List[str]:
    users = set()
    with (data_path / "test_triples.txt").open("r", encoding="utf-8") as fp:
        for line in fp:
            head, relation, tail = line.strip().split("\t")
            if relation.startswith("mlkc"):
                users.add(tail)
    return sorted(users, key=lambda uid: int(uid[3:]) if uid.startswith("uid") and uid[3:].isdigit() else uid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SemanticConvE uid-exercise recommendation scores.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, required=True)
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", required=True)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--ablation", choices=sorted(VALID_MODEL_ABLATIONS), default="full")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=200)
    parser.add_argument("--embedding-shape1", "--embedding_shape1", dest="embedding_shape1", type=int, default=20)
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=9728)
    parser.add_argument("--input-drop", "--input_drop", dest="input_drop", type=float, default=0.2)
    parser.add_argument("--hidden-drop", "--hidden_drop", dest="hidden_drop", type=float, default=0.2)
    parser.add_argument("--feat-drop", "--feat_drop", dest="feat_drop", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.cuda))
    bundle = load_semantic_feature_bundle(args.data_path, args.feature_dir, device=device)
    model = SemanticConvE.from_feature_bundle(
        bundle,
        embedding_dim=args.embedding_dim,
        embedding_shape1=args.embedding_shape1,
        hidden_size=args.hidden_size,
        input_drop=args.input_drop,
        hidden_drop=args.hidden_drop,
        feat_drop=args.feat_drop,
        ablation_mode=args.ablation,
    ).to(device)

    checkpoint_path = args.save_path / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_ablation = checkpoint.get("ablation", "full")
    if checkpoint_ablation != args.ablation:
        raise RuntimeError(
            f"Checkpoint ablation is incompatible: {checkpoint_ablation!r} != {args.ablation!r}. "
            "Use the matching --ablation value for testing."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    q_count = read_q_count(args.data_path)
    exercise_ids = [bundle.entity2id[f"ex{i}"] for i in range(q_count)]
    exercise_tail_ids = torch.tensor(exercise_ids, dtype=torch.long, device=device)
    rec_id = bundle.relation2id["rec"]
    rec_relation = torch.tensor([rec_id], dtype=torch.long, device=device)
    users = test_users_from_triples(args.data_path)

    uid_ex_scores = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(users), args.batch_size):
            batch_users = users[start : start + args.batch_size]
            h = torch.tensor([bundle.entity2id[uid] for uid in batch_users], dtype=torch.long, device=device)
            r = rec_relation.repeat(len(batch_users))
            scores = model.score_tails(h, r, exercise_tail_ids).detach().cpu().tolist()
            uid_ex_scores.extend((uid, row) for uid, row in zip(batch_users, scores))
    inference_seconds = time.perf_counter() - inference_start

    output_file = args.output_file or (args.save_path / "SemanticConvE_uid_ex_scores.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as fp:
        pickle.dump(uid_ex_scores, fp)

    write_json(
        args.save_path / "semantic_conve_inference.json",
        {
            "model": "SemanticConvE",
            "ablation": args.ablation,
            "dataset": args.dataset_name,
            "checkpoint": str(checkpoint_path),
            "output_file": str(output_file),
            "test_user_count": len(users),
            "exercise_count": q_count,
            "inference_seconds": round(inference_seconds, 6),
            "scoring": "type-aware: rec scores are computed only over exercise entities ex0..exN",
        },
    )
    print(json.dumps({"output_file": str(output_file), "users": len(users), "exercises": q_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
