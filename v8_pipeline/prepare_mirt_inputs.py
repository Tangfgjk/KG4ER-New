from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    copy_file,
    default_data_root,
    locate_dataset_dir,
    locate_graph_dir,
    locate_q_file,
    locate_sequence_file,
    read_entity_dict,
    write_json,
)


def _dense_map(values: pd.Series) -> tuple[dict[Any, int], dict[str, str]]:
    unique_values = sorted(values.dropna().astype(str).unique().tolist())
    forward = {value: idx for idx, value in enumerate(unique_values)}
    reverse = {str(idx): value for value, idx in forward.items()}
    return forward, reverse


def normalise_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    required = {"uid", "question", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sequence_interactions.csv missing columns: {sorted(missing)}")

    work = df.copy()
    work["uid_raw"] = work["uid"].astype(str)
    work["item_raw"] = work["question"].astype(str)
    user_map, user_reverse = _dense_map(work["uid_raw"])
    item_map, item_reverse = _dense_map(work["item_raw"])

    work["user_id"] = work["uid_raw"].map(user_map).astype(int)
    work["item_id"] = work["item_raw"].map(item_map).astype(int)
    work["score"] = pd.to_numeric(work["response"], errors="coerce")
    work = work.dropna(subset=["score"])
    work["score"] = (work["score"] > 0).astype(float)
    if "source_split" not in work.columns:
        work["source_split"] = "all"
    else:
        work["source_split"] = work["source_split"].astype(str).str.lower()

    return work, {"user_id_to_raw": user_reverse, "item_id_to_raw": item_reverse}


def build_mirt_frames(df: pd.DataFrame, strategy: str = "all", valid_ratio: float = 0.1, seed: int = 2024) -> dict[str, pd.DataFrame]:
    useful = df[["user_id", "item_id", "score", "source_split"]].copy()
    if strategy not in {"all", "train"}:
        raise ValueError("strategy must be 'all' or 'train'")

    has_explicit_test_split = bool((useful["source_split"] == "test").any())
    if has_explicit_test_split:
        non_test = useful[useful["source_split"] != "test"].copy()
        test = useful[useful["source_split"] == "test"].copy()
    else:
        shuffled = useful.sample(frac=1.0, random_state=seed)
        test_size = max(1, int(len(shuffled) * 0.2)) if len(shuffled) > 5 else 0
        test = shuffled.iloc[:test_size].copy()
        non_test = shuffled.iloc[test_size:].copy()

    train = useful.copy() if strategy == "all" else non_test.copy()
    if has_explicit_test_split:
        valid = non_test.copy()
    elif len(non_test) > 1:
        valid_size = max(1, int(len(non_test) * valid_ratio))
        valid = non_test.sample(n=valid_size, random_state=seed).copy()
    else:
        valid = non_test.copy()

    columns = ["user_id", "item_id", "score"]
    return {
        "train": train[columns].sort_values(columns).reset_index(drop=True),
        "valid": valid[columns].sort_values(columns).reset_index(drop=True),
        "test": test[columns].sort_values(columns).reset_index(drop=True),
        "all": useful[columns].sort_values(columns).reset_index(drop=True),
    }


def write_frames(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V8 no-Q MIRT input CSV files from sequence_interactions.csv.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strategy", choices=["all", "train"], default="all")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = locate_dataset_dir(args.dataset, args.data_root)
    graph_dir = locate_graph_dir(dataset_dir)
    output_dir = args.output_dir or (dataset_dir / "mirt_v8" / "inputs")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite input files.")

    seq_path = locate_sequence_file(dataset_dir)
    df = pd.read_csv(seq_path, low_memory=False)
    normalised, maps = normalise_interactions(df)
    frames = build_mirt_frames(normalised, strategy=args.strategy, valid_ratio=args.valid_ratio, seed=args.seed)
    write_frames(frames, output_dir)
    q_path = locate_q_file(dataset_dir, graph_dir)
    copy_file(q_path, output_dir / "Q_matrix.csv")

    entities = read_entity_dict(graph_dir / "entities.dict")
    manifest = {
        "dataset": args.dataset,
        "strategy": args.strategy,
        "source_sequence_file": seq_path,
        "source_graph_dir": graph_dir,
        "output_dir": output_dir,
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "user_num": int(normalised["user_id"].max()) + 1 if len(normalised) else 0,
        "item_num": int(normalised["item_id"].max()) + 1 if len(normalised) else 0,
        "graph_entity_count": len(entities),
        "maps": maps,
    }
    write_json(output_dir / "mirt_input_manifest.json", manifest)
    print(f"created V8 MIRT inputs: {output_dir}")
    print(f"user_num={manifest['user_num']} item_num={manifest['item_num']} train_rows={manifest['row_counts']['train']}")


if __name__ == "__main__":
    main()
