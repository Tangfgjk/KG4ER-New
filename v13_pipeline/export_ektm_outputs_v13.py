from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from common import (
    ensure_large_csv_field_limit,
    entity_count,
    locate_sequence_interactions,
    output_data_root,
    read_entity_dict,
    read_json,
    read_q_matrix,
    source_data_root,
    source_dataset_dir,
    v13_dataset_dir,
    write_json,
    write_matrix_json,
)


@contextlib.contextmanager
def pushd(path: Path) -> Iterable[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export EKTM_mirt TopicRNNModel topic_v and EKTSeqModel_cdm know_output "
            "from the best checkpoint into v13 front files."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Best EKTM_mirt checkpoint .pth/.pt file.")
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--kt-src", type=Path, default=None, help="KT/MMKT/src directory. Defaults to project KT source.")
    parser.add_argument(
        "--exercise-token-file",
        type=Path,
        default=None,
        help=(
            "Optional exercise token-id mapping used by EKTM_mirt. Supports .pt/.pth/.pkl/.json/.npy. "
            "If omitted for Eedi, the original preprocess_emb_eedi() path is used."
        ),
    )
    parser.add_argument("--a-file", type=Path, default=None, help="MIRT a parameter csv. Defaults to v13/mirt/outputs/a_param_*.csv.")
    parser.add_argument("--b-file", type=Path, default=None, help="MIRT b parameter csv. Defaults to v13/mirt/outputs/b_param_*.csv.")
    parser.add_argument("--interaction-file", type=Path, default=None, help="Optional ordered interaction csv with user_id,item_id,score.")
    parser.add_argument(
        "--interaction-source",
        choices=["source_sequence", "mirt_all"],
        default="source_sequence",
        help="Use source sequence_interactions.csv order by default; mirt_all is only a fallback.",
    )
    parser.add_argument("--knowledge-emb-size", type=int, default=50)
    parser.add_argument("--seq-hidden-size", type=int, default=100)
    parser.add_argument("--text-hidden-size", type=int, default=100)
    parser.add_argument("--score-mode", default="double")
    parser.add_argument("--concept-id-offset", type=int, default=1, help="Q.txt concepts are 0-based; EKTM embedding reserves 0 as padding.")
    parser.add_argument("--empty-mastery-default", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_kt_src() -> Path:
    return Path(__file__).resolve().parents[2].parent / "KT" / "MMKT" / "src"


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("DTransformer", "state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    if not state:
        raise ValueError(f"No tensor state_dict entries found in checkpoint: {path}")
    return state


def load_compatible_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for key, value in state.items():
        if key not in current:
            skipped[key] = "missing_in_model"
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            skipped[key] = f"shape_mismatch checkpoint={tuple(value.shape)} model={tuple(current[key].shape)}"
            continue
        compatible[key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_count": len(compatible),
        "skipped_count": len(skipped),
        "skipped_examples": dict(list(skipped.items())[:20]),
        "missing_after_load": list(missing)[:50],
        "unexpected_after_load": list(unexpected)[:50],
    }


def load_token_mapping(path: Path) -> dict[int, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fp:
            payload = pickle.load(fp)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".npy":
        payload = np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported exercise token mapping format: {path}")

    if isinstance(payload, np.ndarray):
        if payload.dtype == object and payload.shape == ():
            payload = payload.item()
        elif payload.ndim == 2:
            return {idx: torch.as_tensor(row, dtype=torch.long) for idx, row in enumerate(payload)}
    if isinstance(payload, list):
        return {idx: torch.as_tensor(row, dtype=torch.long) for idx, row in enumerate(payload)}
    if isinstance(payload, dict):
        result: dict[int, torch.Tensor] = {}
        for key, value in payload.items():
            result[int(key)] = torch.as_tensor(value, dtype=torch.long)
        return result
    raise ValueError(f"Unsupported exercise token mapping payload in {path}")


def load_default_eedi_tokens(kt_src: Path) -> dict[int, torch.Tensor]:
    if str(kt_src) not in sys.path:
        sys.path.insert(0, str(kt_src))
    from data_prep.preprocess_content import preprocess_emb_eedi

    with pushd(kt_src):
        tokens = preprocess_emb_eedi("data_prep/Eedi2020_content.model")
    return {int(key): torch.as_tensor(value, dtype=torch.long) for key, value in tokens.items()}


def find_single(pattern: str, directory: Path, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cannot locate {label}: {directory / pattern}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {label} files found under {directory}: {[str(x) for x in matches]}")
    return matches[0]


def load_mirt_params(v13_dir: Path, a_file: Path | None, b_file: Path | None, exercise_count: int, concept_count: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    output_dir = v13_dir / "mirt" / "outputs"
    a_path = a_file or find_single("a_param_*.csv", output_dir, "MIRT a parameter")
    b_path = b_file or find_single("b_param_*.csv", output_dir, "MIRT b parameter")
    a = np.loadtxt(a_path, delimiter=",", dtype=np.float32)
    b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if a.shape[0] != exercise_count:
        raise ValueError(f"a rows={a.shape[0]} must match exercise_count={exercise_count}: {a_path}")
    if b.shape[0] != exercise_count:
        raise ValueError(f"b rows={b.shape[0]} must match exercise_count={exercise_count}: {b_path}")
    if a.shape[1] != concept_count:
        raise ValueError(
            "EKTM_mirt EKTSeqModel_cdm expects topic_ab length = concept_count + 1. "
            f"Got a_dim={a.shape[1]}, concept_count={concept_count}: {a_path}"
        )
    return (
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b[:, :1], dtype=torch.float32),
        {"a_file": a_path, "b_file": b_path, "a_shape": list(a.shape), "b_shape": list(b[:, :1].shape)},
    )


def graph_uids_from_manifest(dataset: str, v13_dir: Path, learner_count: int) -> list[str]:
    manifest = read_json(v13_dir / "mirt" / "inputs" / "mirt_input_manifest.json")
    uid_info = manifest.get("uid_alignment", {})
    if "sequence_uid_by_entity" in uid_info:
        mapping = uid_info["sequence_uid_by_entity"]
        return [str(mapping[f"uid{idx}"]) for idx in range(learner_count)]
    if dataset == "Eedi":
        return [str(idx) for idx in range(learner_count)]
    raise ValueError(f"Cannot recover graph uid alignment from {v13_dir / 'mirt/inputs/mirt_input_manifest.json'}")


def load_ordered_interactions(args: argparse.Namespace, v13_dir: Path, learner_count: int, exercise_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.interaction_file is not None:
        df = pd.read_csv(args.interaction_file)
        source_info = {"source": "explicit_interaction_file", "path": args.interaction_file}
    elif args.interaction_source == "mirt_all":
        df = pd.read_csv(v13_dir / "mirt" / "inputs" / "all.csv")
        source_info = {
            "source": "mirt_all",
            "path": v13_dir / "mirt" / "inputs" / "all.csv",
            "warning": "mirt_all.csv may be sorted by item_id and is not preferred for GRU state export.",
        }
    else:
        source_dir = source_dataset_dir(args.dataset, args.source_data_root)
        seq_path = locate_sequence_interactions(source_dir)
        ensure_large_csv_field_limit()
        raw = pd.read_csv(seq_path, low_memory=False)
        required = {"uid", "question", "response"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{seq_path} missing columns: {sorted(missing)}")
        graph_uids = graph_uids_from_manifest(args.dataset, v13_dir, learner_count)
        user_map = {uid: idx for idx, uid in enumerate(graph_uids)}
        work = raw.copy()
        work["_order"] = np.arange(len(work))
        work["uid_key"] = work["uid"].astype(str)
        work = work[work["uid_key"].isin(user_map)].copy()
        work["user_id"] = work["uid_key"].map(user_map).astype(int)
        work["item_id"] = pd.to_numeric(work["question"], errors="coerce")
        work["score"] = pd.to_numeric(work["response"], errors="coerce")
        work = work.dropna(subset=["item_id", "score"])
        work["item_id"] = work["item_id"].astype(int)
        work["score"] = (work["score"] > 0).astype(float)
        work = work[(work["item_id"] >= 0) & (work["item_id"] < exercise_count)]
        df = work[["user_id", "item_id", "score", "_order"]].sort_values(["user_id", "_order"]).reset_index(drop=True)
        source_info = {
            "source": "source_sequence_interactions",
            "path": seq_path,
            "order_policy": "preserve original row order within each graph learner",
        }

    required = {"user_id", "item_id", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interaction file missing columns: {sorted(missing)}")
    df = df[["user_id", "item_id", "score"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["score"] = (pd.to_numeric(df["score"], errors="coerce") > 0).astype(float)
    df = df[(df["user_id"] >= 0) & (df["user_id"] < learner_count)]
    df = df[(df["item_id"] >= 0) & (df["item_id"] < exercise_count)]
    return df.reset_index(drop=True), source_info | {"row_count": int(len(df))}


def move_hidden(hidden: Any, device: torch.device) -> Any:
    if hidden is None:
        return None
    if torch.is_tensor(hidden):
        return hidden.to(device)
    if isinstance(hidden, tuple):
        return tuple(move_hidden(item, device) for item in hidden)
    if isinstance(hidden, list):
        return [move_hidden(item, device) for item in hidden]
    return hidden


def build_model(
    kt_src: Path,
    token_dict: dict[int, torch.Tensor],
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    concept_count: int,
    exercise_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    if str(kt_src) not in sys.path:
        sys.path.insert(0, str(kt_src))
    from model_EKT import EKTM_mirt

    with pushd(kt_src):
        model = EKTM_mirt(
            knowledge_length=concept_count,
            knowledge_emb_size=args.knowledge_emb_size,
            seq_hidden_size=args.seq_hidden_size,
            exercise_num=exercise_count,
            text_hidden_size=args.text_hidden_size,
            gpu=(device.type == "cuda"),
            score_mode=args.score_mode,
            dict=token_dict,
            a_dict=a_tensor,
            b_dict=b_tensor,
        )
    return model.to(device).eval()


def topic_vector(model: torch.nn.Module, token_dict: dict[int, torch.Tensor], ex_idx: int, device: torch.device) -> torch.Tensor:
    if ex_idx not in token_dict:
        raise KeyError(f"exercise token mapping missing ex{ex_idx}")
    topic = token_dict[ex_idx].long().view(-1, 1)
    hidden = move_hidden(model.topic_model.default_hidden(1), device)
    output, _ = model.topic_model(topic, hidden)
    if output.ndim == 2:
        return output[0].detach()
    return output.detach().view(-1)


def knowledge_vector(model: torch.nn.Module, q_matrix: np.ndarray, ex_idx: int, concept_id_offset: int, device: torch.device) -> torch.Tensor:
    concept_ids = np.flatnonzero(q_matrix[ex_idx] > 0)
    if len(concept_ids) == 0:
        ids = torch.zeros(1, dtype=torch.long, device=device)
    else:
        ids = torch.tensor(concept_ids + concept_id_offset, dtype=torch.long, device=device)
    return model.embedding(ids).mean(dim=0).detach()


def concept_embedding_matrix(
    model: torch.nn.Module,
    concept_count: int,
    concept_id_offset: int,
    text_dim: int,
    device: torch.device,
) -> np.ndarray:
    concept_ids = torch.arange(concept_id_offset, concept_id_offset + concept_count, dtype=torch.long, device=device)
    concept_emb = model.embedding(concept_ids).detach().cpu().numpy().astype(np.float32)
    aligned = np.zeros((concept_count, text_dim), dtype=np.float32)
    width = min(text_dim, concept_emb.shape[1])
    aligned[:, :width] = concept_emb[:, :width]
    return aligned


def export_topic_and_mastery(
    model: torch.nn.Module,
    token_dict: dict[int, torch.Tensor],
    q_matrix: np.ndarray,
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    interactions: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    exercise_count, concept_count = q_matrix.shape
    topic_cache: list[torch.Tensor] = []
    k_cache: list[torch.Tensor] = []
    ab_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for ex_idx in range(exercise_count):
            topic_cache.append(topic_vector(model, token_dict, ex_idx, device))
            k_cache.append(knowledge_vector(model, q_matrix, ex_idx, args.concept_id_offset, device))
            ab_cache.append(torch.cat([a_tensor[ex_idx].to(device), b_tensor[ex_idx].view(1).to(device)]).detach())
    topic_matrix = torch.stack(topic_cache).detach().cpu().numpy().astype(np.float32)
    concept_matrix = concept_embedding_matrix(
        model,
        concept_count,
        args.concept_id_offset,
        int(topic_matrix.shape[1]),
        device,
    )

    learner_count = int(interactions["user_id"].max()) + 1 if len(interactions) else 0
    # Prefer the graph learner count from possible missing users in the input range.
    learner_count = max(learner_count, entity_count(read_entity_dict(v13_dataset_dir(args.dataset, args.output_data_root) / "entities.dict"), "uid"))
    mastery = np.full((learner_count, concept_count), float(args.empty_mastery_default), dtype=np.float32)
    seen_users = 0
    total_steps = 0
    with torch.no_grad():
        for user_id, group in interactions.groupby("user_id", sort=True):
            hidden = None
            last_know_output: torch.Tensor | None = None
            for row in group.itertuples(index=False):
                ex_idx = int(row.item_id)
                score = float(row.score)
                pred, hidden, know_output = model.seq_model(
                    topic_cache[ex_idx],
                    k_cache[ex_idx],
                    ab_cache[ex_idx],
                    score,
                    hidden,
                )
                del pred
                last_know_output = know_output.detach()
                total_steps += 1
            if last_know_output is not None:
                mastery[int(user_id)] = last_know_output.detach().cpu().numpy().astype(np.float32)
                seen_users += 1
    mastery = np.clip(mastery, 0.0, 1.0)
    return topic_matrix, concept_matrix, mastery, {"seen_users": seen_users, "total_steps": total_steps}


def write_outputs(
    args: argparse.Namespace,
    v13_dir: Path,
    topic_matrix: np.ndarray,
    concept_matrix: np.ndarray,
    mastery: np.ndarray,
    checkpoint_report: dict[str, Any],
    source_report: dict[str, Any],
    mirt_report: dict[str, Any],
) -> None:
    export_dir = v13_dir / "ektm_mirt" / "exports"
    text_dir = v13_dir / "semantic_kg_features" / "text_embeddings"
    if not args.force:
        targets = [
            export_dir / "exercise_topic_v.npy",
            export_dir / "know_output.npy",
            v13_dir / "stu2know_mastery.json",
            text_dir / "exercise_text_embeddings.npy",
        ]
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"v13 EKTM export targets already exist. Use --force to overwrite: {existing[:5]}")

    export_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir / "exercise_topic_v.npy", topic_matrix)
    np.save(export_dir / "know_output.npy", mastery)
    write_matrix_json(export_dir / "stu2know_mastery.json", mastery, decimals=6)
    write_matrix_json(v13_dir / "stu2know_mastery.json", mastery, decimals=6)
    write_matrix_json(v13_dir / "ektm_mirt" / "stu2know_mastery.json", mastery, decimals=6)

    exercise_count, text_dim = topic_matrix.shape
    np.save(text_dir / "exercise_text_embeddings.npy", topic_matrix.astype(np.float32))
    np.save(text_dir / "concept_text_embeddings.npy", concept_matrix.astype(np.float32))
    write_json(
        text_dir / "text_embedding_manifest.json",
        {
            "dataset": args.dataset,
            "model": "EKTM_mirt.TopicRNNModel",
            "source": "topic_v_from_best_checkpoint",
            "checkpoint": args.checkpoint,
            "embedding_dim": int(text_dim),
            "exercise_entity_ids": [f"ex{idx}" for idx in range(exercise_count)],
            "concept_entity_ids": [f"kc{idx}" for idx in range(concept_matrix.shape[0])],
            "files": {
                "exercise_text_embeddings": "exercise_text_embeddings.npy",
                "concept_text_embeddings": "concept_text_embeddings.npy",
            },
            "concept_embedding_source": "EKTM_mirt knowledge embedding, padded or truncated to topic_v dimension",
        },
    )

    write_json(
        export_dir / "ektm_export_manifest.json",
        {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "kt_src": args.kt_src or default_kt_src(),
            "exercise_token_file": args.exercise_token_file,
            "topic_output_file": export_dir / "exercise_topic_v.npy",
            "concept_output_file": text_dir / "concept_text_embeddings.npy",
            "know_output_file": export_dir / "know_output.npy",
            "mastery_json": v13_dir / "stu2know_mastery.json",
            "text_embedding_dir": text_dir,
            "topic_shape": list(topic_matrix.shape),
            "concept_embedding_shape": list(concept_matrix.shape),
            "mastery_shape": list(mastery.shape),
            "mastery_range": {
                "min": float(np.min(mastery)),
                "max": float(np.max(mastery)),
                "mean": float(np.mean(mastery)),
                "std": float(np.std(mastery)),
            },
            "checkpoint_load": checkpoint_report,
            "mirt_params": mirt_report,
            "interaction_source": source_report,
            "notes": (
                "This exporter loads the best EKTM_mirt checkpoint, reuses TopicRNNModel topic_v as exercise text "
                "embeddings, and exports EKTSeqModel_cdm know_output as stu2know_mastery.json."
            ),
        },
    )


def main() -> None:
    args = parse_args()
    kt_src = (args.kt_src or default_kt_src()).resolve()
    if not kt_src.exists():
        raise FileNotFoundError(f"KT source directory not found: {kt_src}")
    device = torch.device(args.device)
    if device.type == "cpu":
        raise RuntimeError("EKTM_mirt TopicRNNModel uses input.cuda() internally. Please run with --device cuda.")

    v13_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    entity2id = read_entity_dict(v13_dir / "entities.dict")
    learner_count = entity_count(entity2id, "uid")
    exercise_count = entity_count(entity2id, "ex")
    concept_count = entity_count(entity2id, "kc")
    q_matrix = read_q_matrix(v13_dir / "Q.txt")
    if q_matrix.shape != (exercise_count, concept_count):
        raise ValueError(f"Q.txt shape={q_matrix.shape} but graph has exercises={exercise_count}, concepts={concept_count}")

    token_dict = load_token_mapping(args.exercise_token_file) if args.exercise_token_file else load_default_eedi_tokens(kt_src)
    missing_tokens = [idx for idx in range(exercise_count) if idx not in token_dict]
    if missing_tokens:
        raise ValueError(f"exercise token mapping misses {len(missing_tokens)} exercises, examples={missing_tokens[:20]}")

    a_tensor, b_tensor, mirt_report = load_mirt_params(v13_dir, args.a_file, args.b_file, exercise_count, concept_count)
    interactions, source_report = load_ordered_interactions(args, v13_dir, learner_count, exercise_count)
    model = build_model(kt_src, token_dict, a_tensor, b_tensor, concept_count, exercise_count, args, device)
    checkpoint_report = load_compatible_state(model, load_checkpoint_state(args.checkpoint))
    model.eval()

    topic_matrix, concept_matrix, mastery, run_report = export_topic_and_mastery(
        model=model,
        token_dict=token_dict,
        q_matrix=q_matrix,
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        interactions=interactions,
        args=args,
        device=device,
    )
    source_report.update(run_report)
    write_outputs(args, v13_dir, topic_matrix, concept_matrix, mastery, checkpoint_report, source_report, mirt_report)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "topic_shape": list(topic_matrix.shape),
                "concept_embedding_shape": list(concept_matrix.shape),
                "mastery_shape": list(mastery.shape),
                "mastery_min": float(np.min(mastery)),
                "mastery_max": float(np.max(mastery)),
                "seen_users": source_report.get("seen_users"),
                "loaded_checkpoint_tensors": checkpoint_report.get("loaded_count"),
                "output_dir": str(v13_dir / "ektm_mirt" / "exports"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

