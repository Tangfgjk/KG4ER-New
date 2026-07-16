from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import (
    copy_file,
    ensure_large_csv_field_limit,
    locate_sequence_interactions,
    locate_source_graph_dir,
    output_data_root,
    project_root,
    read_entity_dict,
    read_q_matrix,
    resolve_source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
)
from multilabel_pkc_lstm import MultiLabelPKCLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-label PKC-LSTM on exercise-level interactions and "
            "overwrite V11 stu2know_seq.json."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=None,
        help="Original KG4ER data root. Defaults to KG4ER_SOURCE_DATA_ROOT, the local sibling path, or V11 manifest provenance.",
    )
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    # Retained for command compatibility. The corrected model does not import pyKT.
    parser.add_argument("--pykt-root", type=Path, default=project_root() / "ER" / "pykt-toolkit-main")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--emb-size", type=int, default=200, help="PKC-LSTM hidden dimension.")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse the saved best checkpoint and only overwrite stu2know_seq.json.",
    )
    return parser.parse_args()


def parse_list(value: object) -> list[str]:
    if value is None:
        return []
    return [token.strip() for token in str(value).split(",") if token.strip() not in ("", "-1")]


def parse_ints(value: object) -> list[int]:
    values: list[int] = []
    for token in parse_list(value):
        try:
            values.append(int(float(token)))
        except ValueError:
            values.append(-1)
    return values


def parse_masks(value: object, length: int) -> list[int]:
    if value is None or str(value).strip() == "":
        return [1] * length
    masks = parse_ints(value)
    return (masks + [1] * length)[:length]


def is_missing_text(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "-1"}


def q_concept_lookup(q_path: Path) -> dict[int, str]:
    q = read_q_matrix(q_path)
    return {
        qid: "_".join(str(idx) for idx, value in enumerate(q[qid].tolist()) if int(value) > 0)
        for qid in range(q.shape[0])
    }


def sort_timestamp(value: object) -> tuple[int, float | str]:
    if value is None or str(value).strip() == "":
        return (1, "")
    try:
        return (0, float(str(value).strip()))
    except ValueError:
        return (1, str(value).strip())


def write_standard_sequences(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["fold", "uid", "questions", "concepts", "responses", "selectmasks", "orig_len"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_eedi_sequence_interactions_as_standard_files(
    source_dir: Path,
    output_dir: Path,
    learner_count: int,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Convert Eedi interaction rows while preserving one exercise per time step."""
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        manifest_path = output_dir / "eedi_standard_sequence_manifest.json"
        if manifest_path.exists():
            return output_dir, json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite.")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_path = source_dir / "Q.txt"
    concept_lookup = q_concept_lookup(q_path)
    sequence_path = locate_sequence_interactions(source_dir)
    ensure_large_csv_field_limit()
    train_groups: dict[str, list[dict[str, Any]]] = {}
    test_groups: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(learner_count)}
    skipped = {"bad_question": 0, "missing_concept": 0, "bad_response": 0, "outside_test_uid": 0}

    with sequence_path.open("r", encoding="utf-8", newline="") as fp:
        for order, row in enumerate(csv.DictReader(fp)):
            try:
                question = int(float(str(row.get("question", "")).strip()))
            except ValueError:
                skipped["bad_question"] += 1
                continue
            try:
                response = int(float(str(row.get("response", "")).strip())) > 0
            except ValueError:
                skipped["bad_response"] += 1
                continue
            concepts = row.get("concepts", "")
            if is_missing_text(concepts):
                concepts = concept_lookup.get(question, "")
            if is_missing_text(concepts):
                skipped["missing_concept"] += 1
                continue
            record = {
                "question": question,
                "concepts": str(concepts),
                "response": int(response),
                "sort_key": sort_timestamp(row.get("timestamp", "")),
                "order": order,
            }
            split = str(row.get("source_split", "")).strip().lower()
            uid = str(row.get("uid", "")).strip()
            if split == "test":
                try:
                    uid_index = int(float(uid))
                except ValueError:
                    skipped["outside_test_uid"] += 1
                    continue
                if 0 <= uid_index < learner_count:
                    test_groups[uid_index].append(record)
                else:
                    skipped["outside_test_uid"] += 1
            else:
                train_groups.setdefault(uid, []).append(record)

    def make_row(uid: str, fold: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(records, key=lambda item: (item["sort_key"], item["order"]))
        return {
            "fold": fold,
            "uid": uid,
            "questions": ",".join(str(item["question"]) for item in ordered),
            "concepts": ",".join(str(item["concepts"]) for item in ordered),
            "responses": ",".join(str(item["response"]) for item in ordered),
            "selectmasks": ",".join("1" for _ in ordered),
            "orig_len": len(ordered),
        }

    missing = [uid for uid, records in test_groups.items() if not records]
    if missing:
        raise ValueError(f"Eedi source sequence data is missing {len(missing)} ER learner histories: {missing[:20]}")
    train_rows = [make_row(uid, idx % 5, records) for idx, (uid, records) in enumerate(sorted(train_groups.items())) if records]
    test_rows = [make_row(str(uid), -1, test_groups[uid]) for uid in range(learner_count)]
    write_standard_sequences(output_dir / "train_sequences.csv", train_rows)
    write_standard_sequences(output_dir / "test_sequences.csv", test_rows)
    copy_file(q_path, output_dir / "Q.txt")
    manifest = {
        "dataset": "Eedi",
        "source_sequence_interactions": sequence_path,
        "learner_alignment": "test_sequences row i maps to ER learner uidi",
        "train_students": len(train_rows),
        "test_students": len(test_rows),
        "train_interactions": sum(int(row["orig_len"]) for row in train_rows),
        "test_interactions": sum(int(row["orig_len"]) for row in test_rows),
        "target_mode": "next_exercise_multihot_q",
        "skipped": skipped,
    }
    write_json(output_dir / "eedi_standard_sequence_manifest.json", manifest)
    return output_dir, manifest


@dataclass
class PKCSequence:
    uid: str
    questions: np.ndarray
    q_vectors: np.ndarray
    responses: np.ndarray
    select_masks: np.ndarray


def read_exercise_sequences(path: Path, q_matrix: np.ndarray) -> tuple[list[PKCSequence], dict[str, int]]:
    """Read raw exercise steps. Multi-concept exercises are never expanded."""
    ensure_large_csv_field_limit()
    sequences: list[PKCSequence] = []
    stats = {"source_rows": 0, "kept_students": 0, "kept_exercise_steps": 0, "skipped_invalid_question": 0}
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row_index, row in enumerate(csv.DictReader(fp)):
            stats["source_rows"] += 1
            questions = parse_ints(row.get("questions", ""))
            responses = parse_ints(row.get("responses", ""))
            usable = min(len(questions), len(responses))
            masks = parse_masks(row.get("selectmasks", None), usable)
            kept_questions: list[int] = []
            kept_q_vectors: list[np.ndarray] = []
            kept_responses: list[int] = []
            kept_masks: list[int] = []
            for question, response, mask in zip(questions[:usable], responses[:usable], masks):
                if question < 0 or question >= q_matrix.shape[0] or response not in (0, 1):
                    stats["skipped_invalid_question"] += 1
                    continue
                q_vector = q_matrix[question].astype(np.float32, copy=True)
                if float(q_vector.sum()) <= 0.0:
                    stats["skipped_invalid_question"] += 1
                    continue
                kept_questions.append(question)
                kept_q_vectors.append(q_vector)
                kept_responses.append(response)
                kept_masks.append(1 if mask > 0 else 0)
            if not kept_questions:
                continue
            sequences.append(
                PKCSequence(
                    uid=str(row.get("uid", row_index)),
                    questions=np.asarray(kept_questions, dtype=np.int64),
                    q_vectors=np.stack(kept_q_vectors).astype(np.float32),
                    responses=np.asarray(kept_responses, dtype=np.float32),
                    select_masks=np.asarray(kept_masks, dtype=np.float32),
                )
            )
            stats["kept_students"] += 1
            stats["kept_exercise_steps"] += len(kept_questions)
    return sequences, stats


def split_and_chunk_sequences(
    sequences: list[PKCSequence], maxlen: int, valid_ratio: float, seed: int
) -> tuple[list[PKCSequence], list[PKCSequence]]:
    """Split by learner before chunking, so a learner cannot occur in both splits."""
    eligible = [item for item in sequences if len(item.questions) >= 2]
    if len(eligible) < 2:
        raise ValueError("Need at least two learner sequences containing two exercise steps.")
    rng = random.Random(seed)
    indices = list(range(len(eligible)))
    rng.shuffle(indices)
    valid_count = max(1, min(len(eligible) - 1, round(len(eligible) * valid_ratio)))
    valid_indices = set(indices[:valid_count])

    def chunk(items: Iterable[PKCSequence]) -> list[PKCSequence]:
        chunks: list[PKCSequence] = []
        for item in items:
            for start in range(0, len(item.questions), maxlen):
                end = min(start + maxlen, len(item.questions))
                if end - start < 2:
                    continue
                chunks.append(
                    PKCSequence(
                        uid=item.uid,
                        questions=item.questions[start:end],
                        q_vectors=item.q_vectors[start:end],
                        responses=item.responses[start:end],
                        select_masks=item.select_masks[start:end],
                    )
                )
        return chunks

    return chunk(item for idx, item in enumerate(eligible) if idx not in valid_indices), chunk(
        item for idx, item in enumerate(eligible) if idx in valid_indices
    )


class PKCDataset(Dataset):
    def __init__(self, sequences: list[PKCSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> PKCSequence:
        return self.sequences[index]


def collate_pkc(batch: list[PKCSequence]) -> dict[str, torch.Tensor]:
    max_steps = max(len(item.questions) - 1 for item in batch)
    concepts = batch[0].q_vectors.shape[1]
    inputs = torch.zeros(len(batch), max_steps, 2 * concepts, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_steps, concepts, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_steps, 1, dtype=torch.float32)
    for row, item in enumerate(batch):
        q = torch.from_numpy(item.q_vectors)
        # Normalize only the input. The next-exercise target remains binary multi-hot.
        q_input = q / q.sum(dim=1, keepdim=True).clamp_min(1.0)
        response = torch.from_numpy(item.responses).view(-1, 1)
        inputs_all = torch.cat([(1.0 - response) * q_input, response * q_input], dim=1)
        length = len(item.questions) - 1
        inputs[row, :length] = inputs_all[:-1]
        targets[row, :length] = q[1:]
        masks[row, :length, 0] = torch.from_numpy(item.select_masks[1:])
    return {"inputs": inputs, "targets": targets, "masks": masks}


def calculate_positive_weight(sequences: list[PKCSequence]) -> tuple[float, dict[str, int]]:
    positives = 0
    valid_steps = 0
    concepts = sequences[0].q_vectors.shape[1]
    for item in sequences:
        target_q = item.q_vectors[1:]
        target_mask = item.select_masks[1:] > 0
        positives += int(target_q[target_mask].sum())
        valid_steps += int(target_mask.sum())
    negatives = valid_steps * concepts - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"Invalid PKC class counts: positives={positives}, negatives={negatives}")
    return float(negatives / positives), {"positive_labels": positives, "negative_labels": negatives, "valid_steps": valid_steps}


def weighted_multilabel_bce(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, positive_weight: float) -> torch.Tensor:
    base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (float(positive_weight) - 1.0) * targets
    weighted_mask = weights * masks
    return (base_loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model: MultiLabelPKCLSTM, loader: DataLoader, device: torch.device, positive_weight: float) -> dict[str, float]:
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    recall_sum = 0.0
    recall_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        logits = model(inputs)
        base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = 1.0 + (float(positive_weight) - 1.0) * targets
        weighted_mask = weights * masks
        loss_numerator += float((base_loss * weighted_mask).sum().item())
        loss_denominator += float(weighted_mask.sum().item())

        probabilities = torch.sigmoid(logits)
        for row in range(targets.shape[0]):
            for step in range(targets.shape[1]):
                if masks[row, step, 0] <= 0:
                    continue
                positive_count = int(targets[row, step].sum().item())
                if positive_count <= 0:
                    continue
                top_ids = probabilities[row, step].topk(min(positive_count, probabilities.shape[-1])).indices
                hits = float(targets[row, step, top_ids].sum().item())
                recall_sum += hits / positive_count
                recall_count += 1
    model.train()
    return {
        "weighted_bce": loss_numerator / max(loss_denominator, 1.0),
        "recall_at_true_label_count": recall_sum / max(recall_count, 1),
    }


def source_sequence_dir(dataset: str, source_dir: Path, graph_dir: Path, output_dir: Path, force: bool) -> tuple[Path, dict[str, Any] | None]:
    if dataset != "Eedi":
        return graph_dir, None
    entities = read_entity_dict(graph_dir / "entities.dict")
    learner_count = sum(1 for entity in entities if entity.startswith("uid"))
    eedi_dir = output_dir / "pkc_lstm" / "eedi_standard_sequences"
    return prepare_eedi_sequence_interactions_as_standard_files(source_dir, eedi_dir, learner_count, force)


def save_checkpoint(path: Path, model: MultiLabelPKCLSTM, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[MultiLabelPKCLSTM, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = MultiLabelPKCLSTM(
        num_concepts=int(config["num_concepts"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def export_student_seq(model: MultiLabelPKCLSTM, sequences: list[PKCSequence], device: torch.device, maxlen: int) -> list[list[float]]:
    model.eval()
    exported: list[list[float]] = []
    for item in sequences:
        if len(item.questions) == 0:
            exported.append([0.0] * model.num_concepts)
            continue
        q = torch.from_numpy(item.q_vectors[-maxlen:]).to(device)
        response = torch.from_numpy(item.responses[-maxlen:]).to(device).view(-1, 1)
        q_input = q / q.sum(dim=1, keepdim=True).clamp_min(1.0)
        inputs = torch.cat([(1.0 - response) * q_input, response * q_input], dim=1).unsqueeze(0)
        probabilities = torch.sigmoid(model(inputs))[0, -1].detach().cpu().tolist()
        exported.append([float(value) for value in probabilities])
    return exported


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.maxlen < 2:
        raise ValueError("--maxlen must be at least 2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    resolved_source_root = resolve_source_data_root(args.dataset, args.source_data_root, args.output_data_root)
    source_dir = source_dataset_dir(args.dataset, resolved_source_root)
    source_graph_dir = locate_source_graph_dir(source_dir)
    output_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir, eedi_manifest = source_sequence_dir(args.dataset, source_dir, source_graph_dir, output_dir, args.force)
    q_matrix = read_q_matrix(sequence_dir / "Q.txt")
    num_concepts = int(q_matrix.shape[1])
    train_sequences, train_read_stats = read_exercise_sequences(sequence_dir / "train_sequences.csv", q_matrix)
    test_sequences, test_read_stats = read_exercise_sequences(sequence_dir / "test_sequences.csv", q_matrix)
    if not test_sequences:
        raise ValueError(f"No test learner sequences found in {sequence_dir / 'test_sequences.csv'}")

    model_dir = output_dir / "pkc_lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = int(args.hidden_size or args.emb_size)
    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"

    if args.skip_train:
        checkpoint_path = best_path if best_path.exists() else last_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No PKC-LSTM checkpoint found under {model_dir}")
        model, checkpoint_config = load_checkpoint(checkpoint_path, device)
        if int(checkpoint_config["num_concepts"]) != num_concepts:
            raise ValueError("Checkpoint concept dimension does not match Q.txt")
        train_manifest: dict[str, Any] = {"reused_checkpoint": checkpoint_path}
    else:
        train_chunks, valid_chunks = split_and_chunk_sequences(train_sequences, args.maxlen, args.valid_ratio, args.seed)
        positive_weight, class_counts = calculate_positive_weight(train_chunks)
        train_loader = DataLoader(PKCDataset(train_chunks), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkc)
        valid_loader = DataLoader(PKCDataset(valid_chunks), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkc)
        model = MultiLabelPKCLSTM(num_concepts, hidden_size, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_valid = float("inf")
        best_epoch = -1
        history: list[dict[str, float | int]] = []
        config = {
            "model": "MultiLabelPKCLSTM",
            "target_mode": "next_exercise_multihot_q",
            "input_mode": "response_conditioned_l1_normalized_q",
            "num_concepts": num_concepts,
            "hidden_size": hidden_size,
            "dropout": args.dropout,
            "maxlen": args.maxlen,
            "positive_weight": positive_weight,
            "seed": args.seed,
        }
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0
            for batch in train_loader:
                logits = model(batch["inputs"].to(device))
                loss = weighted_multilabel_bce(logits, batch["targets"].to(device), batch["masks"].to(device), positive_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
            valid_metrics = evaluate(model, valid_loader, device, positive_weight)
            epoch_summary = {"epoch": epoch, "train_weighted_bce": total_loss / max(batches, 1), **valid_metrics}
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            save_checkpoint(last_path, model, config)
            if valid_metrics["weighted_bce"] < best_valid:
                best_valid = valid_metrics["weighted_bce"]
                best_epoch = epoch
                save_checkpoint(best_path, model, config)
        model, _ = load_checkpoint(best_path, device)
        train_manifest = {
            "train_chunks": len(train_chunks),
            "valid_chunks": len(valid_chunks),
            "class_counts": class_counts,
            "positive_weight": positive_weight,
            "best_epoch": best_epoch,
            "best_valid_weighted_bce": best_valid,
            "history": history,
        }

    expected_students = sum(1 for entity in read_entity_dict(source_graph_dir / "entities.dict") if entity.startswith("uid"))
    if len(test_sequences) != expected_students:
        raise ValueError(f"Test learner sequences={len(test_sequences)}, ER graph learners={expected_students}")
    exported = export_student_seq(model, test_sequences, device, args.maxlen)
    if len(exported) != expected_students or any(len(row) != num_concepts for row in exported):
        raise RuntimeError("Exported stu2know_seq.json has an invalid shape")
    root_seq_path = output_dir / "stu2know_seq.json"
    model_seq_path = model_dir / "stu2know_seq.json"
    root_seq_path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    model_seq_path.write_text(root_seq_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "dataset": args.dataset,
        "source_data_root": resolved_source_root,
        "source_graph_dir": source_graph_dir,
        "sequence_dir": sequence_dir,
        "output_file": root_seq_path,
        "target_mode": "next_exercise_multihot_q",
        "input_mode": "response_conditioned_l1_normalized_q",
        "multi_concept_exercises_preserved": True,
        "num_concepts": num_concepts,
        "expected_students": expected_students,
        "train_sequence_stats": train_read_stats,
        "test_sequence_stats": test_read_stats,
        "eedi_standard_sequence_manifest": eedi_manifest,
        **train_manifest,
    }
    write_json(model_dir / "pkc_lstm_manifest.json", manifest)
    print(json.dumps({"output_file": str(root_seq_path), "students": len(exported), "concepts": num_concepts, "model_dir": str(model_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
