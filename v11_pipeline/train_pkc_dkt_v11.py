from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    ensure_large_csv_field_limit,
    entity_count,
    jsonable,
    locate_source_graph_dir,
    locate_test_sequences,
    output_data_root,
    read_entity_dict,
    read_q_matrix,
    source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
)
from pkc_dkt_model_v11 import PKCDKT
from prepare_mirt_inputs_v11 import eedi_valid_question_length


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoints saved by this script across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_state_dict(concept_count: int, state_dict: dict[str, torch.Tensor], dropout: float, device: torch.device) -> PKCDKT:
    embed_size = int(state_dict["input_embedding.weight"].shape[1])
    hidden_size = int(state_dict["rnn.weight_hh_l0"].shape[1])
    model = PKCDKT(concept_count, embed_size=embed_size, hidden_size=hidden_size, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    return model


def parse_concept_steps(value: Any) -> list[list[int]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    steps: list[list[int]] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token or token == "-1":
            continue
        concepts = [int(x) for x in token.split("_") if x and x != "-1"]
        if concepts:
            steps.append(concepts)
    return steps


def parse_response_steps(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    responses: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token or token == "-1":
            continue
        try:
            responses.append(1 if int(float(token)) > 0 else 0)
        except ValueError:
            responses.append(0)
    return responses


def locate_train_sequences(dataset: str, source_dir: Path, graph_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        graph_dir / "train_valid_sequences.csv",
        source_dir / "prepared_for_kt" / "train_valid_sequences.csv",
        source_dir / "train_valid_sequences.csv",
    ]
    if dataset == "Eedi":
        candidates.extend(
            [
                source_dir / "train_valid_sequences.csv",
                source_dir.parent.parent / "pykt-toolkit-main" / "data" / "Eedi" / "train_valid_sequences.csv",
                Path(__file__).resolve().parents[2] / "pykt-toolkit-main" / "data" / "Eedi" / "train_valid_sequences.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"cannot locate train_valid_sequences.csv for {dataset}")


class PKCSequenceDataset(Dataset):
    def __init__(self, csv_path: Path, concept_count: int) -> None:
        ensure_large_csv_field_limit()
        rows = pd.read_csv(csv_path, low_memory=False).to_dict("records")
        self.samples: list[tuple[list[int], list[int], np.ndarray]] = []
        self.concept_count = int(concept_count)
        for row in rows:
            concepts = parse_concept_steps(row.get("concepts", ""))
            responses = parse_response_steps(row.get("responses", ""))
            usable = min(len(concepts), len(responses))
            concepts = concepts[:usable]
            responses = responses[:usable]
            if usable < 2:
                continue
            inputs_c: list[int] = []
            inputs_r: list[int] = []
            targets = np.zeros((usable - 1, self.concept_count), dtype=np.float32)
            for idx in range(usable - 1):
                current = [c for c in concepts[idx] if 0 <= c < self.concept_count]
                nxt = [c for c in concepts[idx + 1] if 0 <= c < self.concept_count]
                if not current or not nxt:
                    continue
                inputs_c.append(int(current[0]))
                inputs_r.append(int(responses[idx]))
                targets[len(inputs_c) - 1, nxt] = 1.0
            if inputs_c:
                self.samples.append((inputs_c, inputs_r, targets[: len(inputs_c)]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], np.ndarray]:
        return self.samples[index]


def collate_batch(batch: list[tuple[list[int], list[int], np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item[0]) for item in batch)
    concept_count = batch[0][2].shape[1]
    concepts = torch.zeros((len(batch), max_len), dtype=torch.long)
    responses = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), max_len, concept_count), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row_idx, (c, r, t) in enumerate(batch):
        length = len(c)
        concepts[row_idx, :length] = torch.tensor(c, dtype=torch.long)
        responses[row_idx, :length] = torch.tensor(r, dtype=torch.long)
        targets[row_idx, :length] = torch.tensor(t, dtype=torch.float32)
        mask[row_idx, :length] = True
    return concepts, responses, targets, mask


def split_dataset(dataset: PKCSequenceDataset, valid_ratio: float, seed: int) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    valid_size = max(1, int(len(indices) * valid_ratio)) if len(indices) > 1 else 0
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:] or indices
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, valid_indices or train_indices[:1])


def run_epoch(model: PKCDKT, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for concepts, responses, targets, mask in tqdm(loader, disable=not training):
        concepts = concepts.to(device)
        responses = responses.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        pred = model(concepts, responses)
        loss = F.binary_cross_entropy(pred[mask], targets[mask])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batches += 1
    return total_loss / max(1, batches)


@torch.no_grad()
def export_test_sequence_predictions(
    model: PKCDKT,
    dataset: str,
    source_dir: Path,
    graph_dir: Path,
    output_file: Path,
    concept_count: int,
    learner_count: int,
    device: torch.device,
) -> dict[str, Any]:
    rows_path = locate_test_sequences(dataset, source_dir, graph_dir)
    ensure_large_csv_field_limit()
    if dataset == "Eedi":
        rows: list[dict[str, Any]] = []
        with rows_path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                valid_len = eedi_valid_question_length(row)
                if 10 <= valid_len <= 198:
                    rows.append(dict(row))
        rows = rows[:learner_count]
    else:
        rows = pd.read_csv(rows_path, low_memory=False).head(learner_count).to_dict("records")
    if len(rows) != learner_count:
        raise ValueError(f"aligned test rows={len(rows)} != learner_count={learner_count}")

    model.eval()
    results: list[list[float]] = []
    for row in rows:
        concepts = parse_concept_steps(row.get("concepts", ""))
        responses = parse_response_steps(row.get("responses", ""))
        usable = min(len(concepts), len(responses))
        concepts = concepts[:usable]
        responses = responses[:usable]
        inputs_c = [next((c for c in step if 0 <= c < concept_count), 0) for step in concepts]
        inputs_r = [int(r) for r in responses]
        if not inputs_c:
            results.append([0.0] * concept_count)
            continue
        c = torch.tensor([inputs_c], dtype=torch.long, device=device)
        r = torch.tensor([inputs_r], dtype=torch.long, device=device)
        pred = model(c, r)[0, len(inputs_c) - 1].detach().cpu().numpy()
        results.append([round(float(x), 6) for x in pred.tolist()])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return {"test_sequences": rows_path, "output_file": output_file, "students": len(results), "concepts": concept_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V11 PKC-DKT and export stu2know_seq.json.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--train-sequences", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export stu2know_seq.json from an existing best.pt checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    graph_dir = locate_source_graph_dir(source_dir)
    v11_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    q_matrix = read_q_matrix(graph_dir / "Q.txt")
    entity2id = read_entity_dict(graph_dir / "entities.dict")
    learner_count = entity_count(entity2id, "uid")
    concept_count = int(q_matrix.shape[1])
    train_path = locate_train_sequences(args.dataset, source_dir, graph_dir, args.train_sequences)
    output_dir = v11_dir / "pkc_dkt"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if best_path.exists() and not args.force and not args.export_only:
        raise FileExistsError(f"{best_path} exists. Use --force to retrain.")
    model = PKCDKT(concept_count, embed_size=args.embed_size, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    best_valid = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    if args.export_only:
        if not best_path.exists():
            raise FileNotFoundError(f"{best_path} does not exist. Train PKC-DKT before using --export-only.")
    else:
        dataset = PKCSequenceDataset(train_path, concept_count=concept_count)
        if len(dataset) == 0:
            raise ValueError(f"No usable PKC-DKT sequences in {train_path}")
        train_set, valid_set = split_dataset(dataset, args.valid_ratio, args.seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            with torch.no_grad():
                valid_loss = run_epoch(model, valid_loader, device, None)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
            payload = {
                "model": "PKCDKT",
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": jsonable(vars(args)),
                "concept_count": concept_count,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
            torch.save(payload, last_path)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch + 1
                torch.save(payload, best_path)
            print(f"[Epoch {epoch + 1}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} best_epoch={best_epoch}")

    best_payload = load_checkpoint(best_path, device)
    best_epoch = int(best_payload.get("epoch", best_epoch))
    best_valid = float(best_payload.get("valid_loss", best_valid))
    model = build_model_from_state_dict(concept_count, best_payload["model_state_dict"], args.dropout, device)
    export_summary = export_test_sequence_predictions(
        model,
        args.dataset,
        source_dir,
        graph_dir,
        output_dir / "stu2know_seq.json",
        concept_count,
        learner_count,
        device,
    )
    (v11_dir / "stu2know_seq.json").write_text((output_dir / "stu2know_seq.json").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        output_dir / "pkc_dkt_manifest.json",
        {
            "dataset": args.dataset,
            "source": "V11 PKC-DKT trained locally for next-knowledge-concept occurrence prediction",
            "train_sequences": train_path,
            "graph_dir": graph_dir,
            "concept_count": concept_count,
            "learner_count": learner_count,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid,
            "export_only": bool(args.export_only),
            "history": history,
            "export": export_summary,
        },
    )
    print(f"saved PKC-DKT seq file: {output_dir / 'stu2know_seq.json'}")


if __name__ == "__main__":
    main()
