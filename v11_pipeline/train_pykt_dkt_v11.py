from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from common import (
    copy_file,
    ensure_large_csv_field_limit,
    locate_sequence_interactions,
    output_data_root,
    project_root,
    read_q_matrix,
    source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
)


FIELDNAMES = ["fold", "uid", "questions", "concepts", "responses", "selectmasks", "orig_len"]
STANDARD_SEQUENCE_FIELDS = ["fold", "uid", "questions", "concepts", "responses", "selectmasks", "orig_len"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightly patched pyKT DKT for next-concept occurrence prediction "
            "and export V11 stu2know_seq.json."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--pykt-root", type=Path, default=project_root() / "ER" / "pykt-toolkit-main")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--emb-size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only export stu2know_seq.json from an existing V11 patched pyKT checkpoint.",
    )
    return parser.parse_args()


def parse_list(value: object) -> list[str]:
    if value is None:
        return []
    return [token for token in str(value).split(",") if token != ""]


def expand_row(row: dict[str, str]) -> list[dict[str, int]]:
    questions = parse_list(row.get("questions", ""))
    concepts = parse_list(row.get("concepts", ""))
    responses = parse_list(row.get("responses", ""))
    selectmasks = parse_list(row.get("selectmasks", "")) if "selectmasks" in row else ["1"] * len(questions)
    expanded: list[dict[str, int]] = []
    usable = min(len(questions), len(concepts), len(responses), len(selectmasks))
    for idx in range(usable):
        if questions[idx] == "-1" or concepts[idx] == "-1" or responses[idx] == "-1":
            continue
        for concept in concepts[idx].split("_"):
            if concept in ("", "-1"):
                continue
            expanded.append(
                {
                    "question": int(questions[idx]),
                    "concept": int(concept),
                    "response": int(responses[idx]),
                    "selectmask": int(selectmasks[idx]),
                }
            )
    return expanded


def pad(values: list[str], maxlen: int, pad_value: str = "-1") -> list[str]:
    return values + [pad_value] * max(0, maxlen - len(values))


def make_chunk_row(uid: str, fold: int, chunk: list[dict[str, int]], maxlen: int) -> dict[str, Any]:
    length = len(chunk)
    return {
        "fold": fold,
        "uid": uid,
        "questions": ",".join(pad([str(x["question"]) for x in chunk], maxlen)),
        "concepts": ",".join(pad([str(x["concept"]) for x in chunk], maxlen)),
        "responses": ",".join(pad([str(x["response"]) for x in chunk], maxlen)),
        "selectmasks": ",".join(pad([str(x["selectmask"]) for x in chunk], maxlen)),
        "orig_len": length,
    }


def make_full_row(row: dict[str, str], expanded: list[dict[str, int]]) -> dict[str, Any]:
    return {
        "fold": row.get("fold", -1),
        "uid": row.get("uid", ""),
        "questions": ",".join(str(x["question"]) for x in expanded),
        "concepts": ",".join(str(x["concept"]) for x in expanded),
        "responses": ",".join(str(x["response"]) for x in expanded),
        "selectmasks": ",".join(str(x["selectmask"]) for x in expanded),
        "orig_len": len(expanded),
    }


def is_missing_text(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "-1"}


def q_concept_lookup(q_path: Path) -> dict[int, str]:
    q = read_q_matrix(q_path)
    lookup: dict[int, str] = {}
    for qid in range(q.shape[0]):
        concepts = [str(idx) for idx, value in enumerate(q[qid].tolist()) if int(value) > 0]
        if concepts:
            lookup[qid] = "_".join(concepts)
    return lookup


def sort_timestamp(value: object) -> tuple[int, float | str]:
    if value is None:
        return (1, "")
    text = str(value).strip()
    if text == "":
        return (1, "")
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def make_sequence_row(uid: str, fold: int, records: list[dict[str, Any]]) -> dict[str, Any]:
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


def write_standard_sequences(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=STANDARD_SEQUENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_eedi_sequence_interactions_as_standard_files(
    source_dir: Path,
    output_dir: Path,
    learner_count: int,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Convert Eedi sequence_interactions.csv into train/test sequence files.

    The source Eedi directory does not contain train_sequences.csv/test_sequences.csv.
    Its sequence_interactions.csv already stores source_split=train_valid/test, and
    the test split uses ER learner ids 0..934. We keep that order so row i maps to
    uid{i} in the ER graph.
    """

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        manifest_path = output_dir / "eedi_standard_sequence_manifest.json"
        if manifest_path.exists():
            return output_dir, json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite.")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_path = source_dir / "Q.txt"
    if not q_path.exists():
        raise FileNotFoundError(f"missing Eedi Q matrix: {q_path}")
    concept_lookup = q_concept_lookup(q_path)
    seq_path = locate_sequence_interactions(source_dir)
    ensure_large_csv_field_limit()

    train_groups: dict[str, list[dict[str, Any]]] = {}
    test_groups: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(learner_count)}
    skipped = {"bad_question": 0, "missing_concept": 0, "bad_response": 0, "outside_test_uid": 0}
    row_count = 0
    with seq_path.open("r", encoding="utf-8", newline="") as fp:
        for order, row in enumerate(csv.DictReader(fp)):
            row_count += 1
            try:
                question = int(float(str(row.get("question", "")).strip()))
            except ValueError:
                skipped["bad_question"] += 1
                continue
            try:
                response = int(float(str(row.get("response", "")).strip()))
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
                "response": 1 if response > 0 else 0,
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

    train_rows = [
        make_sequence_row(uid, idx % 5, records)
        for idx, (uid, records) in enumerate(sorted(train_groups.items(), key=lambda kv: kv[0]))
        if records
    ]
    missing_test_uids = [idx for idx, records in test_groups.items() if not records]
    if missing_test_uids:
        raise ValueError(
            "Eedi sequence_interactions.csv does not contain test records for "
            f"{len(missing_test_uids)} ER learners. Examples: {missing_test_uids[:20]}"
        )
    test_rows = [make_sequence_row(str(idx), -1, test_groups[idx]) for idx in range(learner_count)]

    write_standard_sequences(output_dir / "train_sequences.csv", train_rows)
    write_standard_sequences(output_dir / "test_sequences.csv", test_rows)
    copy_file(q_path, output_dir / "Q.txt")

    manifest = {
        "dataset": "Eedi",
        "source_sequence_interactions": seq_path,
        "output_dir": output_dir,
        "row_count": row_count,
        "train_students": len(train_rows),
        "test_students": len(test_rows),
        "train_interactions": sum(int(row["orig_len"]) for row in train_rows),
        "test_interactions": sum(int(row["orig_len"]) for row in test_rows),
        "learner_alignment": "test_sequences row i maps to ER learner uidi",
        "concept_source": "use sequence_interactions.concepts when present; otherwise fill from Q.txt",
        "skipped": skipped,
    }
    write_json(output_dir / "eedi_standard_sequence_manifest.json", manifest)
    return output_dir, manifest


def convert_sequence_file(input_path: Path, chunked_path: Path, full_path: Path, maxlen: int, fold_count: int, is_test: bool) -> dict[str, Any]:
    ensure_large_csv_field_limit()
    chunk_index = 0
    kept = 0
    interactions = 0
    source_students = 0
    chunked_path.parent.mkdir(parents=True, exist_ok=True)
    with chunked_path.open("w", encoding="utf-8", newline="") as chunk_f, full_path.open("w", encoding="utf-8", newline="") as full_f:
        chunk_writer = csv.DictWriter(chunk_f, fieldnames=FIELDNAMES)
        full_writer = csv.DictWriter(full_f, fieldnames=FIELDNAMES)
        chunk_writer.writeheader()
        full_writer.writeheader()
        with input_path.open("r", encoding="utf-8", newline="") as input_f:
            for row_index, row in enumerate(csv.DictReader(input_f)):
                source_students += 1
                expanded = expand_row(row)
                if not expanded:
                    continue
                full_writer.writerow(make_full_row(row, expanded))
                kept += 1
                interactions += len(expanded)
                for start in range(0, len(expanded), maxlen):
                    chunk = expanded[start : start + maxlen]
                    fold = -1 if is_test else chunk_index % fold_count
                    row_uid = str(row.get("uid", row_index))
                    uid = f"{row_uid}_{start // maxlen}" if len(expanded) > maxlen else row_uid
                    chunk_writer.writerow(make_chunk_row(uid, fold, chunk, maxlen))
                    chunk_index += 1
    return {
        "input": input_path,
        "chunked": chunked_path,
        "full": full_path,
        "source_students": source_students,
        "full_students": kept,
        "chunks": chunk_index,
        "interactions": interactions,
    }


def prepare_dkt_sequences(source_graph_dir: Path, output_dir: Path, maxlen: int, fold_count: int, force: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        manifest_path = output_dir / "dkt_concept_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite.")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = convert_sequence_file(
        source_graph_dir / "train_sequences.csv",
        output_dir / "train_valid_sequences.csv",
        output_dir / "train_valid_sequences_full.csv",
        maxlen,
        fold_count,
        is_test=False,
    )
    test_summary = convert_sequence_file(
        source_graph_dir / "test_sequences.csv",
        output_dir / "test_sequences.csv",
        output_dir / "test_sequences_full.csv",
        maxlen,
        fold_count,
        is_test=True,
    )
    q = read_q_matrix(source_graph_dir / "Q.txt")
    manifest = {
        "source_dir": source_graph_dir,
        "output_dir": output_dir,
        "num_q": int(q.shape[0]),
        "num_c": int(q.shape[1]),
        "maxlen": maxlen,
        "fold_count": fold_count,
        "train": train_summary,
        "test": test_summary,
    }
    write_json(output_dir / "dkt_concept_manifest.json", manifest)
    return manifest


def ignore_pykt_copy(dir_name: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache"}
    if Path(dir_name).name == "examples":
        ignored.update({"saved_model", "wandb"})
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def copy_pykt(pykt_root: Path, work_root: Path, force: bool) -> Path:
    target = work_root / "pykt-toolkit-v11"
    if target.exists():
        if force:
            shutil.rmtree(target)
        else:
            return target
    shutil.copytree(pykt_root, target, ignore=ignore_pykt_copy)
    return target


def patch_pykt_for_pkc(pykt_copy: Path) -> dict[str, Any]:
    train_path = pykt_copy / "pykt" / "models" / "train_model.py"
    eval_path = pykt_copy / "pykt" / "models" / "evaluate_model.py"
    train_text = train_path.read_text(encoding="utf-8")
    old_train = (
        'elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:\n'
        "\n"
        "        y = torch.masked_select(ys[0], sm)\n"
        "        t = torch.masked_select(rshft, sm)\n"
        "        loss = binary_cross_entropy(y.double(), t.double())"
    )
    new_train = (
        'elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:\n'
        "\n"
        "        y = torch.masked_select(ys[0], sm)\n"
        "        if model_name == \"dkt\":\n"
        "            # V11 PKC-DKT: only supervise the next observed concept positions as positive labels.\n"
        "            t = torch.masked_select(torch.ones(size=rshft.shape, device=rshft.device, dtype=rshft.dtype), sm)\n"
        "        else:\n"
        "            t = torch.masked_select(rshft, sm)\n"
        "        loss = binary_cross_entropy(y.double(), t.double())"
    )
    if new_train not in train_text:
        if old_train not in train_text:
            raise RuntimeError(f"Cannot locate DKT loss block to patch in {train_path}")
        train_text = train_text.replace(old_train, new_train, 1)

    # pyKT selects the best checkpoint by validation AUC. V11 PKC-DKT sets all
    # observed next-concept labels to 1, so ROC-AUC is undefined. Initialize the
    # variables defensively and let the patched evaluator return accuracy as the
    # selection score for single-class labels.
    train_text = train_text.replace(
        "max_auc, best_epoch = 0, -1\n    train_step = 0",
        "max_auc, best_epoch = 0, -1\n    validauc, validacc = 0.0, 0.0\n    testauc, testacc = -1, -1\n    window_testauc, window_testacc = -1, -1\n    train_step = 0",
    )
    train_text = train_text.replace(
        "        if i - best_epoch >= 10:\n            break",
        "        # V11 PKC-DKT uses the last checkpoint for seq export, so do not early-stop by AUC.\n"
        "        if False and i - best_epoch >= 10:\n"
        "            break",
    )
    train_path.write_text(train_text, encoding="utf-8")

    eval_text = eval_path.read_text(encoding="utf-8")
    old_eval = "t = torch.masked_select(rshft, sm).detach().cpu()"
    new_eval = (
        "# V11 PKC-DKT evaluation target: next observed concept positions are positive labels.\n"
        "            t = torch.masked_select(torch.ones(size=rshft.shape, device=rshft.device, dtype=rshft.dtype), sm).detach().cpu()"
    )
    if new_eval not in eval_text:
        eval_text = eval_text.replace(old_eval, new_eval, 1)

    old_auc_block = (
        "        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)\n"
        "\n"
        "        prelabels = [1 if p >= 0.5 else 0 for p in ps]\n"
        "        acc = metrics.accuracy_score(ts, prelabels)"
    )
    new_auc_block = (
        "        prelabels = [1 if p >= 0.5 else 0 for p in ps]\n"
        "        acc = metrics.accuracy_score(ts, prelabels)\n"
        "        if len(np.unique(ts)) < 2:\n"
        "            # V11 PKC-DKT uses positive-only next-concept labels; ROC-AUC is undefined.\n"
        "            # Use accuracy as the validation selection score so training and checkpointing continue.\n"
        "            auc = float(acc)\n"
        "        else:\n"
        "            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)"
    )
    if new_auc_block not in eval_text:
        if old_auc_block not in eval_text:
            raise RuntimeError(f"Cannot locate evaluate AUC block to patch in {eval_path}")
        eval_text = eval_text.replace(old_auc_block, new_auc_block, 1)
    eval_path.write_text(eval_text, encoding="utf-8")
    return {"train_model": train_path, "evaluate_model": eval_path}


def register_dataset(pykt_copy: Path, dataset_name: str, dkt_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config_path = pykt_copy / "configs" / "data_config.json"
    data_config = json.loads(config_path.read_text(encoding="utf-8"))
    examples_dir = pykt_copy / "examples"
    relative_dpath = os.path.relpath(dkt_dir, examples_dir).replace("\\", "/")
    data_config[dataset_name] = {
        "dpath": relative_dpath,
        "num_q": int(manifest["num_q"]),
        "num_c": int(manifest["num_c"]),
        "input_type": ["concepts"],
        "max_concepts": 1,
        "min_seq_len": 3,
        "maxlen": int(manifest["maxlen"]),
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": list(range(int(manifest["fold_count"]))),
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_sequences.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences.csv",
        "test_file_quelevel": "test_sequences.csv",
        "test_window_file_quelevel": "test_sequences.csv",
        "test_original_file_quelevel": "test_sequences.csv",
    }
    config_path.write_text(json.dumps(data_config, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    return {"dataset_name": dataset_name, "config_path": config_path, "dpath": relative_dpath}


def run_pykt_train(pykt_copy: Path, params: dict[str, Any]) -> Path:
    examples_dir = pykt_copy / "examples"
    old_cwd = Path.cwd()
    sys.path.insert(0, str(pykt_copy))
    sys.path.insert(0, str(examples_dir))
    try:
        os.chdir(examples_dir)
        import wandb_train

        wandb_train.main(params)
    finally:
        os.chdir(old_cwd)
    params_str = "_".join([str(v) for k, v in params.items() if k not in ["other_config"]])
    return examples_dir / params["save_dir"] / params_str


def export_seq_with_checkpoint(pykt_copy: Path, checkpoint_dir: Path, test_sequences: Path, output_file: Path, expected_students: int, expected_concepts: int, device: str) -> dict[str, Any]:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from export_seq_from_dkt_v11 import export_seq, load_pykt_model

    last_checkpoint = checkpoint_dir / "qid_last_model.ckpt"
    checkpoint_file = "qid_last_model.ckpt" if last_checkpoint.exists() else "qid_model.ckpt"
    model, data_config = load_pykt_model(pykt_copy, checkpoint_dir, device, checkpoint_file)
    summary = export_seq(
        model,
        data_config,
        test_sequences,
        output_file,
        device,
        expected_students=expected_students,
        expected_concepts=expected_concepts,
    )
    summary["checkpoint_file"] = checkpoint_file
    summary["checkpoint_policy"] = "prefer_last_model_for_v11_pkc_dkt"
    return summary

def main() -> None:
    args = parse_args()
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    source_graph_dir = source_dir / "prepared_for_kt" if (source_dir / "prepared_for_kt" / "Q.txt").exists() else source_dir
    v11_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    dkt_dir = v11_dir / "pykt_dkt" / "dkt_concept"
    work_root = v11_dir / "pykt_dkt" / "work"
    checkpoint_manifest = v11_dir / "pykt_dkt" / "checkpoint_manifest.json"

    if args.dataset == "Eedi" and not (source_graph_dir / "train_sequences.csv").exists():
        mirt_manifest_path = v11_dir / "mirt" / "inputs" / "mirt_input_manifest.json"
        if not mirt_manifest_path.exists():
            raise FileNotFoundError(
                f"missing {mirt_manifest_path}. Run prepare_mirt_inputs_v11.py before train_pykt_dkt_v11.py."
            )
        mirt_manifest = json.loads(mirt_manifest_path.read_text(encoding="utf-8"))
        learner_count = int(mirt_manifest["user_num"])
        source_graph_dir, eedi_sequence_manifest = prepare_eedi_sequence_interactions_as_standard_files(
            source_dir,
            v11_dir / "pykt_dkt" / "source_sequences",
            learner_count,
            args.force,
        )
    else:
        eedi_sequence_manifest = None

    manifest = prepare_dkt_sequences(source_graph_dir, dkt_dir, args.maxlen, args.fold_count, args.force)
    pykt_copy = copy_pykt(args.pykt_root, work_root, args.force)
    patch_report = patch_pykt_for_pkc(pykt_copy)
    dataset_name = f"{args.dataset}_v11_pkc_dkt"
    register_report = register_dataset(pykt_copy, dataset_name, dkt_dir, manifest)

    checkpoint_dir: Path
    if args.skip_train:
        if not checkpoint_manifest.exists():
            raise FileNotFoundError(f"missing checkpoint manifest for --skip-train: {checkpoint_manifest}")
        checkpoint_dir = Path(json.loads(checkpoint_manifest.read_text(encoding="utf-8"))["checkpoint_dir"])
    else:
        params = {
            "dataset_name": dataset_name,
            "model_name": "dkt",
            "emb_type": "qid",
            "save_dir": "saved_model_v11",
            "seed": args.seed,
            "fold": args.fold,
            "dropout": args.dropout,
            "emb_size": args.emb_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "use_wandb": 0,
            "add_uuid": 0,
        }
        checkpoint_dir = run_pykt_train(pykt_copy, params)
        write_json(
            checkpoint_manifest,
            {
                "dataset": args.dataset,
                "dataset_name": dataset_name,
                "checkpoint_dir": checkpoint_dir,
                "best_checkpoint_file": "qid_model.ckpt",
                "last_checkpoint_file": "qid_last_model.ckpt",
                "export_checkpoint_policy": "prefer qid_last_model.ckpt because V11 PKC-DKT validation AUC is not informative",
                "params": params,
                "patch_report": patch_report,
                "register_report": register_report,
                "eedi_sequence_manifest": eedi_sequence_manifest,
            },
        )

    output_file = v11_dir / "pykt_dkt" / "stu2know_seq.json"
    summary = export_seq_with_checkpoint(
        pykt_copy,
        checkpoint_dir,
        dkt_dir / "test_sequences_full.csv",
        output_file,
        expected_students=int(manifest["test"]["full_students"]),
        expected_concepts=int(manifest["num_c"]),
        device=args.device,
    )
    (v11_dir / "stu2know_seq.json").write_text(output_file.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        v11_dir / "pykt_dkt" / "seq_export_manifest.json",
        {
            "dataset": args.dataset,
            "source": "V11 patched pyKT DKT; next observed concept labels are set to 1",
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_file": summary.get("checkpoint_file"),
            "checkpoint_policy": summary.get("checkpoint_policy"),
            "output_file": output_file,
            "root_copy": pykt_copy,
            "summary": summary,
        },
    )
    print(json.dumps({"checkpoint_dir": str(checkpoint_dir), **summary}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
