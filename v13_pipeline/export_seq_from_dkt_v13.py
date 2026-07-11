from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from common import output_data_root, v13_dataset_dir, write_json


def parse_ints(value: object) -> list[int]:
    if pd.isna(value):
        return []
    return [int(token) for token in str(value).split(",") if token not in ("", "-1")]


def load_pykt_model(pykt_root: Path, checkpoint_dir: Path, device: str, checkpoint_file: str | None = None):
    pykt_root = pykt_root.resolve()
    if str(pykt_root) not in sys.path:
        sys.path.insert(0, str(pykt_root))

    from pykt.models import init_model
    from pykt.models.torch_io import torch_load_file

    checkpoint_dir = checkpoint_dir.resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing pyKT config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = copy.deepcopy(config["model_config"])
    for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
        model_config.pop(remove_item, None)
    params = config["params"]
    model_name = params["model_name"]
    emb_type = params["emb_type"]
    if model_name != "dkt":
        raise ValueError(f"v13 seq exporter expects pyKT DKT, got model_name={model_name!r}")
    model = init_model(model_name, model_config, config["data_config"], emb_type)
    ckpt_name = checkpoint_file or f"{emb_type}_model.ckpt"
    ckpt_path = checkpoint_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint file: {ckpt_path}")
    net = torch_load_file(ckpt_path, map_location=device)
    model.load_state_dict(net)
    model.to(device)
    model.eval()
    return model, config["data_config"]


@torch.no_grad()
def export_seq(
    model,
    data_config: dict,
    test_sequences: Path,
    output_file: Path,
    device: str,
    expected_students: int | None = None,
    expected_concepts: int | None = None,
) -> dict:
    num_c = int(data_config["num_c"])
    if expected_concepts is not None and num_c != expected_concepts:
        raise ValueError(f"checkpoint num_c={num_c}, expected_concepts={expected_concepts}")
    df = pd.read_csv(test_sequences)
    rows: list[list[float]] = []
    for _, row in df.iterrows():
        concepts = parse_ints(row["concepts"])
        responses = parse_ints(row["responses"])
        usable = min(len(concepts), len(responses))
        concepts = concepts[:usable]
        responses = responses[:usable]
        if usable == 0:
            rows.append([0.0] * num_c)
            continue
        if max(concepts) >= num_c:
            raise ValueError(f"concept id {max(concepts)} is outside checkpoint num_c={num_c}")
        c = torch.tensor([concepts], dtype=torch.long, device=device)
        r = torch.tensor([responses], dtype=torch.long, device=device)
        y = model(c, r)
        rows.append([round(float(x), 6) for x in y[0, usable - 1, :].detach().cpu().tolist()])
    if expected_students is not None and len(rows) != expected_students:
        raise ValueError(f"exported students={len(rows)}, expected_students={expected_students}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return {"output_file": output_file, "students": len(rows), "concepts": num_c}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v13 stu2know_seq.json from a trained pyKT DKT checkpoint.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--pykt-root", type=Path, default=Path("ER/pykt-toolkit-main"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--test-sequences", type=Path, required=True)
    parser.add_argument("--expected-students", type=int, default=None)
    parser.add_argument("--expected-concepts", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    output_file = output_dir / "kt_dkt" / "stu2know_seq.json"
    model, data_config = load_pykt_model(args.pykt_root, args.checkpoint_dir, device, args.checkpoint_file)
    summary = export_seq(
        model,
        data_config,
        args.test_sequences,
        output_file,
        device,
        expected_students=args.expected_students,
        expected_concepts=args.expected_concepts,
    )
    write_json(
        output_dir / "kt_dkt" / "seq_export_manifest.json",
        {
            "dataset": args.dataset,
            "source": "pyKT DKT",
            "pykt_root": args.pykt_root,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_file": args.checkpoint_file,
            "test_sequences": args.test_sequences,
            "device": device,
            **summary,
        },
    )
    # Keep the root graph file in sync for build_v13_graph.py.
    (output_dir / "stu2know_seq.json").write_text(output_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

