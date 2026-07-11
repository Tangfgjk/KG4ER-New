from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import output_data_root, project_root, v13_dataset_dir, write_json


def default_educdm_root() -> Path:
    return project_root() / "EduCDM_MIRT_noQ_export_modified" / "EduCDM-main"


def read_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "item_id", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["user_id", "item_id", "score"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["score"] = df["score"].astype(float)
    bad_scores = set(df["score"].unique()) - {0.0, 1.0}
    if bad_scores:
        raise ValueError(f"{path} score must be 0/1. Unexpected values: {sorted(bad_scores)}")
    return df


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(df["user_id"].values, dtype=torch.long),
        torch.tensor(df["item_id"].values, dtype=torch.long),
        torch.tensor(df["score"].values, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modified EduCDM no-Q MIRT for v13 feature estimation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--educdm-root", type=Path, default=default_educdm_root())
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--a-range", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.educdm_root) not in sys.path:
        sys.path.insert(0, str(args.educdm_root))
    from EduCDM.MIRT.MIRT import MIRT

    v13_dir = v13_dataset_dir(args.dataset, args.output_data_root)
    input_dir = args.input_dir or (v13_dir / "mirt" / "inputs")
    output_dir = args.output_dir or (v13_dir / "mirt" / "outputs")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((input_dir / "mirt_input_manifest.json").read_text(encoding="utf-8"))
    user_num = int(manifest["user_num"])
    item_num = int(manifest["item_num"])
    q = pd.read_csv(input_dir / "Q_matrix.csv", header=None)
    latent_dim = int(args.latent_dim or q.shape[1])

    train_df = read_interactions(input_dir / "train.csv")
    valid_df = read_interactions(input_dir / "valid.csv") if (input_dir / "valid.csv").exists() else None
    test_df = read_interactions(input_dir / "test.csv") if (input_dir / "test.csv").exists() else None
    train_loader = make_loader(train_df, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_df, args.batch_size, shuffle=False) if valid_df is not None and len(valid_df) else None
    test_loader = make_loader(test_df, args.batch_size, shuffle=False) if test_df is not None and len(test_df) else None

    cdm = MIRT(user_num=user_num, item_num=item_num, latent_dim=latent_dim, a_range=args.a_range)
    cdm.train(train_loader, valid_loader, epoch=args.epoch, device=args.device, lr=args.lr, weight_decay=args.weight_decay)
    cdm.save(output_dir / "mirt_no_q.params")
    metrics = {}
    if test_loader is not None:
        auc, accuracy = cdm.eval(test_loader, device=args.device)
        metrics = {"test_auc": float(auc), "test_accuracy": float(accuracy)}
        print(f"test_auc={auc:.6f} test_accuracy={accuracy:.6f}")

    a_path = output_dir / f"a_param_{latent_dim}.csv"
    b_path = output_dir / f"b_param_{latent_dim}.csv"
    theta_path = output_dir / f"theta_param_{latent_dim}.csv"
    cdm.export_item_params(a_path, b_path, theta_path=theta_path, device=args.device, transformed=True)
    write_json(
        output_dir / "mirt_export_manifest.json",
        {
            "dataset": args.dataset,
            "version": "v13_graph_subset_no_q_mirt",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "educdm_root": args.educdm_root,
            "user_num": user_num,
            "item_num": item_num,
            "latent_dim": latent_dim,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "a_range": args.a_range,
            "device": args.device,
            "files": {
                "a_param": a_path.name,
                "b_param": b_path.name,
                "theta_param": theta_path.name,
                "model": "mirt_no_q.params",
            },
            "metrics": metrics,
            "notes": "MIRT is used as a pedagogical feature estimator on ER graph learners, not as the final recommendation model.",
        },
    )
    print(f"saved v13 MIRT outputs: {output_dir}")


if __name__ == "__main__":
    main()

