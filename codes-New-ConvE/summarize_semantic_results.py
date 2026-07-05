"""Summarize SemanticConvE experiment outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List

from semantic_experiment_utils import (
    DEFAULT_TOP_KS,
    VALID_ABLATIONS,
    ablation_model_dir,
    default_runs_root,
    parse_ablation_list,
    parse_seed_list,
    parse_top_ks,
    read_json,
    write_json,
)


AVG_WEIGHTS = {
    10: 0.05,
    15: 0.05,
    20: 0.05,
    30: 0.10,
    50: 0.15,
    75: 0.25,
    100: 0.35,
}


def maybe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def weighted_avg(values: Dict[int, float]) -> float | None:
    present = [k for k in AVG_WEIGHTS if k in values]
    if not present:
        return None
    weight_sum = sum(AVG_WEIGHTS[k] for k in present)
    return sum(values[k] * AVG_WEIGHTS[k] for k in present) / weight_sum


def sample_std(values: List[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def metric_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    try:
        return float(metrics[metric][str(top_k)]["mean"])
    except Exception:
        return None


def collect_seed(run_dir: Path, seed: int, top_ks: List[int], ablation: str = "full") -> Dict[str, Any] | None:
    seed_dir = run_dir / ablation_model_dir(ablation) / f"seed{seed}"
    eval_metrics = maybe_read_json(seed_dir / "eval" / "metrics.json")
    if not eval_metrics:
        return None
    train_metrics = maybe_read_json(seed_dir / "metrics.json")
    inference_metrics = maybe_read_json(seed_dir / "semantic_conve_inference.json")
    row: Dict[str, Any] = {
        "ablation": ablation,
        "model": ablation_model_dir(ablation),
        "seed": seed,
        "seed_dir": str(seed_dir),
        "training_seconds": train_metrics.get("training_seconds"),
        "inference_seconds": inference_metrics.get("inference_seconds"),
        "Ep_sim@10": eval_metrics.get("Ep_sim", {}).get("mean"),
    }
    acc_values: Dict[int, float] = {}
    nov_values: Dict[int, float] = {}
    for top_k in top_ks:
        acc = metric_value(eval_metrics, "ACC", top_k)
        nov = metric_value(eval_metrics, "NOV", top_k)
        row[f"ACC@{top_k}"] = acc
        row[f"NOV@{top_k}"] = nov
        if acc is not None:
            acc_values[top_k] = acc
        if nov is not None:
            nov_values[top_k] = nov
    row["ACC-Avg"] = weighted_avg(acc_values)
    row["NOV-Avg"] = weighted_avg(nov_values)
    return row


def collect_gate_rows(run_dir: Path, seeds: List[int], ablations: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ablation in ablations:
        model_name = ablation_model_dir(ablation)
        for seed in seeds:
            seed_dir = run_dir / model_name / f"seed{seed}"
            gate_payload = maybe_read_json(seed_dir / "gate_values.json")
            gate_values = gate_payload.get("gate_values", {})
            if not isinstance(gate_values, dict):
                continue
            for entity_type, feature_values in gate_values.items():
                if not isinstance(feature_values, dict):
                    continue
                for feature_type, value in feature_values.items():
                    try:
                        gate_value = float(value)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "ablation": ablation,
                            "model": model_name,
                            "seed": seed,
                            "seed_dir": str(seed_dir),
                            "entity_type": entity_type,
                            "feature_type": feature_type,
                            "gate_value": gate_value,
                        }
                    )
    return rows


def summarize_gate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[float]] = {}
    for row in rows:
        key = (
            str(row["ablation"]),
            str(row["model"]),
            str(row["entity_type"]),
            str(row["feature_type"]),
        )
        grouped.setdefault(key, []).append(float(row["gate_value"]))
    summaries = []
    for (ablation, model, entity_type, feature_type), values in sorted(grouped.items()):
        summaries.append(
            {
                "ablation": ablation,
                "model": model,
                "entity_type": entity_type,
                "feature_type": feature_type,
                "runs": len(values),
                "mean": sum(values) / len(values),
                "std": sample_std(values),
            }
        )
    return summaries


def summarize_rows(rows: List[Dict[str, Any]], top_ks: List[int]) -> Dict[str, Any]:
    fields = [f"ACC@{k}" for k in top_ks] + ["ACC-Avg"] + [f"NOV@{k}" for k in top_ks] + ["NOV-Avg", "Ep_sim@10", "training_seconds", "inference_seconds"]
    summary: Dict[str, Any] = {"runs": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_mean"] = sum(values) / len(values) if values else None
        summary[f"{field}_std"] = sample_std(values) if values else None
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_paper_table(path: Path, dataset: str, summaries: Dict[str, Dict[str, Any]], top_ks: List[int]) -> None:
    headers = ["Dataset", "Model", "Runs", "ACC-Avg Mean", "ACC-Avg Std", "NOV-Avg Mean", "NOV-Avg Std", "Ep_sim@10 Mean"]
    lines = [
        f"# {dataset} SemanticConvE Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for ablation, summary in summaries.items():
        row = [
            dataset,
            ablation_model_dir(ablation),
            str(summary["runs"]),
            fmt(summary.get("ACC-Avg_mean")),
            fmt(summary.get("ACC-Avg_std")),
            fmt(summary.get("NOV-Avg_mean")),
            fmt(summary.get("NOV-Avg_std")),
            fmt(summary.get("Ep_sim@10_mean")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "Weighted Avg uses weights `0.05, 0.05, 0.05, 0.10, 0.15, 0.25, 0.35` for K=`10,15,20,30,50,75,100`.",
            "",
            "## Full Metrics",
            "",
        ]
    )
    metric_headers = ["Metric"] + [f"@{k} Mean" for k in top_ks] + [f"@{k} Std" for k in top_ks]
    for ablation, summary in summaries.items():
        lines.append(f"### {ablation_model_dir(ablation)}")
        lines.append("")
        lines.append("| " + " | ".join(metric_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(metric_headers)) + " |")
        for metric in ["ACC", "NOV"]:
            metric_row = [metric]
            metric_row.extend(fmt(summary.get(f"{metric}@{k}_mean")) for k in top_ks)
            metric_row.extend(fmt(summary.get(f"{metric}@{k}_std")) for k in top_ks)
            lines.append("| " + " | ".join(metric_row) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_table_csv(path: Path, dataset: str, summaries: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for ablation, summary in summaries.items():
        rows.append(
            {
            "Dataset": dataset,
            "Model": ablation_model_dir(ablation),
            "Ablation": ablation,
            "Runs": summary["runs"],
            "ACC-Avg Mean": summary.get("ACC-Avg_mean"),
            "ACC-Avg Std": summary.get("ACC-Avg_std"),
            "NOV-Avg Mean": summary.get("NOV-Avg_mean"),
            "NOV-Avg Std": summary.get("NOV-Avg_std"),
            "Ep_sim@10 Mean": summary.get("Ep_sim@10_mean"),
            "Training Seconds Mean": summary.get("training_seconds_mean"),
            "Inference Seconds Mean": summary.get("inference_seconds_mean"),
            }
        )
    write_csv(path, rows, list(rows[0].keys()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SemanticConvE runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--ablations", default="full")
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    ablations = parse_ablation_list(args.ablations)
    invalid_ablations = sorted(set(ablations) - set(VALID_ABLATIONS))
    if invalid_ablations:
        raise ValueError(f"Unknown ablations: {','.join(invalid_ablations)}")
    top_ks = parse_top_ks(args.top_ks)
    run_dir = args.runs_root / args.dataset / args.run_id
    output_dir = args.output_dir or (run_dir / "summary")
    rows = []
    missing: Dict[str, List[int]] = {}
    for ablation in ablations:
        for seed in seeds:
            row = collect_seed(run_dir, seed, top_ks, ablation)
            if row is None:
                missing.setdefault(ablation, []).append(seed)
            else:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed SemanticConvE metrics found under {run_dir}")

    summaries = {
        ablation: summarize_rows([row for row in rows if row["ablation"] == ablation], top_ks)
        for ablation in ablations
        if any(row["ablation"] == ablation for row in rows)
    }
    fields = ["ablation", "model", "seed", "seed_dir"] + [f"ACC@{k}" for k in top_ks] + ["ACC-Avg"] + [f"NOV@{k}" for k in top_ks] + ["NOV-Avg", "Ep_sim@10", "training_seconds", "inference_seconds"]
    write_csv(output_dir / "per_seed_metrics.csv", rows, fields)

    gate_rows = collect_gate_rows(run_dir, seeds, ablations)
    gate_summary_rows = summarize_gate_rows(gate_rows)
    if gate_rows:
        write_csv(
            output_dir / "gate_values_per_seed.csv",
            gate_rows,
            ["ablation", "model", "seed", "seed_dir", "entity_type", "feature_type", "gate_value"],
        )
        write_csv(
            output_dir / "gate_values_mean_std.csv",
            gate_summary_rows,
            ["ablation", "model", "entity_type", "feature_type", "runs", "mean", "std"],
        )

    summary_rows = []
    for ablation, summary in summaries.items():
        for key, value in summary.items():
            if key.endswith("_mean"):
                summary_rows.append(
                    {
                        "ablation": ablation,
                        "model": ablation_model_dir(ablation),
                        "metric": key.removesuffix("_mean"),
                        "mean": value,
                        "std": summary.get(key.replace("_mean", "_std")),
                    }
                )
    write_csv(output_dir / "mean_std_metrics.csv", summary_rows, ["ablation", "model", "metric", "mean", "std"])
    write_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "run_id": args.run_id,
            "missing_seeds": missing,
            "summaries": summaries,
            "rows": rows,
            "gate_summaries": gate_summary_rows,
        },
    )
    write_paper_table(output_dir / "paper_table.md", args.dataset, summaries, top_ks)
    write_paper_table_csv(output_dir / "paper_table.csv", args.dataset, summaries)
    print(f"summary written to {output_dir}")


if __name__ == "__main__":
    main()
