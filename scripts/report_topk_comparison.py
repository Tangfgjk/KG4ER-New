"""Create a cross-model top-K table and line chart from completed ER runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


METRICS = ("ACC", "NOV")
TOP_KS = tuple(range(10, 101, 10))


def parse_run_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=RUN_DIR")
    return label, Path(raw_path)


def load_rows(label: str, run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("**/eval/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", path.parents[2].name))
        for metric in METRICS:
            values = payload.get(metric, {})
            for top_k in TOP_KS:
                item = values.get(str(top_k), {})
                value = item.get("mean") if isinstance(item, dict) else None
                if isinstance(value, (int, float)):
                    rows.append(
                        {
                            "group": label,
                            "model": model,
                            "metric": metric,
                            "top_k": top_k,
                            "value": float(value),
                            "source": str(path),
                        }
                    )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["model"], row["metric"], row["top_k"])].append(row["value"])

    output = []
    for (group, model, metric, top_k), values in sorted(grouped.items()):
        output.append(
            {
                "group": group,
                "model": model,
                "metric": metric,
                "top_k": top_k,
                "mean": mean(values),
                "runs": len(values),
            }
        )
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "model", "metric", "top_k", "mean", "runs"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Top-K Result Comparison", ""]
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        names = sorted({(row["group"], row["model"]) for row in metric_rows})
        lookup = {(row["group"], row["model"], row["top_k"]): row for row in metric_rows}
        lines.extend([f"## {metric}", "", "| Group | Model | Runs | " + " | ".join(f"@{k}" for k in TOP_KS) + " |", "| --- | --- | ---: | " + " | ".join("---:" for _ in TOP_KS) + " |"])
        for group, model in names:
            values = [lookup.get((group, model, k)) for k in TOP_KS]
            runs = max((item["runs"] for item in values if item), default=0)
            formatted = [f"{item['mean']:.6f}" if item else "" for item in values]
            lines.append(f"| {group} | {model} | {runs} | " + " | ".join(formatted) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180, constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        names = sorted({(row["group"], row["model"]) for row in metric_rows})
        for group, model in names:
            values = {row["top_k"]: row["mean"] for row in metric_rows if row["group"] == group and row["model"] == model}
            xs = [k for k in TOP_KS if k in values]
            ys = [values[k] for k in xs]
            axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8, label=f"{group}: {model}")
        axis.set_title(metric)
        axis.set_xlabel("Recommendation list size N")
        axis.set_ylabel(metric)
        axis.set_xticks(TOP_KS)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a table and top-K curves for completed ER runs.")
    parser.add_argument("--run-dir", action="append", required=True, type=parse_run_dir, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for label, run_dir in args.run_dir:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        rows.extend(load_rows(label, run_dir))
    if not rows:
        raise ValueError("No eval/metrics.json files were found in the supplied run directories.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(summary, output_dir / "topk_comparison.csv")
    write_markdown(summary, output_dir / "topk_comparison.md")
    plot(summary, output_dir / "topk_comparison.png")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "metric_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
