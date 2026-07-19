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
    def model_key(model: str) -> str:
        if model == "SemanticConvE":
            return "full"
        return model.replace("SemanticConvE_", "")

    semantic_order = ("full", "id_only", "no_learner_id", "no_learner_relation_id", "feature_only")
    model_colors = {
        "TransE": "#111111",
        "TransE-adv": "#6A6A6A",
        "full": "#0072B2",
        "id_only": "#E69F00",
        "no_learner_id": "#D55E00",
        "no_learner_relation_id": "#CC79A7",
        "feature_only": "#009E73",
    }
    style_by_group = {
        "Comparison": ("-", "o"),
        "SemanticConvE include-test": ("-", "^"),
        "SemanticConvE exclude-test": ("--", "s"),
    }

    def series_label(group: str, model: str) -> str:
        key = model_key(model)
        if group == "SemanticConvE include-test":
            return f"{key} (with test edges)"
        if group == "SemanticConvE exclude-test":
            return f"{key} (without test edges)"
        return model

    group_order = {"Comparison": 0, "SemanticConvE include-test": 1, "SemanticConvE exclude-test": 2}
    series = sorted(
        {(row["group"], row["model"]) for row in rows},
        key=lambda item: (
            0 if item[0] == "Comparison" else 1,
            semantic_order.index(model_key(item[1])) if model_key(item[1]) in semantic_order else len(semantic_order),
            group_order.get(item[0], 99),
            item[1],
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(19, 7), dpi=180)
    for axis, metric in zip(axes, METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        for group, model in series:
            values = {row["top_k"]: row["mean"] for row in metric_rows if row["group"] == group and row["model"] == model}
            xs = [k for k in TOP_KS if k in values]
            ys = [values[k] for k in xs]
            if not xs:
                continue
            linestyle, marker = style_by_group.get(group, ("-", "D"))
            axis.plot(
                xs,
                ys,
                color=model_colors.get(model_key(model), "#444444"),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=4.2,
                label=series_label(group, model),
            )
        axis.set_title(metric, fontsize=14, weight="bold")
        axis.set_xlabel("Recommendation list size N")
        axis.set_ylabel(metric)
        axis.set_xticks(TOP_KS)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Curve key", fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(0.84, 0.5))
    figure.subplots_adjust(left=0.06, right=0.82, bottom=0.13, top=0.90, wspace=0.20)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
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
