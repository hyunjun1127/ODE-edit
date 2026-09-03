"""Deterministic code-only figures for CounterFact primary metrics v6."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DPI = 180
FIGSIZE = (12.0, 7.0)
ARM_ORDER = ("LM", "LA", "QM", "QA")
ARM_LABEL = {
    "LM": "Llama / MEMIT",
    "LA": "Llama / AlphaEdit",
    "QM": "Qwen / MEMIT",
    "QA": "Qwen / AlphaEdit",
}
ARM_COLOR = {
    "LM": "#4c78a8",
    "LA": "#f58518",
    "QM": "#54a24b",
    "QA": "#e45756",
}
PRIMARY = (("rs", "RS"), ("ps", "PS"), ("ns", "NS"))
SECONDARY = (
    ("rewrite_acc", "rewrite_acc"),
    ("rephrase_acc", "rephrase_acc"),
    ("rephrase_acc_strict_all_2", "rephrase_acc strict-all-2"),
    ("neighborhood_target_true_teacher_forced_acc", "neighborhood target-true TF acc"),
)
AGE_ORDER = ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT")
AGE_LABEL = {
    "EARLY_FIRST_20PCT": "early 20%",
    "MIDDLE_60PCT": "middle 60%",
    "RECENT_LAST_20PCT": "recent 20%",
}
AGE_STYLE = {
    "EARLY_FIRST_20PCT": "--",
    "MIDDLE_60PCT": "-",
    "RECENT_LAST_20PCT": ":",
}
PNG_METADATA = {"Software": "ODE-edit lifelong_counterfact_figures.py"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
        }
    )


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata=PNG_METADATA, bbox_inches="tight")
    plt.close(fig)


def final_primary(tables: Path, output: Path) -> Path:
    rows = {row["arm"]: row for row in _rows(tables / "counterfact-finalw-full10k.csv")}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(PRIMARY), dtype=float)
    width = 0.19
    for index, arm in enumerate(ARM_ORDER):
        ax.bar(
            x + (index - 1.5) * width,
            [float(rows[arm][metric]) for metric, _ in PRIMARY],
            width,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
        )
    ax.set_title("Final W₁₀₀₀₀ CounterFact Primary Metrics")
    ax.set_ylabel("Prompt-level strict NLL-pair success rate")
    ax.set_xticks(x, [label for _, label in PRIMARY])
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, frameon=False)
    path = output / "counterfact-finalw-full10k-primary.png"
    _save(fig, path)
    return path


def cumulative_primary(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "counterfact-primary-checkpoints.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharex=True, sharey=True)
    for axis, (metric, label) in zip(axes, PRIMARY, strict=True):
        for arm in ARM_ORDER:
            selected = sorted(
                (row for row in rows if row["arm"] == arm),
                key=lambda row: int(row["accepted_edit_count"]),
            )
            axis.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row[metric]) for row in selected],
                color=ARM_COLOR[arm],
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=ARM_LABEL[arm],
            )
        axis.set_title(label)
        axis.set_xlabel("Accepted edits in frozen checkpoint")
        axis.set_ylabel("Strict NLL-pair success rate")
        axis.set_ylim(0.0, 1.0)
    axes[0].legend(ncol=2, frameon=False)
    path = output / "counterfact-cumulative-primary.png"
    _save(fig, path)
    return path


def secondary_accuracy(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "counterfact-primary-checkpoints.csv")
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for axis, (metric, label) in zip(axes.flat, SECONDARY, strict=True):
        for arm in ARM_ORDER:
            selected = sorted(
                (row for row in rows if row["arm"] == arm),
                key=lambda row: int(row["accepted_edit_count"]),
            )
            axis.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row[metric]) for row in selected],
                color=ARM_COLOR[arm],
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                label=ARM_LABEL[arm],
            )
        axis.set_title(label)
        axis.set_xlabel("Accepted edits in frozen checkpoint")
        axis.set_ylabel("Teacher-forced strict accuracy")
        axis.set_ylim(0.0, 1.0)
    axes[0, 0].legend(ncol=2, frameon=False)
    path = output / "counterfact-secondary-accuracy.png"
    _save(fig, path)
    return path


def age_strata(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "counterfact-primary-age-strata.csv")
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 12.0), sharex=True, sharey=True)
    for row_index, arm in enumerate(ARM_ORDER):
        for column_index, (metric, label) in enumerate(PRIMARY):
            axis = axes[row_index, column_index]
            for age in AGE_ORDER:
                selected = sorted(
                    (row for row in rows if row["arm"] == arm and row["age_stratum"] == age),
                    key=lambda row: int(row["accepted_edit_count"]),
                )
                axis.plot(
                    [int(row["accepted_edit_count"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    linestyle=AGE_STYLE[age],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.25,
                    color=ARM_COLOR[arm],
                    label=AGE_LABEL[age],
                )
            axis.set_title(f"{arm} — {label}")
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("Accepted edits")
            axis.set_ylabel("Success rate")
    axes[0, 0].legend(frameon=False)
    path = output / "counterfact-age-strata-primary.png"
    _save(fig, path)
    return path


def nll_advantage(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "counterfact-primary-distributions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharex=True)
    for axis, (metric, label) in zip(axes, PRIMARY, strict=True):
        selected_metric = [row for row in rows if row["metric"] == metric.upper()]
        for arm in ARM_ORDER:
            selected = sorted(
                (row for row in selected_metric if row["arm"] == arm),
                key=lambda row: int(row["accepted_edit_count"]),
            )
            axis.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row["nll_advantage_median"]) for row in selected],
                color=ARM_COLOR[arm],
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                label=f"{arm} median",
            )
            axis.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row["nll_advantage_p90"]) for row in selected],
                color=ARM_COLOR[arm],
                linestyle=":",
                linewidth=1.0,
                label=f"{arm} p90",
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(f"{label} NLL advantage")
        axis.set_xlabel("Accepted edits")
        axis.set_ylabel("Desired minus undesired advantage")
    axes[0].legend(ncol=2, frameon=False, fontsize=7)
    path = output / "counterfact-cumulative-nll-advantage.png"
    _save(fig, path)
    return path


def mechanism_association(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "counterfact-mechanism-associations.csv")
    metrics = sorted({row["mechanism_metric"] for row in rows})
    outcomes = ("rs", "ps", "ns")
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.0), sharey=True)
    for axis, arm in zip(axes, ARM_ORDER, strict=True):
        indexed = {
            (row["mechanism_metric"], row["outcome_metric"]): float(row["spearman"])
            for row in rows
            if row["arm"] == arm
        }
        matrix = np.asarray([[indexed[(metric, outcome)] for outcome in outcomes] for metric in metrics])
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        axis.set_title(arm)
        axis.set_xticks(range(len(outcomes)), [value.upper() for value in outcomes])
        axis.set_yticks(range(len(metrics)), metrics if axis is axes[0] else [""] * len(metrics))
    fig.colorbar(image, ax=axes, shrink=0.78, label="Spearman (n=7 checkpoints/arm)")
    path = output / "counterfact-mechanism-associations.png"
    fig.savefig(path, dpi=DPI, metadata=PNG_METADATA, bbox_inches="tight")
    plt.close(fig)
    return path


def generate(tables: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    _style()
    return [
        final_primary(tables, output),
        cumulative_primary(tables, output),
        secondary_accuracy(tables, output),
        age_strata(tables, output),
        nll_advantage(tables, output),
        mechanism_association(tables, output),
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.tables, args.output):
        print(f"{path.name}\t{sha256_file(path)}")


if __name__ == "__main__":
    main()
