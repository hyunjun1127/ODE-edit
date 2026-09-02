"""Deterministic, code-only plots for lifelong frozen-weight evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

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
METRICS = (
    ("eff", "Eff (rewrite strict)"),
    ("gen_prompt", "Gen prompt"),
    ("gen_strict", "Gen strict"),
    ("loc", "Loc"),
)
PNG_METADATA = {"Software": "ODE-edit lifelong_finalw_figures.py"}


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


def final_performance(tables: Path, output: Path) -> Path:
    rows = {row["arm"]: row for row in _rows(tables / "finalw-full10k-arm-summary.csv")}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(METRICS), dtype=float)
    width = 0.19
    for index, arm in enumerate(ARM_ORDER):
        values = [float(rows[arm][metric]) for metric, _ in METRICS]
        ax.bar(x + (index - 1.5) * width, values, width, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
    ax.set_title("Final W₁₀₀₀₀ on All 10,000 Seen Requests")
    ax.set_ylabel("Strict success rate (higher is better)")
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, frameon=False)
    path = output / "finalw-full10k-performance.png"
    _save(fig, path)
    return path


def cumulative_performance(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "cumulative-seen-prefix-arm-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.flat, METRICS, strict=True):
        for arm in ARM_ORDER:
            selected = sorted((row for row in rows if row["arm"] == arm), key=lambda row: int(row["accepted_edit_count"]))
            ax.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                linewidth=1.6,
                markersize=3.5,
                label=ARM_LABEL[arm],
                color=ARM_COLOR[arm],
            )
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Accepted edits in frozen checkpoint")
        ax.set_ylabel("Strict success rate")
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle("Checkpoint Final W on All Seen Requests", y=1.01, fontsize=13)
    path = output / "cumulative-seen-prefix-performance.png"
    _save(fig, path)
    return path


def age_retention(tables: Path, output: Path) -> Path:
    rows = [
        row for row in _rows(tables / "edit-age-strata-summary.csv")
        if int(row["accepted_edit_count"]) == 10_000
    ]
    strata = ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT")
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    width = 0.19
    x = np.arange(len(strata), dtype=float)
    for ax, (metric, label) in zip(axes.flat, METRICS, strict=True):
        for index, arm in enumerate(ARM_ORDER):
            indexed = {row["age_stratum"]: row for row in rows if row["arm"] == arm}
            ax.bar(
                x + (index - 1.5) * width,
                [float(indexed[stratum][metric]) for stratum in strata],
                width,
                label=ARM_LABEL[arm],
                color=ARM_COLOR[arm],
            )
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, ("earliest 20%", "middle 60%", "recent 20%"), rotation=12)
        ax.set_ylabel("Strict success rate")
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle("Final W₁₀₀₀₀ Retention by Edit Age", y=1.01, fontsize=13)
    path = output / "finalw-full10k-edit-age-retention.png"
    _save(fig, path)
    return path


def final_nll(tables: Path, output: Path) -> Path:
    rows = {row["arm"]: row for row in _rows(tables / "finalw-full10k-arm-summary.csv")}
    fields = (
        ("rewrite_target_new_nll_median", "Rewrite new NLL"),
        ("rewrite_target_true_nll_median", "Rewrite true NLL"),
        ("rephrase_target_new_nll_median", "Rephrase new NLL"),
        ("rephrase_target_true_nll_median", "Rephrase true NLL"),
    )
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True)
    for ax, (field, label) in zip(axes.flat, fields, strict=True):
        ax.bar(
            np.arange(len(ARM_ORDER)),
            [float(rows[arm][field]) for arm in ARM_ORDER],
            color=[ARM_COLOR[arm] for arm in ARM_ORDER],
        )
        ax.set_title(label)
        ax.set_ylabel("Length-normalized NLL (lower is better)")
        ax.set_xticks(np.arange(len(ARM_ORDER)), ARM_ORDER)
    fig.suptitle("Final W₁₀₀₀₀ Target NLL", y=1.01, fontsize=13)
    path = output / "finalw-full10k-target-nll.png"
    _save(fig, path)
    return path


def generate(tables: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    _style()
    return [
        final_performance(tables, output),
        cumulative_performance(tables, output),
        age_retention(tables, output),
        final_nll(tables, output),
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = generate(args.tables, args.output)
    for path in paths:
        print(f"{path.name}\t{sha256_file(path)}")


if __name__ == "__main__":
    main()
