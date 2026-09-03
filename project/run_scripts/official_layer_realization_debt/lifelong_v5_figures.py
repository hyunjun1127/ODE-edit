"""Deterministic code-only figures for the lifelong v5 cumulative report."""

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
CATEGORY_ORDER = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
)
CATEGORY_LABEL = {
    "rewrite_target_new": "Rewrite / target-new",
    "rewrite_target_true": "Rewrite / target-true",
    "rephrase_target_new": "Rephrase / target-new",
    "rephrase_target_true": "Rephrase / target-true",
    "locality_target_true": "Locality / target-true",
}
PERFORMANCE_METRICS = (
    ("eff", "Eff"),
    ("gen_prompt", "Gen prompt"),
    ("gen_strict", "Gen strict"),
    ("loc", "Loc"),
)
AGE_STYLE = {
    "EARLY_FIRST_20PCT": ("earliest 20%", "--"),
    "MIDDLE_60PCT": ("middle 60%", "-"),
    "RECENT_LAST_20PCT": ("recent 20%", ":"),
}
PNG_METADATA = {"Software": "ODE-edit lifelong_v5_figures.py"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
        }
    )


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata=PNG_METADATA, bbox_inches="tight")
    plt.close(fig)
    return path


def final_performance(tables: Path, output: Path) -> Path:
    rows = {row["arm"]: row for row in _rows(tables / "finalw-full10k-arm-summary.csv")}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(PERFORMANCE_METRICS), dtype=float)
    width = 0.19
    for index, arm in enumerate(ARM_ORDER):
        ax.bar(
            x + (index - 1.5) * width,
            [float(rows[arm][metric]) for metric, _ in PERFORMANCE_METRICS],
            width,
            label=ARM_LABEL[arm],
            color=ARM_COLOR[arm],
        )
    ax.set_title("Final W₁₀₀₀₀ on All 10,000 Seen Requests")
    ax.set_ylabel("Success rate (higher is better)")
    ax.set_xticks(x, [label for _, label in PERFORMANCE_METRICS])
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, frameon=False)
    return _save(fig, output / "finalw-full10k-performance.png")


def cumulative_performance(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "cumulative-core-rates.csv")
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.flat, PERFORMANCE_METRICS, strict=True):
        for arm in ARM_ORDER:
            selected = sorted(
                (row for row in rows if row["arm"] == arm),
                key=lambda row: int(row["accepted_edit_count"]),
            )
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
        ax.set_ylabel("Success rate")
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle("Checkpoint Final W on All Seen Requests", y=1.01, fontsize=13)
    return _save(fig, output / "cumulative-performance-trajectories.png")


def _category_trajectory(
    tables: Path, output: Path, *, quantity: str, ylabel: str, filename: str
) -> Path:
    rows = _rows(tables / "cumulative-category-distributions.csv")
    field_prefix = (
        "request_cluster_nll" if quantity == "nll" else "request_cluster_min_margin"
    )
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=True)
    for ax, category in zip(axes.flat[:5], CATEGORY_ORDER, strict=True):
        for arm in ARM_ORDER:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["arm"] == arm and row["category"] == category
                ),
                key=lambda row: int(row["accepted_edit_count"]),
            )
            x = [int(row["accepted_edit_count"]) for row in selected]
            ax.plot(
                x,
                [float(row[f"{field_prefix}_median"]) for row in selected],
                color=ARM_COLOR[arm],
                linewidth=1.6,
                marker="o",
                markersize=2.8,
                label=f"{arm} median",
            )
            ax.plot(
                x,
                [float(row[f"{field_prefix}_p90"]) for row in selected],
                color=ARM_COLOR[arm],
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                label=f"{arm} p90",
            )
        ax.set_title(CATEGORY_LABEL[category])
        ax.set_xlabel("Accepted edits")
        ax.set_ylabel(ylabel)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", ncol=2, frameon=False)
    fig.suptitle(
        "Cumulative Target NLL: Median and p90 Tail"
        if quantity == "nll"
        else "Cumulative Target Margin: Median and p90",
        y=1.005,
        fontsize=13,
    )
    return _save(fig, output / filename)


def preference_trajectories(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "cumulative-core-rates.csv")
    fields = (
        ("rewrite_new_preferred_rate", "Rewrite new preference"),
        ("rephrase_new_preferred_rate", "Rephrase new preference"),
    )
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (field, label) in zip(axes, fields, strict=True):
        for arm in ARM_ORDER:
            selected = sorted(
                (row for row in rows if row["arm"] == arm),
                key=lambda row: int(row["accepted_edit_count"]),
            )
            ax.plot(
                [int(row["accepted_edit_count"]) for row in selected],
                [float(row[field]) for row in selected],
                marker="o",
                linewidth=1.6,
                markersize=3.5,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm],
            )
        ax.set_title(label)
        ax.set_xlabel("Accepted edits")
        ax.set_ylabel("Request preference rate")
        ax.set_ylim(0.0, 1.0)
    axes[0].legend(ncol=2, frameon=False)
    return _save(fig, output / "cumulative-preference-trajectories.png")


def age_performance(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "cumulative-age-strata-full.csv")
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.flat, PERFORMANCE_METRICS, strict=True):
        for arm in ARM_ORDER:
            for stratum, (age_label, linestyle) in AGE_STYLE.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["arm"] == arm and row["age_stratum"] == stratum
                    ),
                    key=lambda row: int(row["accepted_edit_count"]),
                )
                ax.plot(
                    [int(row["accepted_edit_count"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    color=ARM_COLOR[arm],
                    linestyle=linestyle,
                    linewidth=1.25,
                    label=f"{arm} {age_label}",
                )
        ax.set_title(label)
        ax.set_xlabel("Accepted edits")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0.0, 1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Checkpoint Performance by Relative Edit Age", y=1.01, fontsize=13)
    fig.subplots_adjust(bottom=0.18)
    return _save(fig, output / "checkpoint-age-strata-performance.png")


def mechanism_associations(tables: Path, output: Path) -> Path:
    rows = _rows(tables / "mechanism-cumulative-associations.csv")
    mechanism_order = (
        "q_pre_L8_median",
        "q_post_L8_median",
        "mean_rho_median",
        "mean_tau_median",
        "d_parallel_median",
        "d_perp_median",
        "D_TV_median",
        "update_magnitude_total",
        "layer8_update_share",
    )
    outcome_order = ("eff", "gen_prompt", "gen_strict", "loc", "rewrite_new_nll_median")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), sharex=True, sharey=True)
    image = None
    for ax, arm in zip(axes.flat, ARM_ORDER, strict=True):
        indexed = {
            (row["mechanism_metric"], row["outcome_metric"]): float(row["spearman"])
            for row in rows
            if row["arm"] == arm
        }
        matrix = np.array(
            [[indexed[(mechanism, outcome)] for outcome in outcome_order] for mechanism in mechanism_order]
        )
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xticks(np.arange(len(outcome_order)), outcome_order, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(mechanism_order)), mechanism_order)
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Spearman coefficient (n=7 checkpoints)")
    fig.suptitle("Mechanism–Cumulative Performance Descriptive Association", y=1.0, fontsize=13)
    return _save(fig, output / "mechanism-cumulative-associations.png")


def generate(tables: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    _style()
    return [
        final_performance(tables, output),
        cumulative_performance(tables, output),
        _category_trajectory(
            tables,
            output,
            quantity="nll",
            ylabel="Length-normalized NLL (lower is better)",
            filename="cumulative-target-nll-trajectories.png",
        ),
        _category_trajectory(
            tables,
            output,
            quantity="margin",
            ylabel="Minimum target logit margin (higher is better)",
            filename="cumulative-target-margin-trajectories.png",
        ),
        preference_trajectories(tables, output),
        age_performance(tables, output),
        mechanism_associations(tables, output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.tables, args.output):
        print(f"{path.name}\t{sha256_file(path)}")


if __name__ == "__main__":
    main()
