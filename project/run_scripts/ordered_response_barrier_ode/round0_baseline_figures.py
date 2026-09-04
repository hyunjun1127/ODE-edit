"""Deterministic baseline-inclusive figures for the ORBODE round-0 report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PNG_METADATA = {"Software": "ODE-edit deterministic ORBODE baseline report"}
MODEL_ORDER = ("llama3-8b-inst", "qwen2.5-7b-inst")


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "axes.grid": False,
            "path.simplify": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        format="png",
        dpi=120,
        bbox_inches=None,
        metadata=PNG_METADATA,
        pil_kwargs={"compress_level": 9, "optimize": False},
    )
    plt.close(fig)


def baseline_rates(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "baseline-inclusive-performance.csv")
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 8.0), constrained_layout=True)
    colors = ("#4477AA", "#EE6677", "#CCBB44", "#228833")
    for model, ax in zip(MODEL_ORDER, axes, strict=True):
        selected = frame[frame.model == model].reset_index(drop=True)
        x = np.arange(len(selected), dtype=np.float64)
        width = 0.19
        values = (
            ("rewrite_success_rate", "RS", colors[0]),
            ("rephrase_success_rate", "PS", colors[1]),
            ("strict_rephrase_success_rate", "strict PS", colors[2]),
            ("pp_prompt_rate", "PP-prompt", colors[3]),
        )
        for index, (field, label, color) in enumerate(values):
            offset = (index - 1.5) * width
            ax.bar(x + offset, 100.0 * selected[field], width, label=label, color=color)
        ax.set_xticks(x, selected.method_label, rotation=28, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rate (%)")
        ax.set_title(model, loc="left")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    axes[0].legend(ncol=4, fontsize=7, loc="lower right")
    fig.suptitle("Round-0 W0 and explicit Official baselines (canonical endpoint NS unavailable)")
    _save(fig, path)


def preedit_ns(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "preedit-canonical-ns-summary.csv")
    frame = frame.set_index("model").reindex(MODEL_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    x = np.arange(len(frame), dtype=np.float64)
    values = 100.0 * frame.ns_rate.to_numpy(dtype=np.float64)
    bars = ax.bar(x, values, width=0.55, color=("#4477AA", "#EE6677"))
    ax.set_xticks(x, frame.model)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Canonical NS (%)")
    ax.set_title("PRE_EDIT/W0 CounterFact locality\nstrict target-true NLL < target-new NLL; 1,000 prompt pairs/model")
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    for bar, row in zip(bars, frame.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{int(row.ns_numerator)}/{int(row.ns_denominator)}",
            ha="center",
            va="bottom",
        )
    _save(fig, path)


def generate(tables: Path, output: Path) -> list[Path]:
    _configure()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "baseline-rs-ps-pp.png", output / "preedit-canonical-ns.png"]
    baseline_rates(tables, paths[0])
    preedit_ns(tables, paths[1])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = generate(args.tables, args.output)
    print(f"ORBODE_BASELINE_PLOT_PASS files={len(paths)}")


if __name__ == "__main__":
    main()
