"""Deterministic repository-native figures for the ORBODE round-0 review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .round0_analysis_contracts import CELL_BINDINGS, DYNAMIC_ARMS, PRIMARY_ARMS, STAGE_ORDER


PNG_METADATA = {"Software": "ODE-edit deterministic ORBODE round0 analyzer"}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
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


def _title(cell_id: int) -> str:
    model, family, _ = CELL_BINDINGS[cell_id]
    return f"{model} / {family}"


def performance_primary(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "core-performance-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    colors = ("#4477AA", "#EE6677", "#228833")
    for cell_id, ax in zip(CELL_BINDINGS, axes.ravel(), strict=True):
        selected = frame[frame.cell_id == cell_id].set_index("stage").reindex(STAGE_ORDER)
        x = np.arange(len(STAGE_ORDER), dtype=np.float64)
        width = 0.24
        ax.bar(x - width, 100.0 * selected.rewrite_success_rate, width, label="RS", color=colors[0])
        ax.bar(x, 100.0 * selected.rephrase_success_rate, width, label="PS", color=colors[1])
        ns = 100.0 * selected.locality_prediction_preservation_rate
        ax.bar(x + width, ns, width, label="NS", color=colors[2])
        ax.set_xticks(x, STAGE_ORDER, rotation=25, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rate (%)")
        ax.set_title(_title(cell_id), loc="left")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    axes[0, 0].legend(ncol=3, loc="lower right")
    fig.suptitle("Primary CounterFact endpoint rates (PRE_EDIT NS is not applicable)")
    _save(fig, path)


def paired_nll(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "paired-official-deltas.csv")
    metrics = ("rewrite_target_new_nll", "rephrase_target_new_nll")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for cell_id, ax in zip(CELL_BINDINGS, axes.ravel(), strict=True):
        selected = frame[(frame.cell_id == cell_id) & frame.metric.isin(metrics)]
        x = np.arange(len(DYNAMIC_ARMS), dtype=np.float64)
        width = 0.36
        for offset, metric, label, color in (
            (-width / 2, metrics[0], "Rewrite target-new", "#4477AA"),
            (width / 2, metrics[1], "Rephrase target-new", "#EE6677"),
        ):
            values = selected[selected.metric == metric].set_index("arm").reindex(DYNAMIC_ARMS).delta_mean
            ax.bar(x + offset, values, width, label=label, color=color)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xticks(x, DYNAMIC_ARMS)
        ax.set_ylabel("Mean paired NLL delta (ours - O)")
        ax.set_title(_title(cell_id), loc="left")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Request/prompt-matched target-new NLL deltas vs Official O")
    _save(fig, path)


def hit_action_trajectories(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "mechanism-step-summary.csv")
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), constrained_layout=True)
    for row_index, cell_id in enumerate(CELL_BINDINGS):
        selected = frame[frame.cell_id == cell_id].sort_values("visit_ordinal")
        left, right = axes[row_index]
        for arm in DYNAMIC_ARMS:
            arm_rows = selected[selected.arm == arm]
            left.plot(arm_rows.visit_ordinal + 1, arm_rows.residual_norm_mean_after, marker="o", markersize=2, linewidth=0.8, label=arm)
            right.plot(arm_rows.visit_ordinal + 1, arm_rows.semantic_request_strict_count, marker="o", markersize=2, linewidth=0.8, label=arm)
        left.set_ylabel(f"{_title(cell_id)}\nResidual mean after")
        right.set_ylabel("Strict requests / 100")
        left.grid(True, alpha=0.2, linewidth=0.4)
        right.grid(True, alpha=0.2, linewidth=0.4)
        if row_index == 0:
            left.set_title("Action-to-go residual trajectory")
            right.set_title("Strict-predicate trajectory (global hit requires 100/100)")
        if row_index == 3:
            left.set_xlabel("Layer visit ordinal (1..20)")
            right.set_xlabel("Layer visit ordinal (1..20)")
    axes[0, 0].legend(ncol=2, fontsize=7)
    _save(fig, path)


def layer_share(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "layer-update-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))
    for cell_id, ax in zip(CELL_BINDINGS, axes.ravel(), strict=True):
        selected = frame[frame.cell_id == cell_id]
        x = np.arange(len(PRIMARY_ARMS), dtype=np.float64)
        bottom = np.zeros(len(PRIMARY_ARMS), dtype=np.float64)
        for color, layer in zip(colors, (4, 5, 6, 7, 8), strict=True):
            values = selected[selected.layer == layer].set_index("arm").reindex(PRIMARY_ARMS).terminal_net_energy_share.to_numpy(dtype=np.float64)
            ax.bar(x, 100.0 * values, bottom=100.0 * bottom, label=f"L{layer}", color=color)
            bottom += values
        ax.set_xticks(x, PRIMARY_ARMS)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Terminal net squared-Frobenius share (%)")
        ax.set_title(_title(cell_id), loc="left")
    axes[0, 0].legend(ncol=5, fontsize=7)
    fig.suptitle("Per-layer terminal update-energy share")
    _save(fig, path)


def compute_tradeoff(tables: Path, path: Path) -> None:
    frame = pd.read_csv(tables / "compute-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    for cell_id, ax in zip(CELL_BINDINGS, axes.ravel(), strict=True):
        selected = frame[frame.cell_id == cell_id].set_index("arm").reindex(PRIMARY_ARMS)
        x = np.arange(len(PRIMARY_ARMS), dtype=np.float64)
        ax.bar(x - 0.18, selected.wall_ratio_vs_official, 0.36, label="Wall ratio vs O", color="#AA3377")
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, selected.model_forward_invocation_count, 0.36, label="Forward count", color="#33BBEE", alpha=0.75)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set_xticks(x, PRIMARY_ARMS)
        ax.set_ylabel("Wall-time ratio")
        ax2.set_ylabel("Recorded model forwards")
        ax.set_title(_title(cell_id), loc="left")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Endpoint wall time and recorded forward-compute tradeoff")
    _save(fig, path)


def generate(tables: Path, output: Path) -> list[Path]:
    _configure()
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "primary-rs-ps-ns.png",
        output / "paired-target-new-nll-deltas.png",
        output / "hit-action-to-go-trajectories.png",
        output / "layer-update-energy-share.png",
        output / "compute-tradeoff.png",
    ]
    performance_primary(tables, paths[0])
    paired_nll(tables, paths[1])
    hit_action_trajectories(tables, paths[2])
    layer_share(tables, paths[3])
    compute_tradeoff(tables, paths[4])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = generate(args.tables, args.output)
    print(f"ORBODE_ROUND0_PLOT_PASS files={len(paths)}")


if __name__ == "__main__":
    main()
