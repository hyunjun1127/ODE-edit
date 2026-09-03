"""Deterministic repository-native plots for Realization Debt Phase A."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = (
    ("llama3-8b-inst", "memit"),
    ("llama3-8b-inst", "alphaedit"),
    ("qwen2.5-7b-inst", "memit"),
    ("qwen2.5-7b-inst", "alphaedit"),
)
ARM_LABEL = {
    ("llama3-8b-inst", "memit"): "Llama / MEMIT",
    ("llama3-8b-inst", "alphaedit"): "Llama / AlphaEdit",
    ("qwen2.5-7b-inst", "memit"): "Qwen / MEMIT",
    ("qwen2.5-7b-inst", "alphaedit"): "Qwen / AlphaEdit",
}
LAYERS = (4, 5, 6, 7, 8)
PNG_METADATA = {"Software": "ODE-edit deterministic Phase-A plotter"}


def configure() -> None:
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


def heatmap(summary: pd.DataFrame, column: str, title: str, path: Path) -> None:
    matrices: list[np.ndarray] = []
    for model, method in ARM_ORDER:
        selected = summary[(summary.model == model) & (summary.method == method)]
        matrix = (
            selected.pivot(index="layer", columns="batch_index", values=column)
            .reindex(index=LAYERS, columns=range(1, 101))
            .to_numpy(dtype=np.float64)
        )
        if matrix.shape != (5, 100) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid heatmap matrix for {(model, method)} {column}")
        matrices.append(matrix)
    all_values = np.concatenate([value.ravel() for value in matrices])
    vmin, vmax = float(all_values.min()), float(all_values.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    fig, axes = plt.subplots(4, 1, figsize=(13.0, 8.4), sharex=True, constrained_layout=True)
    image = None
    for ax, arm, matrix in zip(axes, ARM_ORDER, matrices, strict=True):
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(ARM_LABEL[arm], loc="left")
        ax.set_yticks(range(5), [str(value) for value in LAYERS])
        ax.set_ylabel("Layer")
    axes[-1].set_xticks([0, 19, 39, 59, 79, 99], ["1", "20", "40", "60", "80", "100"])
    axes[-1].set_xlabel("Sequential B100 batch")
    fig.suptitle(title)
    assert image is not None
    fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    _save(fig, path)


def joint_trajectory(joined: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for ax, arm in zip(axes.ravel(), ARM_ORDER, strict=True):
        model, method = arm
        selected = joined[(joined.model == model) & (joined.method == method)].sort_values(
            ["layer", "batch_index"]
        )
        for layer in LAYERS:
            rows = selected[selected.layer == layer]
            ax.plot(
                rows["frobenius_action_sq"],
                rows["debt_native_mean"],
                linewidth=0.55,
                alpha=0.45,
                label=f"L{layer}",
            )
            ax.scatter(
                rows["frobenius_action_sq"],
                rows["debt_native_mean"],
                c=rows["batch_index"],
                cmap="viridis",
                vmin=1,
                vmax=100,
                s=8,
                alpha=0.8,
                linewidths=0,
            )
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel(r"Batch-layer $||\Delta W||_F^2$")
        ax.set_ylabel("Mean row-wise debt_native")
        ax.legend(ncol=5, fontsize=6, loc="best")
        ax.grid(True, alpha=0.2, linewidth=0.4)
    fig.suptitle("Batch × layer realization debt and squared-Frobenius action (color=batch)")
    _save(fig, path)


def endpoint_relationship(relationship: pd.DataFrame, path: Path) -> None:
    outcomes = (
        ("rs_current_rate", "Immediate RS current rate"),
        ("nll_advantage_mean", "Mean NLL advantage"),
        ("target_new_strict_rate", "Target-new strict rate"),
    )
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 13.5), constrained_layout=True)
    for row_index, arm in enumerate(ARM_ORDER):
        model, method = arm
        rows = relationship[(relationship.model == model) & (relationship.method == method)].sort_values(
            "batch_index"
        )
        x = rows["debt_native_mean"].to_numpy(dtype=np.float64)
        for col_index, (field, label) in enumerate(outcomes):
            ax = axes[row_index, col_index]
            y = rows[field].to_numpy(dtype=np.float64)
            ax.scatter(x, y, c=rows["batch_index"], cmap="viridis", vmin=1, vmax=100, s=16, linewidths=0)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 100)
                ax.plot(grid, slope * grid + intercept, color="black", linewidth=0.7)
            if row_index == 0:
                ax.set_title(label)
            if col_index == 0:
                ax.set_ylabel(f"{ARM_LABEL[arm]}\n{label}")
            else:
                ax.set_ylabel(label)
            if row_index == 3:
                ax.set_xlabel("Mean row-wise debt_native in current B100")
            ax.grid(True, alpha=0.2, linewidth=0.4)
    fig.suptitle("Contemporaneous batch debt vs immediate current-B100 rewrite observations")
    _save(fig, path)


def generate(tables: Path, output: Path) -> list[Path]:
    configure()
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(tables / "batch-layer-debt-summary.csv")
    joined = pd.read_csv(tables / "batch-layer-action-proxy.csv")
    relationship = pd.read_csv(tables / "batch-endpoint-relationship.csv")
    paths = [
        output / "median-rho-batch-layer-heatmap.png",
        output / "median-tau-batch-layer-heatmap.png",
        output / "median-debt-native-batch-layer-heatmap.png",
        output / "debt-frobenius-action-joint-trajectory.png",
        output / "debt-immediate-rewrite-relationship.png",
    ]
    heatmap(summary, "rho_median", "Median rho by batch and layer", paths[0])
    heatmap(summary, "tau_median", "Median tau by batch and layer", paths[1])
    heatmap(summary, "debt_native_median", "Median row-wise debt_native by batch and layer", paths[2])
    joint_trajectory(joined, paths[3])
    endpoint_relationship(relationship, paths[4])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = generate(args.tables, args.output)
    print("PLOT_GENERATION_PASS files=" + str(len(paths)))


if __name__ == "__main__":
    main()
