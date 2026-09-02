"""Deterministic table-only plots for the four-arm lifelong analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import platform

from .lifelong_analysis_io import CELL_ORDER, IDEAL_Q, LAYERS, Q_LABELS, canonical_hash


PLOT_DPI = 180
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_SEED = 660_604
PLOT_INPUTS = (
    "checkpoint-request-complete.csv.gz",
    "checkpoint-layer-complete.csv.gz",
    "checkpoint-weight-complete.csv.gz",
    "production-weight-batch-unit-complete.csv.gz",
    "functional-endpoint-complete.csv.gz",
    "compute-accounting.csv",
)
MODEL_LABEL = {"llama3-8b-inst": "Llama-3-8B-Instruct", "qwen2.5-7b-inst": "Qwen2.5-7B-Instruct"}
METHOD_LABEL = {"memit": "Official MEMIT", "alphaedit": "Official AlphaEdit"}
COLORS = {"memit": "#7c3aed", "alphaedit": "#0f766e"}
ENDPOINT_COLORS = {
    "rewrite_target_new": "#2563eb",
    "rephrase_target_new": "#f97316",
    "locality_target_true": "#16a34a",
}
# Publication axis policy.  The complete, unclipped extrema remain in the
# sealed CSV tables; these shared display ranges keep the four panels directly
# comparable and prevent a few recorded tails from flattening every median.
AXIS_RANGES = {
    "primary_q": [0.0, 4.5],
    "checkpoint_q_color": [0.0, 1.5],
    "terminal_rho": [-0.5, 9.0],
    "terminal_tau": [0.0, 22.0],
    "inherited_debt": [-1.0, 2.0],
    "checkpoint_rho_color": [0.0, 6.0],
    "checkpoint_tau_color": [0.0, 8.0],
    "weight_magnitude_log": [1.0, 1000.0],
    "update_drift_log1p_color": [0.0, 6.5],
    "endpoint_strict_rate": [-0.03, 1.03],
    "compute_hours": [0.0, 14.0],
}


@dataclass(slots=True)
class FigureTables:
    checkpoint_request: list[dict[str, Any]]
    checkpoint_layer: list[dict[str, Any]]
    checkpoint_weight: list[dict[str, Any]]
    production_weight: list[dict[str, Any]]
    functional: list[dict[str, Any]]
    compute: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_csv(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"plot input table is empty: {path}")
    numeric = {
        "batch_index", "accepted_edit_count", "layer", "q_pre_L4", "q_pre_L5",
        "q_pre_L6", "q_pre_L7", "q_pre_L8", "q_post_L8", "q_pre_L8_gt_0_2",
        "d_parallel", "d_perp", "rho", "tau", "frobenius_magnitude", "strict_rate",
        "production_edit_core_wall_seconds_sum", "production_observer_wall_seconds_sum", "cell_wall_seconds",
    }
    integral = {"batch_index", "accepted_edit_count", "layer", "q_pre_L8_gt_0_2"}
    for row in rows:
        for key in numeric & row.keys():
            raw = row[key]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                # Complete tables intentionally carry typed missing markers.
                # Leave those strings intact; plotting code selects only rows
                # whose availability is RECORDED.
                continue
            if not np.isfinite(value):
                raise ValueError(f"nonfinite numeric plot input: {path}:{key}")
            row[key] = int(value) if key in integral else value
    return rows


def load_figure_tables(root: Path) -> FigureTables:
    seal_path = root / "derived-table-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_identity = seal.get("identity_sha256")
    unsigned = dict(seal)
    unsigned.pop("identity_sha256", None)
    if expected_identity != canonical_hash(unsigned):
        raise ValueError("derived table seal canonical identity differs")
    locked = {row["path"]: row for row in seal["tables"]}
    for name in PLOT_INPUTS:
        path = root / name
        row = locked.get(name)
        if row is None or path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise ValueError(f"plot input differs from derived table seal: {name}")
    tables = FigureTables(
        checkpoint_request=_load_csv(root / PLOT_INPUTS[0]),
        checkpoint_layer=_load_csv(root / PLOT_INPUTS[1]),
        checkpoint_weight=_load_csv(root / PLOT_INPUTS[2]),
        production_weight=_load_csv(root / PLOT_INPUTS[3]),
        functional=_load_csv(root / PLOT_INPUTS[4]),
        compute=_load_csv(root / PLOT_INPUTS[5]),
    )
    observed = (
        len(tables.checkpoint_request),
        len(tables.checkpoint_layer),
        len(tables.checkpoint_weight),
        len(tables.production_weight),
        len(tables.functional),
        len(tables.compute),
    )
    for name, count in zip(PLOT_INPUTS, observed, strict=True):
        if count != int(locked[name]["rows"]):
            raise ValueError(f"plot input row count differs from derived table seal: {name}")
    return tables


def _cell_axes(title: str, *, sharex: bool = True, sharey: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=sharex, sharey=sharey)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        axis.set_title(f"{MODEL_LABEL[model]} / {METHOD_LABEL[method]}", fontsize=10)
        axis.grid(alpha=0.16)
    fig.suptitle(title)
    return fig, axes


def primary_residual_figure(tables: FigureTables):
    fig, axes = _cell_axes("Layer-wise Progress toward Target Activation")
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        rows = [r for r in tables.checkpoint_request if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == 10_000 and r["fork"] == "ACCUMULATED_OR_STATIC"]
        q = np.asarray([[r[f"q_{label.replace('-', '_')}"] for label in Q_LABELS] for r in rows])
        for values in q:
            axis.plot(range(6), values, color="0.72", linewidth=0.45, alpha=0.24)
        axis.fill_between(range(6), np.percentile(q, 25, axis=0), np.percentile(q, 75, axis=0), color=COLORS[method], alpha=0.20, label="IQR")
        median = np.median(q, axis=0)
        axis.plot(range(6), median, color=COLORS[method], linewidth=2.4, label="median")
        axis.plot(range(6), IDEAL_Q, color="#b91c1c", linewidth=1.4, linestyle="--", label="ideal")
        count = sum(r["q_pre_L8_gt_0_2"] for r in rows)
        axis.text(0.02, 0.98, f"10k sentinel n={len(rows)}\npre-L8 med={median[4]:.3f}\nq>0.2={count}/{len(rows)}\ndebt med=({np.median([r['d_parallel'] for r in rows]):.3f}, {np.median([r['d_perp'] for r in rows]):.3f})", transform=axis.transAxes, va="top", fontsize=8)
        axis.set_xticks(range(6), Q_LABELS, rotation=20)
        axis.set_ylabel("Normalized Target Progress (remaining residual q)")
        axis.set_ylim(*AXIS_RANGES["primary_q"])
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def checkpoint_q_heatmap_figure(tables: FigureTables):
    fig, axes = _cell_axes("Checkpoint × Layer Median Remaining Residual", sharex=False)
    checkpoints = (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        matrix = []
        for accepted in checkpoints:
            rows = [r for r in tables.checkpoint_request if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == accepted and r["fork"] == "ACCUMULATED_OR_STATIC"]
            matrix.append([np.median([r[f"q_{label.replace('-', '_')}"] for r in rows]) for label in Q_LABELS])
        image = axis.imshow(
            np.asarray(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=AXIS_RANGES["checkpoint_q_color"][0],
            vmax=AXIS_RANGES["checkpoint_q_color"][1],
        )
        axis.set_xticks(range(6), Q_LABELS, rotation=35, ha="right", fontsize=7)
        axis.set_yticks(range(8), [f"{v / 1000:g}k" for v in checkpoints])
        axis.set_ylabel("accepted edits")
        fig.colorbar(image, ax=axis, shrink=0.72, label="median q (sentinel n=100)")
    fig.tight_layout()
    return fig


def _terminal_layer_box(tables: FigureTables, metric: str, title: str):
    fig, axes = _cell_axes(title)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        values = [[r[metric] for r in tables.checkpoint_layer if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == 10_000 and r["fork"] == "ACCUMULATED_OR_STATIC" and r["layer"] == layer] for layer in LAYERS]
        axis.boxplot(values, tick_labels=[str(layer) for layer in LAYERS], showfliers=False)
        if metric == "rho":
            axis.axhline(1.0, color="#b91c1c", linewidth=1, linestyle="--")
            axis.set_ylim(*AXIS_RANGES["terminal_rho"])
        else:
            axis.set_ylim(*AXIS_RANGES["terminal_tau"])
        axis.set_xlabel("editable layer")
        axis.set_ylabel(f"{metric} (sentinel n=100/cell)")
    fig.tight_layout()
    return fig


def rho_figure(tables: FigureTables):
    return _terminal_layer_box(tables, "rho", "Target-aligned Realization Ratio at 10k")


def tau_figure(tables: FigureTables):
    return _terminal_layer_box(tables, "tau", "Orthogonal Activation Distortion at 10k")


def inherited_debt_figure(tables: FigureTables):
    fig, axes = _cell_axes("Inherited Debt Distributions", sharex=False)
    checkpoints = (0, 5000, 10000)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        dpar = [[r["d_parallel"] for r in tables.checkpoint_request if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == accepted and r["fork"] == "ACCUMULATED_OR_STATIC"] for accepted in checkpoints]
        dperp = [[r["d_perp"] for r in tables.checkpoint_request if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == accepted and r["fork"] == "ACCUMULATED_OR_STATIC"] for accepted in checkpoints]
        positions = np.arange(3, dtype=float)
        left = axis.boxplot(dpar, positions=positions - .16, widths=.28, patch_artist=True, showfliers=False)
        right = axis.boxplot(dperp, positions=positions + .16, widths=.28, patch_artist=True, showfliers=False)
        for box in left["boxes"]:
            box.set_facecolor("#2563eb"); box.set_alpha(.55)
        for box in right["boxes"]:
            box.set_facecolor("#f97316"); box.set_alpha(.55)
        axis.set_xticks(positions, ["t0", "5k", "10k"])
        axis.set_xlabel("accepted edits (sentinel n=100/checkpoint)")
        axis.set_ylabel("normalized inherited debt")
        axis.set_ylim(*AXIS_RANGES["inherited_debt"])
        axis.legend([left["boxes"][0], right["boxes"][0]], ["d_parallel", "d_perp"], fontsize=8)
    fig.tight_layout()
    return fig


def checkpoint_layer_heatmap_figure(tables: FigureTables, metric: str):
    fig, axes = _cell_axes(f"Checkpoint × Layer {metric}", sharex=False)
    checkpoints = (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        matrix = []
        for accepted in checkpoints:
            matrix.append([np.median([r[metric] for r in tables.checkpoint_layer if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == accepted and r["fork"] == "ACCUMULATED_OR_STATIC" and r["layer"] == layer]) for layer in LAYERS])
        color_range = AXIS_RANGES[f"checkpoint_{metric}_color"]
        image = axis.imshow(
            np.asarray(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=color_range[0],
            vmax=color_range[1],
        )
        axis.set_xticks(range(5), [str(layer) for layer in LAYERS])
        axis.set_yticks(range(8), [f"{v / 1000:g}k" for v in checkpoints])
        axis.set_xlabel("editable layer"); axis.set_ylabel("accepted edits")
        fig.colorbar(image, ax=axis, shrink=.72, label=f"median {metric} (n=100)")
    fig.tight_layout()
    return fig


def weight_magnitude_figure(tables: FigureTables):
    """Batch-level magnitudes. Deliberately contains no ideal/equal line."""
    fig, axes = _cell_axes("Layer-wise Update Magnitude")
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        medians, lower, upper = [], [], []
        for layer in LAYERS:
            values = np.asarray([r["frobenius_magnitude"] for r in tables.production_weight if r["model"] == model and r["method"] == method and r["layer"] == layer])
            p25, median, p75 = np.percentile(values, (25, 50, 75))
            medians.append(median); lower.append(median - p25); upper.append(p75 - median)
        # Keep the IQR whiskers part of the bar artist.  In particular, do not
        # add a Line2D reference artist: the publication contract forbids an
        # equal-allocation/ideal reference line on this weight-only panel.
        axis.bar(
            range(5),
            medians,
            yerr=[lower, upper],
            color=COLORS[method],
            alpha=.84,
            ecolor="#111827",
            capsize=3,
        )
        axis.set_xticks(range(5), [str(layer) for layer in LAYERS])
        axis.set_xlabel("editable layer")
        axis.set_ylabel("median ||Delta W_l||_F (100 B100 batches)")
        axis.set_yscale("log")
        axis.set_ylim(*AXIS_RANGES["weight_magnitude_log"])
    fig.tight_layout()
    return fig


def update_drift_heatmap_figure(tables: FigureTables):
    fig, axes = _cell_axes("Layer-wise Update Magnitude Drift", sharex=False)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        matrix = np.full((100, 5), np.nan)
        for batch in range(1, 101):
            for column, layer in enumerate(LAYERS):
                values = [r["frobenius_magnitude"] for r in tables.production_weight if r["model"] == model and r["method"] == method and r["batch_index"] == batch and r["layer"] == layer]
                if len(values) != 1:
                    raise ValueError(f"weight plot row count differs: {model}/{method}/B{batch}/L{layer}")
                matrix[batch - 1, column] = values[0]
        image = axis.imshow(
            np.log1p(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=AXIS_RANGES["update_drift_log1p_color"][0],
            vmax=AXIS_RANGES["update_drift_log1p_color"][1],
        )
        axis.set_xticks(range(5), [str(layer) for layer in LAYERS])
        axis.set_yticks([0, 9, 19, 49, 74, 99], ["B1", "B10", "B20", "B50", "B75", "B100"])
        axis.set_xlabel("editable layer"); axis.set_ylabel("sequential B100 batch")
        fig.colorbar(image, ax=axis, shrink=.72, label="log1p(||Delta W_l||_F)")
    fig.tight_layout()
    return fig


def endpoint_checkpoint_figure(tables: FigureTables):
    fig, axes = _cell_axes("Checkpoint Endpoint Strict Rates", sharex=True, sharey=True)
    metrics = ("rewrite_target_new", "rephrase_target_new", "locality_target_true")
    labels = ("rewrite", "rephrase", "locality")
    checkpoints = (1000, 1500, 2000, 3000, 5000, 7500, 10000)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        for metric, label in zip(metrics, labels, strict=True):
            values = []
            for accepted in checkpoints:
                selected = [r["strict_rate"] for r in tables.functional if r["model"] == model and r["method"] == method and r["accepted_edit_count"] == accepted and r["cohort"] == "current" and r["metric"] == metric and r["availability"] == "RECORDED"]
                values.append(float(np.mean(selected)))
            axis.plot(
                checkpoints,
                values,
                marker="o",
                linewidth=1.6,
                color=ENDPOINT_COLORS[metric],
                label=label,
            )
        axis.set_xlabel("accepted edits")
        axis.set_ylabel("request-mean strict rate")
        axis.set_ylim(*AXIS_RANGES["endpoint_strict_rate"])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def compute_comparison_figure(tables: FigureTables):
    fig, axis = plt.subplots(figsize=(11, 6))
    labels = [f"{MODEL_LABEL[r['model']]}\n{METHOD_LABEL[r['method']]}" for r in tables.compute]
    edit = np.asarray([r["production_edit_core_wall_seconds_sum"] for r in tables.compute]) / 3600
    observer = np.asarray([r["production_observer_wall_seconds_sum"] for r in tables.compute]) / 3600
    total = np.asarray([r["cell_wall_seconds"] for r in tables.compute]) / 3600
    x = np.arange(len(labels))
    axis.bar(x - .22, edit, width=.22, label="production edit-core")
    axis.bar(x, observer, width=.22, label="observer recorded")
    axis.bar(x + .22, total, width=.22, label="cell wall")
    axis.set_xticks(x, labels)
    axis.set_ylabel("hours")
    axis.set_title("Recorded Compute Time (GPU peak memory NOT_RECORDED_SCHEMA)")
    axis.set_ylim(*AXIS_RANGES["compute_hours"])
    axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_figures(output: Path, tables: FigureTables) -> list[str]:
    figures = {
        "primary-residual-trajectory-2x2.png": primary_residual_figure(tables),
        "checkpoint-layer-q-heatmap.png": checkpoint_q_heatmap_figure(tables),
        "rho-by-layer-terminal.png": rho_figure(tables),
        "tau-by-layer-terminal.png": tau_figure(tables),
        "inherited-debt-distributions.png": inherited_debt_figure(tables),
        "checkpoint-layer-rho-heatmap.png": checkpoint_layer_heatmap_figure(tables, "rho"),
        "checkpoint-layer-tau-heatmap.png": checkpoint_layer_heatmap_figure(tables, "tau"),
        "layer-wise-update-magnitude.png": weight_magnitude_figure(tables),
        "layer-wise-update-magnitude-drift.png": update_drift_heatmap_figure(tables),
        "endpoint-checkpoint-strict-rates.png": endpoint_checkpoint_figure(tables),
        "compute-time-comparison.png": compute_comparison_figure(tables),
    }
    names = []
    for name, figure in figures.items():
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite plot: {path}")
        figure.savefig(
            path,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            metadata={"Software": "ODE-edit deterministic lifelong_analysis_figures"},
        )
        plt.close(figure)
        names.append(name)
    return names


def _python_executable() -> str:
    """Return the invoked interpreter path without resolving a venv symlink."""

    return str(Path(sys.executable).absolute())


def _reproduction_command(tables_root: Path, output: Path) -> str:
    command_argv = [
        "env",
        "MPLCONFIGDIR=/tmp/odeedit-lifelong-analysis-mpl",
        _python_executable(),
        "-m",
        "project.run_scripts.official_layer_realization_debt.lifelong_analysis_figures",
        "--tables",
        str(tables_root.absolute()),
        "--verify-existing",
        str(output.absolute()),
    ]
    return shlex.join(command_argv)


def run_cli(tables_root: Path, output: Path) -> dict[str, Any]:
    np.random.seed(PLOT_SEED)
    plt.style.use(PLOT_STYLE)
    tables = load_figure_tables(tables_root)
    names = save_figures(output, tables)
    command = _reproduction_command(tables_root, output)
    receipt = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-fourarm-plot-reproduction.v2",
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "command": command,
        "working_directory": str(Path.cwd().resolve()),
        "seed": PLOT_SEED,
        "style": PLOT_STYLE,
        "dpi": PLOT_DPI,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "backend": matplotlib.get_backend(),
        },
        "figure_geometry": {
            "cell_panels_inches": [13, 9],
            "compute_inches": [11, 6],
            "cell_panel_layout": "rows=Llama,Qwen; columns=MEMIT,AlphaEdit",
            "q_axis": "pre-L4,pre-L5,pre-L6,pre-L7,pre-L8,post-L8",
            "layer_axis": [4, 5, 6, 7, 8],
            "axis_ranges": AXIS_RANGES,
            "axis_units": {
                "q": "dimensionless residual norm / entry residual norm",
                "rho_tau_debt": "dimensionless",
                "weight_magnitude": "Frobenius norm, log display axis",
                "strict_rate": "fraction",
                "compute": "hours",
            },
        },
        "panel_order": [[model, method] for model, method in CELL_ORDER],
        "missing_policy": "NO_INTERPOLATION_NO_IMPUTATION",
        "derived_table_seal": {
            "path": "derived-table-seal.json",
            "bytes": (tables_root / "derived-table-seal.json").stat().st_size,
            "sha256": _sha256(tables_root / "derived-table-seal.json"),
        },
        "inputs": [{"path": name, "bytes": (tables_root / name).stat().st_size, "sha256": _sha256(tables_root / name)} for name in PLOT_INPUTS],
        "input_row_counts": {"checkpoint_request": len(tables.checkpoint_request), "checkpoint_layer": len(tables.checkpoint_layer), "checkpoint_weight": len(tables.checkpoint_weight), "production_weight": len(tables.production_weight), "functional": len(tables.functional), "compute": len(tables.compute)},
        "captions": {
            "primary-residual-trajectory-2x2.png": "10k sentinel n=100/cell; display range q=[0,4.5]; full extrema remain in CSV; missing values are not interpolated or imputed.",
            "checkpoint-layer-q-heatmap.png": "Sentinel n=100/checkpoint/cell; medians by fixed checkpoint schedule; shared color range q=[0,1.5].",
            "rho-by-layer-terminal.png": "10k sentinel n=100/layer/cell; shared display range rho=[-0.5,9].",
            "tau-by-layer-terminal.png": "10k sentinel n=100/layer/cell; shared display range tau=[0,22].",
            "inherited-debt-distributions.png": "Sentinel n=100/checkpoint/cell at t0,5k,10k; display range [-1,2], full extrema in CSV.",
            "checkpoint-layer-rho-heatmap.png": "Sentinel n=100/checkpoint/layer/cell; shared median color range rho=[0,6].",
            "checkpoint-layer-tau-heatmap.png": "Sentinel n=100/checkpoint/layer/cell; shared median color range tau=[0,8].",
            "layer-wise-update-magnitude.png": "Weight observation unit is one B100 batch; n=100/layer/cell; log display range [1,1000].",
            "layer-wise-update-magnitude-drift.png": "One observed B100 update per batch/layer/cell; log1p color range [0,6.5]; no interpolation.",
            "endpoint-checkpoint-strict-rates.png": "Current B100 functional panel n=100 requests/checkpoint/cell; prompt-count weighted within request only.",
            "compute-time-comparison.png": "Recorded edit-core, observer and cell wall time; GPU peak memory is not inferred.",
        },
        "figures": [{"path": name, "bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)} for name in names],
    }
    path = output / "plot-reproduction.json"
    raw = (json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    return receipt


def verify_existing(tables_root: Path, existing_output: Path) -> dict[str, Any]:
    """Re-render into an ephemeral directory and verify byte-stable PNGs."""

    receipt_path = existing_output / "plot-reproduction.json"
    expected = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_inputs = {row["path"]: row["sha256"] for row in expected["inputs"]}
    actual_inputs = {name: _sha256(tables_root / name) for name in PLOT_INPUTS}
    if actual_inputs != expected_inputs:
        raise RuntimeError("plot input table SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="odeedit-lifelong-plot-verify-") as temporary:
        rendered = run_cli(tables_root, Path(temporary))
        expected_figures = {
            row["path"]: (row["bytes"], row["sha256"])
            for row in expected["figures"]
        }
        actual_figures = {
            row["path"]: (row["bytes"], row["sha256"])
            for row in rendered["figures"]
        }
        if actual_figures != expected_figures:
            raise RuntimeError("re-rendered figure bytes/SHA differ from sealed figures")
    return {
        "status": "BYTE_STABLE_REPRODUCTION_PASS",
        "figure_count": len(expected_figures),
        "input_count": len(expected_inputs),
        "existing_output": str(existing_output.absolute()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tables", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        run_cli(args.tables, args.output)
    else:
        print(json.dumps(verify_existing(args.tables, args.verify_existing), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["FigureTables", "PLOT_DPI", "PLOT_INPUTS", "PLOT_SEED", "PLOT_STYLE", "checkpoint_layer_heatmap_figure", "checkpoint_q_heatmap_figure", "compute_comparison_figure", "endpoint_checkpoint_figure", "inherited_debt_figure", "load_figure_tables", "primary_residual_figure", "rho_figure", "run_cli", "save_figures", "tau_figure", "update_drift_heatmap_figure", "verify_existing", "weight_magnitude_figure"]
