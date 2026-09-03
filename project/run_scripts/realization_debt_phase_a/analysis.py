"""Build the analysis-only Realization Debt Phase A canonical package.

The builder reads the three sealed v3 production tables, validates their exact
denominators and joins, computes row-wise debt before aggregation, and emits a
create-once package.  It never imports torch or an experiment runtime.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from .contracts import (
    ACTION_KEYS,
    AnalysisBoundary,
    CONTEXT_MEMBERS,
    CONTRACT_SHA256,
    EXPECTED_ACTION_ROWS,
    EXPECTED_ARMS,
    EXPECTED_BATCHES_PER_ARM,
    EXPECTED_LAYER_ROWS,
    EXPECTED_LAYERS,
    EXPECTED_REQUEST_ROWS,
    EXPECTED_REQUESTS_PER_BATCH,
    INSTRUCTION_ID,
    LAYER_KEYS,
    PRIMARY_INPUTS,
    REQUEST_KEYS,
    SOURCE_BASE_HEAD,
    SOURCE_BASE_TREE,
    canonical_hash,
    member,
    regular_file,
    sha256_file,
    write_json_once,
)


V3_DIR = "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3"
VERSION_DIRS = {
    "v3": V3_DIR,
    "v4-r2": "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2",
    "v5": "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v5",
    "v6": "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6",
}
ENDPOINT_FIELDS = (
    "target_new_nll",
    "target_true_nll",
    "target_new_margin",
    "target_true_margin",
    "target_new_strict",
    "target_true_strict",
)
ROW_METRICS = (
    "rho",
    "tau",
    "debt_native",
    "debt_under",
    "debt_over",
    "debt_opposite",
    "debt_orthogonal",
    "normalized_potential_reduction",
)
DEBT_METRICS = (
    "debt_native",
    "debt_under",
    "debt_over",
    "debt_opposite",
    "debt_orthogonal",
    "normalized_potential_reduction",
)
SOURCE_FILES = (
    "project/run_scripts/realization_debt_phase_a/__init__.py",
    "project/run_scripts/realization_debt_phase_a/contracts.py",
    "project/run_scripts/realization_debt_phase_a/analysis.py",
    "project/run_scripts/realization_debt_phase_a/figures.py",
    "project/run_scripts/realization_debt_phase_a/package_verify.py",
    "project/run_scripts/realization_debt_phase_a/tests/__init__.py",
    "project/run_scripts/realization_debt_phase_a/tests/test_analysis.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def write_csv_once(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> None:
    if frame.empty:
        raise AnalysisBoundary(f"refusing empty derived table: {path.name}")
    raw = _csv_bytes(frame)
    if compressed:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
            handle.write(raw)
        raw = buffer.getvalue()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def _read_csv(path: Path) -> pd.DataFrame:
    regular_file(path)
    return pd.read_csv(path, low_memory=False)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AnalysisBoundary(f"{label} missing columns: {missing}")


def _numeric_finite(frame: pd.DataFrame, fields: Sequence[str], label: str) -> None:
    _require_columns(frame, fields, label)
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    values = frame[list(fields)].to_numpy(dtype=np.float64)
    count = int((~np.isfinite(values)).sum())
    if count:
        raise AnalysisBoundary(f"{label} core nonfinite count={count}")


def _unique_key(frame: pd.DataFrame, keys: Sequence[str], expected: int, label: str) -> None:
    if len(frame) != expected:
        raise AnalysisBoundary(f"{label} rows differ: {len(frame)} != {expected}")
    duplicate_count = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        raise AnalysisBoundary(f"{label} duplicate composite-key rows={duplicate_count}")


def _fixed_group_summary(
    frame: pd.DataFrame,
    keys: Sequence[str],
    metrics: Sequence[str],
    group_size: int,
) -> pd.DataFrame:
    ordered = frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    sizes = ordered.groupby(list(keys), sort=True, dropna=False).size()
    if sizes.empty or not (sizes.to_numpy() == group_size).all():
        bad = sizes[sizes != group_size].head(10).to_dict()
        raise AnalysisBoundary(f"fixed group size differs from {group_size}: {bad}")
    groups = len(ordered) // group_size
    key_rows = ordered.loc[np.arange(groups) * group_size, list(keys)].reset_index(drop=True)
    result = key_rows.copy()
    top_count = max(1, int(math.ceil(0.1 * group_size)))
    for metric in metrics:
        values = ordered[metric].to_numpy(dtype=np.float64).reshape(groups, group_size)
        sorted_values = np.sort(values, axis=1)
        result[f"{metric}_mean"] = values.mean(axis=1)
        result[f"{metric}_median"] = np.quantile(values, 0.5, axis=1, method="linear")
        result[f"{metric}_p90"] = np.quantile(values, 0.9, axis=1, method="linear")
        result[f"{metric}_p99"] = np.quantile(values, 0.99, axis=1, method="linear")
        result[f"{metric}_cvar90"] = sorted_values[:, -top_count:].mean(axis=1)
        result[f"{metric}_max"] = sorted_values[:, -1]
    result["row_denominator"] = group_size
    return result


def _stat(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise AnalysisBoundary("empty/nonfinite summary input")
    top_count = max(1, int(math.ceil(0.1 * len(values))))
    ordered = np.sort(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5, method="linear")),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "cvar90": float(ordered[-top_count:].mean()),
        "max": float(ordered[-1]),
    }


def _input_inventory(repo: Path) -> list[dict[str, Any]]:
    base = repo / "experiment-reports/servers/server4"
    rows: list[dict[str, Any]] = []
    for name, expected_sha in PRIMARY_INPUTS.items():
        path = base / V3_DIR / name
        row = member(path, relative_to=repo, kind="primary")
        if row["sha256"] != expected_sha:
            raise AnalysisBoundary(f"sealed primary SHA differs: {name}")
        rows.append(row)
    for version, name in CONTEXT_MEMBERS:
        path = base / VERSION_DIRS[version] / name
        rows.append(member(path, relative_to=repo, kind=f"context_{version}"))
    return rows


def _load_validate(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = repo / "experiment-reports/servers/server4" / V3_DIR
    layer = _read_csv(base / "production-layer-complete.csv.gz")
    action = _read_csv(base / "production-weight-batch-unit-complete.csv.gz")
    endpoint = _read_csv(base / "production-request-complete.csv.gz")
    _require_columns(layer, LAYER_KEYS + ("case_identity_sha256", "q_pre", "q_post", "rho", "tau"), "layer")
    _require_columns(action, ACTION_KEYS + ("frobenius_squared_telemetry",), "action")
    _require_columns(endpoint, REQUEST_KEYS + ("case_identity_sha256",) + ENDPOINT_FIELDS, "endpoint")
    _unique_key(layer, LAYER_KEYS, EXPECTED_LAYER_ROWS, "layer")
    _unique_key(action, ACTION_KEYS, EXPECTED_ACTION_ROWS, "action")
    _unique_key(endpoint, REQUEST_KEYS, EXPECTED_REQUEST_ROWS, "endpoint")
    _numeric_finite(layer, ("q_pre", "q_post", "rho", "tau"), "layer")
    _numeric_finite(action, ("frobenius_squared_telemetry",), "action")
    _numeric_finite(endpoint, ENDPOINT_FIELDS, "endpoint")
    for frame in (layer, action, endpoint):
        frame["batch_index"] = pd.to_numeric(frame["batch_index"], errors="raise").astype(int)
    layer["layer"] = pd.to_numeric(layer["layer"], errors="raise").astype(int)
    action["layer"] = pd.to_numeric(action["layer"], errors="raise").astype(int)
    arms = layer[["model", "method"]].drop_duplicates().sort_values(["model", "method"])
    if len(arms) != EXPECTED_ARMS:
        raise AnalysisBoundary(f"arm denominator differs: {len(arms)}")
    arm_set = set(map(tuple, arms.to_numpy().tolist()))
    if set(map(tuple, action[["model", "method"]].drop_duplicates().to_numpy().tolist())) != arm_set:
        raise AnalysisBoundary("action arm identity differs")
    if set(map(tuple, endpoint[["model", "method"]].drop_duplicates().to_numpy().tolist())) != arm_set:
        raise AnalysisBoundary("endpoint arm identity differs")
    expected_batches = set(range(1, EXPECTED_BATCHES_PER_ARM + 1))
    for arm in sorted(arm_set):
        for frame, label in ((layer, "layer"), (action, "action"), (endpoint, "endpoint")):
            selected = frame[(frame.model == arm[0]) & (frame.method == arm[1])]
            if set(selected.batch_index.unique()) != expected_batches:
                raise AnalysisBoundary(f"{label} batch identity differs: {arm}")
        counts = endpoint[(endpoint.model == arm[0]) & (endpoint.method == arm[1])].groupby("batch_index").size()
        if not (counts.to_numpy() == EXPECTED_REQUESTS_PER_BATCH).all():
            raise AnalysisBoundary(f"request denominator differs: {arm}")
    if set(layer.layer.unique()) != set(EXPECTED_LAYERS) or set(action.layer.unique()) != set(EXPECTED_LAYERS):
        raise AnalysisBoundary("layer identity differs from L4-L8")
    layer_counts = layer.groupby(list(REQUEST_KEYS)).size()
    if not (layer_counts.to_numpy() == len(EXPECTED_LAYERS)).all():
        raise AnalysisBoundary("request-layer denominator differs from five")
    layer_request = layer[list(REQUEST_KEYS)].drop_duplicates()
    endpoint_request = endpoint[list(REQUEST_KEYS)].drop_duplicates()
    request_join = layer_request.merge(endpoint_request, on=list(REQUEST_KEYS), how="outer", indicator=True)
    endpoint_missing = int((request_join._merge != "both").sum())
    layer_action = layer[["model", "method", "batch_index", "layer"]].drop_duplicates()
    action_key = action[list(ACTION_KEYS)].drop_duplicates()
    action_join = layer_action.merge(action_key, on=list(ACTION_KEYS), how="outer", indicator=True)
    action_missing = int((action_join._merge != "both").sum())
    if endpoint_missing or action_missing:
        raise AnalysisBoundary(f"join identity differs: endpoint={endpoint_missing} action={action_missing}")
    for field in ("target_new_strict", "target_true_strict"):
        if not endpoint[field].isin([0, 1]).all():
            raise AnalysisBoundary(f"strict field outside 0/1: {field}")
    gates = {
        "arm_count": len(arm_set),
        "batch_count_per_arm": EXPECTED_BATCHES_PER_ARM,
        "requests_per_arm_batch": EXPECTED_REQUESTS_PER_BATCH,
        "layers_per_request": len(EXPECTED_LAYERS),
        "layer_rows": len(layer),
        "request_rows": len(endpoint),
        "action_rows": len(action),
        "layer_key_duplicate_rows": 0,
        "request_key_duplicate_rows": 0,
        "action_key_duplicate_rows": 0,
        "layer_to_endpoint_join_missing": endpoint_missing,
        "layer_to_action_join_missing": action_missing,
        "core_nonfinite_count": 0,
    }
    return layer, action, endpoint, gates


def _derive_layer(layer: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = layer[list(LAYER_KEYS) + ["case_identity_sha256", "q_pre", "q_post", "rho", "tau"]].copy()
    rho = out.rho.to_numpy(dtype=np.float64)
    tau = out.tau.to_numpy(dtype=np.float64)
    parallel = np.square(1.0 - rho)
    out["debt_under"] = np.where((rho >= 0.0) & (rho < 1.0), parallel, 0.0)
    out["debt_over"] = np.where(rho > 1.0, np.square(rho - 1.0), 0.0)
    out["debt_opposite"] = np.where(rho < 0.0, parallel, 0.0)
    out["debt_orthogonal"] = np.square(tau)
    out["debt_native"] = parallel + out["debt_orthogonal"].to_numpy(dtype=np.float64)
    out["normalized_potential_reduction"] = 0.5 * (
        np.square(out.q_pre.to_numpy(dtype=np.float64)) - np.square(out.q_post.to_numpy(dtype=np.float64))
    )
    decomposition = (
        out.debt_under.to_numpy(dtype=np.float64)
        + out.debt_over.to_numpy(dtype=np.float64)
        + out.debt_opposite.to_numpy(dtype=np.float64)
        + out.debt_orthogonal.to_numpy(dtype=np.float64)
    )
    absolute_error = np.abs(out.debt_native.to_numpy(dtype=np.float64) - decomposition)
    tolerance = np.finfo(np.float64).eps * 8.0 * np.maximum(1.0, np.abs(out.debt_native.to_numpy(dtype=np.float64)))
    failures = int((absolute_error > tolerance).sum())
    if failures:
        raise AnalysisBoundary(f"debt decomposition identity failures={failures}")
    if not np.isfinite(out[list(ROW_METRICS)].to_numpy(dtype=np.float64)).all():
        raise AnalysisBoundary("derived row debt contains nonfinite values")
    out = out.sort_values(list(LAYER_KEYS), kind="mergesort").reset_index(drop=True)
    return out, {
        "debt_decomposition_identity_failure_count": failures,
        "debt_decomposition_max_abs_error": float(absolute_error.max()),
        "debt_decomposition_tolerance_rule": "8*eps_fp64*max(1,abs(debt_native)) row-wise",
    }


def _request_endpoint_join(derived: pd.DataFrame, endpoint: pd.DataFrame) -> pd.DataFrame:
    request_summary = _fixed_group_summary(derived, REQUEST_KEYS, DEBT_METRICS, len(EXPECTED_LAYERS))
    rho_tau = derived.groupby(list(REQUEST_KEYS), sort=True, as_index=False).agg(
        rho_mean=("rho", "mean"),
        rho_median=("rho", "median"),
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        tau_mean=("tau", "mean"),
        tau_median=("tau", "median"),
        tau_max=("tau", "max"),
    )
    request_summary = request_summary.merge(rho_tau, on=list(REQUEST_KEYS), validate="one_to_one")
    endpoint_subset = endpoint[list(REQUEST_KEYS) + ["case_identity_sha256"] + list(ENDPOINT_FIELDS)].copy()
    joined = request_summary.merge(endpoint_subset, on=list(REQUEST_KEYS), how="left", validate="one_to_one", indicator=True)
    missing = int((joined._merge != "both").sum())
    if missing:
        raise AnalysisBoundary(f"request endpoint join missing={missing}")
    joined = joined.drop(columns="_merge")
    joined["rs_current"] = (joined.target_new_nll < joined.target_true_nll).astype(int)
    joined["nll_tie"] = (joined.target_new_nll == joined.target_true_nll).astype(int)
    joined["nll_advantage"] = joined.target_true_nll - joined.target_new_nll
    return joined.sort_values(list(REQUEST_KEYS), kind="mergesort").reset_index(drop=True)


def _batch_relationship(request: pd.DataFrame, action_proxy: pd.DataFrame) -> pd.DataFrame:
    relationship = request.groupby(["model", "method", "batch_index"], sort=True, as_index=False).agg(
        debt_native_mean=("debt_native_mean", "mean"),
        debt_native_median=("debt_native_mean", "median"),
        debt_native_p90=("debt_native_mean", lambda value: value.quantile(0.9, interpolation="linear")),
        debt_native_max=("debt_native_max", "max"),
        rs_current_count=("rs_current", "sum"),
        rs_current_rate=("rs_current", "mean"),
        nll_tie_count=("nll_tie", "sum"),
        nll_advantage_mean=("nll_advantage", "mean"),
        nll_advantage_median=("nll_advantage", "median"),
        target_new_nll_mean=("target_new_nll", "mean"),
        target_true_nll_mean=("target_true_nll", "mean"),
        target_new_strict_count=("target_new_strict", "sum"),
        target_new_strict_rate=("target_new_strict", "mean"),
        target_true_strict_count=("target_true_strict", "sum"),
        target_true_strict_rate=("target_true_strict", "mean"),
        request_denominator=("request_sha256", "size"),
    )
    action_batch = action_proxy.groupby(["model", "method", "batch_index"], sort=True, as_index=False).agg(
        frobenius_action_sq_sum_layers=("frobenius_action_sq", "sum"),
        frobenius_action_sq_mean_layers=("frobenius_action_sq", "mean"),
        progress_eff_frobenius_proxy_mean_layers=("progress_eff_frobenius_proxy", "mean"),
        debt_per_action_frobenius_proxy_mean_layers=("debt_per_action_frobenius_proxy", "mean"),
    )
    relationship = relationship.merge(action_batch, on=["model", "method", "batch_index"], validate="one_to_one")
    if not (relationship.request_denominator == EXPECTED_REQUESTS_PER_BATCH).all():
        raise AnalysisBoundary("batch relationship request denominator differs")
    return relationship


def _association_summary(relationship: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcomes = ("rs_current_rate", "nll_advantage_mean", "target_new_strict_rate")
    for (model, method), frame in relationship.groupby(["model", "method"], sort=True):
        x = frame.debt_native_mean.astype(float)
        for outcome in outcomes:
            y = frame[outcome].astype(float)
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "debt_metric": "debt_native_mean",
                    "immediate_outcome": outcome,
                    "batch_denominator": len(frame),
                    "pearson_r": float(x.corr(y, method="pearson")),
                    "spearman_rho": float(x.rank(method="average").corr(y.rank(method="average"), method="pearson")),
                    "interpretation_boundary": "CONTEMPORANEOUS_DESCRIPTIVE_ASSOCIATION_ONLY",
                }
            )
    return pd.DataFrame(rows)


def _arm_summary(derived: pd.DataFrame, request: pd.DataFrame, action_proxy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method), layer_rows in derived.groupby(["model", "method"], sort=True):
        req = request[(request.model == model) & (request.method == method)]
        action = action_proxy[(action_proxy.model == model) & (action_proxy.method == method)]
        row: dict[str, Any] = {
            "model": model,
            "method": method,
            "layer_row_denominator": len(layer_rows),
            "request_denominator": len(req),
            "batch_layer_action_denominator": len(action),
        }
        for metric in ROW_METRICS:
            for name, value in _stat(layer_rows[metric].to_numpy(dtype=np.float64)).items():
                row[f"{metric}_{name}"] = value
        for metric in ("target_new_nll", "target_true_nll", "nll_advantage"):
            for name, value in _stat(req[metric].to_numpy(dtype=np.float64)).items():
                row[f"{metric}_{name}"] = value
        row.update(
            {
                "rs_current_count": int(req.rs_current.sum()),
                "rs_current_rate": float(req.rs_current.mean()),
                "nll_tie_count": int(req.nll_tie.sum()),
                "target_new_strict_count": int(req.target_new_strict.sum()),
                "target_new_strict_rate": float(req.target_new_strict.mean()),
                "target_true_strict_count": int(req.target_true_strict.sum()),
                "target_true_strict_rate": float(req.target_true_strict.mean()),
                "rho_under_fraction": float(((layer_rows.rho >= 0) & (layer_rows.rho < 1)).mean()),
                "rho_over_fraction": float((layer_rows.rho > 1).mean()),
                "rho_opposite_fraction": float((layer_rows.rho < 0).mean()),
                "frobenius_action_sq_sum": float(action.frobenius_action_sq.sum()),
                "frobenius_action_sq_mean": float(action.frobenius_action_sq.mean()),
                "progress_eff_frobenius_proxy_mean": float(action.progress_eff_frobenius_proxy.mean()),
                "debt_per_action_frobenius_proxy_mean": float(action.debt_per_action_frobenius_proxy.mean()),
                "exact_g_metric_action": "NOT_COMPUTABLE_FROM_SCHEMA",
                "absolute_allocation_energy_weighted_debt": "NOT_COMPUTABLE_FROM_SCHEMA",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _arm_layer_tail(derived: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method, layer), frame in derived.groupby(["model", "method", "layer"], sort=True):
        row: dict[str, Any] = {
            "model": model,
            "method": method,
            "layer": int(layer),
            "request_layer_denominator": len(frame),
        }
        for metric in ROW_METRICS:
            for name, value in _stat(frame[metric].to_numpy(dtype=np.float64)).items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_method_batch(relationship: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "debt_native_mean",
        "debt_native_p90",
        "debt_native_max",
        "rs_current_rate",
        "nll_advantage_mean",
        "target_new_strict_rate",
        "frobenius_action_sq_sum_layers",
    )
    memit = relationship[relationship.method == "memit"].set_index(["model", "batch_index"])
    alpha = relationship[relationship.method == "alphaedit"].set_index(["model", "batch_index"])
    if not memit.index.equals(alpha.index):
        raise AnalysisBoundary("paired AlphaEdit/MEMIT batch identity differs")
    rows = alpha[list(fields)] - memit[list(fields)]
    rows.columns = [f"alphaedit_minus_memit_{field}" for field in fields]
    return rows.reset_index()


def _format(value: Any, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_format(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    inputs: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    arm: pd.DataFrame,
    tail: pd.DataFrame,
    assoc: pd.DataFrame,
    output_inventory: Sequence[Mapping[str, Any]],
) -> str:
    arm_rows = []
    for row in arm.to_dict("records"):
        arm_rows.append(
            (
                row["model"], row["method"], row["debt_native_mean"], row["debt_native_median"],
                row["debt_native_p90"], row["debt_native_cvar90"], row["debt_native_max"],
                100.0 * row["rs_current_rate"], row["nll_advantage_mean"],
                100.0 * row["target_new_strict_rate"], row["frobenius_action_sq_sum"],
            )
        )
    tail_rows = []
    for row in tail.to_dict("records"):
        tail_rows.append(
            (row["model"], row["method"], row["layer"], row["debt_native_median"],
             row["debt_native_p90"], row["debt_native_p99"], row["debt_native_cvar90"], row["debt_native_max"])
        )
    assoc_rows = []
    for row in assoc.to_dict("records"):
        assoc_rows.append((row["model"], row["method"], row["immediate_outcome"], row["pearson_r"], row["spearman_rho"]))
    input_rows = [(row["kind"], row["path"], row["bytes"], row["sha256"]) for row in inputs]
    output_rows = [(row["path"], row["bytes"], row["sha256"]) for row in output_inventory]
    lowest = arm.loc[arm.debt_native_mean.idxmin()]
    highest = arm.loc[arm.debt_native_mean.idxmax()]
    lines = [
        "# Realization Debt Phase A — lifelong v6 추가 분석 사실 보고서",
        "",
        "## 0. 판정과 해석 경계",
        "",
        "- 상태: `ANALYSIS_ONLY_TERMINAL_PASS`",
        "- 새 editing run/checkpoint evaluation/model load/GPU/Slurm action: `0`.",
        "- 이 보고서는 동일 B100 write의 request-layer realization debt와 직후 current-B100 rewrite 관측 간의 **동시점 기술적 연관**만 기술한다.",
        "- 인과, 미래 forgetting, global locality, 자동 promotion을 주장하지 않는다. `scientific_promotion=false`.",
        "- `absolute allocation-energy weighted debt`와 exact `G`-metric action은 raw schema가 없어 `NOT_COMPUTABLE_FROM_SCHEMA`이다.",
        "- 기존 `d_parallel/d_perp`는 inherited L8 trajectory debt이므로 본 보고서의 per-write debt 명칭으로 재사용하지 않았다.",
        "",
        "## 1. 한눈에 보는 arm별 사실값",
        "",
        _markdown_table(
            ["model", "method", "debt mean", "median", "p90", "CVaR90", "max", "RS current %", "NLL advantage mean", "target-new strict %", "Σ Frobenius action²"],
            arm_rows,
        ),
        "",
        f"행 단위 평균 debt_native 최솟값은 `{lowest['model']}/{lowest['method']}`의 {_format(lowest['debt_native_mean'])}, 최댓값은 `{highest['model']}/{highest['method']}`의 {_format(highest['debt_native_mean'])}이다. 이는 서술적 순위이며 성능 원인의 증거가 아니다.",
        "",
        "## 2. 정의와 산출 순서",
        "",
        "각 raw request-layer 행에서 먼저 다음을 계산했다.",
        "",
        "- `debt_native=(1-rho)^2+tau^2`",
        "- `debt_under=1[0<=rho<1](1-rho)^2`",
        "- `debt_over=1[rho>1](rho-1)^2`",
        "- `debt_opposite=1[rho<0](1-rho)^2`",
        "- `debt_orthogonal=tau^2`",
        "- `normalized_potential_reduction=0.5*(q_pre^2-q_post^2)`",
        "",
        "그 뒤 arm×batch×layer의 정확히 100개 행에서 mean/median/p90/p99/top-decile CVaR0.9/max를 계산했다. percentile은 NumPy linear quantile, CVaR0.9는 정렬된 상위 10개 행의 산술평균이다. 요약 rho/tau에 debt 식을 다시 적용하지 않았다.",
        "",
        "weight action은 request별로 복제하지 않았다. 먼저 100개 request를 batch×layer로 집계한 뒤 한 개의 `frobenius_squared_telemetry`와 1:1 결합했다. 분모 `+1e-12`를 사용한 두 proxy만 `progress_eff_frobenius_proxy`, `debt_per_action_frobenius_proxy`로 명명했다.",
        "",
        "request endpoint 결합은 `(model,method,batch_index,request_sha256)` 1:1이며 `RS_current=1[target_new_nll<target_true_nll]`; tie는 failure이다. `nll_advantage=target_true_nll-target_new_nll`이다. PS/NS는 만들지 않았다.",
        "",
        "## 3. 정확한 분모와 hard gate",
        "",
        _markdown_table(["gate", "value"], [(key, value) for key, value in sorted(gates.items())]),
        "",
        "모든 composite key는 unique이고 layer→endpoint, layer→action join 누락은 0이다. core/derived nonfinite, debt decomposition failure, interpolation, imputation은 모두 0이다.",
        "",
        "## 4. Layer별 tail",
        "",
        _markdown_table(["model", "method", "layer", "median", "p90", "p99", "CVaR90", "max"], tail_rows),
        "",
        "전체 batch×layer 세부값은 `batch-layer-debt-summary.csv`, action 결합은 `batch-layer-action-proxy.csv`에 있다.",
        "",
        "## 5. Immediate rewrite와의 동시점 연관",
        "",
        _markdown_table(["model", "method", "outcome", "Pearson r", "Spearman rho"], assoc_rows),
        "",
        "상관계수의 관측 단위는 arm별 100개 batch이다. 같은 batch의 debt와 바로 뒤 current-B100 endpoint를 연결했을 뿐, 미래 forgetting이나 causal mediation으로 해석하지 않는다.",
        "",
        "## 6. Figure",
        "",
        "- `median-rho-batch-layer-heatmap.png`: arm별 batch1–100×L4–L8 median rho",
        "- `median-tau-batch-layer-heatmap.png`: arm별 batch1–100×L4–L8 median tau",
        "- `median-debt-native-batch-layer-heatmap.png`: arm별 row-wise debt_native median",
        "- `debt-frobenius-action-joint-trajectory.png`: batch×layer debt/action joint trajectory",
        "- `debt-immediate-rewrite-relationship.png`: batch debt와 RS/NLL advantage/strict rate",
        "",
        "모든 PNG는 저장소 Python CLI로 동일 CSV를 두 번 실제 실행해 byte SHA가 같은지 검증했다. 정확한 명령·입력/출력 SHA·Python/NumPy/pandas/Matplotlib/Pillow 버전은 `plot-reproduction.json`에 있다.",
        "",
        "## 7. 입력 identity",
        "",
        _markdown_table(["kind", "path", "bytes", "sha256"], input_rows),
        "",
        "세 primary gzip SHA는 산출 전후 재검증해 동일했다. v3–v6 manifest/receipt/report는 context binding에만 사용했고 원본 bytes를 수정하지 않았다.",
        "",
        "## 8. 출력 identity (report/manifest/receipt 이전 산출물)",
        "",
        _markdown_table(["path", "bytes", "sha256"], output_rows),
        "",
        "최종 전체 member root와 report/manifest/receipt SHA는 `analysis-manifest.json` 및 `rooted-analysis-receipt.json`이 결속한다.",
        "",
        "## 9. 한계",
        "",
        "- raw schema에는 `R_entry_norm_sq`, `A_norm_sq`, exact allocation-energy weight, `DeltaW^T G DeltaW`가 없다.",
        "- Frobenius proxy를 exact G-action 또는 request-level action으로 해석하지 않는다.",
        "- batch action을 100 request에 복제하지 않았다.",
        "- current-B100 endpoint만 사용했으며 canonical PS/NS 또는 미래 checkpoint 결과를 재구성하지 않았다.",
        "- interpolation/imputation/reconstruction은 모두 0이다.",
        "",
    ]
    return "\n".join(lines)


def build(repo: Path, output: Path, repro_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    if output.exists() or output.is_symlink():
        raise AnalysisBoundary(f"create-once output already exists: {output}")
    if _git(repo, "rev-parse", f"{SOURCE_BASE_HEAD}^{{tree}}") != SOURCE_BASE_TREE:
        raise AnalysisBoundary("sealed analysis base tree differs")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", SOURCE_BASE_HEAD, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise AnalysisBoundary("sealed analysis base is not an ancestor of source HEAD")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        # Source edits are expected while developing, but the production build is
        # only allowed after they are staged/committed by the caller.
        raise AnalysisBoundary("tracked source worktree is not clean")
    inputs_before = _input_inventory(repo)
    input_root = canonical_hash(inputs_before)
    layer, action, endpoint, gates = _load_validate(repo)
    derived, identity_gate = _derive_layer(layer)
    gates.update(identity_gate)
    gates.update(
        {
            "imputation_count": 0,
            "interpolation_count": 0,
            "reconstruction_count": 0,
            "batch_action_request_replication_count": 0,
            "scientific_promotion": False,
        }
    )
    batch_layer = _fixed_group_summary(derived, ("model", "method", "batch_index", "layer"), ROW_METRICS, 100)
    action_input = action[list(ACTION_KEYS) + ["frobenius_squared_telemetry"]].rename(
        columns={"frobenius_squared_telemetry": "frobenius_action_sq"}
    )
    action_proxy = batch_layer.merge(action_input, on=list(ACTION_KEYS), how="left", validate="one_to_one", indicator=True)
    missing_action = int((action_proxy._merge != "both").sum())
    if missing_action:
        raise AnalysisBoundary(f"batch-layer action join missing={missing_action}")
    action_proxy = action_proxy.drop(columns="_merge")
    action_proxy["progress_eff_frobenius_proxy"] = (
        action_proxy.normalized_potential_reduction_mean / (action_proxy.frobenius_action_sq + 1e-12)
    )
    action_proxy["debt_per_action_frobenius_proxy"] = (
        action_proxy.debt_native_mean / (action_proxy.frobenius_action_sq + 1e-12)
    )
    request = _request_endpoint_join(derived, endpoint)
    relationship = _batch_relationship(request, action_proxy)
    association = _association_summary(relationship)
    arm = _arm_summary(derived, request, action_proxy)
    tail = _arm_layer_tail(derived)
    paired = _paired_method_batch(relationship)
    output.mkdir(parents=True, mode=0o755)
    table_map = {
        "request-layer-debt.csv.gz": (derived, True),
        "batch-layer-debt-summary.csv": (batch_layer, False),
        "batch-layer-action-proxy.csv": (action_proxy, False),
        "request-debt-endpoint.csv.gz": (request, True),
        "batch-endpoint-relationship.csv": (relationship, False),
        "association-summary.csv": (association, False),
        "arm-debt-summary.csv": (arm, False),
        "arm-layer-tail-summary.csv": (tail, False),
        "paired-method-batch-deltas.csv": (paired, False),
    }
    for name, (frame, compressed) in table_map.items():
        write_csv_once(output / name, frame, compressed=compressed)
    repro_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase-a-plot-repro-", dir=repro_root) as temporary:
        reproduce = Path(temporary)
        command_final = [sys.executable, "-m", "project.run_scripts.realization_debt_phase_a.figures", "--tables", str(output), "--output", str(output)]
        command_repro = [sys.executable, "-m", "project.run_scripts.realization_debt_phase_a.figures", "--tables", str(output), "--output", str(reproduce)]
        environment = dict(os.environ)
        environment.setdefault("MPLCONFIGDIR", str(repro_root / "mplconfig"))
        environment.setdefault("SOURCE_DATE_EPOCH", "0")
        subprocess.run(command_final, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        subprocess.run(command_repro, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        plot_rows: list[dict[str, Any]] = []
        for path in sorted(output.glob("*.png")):
            other = reproduce / path.name
            if not other.is_file() or other.is_symlink():
                raise AnalysisBoundary(f"missing reproduced plot: {path.name}")
            first_sha, second_sha = sha256_file(path), sha256_file(other)
            if first_sha != second_sha or path.stat().st_size != other.stat().st_size:
                raise AnalysisBoundary(f"plot byte reproduction differs: {path.name}")
            plot_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": first_sha,
                    "reproduced_bytes": other.stat().st_size,
                    "reproduced_sha256": second_sha,
                    "byte_stable": True,
                }
            )
    plot_inputs = [member(output / name, relative_to=output, kind="plot_input") for name in (
        "batch-layer-debt-summary.csv", "batch-layer-action-proxy.csv", "batch-endpoint-relationship.csv"
    )]
    plot_receipt = {
        "schema": "odeedit.s06.realization-debt-phase-a.plot-reproduction.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "BYTE_STABLE_REPRODUCTION_PASS",
        "actual_commands": [shlex.join(command_final), shlex.join(command_repro)],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": __import__("PIL").__version__,
            "backend": matplotlib.get_backend(),
            "MPLCONFIGDIR": environment["MPLCONFIGDIR"],
            "SOURCE_DATE_EPOCH": environment["SOURCE_DATE_EPOCH"],
        },
        "inputs": plot_inputs,
        "outputs": plot_rows,
    }
    plot_receipt["identity_sha256"] = canonical_hash(plot_receipt)
    write_json_once(output / "plot-reproduction.json", plot_receipt)
    inputs_after = _input_inventory(repo)
    if inputs_before != inputs_after or input_root != canonical_hash(inputs_after):
        raise AnalysisBoundary("sealed input identity changed during analysis")
    gates["input_sha_before_after_unchanged"] = True
    gates["plot_count"] = len(plot_rows)
    gates["plot_byte_reproduction_failure_count"] = 0
    gates["absolute_allocation_energy_weighted_debt"] = "NOT_COMPUTABLE_FROM_SCHEMA"
    gates["exact_g_metric_action"] = "NOT_COMPUTABLE_FROM_SCHEMA"
    audit = {
        "schema": "odeedit.s06.realization-debt-phase-a.analysis-gate.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "contract_sha256": CONTRACT_SHA256,
        "source_base": {"head": SOURCE_BASE_HEAD, "tree": SOURCE_BASE_TREE},
        "gates": gates,
        "inputs_before": inputs_before,
        "inputs_after": inputs_after,
        "input_root": input_root,
        "claim_boundary": "CONTEMPORANEOUS_DESCRIPTIVE_ASSOCIATION_ONLY",
        "scientific_promotion": False,
        "model_gpu_slurm_action_count": 0,
    }
    audit["identity_sha256"] = canonical_hash(audit)
    write_json_once(output / "analysis-audit.json", audit)
    pre_report_members = [
        member(path, relative_to=output, kind="derived_output")
        for path in sorted(output.iterdir())
        if path.is_file() and not path.is_symlink()
    ]
    inventory_frame = pd.DataFrame(pre_report_members)
    write_csv_once(output / "output-member-inventory.csv", inventory_frame)
    report_inventory = [member(path, relative_to=output, kind="derived_output") for path in sorted(output.iterdir()) if path.is_file() and not path.is_symlink()]
    report = _report(inputs_before, gates, arm, tail, association, report_inventory)
    report_path = output / "realization-debt-phase-a-lifelong-v6-detailed-factual-ko.md"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    sources = [member(repo / name, relative_to=repo, kind="analysis_source") for name in SOURCE_FILES]
    package_members = [
        member(path, relative_to=output, kind="canonical_output")
        for path in sorted(output.iterdir())
        if path.is_file() and not path.is_symlink()
    ]
    manifest = {
        "schema": "odeedit.s06.realization-debt-phase-a.analysis-manifest.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "ANALYSIS_ONLY_TERMINAL_PASS",
        "source_base": {"head": SOURCE_BASE_HEAD, "tree": SOURCE_BASE_TREE},
        "contract_sha256": CONTRACT_SHA256,
        "inputs": inputs_before,
        "input_root": input_root,
        "analysis_sources": sources,
        "analysis_source_root": canonical_hash(sources),
        "members": package_members,
        "member_root": canonical_hash(package_members),
        "denominators_and_gates": gates,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = output / "analysis-manifest.json"
    write_json_once(manifest_path, manifest)
    receipt = {
        "schema": "odeedit.s06.realization-debt-phase-a.rooted-analysis-receipt.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "ANALYSIS_ONLY_TERMINAL_PASS",
        "source_base": manifest["source_base"],
        "contract_sha256": CONTRACT_SHA256,
        "input_root": input_root,
        "analysis_source_root": manifest["analysis_source_root"],
        "member_root": manifest["member_root"],
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "identity_sha256": manifest["identity_sha256"],
        },
        "report": {
            "path": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "gates": gates,
        "scientific_promotion": False,
        "model_gpu_slurm_action_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = output / "rooted-analysis-receipt.json"
    write_json_once(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output": str(output),
        "report_sha256": receipt["report"]["sha256"],
        "manifest_sha256": receipt["manifest"]["sha256"],
        "manifest_identity": receipt["manifest"]["identity_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_identity": receipt["identity_sha256"],
        "member_root": receipt["member_root"],
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output, args.repro_root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
