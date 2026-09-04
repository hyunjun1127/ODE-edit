"""Build the analysis-only ORBODE round-0 B100 canonical package.

The builder rehashes immutable job 35694 results, independently recomputes all
CounterFact endpoint metrics from prompt rows, summarizes recorded mechanism
telemetry, and emits deterministic tables/figures/receipts.  It never imports
torch and never invokes a model, GPU, or Slurm submission command.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from .round0_analysis_contracts import (
    AUTHORITATIVE_DOCUMENTS,
    CELL_BINDINGS,
    DYNAMIC_ARMS,
    EXPECTED_DYNAMIC_STEPS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_PRIMARY_ENDPOINTS,
    EXPECTED_REQUEST_STEPS,
    INSTRUCTION_ID,
    KIND_COUNTS,
    LAYERS,
    NONCE,
    ORDER_ROOT,
    PRIMARY_ARMS,
    RAW_SOURCE_HEAD,
    RAW_SOURCE_TREE,
    ROUND_JOB_ID,
    STAGE_ORDER,
    STREAM_ROOT,
    SWEEPS,
    AnalysisBoundary,
    canonical_hash,
    member,
    regular_file,
    sha256_file,
    verify_canonical_identity,
    write_json_once,
)


SOURCE_FILES = (
    "project/run_scripts/ordered_response_barrier_ode/round0_analysis_contracts.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_analysis.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_figures.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_package_verify.py",
    "project/run_scripts/ordered_response_barrier_ode/tests/test_round0_analysis.py",
)
REFERENCE_FILES: tuple[str, ...] = ()
REFERENCE_CORE_PERFORMANCE_FILE = ""
KIND_ORDER = tuple(KIND_COUNTS)
ARRAY_FIELDS = (
    "per_request_D_R0",
    "per_request_D_model_R0",
    "per_request_actual_potential_change",
    "per_request_g",
    "per_request_q_res",
    "per_request_r",
    "per_request_response_orthogonal_R0_squared",
    "per_request_writer_command_R0_squared",
    "per_request_zero_command_cross_response_R0_squared",
)
STEP_SCALARS = (
    "actual_reduction",
    "actual_transition_norm_mean",
    "applied_update_frobenius",
    "applied_update_frobenius_squared",
    "coefficient_u",
    "command_norm_mean",
    "direction_frobenius",
    "direction_frobenius_squared",
    "discretization_defect",
    "entry_sublevel_violation",
    "euler_alpha",
    "factor_rank",
    "g",
    "model_error_norm_mean",
    "potential_after",
    "potential_before",
    "predicted_reduction",
    "predicted_transition_norm_mean",
    "r",
    "realization_error_norm_mean",
    "residual_denominator",
    "residual_norm_mean_after",
    "residual_norm_mean_before",
    "resolution_stable_path_increment_frobenius_squared",
    "response_norm_mean",
    "writer_command_norm_mean",
)
LINEAGE = (
    ("35520", "round0-tech-r1", "c747b3ea67c44e20be3fa8ec8129f8a9503fe825", "TECHNICAL_ATTEMPT"),
    ("35567", "round0-tech-r2", "396496710c43e7b6875d219fd44364141f5cb300", "TECHNICAL_ATTEMPT"),
    ("35582", "round0-tech-r4", "346a592257be6dc89d378671026c4253c3938341", "TECHNICAL_ATTEMPT"),
    ("35615", "b1-tech-r1", RAW_SOURCE_HEAD, "B1_PILOT"),
    ("35694", "round0-b1-gated-tech-r1", RAW_SOURCE_HEAD, "CANONICAL_B100"),
)
INCLUDE_LEGACY_DRY_PLAN = True


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisBoundary(f"JSON object expected: {path}")
    return payload


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _write_csv_once(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> None:
    if frame.empty:
        raise AnalysisBoundary(f"refusing empty table: {path.name}")
    raw = _csv_bytes(frame)
    if compressed:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
            handle.write(raw)
        raw = buffer.getvalue()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def _stat(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise AnalysisBoundary("stat input is empty or nonfinite")
    return {
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "p90": float(np.quantile(array, 0.9, method="linear")),
        "max": float(array.max()),
    }


def _near(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 32.0 * np.finfo(np.float64).eps * scale


def _verify_document_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, expected_sha, expected_bytes, expected_lines in AUTHORITATIVE_DOCUMENTS:
        path = Path(name)
        row = member(path, kind="authoritative_document")
        lines = sum(1 for _ in path.open("rb"))
        if row["sha256"] != expected_sha or row["bytes"] != expected_bytes or lines != expected_lines:
            raise AnalysisBoundary(f"authoritative document identity differs: {path}")
        row["lines"] = lines
        rows.append(row)
    return rows


def _verify_reference_inputs(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in REFERENCE_FILES:
        rows.append(member(repo / relative, relative_to=repo, kind="immutable_prior_report_reference"))
    return rows


def _prior_metric_parity(repo: Path, current: pd.DataFrame) -> pd.DataFrame:
    if not REFERENCE_CORE_PERFORMANCE_FILE:
        return pd.DataFrame(
            [{
                "metric": "NOT_CONFIGURED",
                "paired_row_count": 0,
                "exact_equal_count": 0,
                "nonzero_delta_count": 0,
                "delta_mean": np.nan,
                "delta_median": np.nan,
                "delta_p90": np.nan,
                "max_absolute_delta": np.nan,
            }]
        )
    prior_path = repo / REFERENCE_CORE_PERFORMANCE_FILE
    regular_file(prior_path)
    prior = pd.read_csv(prior_path)
    keys = ["cell_id", "model", "writer_family", "stage"]
    if len(prior) != 24 or prior.duplicated(keys).any() or len(current) != 24 or current.duplicated(keys).any():
        raise AnalysisBoundary("prior/current performance key denominator differs")
    metrics = (
        "rewrite_success_rate",
        "rephrase_success_rate",
        "strict_rephrase_success_rate",
        "locality_prediction_preservation_rate",
        "rewrite_target_new_accuracy_rate",
        "rewrite_target_true_accuracy_rate",
        "rephrase_target_new_accuracy_rate",
        "rephrase_target_true_accuracy_rate",
        "locality_target_true_accuracy_rate",
    )
    merged = prior[keys + list(metrics)].merge(
        current[keys + list(metrics)], on=keys, how="outer", validate="one_to_one",
        suffixes=("_prior", "_rerun"), indicator=True,
    )
    if len(merged) != 24 or not (merged["_merge"] == "both").all():
        raise AnalysisBoundary("prior/current performance join differs")
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        left = merged[f"{metric}_prior"].astype(np.float64)
        right = merged[f"{metric}_rerun"].astype(np.float64)
        if left.isna().any() or right.isna().any() or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AnalysisBoundary(f"prior/current nonfinite metric: {metric}")
        delta = right - left
        rows.append(
            {
                "metric": metric,
                "paired_row_count": len(delta),
                "exact_equal_count": int((delta == 0.0).sum()),
                "nonzero_delta_count": int((delta != 0.0).sum()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(np.quantile(delta, 0.5, method="linear")),
                "delta_p90": float(np.quantile(delta, 0.9, method="linear")),
                "max_absolute_delta": float(np.abs(delta).max()),
            }
        )
    return pd.DataFrame(rows)


def _validate_evaluation(
    evaluation: Mapping[str, Any], request_order: Sequence[str], *, endpoint: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows = pd.DataFrame(evaluation.get("rows", []))
    required = {
        "request_sha256", "case_id", "kind", "prompt_index", "nll",
        "all_tokens_correct", "correct_token_count", "target_token_count",
        "input_identity_sha256", "observation_identity_sha256",
    }
    if len(rows) != EXPECTED_EVALUATION_ROWS or not required.issubset(rows.columns):
        raise AnalysisBoundary("evaluation row schema/denominator differs")
    if int(rows.duplicated(["request_sha256", "kind", "prompt_index"], keep=False).sum()):
        raise AnalysisBoundary("evaluation composite-key duplicate")
    if set(rows.request_sha256) != set(request_order):
        raise AnalysisBoundary("evaluation request identity differs")
    rows["nll"] = pd.to_numeric(rows.nll, errors="raise")
    if not np.isfinite(rows.nll.to_numpy(dtype=np.float64)).all():
        raise AnalysisBoundary("evaluation NLL nonfinite")
    for kind, count in KIND_COUNTS.items():
        selected = rows[rows.kind == kind]
        if len(selected) != count:
            raise AnalysisBoundary(f"evaluation kind denominator differs: {kind}")
    if set(rows.kind) != set(KIND_COUNTS):
        raise AnalysisBoundary("unexpected evaluation kind")
    summaries: list[dict[str, Any]] = []
    stored_summaries = evaluation.get("kind_summaries", {})
    for kind in KIND_ORDER:
        selected = rows[rows.kind == kind]
        stats = _stat(selected.nll)
        correct = int(selected.all_tokens_correct.astype(bool).sum())
        stored = stored_summaries.get(kind, {})
        for field, value in stats.items():
            if not _near(value, float(stored.get(f"nll_{field}", float("nan")))):
                raise AnalysisBoundary(f"stored NLL summary differs: {kind}/{field}")
        if correct != int(stored.get("all_tokens_correct_count", -1)):
            raise AnalysisBoundary(f"stored accuracy differs: {kind}")
        summaries.append(
            {
                "kind": kind,
                "row_count": len(selected),
                **{f"nll_{key}": value for key, value in stats.items()},
                "all_tokens_correct_count": correct,
                "all_tokens_correct_denominator": len(selected),
                "all_tokens_correct_rate": correct / len(selected),
            }
        )

    pairs: dict[str, pd.DataFrame] = {}
    preference: dict[str, dict[str, Any]] = {}
    for prefix, denominator in (("rewrite", 100), ("rephrase", 200)):
        new = rows[rows.kind == f"{prefix}_target_new"].copy()
        true = rows[rows.kind == f"{prefix}_target_true"].copy()
        pair = new.merge(
            true,
            on=["request_sha256", "case_id", "prompt_index"],
            suffixes=("_new", "_true"),
            validate="one_to_one",
        )
        if len(pair) != denominator:
            raise AnalysisBoundary(f"NLL pair denominator differs: {prefix}")
        pair["margin_true_minus_new"] = pair.nll_true - pair.nll_new
        pair["success"] = pair.nll_new < pair.nll_true
        prompt_count = int(pair.success.sum())
        strict = pair.groupby("request_sha256", sort=False).success.all()
        strict_count = int(strict.sum())
        stored = evaluation.get("preference", {}).get(prefix, {})
        if (
            prompt_count != int(stored.get("prompt_success_count", -1))
            or strict_count != int(stored.get("strict_request_success_count", -1))
            or stored.get("tie_is_failure") is not True
        ):
            raise AnalysisBoundary(f"stored preference differs: {prefix}")
        pairs[prefix] = pair
        preference[prefix] = {
            "prompt_success_count": prompt_count,
            "prompt_denominator": denominator,
            "prompt_success_rate": prompt_count / denominator,
            "strict_request_success_count": strict_count,
            "strict_request_denominator": 100,
            "strict_request_success_rate": strict_count / 100,
            **{f"margin_{key}": value for key, value in _stat(pair.margin_true_minus_new).items()},
        }

    request_rows: list[dict[str, Any]] = []
    locality = rows[rows.kind == "locality_target_true"]
    locality_new = rows[rows.kind == "locality_target_new"] if "locality_target_new" in set(rows.kind) else None
    for ordinal, request_sha in enumerate(request_order):
        rewrite = pairs["rewrite"][pairs["rewrite"].request_sha256 == request_sha]
        rephrase = pairs["rephrase"][pairs["rephrase"].request_sha256 == request_sha].sort_values("prompt_index")
        local = locality[locality.request_sha256 == request_sha].sort_values("prompt_index")
        if len(rewrite) != 1 or len(rephrase) != 2 or len(local) != 10:
            raise AnalysisBoundary("per-request prompt denominator differs")
        row: dict[str, Any] = {
            "request_ordinal": ordinal,
            "request_sha256": request_sha,
            "case_id": int(rewrite.case_id.iloc[0]),
            "rewrite_target_new_nll": float(rewrite.nll_new.iloc[0]),
            "rewrite_target_true_nll": float(rewrite.nll_true.iloc[0]),
            "rewrite_margin_true_minus_new": float(rewrite.margin_true_minus_new.iloc[0]),
            "rewrite_success": int(rewrite.success.iloc[0]),
            "rewrite_target_new_accuracy": int(bool(rewrite.all_tokens_correct_new.iloc[0])),
            "rewrite_target_true_accuracy": int(bool(rewrite.all_tokens_correct_true.iloc[0])),
            "rephrase_target_new_nll_mean": float(rephrase.nll_new.mean()),
            "rephrase_target_new_nll_max": float(rephrase.nll_new.max()),
            "rephrase_target_true_nll_mean": float(rephrase.nll_true.mean()),
            "rephrase_target_true_nll_max": float(rephrase.nll_true.max()),
            "rephrase_margin_true_minus_new_mean": float(rephrase.margin_true_minus_new.mean()),
            "rephrase_prompt_success_count": int(rephrase.success.sum()),
            "rephrase_prompt_denominator": 2,
            "rephrase_strict_success": int(rephrase.success.all()),
            "rephrase_target_new_accuracy_count": int(rephrase.all_tokens_correct_new.astype(bool).sum()),
            "rephrase_target_true_accuracy_count": int(rephrase.all_tokens_correct_true.astype(bool).sum()),
            "locality_target_true_nll_mean": float(local.nll.mean()),
            "locality_target_true_accuracy_count": int(local.all_tokens_correct.astype(bool).sum()),
            "locality_prompt_denominator": 10,
        }
        if locality_new is not None:
            local_new = locality_new[locality_new.request_sha256 == request_sha].sort_values("prompt_index")
            local_pair = local_new.merge(
                local,
                on=["request_sha256", "case_id", "prompt_index"],
                suffixes=("_new", "_true"),
                validate="one_to_one",
            )
            if len(local_pair) != 10:
                raise AnalysisBoundary("per-request canonical NS denominator differs")
            local_pair["canonical_ns_success"] = local_pair.nll_true < local_pair.nll_new
            row.update(
                {
                    "canonical_ns_numerator": int(local_pair.canonical_ns_success.sum()),
                    "canonical_ns_denominator": 10,
                    "canonical_ns_rate": float(local_pair.canonical_ns_success.mean()),
                    "canonical_ns_tie_count": int((local_pair.nll_true == local_pair.nll_new).sum()),
                    "locality_target_new_nll_mean": float(local_pair.nll_new.mean()),
                    "locality_target_true_minus_new_nll_mean": float(
                        (local_pair.nll_true - local_pair.nll_new).mean()
                    ),
                }
            )
        else:
            row.update(
                {
                    "canonical_ns_numerator": np.nan,
                    "canonical_ns_denominator": np.nan,
                    "canonical_ns_rate": np.nan,
                    "canonical_ns_tie_count": np.nan,
                    "locality_target_new_nll_mean": np.nan,
                    "locality_target_true_minus_new_nll_mean": np.nan,
                }
            )
        if endpoint:
            needed = {"prediction_preserved_count", "prediction_token_count", "all_predictions_preserved"}
            if not needed.issubset(local.columns):
                raise AnalysisBoundary("endpoint locality preservation fields missing")
            numerator = int(pd.to_numeric(local.prediction_preserved_count).sum())
            denominator = int(pd.to_numeric(local.prediction_token_count).sum())
            row.update(
                {
                    "locality_prediction_preservation_numerator": numerator,
                    "locality_prediction_preservation_denominator": denominator,
                    "locality_prediction_preservation_rate": numerator / denominator,
                    "locality_all_prompt_preserved_count": int(local.all_predictions_preserved.astype(bool).sum()),
                }
            )
        else:
            row.update(
                {
                    "locality_prediction_preservation_numerator": np.nan,
                    "locality_prediction_preservation_denominator": np.nan,
                    "locality_prediction_preservation_rate": np.nan,
                    "locality_all_prompt_preserved_count": np.nan,
                }
            )
        request_rows.append(row)
    request_frame = pd.DataFrame(request_rows)
    if endpoint:
        stored_locality = evaluation.get("locality", {})
        numerator = int(request_frame.locality_prediction_preservation_numerator.sum())
        denominator = int(request_frame.locality_prediction_preservation_denominator.sum())
        prompt_preserved = int(request_frame.locality_all_prompt_preserved_count.sum())
        if (
            numerator != int(stored_locality.get("prediction_preservation_numerator", -1))
            or denominator != int(stored_locality.get("prediction_preservation_denominator", -1))
            or prompt_preserved != int(stored_locality.get("all_prompt_predictions_preserved_count", -1))
        ):
            raise AnalysisBoundary("stored locality preservation differs")
    if locality_new is not None:
        stored_ns = evaluation.get("locality", {}).get("canonical_ns", {})
        ns_numerator = int(request_frame.canonical_ns_numerator.sum())
        ns_denominator = int(request_frame.canonical_ns_denominator.sum())
        ns_ties = int(request_frame.canonical_ns_tie_count.sum())
        if (
            stored_ns.get("predicate") != "target_true_nll < target_new_nll"
            or stored_ns.get("tie_is_failure") is not True
            or ns_numerator != int(stored_ns.get("prompt_success_count", -1))
            or ns_denominator != int(stored_ns.get("prompt_denominator", -1))
            or ns_ties != int(stored_ns.get("tie_count", -1))
        ):
            raise AnalysisBoundary("stored canonical NS differs")
    performance = {
        "rewrite_success_count": preference["rewrite"]["prompt_success_count"],
        "rewrite_success_denominator": 100,
        "rewrite_success_rate": preference["rewrite"]["prompt_success_rate"],
        "rephrase_success_count": preference["rephrase"]["prompt_success_count"],
        "rephrase_success_denominator": 200,
        "rephrase_success_rate": preference["rephrase"]["prompt_success_rate"],
        "strict_rephrase_success_count": preference["rephrase"]["strict_request_success_count"],
        "strict_rephrase_success_denominator": 100,
        "strict_rephrase_success_rate": preference["rephrase"]["strict_request_success_rate"],
        "locality_prediction_preservation_numerator": (
            int(request_frame.locality_prediction_preservation_numerator.sum()) if endpoint else np.nan
        ),
        "locality_prediction_preservation_denominator": (
            int(request_frame.locality_prediction_preservation_denominator.sum()) if endpoint else np.nan
        ),
        "locality_prediction_preservation_rate": (
            float(request_frame.locality_prediction_preservation_numerator.sum()
                  / request_frame.locality_prediction_preservation_denominator.sum()) if endpoint else np.nan
        ),
        "canonical_ns_numerator": (
            int(request_frame.canonical_ns_numerator.sum()) if locality_new is not None else np.nan
        ),
        "canonical_ns_denominator": (
            int(request_frame.canonical_ns_denominator.sum()) if locality_new is not None else np.nan
        ),
        "canonical_ns_rate": (
            float(request_frame.canonical_ns_numerator.sum() / request_frame.canonical_ns_denominator.sum())
            if locality_new is not None
            else np.nan
        ),
        "canonical_ns_tie_count": (
            int(request_frame.canonical_ns_tie_count.sum()) if locality_new is not None else np.nan
        ),
    }
    for summary in summaries:
        kind = summary["kind"]
        performance[f"{kind}_accuracy_count"] = summary["all_tokens_correct_count"]
        performance[f"{kind}_accuracy_denominator"] = summary["all_tokens_correct_denominator"]
        performance[f"{kind}_accuracy_rate"] = summary["all_tokens_correct_rate"]
    return rows, request_frame, performance, summaries


def _load_cells(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_request_order: list[str] | None = None
    reference_case_ids: list[int] | None = None
    model_entry_identity: dict[str, str] = {}
    gates = {
        "canonical_cell_count": 0,
        "canonical_request_count": 0,
        "primary_endpoint_count": 0,
        "evaluation_prompt_row_count": 0,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
        "rerun_count_for_analysis": 0,
        "dynamic_z_recompute_count": 0,
        "forbidden_controller_evaluator_access_count": 0,
        "retry_count": 0,
        "nonfinite_count": 0,
        "entry_already_hit_count": 0,
    }
    for cell_id, (expected_model, expected_family, child_job_id) in CELL_BINDINGS.items():
        task_root = raw_root / f"task-{cell_id}"
        if task_root.is_symlink() or not task_root.is_dir():
            raise AnalysisBoundary(f"invalid task root: {task_root}")
        result_path = task_root / "result.json"
        round_path = task_root / "round-00-result.json"
        receipt_path = task_root / "terminal-receipt.json"
        for path, kind in ((result_path, "canonical_result"), (round_path, "canonical_round"), (receipt_path, "terminal_receipt")):
            row = member(path, kind=kind)
            row.update({"cell_id": cell_id, "child_job_id": child_job_id})
            inputs.append(row)
        result, round_result, receipt = map(_load_json, (result_path, round_path, receipt_path))
        verify_canonical_identity(round_result)
        verify_canonical_identity(receipt, field="receipt_identity_sha256")
        publication = result.get("round_publication_receipts", [])
        if len(publication) != 1 or publication[0].get("file_sha256") != sha256_file(round_path):
            raise AnalysisBoundary(f"round publication SHA differs: cell {cell_id}")
        if receipt.get("result_sha256") != sha256_file(result_path):
            raise AnalysisBoundary(f"terminal result SHA differs: cell {cell_id}")
        expected = {
            "cell_id": cell_id,
            "model_alias": expected_model,
            "writer_family": expected_family,
            "status": "TERMINAL_VALID",
            "source_head": RAW_SOURCE_HEAD,
            "source_tree": RAW_SOURCE_TREE,
            "stream_root": STREAM_ROOT,
            "order_root": ORDER_ROOT,
            "request_count": 100,
            "primary_endpoint_count": 500,
            "fixed_z_compute_count": 100,
            "fixed_z_recompute_count": 0,
            "technical_failure_count": 0,
            "scientific_failure_count": 0,
            "imputation_count": 0,
            "entry_already_hit_count": 0,
            "full_fp32": True,
            "transaction_pass": True,
            "w0_restore_pass": True,
            "cache_restore_pass": True,
            "jvp_gate_pass": True,
            "overlay_gate_pass": True,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise AnalysisBoundary(f"terminal receipt differs: cell={cell_id} field={key}")
        if result.get("primary_arms") != list(PRIMARY_ARMS) or round_result.get("primary_arm_order") != list(PRIMARY_ARMS):
            raise AnalysisBoundary(f"arm order differs: cell {cell_id}")
        request_order = list(round_result.get("request_sha256", []))
        case_ids = [int(value) for value in round_result.get("case_ids", [])]
        if len(request_order) != 100 or len(set(request_order)) != 100 or len(case_ids) != 100:
            raise AnalysisBoundary(f"B100 request denominator differs: cell {cell_id}")
        if canonical_hash(request_order) != round_result.get("request_order_sha256"):
            raise AnalysisBoundary(f"B100 request order hash differs: cell {cell_id}")
        if reference_request_order is None:
            reference_request_order, reference_case_ids = request_order, case_ids
        elif request_order != reference_request_order or case_ids != reference_case_ids:
            raise AnalysisBoundary(f"cross-cell request/case order differs: cell {cell_id}")
        entry_identity = str(round_result["entry_evaluation"]["identity_sha256"])
        previous = model_entry_identity.setdefault(expected_model, entry_identity)
        if previous != entry_identity:
            raise AnalysisBoundary(f"cross-writer PRE_EDIT identity differs: {expected_model}")
        if round_result.get("fixed_z_recompute_count") != 0 or result.get("runtime_lock", {}).get("dynamic_z_recompute_count") != 0:
            raise AnalysisBoundary(f"dynamic z recompute detected: cell {cell_id}")
        if result.get("runtime_lock", {}).get("retry_count") != 0:
            raise AnalysisBoundary(f"retry detected: cell {cell_id}")
        if result.get("parameter_inventory", {}).get("quantized") is not False:
            raise AnalysisBoundary(f"quantized model detected: cell {cell_id}")
        if set(result.get("parameter_inventory", {}).get("parameter_tensors_by_dtype", {})) != {"torch.float32"}:
            raise AnalysisBoundary(f"parameter dtype differs: cell {cell_id}")
        if result.get("runtime_lock", {}).get("controller_heldout_influence_count") != 0:
            raise AnalysisBoundary(f"controller heldout influence detected: cell {cell_id}")
        cells.append(
            {
                "cell_id": cell_id,
                "child_job_id": child_job_id,
                "model": expected_model,
                "writer_family": expected_family,
                "task_root": task_root,
                "result": result,
                "round": round_result,
                "receipt": receipt,
                "request_order": request_order,
                "case_ids": case_ids,
            }
        )
        gates["canonical_cell_count"] += 1
        gates["canonical_request_count"] += 100
        gates["primary_endpoint_count"] += 500
        gates["entry_already_hit_count"] += int(receipt["entry_already_hit_count"])
    if gates["canonical_cell_count"] != 4 or gates["canonical_request_count"] != 400:
        raise AnalysisBoundary("canonical cell/request denominator differs")
    if gates["primary_endpoint_count"] != EXPECTED_PRIMARY_ENDPOINTS:
        raise AnalysisBoundary("primary endpoint denominator differs")
    gates["same_request_order_cross_cell"] = True
    gates["same_pre_edit_within_model"] = True
    gates["source_head_tree_pass"] = True
    gates["stream_order_root_pass"] = True
    gates["full_fp32_scope"] = "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH"
    gates["official_memit_ephemeral_fp64_exception_cell_count"] = 2
    gates["unqualified_all_algorithm_full_fp32_cell_count"] = 2
    return cells, inputs, gates


def _endpoint_tables(cells: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, str], pd.DataFrame], dict[str, Any]]:
    performance_rows: list[dict[str, Any]] = []
    nll_rows: list[dict[str, Any]] = []
    prompt_frames: list[pd.DataFrame] = []
    request_frames: list[pd.DataFrame] = []
    indexed_prompts: dict[tuple[int, str], pd.DataFrame] = {}
    gates = {"evaluation_summary_recompute_failure_count": 0, "evaluation_entry_binding_failure_count": 0}
    for cell in cells:
        round_result = cell["round"]
        stages: list[tuple[str, Mapping[str, Any], str]] = [
            ("PRE_EDIT", round_result["entry_evaluation"], "ENTRY_REFERENCE")
        ]
        endpoints = list(round_result["primary_endpoints"])
        if [row.get("arm") for row in endpoints] != list(PRIMARY_ARMS):
            raise AnalysisBoundary("primary endpoint arm order differs")
        stages.extend((str(row["arm"]), row["evaluation"], str(row["status"])) for row in endpoints)
        entry_by_key: pd.DataFrame | None = None
        for stage, evaluation, status in stages:
            endpoint = stage != "PRE_EDIT"
            prompts, requests, performance, summaries = _validate_evaluation(
                evaluation, cell["request_order"], endpoint=endpoint
            )
            verify_canonical_identity(evaluation)
            tagged_prompt = prompts.copy()
            tagged_prompt.insert(0, "stage", stage)
            tagged_prompt.insert(0, "writer_family", cell["writer_family"])
            tagged_prompt.insert(0, "model", cell["model"])
            tagged_prompt.insert(0, "cell_id", cell["cell_id"])
            prompt_frames.append(tagged_prompt)
            indexed_prompts[(cell["cell_id"], stage)] = tagged_prompt
            tagged_request = requests.copy()
            tagged_request.insert(0, "stage", stage)
            tagged_request.insert(0, "writer_family", cell["writer_family"])
            tagged_request.insert(0, "model", cell["model"])
            tagged_request.insert(0, "cell_id", cell["cell_id"])
            request_frames.append(tagged_request)
            performance_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "model": cell["model"],
                    "writer_family": cell["writer_family"],
                    "stage": stage,
                    "stage_status": status,
                    **performance,
                }
            )
            for summary in summaries:
                nll_rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "model": cell["model"],
                        "writer_family": cell["writer_family"],
                        "stage": stage,
                        **summary,
                    }
                )
            key_columns = ["request_sha256", "kind", "prompt_index", "input_identity_sha256"]
            current = prompts.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
            if not endpoint:
                entry_by_key = current
            else:
                assert entry_by_key is not None
                if not current[key_columns].equals(entry_by_key[key_columns]):
                    raise AnalysisBoundary(f"endpoint evaluator input differs: cell={cell['cell_id']} arm={stage}")
                for field in ("entry_nll", "entry_observation_identity_sha256"):
                    if field not in current.columns:
                        raise AnalysisBoundary(f"endpoint entry binding field missing: {field}")
                if not np.array_equal(
                    current.entry_nll.to_numpy(dtype=np.float64), entry_by_key.nll.to_numpy(dtype=np.float64)
                ):
                    raise AnalysisBoundary(f"endpoint entry NLL differs: cell={cell['cell_id']} arm={stage}")
                if current.entry_observation_identity_sha256.tolist() != entry_by_key.observation_identity_sha256.tolist():
                    raise AnalysisBoundary(f"endpoint entry observation differs: cell={cell['cell_id']} arm={stage}")
    prompt = pd.concat(prompt_frames, ignore_index=True)
    request = pd.concat(request_frames, ignore_index=True)
    performance = pd.DataFrame(performance_rows)
    nll = pd.DataFrame(nll_rows)
    expected_prompt_rows = 4 * 6 * EXPECTED_EVALUATION_ROWS
    if (
        len(prompt) != expected_prompt_rows
        or len(request) != 4 * 6 * 100
        or len(performance) != 24
        or len(nll) != 4 * 6 * len(KIND_COUNTS)
    ):
        raise AnalysisBoundary("derived endpoint table denominator differs")
    gates.update(
        {
            "evaluation_prompt_row_count": len(prompt),
            "evaluation_request_row_count": len(request),
            "performance_summary_row_count": len(performance),
            "nll_summary_row_count": len(nll),
        }
    )
    return performance, nll, prompt, request, indexed_prompts, gates


def _paired_delta_table(
    indexed_prompts: Mapping[tuple[int, str], pd.DataFrame], request: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_BINDINGS:
        official = indexed_prompts[(cell_id, "O")]
        for arm in DYNAMIC_ARMS:
            ours = indexed_prompts[(cell_id, arm)]
            for kind in KIND_ORDER[:4]:
                keys = ["request_sha256", "case_id", "kind", "prompt_index", "input_identity_sha256"]
                left = official[official.kind == kind][keys + ["nll"]]
                right = ours[ours.kind == kind][keys + ["nll"]]
                paired = left.merge(right, on=keys, suffixes=("_official", "_ours"), validate="one_to_one")
                delta = paired.nll_ours.to_numpy(dtype=np.float64) - paired.nll_official.to_numpy(dtype=np.float64)
                stats = _stat(delta)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "model": CELL_BINDINGS[cell_id][0],
                        "writer_family": CELL_BINDINGS[cell_id][1],
                        "arm": arm,
                        "metric": f"{kind}_nll",
                        "comparison": "OURS_MINUS_OFFICIAL",
                        "descriptive_direction": "LOWER_NLL_IS_LOWER",
                        "paired_denominator": len(delta),
                        **{f"delta_{key}": value for key, value in stats.items()},
                        "better_count": int((delta < 0).sum()),
                        "exact_equal_count": int((delta == 0).sum()),
                        "worse_count": int((delta > 0).sum()),
                    }
                )
            for prefix, kind_count in (("rewrite", 100), ("rephrase", 200)):
                keys = ["request_sha256", "case_id", "prompt_index"]
                def margin(frame: pd.DataFrame) -> pd.DataFrame:
                    new = frame[frame.kind == f"{prefix}_target_new"][keys + ["nll"]]
                    true = frame[frame.kind == f"{prefix}_target_true"][keys + ["nll"]]
                    out = new.merge(true, on=keys, suffixes=("_new", "_true"), validate="one_to_one")
                    out["value"] = out.nll_true - out.nll_new
                    return out[keys + ["value"]]
                left = margin(official)
                right = margin(ours)
                paired = left.merge(right, on=keys, suffixes=("_official", "_ours"), validate="one_to_one")
                delta = paired.value_ours.to_numpy(dtype=np.float64) - paired.value_official.to_numpy(dtype=np.float64)
                stats = _stat(delta)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "model": CELL_BINDINGS[cell_id][0],
                        "writer_family": CELL_BINDINGS[cell_id][1],
                        "arm": arm,
                        "metric": f"{prefix}_margin_true_minus_new",
                        "comparison": "OURS_MINUS_OFFICIAL",
                        "descriptive_direction": "HIGHER_MARGIN_FAVORS_TARGET_NEW",
                        "paired_denominator": kind_count,
                        **{f"delta_{key}": value for key, value in stats.items()},
                        "better_count": int((delta > 0).sum()),
                        "exact_equal_count": int((delta == 0).sum()),
                        "worse_count": int((delta < 0).sum()),
                    }
                )
            o_req = request[(request.cell_id == cell_id) & (request.stage == "O")]
            a_req = request[(request.cell_id == cell_id) & (request.stage == arm)]
            paired = o_req[["request_sha256", "locality_prediction_preservation_rate"]].merge(
                a_req[["request_sha256", "locality_prediction_preservation_rate"]],
                on="request_sha256", suffixes=("_official", "_ours"), validate="one_to_one",
            )
            delta = paired.locality_prediction_preservation_rate_ours.to_numpy(dtype=np.float64) - paired.locality_prediction_preservation_rate_official.to_numpy(dtype=np.float64)
            stats = _stat(delta)
            rows.append(
                {
                    "cell_id": cell_id,
                    "model": CELL_BINDINGS[cell_id][0],
                    "writer_family": CELL_BINDINGS[cell_id][1],
                    "arm": arm,
                    "metric": "locality_prediction_preservation_rate",
                    "comparison": "OURS_MINUS_OFFICIAL",
                    "descriptive_direction": "HIGHER_PRESERVATION_IS_HIGHER",
                    "paired_denominator": len(delta),
                    **{f"delta_{key}": value for key, value in stats.items()},
                    "better_count": int((delta > 0).sum()),
                    "exact_equal_count": int((delta == 0).sum()),
                    "worse_count": int((delta < 0).sum()),
                }
            )
            if "canonical_ns_rate" in request.columns and not a_req.canonical_ns_rate.isna().all():
                paired_ns = o_req[["request_sha256", "canonical_ns_rate"]].merge(
                    a_req[["request_sha256", "canonical_ns_rate"]],
                    on="request_sha256",
                    suffixes=("_official", "_ours"),
                    validate="one_to_one",
                )
                delta_ns = (
                    paired_ns.canonical_ns_rate_ours.to_numpy(dtype=np.float64)
                    - paired_ns.canonical_ns_rate_official.to_numpy(dtype=np.float64)
                )
                stats_ns = _stat(delta_ns)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "model": CELL_BINDINGS[cell_id][0],
                        "writer_family": CELL_BINDINGS[cell_id][1],
                        "arm": arm,
                        "metric": "canonical_ns_rate",
                        "comparison": "OURS_MINUS_OFFICIAL",
                        "descriptive_direction": "HIGHER_CANONICAL_NS_IS_HIGHER",
                        "paired_denominator": len(delta_ns),
                        **{f"delta_{key}": value for key, value in stats_ns.items()},
                        "better_count": int((delta_ns > 0).sum()),
                        "exact_equal_count": int((delta_ns == 0).sum()),
                        "worse_count": int((delta_ns < 0).sum()),
                    }
                )
    result = pd.DataFrame(rows)
    metrics_per_comparison = 8 if "canonical_ns_rate" in set(result.metric) else 7
    if len(result) != 4 * 4 * metrics_per_comparison:
        raise AnalysisBoundary("paired delta row denominator differs")
    return result


def _numeric_array_summary(values: Sequence[Any]) -> dict[str, Any]:
    recorded = [float(value) for value in values if value is not None]
    if not recorded:
        return {"recorded_count": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    stats = _stat(recorded)
    return {"recorded_count": len(recorded), **stats}


def _mechanism_tables(cells: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    arm_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    request_step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    preamble_rows: list[dict[str, Any]] = []
    for cell in cells:
        result = cell["result"]
        round_result = cell["round"]
        preamble = round_result["runtime_preamble"]
        verify_canonical_identity(preamble)
        facts = preamble["facts"]
        fd = facts["P1_jvp"]["finite_difference"]
        preamble_rows.append(
            {
                "cell_id": cell["cell_id"],
                "model": cell["model"],
                "writer_family": cell["writer_family"],
                "status": facts["status"],
                "request_case_id": facts["request_case_id"],
                "request_sha256": facts["request_sha256"],
                "official_wrapper_direct_exact_fidelity": facts["P0_official_scaling"]["stock_official_parity"]["exact_parity"],
                "residual_scaling_max_abs_error": facts["P0_official_scaling"]["residual_scaling_max_abs_error"],
                "jvp_fd_epsilon_count": len(fd),
                "jvp_fd_allclose_count": sum(bool(row["allclose"]) for row in fd),
                "jvp_fd_min_cosine": min(float(row["cosine_similarity"]) for row in fd),
                "jvp_fd_max_abs_error": max(float(row["maximum_absolute_error"]) for row in fd),
                "jvp_fd_max_relative_l2_error": max(float(row["relative_l2_error"]) for row in fd),
                "state_changed_key_observed": facts["P2_state_transaction"]["l5_keys_changed_observed"],
                "inner_persistent_mutation_count": facts["P2_state_transaction"]["inner_persistent_mutation_count"],
                "algorithmic_retry_count": facts["P2_state_transaction"]["algorithmic_retry_count"],
                "w0_pointer_bytes_restore_pass": facts["w0_pointer_bytes_restore_pass"],
                "orbhit_direct_prefix_identity_status": facts["P3_orbhit_direct_prefix_identity"]["status"],
                "outcome_comparison_blocking_gate_count": facts["outcome_comparison_gate_policy"]["blocking_outcome_comparison_count"],
                "scientific_selection_influence_count": facts["scientific_selection_influence_count"],
            }
        )
        for endpoint in round_result["primary_endpoints"]:
            verify_canonical_identity(endpoint)
            arm = str(endpoint["arm"])
            telemetry = endpoint["mechanism_telemetry"]
            verify_canonical_identity(telemetry)
            mechanism = telemetry["telemetry"]
            facts_endpoint = endpoint["endpoint_facts"]["facts"]
            accounting = endpoint["accounting_reconciliation"]["facts"]
            terminal_sq = float(facts_endpoint["terminal_net_frobenius_squared"])
            arm_row: dict[str, Any] = {
                "cell_id": cell["cell_id"],
                "model": cell["model"],
                "writer_family": cell["writer_family"],
                "arm": arm,
                "status": endpoint["status"],
                "endpoint_status": endpoint["endpoint_status"],
                "terminal_semantic_status": mechanism.get("terminal_status", "OFFICIAL_BYPASS"),
                "global_first_hit": "NONE" if mechanism.get("first_hit") is None else json.dumps(mechanism["first_hit"], sort_keys=True),
                "entry_already_hit": endpoint["status"] == "ENTRY_ALREADY_HIT",
                "anchor_active_request_count": mechanism.get("anchor_active_request_count", np.nan),
                "anchor_zero_request_count": mechanism.get("anchor_zero_request_count", np.nan),
                "step_count": mechanism.get("step_count", 0),
                "zero_action_visit_count": mechanism.get("zero_action_visit_count", 0),
                "nonzero_action_visit_count": mechanism.get("nonzero_action_visit_count", 0),
                "terminal_request_strict_count": mechanism.get("terminal_semantic", {}).get("request_strict_count", mechanism.get("terminal_semantic", {}).get("request_strict_count", np.nan)),
                "terminal_strict_event_count": mechanism.get("terminal_semantic", {}).get("strict_event_count", np.nan),
                "terminal_semantic_event_count": mechanism.get("terminal_semantic", {}).get("event_count", np.nan),
                "terminal_net_frobenius": float(facts_endpoint["terminal_net_frobenius"]),
                "terminal_net_frobenius_squared": terminal_sq,
                "physical_euler_path_frobenius_length": mechanism.get("physical_euler_path_frobenius_length", np.nan),
                "resolution_stable_path_frobenius": mechanism.get("resolution_stable_path_frobenius", np.nan),
                "resolution_stable_path_frobenius_squared": mechanism.get("resolution_stable_path_frobenius_squared", np.nan),
                "sum_step_action_frobenius_squared": mechanism.get("sum_step_action_frobenius_squared", terminal_sq),
                "history_append_count": int(endpoint["history_append_count"]),
                "inner_history_append_count": int(endpoint["inner_history_append_count"]),
                "inner_cache_mutation_count": int(endpoint["inner_cache_mutation_count"]),
                "physical_write_count": int(endpoint["physical_write_count"]),
                "selected_weight_endpoint_sha256": endpoint["selected_weight_endpoint_sha256"],
                "wall_seconds": float(endpoint["wall_seconds"]),
                "model_forward_invocation_count": int(endpoint["model_forward_invocation_count"]),
                "jvp_call_count": int(endpoint["jvp_counts"]["jvp_call_count"]),
                "jvp_wall_seconds": float(endpoint["jvp_counts"]["wall_seconds"]),
                "key_capture_count": int(endpoint["adapter_counts"]["key_capture_count"]),
                "factor_build_count": int(mechanism.get("factor_build_count", endpoint["adapter_counts"]["layer_factorization_count"])),
                "solve_count": int(endpoint["adapter_counts"]["solve_count"]),
                "terminal_capture_count": int(endpoint["adapter_counts"]["terminal_capture_count"]),
                "endpoint_evaluator_count": int(accounting["endpoint_evaluator_count"]),
                "derived_endpoint_evaluator_count": int(accounting["derived_endpoint_evaluator_count"]),
                "shadow_materialization_count": accounting["shadow_materialization_count"],
                "method_state_kind": accounting["method_state_kind"],
                "method_state_content_identity_equal": accounting["method_state_content_identity_equal"],
                "factor_condition": "NOT_RECORDED_SCHEMA_GAP",
            }
            arm_rows.append(arm_row)
            weights = facts_endpoint["terminal_net_frobenius_squared_by_weight"]
            step_by_layer: dict[int, list[Mapping[str, Any]]] = {layer: [] for layer in LAYERS}
            for step in mechanism.get("steps", []):
                layer = int(step["layer"])
                if layer not in step_by_layer:
                    raise AnalysisBoundary("unexpected dynamic layer")
                step_by_layer[layer].append(step)
            for weight_name, value in sorted(weights.items()):
                match = re.search(r"layers\.(\d+)\.", weight_name)
                if not match:
                    raise AnalysisBoundary(f"cannot parse layer from weight: {weight_name}")
                layer = int(match.group(1))
                selected_steps = step_by_layer[layer]
                net_sq = float(value)
                layer_rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "model": cell["model"],
                        "writer_family": cell["writer_family"],
                        "arm": arm,
                        "layer": layer,
                        "weight_name": weight_name,
                        "terminal_net_frobenius_squared": net_sq,
                        "terminal_net_energy_share": net_sq / terminal_sq if terminal_sq else np.nan,
                        "sum_applied_update_frobenius_squared": (
                            sum(float(step["applied_update_frobenius_squared"]) for step in selected_steps)
                            if selected_steps else net_sq
                        ),
                        "sum_resolution_stable_path_increment_frobenius_squared": (
                            sum(float(step["resolution_stable_path_increment_frobenius_squared"]) for step in selected_steps)
                            if selected_steps else net_sq
                        ),
                        "visit_count": len(selected_steps) if selected_steps else (1 if arm == "O" else 0),
                        "factor_rank_mean": (
                            float(np.mean([float(step["factor_rank"]) for step in selected_steps]))
                            if selected_steps else np.nan
                        ),
                        "factor_rank_min": (
                            float(np.min([float(step["factor_rank"]) for step in selected_steps]))
                            if selected_steps else np.nan
                        ),
                    }
                )
            if arm in DYNAMIC_ARMS:
                steps = list(mechanism.get("steps", []))
                if len(steps) != 20:
                    raise AnalysisBoundary(f"dynamic step denominator differs: cell={cell['cell_id']} arm={arm}")
                if [(int(row["sweep"]), int(row["layer"])) for row in steps] != [
                    (sweep, layer) for sweep in SWEEPS for layer in LAYERS
                ]:
                    raise AnalysisBoundary(f"dynamic sweep/layer order differs: cell={cell['cell_id']} arm={arm}")
                for step in steps:
                    summary: dict[str, Any] = {
                        "cell_id": cell["cell_id"], "model": cell["model"],
                        "writer_family": cell["writer_family"], "arm": arm,
                        "sweep": int(step["sweep"]), "layer": int(step["layer"]),
                        "visit_ordinal": int(step["visit_ordinal"]),
                        "action_geometry_status": step["action_geometry_status"],
                        "native_creg_action_status": step["native_creg_action_status"],
                        "built_state_version": int(step["built_state_version"]),
                        "command_reference_state_version": int(step["command_reference_state_version"]),
                        "resulting_state_version": int(step["resulting_state_version"]),
                        "semantic_all_strict": bool(step["semantic_all_strict"]),
                        "semantic_request_strict_count": int(step["semantic_request_strict_count"]),
                        "semantic_request_count": int(step["semantic_request_count"]),
                        "semantic_strict_event_count": int(step["semantic_strict_event_count"]),
                        "semantic_event_count": int(step["semantic_event_count"]),
                        "semantic_tie_count": int(step["semantic_tie_count"]),
                        "build_identity": step["build_identity"],
                        "keys_sha256": step["keys_sha256"],
                        "residual_sha256": step["residual_sha256"],
                        "solver_identity": step["solver_identity"],
                        "virtual_state_identity_sha256": step["virtual_state_identity_sha256"],
                    }
                    for field in STEP_SCALARS:
                        summary[field] = step[field]
                    for field in ARRAY_FIELDS:
                        values = list(step[field])
                        if len(values) != 100:
                            raise AnalysisBoundary(f"per-request mechanism denominator differs: {field}")
                        stats = _numeric_array_summary(values)
                        for key, value in stats.items():
                            summary[f"{field}_{key}"] = value
                    worsened = list(step["per_request_actual_potential_worsened"])
                    statuses = list(step["per_request_metric_status"])
                    if len(worsened) != 100 or len(statuses) != 100:
                        raise AnalysisBoundary("per-request mechanism status denominator differs")
                    summary["actual_potential_worsened_count"] = sum(bool(value) for value in worsened)
                    summary["metric_defined_count"] = sum(str(value).startswith("DEFINED") for value in statuses)
                    summary["metric_skipped_zero_command_count"] = sum(value == "SKIPPED_ZERO_COMMAND" for value in statuses)
                    step_rows.append(summary)
                    for index, request_sha in enumerate(cell["request_order"]):
                        request_row: dict[str, Any] = {
                            "cell_id": cell["cell_id"], "model": cell["model"],
                            "writer_family": cell["writer_family"], "arm": arm,
                            "sweep": int(step["sweep"]), "layer": int(step["layer"]),
                            "visit_ordinal": int(step["visit_ordinal"]),
                            "request_ordinal": index, "request_sha256": request_sha,
                            "case_id": cell["case_ids"][index],
                            "actual_potential_worsened": int(bool(worsened[index])),
                            "metric_status": statuses[index],
                        }
                        for field in ARRAY_FIELDS:
                            request_row[field] = step[field][index]
                        request_step_rows.append(request_row)
        for endpoint in round_result["primary_endpoints"]:
            accounting = endpoint["accounting_reconciliation"]["facts"]
            solve = accounting["solve"]
            compute_rows.append(
                {
                    "cell_id": cell["cell_id"], "model": cell["model"],
                    "writer_family": cell["writer_family"], "arm": endpoint["arm"],
                    "wall_seconds": endpoint["wall_seconds"],
                    "model_forward_invocation_count": endpoint["model_forward_invocation_count"],
                    "key_capture_count": endpoint["adapter_counts"]["key_capture_count"],
                    "factor_build_count": endpoint["mechanism_telemetry"]["telemetry"].get("factor_build_count", 0),
                    "solve_count_recorded": endpoint["adapter_counts"]["solve_count"],
                    "stock_expected_layer_solve_count": solve.get("expected_layer_solve_count", np.nan),
                    "stock_actual_solve_count": solve.get("actual_torch_solve_invocation_count", np.nan),
                    "jvp_call_count": endpoint["jvp_counts"]["jvp_call_count"],
                    "jvp_model_forward_invocation_count": endpoint["jvp_counts"]["model_forward_invocation_count"],
                    "jvp_wall_seconds": endpoint["jvp_counts"]["wall_seconds"],
                    "terminal_capture_count": endpoint["adapter_counts"]["terminal_capture_count"],
                    "physical_write_count": endpoint["physical_write_count"],
                    "shadow_materialization_count": accounting["shadow_materialization_count"],
                    "endpoint_evaluator_count": accounting["endpoint_evaluator_count"],
                    "derived_endpoint_evaluator_count": accounting["derived_endpoint_evaluator_count"],
                    "peak_gpu_allocated_bytes_cell_scope": result["memory"]["peak_allocated_bytes"],
                    "peak_gpu_reserved_bytes_cell_scope": result["memory"]["peak_reserved_bytes"],
                    "model_load_seconds_cell_scope": result["timing"]["model_load_seconds"],
                    "job_total_seconds_cell_scope": result["timing"]["job_total_seconds"],
                }
            )
    arm = pd.DataFrame(arm_rows)
    step = pd.DataFrame(step_rows)
    request_step = pd.DataFrame(request_step_rows)
    layer = pd.DataFrame(layer_rows)
    compute = pd.DataFrame(compute_rows)
    preamble = pd.DataFrame(preamble_rows)
    if len(arm) != 20 or len(step) != EXPECTED_DYNAMIC_STEPS or len(request_step) != EXPECTED_REQUEST_STEPS or len(layer) != 100 or len(compute) != 20 or len(preamble) != 4:
        raise AnalysisBoundary("mechanism/compute derived denominator differs")
    if int(step.semantic_all_strict.sum()) != 0:
        raise AnalysisBoundary("unexpected recorded global first hit")
    gates = {
        "mechanism_arm_row_count": len(arm),
        "mechanism_step_row_count": len(step),
        "mechanism_request_step_row_count": len(request_step),
        "layer_update_row_count": len(layer),
        "compute_row_count": len(compute),
        "runtime_preamble_row_count": len(preamble),
        "global_first_hit_count": 0,
        "horizon_semantic_miss_dynamic_endpoint_count": int((arm[arm.arm != "O"].status == "HORIZON_SEMANTIC_MISS").sum()),
        "entry_already_hit_count": int(arm.entry_already_hit.sum()),
        "method_state_content_mismatch_count": int((~arm.method_state_content_identity_equal.astype(bool)).sum()),
        "history_append_count": int(arm.history_append_count.sum()),
        "inner_history_append_count": int(arm.inner_history_append_count.sum()),
        "inner_cache_mutation_count": int(arm.inner_cache_mutation_count.sum()),
        "runtime_preamble_failure_count": int((preamble.status != "FAST_RUNTIME_PREAMBLE_PASS").sum()),
        "official_wrapper_direct_fidelity_failure_count": int((~preamble.official_wrapper_direct_exact_fidelity.astype(bool)).sum()),
    }
    return arm, step, request_step, layer, compute, preamble, gates


def _derived_orbhit_table(cells: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        payload = cell["round"]["derived_orbhit"]
        verify_canonical_identity(payload)
        _, _, performance, _ = _validate_evaluation(payload["evaluation"], cell["request_order"], endpoint=True)
        rows.append(
            {
                "cell_id": cell["cell_id"], "model": cell["model"],
                "writer_family": cell["writer_family"], "derived_arm": "ORBHit",
                "status": payload["status"], "withheld_technical": payload["withheld_technical"],
                "factor_count": payload["factor_count"],
                "source_derived_identity_sha256": payload["source_derived_identity_sha256"],
                "selected_weight_endpoint_sha256": payload["selected_weight_endpoint_sha256"],
                **performance,
            }
        )
    return pd.DataFrame(rows)


def _method_contrast_table(request: pd.DataFrame) -> pd.DataFrame:
    comparisons = (("QCL", "NQFIX"), ("NQFIX", "ORBFH"), ("ORBFH", "JAC"))
    metrics = [
        ("rewrite_target_new_nll", "LOWER_IS_LOWER"),
        ("rewrite_target_true_nll", "LOWER_IS_LOWER"),
        ("rewrite_margin_true_minus_new", "HIGHER_FAVORS_TARGET_NEW"),
        ("rephrase_target_new_nll_mean", "LOWER_IS_LOWER"),
        ("rephrase_target_new_nll_max", "LOWER_IS_LOWER"),
        ("rephrase_margin_true_minus_new_mean", "HIGHER_FAVORS_TARGET_NEW"),
        ("locality_prediction_preservation_rate", "HIGHER_IS_HIGHER"),
    ]
    if "canonical_ns_rate" in request.columns and not request.canonical_ns_rate.isna().all():
        metrics.append(("canonical_ns_rate", "HIGHER_IS_HIGHER"))
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_BINDINGS:
        selected = request[request.cell_id == cell_id]
        for left_arm, right_arm in comparisons:
            left = selected[selected.stage == left_arm]
            right = selected[selected.stage == right_arm]
            for metric, direction in metrics:
                pair = left[["request_sha256", metric]].merge(
                    right[["request_sha256", metric]], on="request_sha256",
                    suffixes=("_left", "_right"), validate="one_to_one",
                )
                delta = pair[f"{metric}_right"].to_numpy(dtype=np.float64) - pair[f"{metric}_left"].to_numpy(dtype=np.float64)
                stats = _stat(delta)
                higher = direction.startswith("HIGHER")
                rows.append(
                    {
                        "cell_id": cell_id, "model": CELL_BINDINGS[cell_id][0],
                        "writer_family": CELL_BINDINGS[cell_id][1],
                        "left_arm": left_arm, "right_arm": right_arm,
                        "metric": metric, "comparison": "RIGHT_MINUS_LEFT",
                        "descriptive_direction": direction, "paired_denominator": len(delta),
                        **{f"delta_{key}": value for key, value in stats.items()},
                        "right_better_count": int(((delta > 0) if higher else (delta < 0)).sum()),
                        "exact_equal_count": int((delta == 0).sum()),
                        "right_worse_count": int(((delta < 0) if higher else (delta > 0)).sum()),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 4 * 3 * len(metrics):
        raise AnalysisBoundary("method contrast denominator differs")
    return result


def _mechanism_arm_aggregate(step: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "coefficient_u", "applied_update_frobenius", "residual_norm_mean_before",
        "residual_norm_mean_after", "response_norm_mean", "potential_before",
        "potential_after", "actual_reduction", "predicted_reduction",
        "realization_error_norm_mean", "model_error_norm_mean", "discretization_defect",
        "per_request_q_res_mean", "per_request_D_R0_mean", "per_request_D_model_R0_mean",
    )
    for keys, selected in step.groupby(["cell_id", "model", "writer_family", "arm"], sort=True):
        row: dict[str, Any] = dict(zip(("cell_id", "model", "writer_family", "arm"), keys, strict=True))
        row["step_count"] = len(selected)
        row["semantic_global_hit_step_count"] = int(selected.semantic_all_strict.sum())
        row["semantic_request_strict_count_terminal"] = int(selected.sort_values("visit_ordinal").semantic_request_strict_count.iloc[-1])
        row["zero_command_step_count"] = int((selected.metric_skipped_zero_command_count == 100).sum())
        row["potential_worsened_request_visit_count"] = int(selected.actual_potential_worsened_count.sum())
        for metric in metrics:
            values = pd.to_numeric(selected[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(values):
                for name, value in _stat(values).items():
                    row[f"{metric}_{name}"] = value
                row[f"{metric}_recorded_count"] = len(values)
            else:
                for name in ("mean", "median", "p90", "max"):
                    row[f"{metric}_{name}"] = np.nan
                row[f"{metric}_recorded_count"] = 0
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 16:
        raise AnalysisBoundary("mechanism arm aggregate denominator differs")
    return result


def _scheduler_rows(parent_job: str) -> list[dict[str, Any]]:
    command = [
        "sacct", "-X", "-j", parent_job,
        "--format=JobID,JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList",
        "-n", "-P",
    ]
    output = subprocess.check_output(command, text=True)
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 9:
            raise AnalysisBoundary(f"unexpected sacct row: {line}")
        job_id, raw_id, job_name, state, exit_code, elapsed, start, end, node = fields
        if "." in job_id:
            continue
        match = re.fullmatch(rf"{re.escape(parent_job)}_(\d+)", job_id)
        if not match:
            continue
        rows.append(
            {
                "parent_job_id": parent_job,
                "array_task_id": int(match.group(1)),
                "child_job_id_raw": raw_id,
                "job_name": job_name,
                "scheduler_state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "start": start,
                "end": end,
                "node": node,
            }
        )
    if len(rows) != 4:
        raise AnalysisBoundary(f"scheduler child denominator differs: {parent_job}")
    return sorted(rows, key=lambda row: row["array_task_id"])


def _lineage_table(state_base: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parent_job, namespace, source_head, category in LINEAGE:
        scheduler = _scheduler_rows(parent_job)
        for scheduler_row in scheduler:
            task = int(scheduler_row["array_task_id"])
            task_root = state_base / namespace / f"task-{task}"
            failure_path = task_root / "failure-boundary.json"
            terminal_path = task_root / "terminal-receipt.json"
            failure: Mapping[str, Any] = _load_json(failure_path) if failure_path.exists() else {}
            terminal: Mapping[str, Any] = _load_json(terminal_path) if terminal_path.exists() else {}
            if category == "CANONICAL_B100":
                inclusion = "INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR"
            elif category == "B1_PILOT":
                inclusion = "EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY"
            elif scheduler_row["scheduler_state"].startswith("CANCELLED"):
                inclusion = "EXCLUDED_USER_CANCELLED_PARTIAL_DENOMINATOR_ZERO"
            else:
                inclusion = "EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO"
            rows.append(
                {
                    **scheduler_row,
                    "namespace": namespace,
                    "category": category,
                    "source_head": source_head,
                    "result_inclusion": inclusion,
                    "failure_boundary_present": bool(failure),
                    "failure_exception_type": failure.get("exception_type", ""),
                    "failure_exception": failure.get("exception", ""),
                    "failure_stage": failure.get("progress", {}).get("stage", ""),
                    "completed_primary_arms_before_failure": len(failure.get("progress", {}).get("completed_primary_arms_current_round", [])),
                    "science_change_count": failure.get("science_change_count", 0),
                    "threshold_change_count": failure.get("threshold_change_count", 0),
                    "tolerance_change_count": failure.get("tolerance_change_count", 0),
                    "w0_restore_pass": failure.get("w0_pointer_bytes_restore_pass", terminal.get("w0_restore_pass", "NOT_RECORDED_CANCELLED_PARTIAL")),
                    "terminal_status": terminal.get("status", ""),
                }
            )
    # The immutable v1 package records its prepared-but-unsubmitted TECH-R3.
    if INCLUDE_LEGACY_DRY_PLAN:
        rows.append({
            "parent_job_id": "NOT_SUBMITTED", "array_task_id": -1, "child_job_id_raw": "NOT_APPLICABLE",
            "job_name": "NOT_SUBMITTED", "scheduler_state": "DRY_PLAN_ONLY", "exit_code": "NOT_APPLICABLE",
            "elapsed": "00:00:00", "start": "", "end": "", "node": "",
            "namespace": "round0-tech-r3", "category": "DRY_PLAN_ONLY",
            "source_head": "ed627c93e136a5a8fb187068169ec337972ffed1",
            "result_inclusion": "EXCLUDED_NOT_SUBMITTED_DENOMINATOR_ZERO",
            "failure_boundary_present": False, "failure_exception_type": "", "failure_exception": "",
            "failure_stage": "", "completed_primary_arms_before_failure": 0,
            "science_change_count": 0, "threshold_change_count": 0, "tolerance_change_count": 0,
            "w0_restore_pass": "NOT_APPLICABLE_NO_RUN", "terminal_status": "",
        })
    return pd.DataFrame(rows)


def _artifact_inventory(state_base: Path, log_base: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root, kind in ((state_base, "LOCAL_STATE"), (log_base, "LOCAL_LOG")):
        if root.is_symlink() or not root.is_dir():
            raise AnalysisBoundary(f"artifact root is invalid: {root}")
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise AnalysisBoundary(f"artifact symlink rejected: {path}")
            if not stat.S_ISREG(info.st_mode):
                continue
            rows.append(
                {
                    "artifact_class": kind,
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": info.st_size,
                    "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def _assert_no_active_task_jobs() -> dict[str, Any]:
    output = subprocess.check_output(["squeue", "-h", "-u", os.environ.get("USER", "janghj"), "-o", "%A|%a|%j|%T"], text=True)
    active = [line for line in output.splitlines() if "odeedit_orbode" in line]
    if active:
        raise AnalysisBoundary(f"task-owned active jobs are not zero: {active}")
    return {"task_owned_active_job_count": 0, "squeue_checked": True}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _rate(count: Any, denominator: Any, value: Any) -> str:
    if pd.isna(value):
        return "N/A (entry reference)"
    return f"{int(count)}/{int(denominator)} ({100.0 * float(value):.2f}%)"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    performance: pd.DataFrame,
    nll: pd.DataFrame,
    paired: pd.DataFrame,
    contrast: pd.DataFrame,
    arm: pd.DataFrame,
    step: pd.DataFrame,
    layer: pd.DataFrame,
    compute: pd.DataFrame,
    preamble: pd.DataFrame,
    orbhit: pd.DataFrame,
    lineage: pd.DataFrame,
    prior_parity: pd.DataFrame,
    gates: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
) -> str:
    performance = performance.copy()
    canonical_ns_recorded = (
        "canonical_ns_rate" in performance.columns
        and not performance.canonical_ns_rate.isna().all()
    )
    performance["stage_order"] = performance.stage.map({name: index for index, name in enumerate(STAGE_ORDER)})
    performance = performance.sort_values(["cell_id", "stage_order"])
    core_rows = []
    for row in performance.to_dict("records"):
        core_rows.append(
            (
                row["model"], row["writer_family"], row["stage"], row["stage_status"],
                _rate(row["rewrite_success_count"], row["rewrite_success_denominator"], row["rewrite_success_rate"]),
                _rate(row["rephrase_success_count"], row["rephrase_success_denominator"], row["rephrase_success_rate"]),
                _rate(row["strict_rephrase_success_count"], row["strict_rephrase_success_denominator"], row["strict_rephrase_success_rate"]),
                _rate(
                    row["canonical_ns_numerator"],
                    row["canonical_ns_denominator"],
                    row["canonical_ns_rate"],
                ) if canonical_ns_recorded else "N/A",
                _rate(row["locality_prediction_preservation_numerator"], row["locality_prediction_preservation_denominator"], row["locality_prediction_preservation_rate"]),
                _rate(row["rewrite_target_new_accuracy_count"], row["rewrite_target_new_accuracy_denominator"], row["rewrite_target_new_accuracy_rate"]),
                _rate(row["rewrite_target_true_accuracy_count"], row["rewrite_target_true_accuracy_denominator"], row["rewrite_target_true_accuracy_rate"]),
                _rate(row["rephrase_target_new_accuracy_count"], row["rephrase_target_new_accuracy_denominator"], row["rephrase_target_new_accuracy_rate"]),
                _rate(row["rephrase_target_true_accuracy_count"], row["rephrase_target_true_accuracy_denominator"], row["rephrase_target_true_accuracy_rate"]),
                _rate(row["locality_target_true_accuracy_count"], row["locality_target_true_accuracy_denominator"], row["locality_target_true_accuracy_rate"]),
            )
        )

    def nll_rows(prefix: str) -> list[tuple[Any, ...]]:
        selected = nll[nll.kind.isin((f"{prefix}_target_new", f"{prefix}_target_true"))]
        rows: list[tuple[Any, ...]] = []
        for (cell_id, model, family, stage), group in selected.groupby(["cell_id", "model", "writer_family", "stage"], sort=False):
            values = {row["kind"]: row for row in group.to_dict("records")}
            new, true = values[f"{prefix}_target_new"], values[f"{prefix}_target_true"]
            rows.append(
                (
                    model, family, stage,
                    *(_fmt(new[f"nll_{key}"]) for key in ("mean", "median", "p90", "max")),
                    *(_fmt(true[f"nll_{key}"]) for key in ("mean", "median", "p90", "max")),
                )
            )
        order = {(CELL_BINDINGS[c][0], CELL_BINDINGS[c][1], s): c * 10 + STAGE_ORDER.index(s) for c in CELL_BINDINGS for s in STAGE_ORDER}
        return sorted(rows, key=lambda row: order[(row[0], row[1], row[2])])

    paired_focus = paired[
        paired.metric.isin(
            (
                "rewrite_target_new_nll",
                "rephrase_target_new_nll",
                "rewrite_margin_true_minus_new",
                "rephrase_margin_true_minus_new",
                "canonical_ns_rate",
                "locality_prediction_preservation_rate",
            )
        )
    ]
    paired_rows = [
        (
            row.model, row.writer_family, row.arm, row.metric, row.paired_denominator,
            _fmt(row.delta_mean), _fmt(row.delta_median), _fmt(row.delta_p90), _fmt(row.delta_max),
            f"{row.better_count}/{row.exact_equal_count}/{row.worse_count}",
        )
        for row in paired_focus.itertuples()
    ]
    arm_rows = [
        (
            row.model, row.writer_family, row.arm, row.status, row.endpoint_status,
            row.step_count, row.nonzero_action_visit_count, row.zero_action_visit_count,
            row.terminal_request_strict_count, row.global_first_hit,
            _fmt(row.terminal_net_frobenius), _fmt(row.resolution_stable_path_frobenius),
        )
        for row in arm.itertuples()
    ]
    layer_pivot = layer.pivot_table(
        index=["cell_id", "model", "writer_family", "arm"], columns="layer",
        values="terminal_net_energy_share", aggfunc="first",
    ).reset_index()
    layer_rows = [
        (
            row["model"], row["writer_family"], row["arm"],
            *(_fmt(100.0 * float(row[layer_id]), 2) for layer_id in LAYERS),
        )
        for row in layer_pivot.to_dict("records")
    ]
    compute_rows = [
        (
            row.model, row.writer_family, row.arm, _fmt(row.wall_seconds, 2),
            _fmt(row.wall_delta_vs_official_seconds, 2), _fmt(row.wall_ratio_vs_official, 3),
            row.model_forward_invocation_count, row.key_capture_count, row.factor_build_count,
            row.solve_count_recorded, row.jvp_call_count, row.physical_write_count,
            _fmt(row.jvp_wall_seconds, 2),
        )
        for row in compute.itertuples()
    ]
    compute_detail_rows = [
        (
            row.model, row.writer_family, row.arm,
            row.terminal_capture_count,
            row.stock_expected_layer_solve_count,
            _fmt(row.stock_actual_solve_count),
            row.jvp_model_forward_invocation_count,
            row.shadow_materialization_count,
            row.endpoint_evaluator_count,
            row.derived_endpoint_evaluator_count,
        )
        for row in compute.itertuples()
    ]
    compute_cell_rows = []
    for cell_id, selected in compute.groupby("cell_id", sort=True):
        first = selected.iloc[0]
        compute_cell_rows.append(
            (
                first["model"], first["writer_family"],
                _fmt(first["model_load_seconds_cell_scope"], 2),
                _fmt(first["job_total_seconds_cell_scope"], 2),
                _fmt(float(first["peak_gpu_allocated_bytes_cell_scope"]) / (1024**3), 3),
                _fmt(float(first["peak_gpu_reserved_bytes_cell_scope"]) / (1024**3), 3),
                int(selected["model_forward_invocation_count"].sum()),
                int(selected["jvp_call_count"].sum()),
                _fmt(float(selected["wall_seconds"].sum()), 2),
            )
        )
    preamble_rows = [
        (
            row.model, row.writer_family, row.status, row.official_wrapper_direct_exact_fidelity,
            f"{row.jvp_fd_allclose_count}/{row.jvp_fd_epsilon_count}", _fmt(row.jvp_fd_min_cosine, 7),
            _fmt(row.jvp_fd_max_abs_error, 8), row.state_changed_key_observed,
            row.inner_persistent_mutation_count, row.w0_pointer_bytes_restore_pass,
            row.orbhit_direct_prefix_identity_status,
        )
        for row in preamble.itertuples()
    ]
    lineage_rows = [
        (
            row.parent_job_id, row.array_task_id, row.child_job_id_raw, row.namespace,
            row.scheduler_state, row.exit_code, row.source_head[:12], row.result_inclusion,
            row.failure_exception_type or "-", (row.failure_exception[:72] + "…") if len(row.failure_exception) > 72 else (row.failure_exception or "-"),
            row.w0_restore_pass,
        )
        for row in lineage.itertuples()
    ]
    input_rows = [(row["kind"], row["path"], row["bytes"], row["mode"], row["sha256"]) for row in inputs]
    prior_parity_rows = [
        (
            row.metric, row.paired_row_count, row.exact_equal_count,
            row.nonzero_delta_count, _fmt(row.delta_mean), _fmt(row.delta_median),
            _fmt(row.delta_p90), _fmt(row.max_absolute_delta),
        )
        for row in prior_parity.itertuples()
    ]
    gap_rows = [(key, value) for key, value in (
        ("exact FLOPs", "NOT_RECORDED_SCHEMA_GAP; 호출 수와 actual wall time만 보고"),
        ("per-layer factor condition number", "NOT_RECORDED_SCHEMA_GAP"),
        ("per-layer SVD spectrum", "NOT_RECORDED_SCHEMA_GAP"),
        ("stock Official actual torch.linalg.solve call count", "NOT_RECORDED_STOCK_SOURCE; expected logical layer count only"),
        ("per-arm peak GPU memory", "NOT_RECORDED_SCHEMA_GAP; cell-level peak only"),
        ("per-request first-hit sweep/layer", "NOT_RECORDED_SCHEMA_GAP; only global all-request hit and per-step strict counts"),
        ("post-hit drift", "NOT_APPLICABLE_NO_GLOBAL_FIRST_HIT"),
        ("exact C/P matrix telemetry", "NOT_RECORDED_SCHEMA_GAP"),
        ("raw generation text/tokens", "INTENTIONALLY_NOT_PUBLISHED"),
    )]
    top = []
    for cell_id in CELL_BINDINGS:
        selected = performance[(performance.cell_id == cell_id) & performance.stage.isin(PRIMARY_ARMS)]
        top_metrics = [("rewrite_success_rate", "RS"), ("rephrase_success_rate", "PS")]
        top_metrics.append(
            ("canonical_ns_rate", "canonical NS")
            if canonical_ns_recorded
            else ("locality_prediction_preservation_rate", "PP-token")
        )
        for metric, label in top_metrics:
            maximum = selected[metric].max()
            names = ",".join(selected[selected[metric] == maximum].stage)
            top.append((CELL_BINDINGS[cell_id][0], CELL_BINDINGS[cell_id][1], label, names, f"{100*maximum:.2f}%"))
    memory_facts = []
    for cell_id, binding in CELL_BINDINGS.items():
        selected = compute[compute.cell_id == cell_id]
        memory_facts.append(
            f"{binding[0]}/{binding[1]} "
            f"{float(selected.peak_gpu_allocated_bytes_cell_scope.iloc[0]) / 1e9:.2f}/"
            f"{float(selected.peak_gpu_reserved_bytes_cell_scope.iloc[0]) / 1e9:.2f}GB"
        )

    return "\n".join(
        [
            "# Ordered Response-Barrier ODE-Edit — Server1 round0 B100 기존 분석 + canonical-NS 재실행 통합 사실 보고서"
            if canonical_ns_recorded
            else "# Ordered Response-Barrier ODE-Edit — Server1 round0 B100 상세 사실 보고서",
            "",
            "## 0. 판정과 실행 경계",
            "",
            f"- 상태: `ANALYSIS_ONLY_TERMINAL_PASS`; canonical 실행은 Slurm array `{ROUND_JOB_ID}`, round0 B100 네 cell이다.",
            "- scheduler child: " + ", ".join(
                f"`{ROUND_JOB_ID}_{cell_id}→{binding[2]}`"
                for cell_id, binding in CELL_BINDINGS.items()
            ) + "; 모두 `COMPLETED/0:0`이다.",
            "- canonical 과학 분모: `4 cells × 5 primary arms × 100 requests = 2,000 request-arm endpoints`; 입력 request는 cell별 100, 전체 실행 관측 400이다.",
            "- `ORBHit`은 ORBFH 경로의 derived materialized prefix라 primary arm이나 2,000 분모에 중복 포함하지 않았다.",
            "- 분석 중 새 model/GPU/Slurm/editing run/retry/imputation은 모두 0이며 remaining-nine은 HOLD다.",
            "- 모든 비교는 동일 round0 request/order 안의 기술적·서술적 비교다. 인과, 보편성, 자동 promotion을 주장하지 않는다. `scientific_promotion=false`.",
            "",
            "## 1. 지표 정의와 정확한 분모",
            "",
            "- NLL은 teacher-forced target token의 평균 negative log likelihood다. `target-new`는 새 사실 정답, `target-true`는 원래 사실 정답이며 NLL 수치 자체는 낮을수록 해당 target에 더 높은 확률을 준다.",
            "- `RS`는 100 rewrite pair에서 `NLL(new) < NLL(true)`인 strict count/rate다. tie는 failure다.",
            "- `PS`는 200 rephrase prompt pair에서 같은 strict 비교를 한 count/rate다. `strict PS`는 request별 두 rephrase가 모두 성공한 100-request count/rate다.",
            "- `canonical NS`는 1,000 neighborhood prompt pair에서 `NLL(target-true) < NLL(target-new)`인 strict count/rate다. tie는 failure다.",
            "- `PP-token`은 endpoint token prediction이 PRE_EDIT과 같은 token 수/전체 target token 수다. target 길이 때문에 분모가 1,010이며 canonical NS가 아니다. `PP-prompt`의 분모는 1,000이다.",
            "- rewrite/rephrase accuracy는 target token을 모두 맞힌 prompt 수로, pairwise RS/PS와 다른 secondary 지표다.",
            "- PRE_EDIT은 endpoint 이전 W0 관측이며 이번 v2 evaluator에서 canonical NS 분모 1,000을 직접 기록했다. PRE_EDIT에는 전후 비교인 PP-token/PP-prompt가 적용되지 않는다.",
            "- `ENTRY_ALREADY_HIT`은 계약상 정상 W0 no-op endpoint로 분모에 들어가지만 이번 네 cell에서는 0건이었다.",
            "",
            "## 2. 핵심 endpoint 표 — PRE_EDIT, O, QCL, NQFIX, ORBFH, JAC",
            "",
            _markdown_table(
                ["model", "writer", "stage", "status", "RS", "PS", "strict PS", "canonical NS", "PP-token", "RW new acc", "RW true acc", "RP new acc", "RP true acc", "LOC true acc"],
                core_rows,
            ),
            "",
            "각 cell에서 primary rate의 단순 최댓값(선택·promotion 규칙 아님):",
            "",
            _markdown_table(["model", "writer", "metric", "max arm(s)", "rate"], top),
            "",
            "## 2.1 기존 v1/v2 보고서 통합과 재실행 parity",
            "",
            "기존 `exhaustive-v1`은 mechanics·RS/PS·PP-token을 상세 분석했지만 endpoint locality target-new NLL이 없었다. `baseline-inclusive-v2`는 별도 PRE_EDIT canonical NS만 보완했으며 endpoint canonical NS는 schema gap으로 남았다. 이 통합판은 두 package를 immutable reference로 결속하고, 새 v2 evaluator raw에서 PRE_EDIT와 O/QCL/NQFIX/ORBFH/JAC 전체 canonical NS를 다시 계산했다.",
            "",
            _markdown_table(
                ["shared metric", "paired rows", "exact equal", "nonzero", "Δ mean", "Δ median", "Δ p90", "max |Δ|"],
                prior_parity_rows,
            ),
            "",
            "여기서 delta는 `canonical-NS rerun - 기존 exhaustive-v1`이다. canonical NS 자체는 기존 endpoint schema에 없었으므로 parity 대상으로 만들지 않았다. 기존 두 보고서와 manifest/receipt의 exact SHA는 §15 input inventory에 포함된다.",
            "",
            "## 3. Rewrite NLL 분포",
            "",
            _markdown_table(
                ["model", "writer", "stage", "new mean", "new median", "new p90", "new max", "true mean", "true median", "true p90", "true max"],
                nll_rows("rewrite"),
            ),
            "",
            "## 4. Rephrase NLL 분포",
            "",
            _markdown_table(
                ["model", "writer", "stage", "new mean", "new median", "new p90", "new max", "true mean", "true median", "true p90", "true max"],
                nll_rows("rephrase"),
            ),
            "",
            "## 4.1 Neighborhood locality NLL 분포와 canonical NS 입력",
            "",
            _markdown_table(
                ["model", "writer", "stage", "new mean", "new median", "new p90", "new max", "true mean", "true median", "true p90", "true max"],
                nll_rows("locality"),
            ) if canonical_ns_recorded else "`locality_target_new`은 legacy raw에 없어 `NOT_RECORDED_SCHEMA_GAP`이다.",
            "",
            "이 표의 각 stage는 target-new 1,000행과 target-true 1,000행을 같은 `(request, prompt_index)`로 결속한다. canonical NS count는 두 NLL의 strict 비교에서 직접 계산하며 token prediction-preservation과 섞지 않는다."
            if canonical_ns_recorded
            else "",
            "",
            f"`prompt-nll.csv.gz`는 {4 * 6 * EXPECTED_EVALUATION_ROWS:,}개 PRE_EDIT/endpoint prompt 행을 모두 보존하고, `request-endpoint-metrics.csv.gz`는 2,400개 request-stage 행에서 rewrite, 두 rephrase, 열 neighborhood target-new/target-true 관측을 결속한다.",
            "",
            "## 5. 동일 request의 Official O 대비 paired delta",
            "",
            "delta는 `ours - O`다. NLL은 음수가 더 낮은 수치, target-new margin과 NS는 양수가 더 높은 수치다. equal은 tolerance를 만들지 않은 exact arithmetic equality다.",
            "",
            _markdown_table(
                ["model", "writer", "ours", "metric", "n", "Δ mean", "Δ median", "Δ p90", "Δ max", "better/equal/worse"],
                paired_rows,
            ),
            "",
            f"모든 target-new/target-true NLL, rewrite/rephrase margin, canonical NS와 PP-token의 {len(paired)}개 paired summary는 `paired-official-deltas.csv`; QCL→NQFIX→ORBFH→JAC의 {len(contrast)}개 predeclared adjacent contrast는 `method-contrast-deltas.csv`에 있다.",
            "",
            "## 6. 다섯 primary arm의 구현상 차이",
            "",
            "- `O`: stock Official MEMIT 또는 AlphaEdit wrapper. B100 family-native fixed z 100개를 cohort entry에서 계산하고 stock entrypoint를 한 번 실행한다.",
            "- `QCL`: 각 sweep의 L4→L8 visit마다 current residual을 remaining-layer count `n_l=(5,4,3,2,1)`로 나눈 `R/n_l`, fixed `u=1`, per-visit rebuild.",
            "- `NQFIX`: 같은 20-visit clock에서 divisor를 제거한 full current residual `R`, fixed `u=1`, per-visit rebuild.",
            "- `ORBFH`: full current residual과 current response-derived `u(W)`, 실제 upstream transition 뒤 downstream factor/key/response를 per visit rebuild.",
            "- `JAC`: 각 sweep entry에서 full residual과 response `u(W)`를 정하고, sweep 내부에서는 그 command reference를 유지하는 per-sweep arm. fully entry-frozen arm은 아니다.",
            "- 이 차이는 matched descriptive attribution axis일 뿐, 단일 round0 결과만으로 causal layer importance나 universal superiority를 주장하지 않는다.",
            "",
            "## 7. Mechanism terminal 및 hit 사실",
            "",
            _markdown_table(
                ["model", "writer", "arm", "status", "endpoint", "steps", "nonzero", "zero", "terminal strict req", "global hit", "||ΔW||F", "path ||·||F"],
                arm_rows,
            ),
            "",
            "16개 dynamic endpoint 모두 20 visits를 완료하고 기술적으로 valid endpoint를 냈지만, 100 requests가 동시에 strict가 되는 global prefix는 한 번도 없었다. 따라서 16/16 상태는 `HORIZON_SEMANTIC_MISS`; 이는 계약상 과학 관측이지 기술 실패가 아니다. JAC 및 한 Qwen AlphaEdit ORBFH에서 zero-command visit이 기록됐고 나머지는 nonzero였다. per-step strict-request count와 residual 궤적은 `mechanism-step-summary.csv` 및 그림에 있다. per-request first-hit identity는 raw schema에 없어 재구성하지 않았다.",
            "",
            "## 8. Layer별 terminal net update energy share (%)",
            "",
            _markdown_table(["model", "writer", "arm", "L4", "L5", "L6", "L7", "L8"], layer_rows),
            "",
            "표는 서로 다른 parameter block의 squared Frobenius terminal-net share다. 모델/방법 사이 raw residual norm을 직접 비교하지 않았으며, absolute update와 path energy는 `layer-update-summary.csv`에 별도로 있다.",
            "",
            "## 9. Compute/time/memory",
            "",
            _markdown_table(
                ["model", "writer", "arm", "wall s", "Δs vs O", "ratio vs O", "forwards", "keys", "builds", "solves", "JVP", "writes", "JVP s"],
                compute_rows,
            ),
            "",
            "위 표의 `wall s`는 endpoint edit-core 관측 시간이고 `Δs/ratio`는 같은 model/writer cell의 Official O 대비다. JVP 시간은 endpoint wall의 구성 요소이며 별도 합산하지 않는다.",
            "",
            _markdown_table(
                ["model", "writer", "arm", "terminal captures", "stock logical solves", "stock actual solves", "JVP forwards", "shadow mats", "endpoint eval", "derived eval"],
                compute_detail_rows,
            ),
            "",
            _markdown_table(
                ["model", "writer", "model load s", "job total s", "peak alloc GiB", "peak reserved GiB", "arm forwards sum", "arm JVP sum", "arm wall sum s"],
                compute_cell_rows,
            ),
            "",
            "Official stock 내부 actual solve call은 intercept하지 않아 logical expected 5만 기록됐고, dynamic arms는 adapter-intercepted 20 solves/builds가 기록됐다. `endpoint eval`은 각 primary endpoint의 evaluator 호출, `derived eval`은 ORBHit observation-only 평가다. canonical NS v2는 같은 1,000 neighborhood prompt에 target-new와 target-true를 모두 실행하므로 그 비용은 현재 job의 actual model-forward/wall 카운터에 포함된다. 이전 schema와의 FLOP 차이를 시간으로 역산하지 않는다.",
            "",
            "peak GPU memory는 arm별이 아니라 cell-level allocated/reserved다: " + "; ".join(memory_facts) + ". model load는 cell별 1회이며 wave 분리 시 reload가 필요하다. exact FLOPs는 raw schema에 없으므로 만들지 않았고, 실제 forward/key/factor/solve/JVP/capture/write/materialization/evaluator 호출 수와 CUDA 동기화된 wall time을 계산량의 재현 가능한 대리 계정으로 보고한다.",
            "",
            "## 10. Runtime B1 preamble와 integrity",
            "",
            _markdown_table(
                ["model", "writer", "status", "stock fidelity", "FD pass", "min cosine", "max abs err", "changed-state key", "inner mutation", "W0 restore", "ORBHit audit"],
                preamble_rows,
            ),
            "",
            "모든 cell은 stock O wrapper↔direct exact fidelity, multi-epsilon JVP/FD, changed-state observation, overlay/shadow/transaction, ORBHit direct-prefix audit를 통과했다. W0 pointer/bytes, cache/method-state content, arm contamination, forbidden evaluator influence, dynamic-z, retry, inner history/cache mutation gate도 PASS/0이다.",
            "",
            "FULL-FP32 claim scope는 `MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH`다. 모든 model parameter/storage와 ORBODE controller/write path는 FP32, autocast/TF32/BF16/FP16/quantization/numeric storage cast는 0이다. MEMIT 두 cell은 stock native solve가 일시적 FP64이므로 `unqualified all-algorithm FP32`는 false이며 receipt가 이를 명시한다; AlphaEdit 두 cell은 true다.",
            "",
            "MEMIT의 method state는 `MEMIT_STATIC_COV_CACHE_CONTENT_SHA256`; AlphaEdit은 `ALPHA_CACHE_C_CONTENT_SHA256`이다. 모든 arm에서 entry/after content identity가 같고 inner append/mutation은 0이다. AlphaEdit primary endpoint당 terminal history append 1이 관측되며 transaction 종료 후 entry content로 restore됐다. MEMIT을 Alpha cache로 부르지 않는다.",
            "",
            "## 11. 기술 시도, B1 pilot, canonical B100 계보",
            "",
            _markdown_table(
                ["parent", "task", "child", "namespace", "state", "exit", "source", "denominator", "exception", "detail", "W0 restore"],
                lineage_rows,
            ),
            "",
            f"이전 기술 시도와 B1 pilot은 B100 과학 분모에서 제외했다. canonical 과학 분모는 source `{RAW_SOURCE_HEAD[:12]}…`, job `{ROUND_JOB_ID}`만 사용한다.",
            "",
            "## 12. ORBHit derived endpoint",
            "",
            _markdown_table(
                ["model", "writer", "status", "factors", "RS", "PS", "strict PS", "canonical NS", "PP-token"],
                [
                    (
                        row.model, row.writer_family, row.status, row.factor_count,
                        _rate(row.rewrite_success_count, row.rewrite_success_denominator, row.rewrite_success_rate),
                        _rate(row.rephrase_success_count, row.rephrase_success_denominator, row.rephrase_success_rate),
                        _rate(row.strict_rephrase_success_count, row.strict_rephrase_success_denominator, row.strict_rephrase_success_rate),
                        _rate(row.canonical_ns_numerator, row.canonical_ns_denominator, row.canonical_ns_rate),
                        _rate(row.locality_prediction_preservation_numerator, row.locality_prediction_preservation_denominator, row.locality_prediction_preservation_rate),
                    ) for row in orbhit.itertuples()
                ],
            ),
            "",
            "global hit이 없었으므로 derived prefix는 horizon prefix를 materialize했다. 이 표는 observation-only이며 primary arm 분모와 aggregate에 합치지 않았다.",
            "",
            "## 13. 기록 한계와 금지된 추정",
            "",
            _markdown_table(["field", "status"], gap_rows),
            "",
            "누락 필드는 시간이나 aggregate 이름으로 역추정하지 않았고 imputation/reconstruction/rerun은 0이다.",
            "",
            "## 14. Figures와 재현성",
            "",
            "- `primary-rs-ps-ns.png`: PRE_EDIT/O/QCL/NQFIX/ORBFH/JAC의 primary RS/PS/canonical NS.",
            "- `paired-target-new-nll-deltas.png`: 동일 prompt의 ours−O target-new NLL.",
            "- `hit-action-to-go-trajectories.png`: 20 visits의 residual-after와 strict-request count.",
            "- `layer-update-energy-share.png`: L4–L8 terminal-net squared-Frobenius share.",
            "- `compute-tradeoff.png`: endpoint wall ratio와 recorded forward count.",
            "",
            "모든 그림은 repository Python CLI를 같은 입력으로 두 번 실행해 PNG bytes/SHA가 동일함을 검증했다. Codex image/visualization/manual edit는 사용하지 않았다. 명령·환경·입출력 SHA는 `plot-reproduction.json`에 있다.",
            "",
            "## 15. Raw identity와 hard gates",
            "",
            _markdown_table(["kind", "absolute path", "bytes", "mode", "sha256"], input_rows),
            "",
            _markdown_table(["gate", "value"], [(key, value) for key, value in sorted(gates.items())]),
            "",
            "전체 local state/log의 SHA/bytes/mode inventory는 `artifact-inventory.csv`; canonical raw→derived table→PNG→manifest/receipt 결속은 `analysis-manifest.json`과 `rooted-analysis-receipt.json`에 있다.",
            "",
        ]
    )


def build(repo: Path, raw_root: Path, log_base: Path, output: Path, repro_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    raw_root = raw_root.resolve()
    log_base = log_base.resolve()
    if output.exists() or output.is_symlink():
        raise AnalysisBoundary(f"create-once output exists: {output}")
    if _git(repo, "rev-parse", f"{RAW_SOURCE_HEAD}^{{tree}}") != RAW_SOURCE_TREE:
        raise AnalysisBoundary("raw source head/tree binding differs")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", RAW_SOURCE_HEAD, "HEAD"], check=False).returncode:
        raise AnalysisBoundary("raw source head is not an ancestor of analysis source")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise AnalysisBoundary("tracked analysis worktree must be clean")
    active_gate = _assert_no_active_task_jobs()
    documents = _verify_document_inputs()
    references = _verify_reference_inputs(repo)
    cells, raw_inputs, gates = _load_cells(raw_root)
    inputs_before = documents + references + raw_inputs
    input_root = canonical_hash(inputs_before)
    gates.update(active_gate)
    performance, nll, prompt, request, indexed_prompts, endpoint_gates = _endpoint_tables(cells)
    gates.update(endpoint_gates)
    prior_parity = _prior_metric_parity(repo, performance)
    paired = _paired_delta_table(indexed_prompts, request)
    contrast = _method_contrast_table(request)
    arm, step, request_step, layer, compute, preamble, mechanism_gates = _mechanism_tables(cells)
    gates.update(mechanism_gates)
    mechanism_aggregate = _mechanism_arm_aggregate(step)
    orbhit = _derived_orbhit_table(cells)
    state_base = raw_root.parent
    lineage = _lineage_table(state_base)
    artifact_inventory = _artifact_inventory(state_base, log_base)
    for cell_id in CELL_BINDINGS:
        official_wall = float(compute[(compute.cell_id == cell_id) & (compute.arm == "O")].wall_seconds.iloc[0])
        selected = compute.cell_id == cell_id
        compute.loc[selected, "wall_delta_vs_official_seconds"] = compute.loc[selected, "wall_seconds"] - official_wall
        compute.loc[selected, "wall_ratio_vs_official"] = compute.loc[selected, "wall_seconds"] / official_wall
    gates.update(
        {
            "paired_official_summary_row_count": len(paired),
            "method_contrast_summary_row_count": len(contrast),
            "mechanism_arm_aggregate_row_count": len(mechanism_aggregate),
            "derived_orbhit_row_count": len(orbhit),
            "derived_orbhit_primary_denominator_influence_count": 0,
            "lineage_row_count": len(lineage),
            "artifact_inventory_row_count": len(artifact_inventory),
            "immutable_prior_report_reference_count": len(references),
            "prior_rerun_parity_metric_count": len(prior_parity),
            "prior_rerun_parity_nonzero_delta_count": int(prior_parity["nonzero_delta_count"].sum()),
            "analysis_only_model_action_count": 0,
            "analysis_only_gpu_action_count": 0,
            "analysis_only_slurm_submit_count": 0,
            "remaining_nine_submit_count": 0,
            "scientific_promotion": False,
            "raw_input_sha_before_after_unchanged": True,
        }
    )
    output.mkdir(parents=True, mode=0o755)
    table_map = {
        "core-performance-summary.csv": (performance, False),
        "nll-distribution-summary.csv": (nll, False),
        "prompt-nll.csv.gz": (prompt, True),
        "request-endpoint-metrics.csv.gz": (request, True),
        "paired-official-deltas.csv": (paired, False),
        "method-contrast-deltas.csv": (contrast, False),
        "mechanism-arm-summary.csv": (arm, False),
        "mechanism-step-summary.csv": (step, False),
        "mechanism-request-step.csv.gz": (request_step, True),
        "mechanism-arm-aggregate.csv": (mechanism_aggregate, False),
        "layer-update-summary.csv": (layer, False),
        "compute-summary.csv": (compute, False),
        "runtime-preamble-summary.csv": (preamble, False),
        "derived-orbhit-summary.csv": (orbhit, False),
        "lineage-inventory.csv": (lineage, False),
        "artifact-inventory.csv": (artifact_inventory, False),
        "previous-vs-rerun-metric-parity.csv": (prior_parity, False),
    }
    for name, (frame, compressed) in table_map.items():
        _write_csv_once(output / name, frame, compressed=compressed)

    scheduler_payload = {
        "schema": "odeedit.s06.orbode.round0.scheduler-terminal.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": f"CANONICAL_JOB_{ROUND_JOB_ID}_4_OF_4_COMPLETED",
        "task_owned_active_job_count": 0,
        "canonical_rows": lineage[lineage.parent_job_id == ROUND_JOB_ID].to_dict("records"),
        "all_lineage_row_count": len(lineage),
    }
    scheduler_payload["identity_sha256"] = canonical_hash(scheduler_payload)
    write_json_once(output / "scheduler-terminal.json", scheduler_payload)

    summary_payload = {
        "schema": "odeedit.s06.orbode.round0.analysis-summary.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "ANALYSIS_ONLY_TERMINAL_PASS",
        "canonical_job_id": ROUND_JOB_ID,
        "raw_source": {"head": RAW_SOURCE_HEAD, "tree": RAW_SOURCE_TREE},
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "denominators_and_gates": gates,
        "cell_terminal_receipt_identities": [cell["receipt"]["receipt_identity_sha256"] for cell in cells],
        "cell_terminal_receipt_identity_root": canonical_hash([cell["receipt"]["receipt_identity_sha256"] for cell in cells]),
        "claim_boundary": "MATCHED_ROUND0_DESCRIPTIVE_FACTS_ONLY",
        "scientific_promotion": False,
    }
    summary_payload["identity_sha256"] = canonical_hash(summary_payload)
    write_json_once(output / "analysis-summary.json", summary_payload)

    repro_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orbode-round0-plot-repro-", dir=repro_root) as temporary:
        reproduced = Path(temporary)
        final_command = [sys.executable, "-m", "project.run_scripts.ordered_response_barrier_ode.round0_figures", "--tables", str(output), "--output", str(output)]
        repro_command = [sys.executable, "-m", "project.run_scripts.ordered_response_barrier_ode.round0_figures", "--tables", str(output), "--output", str(reproduced)]
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(repro_root / "mplconfig")
        environment["SOURCE_DATE_EPOCH"] = "0"
        subprocess.run(final_command, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        subprocess.run(repro_command, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        plot_rows: list[dict[str, Any]] = []
        for path in sorted(output.glob("*.png")):
            counterpart = reproduced / path.name
            if not counterpart.is_file() or counterpart.is_symlink():
                raise AnalysisBoundary(f"reproduced plot absent: {path.name}")
            left_sha, right_sha = sha256_file(path), sha256_file(counterpart)
            if left_sha != right_sha or path.stat().st_size != counterpart.stat().st_size:
                raise AnalysisBoundary(f"plot byte reproduction differs: {path.name}")
            plot_rows.append(
                {
                    "path": path.name, "bytes": path.stat().st_size, "sha256": left_sha,
                    "reproduced_bytes": counterpart.stat().st_size,
                    "reproduced_sha256": right_sha, "byte_stable": True,
                }
            )
    plot_inputs = [
        member(output / name, relative_to=output, kind="plot_input")
        for name in (
            "core-performance-summary.csv", "paired-official-deltas.csv",
            "mechanism-step-summary.csv", "layer-update-summary.csv", "compute-summary.csv",
        )
    ]
    plot_receipt = {
        "schema": "odeedit.s06.orbode.round0.plot-reproduction.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "BYTE_STABLE_REPRODUCTION_PASS",
        "actual_commands": [shlex.join(final_command), shlex.join(repro_command)],
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "pillow": __import__("PIL").__version__,
            "backend": matplotlib.get_backend(), "MPLCONFIGDIR": environment["MPLCONFIGDIR"],
            "SOURCE_DATE_EPOCH": environment["SOURCE_DATE_EPOCH"],
        },
        "inputs": plot_inputs,
        "outputs": plot_rows,
    }
    plot_receipt["identity_sha256"] = canonical_hash(plot_receipt)
    write_json_once(output / "plot-reproduction.json", plot_receipt)
    gates["plot_count"] = len(plot_rows)
    gates["plot_byte_reproduction_failure_count"] = 0

    inputs_after = _verify_document_inputs() + _verify_reference_inputs(repo)
    for cell_id, (_, _, child_job) in CELL_BINDINGS.items():
        task_root = raw_root / f"task-{cell_id}"
        for path, kind in ((task_root / "result.json", "canonical_result"), (task_root / "round-00-result.json", "canonical_round"), (task_root / "terminal-receipt.json", "terminal_receipt")):
            row = member(path, kind=kind)
            row.update({"cell_id": cell_id, "child_job_id": child_job})
            inputs_after.append(row)
    if inputs_after != inputs_before or canonical_hash(inputs_after) != input_root:
        raise AnalysisBoundary("canonical input identity changed during analysis")

    audit_payload = {
        "schema": "odeedit.s06.orbode.round0.analysis-audit.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "analysis_source": {"head": _git(repo, "rev-parse", "HEAD"), "tree": _git(repo, "rev-parse", "HEAD^{tree}")},
        "raw_source": {"head": RAW_SOURCE_HEAD, "tree": RAW_SOURCE_TREE},
        "input_root": input_root,
        "inputs_before": inputs_before,
        "inputs_after": inputs_after,
        "gates": gates,
        "imputation_count": 0,
        "scientific_promotion": False,
        "new_model_gpu_slurm_action_count": 0,
    }
    audit_payload["identity_sha256"] = canonical_hash(audit_payload)
    write_json_once(output / "analysis-audit.json", audit_payload)

    report_text = _report(
        performance, nll, paired, contrast, arm, step, layer, compute,
        preamble, orbhit, lineage, prior_parity, gates, inputs_before,
    )
    report_path = output / "ordered-response-barrier-ode-round0-b100-exhaustive-factual-ko.md"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report_text)
    sources = [member(repo / name, relative_to=repo, kind="analysis_source") for name in SOURCE_FILES]
    package_members = [
        member(path, relative_to=output, kind="canonical_output")
        for path in sorted(output.iterdir()) if path.is_file() and not path.is_symlink()
    ]
    manifest = {
        "schema": "odeedit.s06.orbode.round0.analysis-manifest.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "ANALYSIS_ONLY_TERMINAL_PASS",
        "analysis_source": audit_payload["analysis_source"],
        "raw_source": audit_payload["raw_source"],
        "stream_root": STREAM_ROOT, "order_root": ORDER_ROOT,
        "inputs": inputs_before, "input_root": input_root,
        "analysis_sources": sources, "analysis_source_root": canonical_hash(sources),
        "members": package_members, "member_root": canonical_hash(package_members),
        "denominators_and_gates": gates,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = output / "analysis-manifest.json"
    write_json_once(manifest_path, manifest)
    receipt = {
        "schema": "odeedit.s06.orbode.round0.rooted-analysis-receipt.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "ANALYSIS_ONLY_TERMINAL_PASS",
        "analysis_source": manifest["analysis_source"], "raw_source": manifest["raw_source"],
        "stream_root": STREAM_ROOT, "order_root": ORDER_ROOT,
        "input_root": input_root, "analysis_source_root": manifest["analysis_source_root"],
        "member_root": manifest["member_root"],
        "manifest": {
            "path": manifest_path.name, "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path), "identity_sha256": manifest["identity_sha256"],
        },
        "report": {"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "gates": gates,
        "scientific_promotion": False,
        "new_model_gpu_slurm_action_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = output / "rooted-analysis-receipt.json"
    write_json_once(receipt_path, receipt)
    return {
        "status": receipt["status"], "output": str(output),
        "report_sha256": receipt["report"]["sha256"],
        "manifest_sha256": receipt["manifest"]["sha256"],
        "manifest_identity": receipt["manifest"]["identity_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_identity": receipt["identity_sha256"],
        "member_root": receipt["member_root"], "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--log-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo, args.raw_root, args.log_base, args.output, args.repro_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
