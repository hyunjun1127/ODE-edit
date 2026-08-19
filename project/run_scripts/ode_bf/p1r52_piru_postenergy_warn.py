"""Narrow helpers for the PIR-U post-solve-energy-WARN amendment.

The helpers are deliberately model-free.  They classify the unchanged
Structural-H certificate, combine already-produced z/W evaluator receipts,
and derive layer concentration from realized BF16 materializer energy.
"""

from __future__ import annotations

import math
import statistics
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, canonical_hash


INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-PIR-U-SEQUENTIAL-10XB100-STRUCTURALH-POSTENERGY-WARN-R1-V1"
)
METHOD_ID = "P1R52-PIR-U-SEQUENTIAL-STRUCTURAL-H-POSTENERGY-WARN-R1"
ATTEMPT_SUFFIX = "pir-u-structuralh-postenergy-warn-r1-v1"
RESULT_NAME = (
    "s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-"
    "postenergy-warn-r1-v1"
)
RUN_TOKEN = "p1r52-pir-u-sequential-postenergy-warn-r1-v1"


def classify_h_postsolve_certificate(
    *,
    solver_stage_success: Mapping[str, bool],
    strength_residual: float,
    energy_residual: float,
    p_residual: float,
    selected_minimum: float,
    primal_tolerance: float,
    energy_tolerance: float,
    postsolve_energy_warn_enabled: bool,
) -> dict[str, Any]:
    """Classify the four external gates without altering their tolerances."""

    scalar_values = (
        strength_residual,
        energy_residual,
        p_residual,
        selected_minimum,
        primal_tolerance,
        energy_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ODEBFContractError("PIR-U H certificate scalar is nonfinite")
    if primal_tolerance < 0.0 or energy_tolerance < 0.0:
        raise ODEBFContractError("PIR-U H certificate tolerance differs")
    stages = {str(key): bool(value) for key, value in solver_stage_success.items()}
    expected_stages = ("H", "P", "CAPACITY")
    if tuple(stages) != expected_stages:
        raise ODEBFContractError("PIR-U H solver-stage inventory differs")

    first_false: str | None = None
    if not all(stages.values()):
        first_false = "SOLVER_STAGE_SUCCESS"
    elif strength_residual > primal_tolerance:
        first_false = "STRENGTH_RESIDUAL"
    elif p_residual > primal_tolerance:
        first_false = "PRETRAINED_P_RESIDUAL"
    elif selected_minimum < -primal_tolerance:
        first_false = "NONNEGATIVE_SELECTED"
    elif energy_residual > energy_tolerance and not postsolve_energy_warn_enabled:
        first_false = "POSTSOLVE_ENERGY_RESIDUAL"

    warning = bool(postsolve_energy_warn_enabled and energy_residual > 0.0)
    status = (
        "HARD_FAIL"
        if first_false is not None
        else "WARN_POSTSOLVE_ENERGY_RESIDUAL"
        if warning
        else "PASS"
    )
    payload = {
        "status": status,
        "first_false_gate": first_false,
        "solver_stage_success": stages,
        "strength_residual": float(strength_residual),
        "energy_residual": float(energy_residual),
        "p_residual": float(p_residual),
        "selected_minimum": float(selected_minimum),
        "primal_tolerance": float(primal_tolerance),
        "energy_tolerance": float(energy_tolerance),
        "postsolve_energy_warn_enabled": bool(postsolve_energy_warn_enabled),
        "postsolve_energy_decision_influence_count": 0 if postsolve_energy_warn_enabled else 1,
        "replacement_energy_threshold_count": 0,
        "polish_count": 0,
        "coefficient_shrink_count": 0,
        "retry_count": 0,
        "tolerance_relaxation_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def actual_update_norm_share_receipt(
    realized_bf16_step_energy: Mapping[str, float],
) -> dict[str, Any]:
    """Derive primary norm share and secondary squared-energy share."""

    if not realized_bf16_step_energy:
        raise ODEBFContractError("realized BF16 step energy is empty")
    ordered = [(str(key), float(value)) for key, value in realized_bf16_step_energy.items()]
    if any(not math.isfinite(value) or value < 0.0 for _, value in ordered):
        raise ODEBFContractError("realized BF16 step energy differs")
    norms = [math.sqrt(value) for _, value in ordered]
    norm_total = math.fsum(norms)
    energy_total = math.fsum(value for _, value in ordered)
    norm_shares = [value / norm_total if norm_total > 0.0 else 0.0 for value in norms]
    energy_shares = [
        value / energy_total if energy_total > 0.0 else 0.0
        for _, value in ordered
    ]
    layers = []
    for (weight_name, energy), norm, norm_share, energy_share in zip(
        ordered, norms, norm_shares, energy_shares, strict=True
    ):
        marker = weight_name.split(".layers.", 1)
        layer = int(marker[1].split(".", 1)[0]) if len(marker) == 2 else None
        layers.append(
            {
                "weight_name": weight_name,
                "layer": layer,
                "realized_bf16_step_energy": energy,
                "actual_update_norm": norm,
                "actual_update_norm_share": norm_share,
                "squared_energy_share": energy_share,
            }
        )
    payload = {
        "schema": "ode-edit-s05-p1r52-piru-actual-update-norm-share/v1",
        "status": "ZERO_ENERGY" if norm_total == 0.0 else "FINITE",
        "layers": layers,
        "layer_count": len(layers),
        "actual_update_norm_total": norm_total,
        "realized_bf16_step_energy_total": energy_total,
        "actual_update_norm_share_sum": math.fsum(norm_shares),
        "squared_energy_share_sum": math.fsum(energy_shares),
        "primary_concentration_metric": "SQRT_REALIZED_BF16_STEP_ENERGY_SHARE",
        "secondary_concentration_metric": "REALIZED_BF16_STEP_ENERGY_SHARE",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def load_actual_update_norm_rows(ode_raw_root: Path) -> list[dict[str, Any]]:
    """Read the eight accepted materializer receipts from one completed B100."""

    paths = sorted(
        ode_raw_root.rglob("accepted-k*.json"),
        key=lambda path: int(path.stem.removeprefix("accepted-k")),
    )
    if [int(path.stem.removeprefix("accepted-k")) for path in paths] != list(range(1, 9)):
        raise ODEBFContractError("PIR-U accepted materializer step inventory differs")
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        materialization = source.get("materialization")
        if not isinstance(materialization, Mapping):
            raise ODEBFContractError("PIR-U materialization receipt is absent")
        energy = materialization.get("realized_bf16_step_energy")
        if not isinstance(energy, Mapping):
            raise ODEBFContractError("PIR-U realized BF16 step energy is absent")
        receipt = actual_update_norm_share_receipt(energy)
        receipt["outer_k"] = int(path.stem.removeprefix("accepted-k"))
        receipt["accepted_receipt_identity_sha256"] = source.get("identity_sha256")
        receipt["identity_sha256"] = canonical_hash(
            {key: value for key, value in receipt.items() if key != "identity_sha256"}
        )
        rows.append(receipt)
    return rows


def _flatten_nll(value: Sequence[Sequence[float]]) -> list[float]:
    flattened = [float(item) for row in value for item in row]
    if not flattened or not all(math.isfinite(item) for item in flattened):
        raise ODEBFContractError("z/W NLL panel differs")
    return flattened


def _quantiles(value: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(item) for item in value)
    lower = math.floor(0.9 * (len(ordered) - 1))
    p90 = ordered[lower]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90_lower_order_statistic": p90,
        "raw_max": ordered[-1],
    }


def _paired_nll_panel(
    z_panel: Mapping[str, Any],
    w_panel: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    z_nested = z_panel.get("target_new_nll_by_request")
    w_nested = w_panel.get("target_new_nll_by_request")
    if not isinstance(z_nested, list) or not isinstance(w_nested, list) or len(z_nested) != len(w_nested):
        raise ODEBFContractError("z/W request denominator differs")
    if any(len(z_row) != len(w_row) for z_row, w_row in zip(z_nested, w_nested, strict=True)):
        raise ODEBFContractError("z/W prompt denominator differs")
    z_values = _flatten_nll(z_nested)
    w_values = _flatten_nll(w_nested)
    gaps = [w_value - z_value for z_value, w_value in zip(z_values, w_values, strict=True)]
    ratios: list[float | None] = [
        w_value / z_value if abs(z_value) > 1.0e-12 else None
        for z_value, w_value in zip(z_values, w_values, strict=True)
    ]
    payload = {
        "name": name,
        "request_denominator": len(z_nested),
        "prompt_denominator": len(z_values),
        "z_target_new_nll": _quantiles(z_values),
        "physical_w_target_new_nll": _quantiles(w_values),
        "writer_realization_gap_w_minus_z": _quantiles(gaps),
        "writer_realization_ratio_w_over_z": _quantiles(
            [value for value in ratios if value is not None]
        ),
        "physical_w_worse_than_z_numerator": sum(value > 0.0 for value in gaps),
        "physical_w_worse_than_z_denominator": len(gaps),
        "per_request_z_target_new_nll": z_nested,
        "per_request_physical_w_target_new_nll": w_nested,
        "per_request_writer_realization_gap_w_minus_z": [
            [float(w) - float(z) for z, w in zip(z_row, w_row, strict=True)]
            for z_row, w_row in zip(z_nested, w_nested, strict=True)
        ],
        "per_request_writer_realization_ratio_w_over_z": [
            [float(w) / float(z) if abs(float(z)) > 1.0e-12 else None for z, w in zip(z_row, w_row, strict=True)]
            for z_row, w_row in zip(z_nested, w_nested, strict=True)
        ],
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def z_w_realization_receipt(
    z_scores: Mapping[str, Any],
    w_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair direct-z and the one existing physical-W evaluator receipt."""

    rewrite = _paired_nll_panel(
        z_scores["rewrite_success"], w_scores["rewrite_success"], name="rewrite"
    )
    rephrase = _paired_nll_panel(
        z_scores["paraphrase_success"],
        w_scores["paraphrase_success"],
        name="rephrase",
    )
    payload = {
        "schema": "ode-edit-s05-p1r52-piru-z-w-writer-realization/v1",
        "rewrite": rewrite,
        "rephrase": rephrase,
        "direct_z_rewrite_success": z_scores["rewrite_success"],
        "direct_z_rewrite_accuracy": z_scores["rewrite_acc"],
        "direct_z_rephrase_success": z_scores["paraphrase_success"],
        "direct_z_rephrase_accuracy": z_scores["paraphrase_acc"],
        "physical_w_rewrite_success": w_scores["rewrite_success"],
        "physical_w_rewrite_accuracy": w_scores["rewrite_acc"],
        "physical_w_rephrase_success": w_scores["paraphrase_success"],
        "physical_w_rephrase_accuracy": w_scores["paraphrase_acc"],
        "canonical_gap_term": "WRITER_REALIZATION_GAP_W_MINUS_Z",
        "duplicate_physical_w_evaluator_forward_count": 0,
        "action_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "ATTEMPT_SUFFIX",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "RESULT_NAME",
    "RUN_TOKEN",
    "actual_update_norm_share_receipt",
    "classify_h_postsolve_certificate",
    "load_actual_update_norm_rows",
    "z_w_realization_receipt",
]
