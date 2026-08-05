"""Detached, raw-free stepwise layer-routing telemetry.

The helpers in this module receive numeric controller outputs only.  They do
not import torch, inspect model state, or participate in any controller
decision.  Their sole consumer is the create-once S05 receipt path.
"""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from .contracts import ODEBFContractError, canonical_hash


LAYER_IDS = (4, 5, 6, 7, 8)
SEVERE_TOP1_SHARE = 0.80
SEVERE_EFFECTIVE_LAYER_COUNT = 1.50
_DISTRIBUTION_KEYS = (
    "coefficient_share",
    "positive_predicted_progress_share",
    "prequantized_update_energy_share",
    "realized_bf16_update_energy_share",
)


def _vector(
    name: str,
    values: Sequence[float],
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    typed = tuple(values)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in typed):
        raise ODEBFContractError(f"layer-routing {name} vector type differs")
    observed = np.asarray(tuple(float(value) for value in typed), dtype=np.float64)
    if observed.shape != (len(LAYER_IDS),) or not np.isfinite(observed).all():
        raise ODEBFContractError(f"layer-routing {name} vector differs")
    if nonnegative and np.any(observed < 0.0):
        raise ODEBFContractError(f"layer-routing {name} contains negative values")
    return observed


def _mask(name: str, values: Sequence[bool | int]) -> tuple[int, ...]:
    typed = tuple(values)
    if any(
        not isinstance(value, (bool, np.bool_, Integral))
        for value in typed
    ):
        raise ODEBFContractError(f"layer-routing {name} mask type differs")
    observed = tuple(int(value) for value in typed)
    if len(observed) != len(LAYER_IDS) or any(value not in (0, 1) for value in observed):
        raise ODEBFContractError(f"layer-routing {name} mask differs")
    return observed


def _identity(name: str, value: str | None) -> str | None:
    if value is not None and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise ODEBFContractError(f"layer-routing {name} identity differs")
    return value


def _normalized_share(values: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "absolute_l1":
        mass = np.abs(values)
    elif normalization == "positive":
        mass = np.maximum(values, 0.0)
    elif normalization == "nonnegative":
        if np.any(values < 0.0):
            raise ODEBFContractError("layer-routing energy distribution is negative")
        mass = values
    else:
        raise ODEBFContractError("layer-routing normalization differs")
    total = float(mass.sum())
    return np.zeros_like(mass) if total == 0.0 else mass / total


def distribution_summary(
    values: Sequence[float],
    *,
    normalization: str,
) -> dict[str, Any]:
    """Return one recomputable distribution over the exact five layers."""

    observed = _vector("distribution", values)
    share = _normalized_share(observed, normalization)
    support = (
        np.abs(observed)
        if normalization == "absolute_l1"
        else np.maximum(observed, 0.0)
        if normalization == "positive"
        else observed
    )
    ranking = sorted(
        range(len(LAYER_IDS)),
        key=lambda index: (-float(share[index]), LAYER_IDS[index]),
    )
    nonzero = share[share > 0.0]
    hhi = float(share @ share)
    entropy = (
        0.0
        if nonzero.size == 0
        else float(-(nonzero * np.log(nonzero)).sum() / math.log(len(LAYER_IDS)))
    )
    ordered = np.sort(share)
    gini = (
        0.0
        if float(ordered.sum()) == 0.0
        else float(
            (2.0 * np.arange(1, len(LAYER_IDS) + 1) @ ordered)
            / (len(LAYER_IDS) * ordered.sum())
            - (len(LAYER_IDS) + 1) / len(LAYER_IDS)
        )
    )
    effective = 0.0 if hhi == 0.0 else 1.0 / hhi
    top1 = ranking[0]
    top2 = ranking[1]
    severe_types = []
    if nonzero.size:
        if float(share[top1]) >= SEVERE_TOP1_SHARE:
            severe_types.append("TOP1_SHARE_GTE_0_80")
        if effective <= SEVERE_EFFECTIVE_LAYER_COUNT:
            severe_types.append("EFFECTIVE_LAYER_COUNT_LTE_1_50")
    payload = {
        "layer_ids": list(LAYER_IDS),
        "normalization": normalization,
        "numeric_vector": observed.tolist(),
        "share": share.tolist(),
        "top1_layer_id": LAYER_IDS[top1],
        "top1_share": float(share[top1]),
        "top2_layer_id": LAYER_IDS[top2],
        "top2_share": float(share[top2]),
        "top2_cumulative_share": float(share[top1] + share[top2]),
        "hhi": hhi,
        "normalized_entropy": entropy,
        "gini": gini,
        "effective_layer_count": effective,
        "zero_count": int(np.count_nonzero(support == 0.0)),
        "active_layer_count": int(np.count_nonzero(share > 0.0)),
        "severe_concentration": bool(severe_types),
        "severe_metric_types": severe_types,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def relabel_layer_routing_telemetry(
    payload: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Copy one numeric observation into a later receipt category.

    Relabeling never changes the numeric vectors.  Linking the source digest
    makes rejected/accepted/first-hit/final receipt provenance explicit while
    keeping the controller-independent observation immutable.
    """

    if payload.get("schema") != "ode-edit-stepwise-layer-routing-telemetry/v1":
        raise ODEBFContractError("layer-routing source telemetry schema differs")
    if stage not in (
        "REJECTED_TRANSITION",
        "ACCEPTED_TRANSITION",
        "FIRST_HIT",
        "SHADOW_ENDPOINT",
        "FINAL_ENDPOINT",
    ):
        raise ODEBFContractError("layer-routing relabeled stage differs")
    source_sha256 = str(payload.get("identity_sha256", ""))
    _identity("source telemetry", source_sha256)
    result = deepcopy(dict(payload))
    result["stage"] = stage
    result["source_telemetry_sha256"] = source_sha256
    result.pop("identity_sha256", None)
    result["identity_sha256"] = canonical_hash(result)
    return result


def _coefficient_change(
    raw_velocity: np.ndarray,
    bf_velocity: np.ndarray,
) -> dict[str, Any]:
    difference = bf_velocity - raw_velocity
    raw_norm = float(np.linalg.norm(raw_velocity))
    bf_norm = float(np.linalg.norm(bf_velocity))
    cosine = (
        1.0
        if raw_norm == 0.0 and bf_norm == 0.0
        else 0.0
        if raw_norm == 0.0 or bf_norm == 0.0
        else float(raw_velocity @ bf_velocity / (raw_norm * bf_norm))
    )
    raw_share = _normalized_share(raw_velocity, "absolute_l1")
    bf_share = _normalized_share(bf_velocity, "absolute_l1")
    raw_top = int(np.argmax(raw_share))
    bf_top = int(np.argmax(bf_share))
    return {
        "l1": float(np.abs(difference).sum()),
        "l2": float(np.linalg.norm(difference)),
        "cosine": cosine,
        "raw_top_layer_id": LAYER_IDS[raw_top],
        "bf_top_layer_id": LAYER_IDS[bf_top],
        "top_layer_changed": LAYER_IDS[raw_top] != LAYER_IDS[bf_top],
    }


def build_layer_routing_telemetry(
    *,
    step_index: int,
    stage: str,
    layer_ids: Sequence[int],
    signed_efficiency: Sequence[float],
    raw_velocity: Sequence[float],
    bf_velocity: Sequence[float],
    applied_coefficient: Sequence[float],
    active_direction_mask: Sequence[bool | int],
    raw_cap_bound_mask: Sequence[bool | int],
    bf_cap_bound_mask: Sequence[bool | int],
    raw_zero_bound_mask: Sequence[bool | int],
    bf_zero_bound_mask: Sequence[bool | int],
    predicted_progress_contribution: Sequence[float],
    prequantized_update_energy: Sequence[float],
    realized_bf16_update_energy: Sequence[float],
    cumulative_bf16_capacity: Sequence[float],
    bf16_capacity_contribution: Sequence[float],
    structural_h_contribution: Sequence[float],
    structural_p_contribution: Sequence[float],
    trust_contribution: Sequence[float],
    field_sha256: str,
    state_sha256: str,
    target_z_sha256: str,
    factor_state_sha256: str,
    raw_solver_certificate_sha256: str,
    bf_solver_certificate_sha256: str,
    candidate_sha256: str | None,
) -> dict[str, Any]:
    """Assemble one numeric-only field/trial/transition observation."""

    if tuple(int(value) for value in layer_ids) != LAYER_IDS:
        raise ODEBFContractError("layer-routing candidate layer order differs")
    if int(step_index) < 0 or stage not in (
        "FIELD",
        "TRIAL",
        "REJECTED_TRANSITION",
        "ACCEPTED_TRANSITION",
        "FIRST_HIT",
        "SHADOW_ENDPOINT",
        "FINAL_ENDPOINT",
    ):
        raise ODEBFContractError("layer-routing stage differs")
    signed = _vector("signed efficiency", signed_efficiency)
    raw = _vector("raw velocity", raw_velocity)
    bf = _vector("BF velocity", bf_velocity)
    applied = _vector("applied coefficient", applied_coefficient)
    predicted = _vector("predicted progress", predicted_progress_contribution)
    prequantized = _vector(
        "prequantized update energy", prequantized_update_energy, nonnegative=True
    )
    realized = _vector(
        "realized BF16 update energy", realized_bf16_update_energy, nonnegative=True
    )
    cumulative = _vector(
        "cumulative BF16 capacity", cumulative_bf16_capacity, nonnegative=True
    )
    capacity_contribution = _vector(
        "BF16 capacity contribution", bf16_capacity_contribution, nonnegative=True
    )
    historical = _vector("structural H", structural_h_contribution)
    pretrained = _vector("structural P", structural_p_contribution)
    trust = _vector("trust", trust_contribution)
    coefficient_share_source = bf if stage == "FIELD" else applied
    distribution_defined = {
        "coefficient_share": True,
        "positive_predicted_progress_share": True,
        "prequantized_update_energy_share": stage != "FIELD",
        "realized_bf16_update_energy_share": stage != "FIELD",
    }
    payload = {
        "schema": "ode-edit-stepwise-layer-routing-telemetry/v1",
        "step_index": int(step_index),
        "stage": stage,
        "layer_ids": list(LAYER_IDS),
        "signed_routing_efficiency": signed.tolist(),
        "raw_velocity_coefficient": raw.tolist(),
        "bf_velocity_coefficient": bf.tolist(),
        "applied_coefficient": applied.tolist(),
        "active_direction_mask": list(_mask("active direction", active_direction_mask)),
        "raw_cap_bound_mask": list(_mask("raw cap bound", raw_cap_bound_mask)),
        "bf_cap_bound_mask": list(_mask("BF cap bound", bf_cap_bound_mask)),
        "raw_zero_bound_mask": list(_mask("raw zero bound", raw_zero_bound_mask)),
        "bf_zero_bound_mask": list(_mask("BF zero bound", bf_zero_bound_mask)),
        "predicted_progress_contribution": predicted.tolist(),
        "prequantized_update_frobenius_energy": prequantized.tolist(),
        "realized_bf16_parameter_delta_frobenius_energy": realized.tolist(),
        "cumulative_bf16_capacity": cumulative.tolist(),
        "bf16_capacity_contribution": capacity_contribution.tolist(),
        "structural_h_contribution": historical.tolist(),
        "structural_p_contribution": pretrained.tolist(),
        "trust_contribution": trust.tolist(),
        "raw_coefficient_share": distribution_summary(
            raw, normalization="absolute_l1"
        ),
        "bf_coefficient_share": distribution_summary(
            bf, normalization="absolute_l1"
        ),
        "coefficient_share": distribution_summary(
            coefficient_share_source, normalization="absolute_l1"
        ),
        "coefficient_share_source": (
            "bf_velocity_coefficient"
            if stage == "FIELD"
            else "applied_coefficient"
        ),
        "positive_predicted_progress_share": distribution_summary(
            predicted, normalization="positive"
        ),
        "prequantized_update_energy_share": distribution_summary(
            prequantized, normalization="nonnegative"
        ),
        "realized_bf16_update_energy_share": distribution_summary(
            realized, normalization="nonnegative"
        ),
        "raw_to_bf": _coefficient_change(raw, bf),
        "distribution_defined": distribution_defined,
        "identity_links": {
            "field_sha256": _identity("field", field_sha256),
            "state_sha256": _identity("state", state_sha256),
            "target_z_sha256": _identity("target-z", target_z_sha256),
            "factor_state_sha256": _identity("factor state", factor_state_sha256),
            "raw_solver_certificate_sha256": _identity(
                "raw solver certificate", raw_solver_certificate_sha256
            ),
            "bf_solver_certificate_sha256": _identity(
                "BF solver certificate", bf_solver_certificate_sha256
            ),
            "candidate_sha256": _identity("candidate", candidate_sha256),
        },
        "observation_only": True,
        "controller_dependency_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    p = _vector("left share", left, nonnegative=True)
    q = _vector("right share", right, nonnegative=True)
    if not math.isclose(float(p.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12) and float(p.sum()) != 0.0:
        raise ODEBFContractError("layer-routing left share is not normalized")
    if not math.isclose(float(q.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12) and float(q.sum()) != 0.0:
        raise ODEBFContractError("layer-routing right share is not normalized")
    midpoint = 0.5 * (p + q)

    def kl(value: np.ndarray) -> float:
        mask = value > 0.0
        return float((value[mask] * np.log(value[mask] / midpoint[mask])).sum())

    return 0.0 if float(midpoint.sum()) == 0.0 else 0.5 * (kl(p) + kl(q))


def _distribution_trajectory(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    defined_rows = [
        row
        for row in rows
        if bool(row.get("distribution_defined", {}).get(key, True))
    ]
    if not defined_rows:
        return {
            "status": "NOT_DEFINED_ON_RECORDED_STAGES",
            "maximum_top1_share": None,
            "minimum_effective_layer_count": None,
            "maximum_hhi": None,
            "longest_consecutive_severe_run": 0,
            "severe_step_count": 0,
            "severe_metric_types": [],
            "severe_run_by_metric_type": {},
            "top_layer_switch_count": 0,
            "successive_step_jensen_shannon": [],
            "initial_to_final_share_delta": [0.0] * len(LAYER_IDS),
        }
    distributions = [row[key] for row in defined_rows]
    top_ids = [int(value["top1_layer_id"]) for value in distributions]
    severe = [bool(value["severe_concentration"]) for value in distributions]
    longest = 0
    current = 0
    for value in severe:
        current = current + 1 if value else 0
        longest = max(longest, current)
    successive_js = [
        _jensen_shannon(left["share"], right["share"])
        for left, right in zip(distributions, distributions[1:], strict=False)
    ]
    initial = np.asarray(distributions[0]["share"], dtype=np.float64)
    final = np.asarray(distributions[-1]["share"], dtype=np.float64)
    metric_types = sorted(
        {
            str(metric)
            for value in distributions
            for metric in value["severe_metric_types"]
        }
    )
    severe_run_by_metric_type: dict[str, dict[str, int]] = {}
    for metric_type in metric_types:
        flags = [
            metric_type in value["severe_metric_types"]
            for value in distributions
        ]
        longest_for_type = 0
        current_for_type = 0
        for flag in flags:
            current_for_type = current_for_type + 1 if flag else 0
            longest_for_type = max(longest_for_type, current_for_type)
        severe_run_by_metric_type[metric_type] = {
            "step_count": sum(int(flag) for flag in flags),
            "longest_consecutive_run": longest_for_type,
        }
    return {
        "status": "RECORDED",
        "maximum_top1_share": max(float(value["top1_share"]) for value in distributions),
        "minimum_effective_layer_count": min(
            float(value["effective_layer_count"]) for value in distributions
        ),
        "maximum_hhi": max(float(value["hhi"]) for value in distributions),
        "longest_consecutive_severe_run": longest,
        "severe_step_count": sum(int(value) for value in severe),
        "severe_metric_types": metric_types,
        "severe_run_by_metric_type": severe_run_by_metric_type,
        "top_layer_switch_count": sum(
            int(left != right) for left, right in zip(top_ids, top_ids[1:], strict=False)
        ),
        "successive_step_jensen_shannon": successive_js,
        "initial_to_final_share_delta": (final - initial).tolist(),
    }


def summarize_layer_routing_trajectory(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_hit_step_index: int | None,
) -> dict[str, Any]:
    """Summarize initial/every-accepted/first-hit/final routing distributions."""

    values = tuple(rows)
    if not values:
        raise ODEBFContractError("layer-routing trajectory is empty")
    indices = tuple(int(row["step_index"]) for row in values)
    if indices != tuple(sorted(set(indices))):
        raise ODEBFContractError("layer-routing trajectory step order differs")
    if str(values[0]["stage"]) != "FIELD" or any(
        str(row["stage"]) != "ACCEPTED_TRANSITION" for row in values[1:]
    ):
        raise ODEBFContractError("layer-routing trajectory stage selection differs")
    for row in values:
        if tuple(int(value) for value in row["layer_ids"]) != LAYER_IDS:
            raise ODEBFContractError("layer-routing trajectory layer order differs")
        for key in _DISTRIBUTION_KEYS:
            if key not in row:
                raise ODEBFContractError("layer-routing trajectory distribution missing")
    first_hit = None
    if first_hit_step_index is not None:
        matches = [
            row
            for row in values[1:]
            if int(row["step_index"]) == first_hit_step_index
        ]
        if len(matches) != 1:
            raise ODEBFContractError("layer-routing first-hit row differs")
        first_hit = matches[0]
    transition_values = tuple(
        row for row in values if str(row["stage"]) != "FIELD"
    )
    applied_utilization = np.sum(
        [
            np.abs(_vector("applied", row["applied_coefficient"]))
            for row in transition_values
        ],
        axis=0,
    ) if transition_values else np.zeros(len(LAYER_IDS), dtype=np.float64)
    realized_utilization = np.sum(
        [
            _vector(
                "realized",
                row["realized_bf16_parameter_delta_frobenius_energy"],
                nonnegative=True,
            )
            for row in transition_values
        ],
        axis=0,
    ) if transition_values else np.zeros(len(LAYER_IDS), dtype=np.float64)
    table = []
    for ordinal, row in enumerate(values):
        item = deepcopy(dict(row))
        roles = ["INITIAL"] if ordinal == 0 else ["ACCEPTED"]
        if first_hit_step_index is not None and int(row["step_index"]) == first_hit_step_index:
            roles.append("FIRST_HIT")
        if ordinal == len(values) - 1:
            roles.append("FINAL")
        item["trajectory_roles"] = roles
        table.append(item)
    vector_keys = (
        "signed_routing_efficiency",
        "raw_velocity_coefficient",
        "bf_velocity_coefficient",
        "applied_coefficient",
        "predicted_progress_contribution",
        "prequantized_update_frobenius_energy",
        "realized_bf16_parameter_delta_frobenius_energy",
        "cumulative_bf16_capacity",
        "bf16_capacity_contribution",
        "structural_h_contribution",
        "structural_p_contribution",
        "trust_contribution",
    )
    initial_to_final_vector_delta = {
        key: (
            _vector(key, values[-1][key]) - _vector(key, values[0][key])
        ).tolist()
        for key in vector_keys
    }
    result = {
        "schema": "ode-edit-stepwise-layer-routing-trajectory/v1",
        "layer_ids": list(LAYER_IDS),
        "table": table,
        "initial_step_index": indices[0],
        "final_step_index": indices[-1],
        "first_hit_step_index": first_hit_step_index,
        "first_hit_row_sha256": (
            None if first_hit is None else str(first_hit["identity_sha256"])
        ),
        "distribution_trajectory": {
            key: _distribution_trajectory(values, key)
            for key in _DISTRIBUTION_KEYS
        },
        "initial_to_final_vector_delta": initial_to_final_vector_delta,
        "cumulative_per_layer_utilization": applied_utilization.tolist(),
        "cumulative_per_layer_omega": realized_utilization.tolist(),
        "observation_only": True,
        "controller_dependency_count": 0,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result
