"""Outcome-free P1 mechanism diagnostics from existing controller records.

The helpers in this module never evaluate a prompt and never mutate a model.
They summarize already-built low-rank fields and :class:`ArmRunResult` steps.
Exact C cosine is deferred so authoritative controller timing and peak memory
stay clean; this module retains no factor tensor for later diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import Arm, ArmRunResult, MethodContractError, ProposalBatch
from .hooks import FactorDirection


MECHANISM_SCHEMA = "ode-edit-session02-p1-mechanism/v1"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise MethodContractError(f"{name} is non-finite")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def capture_field_mechanism(
    batch: ProposalBatch,
    previous: Mapping[int, str] | None,
    *,
    normalization_epsilon: float,
) -> tuple[dict[str, Any], dict[int, str]]:
    """Capture one safe synchronous field record and transition metadata."""

    epsilon = _finite("mechanism normalization epsilon", normalization_epsilon)
    if epsilon <= 0.0:
        raise MethodContractError("mechanism normalization epsilon must be positive")
    direction_ids: dict[int, str] = {}
    for proposal in batch.proposals:
        direction = proposal.payload
        if not isinstance(direction, FactorDirection):
            raise MethodContractError("mechanism field payload is not low rank")
        # The proposal builder already computed and locked this identity.
        # Re-hashing U/V here would add tensor transfers/work inside the
        # authoritative controller timer.
        direction_ids[proposal.layer] = proposal.direction_id
    if tuple(direction_ids) != batch.layers:
        raise MethodContractError("mechanism field layer order differs")

    slope_total = sum(batch.slopes)
    normalized_slopes = (
        [0.0 for _ in batch.slopes]
        if slope_total <= epsilon
        else [float(value / slope_total) for value in batch.slopes]
    )
    ranking = [
        layer
        for _value, layer in sorted(
            ((-normalized_slopes[index], layer) for index, layer in enumerate(batch.layers))
        )
    ]
    transition: dict[str, Any] | None = None
    if previous is not None:
        if tuple(previous) != batch.layers:
            raise MethodContractError("mechanism transition layer set differs")
        changed = [
            previous[layer] != direction_ids[layer]
            for layer in batch.layers
        ]
        transition = {
            "changed_layers": sum(changed),
            "compared_layers": len(changed),
            "direction_id_transition_rate": sum(changed) / len(changed),
        }

    record = {
        "field_index": -1,
        "snapshot_id": batch.snapshot_id,
        "layers": list(batch.layers),
        "direction_ids": list(batch.direction_ids),
        "slopes": list(batch.slopes),
        "normalized_layer_efficiency": normalized_slopes,
        "layer_ranking": ranking,
        "transition_from_previous": transition,
    }
    # Return immutable string identities only.  P1 must never extend the
    # lifetime of U/V tensors or their graphs for diagnostics.
    return record, dict(direction_ids)


def _ranking_distance(left: Sequence[int], right: Sequence[int]) -> float:
    if tuple(sorted(left)) != tuple(sorted(right)) or len(left) != len(set(left)):
        raise MethodContractError("ranking drift layer sets differ")
    if len(left) < 2:
        return 0.0
    right_position = {layer: index for index, layer in enumerate(right)}
    discordant = 0
    total = 0
    for index, first in enumerate(left):
        for second in left[index + 1 :]:
            total += 1
            if right_position[first] > right_position[second]:
                discordant += 1
    return discordant / total


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise MethodContractError("allocation cosine dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise MethodContractError("accepted allocation has zero norm")
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return min(1.0, max(-1.0, value))


def trajectory_summary(
    result: ArmRunResult,
    event_history: Sequence[Mapping[str, Any]],
    *,
    controller_gpu_seconds: float,
    progress_epsilon: float,
    failure_rollback_required: bool,
    failure_rollback_exact: bool | None,
    successful_terminal_endpoint_exact: bool | None,
    arm_isolation_restore_exact: bool,
) -> dict[str, Any]:
    """Derive A1 progress fields without another model evaluation."""

    epsilon = _finite("trajectory progress epsilon", progress_epsilon)
    gpu_seconds = _finite("trajectory controller GPU seconds", controller_gpu_seconds)
    if epsilon <= 0.0 or gpu_seconds < 0.0:
        raise MethodContractError("trajectory normalization inputs are invalid")

    if result.steps:
        start_hard = result.steps[0].hard_phi_before
        start_smooth = result.steps[0].smooth_phi_before
    elif event_history:
        start_hard = _finite("entry hard phi", event_history[0]["hard_phi"])
        start_smooth = _finite("entry smooth phi", event_history[0]["smooth_phi"])
    else:
        raise MethodContractError("trajectory lacks both steps and entry event")

    accepted = [step for step in result.steps if step.accepted]
    if accepted:
        end_hard = accepted[-1].hard_phi_after
        end_smooth = accepted[-1].smooth_phi_after
    else:
        end_hard = start_hard
        end_smooth = start_smooth
    if result.steps:
        last_trial_hard = result.steps[-1].hard_phi_after
        last_trial_smooth = result.steps[-1].smooth_phi_after
    else:
        last_trial_hard = start_hard
        last_trial_smooth = start_smooth

    hard_scale = max(abs(start_hard), epsilon)
    smooth_scale = max(abs(start_smooth), epsilon)
    hard_progress = start_hard - end_hard
    smooth_progress = start_smooth - end_smooth
    accepted_rows = []
    for accepted_index, step in enumerate(accepted):
        coefficient_norm = math.sqrt(
            sum(value * value for value in step.applied_coefficients)
        )
        accepted_rows.append(
            {
                "accepted_index": accepted_index,
                "step_position": step.position,
                "direction_ids": list(step.direction_ids),
                "coefficients": list(step.applied_coefficients),
                "coefficient_l2": coefficient_norm,
                "hard_phi_before": step.hard_phi_before,
                "hard_phi_after": step.hard_phi_after,
                "smooth_phi_before": step.smooth_phi_before,
                "smooth_phi_after": step.smooth_phi_after,
                "hard_progress": step.hard_phi_before - step.hard_phi_after,
                "smooth_progress": step.smooth_phi_before - step.smooth_phi_after,
                "hard_progress_normalized_to_start": (
                    step.hard_phi_before - step.hard_phi_after
                )
                / hard_scale,
                "smooth_progress_normalized_to_start": (
                    step.smooth_phi_before - step.smooth_phi_after
                )
                / smooth_scale,
                "hard_progress_per_c_step": (
                    (step.hard_phi_before - step.hard_phi_after) / coefficient_norm
                    if coefficient_norm > epsilon
                    else None
                ),
            }
        )

    return {
        "definition": "entry-to-last-accepted-controller-trajectory; rollback-terminal-separated",
        "start_hard_phi": start_hard,
        "end_accepted_hard_phi": end_hard,
        "last_trial_hard_phi": last_trial_hard,
        "start_smooth_phi": start_smooth,
        "end_accepted_smooth_phi": end_smooth,
        "last_trial_smooth_phi": last_trial_smooth,
        "hard_progress": hard_progress,
        "smooth_progress": smooth_progress,
        "normalized_hard_progress": hard_progress / hard_scale,
        "normalized_smooth_progress": smooth_progress / smooth_scale,
        "hard_progress_per_controller_gpu_second": (
            hard_progress / gpu_seconds if gpu_seconds > epsilon else None
        ),
        "smooth_progress_per_controller_gpu_second": (
            smooth_progress / gpu_seconds if gpu_seconds > epsilon else None
        ),
        "controller_gpu_seconds": gpu_seconds,
        "accepted_steps": accepted_rows,
        "accepted_step_count": len(accepted_rows),
        "trial_step_count": len(result.steps),
        "transaction_state": {
            "omega_appended": result.omega_appended,
            "omega_used_for_rollback_inference": False,
            "failure_rollback_required": failure_rollback_required,
            "failure_rollback_exact": failure_rollback_exact,
            "successful_terminal_endpoint_exact": successful_terminal_endpoint_exact,
            "arm_isolation_restore_exact": arm_isolation_restore_exact,
            "state_fact_source": "explicit-runner-validation-not-omega-receipt",
        },
    }


def summarize_mechanism(
    arm: Arm,
    result: ArmRunResult,
    field_history: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    support_tolerance: float,
    controller_gpu_seconds: float,
    event_history: Sequence[Mapping[str, Any]],
    progress_epsilon: float,
    failure_rollback_required: bool,
    failure_rollback_exact: bool | None,
    successful_terminal_endpoint_exact: bool | None,
    arm_isolation_restore_exact: bool,
) -> dict[str, Any]:
    """Summarize direction, ranking, allocation, support, and progress drift."""

    support_epsilon = _finite("support tolerance", support_tolerance)
    if support_epsilon <= 0.0:
        raise MethodContractError("support tolerance must be positive")
    layer_tuple = tuple(int(layer) for layer in layers)
    adaptive = arm in {Arm.ONE_REFRESH, Arm.FULL_ODE_EDIT}
    history = [dict(record) for record in field_history] if adaptive else []
    if adaptive:
        for index, record in enumerate(history):
            record["field_index"] = index

    transition_rates = [
        float(transition["direction_id_transition_rate"])
        for record in history
        if isinstance((transition := record.get("transition_from_previous")), Mapping)
    ]
    ranking_drifts = [
        _ranking_distance(
            history[index - 1]["layer_ranking"], history[index]["layer_ranking"]
        )
        for index in range(1, len(history))
    ]

    accepted_vectors = []
    for step in result.steps:
        if not step.accepted:
            continue
        if adaptive and len(step.applied_coefficients) != len(layer_tuple):
            raise MethodContractError("adaptive accepted allocation layer count differs")
        coefficients = list(step.applied_coefficients)
        norm = math.sqrt(sum(value * value for value in coefficients))
        if norm <= 0.0:
            raise MethodContractError("accepted allocation norm is zero")
        support = [
            layer
            for layer, value in zip(layer_tuple, coefficients, strict=True)
            if value > support_epsilon
        ] if adaptive else []
        accepted_vectors.append(
            {
                "step_position": step.position,
                "coefficients": coefficients,
                "l2_normalized": [value / norm for value in coefficients],
                "support_layers": support,
            }
        )
    allocation_cosines = [
        _cosine(
            accepted_vectors[index - 1]["coefficients"],
            accepted_vectors[index]["coefficients"],
        )
        for index in range(1, len(accepted_vectors))
    ] if adaptive else []
    support_turnover = []
    if adaptive:
        for index in range(1, len(accepted_vectors)):
            left = set(accepted_vectors[index - 1]["support_layers"])
            right = set(accepted_vectors[index]["support_layers"])
            union = left | right
            support_turnover.append(0.0 if not union else len(left ^ right) / len(union))

    return {
        "schema_version": MECHANISM_SCHEMA,
        "arm": arm.value,
        "adaptive_drift_applicable": adaptive,
        "field_count": len(history),
        "fields": history,
        "direction_id_transition_rate": {
            "comparisons": len(transition_rates),
            "values": transition_rates,
            "mean": _mean(transition_rates),
        },
        "layer_ranking_drift": {
            "metric": "normalized-kendall-discordance",
            "comparisons": len(ranking_drifts),
            "values": ranking_drifts,
            "mean": _mean(ranking_drifts),
            "max": max(ranking_drifts) if ranking_drifts else None,
        },
        "allocation_coefficient_cosine": {
            "comparisons": len(allocation_cosines),
            "values": allocation_cosines,
            "mean": _mean(allocation_cosines),
            "min": min(allocation_cosines) if allocation_cosines else None,
        },
        "support_turnover": {
            "support_tolerance": support_epsilon,
            "metric": "jaccard-distance",
            "comparisons": len(support_turnover),
            "values": support_turnover,
            "mean": _mean(support_turnover),
            "max": max(support_turnover) if support_turnover else None,
        },
        "accepted_allocations": accepted_vectors if adaptive else [],
        "c_geometry": {
            "available": False,
            "backend": None,
            "dense_delta_materialized": False,
            "deferred_reason": "authoritative-compute-timer-and-peak-separation",
            "controller_compute_contaminated": False,
            "covariance_copy_or_recompute": False,
        },
        "trajectory": trajectory_summary(
            result,
            event_history,
            controller_gpu_seconds=controller_gpu_seconds,
            progress_epsilon=progress_epsilon,
            failure_rollback_required=failure_rollback_required,
            failure_rollback_exact=failure_rollback_exact,
            successful_terminal_endpoint_exact=successful_terminal_endpoint_exact,
            arm_isolation_restore_exact=arm_isolation_restore_exact,
        ),
    }
