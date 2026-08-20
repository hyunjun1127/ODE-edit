"""Prefix-refreshed residual writers for P1R52-PIR.

The P1R52 target controller, finite demand, and entry Soft routing remain
authoritative.  This module changes only the writer factors.  It reuses the
FPiQ exact-BF16 virtual-prefix machinery while omitting every per-prefix NLL
decision and current semantic-slope backward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import (
    CachedBF16FunctionalTrial,
    WaypointFactor,
    assemble_effective_bf16,
    tensor_sha256,
)
from .p1_backend import P1DynamicField, P1LayerField, PinnedCovarianceRegistry
from .p1r24_atomic_strength import P1R24_Q_EPSILON
from .p1r52_frozen_pi_quota_writer import (
    build_current_layer_field,
    build_entry_layer_field,
    build_velocity_factor,
    merge_prefix_factors,
    parameter_state,
)
from .progress_simplex_routing import SIMPLEX_PRIMAL_TOLERANCE
from .scalable_batched_model import (
    ScalableCapturePlan,
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    _parameter_inventory_sha256,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import P1R23_H, P1R23_LAYER_ORDER


P1R52_PIR_INSTRUCTION_ID = "ODEEDIT-S05-P1R52-PIR-SEQUENTIAL-WRITER-V1"
P1R52_PIR_METHOD_ID = "P1R52-PIR-SEQUENTIAL-WRITER-V1"
P1R52_PIR_CONTRACT_SHA256 = (
    "60227762fb804d4f52e7501b60c6016014468d112183cc0c2ecf9cc6fbcf730a"
)
PIR_P_RECEIPT_STATUS = "ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE"


class P1R52PIRPolicy(str, Enum):
    J0 = "J0"
    PIR_G = "PIR-G"
    PIR_U = "PIR-U"


PIR_SEQUENTIAL_POLICIES = (P1R52PIRPolicy.PIR_G, P1R52PIRPolicy.PIR_U)


class PIRTypedBoundary(ODEBFStateError):
    """Typed scientific boundary for the locked PIR-G denominator."""


@dataclass(frozen=True, slots=True)
class PIRWriterResult:
    policy: P1R52PIRPolicy
    increment: Mapping[str, WaypointFactor]
    velocity: tuple[float, ...]
    layer_fields: tuple[P1LayerField, ...]
    predicted_progress: float
    final_virtual_objective: ScalableObjectiveResult
    expected_bf16_sha256: Mapping[str, str]
    target_state: torch.Tensor
    last_prefix_terminal: torch.Tensor
    last_coefficient: float
    receipt: Mapping[str, Any]


def remaining_pi_beta(entry_pi: Sequence[float]) -> tuple[float, ...]:
    """Return beta_l=pi_l/sum_{j>=l} pi_j without an epsilon shift."""

    pi0 = tuple(float(item) for item in entry_pi)
    if (
        len(pi0) != len(P1R23_LAYER_ORDER)
        or any(not math.isfinite(item) or item < 0.0 for item in pi0)
    ):
        raise ODEBFContractError("P1R52-PIR entry pi differs")
    beta: list[float] = []
    for index, value in enumerate(pi0):
        suffix = math.fsum(pi0[index:])
        current = value / suffix if suffix > 0.0 else 0.0
        if not math.isfinite(current) or current < 0.0 or current > 1.0:
            raise ODEBFContractError("P1R52-PIR beta differs")
        beta.append(current)
    return tuple(beta)


def pir_gamma_policy(
    *,
    policy: P1R52PIRPolicy | str,
    alpha_star: float,
    entry_applied_slopes: Sequence[float],
    beta: Sequence[float],
) -> dict[str, Any]:
    """Apply the exact PIR-G/PIR-U gamma branch and strength certificate."""

    selected = P1R52PIRPolicy(policy)
    slopes = tuple(float(item) for item in entry_applied_slopes)
    shares = tuple(float(item) for item in beta)
    if (
        selected not in PIR_SEQUENTIAL_POLICIES
        or len(slopes) != len(P1R23_LAYER_ORDER)
        or len(shares) != len(P1R23_LAYER_ORDER)
        or not math.isfinite(alpha_star)
        or alpha_star < 0.0
        or any(not math.isfinite(item) for item in (*slopes, *shares))
    ):
        raise ODEBFContractError("P1R52-PIR gamma input differs")
    denominator = math.fsum(a * b for a, b in zip(slopes, shares, strict=True))
    if alpha_star <= P1R24_Q_EPSILON:
        gamma = 0.0
        status = "ZERO_FINITE_DEMAND"
    elif selected is P1R52PIRPolicy.PIR_U:
        gamma = 1.0
        status = "PIR_U_UNIT_GAMMA"
    else:
        if not math.isfinite(denominator) or denominator <= P1R24_Q_EPSILON:
            raise PIRTypedBoundary("PIR_G_ENTRY_DENOMINATOR_NONPOSITIVE_OR_NONFINITE")
        gamma = alpha_star / denominator
        status = "PIR_G_ENTRY_LINEAR_STRENGTH_MATCH"
    coefficients = tuple(gamma * item for item in shares)
    predicted = math.fsum(
        a * value for a, value in zip(slopes, coefficients, strict=True)
    )
    residual = abs(predicted - alpha_star)
    if (
        selected is P1R52PIRPolicy.PIR_G
        and alpha_star > P1R24_Q_EPSILON
        and residual > SIMPLEX_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("P1R52-PIR entry strength identity differs")
    if alpha_star <= P1R24_Q_EPSILON and any(value != 0.0 for value in coefficients):
        raise ODEBFContractError("P1R52-PIR zero-demand coefficient differs")
    return {
        "denominator": denominator,
        "gamma": gamma,
        "coefficients": coefficients,
        "predicted_progress": predicted,
        "strength_identity_residual": residual,
        "status": status,
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    lhs = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    rhs = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = float(torch.linalg.norm(lhs) * torch.linalg.norm(rhs))
    if denominator <= P1R24_Q_EPSILON:
        return None
    value = float(torch.dot(lhs, rhs) / denominator)
    return max(-1.0, min(1.0, value))


def _finish_prefix_observation(
    receipt: dict[str, Any],
    *,
    target_state: torch.Tensor,
    source_terminal: torch.Tensor,
    destination_terminal: torch.Tensor,
    coefficient: float,
) -> None:
    target = target_state.detach().to(device="cpu", dtype=torch.float32)
    source = source_terminal.detach().to(device="cpu", dtype=torch.float32)
    destination = destination_terminal.detach().to(device="cpu", dtype=torch.float32)
    before = target - source
    after = target - destination
    realized = destination - source
    intended = coefficient * before
    before_norm = float(torch.linalg.norm(before.double()))
    after_norm = float(torch.linalg.norm(after.double()))
    receipt["next_residual_norm"] = after_norm
    receipt["residual_reduction"] = before_norm - after_norm
    receipt["residual_reduction_ratio"] = (
        (before_norm - after_norm) / before_norm
        if before_norm > P1R24_Q_EPSILON
        else 0.0
    )
    receipt["negative_residual_progress"] = bool(after_norm > before_norm)
    receipt["intended_realized_cosine"] = _cosine(intended, realized)
    receipt["realized_displacement_norm"] = float(
        torch.linalg.norm(realized.double())
    )


def plan_pir_writer(
    model: torch.nn.Module,
    *,
    policy: P1R52PIRPolicy | str,
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    objective_plan: ScalableObjectivePlan,
    capture_plan: ScalableCapturePlan,
    base_values: Mapping[str, torch.Tensor],
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    entry_field: P1DynamicField,
    entry_applied_slopes: Sequence[float],
    entry_pi: Sequence[float],
    entry_velocity: Sequence[float],
    target_state: torch.Tensor,
    alpha_star: float,
    endpoint_nll: float,
    entry_nll: float,
    step_index: int,
    prefix_history_keys_by_layer: Mapping[int, torch.Tensor] | None = None,
    prefix_history_policy: str = "PIRU-LEGACY",
) -> PIRWriterResult:
    """Plan five PIR factors against exact cumulative BF16 virtual prefixes."""

    selected = P1R52PIRPolicy(policy)
    if prefix_history_policy not in ("PIRU-LEGACY", "PIRU-CACHE-COMPLETE"):
        raise ODEBFContractError("P1R52-PIR prefix history policy differs")
    if selected not in PIR_SEQUENTIAL_POLICIES:
        raise ODEBFContractError("P1R52-PIR sequential policy differs")
    layers = tuple(int(item) for item in hparams.layers)
    slopes0 = tuple(float(item) for item in entry_applied_slopes)
    pi0 = tuple(float(item) for item in entry_pi)
    velocity0 = tuple(float(item) for item in entry_velocity)
    if (
        layers != P1R23_LAYER_ORDER
        or tuple(item.layer for item in entry_field.layers) != layers
        or any(len(values) != len(layers) for values in (slopes0, pi0, velocity0))
        or any(not math.isfinite(item) for item in (*slopes0, *pi0, *velocity0))
        or any(item < 0.0 for item in (*pi0, *velocity0))
        or not math.isfinite(alpha_star)
        or alpha_star < 0.0
        or abs(max(float(entry_nll) - float(endpoint_nll), 0.0) - alpha_star)
        > SIMPLEX_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("P1R52-PIR entry identity differs")
    beta = remaining_pi_beta(pi0)
    gamma_receipt = pir_gamma_policy(
        policy=selected,
        alpha_star=alpha_star,
        entry_applied_slopes=slopes0,
        beta=beta,
    )
    gamma = float(gamma_receipt["gamma"])
    coefficients = tuple(float(item) for item in gamma_receipt["coefficients"])
    predicted = float(gamma_receipt["predicted_progress"])
    predicted_shares = tuple(
        (slopes0[index] * coefficients[index] / predicted)
        if abs(predicted) > P1R24_Q_EPSILON
        else 0.0
        for index in range(len(layers))
    )

    weight_names = tuple(item.weight_name for item in entry_field.layers)
    before_inventory = _parameter_inventory_sha256(model)
    before_state = parameter_state(model, weight_names)
    planned: dict[str, WaypointFactor] = {}
    selected_fields: list[P1LayerField] = []
    layer_receipts: list[dict[str, Any]] = []
    layer_terminals: list[torch.Tensor] = []
    capture_count = 0
    capture_forward_count = 0
    capture_processed_tokens = 0
    capture_padded_tokens = 0
    q_solve_count = 0
    expected_history_layers = set(layers)
    if prefix_history_keys_by_layer is not None and (
        set(prefix_history_keys_by_layer) != expected_history_layers
        or any(
            value.ndim != 2 or not torch.isfinite(value).all()
            for value in prefix_history_keys_by_layer.values()
        )
    ):
        raise ODEBFContractError("P1R52-PIR prefix history inventory differs")

    for ordinal, layer in enumerate(layers):
        if ordinal == 0:
            current_terminal = entry_field.current_z.detach().cpu().float()
            field, residual_identity = build_entry_layer_field(
                entry_field.layers[0],
                target_state=target_state,
                current_terminal=current_terminal,
                step_index=step_index,
                factor_ordinal=ordinal,
            )
            capture_sha = None
            field_source = "ENTRY_FIELD_REUSE"
        else:
            candidate = merge_prefix_factors(current_factors, planned)
            entry_subset = {name: base_values[name] for name in candidate}
            with CachedBF16FunctionalTrial(
                model, entry_subset, candidate, row_block=64
            ):
                physical = capture_scalable_physical_state(
                    model, capture_plan, hparams
                )
                current_terminal = physical.terminal_z.detach().cpu().float()
                field, residual_identity = build_current_layer_field(
                    model,
                    hparams,
                    projector,
                    covariance_registry,
                    layer=layer,
                    key=physical.keys_by_layer[layer],
                    target_state=target_state,
                    current_terminal=current_terminal,
                    step_index=step_index,
                    factor_ordinal=ordinal,
                    projector_sha256=projector_sha256,
                    residual_tolerance=residual_tolerance,
                    q_only=True,
                    history_keys=(
                        prefix_history_keys_by_layer[layer]
                        if prefix_history_keys_by_layer is not None
                        and prefix_history_policy == "PIRU-CACHE-COMPLETE"
                        else None
                    ),
                )
            capture_count += 1
            capture_forward_count += physical.physical_forward_count
            capture_processed_tokens += physical.processed_token_count
            capture_padded_tokens += physical.padded_token_count
            q_solve_count += 1
            capture_sha = physical.identity_sha256
            field_source = "CURRENT_PREFIX_Q_ONLY"
            _finish_prefix_observation(
                layer_receipts[-1],
                target_state=target_state,
                source_terminal=layer_terminals[-1],
                destination_terminal=current_terminal,
                coefficient=coefficients[ordinal - 1],
            )

        coefficient = coefficients[ordinal]
        factor = build_velocity_factor(
            field,
            coefficient,
            step_index=step_index,
            factor_ordinal=ordinal,
        )
        planned[field.weight_name] = factor
        selected_fields.append(field)
        layer_terminals.append(current_terminal.clone())
        factor_energy = float(factor.theta * factor.theta) * float(
            field.factor_frobenius_sq
        )
        layer_receipts.append(
            {
                "layer": layer,
                "field_source": field_source,
                "current_residual_norm": float(
                    torch.linalg.norm(
                        (target_state.detach().cpu().float() - current_terminal).double()
                    )
                ),
                "full_residual_h_identity_max_abs": residual_identity,
                "key_sha256": tensor_sha256(field.key),
                "key_norm": float(torch.linalg.norm(field.key.double())),
                "q_sha256": tensor_sha256(field.q),
                "q_norm": float(torch.linalg.norm(field.q.double())),
                "solve_history_policy": prefix_history_policy,
                "solve_history_width": (
                    int(prefix_history_keys_by_layer[layer].shape[1])
                    if ordinal == 0 and prefix_history_keys_by_layer is not None
                    else int(prefix_history_keys_by_layer[layer].shape[1])
                    if prefix_history_keys_by_layer is not None
                    and prefix_history_policy == "PIRU-CACHE-COMPLETE"
                    else 0
                ),
                "solve_history_sha256": (
                    tensor_sha256(prefix_history_keys_by_layer[layer])
                    if ordinal == 0 and prefix_history_keys_by_layer is not None
                    else tensor_sha256(prefix_history_keys_by_layer[layer])
                    if prefix_history_keys_by_layer is not None
                    and prefix_history_policy == "PIRU-CACHE-COMPLETE"
                    else tensor_sha256(
                        torch.empty((field.key.shape[0], 0), dtype=torch.float32)
                    )
                ),
                "alpha_solve_branch": field.woodbury_certificate.method.value,
                "alpha_solve_small_dimension": field.woodbury_certificate.small_dimension,
                "alpha_solve_condition": field.woodbury_certificate.small_condition,
                "alpha_solve_residual": field.woodbury_certificate.alpha_linear_residual,
                "historical_overlap_norm": (
                    0.0
                    if ordinal == 0
                    or prefix_history_keys_by_layer is None
                    or prefix_history_policy != "PIRU-CACHE-COMPLETE"
                    else float(
                        torch.linalg.norm(
                            prefix_history_keys_by_layer[layer]
                            .to(dtype=torch.float64)
                            .T
                            @ field.q.to(dtype=torch.float64)
                        )
                    )
                ),
                "capture_sha256": capture_sha,
                "entry_pi": pi0[ordinal],
                "beta": beta[ordinal],
                "gamma": gamma,
                "gamma_beta": coefficient,
                "entry_applied_slope": slopes0[ordinal],
                "applied_slope": slopes0[ordinal],
                "predicted_semantic_share": predicted_shares[ordinal],
                "entry_velocity": velocity0[ordinal],
                "coefficient_over_j0": (
                    coefficient / velocity0[ordinal]
                    if velocity0[ordinal] > P1R24_Q_EPSILON
                    else None
                ),
                "theta": factor.theta,
                "factor_energy": factor_energy,
                "actual_bf16_energy_share": None,
                "residual_inverse_h_application_count": 1,
                "physical_h_application_count": 1,
                "second_h_application_count": 0,
                "pi_application_count": 1,
                "beta_application_count": 1,
                "current_semantic_slope_backward_count": 0,
                "current_w_only_nll_decision_count": 0,
                "next_residual_norm": None,
                "residual_reduction": None,
                "residual_reduction_ratio": None,
                "negative_residual_progress": None,
                "intended_realized_cosine": None,
                "realized_displacement_norm": None,
            }
        )

    total_energy = math.fsum(float(item["factor_energy"]) for item in layer_receipts)
    for item in layer_receipts:
        item["actual_bf16_energy_share"] = (
            float(item["factor_energy"]) / total_energy
            if total_energy > P1R24_Q_EPSILON
            else 0.0
        )

    # Prefix refresh is deliberately q-only: mixed entry/current covariance is
    # not a valid same-outer PIR decision metric.  The committed q vectors must
    # nevertheless enter the unchanged next-outer Soft entry router, whose
    # cumulative-P geometry consumes accepted covariance actions.  Bind those
    # actions only after every PIR factor has been selected; this has zero
    # influence on the current writer plan and adds no model forward/backward.
    committed_fields: list[P1LayerField] = [selected_fields[0]]
    for field in selected_fields[1:]:
        action, gram, covariance_receipt = covariance_registry.action(
            field.layer,
            field.q,
            expected_batch_size=field.q.shape[1],
        )
        committed_fields.append(
            replace(
                field,
                covariance_action=action,
                covariance_gram=gram,
                covariance_receipt=covariance_receipt,
            )
        )

    final_candidate = merge_prefix_factors(current_factors, planned)
    final_subset = {name: base_values[name] for name in final_candidate}
    with CachedBF16FunctionalTrial(
        model, final_subset, final_candidate, row_block=64
    ):
        final_objective = evaluate_scalable_target_new_objective(
            model, objective_plan, target_gradient_required=False
        )
    expected_hashes: dict[str, str] = {}
    for name, factors in sorted(final_candidate.items()):
        effective, stats = assemble_effective_bf16(
            base_values[name].detach().to(
                device=next(model.parameters()).device,
                dtype=torch.bfloat16,
            ),
            factors,
            row_block=64,
        )
        expected_hashes[name] = stats.effective_bf16_sha256
        del effective
    if (
        before_inventory != _parameter_inventory_sha256(model)
        or before_state != parameter_state(model, weight_names)
    ):
        raise ODEBFStateError("P1R52-PIR virtual prefix mutated live parameters")
    coverage = (
        1.0
        if alpha_star <= P1R24_Q_EPSILON
        else (float(entry_nll) - float(final_objective.loss)) / alpha_star
    )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-pir-sequential-writer/v1",
        "instruction_id": P1R52_PIR_INSTRUCTION_ID,
        "method_id": P1R52_PIR_METHOD_ID,
        "policy": selected.value,
        "step_index": step_index,
        "layer_order": list(layers),
        "alpha_star": alpha_star,
        "alpha_star_source": "finite_demand.rho_write=[L0-Lz]+",
        "target_step_rho_write_authority_count": 0,
        "entry_nll": float(entry_nll),
        "endpoint_nll": float(endpoint_nll),
        "entry_applied_slopes": list(slopes0),
        "entry_pi": list(pi0),
        "beta": list(beta),
        "D_beta0": float(gamma_receipt["denominator"]),
        "gamma": gamma,
        "gamma_status": gamma_receipt["status"],
        "entry_strength_identity_residual": float(
            gamma_receipt["strength_identity_residual"]
        ),
        "predicted_semantic_share": list(predicted_shares),
        "entry_velocity": list(velocity0),
        "selected_velocity": list(coefficients),
        "layers": layer_receipts,
        "predicted_progress": predicted,
        "final_virtual_w_only_nll": float(final_objective.loss),
        "writer_coverage": coverage,
        "prefix_capture_count": capture_count,
        "prefix_capture_forward_count": capture_forward_count,
        "prefix_capture_processed_tokens": capture_processed_tokens,
        "prefix_capture_padded_tokens": capture_padded_tokens,
        "current_q_solve_count": q_solve_count,
        "prefix_history_policy": prefix_history_policy,
        "prefix_empty_history_solve_count": (
            len(layer_receipts) - 1 if prefix_history_policy == "PIRU-LEGACY" else 0
        ),
        "prefix_committed_history_solve_count": (
            len(layer_receipts) - 1
            if prefix_history_policy == "PIRU-CACHE-COMPLETE"
            else 0
        ),
        "current_batch_history_inclusion_count": 0,
        "current_prefix_history_inclusion_count": 0,
        "q_only_builder_count": q_solve_count,
        "q_only_virtual_prefix_covariance_action_count": 0,
        "entry_router_covariance_rebind_count": len(committed_fields) - 1,
        "entry_router_covariance_current_writer_decision_influence_count": 0,
        "layer4_entry_field_reuse_count": 1,
        "layer4_capture_count": 0,
        "layer4_added_backward_count": 0,
        "additional_slope_group_count": 0,
        "additional_slope_backward_count": 0,
        "additional_slope_forward_count": 0,
        "additional_slope_processed_tokens": 0,
        "additional_slope_padded_tokens": 0,
        "expected_additional_slope_groups_per_outer": 0,
        "final_virtual_objective_forward_count": final_objective.model_forward_count,
        "final_virtual_objective_backward_count": final_objective.backward_count,
        "live_parameter_pointer_version_hash_change_count": 0,
        "final_virtual_bf16_sha256": expected_hashes,
        "residual_inverse_h_application_count_per_layer": 1,
        "physical_h_application_count_per_layer": 1,
        "second_h_application_count": 0,
        "pi_application_count_per_layer": 1,
        "beta_application_count_per_layer": 1,
        "current_semantic_slope_backward_count": 0,
        "per_layer_current_nll_decision_count": 0,
        "inverse_current_slope_division_count": 0,
        "semantic_quota_carry_debt_count": 0,
        "sequential_p_receipt": PIR_P_RECEIPT_STATUS,
        "sequential_p_decision_influence_count": 0,
        "atomic_history_mode": "OFF",
        "atomic_history_append_count": 0,
        "hard_p_h_gate_influence_count": 0,
        "cap_influence_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "contraction_count": 0,
        "functional_veto_count": 0,
        "heldout_writer_decision_access_count": 0,
        "request_layer_router_count": 0,
        "total_factor_energy": total_energy,
    }
    if capture_count != 4 or q_solve_count != 4:
        raise ODEBFContractError("P1R52-PIR optimized prefix compute differs")
    receipt["identity_sha256"] = canonical_hash(receipt)
    return PIRWriterResult(
        selected,
        dict(planned),
        coefficients,
        tuple(committed_fields),
        predicted,
        final_objective,
        expected_hashes,
        target_state.detach().cpu().float().clone(),
        layer_terminals[-1].clone(),
        coefficients[-1],
        receipt,
    )


def post_commit_pir_identity(
    result: PIRWriterResult,
    materialization: Mapping[str, Any],
    post_terminal: torch.Tensor,
) -> dict[str, Any]:
    """Certify one-shot BF16 identity and the final layer realization."""

    observed = materialization.get("effective_bf16_sha256")
    if observed != dict(result.expected_bf16_sha256):
        raise ODEBFContractError("P1R52-PIR virtual/physical BF16 identity differs")
    last: dict[str, Any] = {}
    _finish_prefix_observation(
        last,
        target_state=result.target_state,
        source_terminal=result.last_prefix_terminal,
        destination_terminal=post_terminal,
        coefficient=result.last_coefficient,
    )
    payload = {
        "policy": result.policy.value,
        "final_virtual_bf16_sha256": dict(result.expected_bf16_sha256),
        "post_commit_bf16_sha256": dict(observed),
        "exact_hash_identity": True,
        "physical_commit_count": 1,
        "prefix_physical_commit_count": 0,
        "last_layer_realization": last,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P1R52_PIR_CONTRACT_SHA256",
    "P1R52_PIR_INSTRUCTION_ID",
    "P1R52_PIR_METHOD_ID",
    "PIR_P_RECEIPT_STATUS",
    "PIR_SEQUENTIAL_POLICIES",
    "PIRTypedBoundary",
    "PIRWriterResult",
    "P1R52PIRPolicy",
    "pir_gamma_policy",
    "plan_pir_writer",
    "post_commit_pir_identity",
    "remaining_pi_beta",
]
