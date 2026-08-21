"""C1/C2 execution adapters for the P1R52 joint-P/C allocation.

C1 delegates to the unchanged PIR-U remaining-residual implementation.  C2
uses the same frozen entry allocation but applies a fixed contribution quota
from the entry residual while reusing the existing current-key/q field
builder and exact BF16 virtual-prefix machinery.
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
    merge_prefix_factors,
    parameter_state,
)
from .p1r52_pir_writer import (
    PIRWriterResult,
    P1R52PIRPolicy,
    plan_pir_writer,
    remaining_pi_beta,
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


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-JOINT-PC-C1-C2-INDEPENDENT-B100-V1"
METHOD_ID = "P1R52-JOINT-PC-C1-C2-WRITER-V1"


class JointPCWriterArm(str, Enum):
    C0 = "C0-PIRU-CONTROL"
    C1 = "C1-JOINT-PC-REMAINING"
    C2 = "C2-JOINT-PC-FIXED-QUOTA"


@dataclass(frozen=True, slots=True)
class JointPCWriterResult:
    arm: JointPCWriterArm
    increment: Mapping[str, WaypointFactor]
    layer_fields: tuple[P1LayerField, ...]
    final_virtual_objective: ScalableObjectiveResult
    expected_bf16_sha256: Mapping[str, str]
    target_state: torch.Tensor
    last_prefix_terminal: torch.Tensor
    last_intended_quota: torch.Tensor
    receipt: Mapping[str, Any]
    inherited_piru_result: PIRWriterResult | None = None


def _pi_tuple(entry_pi: Sequence[float]) -> tuple[float, ...]:
    pi = tuple(float(item) for item in entry_pi)
    if (
        len(pi) != len(P1R23_LAYER_ORDER)
        or any(not math.isfinite(item) or item < 0.0 for item in pi)
        or abs(math.fsum(pi) - 1.0) > SIMPLEX_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("joint P/C execution pi differs")
    return pi


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    lhs = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    rhs = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = float(torch.linalg.norm(lhs) * torch.linalg.norm(rhs))
    if denominator <= P1R24_Q_EPSILON:
        return None
    return max(-1.0, min(1.0, float(torch.dot(lhs, rhs) / denominator)))


def _factor_energy(factor: WaypointFactor) -> float:
    left_gram = factor.left.T.to(dtype=torch.float64) @ factor.left.to(
        dtype=torch.float64
    )
    right_gram = factor.right.T.to(dtype=torch.float64) @ factor.right.to(
        dtype=torch.float64
    )
    value = float(factor.theta * factor.theta) * float(
        torch.sum(left_gram * right_gram)
    )
    if not math.isfinite(value) or value < 0.0:
        raise ODEBFContractError("joint P/C factor energy differs")
    return value


def plan_c1_writer(
    model: torch.nn.Module,
    *,
    arm: JointPCWriterArm | str = JointPCWriterArm.C1,
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
) -> JointPCWriterResult:
    """Use the unchanged PIR-U source for exact C1 suffix execution."""

    selected_arm = JointPCWriterArm(arm)
    if selected_arm not in (JointPCWriterArm.C0, JointPCWriterArm.C1):
        raise ODEBFContractError("PIR-U suffix adapter arm differs")
    pi = _pi_tuple(entry_pi)
    inherited = plan_pir_writer(
        model,
        policy=P1R52PIRPolicy.PIR_U,
        hparams=hparams,
        projector=projector,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=residual_tolerance,
        objective_plan=objective_plan,
        capture_plan=capture_plan,
        base_values=base_values,
        current_factors=current_factors,
        entry_field=entry_field,
        entry_applied_slopes=entry_applied_slopes,
        entry_pi=pi,
        entry_velocity=entry_velocity,
        target_state=target_state,
        alpha_star=alpha_star,
        endpoint_nll=endpoint_nll,
        entry_nll=entry_nll,
        step_index=step_index,
    )
    beta = remaining_pi_beta(pi)
    if beta[-1] != 1.0 or tuple(inherited.velocity) != beta:
        raise ODEBFStateError("C1 remaining-residual suffix identity differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-joint-pc-suffix-writer/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "arm": selected_arm.value,
        "entry_pi": list(pi),
        "suffix_mass": [math.fsum(pi[index:]) for index in range(len(pi))],
        "beta": list(beta),
        "last_beta": beta[-1],
        "inherited_piru_receipt": inherited.receipt,
        "inherited_source_decision_delta_count": 0,
        "entry_route_source": (
            "UNCHANGED_P1R52_PIRU_P_H_FIRST_CAPACITY_LATE"
            if selected_arm is JointPCWriterArm.C0
            else "JOINT_PC_EPIGRAPH_MINIMAX"
        ),
        "current_slope_inverse_count": 0,
        "candidate_rejection_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "second_pass_count": 0,
        "debt_count": 0,
        "catchup_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    last_quota = (
        target_state.detach().to(device="cpu", dtype=torch.float32)
        - inherited.last_prefix_terminal.detach().to(device="cpu", dtype=torch.float32)
    )
    return JointPCWriterResult(
        selected_arm,
        inherited.increment,
        inherited.layer_fields,
        inherited.final_virtual_objective,
        inherited.expected_bf16_sha256,
        inherited.target_state,
        inherited.last_prefix_terminal,
        last_quota,
        receipt,
        inherited,
    )


def plan_c2_writer(
    model: torch.nn.Module,
    *,
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
    entry_pi: Sequence[float],
    target_state: torch.Tensor,
    step_index: int,
    prefix_history_keys_by_layer: Mapping[int, torch.Tensor] | None = None,
) -> JointPCWriterResult:
    """Plan fixed entry-residual quotas with current key/q recapture."""

    layers = tuple(int(item) for item in hparams.layers)
    pi = _pi_tuple(entry_pi)
    if layers != P1R23_LAYER_ORDER or tuple(item.layer for item in entry_field.layers) != layers:
        raise ODEBFContractError("C2 writer layer inventory differs")
    if prefix_history_keys_by_layer is not None and (
        set(prefix_history_keys_by_layer) != set(layers)
        or any(value.ndim != 2 or not torch.isfinite(value).all() for value in prefix_history_keys_by_layer.values())
    ):
        raise ODEBFContractError("C2 writer history inventory differs")

    target = target_state.detach().to(device="cpu", dtype=torch.float32).contiguous()
    entry_terminal = entry_field.current_z.detach().to(device="cpu", dtype=torch.float32)
    entry_residual = (target - entry_terminal).contiguous()
    weight_names = tuple(item.weight_name for item in entry_field.layers)
    before_inventory = _parameter_inventory_sha256(model)
    before_state = parameter_state(model, weight_names)
    planned: dict[str, WaypointFactor] = {}
    selected_fields: list[P1LayerField] = []
    layer_receipts: list[dict[str, Any]] = []
    terminals: list[torch.Tensor] = []
    capture_count = 0
    capture_forward_count = 0
    capture_processed_tokens = 0
    capture_padded_tokens = 0
    q_solve_count = 0

    for ordinal, layer in enumerate(layers):
        if ordinal == 0:
            current_terminal = entry_terminal.clone()
            field = entry_field.layers[0]
            capture_sha = None
            field_source = "ENTRY_FIELD_REUSE"
        else:
            candidate = merge_prefix_factors(current_factors, planned)
            entry_subset = {name: base_values[name] for name in candidate}
            with CachedBF16FunctionalTrial(model, entry_subset, candidate, row_block=64):
                physical = capture_scalable_physical_state(model, capture_plan, hparams)
                current_terminal = physical.terminal_z.detach().to(device="cpu", dtype=torch.float32)
                # The current state determines key/q.  The semantic residual fed
                # to the field is the frozen entry quota direction, not z*-y(W).
                quota_target = (current_terminal + entry_residual).contiguous()
                field, _ = build_current_layer_field(
                    model,
                    hparams,
                    projector,
                    covariance_registry,
                    layer=layer,
                    key=physical.keys_by_layer[layer],
                    target_state=quota_target,
                    current_terminal=current_terminal,
                    step_index=step_index,
                    factor_ordinal=ordinal,
                    projector_sha256=projector_sha256,
                    residual_tolerance=residual_tolerance,
                    q_only=True,
                    history_keys=(
                        None
                        if prefix_history_keys_by_layer is None
                        else prefix_history_keys_by_layer[layer]
                    ),
                )
            capture_count += 1
            capture_forward_count += physical.physical_forward_count
            capture_processed_tokens += physical.processed_token_count
            capture_padded_tokens += physical.padded_token_count
            q_solve_count += 1
            capture_sha = physical.identity_sha256
            field_source = "CURRENT_PREFIX_KEY_Q_FIXED_ENTRY_RESIDUAL"

        quota = (float(pi[ordinal]) * entry_residual).contiguous()
        # C2 is defined directly in the finite contribution coordinate:
        # theta=pi_l and left=e0.  No residual/h then h round trip is allowed.
        factor = WaypointFactor(
            field.weight_name,
            field.layer,
            0,
            step_index,
            ordinal,
            float(pi[ordinal]),
            entry_residual.clone(),
            field.q.clone(),
            global_batch_size=field.factor.global_batch_size,
        )
        intended_max_abs = float(
            torch.max(
                torch.abs(
                    float(factor.theta) * factor.left
                    - quota
                )
            )
        )
        if intended_max_abs > SIMPLEX_PRIMAL_TOLERANCE:
            raise ODEBFStateError("C2 fixed entry quota factor identity differs")
        planned[field.weight_name] = factor
        selected_fields.append(field)
        terminals.append(current_terminal.clone())
        current_residual = target - current_terminal
        layer_receipts.append(
            {
                "layer": layer,
                "field_source": field_source,
                "capture_sha256": capture_sha,
                "entry_pi": pi[ordinal],
                "fixed_quota_sha256": tensor_sha256(quota),
                "fixed_quota_norm": float(torch.linalg.norm(quota.double())),
                "current_residual_norm": float(torch.linalg.norm(current_residual.double())),
                "entry_residual_norm": float(torch.linalg.norm(entry_residual.double())),
                "key_sha256": tensor_sha256(field.key),
                "key_norm": float(torch.linalg.norm(field.key.double())),
                "q_sha256": tensor_sha256(field.q),
                "q_norm": float(torch.linalg.norm(field.q.double())),
                "solve_residual": field.woodbury_certificate.alpha_linear_residual,
                "solve_condition": field.woodbury_certificate.small_condition,
                "theta": factor.theta,
                "intended_quota_identity_max_abs": intended_max_abs,
                "factor_energy": _factor_energy(factor),
                "numeric_h_multiplication_count": 0,
                "actual_bf16_energy_share": None,
                "suffix_normalization_count": 0,
                "current_slope_inverse_count": 0,
                "quota_remainder_reallocation_count": 0,
            }
        )

    total_energy = math.fsum(float(item["factor_energy"]) for item in layer_receipts)
    for item in layer_receipts:
        item["actual_bf16_energy_share"] = (
            float(item["factor_energy"]) / total_energy if total_energy > P1R24_Q_EPSILON else 0.0
        )

    committed_fields: list[P1LayerField] = [selected_fields[0]]
    for field in selected_fields[1:]:
        action, gram, covariance_receipt = covariance_registry.action(
            field.layer, field.q, expected_batch_size=field.q.shape[1]
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
    with CachedBF16FunctionalTrial(model, final_subset, final_candidate, row_block=64):
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
    if before_inventory != _parameter_inventory_sha256(model) or before_state != parameter_state(model, weight_names):
        raise ODEBFStateError("C2 virtual prefix mutated live parameters")

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c2-writer/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "arm": JointPCWriterArm.C2.value,
        "step_index": step_index,
        "layer_order": list(layers),
        "entry_pi": list(pi),
        "entry_residual_sha256": tensor_sha256(entry_residual),
        "entry_residual_norm": float(torch.linalg.norm(entry_residual.double())),
        "layers": layer_receipts,
        "prefix_capture_count": capture_count,
        "prefix_capture_forward_count": capture_forward_count,
        "prefix_capture_processed_tokens": capture_processed_tokens,
        "prefix_capture_padded_tokens": capture_padded_tokens,
        "current_q_solve_count": q_solve_count,
        "entry_field_solve_count": len(layers),
        "total_writer_solve_count": len(layers) + q_solve_count,
        "layer4_entry_field_reuse_count": 1,
        "layer4_capture_count": 0,
        "final_virtual_w_only_nll": float(final_objective.loss),
        "final_virtual_objective_forward_count": final_objective.model_forward_count,
        "final_virtual_objective_backward_count": final_objective.backward_count,
        "final_virtual_bf16_sha256": expected_hashes,
        "total_factor_energy": total_energy,
        "suffix_normalization_count": 0,
        "debt_count": 0,
        "catchup_count": 0,
        "second_pass_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "candidate_rejection_count": 0,
        "current_slope_inverse_count": 0,
        "hard_p_h_budget_influence_count": 0,
        "heldout_writer_decision_access_count": 0,
        "target_recompute_count": 0,
        "physical_materialization_count": 0,
    }
    if capture_count != 4 or q_solve_count != 4 or receipt["total_writer_solve_count"] > 9:
        raise ODEBFStateError("C2 optimized prefix compute differs")
    receipt["identity_sha256"] = canonical_hash(receipt)
    return JointPCWriterResult(
        JointPCWriterArm.C2,
        dict(planned),
        tuple(committed_fields),
        final_objective,
        expected_hashes,
        target.clone(),
        terminals[-1].clone(),
        (float(pi[-1]) * entry_residual).contiguous(),
        receipt,
        None,
    )


def post_commit_joint_pc_identity(
    result: JointPCWriterResult,
    materialization: Mapping[str, Any],
    post_terminal: torch.Tensor,
) -> dict[str, Any]:
    observed = materialization.get("effective_bf16_sha256")
    if observed != dict(result.expected_bf16_sha256):
        raise ODEBFStateError("joint P/C virtual/physical BF16 identity differs")
    source = result.last_prefix_terminal.detach().to(device="cpu", dtype=torch.float32)
    destination = post_terminal.detach().to(device="cpu", dtype=torch.float32)
    realized = destination - source
    payload = {
        "arm": result.arm.value,
        "exact_hash_identity": True,
        "final_virtual_bf16_sha256": dict(result.expected_bf16_sha256),
        "post_commit_bf16_sha256": dict(observed),
        "physical_commit_count": 1,
        "prefix_physical_commit_count": 0,
        "last_intended_quota_norm": float(torch.linalg.norm(result.last_intended_quota.double())),
        "last_realized_displacement_norm": float(torch.linalg.norm(realized.double())),
        "last_intended_realized_cosine": _cosine(result.last_intended_quota, realized),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "INSTRUCTION_ID",
    "JointPCWriterArm",
    "JointPCWriterResult",
    "METHOD_ID",
    "plan_c1_writer",
    "plan_c2_writer",
    "post_commit_joint_pc_identity",
]
