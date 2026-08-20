"""Reusable P1R52 target-depth schedule and authoritative outer reassembly.

The scheduler repeats the existing P1R52 target operator at a fixed physical
outer state.  This module deliberately owns no model, writer, factor,
materializer, evaluator, or history implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
    P1R24TargetStep,
)
from .p1r39_normalized_gradient_target import _full_current_residual_step
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .p1r52_r42_safe_kdc import (
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
    select_p1r52_target_proposal,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R52_TARGET_DEPTH_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL1-IL3FULL-V1"
)
P1R52_TARGET_DEPTH_METHOD_ID = "P1R52-TARGET-DEPTH-IL1-IL3FULL"
P1R52_TARGET_DEPTH_EXTENSION_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL8-IL10-IL15-V1"
)
P1R52_TARGET_DEPTH_EXTENSION_METHOD_ID = "P1R52-TARGET-DEPTH-IL8-IL10-IL15"


class P1R52TargetDepth(str, Enum):
    IL1 = "IL1"
    IL3_FULL = "IL3-FULL"
    IL8_FULL = "IL8-FULL"
    IL10_FULL = "IL10-FULL"
    IL15_FULL = "IL15-FULL"

    @property
    def inner_count(self) -> int:
        return {
            P1R52TargetDepth.IL1: 1,
            P1R52TargetDepth.IL3_FULL: 3,
            P1R52TargetDepth.IL8_FULL: 8,
            P1R52TargetDepth.IL10_FULL: 10,
            P1R52TargetDepth.IL15_FULL: 15,
        }[self]

    @property
    def instruction_id(self) -> str:
        return (
            P1R52_TARGET_DEPTH_INSTRUCTION_ID
            if self in (P1R52TargetDepth.IL1, P1R52TargetDepth.IL3_FULL)
            else P1R52_TARGET_DEPTH_EXTENSION_INSTRUCTION_ID
        )

    @property
    def method_id(self) -> str:
        return (
            P1R52_TARGET_DEPTH_METHOD_ID
            if self in (P1R52TargetDepth.IL1, P1R52TargetDepth.IL3_FULL)
            else P1R52_TARGET_DEPTH_EXTENSION_METHOD_ID
        )

    @classmethod
    def from_inner_count(cls, inner_count: int) -> "P1R52TargetDepth":
        by_count = {item.inner_count: item for item in cls}
        try:
            return by_count[inner_count]
        except KeyError as exc:
            raise ODEBFContractError(
                "P1R52 target depth must be exactly IL1, IL3-FULL, IL8-FULL, "
                "IL10-FULL, or IL15-FULL"
            ) from exc


P1R52_TARGET_DEPTH_INNER_COUNTS = tuple(
    item.inner_count for item in P1R52TargetDepth
)


@dataclass(frozen=True, slots=True)
class P1R52TargetDepthInner:
    inner_index: int
    global_target_update_ordinal: int
    target_step: P1R24TargetStep
    selected_endpoint: Any
    next_state: P1R51ControllerState
    observation: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class P1R52TargetDepthOuter:
    target_step: P1R24TargetStep
    selected_endpoint: Any
    next_state: P1R51ControllerState
    inner_steps: tuple[P1R52TargetDepthInner, ...]
    target_results: tuple[ScalableObjectiveResult, ...]
    kl_results: tuple[P1R24KLResult, ...]
    early_stop_byte_identical: bool
    receipt: Mapping[str, Any]


def requestwise_byte_identical_mask(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[bool, ...]:
    """Return exact per-request tensor identity without numeric tolerance."""

    if left.shape != right.shape or left.ndim != 2 or left.dtype != right.dtype:
        raise ODEBFContractError("P1R52 target-depth request geometry differs")
    left_cpu = left.detach().to(device="cpu").contiguous()
    right_cpu = right.detach().to(device="cpu").contiguous()
    return tuple(
        bool(torch.equal(left_cpu[:, index], right_cpu[:, index]))
        for index in range(left_cpu.shape[1])
    )


def reassemble_p1r52_outer_target(
    *,
    outer_step_index: int,
    depth: P1R52TargetDepth | str,
    outer_entry_target: torch.Tensor,
    current_terminal: torch.Tensor,
    inner_steps: Sequence[P1R52TargetDepthInner],
    outer_entry_nll_gradient: torch.Tensor,
    outer_entry_aggregate_gradient: torch.Tensor,
    physical_state_sha256_before: str,
    physical_state_sha256_after_inners: str,
    teacher_sha256_before: str,
    teacher_sha256_after_inners: str,
    history_cache_sha256_before: str,
    history_cache_sha256_after_inners: str,
    factor_inventory_sha256_before: str,
    factor_inventory_sha256_after_inners: str,
    early_stop_byte_identical: bool | None = None,
) -> P1R52TargetDepthOuter:
    """Rebuild the one authoritative writer input from outer entry to final z.

    The entry gradient is retained only for legacy observation fields produced
    by ``_full_current_residual_step``.  Finite demand is rebuilt separately by
    the unchanged P1R34 builder and is the sole writer-strength authority.
    """

    policy = depth if isinstance(depth, P1R52TargetDepth) else P1R52TargetDepth(depth)
    if outer_step_index < 0 or outer_step_index >= 8:
        raise ODEBFContractError("P1R52 target-depth outer step differs")
    if not inner_steps or len(inner_steps) > policy.inner_count:
        raise ODEBFContractError("P1R52 target-depth inner count differs")
    if tuple(item.inner_index for item in inner_steps) != tuple(range(len(inner_steps))):
        raise ODEBFContractError("P1R52 target-depth inner ordering differs")
    if any(
        item.global_target_update_ordinal
        != outer_step_index * policy.inner_count + item.inner_index
        for item in inner_steps
    ):
        raise ODEBFContractError("P1R52 global target update ordinal differs")
    fixed_pairs = (
        (physical_state_sha256_before, physical_state_sha256_after_inners),
        (teacher_sha256_before, teacher_sha256_after_inners),
        (history_cache_sha256_before, history_cache_sha256_after_inners),
        (factor_inventory_sha256_before, factor_inventory_sha256_after_inners),
    )
    if any(before != after for before, after in fixed_pairs):
        raise ODEBFContractError("P1R52 fixed outer state changed inside target inners")

    last = inner_steps[-1]
    target_star = last.target_step.target_next
    last_inner_entry = (
        outer_entry_target if len(inner_steps) == 1 else inner_steps[-2].target_step.target_next
    )
    last_inner_no_move = requestwise_byte_identical_mask(last_inner_entry, target_star)
    outer_no_move = requestwise_byte_identical_mask(outer_entry_target, target_star)
    early_stop = (
        bool(torch.equal(last_inner_entry, target_star))
        if early_stop_byte_identical is None
        else bool(early_stop_byte_identical)
    )
    last_displacement = (
        target_star.detach().to(device="cpu", dtype=torch.float64)
        - last_inner_entry.detach().to(device="cpu", dtype=torch.float64)
    )
    outer_displacement = (
        target_star.detach().to(device="cpu", dtype=torch.float64)
        - outer_entry_target.detach().to(device="cpu", dtype=torch.float64)
    )
    full_residual = (
        target_star.detach().to(device="cpu", dtype=torch.float64)
        - current_terminal.detach().to(device="cpu", dtype=torch.float64)
    )
    inner_receipts = [
        {
            "inner_index": item.inner_index,
            "global_target_update_ordinal": item.global_target_update_ordinal,
            "target_step_sha256": item.target_step.receipt["identity_sha256"],
            "target_next_sha256": tensor_sha256(item.target_step.target_next),
            "selection_by_request": list(item.target_step.receipt["selection_by_request"]),
            "selected_endpoint_sha256": item.selected_endpoint.identity_sha256,
            "selected_endpoint_nll": float(item.selected_endpoint.loss),
            "accepted_z_observation": item.observation,
        }
        for item in inner_steps
    ]
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-depth-outer-reassembly/v1",
        "instruction_id": policy.instruction_id,
        "method_id": policy.method_id,
        "depth_policy": policy.value,
        "configured_inner_count": policy.inner_count,
        "executed_inner_count": len(inner_steps),
        "outer_step_index": outer_step_index,
        "inner_step_index_policy": "ALL_INNERS_USE_OUTER_K",
        "global_target_update_ordinal_observation_only": True,
        "early_stop_policy": "ENTIRE_SELECTED_FP32_TARGET_BYTE_IDENTICAL_ONLY",
        "early_stop_byte_identical": early_stop,
        "inner_trajectory": inner_receipts,
        "selection_by_request": list(
            last.target_step.receipt["selection_by_request"]
        ),
        "semantic_held_mask": [bool(item) for item in outer_no_move],
        "target_hold_mask": [bool(item) for item in outer_no_move],
        "last_inner_target_displacement_sha256": tensor_sha256(last_displacement),
        "last_inner_target_displacement_norm": float(torch.linalg.vector_norm(last_displacement)),
        "outer_net_target_displacement_sha256": tensor_sha256(outer_displacement),
        "outer_net_target_displacement_norm": float(torch.linalg.vector_norm(outer_displacement)),
        "final_full_current_residual_sha256": tensor_sha256(full_residual),
        "final_full_current_residual_norm": float(torch.linalg.vector_norm(full_residual)),
        "last_inner_selection_mask": [bool(item) for item in last_inner_no_move],
        "last_inner_current_mask": [
            str(item) == "CURRENT"
            for item in last.target_step.receipt["selection_by_request"]
        ],
        "outer_net_no_move_mask": [bool(item) for item in outer_no_move],
        "outer_realization_mask_rebuilt_from_entry_to_final": True,
        "last_inner_gradient_writer_authority_count": 0,
        "outer_entry_linear_diagnostic_writer_authority_count": 0,
        "inner_write_velocity_authority_count": 0,
        "inner_required_displacement_authority_count": 0,
        "inner_rho_sum_count": 0,
        "inner_alpha_target_sum_count": 0,
        "finite_demand_builder_expected_count_per_outer": 1,
        "inner_writer_materialization_count": 0,
        "inner_factor_build_count": 0,
        "inner_weight_mutation_count": 0,
        "inner_history_append_count": 0,
        "inner_covariance_recompute_count": 0,
        "inner_heldout_access_count": sum(
            int(item.observation is not None) for item in inner_steps
        ),
        "inner_observation_pass_count": sum(
            int(item.observation is not None) for item in inner_steps
        ),
        "inner_observation_added_model_forward_count": sum(
            int(item.observation["added_model_forward_count"])
            for item in inner_steps
            if item.observation is not None
        ),
        "inner_observation_added_backward_count": 0,
        "inner_observation_added_generation_count": 0,
        "inner_observation_action_influence_count": 0,
        "inner_observation_duplicate_evaluation_count": 0,
        "outer_writer_materialization_expected_count": 1,
        "entry_norm_calibration_count": sum(
            int(item.target_step.receipt["entry_norm_calibration_count"])
            for item in inner_steps
        ),
        "scratch_state_commit_expected_count": 1,
        "physical_state_sha256": physical_state_sha256_before,
        "teacher_sha256": teacher_sha256_before,
        "kl_teacher_input_sha256": teacher_sha256_before,
        "history_cache_sha256": history_cache_sha256_before,
        "factor_inventory_sha256": factor_inventory_sha256_before,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    step = _full_current_residual_step(
        target_next=target_star,
        current_target=outer_entry_target,
        current_terminal=current_terminal,
        nll_gradient=outer_entry_nll_gradient,
        aggregate_gradient=outer_entry_aggregate_gradient,
        held=outer_no_move,
        receipt=receipt,
    )
    return P1R52TargetDepthOuter(
        step,
        last.selected_endpoint,
        last.next_state,
        tuple(inner_steps),
        (),
        (),
        early_stop,
        step.receipt,
    )


def run_p1r52_target_depth_scheduler(
    *,
    outer_step_index: int,
    depth: P1R52TargetDepth | str,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    state: P1R51ControllerState,
    lock: P1R24AliasTargetLock,
    alias: str,
    shared_speed: float,
    teacher_sha256: str,
    evaluate_target: Callable[[torch.Tensor], ScalableObjectiveResult],
    evaluate_kl: Callable[[torch.Tensor], P1R24KLResult],
    evaluate_endpoint: Callable[[torch.Tensor, str], ScalableObjectiveResult],
    fixed_state_identities: Callable[[], tuple[str, str, str]],
    observe_inner: Callable[..., Mapping[str, Any]] | None = None,
    first_target_result: ScalableObjectiveResult | None = None,
    first_kl_result: P1R24KLResult | None = None,
) -> P1R52TargetDepthOuter:
    """Run IL1/IL3 target operators while the physical outer state is fixed."""

    policy = depth if isinstance(depth, P1R52TargetDepth) else P1R52TargetDepth(depth)
    if (first_target_result is None) != (first_kl_result is None):
        raise ODEBFContractError("P1R52 first inner result pair differs")
    physical_before, history_before, factors_before = fixed_state_identities()
    scratch_target = current_target
    scratch_state = state
    target_results: list[ScalableObjectiveResult] = []
    kl_results: list[P1R24KLResult] = []
    inner_steps: list[P1R52TargetDepthInner] = []
    stopped_early = False
    entry_nll_gradient: torch.Tensor | None = None
    entry_aggregate_gradient: torch.Tensor | None = None
    for inner_index in range(policy.inner_count):
        if inner_index == 0 and first_target_result is not None:
            target_result = first_target_result
            assert first_kl_result is not None
            kl_result = first_kl_result
        else:
            target_result = evaluate_target(scratch_target)
            kl_result = evaluate_kl(scratch_target)
        target_results.append(target_result)
        kl_results.append(kl_result)
        proposal = prepare_p1r52_target_proposal(
            scratch_target,
            current_terminal,
            target_origin,
            target_result,
            kl_result,
            lock,
            scratch_state,
            alias=alias,
            step_index=outer_step_index,
            shared_speed=shared_speed,
            kl_teacher_input_sha256=teacher_sha256,
        )
        if inner_index == 0:
            entry_nll_gradient = proposal.primary_step.nll_gradient
            entry_aggregate_gradient = proposal.primary_step.combined_gradient
        primary_endpoint = evaluate_endpoint(
            proposal.primary_step.target_next, "PRIMARY"
        )
        rescue = prepare_p1r52_rescue_proposal(
            proposal,
            scratch_target,
            current_terminal,
            target_result,
            primary_endpoint,
            step_index=outer_step_index,
        )
        rescue_endpoint = (
            None
            if rescue.rescue_step is None
            else evaluate_endpoint(rescue.rescue_step.target_next, "RESCUE")
        )
        selected = select_p1r52_target_proposal(
            proposal,
            rescue,
            scratch_target,
            current_terminal,
            target_result,
            primary_endpoint,
            rescue_endpoint,
            step_index=outer_step_index,
        )
        observation = (
            None
            if observe_inner is None
            else observe_inner(
                depth_policy=policy.value,
                configured_inner_count=policy.inner_count,
                outer_step_index=outer_step_index,
                inner_index=inner_index,
                global_target_update_ordinal=(
                    outer_step_index * policy.inner_count + inner_index
                ),
                outer_entry_target=current_target,
                previous_target=scratch_target,
                accepted_target=selected.target_step.target_next,
                current_terminal=current_terminal,
                selected_endpoint=selected.selected_endpoint,
            )
        )
        inner_steps.append(
            P1R52TargetDepthInner(
                inner_index,
                outer_step_index * policy.inner_count + inner_index,
                selected.target_step,
                selected.selected_endpoint,
                selected.next_state,
                observation,
            )
        )
        byte_identical = torch.equal(selected.target_step.target_next, scratch_target)
        scratch_target = selected.target_step.target_next
        scratch_state = selected.next_state
        if byte_identical:
            stopped_early = True
            break
    assert entry_nll_gradient is not None and entry_aggregate_gradient is not None
    physical_after, history_after, factors_after = fixed_state_identities()
    outer = reassemble_p1r52_outer_target(
        outer_step_index=outer_step_index,
        depth=policy,
        outer_entry_target=current_target,
        current_terminal=current_terminal,
        inner_steps=inner_steps,
        outer_entry_nll_gradient=entry_nll_gradient,
        outer_entry_aggregate_gradient=entry_aggregate_gradient,
        physical_state_sha256_before=physical_before,
        physical_state_sha256_after_inners=physical_after,
        teacher_sha256_before=teacher_sha256,
        teacher_sha256_after_inners=teacher_sha256,
        history_cache_sha256_before=history_before,
        history_cache_sha256_after_inners=history_after,
        factor_inventory_sha256_before=factors_before,
        factor_inventory_sha256_after_inners=factors_after,
        early_stop_byte_identical=stopped_early,
    )
    return P1R52TargetDepthOuter(
        outer.target_step,
        outer.selected_endpoint,
        outer.next_state,
        outer.inner_steps,
        tuple(target_results),
        tuple(kl_results),
        outer.early_stop_byte_identical,
        outer.receipt,
    )


__all__ = [
    "P1R52_TARGET_DEPTH_INSTRUCTION_ID",
    "P1R52_TARGET_DEPTH_METHOD_ID",
    "P1R52_TARGET_DEPTH_EXTENSION_INSTRUCTION_ID",
    "P1R52_TARGET_DEPTH_EXTENSION_METHOD_ID",
    "P1R52_TARGET_DEPTH_INNER_COUNTS",
    "P1R52TargetDepth",
    "P1R52TargetDepthInner",
    "P1R52TargetDepthOuter",
    "reassemble_p1r52_outer_target",
    "requestwise_byte_identical_mask",
    "run_p1r52_target_depth_scheduler",
]
