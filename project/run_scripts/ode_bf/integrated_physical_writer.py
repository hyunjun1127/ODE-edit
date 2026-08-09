"""Integrated matrix-free physical-writer routing for P1R14.

This module is deliberately additive to the frozen R12/R13 implementations.
It owns the numerical coordinate, the three-stage lexicographic program, and
the online compute contract for
``FIXED_E8_DYNAMIC_BG_MATRIXFREE_PHYSICAL_WRITER_P_TIEBREAK_V1``.

The coefficient VJP coordinate is ``theta`` in
``W(theta) = W + sum_l theta_l B_l``.  The optimizer variable is the velocity
``v`` and the only physical step is ``theta = h * v``.  Consequently the VJP
graph never contains ``h``; predicted reductions and the authoritative factor
assembler each apply it exactly once.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .contracts import ODEBFContractError, canonical_hash
from .integrated_physical_writer_solver import (
    PHYSICAL_SOLVER_FTOL,
    PHYSICAL_SOLVER_KKT_TOLERANCE,
    PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
    PhysicalWriterConstraint,
    PhysicalWriterSolverCertificate,
    physical_writer_technical_constraints,
    solve_certified_physical_qp,
)
from .routing import RoutingProblem


INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1"
)
INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS = (
    "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1-A1",
    "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1-A2",
    "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1-A3",
)
INTEGRATED_PHYSICAL_WRITER_METHOD_ID = (
    "FIXED_E8_DYNAMIC_BG_MATRIXFREE_PHYSICAL_WRITER_P_TIEBREAK_V1"
)
INTEGRATED_PHYSICAL_WRITER_SCHEMA = (
    "ode-edit-s05-integrated-physical-writer/v1"
)

PHYSICAL_WRITER_LAYER_ORDER = (4, 5, 6, 7, 8)
PHYSICAL_WRITER_GRID_COUNT = 8
PHYSICAL_WRITER_H = 1.0 / 8.0
PHYSICAL_WRITER_CONTEXT_COUNT = 60
PHYSICAL_WRITER_MICROBATCH_SIZE = 10
PHYSICAL_WRITER_MICROBATCH_GROUPS = 6
PHYSICAL_WRITER_EPSILON_E = 1.0e-8
PHYSICAL_WRITER_EPSILON_P = 1.0e-8
PHYSICAL_WRITER_LAMBDA_TRANSPORT = 1.0
PHYSICAL_WRITER_ONLINE_FORWARD_GROUP_CEILING = 110
PHYSICAL_WRITER_FIXED_FORWARD_GROUP_LIMIT = 6


class PhysicalWriterStepStatus(str, Enum):
    """The exhaustive scientific step outcomes under the A2 contract."""

    JOINT_WRITE = "JOINT_WRITE"
    EDIT_DESCENT_INACTIVE_GOAL_MET = "EDIT_DESCENT_INACTIVE_GOAL_MET"
    ZERO_WRITE_GOAL_MET = "ZERO_WRITE_GOAL_MET"
    NO_PHYSICAL_W_ONLY_DIRECTION = "NO_PHYSICAL_W_ONLY_DIRECTION"


@dataclass(frozen=True, slots=True)
class PhysicalWriterComputeContract:
    """Static online forward-group contract, separate from model-call counts."""

    grid_steps: int
    controller_context_count: int
    microbatch_size: int
    microbatch_groups: int
    target_factor_groups_per_step: int
    writer_vjp_logical_groups_per_step: int
    writer_vjp_forward_groups_per_step: int
    writer_vjp_microbatch_graphs_per_step: int
    writer_vjp_autograd_invocations_per_step: int
    p_vjp_logical_groups_per_step: int
    p_vjp_forward_groups_per_step: int
    p_vjp_microbatch_graphs_per_step: int
    p_vjp_autograd_invocations_per_step: int
    online_forward_groups_per_step: int
    integration_online_forward_groups: int
    fixed_online_forward_groups: int
    total_online_forward_groups: int
    online_forward_group_ceiling: int
    terminal_evaluation_in_online_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def physical_writer_compute_contract(
    *, fixed_online_forward_groups: int
) -> PhysicalWriterComputeContract:
    """Return the sealed ``8 * (1 + 6 + 6) + fixed`` accounting.

    A VJP *logical group* is one semantic multi-output operation.  Each such
    group is implemented by six physical M=10 graphs and six autograd backend
    invocations.  The forward-group ceiling counts those physical microbatch
    groups, while the receipt keeps both views explicit.
    """

    fixed = int(fixed_online_forward_groups)
    if (
        isinstance(fixed_online_forward_groups, bool)
        or fixed != fixed_online_forward_groups
        or fixed != 0
    ):
        raise ODEBFContractError(
            "integrated physical writer fixed forward groups are not zero"
        )
    per_step = 1 + PHYSICAL_WRITER_MICROBATCH_GROUPS * 2
    integration = PHYSICAL_WRITER_GRID_COUNT * per_step
    total = integration + fixed
    if total >= PHYSICAL_WRITER_ONLINE_FORWARD_GROUP_CEILING:
        raise ODEBFContractError(
            "PRECHECKPOINT_COMPUTE_CEILING_BLOCKED: online forward groups"
        )
    payload = {
        "schema": f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-compute-contract",
        "grid_steps": PHYSICAL_WRITER_GRID_COUNT,
        "controller_context_count": PHYSICAL_WRITER_CONTEXT_COUNT,
        "microbatch_size": PHYSICAL_WRITER_MICROBATCH_SIZE,
        "microbatch_groups": PHYSICAL_WRITER_MICROBATCH_GROUPS,
        "target_factor_groups_per_step": 1,
        "writer_vjp_logical_groups_per_step": 1,
        "writer_vjp_forward_groups_per_step": (
            PHYSICAL_WRITER_MICROBATCH_GROUPS
        ),
        "writer_vjp_microbatch_graphs_per_step": (
            PHYSICAL_WRITER_MICROBATCH_GROUPS
        ),
        "writer_vjp_autograd_invocations_per_step": (
            PHYSICAL_WRITER_MICROBATCH_GROUPS
        ),
        "p_vjp_logical_groups_per_step": 1,
        "p_vjp_forward_groups_per_step": PHYSICAL_WRITER_MICROBATCH_GROUPS,
        "p_vjp_microbatch_graphs_per_step": PHYSICAL_WRITER_MICROBATCH_GROUPS,
        "p_vjp_autograd_invocations_per_step": (
            PHYSICAL_WRITER_MICROBATCH_GROUPS
        ),
        "online_forward_groups_per_step": per_step,
        "integration_online_forward_groups": integration,
        "fixed_online_forward_groups": fixed,
        "total_online_forward_groups": total,
        "online_forward_group_ceiling": (
            PHYSICAL_WRITER_ONLINE_FORWARD_GROUP_CEILING
        ),
        "terminal_evaluation_in_online_count": 0,
    }
    return PhysicalWriterComputeContract(
        **{
            key: value
            for key, value in payload.items()
            if key != "schema"
        },
        identity_sha256=canonical_hash(payload),
    )


class PhysicalWriterComputePhase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    PRODUCTION = "PRODUCTION"
    TERMINAL = "TERMINAL"


class PhysicalWriterForwardRole(str, Enum):
    TARGET_FACTOR = "SHARED_MULTIHOOK_DYNAMIC_TARGET_FACTOR"
    WRITER_VJP = "NOHOOK_WRITER_EDIT_TRANSPORT_M10"
    P_VJP = "SEALED_ANCHOR_P_VJP_M10"
    FIXED = "ONLINE_FIXED_OVERHEAD"
    BOOTSTRAP = "TARGET_ONLY_BG_BOOTSTRAP"
    TERMINAL = "POST_ACTION_FREEZE_TERMINAL_EVALUATION"


@dataclass(frozen=True, slots=True)
class PhysicalWriterGroupRecord:
    group_id: str
    phase: PhysicalWriterComputePhase
    role: PhysicalWriterForwardRole
    step_index: int | None
    microbatch_ordinal: int | None
    model_forward_calls: int
    physical_microbatch_graphs: int
    autograd_backend_invocations: int
    backward_calls: int
    processed_tokens: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["role"] = self.role.value
        return payload


class PhysicalWriterGroupLedger:
    """Append-only, phase-separated forward and autograd accounting."""

    def __init__(self) -> None:
        self._records: list[PhysicalWriterGroupRecord] = []
        self._ids: set[str] = set()
        self._action_freeze_sha256: str | None = None

    @property
    def records(self) -> tuple[PhysicalWriterGroupRecord, ...]:
        return tuple(self._records)

    def freeze_actions(self, action_freeze_sha256: str) -> None:
        if (
            self._action_freeze_sha256 is not None
            or not isinstance(action_freeze_sha256, str)
            or len(action_freeze_sha256) != 64
        ):
            raise ODEBFContractError(
                "integrated physical writer action freeze differs"
            )
        self._action_freeze_sha256 = action_freeze_sha256

    def record(
        self,
        *,
        group_id: str,
        phase: PhysicalWriterComputePhase | str,
        role: PhysicalWriterForwardRole | str,
        step_index: int | None,
        microbatch_ordinal: int | None,
        model_forward_calls: int,
        physical_microbatch_graphs: int,
        autograd_backend_invocations: int,
        backward_calls: int,
        processed_tokens: int,
    ) -> PhysicalWriterGroupRecord:
        selected_phase = PhysicalWriterComputePhase(phase)
        selected_role = PhysicalWriterForwardRole(role)
        if not isinstance(group_id, str) or not group_id or group_id in self._ids:
            raise ODEBFContractError(
                "integrated physical writer compute group identity differs"
            )
        if selected_phase is PhysicalWriterComputePhase.TERMINAL and (
            self._action_freeze_sha256 is None
        ):
            raise ODEBFContractError(
                "integrated physical writer terminal ledger opened pre-freeze"
            )
        if self._action_freeze_sha256 is not None and (
            selected_phase is not PhysicalWriterComputePhase.TERMINAL
        ):
            raise ODEBFContractError(
                "integrated physical writer action changed after freeze"
            )
        if selected_phase is PhysicalWriterComputePhase.PRODUCTION:
            if selected_role in (
                PhysicalWriterForwardRole.WRITER_VJP,
                PhysicalWriterForwardRole.P_VJP,
            ):
                if (
                    step_index is None
                    or not 0 <= step_index < PHYSICAL_WRITER_GRID_COUNT
                    or
                    microbatch_ordinal is None
                    or not 0
                    <= microbatch_ordinal
                    < PHYSICAL_WRITER_MICROBATCH_GROUPS
                ):
                    raise ODEBFContractError(
                        "integrated physical writer microbatch differs"
                    )
            elif selected_role is PhysicalWriterForwardRole.TARGET_FACTOR:
                if (
                    step_index is None
                    or not 0 <= step_index < PHYSICAL_WRITER_GRID_COUNT
                    or microbatch_ordinal is not None
                ):
                    raise ODEBFContractError(
                        "integrated physical writer target group is split"
                    )
            elif selected_role is PhysicalWriterForwardRole.FIXED:
                if step_index is not None or microbatch_ordinal is not None:
                    raise ODEBFContractError(
                        "integrated physical writer fixed group differs"
                    )
            else:
                raise ODEBFContractError(
                    "integrated physical writer production role differs"
                )
        elif selected_phase is PhysicalWriterComputePhase.BOOTSTRAP:
            if selected_role is not PhysicalWriterForwardRole.BOOTSTRAP:
                raise ODEBFContractError(
                    "integrated physical writer bootstrap role differs"
                )
        elif selected_role is not PhysicalWriterForwardRole.TERMINAL:
            raise ODEBFContractError(
                "integrated physical writer terminal role differs"
            )
        counters = (
            model_forward_calls,
            physical_microbatch_graphs,
            autograd_backend_invocations,
            backward_calls,
            processed_tokens,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counters
        ):
            raise ODEBFContractError(
                "integrated physical writer compute counter differs"
            )
        if selected_phase is PhysicalWriterComputePhase.PRODUCTION:
            if selected_role in (
                PhysicalWriterForwardRole.WRITER_VJP,
                PhysicalWriterForwardRole.P_VJP,
            ) and (
                model_forward_calls != 1
                or physical_microbatch_graphs != 1
                or autograd_backend_invocations != 1
                or backward_calls != 1
                or processed_tokens <= 0
            ):
                raise ODEBFContractError(
                    "integrated physical writer physical VJP accounting differs"
                )
            if selected_role is PhysicalWriterForwardRole.TARGET_FACTOR and (
                model_forward_calls <= 0
                or physical_microbatch_graphs != 1
                or autograd_backend_invocations <= 0
                or backward_calls <= 0
                or processed_tokens <= 0
            ):
                raise ODEBFContractError(
                    "integrated physical writer target/factor accounting differs"
                )
            if selected_role is PhysicalWriterForwardRole.FIXED and (
                model_forward_calls <= 0
                or physical_microbatch_graphs != 1
                or autograd_backend_invocations != 0
                or backward_calls != 0
                or processed_tokens <= 0
            ):
                raise ODEBFContractError(
                    "integrated physical writer fixed parity accounting differs"
                )
        payload = {
            "schema": f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-group-record",
            "group_id": group_id,
            "phase": selected_phase.value,
            "role": selected_role.value,
            "step_index": step_index,
            "microbatch_ordinal": microbatch_ordinal,
            "model_forward_calls": model_forward_calls,
            "physical_microbatch_graphs": physical_microbatch_graphs,
            "autograd_backend_invocations": autograd_backend_invocations,
            "backward_calls": backward_calls,
            "processed_tokens": processed_tokens,
        }
        receipt = PhysicalWriterGroupRecord(
            group_id,
            selected_phase,
            selected_role,
            step_index,
            microbatch_ordinal,
            model_forward_calls,
            physical_microbatch_graphs,
            autograd_backend_invocations,
            backward_calls,
            processed_tokens,
            canonical_hash(payload),
        )
        self._ids.add(group_id)
        self._records.append(receipt)
        return receipt

    def validate_production(
        self,
        *,
        fixed_online_forward_groups: int,
        completed_steps: int = PHYSICAL_WRITER_GRID_COUNT,
        attempted_field_count: int | None = None,
    ) -> dict[str, Any]:
        contract = physical_writer_compute_contract(
            fixed_online_forward_groups=fixed_online_forward_groups
        )
        attempted = (
            completed_steps
            if attempted_field_count is None
            else int(attempted_field_count)
        )
        if (
            completed_steps < 0
            or completed_steps > PHYSICAL_WRITER_GRID_COUNT
            or attempted < completed_steps
            or attempted > PHYSICAL_WRITER_GRID_COUNT
            or attempted - completed_steps > 1
        ):
            raise ODEBFContractError(
                "integrated physical writer field/transition counts differ"
            )
        production = tuple(
            item
            for item in self._records
            if item.phase is PhysicalWriterComputePhase.PRODUCTION
        )
        if any(
            item.step_index is not None and item.step_index >= attempted
            for item in production
        ):
            raise ODEBFContractError(
                "integrated physical writer ledger exceeds attempted fields"
            )
        for step in range(attempted):
            step_records = tuple(
                item for item in production if item.step_index == step
            )
            by_role = {
                role: tuple(item for item in step_records if item.role is role)
                for role in (
                    PhysicalWriterForwardRole.TARGET_FACTOR,
                    PhysicalWriterForwardRole.WRITER_VJP,
                    PhysicalWriterForwardRole.P_VJP,
                )
            }
            if (
                len(by_role[PhysicalWriterForwardRole.TARGET_FACTOR]) != 1
                or len(by_role[PhysicalWriterForwardRole.WRITER_VJP])
                != PHYSICAL_WRITER_MICROBATCH_GROUPS
                or len(by_role[PhysicalWriterForwardRole.P_VJP])
                != PHYSICAL_WRITER_MICROBATCH_GROUPS
                or {
                    item.microbatch_ordinal
                    for item in by_role[PhysicalWriterForwardRole.WRITER_VJP]
                }
                != set(range(PHYSICAL_WRITER_MICROBATCH_GROUPS))
                or {
                    item.microbatch_ordinal
                    for item in by_role[PhysicalWriterForwardRole.P_VJP]
                }
                != set(range(PHYSICAL_WRITER_MICROBATCH_GROUPS))
            ):
                raise ODEBFContractError(
                    "PRECHECKPOINT_COMPUTE_CEILING_BLOCKED: step groups differ"
                )
        fixed = tuple(
            item
            for item in production
            if item.role is PhysicalWriterForwardRole.FIXED
        )
        if len(fixed) != fixed_online_forward_groups:
            raise ODEBFContractError(
                "PRECHECKPOINT_COMPUTE_CEILING_BLOCKED: fixed groups differ"
            )
        expected = attempted * 13 + fixed_online_forward_groups
        observed = sum(
            item.physical_microbatch_graphs
            if item.role
            in (
                PhysicalWriterForwardRole.WRITER_VJP,
                PhysicalWriterForwardRole.P_VJP,
            )
            else 1
            for item in production
        )
        # Each microbatch is persisted as its own record, so its physical graph
        # count must be one.  The explicit sum catches hidden endpoint work.
        if observed != expected or any(
            item.physical_microbatch_graphs != 1 for item in production
        ):
            raise ODEBFContractError(
                "PRECHECKPOINT_COMPUTE_CEILING_BLOCKED: online groups differ"
            )
        payload = {
            "schema": f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-group-ledger",
            "attempted_field_count": attempted,
            "accepted_transition_count": completed_steps,
            "typed_pre_guard_boundary_count": attempted - completed_steps,
            "production_forward_group_count": observed,
            "fixed_online_forward_group_count": len(fixed),
            "writer_vjp_logical_group_count": attempted,
            "writer_vjp_physical_microbatch_graph_count": (
                attempted * PHYSICAL_WRITER_MICROBATCH_GROUPS
            ),
            "writer_vjp_autograd_backend_invocation_count": sum(
                item.autograd_backend_invocations
                for item in production
                if item.role is PhysicalWriterForwardRole.WRITER_VJP
            ),
            "p_vjp_logical_group_count": attempted,
            "p_vjp_physical_microbatch_graph_count": (
                attempted * PHYSICAL_WRITER_MICROBATCH_GROUPS
            ),
            "p_vjp_autograd_backend_invocation_count": sum(
                item.autograd_backend_invocations
                for item in production
                if item.role is PhysicalWriterForwardRole.P_VJP
            ),
            "model_forward_call_count": sum(
                item.model_forward_calls for item in production
            ),
            "backward_call_count": sum(
                item.backward_calls for item in production
            ),
            "processed_token_count": sum(
                item.processed_tokens for item in production
            ),
            "terminal_group_count": sum(
                item.phase is PhysicalWriterComputePhase.TERMINAL
                for item in self._records
            ),
            "bootstrap_group_count": sum(
                item.phase is PhysicalWriterComputePhase.BOOTSTRAP
                for item in self._records
            ),
            "terminal_open_before_action_freeze_count": 0,
            "contract_sha256": contract.identity_sha256,
            "records": [item.raw_free_payload() for item in self._records],
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class TwoOutputVJP:
    """One M=10 writer microbatch's two-output coefficient VJP."""

    loss_new: float
    loss_transport: float
    a_edit: tuple[float, ...]
    a_transport: tuple[float, ...]
    theta_coordinate: str
    h_in_vjp_graph_count: int
    logical_group_count: int
    physical_microbatch_graph_count: int
    autograd_backend_invocation_count: int
    backward_call_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def batched_two_output_coefficient_vjp(
    loss_new: torch.Tensor,
    loss_transport: torch.Tensor,
    theta: torch.Tensor,
) -> TwoOutputVJP:
    """Differentiate two scalar losses with one batched autograd invocation.

    ``theta`` is the unscaled physical coefficient coordinate.  The caller
    accumulates six such M=10 receipts into the single writer VJP logical
    group for a complete 10x6 controller field.
    """

    if (
        not isinstance(loss_new, torch.Tensor)
        or not isinstance(loss_transport, torch.Tensor)
        or loss_new.ndim != 0
        or loss_transport.ndim != 0
        or not isinstance(theta, torch.Tensor)
        or theta.ndim != 1
        or theta.numel() != len(PHYSICAL_WRITER_LAYER_ORDER)
        or not theta.requires_grad
        or not loss_new.requires_grad
        or not loss_transport.requires_grad
    ):
        raise ODEBFContractError("integrated physical writer VJP input differs")
    if not bool(
        torch.isfinite(loss_new.detach()).all()
        and torch.isfinite(loss_transport.detach()).all()
        and torch.isfinite(theta.detach()).all()
    ):
        raise ODEBFContractError("integrated physical writer VJP is non-finite")
    outputs = torch.stack((loss_new, loss_transport))
    basis = torch.eye(2, dtype=outputs.dtype, device=outputs.device)
    gradient = torch.autograd.grad(
        outputs,
        theta,
        grad_outputs=basis,
        retain_graph=False,
        create_graph=False,
        is_grads_batched=True,
    )[0]
    if gradient.shape != (2, theta.numel()) or not bool(
        torch.isfinite(gradient).all()
    ):
        raise ODEBFContractError("integrated physical writer VJP shape differs")
    edit = -gradient[0].detach().to(device="cpu", dtype=torch.float64)
    transport = -gradient[1].detach().to(
        device="cpu", dtype=torch.float64
    )
    payload = {
        "schema": f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-two-output-vjp",
        "loss_new": float(loss_new.detach().to(device="cpu", dtype=torch.float64)),
        "loss_transport": float(
            loss_transport.detach().to(device="cpu", dtype=torch.float64)
        ),
        "a_edit": [float(item) for item in edit],
        "a_transport": [float(item) for item in transport],
        "theta_coordinate": "W(theta)=W+sum_l(theta_l*B_l)",
        "h_in_vjp_graph_count": 0,
        "logical_group_count": 1,
        "physical_microbatch_graph_count": 1,
        "autograd_backend_invocation_count": 1,
        "backward_call_count": 1,
    }
    converted = {
        key: value for key, value in payload.items() if key != "schema"
    }
    converted["a_edit"] = tuple(payload["a_edit"])
    converted["a_transport"] = tuple(payload["a_transport"])
    converted["identity_sha256"] = canonical_hash(payload)
    return TwoOutputVJP(**converted)


@dataclass(frozen=True, slots=True)
class PhysicalWriterRoutingResult:
    """Complete A1/A2 lexicographic selection receipt for one grid state."""

    status: PhysicalWriterStepStatus
    step_index: int
    a_edit: tuple[float, ...]
    a_transport: tuple[float, ...]
    a_edit_normalized: tuple[float, ...]
    a_transport_normalized: tuple[float, ...]
    a_physical: tuple[float, ...]
    p_edit: float
    p_transport: float
    current_nohook_loss: float
    goal_loss: float
    deficit: float
    requested_applied_reduction: float
    velocity: tuple[float, ...]
    applied_theta: tuple[float, ...]
    e_star: float | None
    p_proxy_derivative_step: tuple[float, ...]
    p_structural_matrix_step: tuple[tuple[float, ...], ...]
    p_scale: float
    p_bar_star: float | None
    edit_descent_status: str
    transport_descent_status: str
    p_tie_status: str
    clock_advance_count: int
    candidate_count: int
    all_five_layer_domain: bool
    overlay_access_count: int
    h_in_vjp_graph_count: int
    h_in_applied_theta_count: int
    certificates: tuple[PhysicalWriterSolverCertificate, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["certificates"] = [
            item.raw_free_payload() for item in self.certificates
        ]
        return payload


def _array(
    value: Sequence[float] | np.ndarray,
    *,
    label: str,
    nonnegative: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (len(PHYSICAL_WRITER_LAYER_ORDER),)
        or not np.all(np.isfinite(result))
        or (nonnegative and np.any(result < 0.0))
    ):
        raise ODEBFContractError(
            f"integrated physical writer {label} differs"
        )
    return result.copy()


def _matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    dimension = len(PHYSICAL_WRITER_LAYER_ORDER)
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (dimension, dimension)
        or not np.all(np.isfinite(result))
        or not np.allclose(result, result.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ODEBFContractError(
            f"integrated physical writer {label} differs"
        )
    if float(np.min(np.linalg.eigvalsh(result))) < -1.0e-10:
        raise ODEBFContractError(
            f"integrated physical writer {label} is not PSD"
        )
    return result.copy()


def physical_theta_from_velocity(
    velocity: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply the sole scientific ``h`` to a certified velocity."""

    value = _array(velocity, label="velocity")
    return (PHYSICAL_WRITER_H * value).astype(np.float64, copy=False)


def _constraint(
    name: str,
    function: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
) -> PhysicalWriterConstraint:
    return PhysicalWriterConstraint(name, function, jacobian, hessian)


def _observe(
    observer: Callable[[PhysicalWriterSolverCertificate], None] | None,
    certificate: PhysicalWriterSolverCertificate,
) -> None:
    if observer is not None:
        observer(certificate)


def _routing_result(
    payload: Mapping[str, Any],
    certificates: Sequence[PhysicalWriterSolverCertificate],
    identity_sha256: str,
) -> PhysicalWriterRoutingResult:
    converted = dict(payload)
    converted["status"] = PhysicalWriterStepStatus(str(payload["status"]))
    for name in (
        "a_edit",
        "a_transport",
        "a_edit_normalized",
        "a_transport_normalized",
        "a_physical",
        "velocity",
        "applied_theta",
        "p_proxy_derivative_step",
    ):
        converted[name] = tuple(float(item) for item in payload[name])
    converted["p_structural_matrix_step"] = tuple(
        tuple(float(item) for item in row)
        for row in payload["p_structural_matrix_step"]
    )
    converted["certificates"] = tuple(certificates)
    converted["identity_sha256"] = identity_sha256
    return PhysicalWriterRoutingResult(**converted)


def solve_integrated_physical_writer(
    problem: RoutingProblem,
    *,
    a_edit: Sequence[float] | np.ndarray,
    a_transport: Sequence[float] | np.ndarray,
    current_nohook_loss: float,
    goal_loss: float,
    step_index: int,
    functional_p_derivative: Sequence[float] | np.ndarray,
    structural_p_matrix_raw: Sequence[Sequence[float]] | np.ndarray,
    certificate_observer: Callable[[PhysicalWriterSolverCertificate], None] | None = None,
    pre_guard_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> PhysicalWriterRoutingResult:
    """Solve the A1/A2 E -> P -> capacity program over all five layers."""

    if step_index < 0 or step_index >= PHYSICAL_WRITER_GRID_COUNT:
        raise ODEBFContractError("integrated physical writer step differs")
    if not math.isfinite(current_nohook_loss) or not math.isfinite(goal_loss):
        raise ODEBFContractError("integrated physical writer loss differs")
    if problem.signed_progress.shape != (len(PHYSICAL_WRITER_LAYER_ORDER),):
        raise ODEBFContractError("integrated physical writer layer domain differs")
    edit = _array(a_edit, label="edit slope")
    transport = _array(a_transport, label="transport slope")
    p_derivative = _array(
        functional_p_derivative,
        label="functional P derivative",
    )
    p_raw = _matrix(structural_p_matrix_raw, label="structural P matrix")
    # ``RoutingProblem`` is retained solely for the inherited technical
    # caps/trust/capacity geometry and independent certificate machinery.
    certificate_problem = replace(problem, signed_progress=edit)
    transport_certificate_problem = replace(problem, signed_progress=transport)
    caps = np.minimum(certificate_problem.layer_caps, 1.0)
    lower = np.zeros_like(caps)
    technical = physical_writer_technical_constraints(
        certificate_problem,
        requested_progress=None,
    )
    zero_hessian = lambda value: np.zeros(  # noqa: E731
        (value.size, value.size), dtype=np.float64
    )

    maximum_edit, edit_certificate = solve_certified_physical_qp(
        phase="maximum-physical-edit",
        problem=certificate_problem,
        objective=lambda value: float(-edit @ value),
        jacobian=lambda value: -edit.copy(),
        hessian=zero_hessian,
        constraints=technical,
        lower=lower,
        upper=caps,
        initial=np.zeros_like(caps),
        p_max=0.0,
        requested_progress=0.0,
        authority_role="AUTHORITATIVE_PHYSICAL_EDIT_MAXIMUM",
    )
    _observe(certificate_observer, edit_certificate)
    p_edit = float(edit @ maximum_edit)
    if p_edit < -PHYSICAL_SOLVER_PRIMAL_TOLERANCE or not math.isfinite(p_edit):
        raise ODEBFContractError("integrated physical edit maximum differs")
    p_edit = max(p_edit, 0.0)

    maximum_transport, transport_certificate = solve_certified_physical_qp(
        phase="maximum-physical-transport",
        problem=transport_certificate_problem,
        objective=lambda value: float(-transport @ value),
        jacobian=lambda value: -transport.copy(),
        hessian=zero_hessian,
        constraints=technical,
        lower=lower,
        upper=caps,
        initial=np.zeros_like(caps),
        p_max=p_edit,
        requested_progress=0.0,
        authority_role="AUTHORITATIVE_TRANSPORT_MAXIMUM",
    )
    _observe(certificate_observer, transport_certificate)
    p_transport = float(transport @ maximum_transport)
    if p_transport < -PHYSICAL_SOLVER_PRIMAL_TOLERANCE or not math.isfinite(
        p_transport
    ):
        raise ODEBFContractError("integrated transport maximum differs")
    p_transport = max(p_transport, 0.0)

    deficit = max(float(current_nohook_loss - goal_loss), 0.0)
    remaining_steps = PHYSICAL_WRITER_GRID_COUNT - step_index
    requested = min(
        PHYSICAL_WRITER_H * p_edit,
        deficit / float(remaining_steps),
    )
    certificates: list[PhysicalWriterSolverCertificate] = [
        edit_certificate,
        transport_certificate,
    ]
    edit_active = p_edit > 0.0
    transport_active = p_transport > 0.0
    edit_bar = edit / p_edit if edit_active else np.zeros_like(edit)
    transport_bar = (
        transport / p_transport
        if transport_active
        else np.zeros_like(transport)
    )
    physical = edit_bar + PHYSICAL_WRITER_LAMBDA_TRANSPORT * transport_bar

    if pre_guard_observer is not None:
        pre_guard_observer(
            {
                "schema": (
                    f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-pre-physical-guard"
                ),
                "step_index": step_index,
                "a_edit": edit.tolist(),
                "a_transport": transport.tolist(),
                "p_edit": p_edit,
                "p_transport": p_transport,
                "current_nohook_loss": float(current_nohook_loss),
                "goal_loss": float(goal_loss),
                "deficit": deficit,
                "requested_applied_reduction": requested,
                "all_five_layer_domain": True,
                "overlay_access_count": 0,
                "observer_decision_influence_count": 0,
            }
        )

    if deficit > 0.0 and not edit_active:
        velocity = np.zeros_like(edit)
        theta = physical_theta_from_velocity(velocity)
        payload = {
            "status": PhysicalWriterStepStatus.NO_PHYSICAL_W_ONLY_DIRECTION.value,
            "step_index": step_index,
            "a_edit": edit.tolist(),
            "a_transport": transport.tolist(),
            "a_edit_normalized": edit_bar.tolist(),
            "a_transport_normalized": transport_bar.tolist(),
            "a_physical": physical.tolist(),
            "p_edit": p_edit,
            "p_transport": p_transport,
            "current_nohook_loss": current_nohook_loss,
            "goal_loss": goal_loss,
            "deficit": deficit,
            "requested_applied_reduction": requested,
            "velocity": velocity.tolist(),
            "applied_theta": theta.tolist(),
            "e_star": None,
            "p_proxy_derivative_step": (
                PHYSICAL_WRITER_H * np.maximum(p_derivative, 0.0)
            ).tolist(),
            "p_structural_matrix_step": (
                PHYSICAL_WRITER_H**2 * p_raw
            ).tolist(),
            "p_scale": 0.0,
            "p_bar_star": None,
            "edit_descent_status": "NO_PHYSICAL_W_ONLY_DIRECTION",
            "transport_descent_status": (
                "ACTIVE" if transport_active else "TRANSPORT_DESCENT_INACTIVE"
            ),
            "p_tie_status": "NOT_REACHED",
            "clock_advance_count": 0,
            "candidate_count": 0,
            "all_five_layer_domain": True,
            "overlay_access_count": 0,
            "h_in_vjp_graph_count": 0,
            "h_in_applied_theta_count": 1,
        }
        identity = canonical_hash(
            {
                **payload,
                "certificates": [
                    item.raw_free_payload() for item in certificates
                ],
            }
        )
        return _routing_result(payload, certificates, identity)

    progress_constraint = _constraint(
        "physical_edit_applied_progress",
        lambda value: float(PHYSICAL_WRITER_H * edit @ value - requested),
        lambda value: PHYSICAL_WRITER_H * edit.copy(),
        zero_hessian,
    )
    feasible = [*technical, progress_constraint]
    if edit_active:
        scale = (
            requested / (PHYSICAL_WRITER_H * p_edit)
            if requested > 0.0
            else 0.0
        )
        initial = maximum_edit * scale
    else:
        initial = np.zeros_like(edit)

    e_objective = lambda value: float(-PHYSICAL_WRITER_H * physical @ value)
    e_gradient = lambda value: -PHYSICAL_WRITER_H * physical.copy()
    stage1, e_certificate = solve_certified_physical_qp(
        phase="stage1-minimum-dimensionless-physical-transport-E",
        problem=certificate_problem,
        objective=e_objective,
        jacobian=e_gradient,
        hessian=zero_hessian,
        constraints=feasible,
        lower=lower,
        upper=caps,
        initial=initial,
        p_max=p_edit,
        requested_progress=requested,
        authority_role="AUTHORITATIVE_E_STAGE",
    )
    _observe(certificate_observer, e_certificate)
    certificates.append(e_certificate)
    e_star = float(e_objective(stage1))
    e_tie = _constraint(
        "dimensionless_E_tie",
        lambda value: float(
            e_star + PHYSICAL_WRITER_EPSILON_E - e_objective(value)
        ),
        lambda value: PHYSICAL_WRITER_H * physical.copy(),
        zero_hessian,
    )

    pi = PHYSICAL_WRITER_H * np.maximum(p_derivative, 0.0)
    m_step = PHYSICAL_WRITER_H**2 * p_raw
    p_scale = float(pi @ caps + caps @ np.abs(m_step) @ caps)
    p_active = p_scale > 0.0

    def p_bar(value: np.ndarray) -> float:
        if not p_active:
            return 0.0
        return float((pi @ value + value @ m_step @ value) / p_scale)

    def p_gradient(value: np.ndarray) -> np.ndarray:
        if not p_active:
            return np.zeros_like(value)
        return (pi + 2.0 * m_step @ value) / p_scale

    def p_hessian(value: np.ndarray) -> np.ndarray:
        del value
        if not p_active:
            return np.zeros_like(m_step)
        return 2.0 * m_step / p_scale

    stage2, p_certificate = solve_certified_physical_qp(
        phase="stage2-minimum-dimensionless-P-tie",
        problem=certificate_problem,
        objective=p_bar,
        jacobian=p_gradient,
        hessian=p_hessian,
        constraints=[*feasible, e_tie],
        lower=lower,
        upper=caps,
        initial=stage1,
        p_max=p_edit,
        requested_progress=requested,
        authority_role="AUTHORITATIVE_P_TIE_STAGE",
    )
    _observe(certificate_observer, p_certificate)
    certificates.append(p_certificate)
    p_star = float(p_bar(stage2))
    p_tie = _constraint(
        "dimensionless_P_tie",
        lambda value: float(
            p_star + PHYSICAL_WRITER_EPSILON_P - p_bar(value)
        ),
        lambda value: -p_gradient(value),
        lambda value: -p_hessian(value),
    )

    capacity = certificate_problem.capacity_metric
    stage3, capacity_certificate = solve_certified_physical_qp(
        phase="stage3-minimum-capacity-within-E-P-ties",
        problem=certificate_problem,
        objective=lambda value: float(0.5 * value @ capacity @ value),
        jacobian=lambda value: capacity @ value,
        hessian=lambda value: capacity.copy(),
        constraints=[*feasible, e_tie, p_tie],
        lower=lower,
        upper=caps,
        initial=stage2,
        p_max=p_edit,
        requested_progress=requested,
        authority_role="AUTHORITATIVE_CAPACITY_TIE_STAGE",
    )
    _observe(certificate_observer, capacity_certificate)
    certificates.append(capacity_certificate)
    theta = physical_theta_from_velocity(stage3)
    if not np.array_equal(theta, PHYSICAL_WRITER_H * stage3):
        raise ODEBFContractError(
            "integrated physical writer applied theta double-h differs"
        )
    zero_write = bool(np.count_nonzero(stage3) == 0)
    if not edit_active:
        status = (
            PhysicalWriterStepStatus.ZERO_WRITE_GOAL_MET
            if zero_write
            else PhysicalWriterStepStatus.EDIT_DESCENT_INACTIVE_GOAL_MET
        )
        edit_status = "EDIT_DESCENT_INACTIVE_GOAL_MET"
    else:
        status = PhysicalWriterStepStatus.JOINT_WRITE
        edit_status = "ACTIVE"
    payload = {
        "status": status.value,
        "step_index": step_index,
        "a_edit": edit.tolist(),
        "a_transport": transport.tolist(),
        "a_edit_normalized": edit_bar.tolist(),
        "a_transport_normalized": transport_bar.tolist(),
        "a_physical": physical.tolist(),
        "p_edit": p_edit,
        "p_transport": p_transport,
        "current_nohook_loss": float(current_nohook_loss),
        "goal_loss": float(goal_loss),
        "deficit": deficit,
        "requested_applied_reduction": requested,
        "velocity": stage3.tolist(),
        "applied_theta": theta.tolist(),
        "e_star": e_star,
        "p_proxy_derivative_step": pi.tolist(),
        "p_structural_matrix_step": m_step.tolist(),
        "p_scale": p_scale,
        "p_bar_star": p_star,
        "edit_descent_status": edit_status,
        "transport_descent_status": (
            "ACTIVE" if transport_active else "TRANSPORT_DESCENT_INACTIVE"
        ),
        "p_tie_status": "ACTIVE" if p_active else "P_TIE_INACTIVE",
        "clock_advance_count": 1,
        "candidate_count": 0 if zero_write else 1,
        "all_five_layer_domain": True,
        "overlay_access_count": 0,
        "h_in_vjp_graph_count": 0,
        "h_in_applied_theta_count": 1,
    }
    identity = canonical_hash(
        {
            **payload,
            "certificates": [item.raw_free_payload() for item in certificates],
        }
    )
    return _routing_result(payload, certificates, identity)


def integrated_physical_writer_source_contract() -> dict[str, Any]:
    payload = {
        "schema": f"{INTEGRATED_PHYSICAL_WRITER_SCHEMA}-source-contract",
        "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
        "amendment_ids": list(INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS),
        "method_id": INTEGRATED_PHYSICAL_WRITER_METHOD_ID,
        "layer_order": list(PHYSICAL_WRITER_LAYER_ORDER),
        "grid_count": PHYSICAL_WRITER_GRID_COUNT,
        "h": PHYSICAL_WRITER_H,
        "coefficient_coordinate": "W(theta)=W+sum_l(theta_l*B_l)",
        "optimizer_variable": "v",
        "applied_coefficient": "theta=h*v",
        "h_in_vjp_graph_count": 0,
        "h_in_physical_assembler_count": 1,
        "all_five_layer_domain": True,
        "lambda_transport": PHYSICAL_WRITER_LAMBDA_TRANSPORT,
        "epsilon_E": PHYSICAL_WRITER_EPSILON_E,
        "epsilon_E_units": "DIMENSIONLESS_E_ABSOLUTE_FP64",
        "epsilon_P": PHYSICAL_WRITER_EPSILON_P,
        "epsilon_P_units": "DIMENSIONLESS_P_BAR_ABSOLUTE_FP64",
        "solver_ftol": PHYSICAL_SOLVER_FTOL,
        "solver_primal_tolerance": PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
        "solver_kkt_tolerance": PHYSICAL_SOLVER_KKT_TOLERANCE,
        "overlay_guard_objective_termination_access_count": 0,
        "hard_h_p_budget_veto_retry_count": 0,
        "native_or_direct_z_controller_access_count": 0,
        "committed_load_semantics": "PERSISTENT_OUTER_HISTORY_ONLY",
        "outer_history_item_count": 0,
        "current_virtual_factor_load_in_committed_load_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS",
    "INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID",
    "INTEGRATED_PHYSICAL_WRITER_METHOD_ID",
    "PHYSICAL_WRITER_EPSILON_E",
    "PHYSICAL_WRITER_EPSILON_P",
    "PHYSICAL_WRITER_GRID_COUNT",
    "PHYSICAL_WRITER_H",
    "PHYSICAL_WRITER_LAYER_ORDER",
    "PhysicalWriterComputeContract",
    "PhysicalWriterComputePhase",
    "PhysicalWriterForwardRole",
    "PhysicalWriterGroupLedger",
    "PhysicalWriterGroupRecord",
    "PhysicalWriterRoutingResult",
    "PhysicalWriterStepStatus",
    "TwoOutputVJP",
    "batched_two_output_coefficient_vjp",
    "integrated_physical_writer_source_contract",
    "physical_theta_from_velocity",
    "physical_writer_compute_contract",
    "solve_integrated_physical_writer",
]
