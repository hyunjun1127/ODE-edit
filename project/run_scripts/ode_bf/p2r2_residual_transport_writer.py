"""P2R2 request-conserving residual transport writer and lexicographic routers.

The module is intentionally independent of experiment launchers.  It consumes
one current-state P2R1 target residual field, measures the exact request by
request/layer physical response with a batched VJP, and returns one certified
joint allocation.  It does not own a target policy, held-out evaluator, or
weight transaction.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linprog, minimize

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p1_backend import P1DynamicField
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .scalable_batched_model import (
    ScalableObjectivePlan,
    _parameter_inventory_sha256,
)
from .target_new_nll import _score_prepared_target_new_batch


P2R2_INSTRUCTION_ID = (
    "ODEEDIT-S05-P2R2-P2R1-SEMANTIC-CONSERVING-RESIDUAL-TRANSPORT-WRITER-V1"
)
P2R2_METHOD_ID = "P2R2-SEMANTIC-CONSERVING-RESIDUAL-TRANSPORT-WRITER-V1"
P2R2_LAYER_COUNT = 5
P2R2_REQUEST_COUNT = 10
P2R2_ALPHA_COUNT = P2R2_LAYER_COUNT * P2R2_REQUEST_COUNT
P2R2_NUMERICAL_EPSILON = SIMPLEX_PRIMAL_TOLERANCE


def _unwrap(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise ODEBFContractError("P2R2 hooked activation differs")


class RequestLayerCoefficientOverlay:
    """Apply ``sum_i alpha[l,i] R_i Q_i^T`` at every rewrite layer."""

    def __init__(
        self,
        model: torch.nn.Module,
        field: P1DynamicField,
        coefficients: torch.Tensor,
    ) -> None:
        if (
            coefficients.ndim != 2
            or coefficients.shape != (len(field.layers), field.current_z.shape[1])
            or not coefficients.requires_grad
            or not bool(torch.isfinite(coefficients).all())
        ):
            raise ODEBFContractError("P2R2 request/layer coefficient geometry differs")
        self.model = model
        self.field = field
        self.coefficients = coefficients
        self._handles: list[Any] = []
        self.fire_count = [0] * len(field.layers)

    def __enter__(self) -> "RequestLayerCoefficientOverlay":
        for ordinal, layer in enumerate(self.field.layers):
            module_name = layer.weight_name.removesuffix(".weight")
            module = self.model.get_submodule(module_name)

            def hook(
                _module: Any,
                inputs: Any,
                output: Any,
                *,
                layer_ordinal: int = ordinal,
                layer_field: Any = layer,
            ) -> Any:
                raw = _unwrap(output)
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise ODEBFContractError("P2R2 rewrite input differs")
                hidden = inputs[0]
                if hidden.shape[:-1] != raw.shape[:-1]:
                    raise ODEBFContractError("P2R2 rewrite input/output axes differ")
                q = layer_field.q.to(device=hidden.device, dtype=torch.float32)
                residual = layer_field.residual.to(device=hidden.device, dtype=torch.float32)
                coefficient = self.coefficients[layer_ordinal].to(
                    device=hidden.device, dtype=torch.float32
                )
                if (
                    q.ndim != 2
                    or residual.ndim != 2
                    or q.shape[1] != coefficient.numel()
                    or residual.shape[1] != coefficient.numel()
                    or hidden.shape[-1] != q.shape[0]
                    or raw.shape[-1] != residual.shape[0]
                ):
                    raise ODEBFContractError("P2R2 Alpha unit proposal geometry differs")
                projected = hidden.to(torch.float32) @ q
                delta = (projected * coefficient) @ residual.T
                replaced = raw + delta.to(dtype=raw.dtype)
                self.fire_count[layer_ordinal] += 1
                if isinstance(output, tuple):
                    return (replaced, *output[1:])
                return replaced

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


@dataclass(frozen=True, slots=True)
class PhysicalResponseReceipt:
    response: torch.Tensor
    current_nll_by_request: tuple[float, ...]
    model_forward_count: int
    batched_vjp_count: int
    processed_token_count: int
    overlay_fire_count: tuple[int, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r2-request-layer-physical-response/v1",
            "response_shape": list(self.response.shape),
            "response_sha256": tensor_sha256(self.response),
            "current_nll_by_request": list(self.current_nll_by_request),
            "model_forward_count": self.model_forward_count,
            "batched_vjp_count": self.batched_vjp_count,
            "candidate_forward_count": 0,
            "candidate_materialization_count": 0,
            "processed_token_count": self.processed_token_count,
            "overlay_fire_count": list(self.overlay_fire_count),
            "request_axis": 0,
            "alpha_axis": 1,
            "alpha_axis_order": "LAYER_MAJOR_THEN_REQUEST",
            "identity_sha256": self.identity_sha256,
        }


def measure_request_layer_response(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    field: P1DynamicField,
) -> PhysicalResponseReceipt:
    """Return authoritative ``S=-d L_i / d alpha_(l,j)`` in one VJP per microbatch."""

    layer_count = len(field.layers)
    request_count = plan.request_count
    if layer_count != P2R2_LAYER_COUNT or request_count != P2R2_REQUEST_COUNT:
        raise ODEBFContractError("P2R2 response inventory differs")
    if field.current_z.shape[1] != request_count:
        raise ODEBFContractError("P2R2 field request axis differs")
    device = next(model.parameters()).device
    coefficients = torch.zeros(
        (layer_count, request_count),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    response = torch.empty(
        (request_count, layer_count * request_count), dtype=torch.float64
    )
    values: list[float | None] = [None] * request_count
    before = _parameter_inventory_sha256(model)
    processed = 0
    vjp_count = 0
    overlay: RequestLayerCoefficientOverlay
    with RequestLayerCoefficientOverlay(model, field, coefficients) as overlay:
        for batch in plan.batches:
            observed, batch_processed = _score_prepared_target_new_batch(
                model,
                batch.prepared,
                device=device,
                llama=plan.llama,
                context_sha256=plan.context_sha256,
            )
            local_values = torch.stack([item[1] for item in observed])
            eye = torch.eye(
                local_values.numel(), device=local_values.device, dtype=local_values.dtype
            )
            gradient = torch.autograd.grad(
                local_values,
                coefficients,
                grad_outputs=eye,
                is_grads_batched=True,
                retain_graph=False,
                create_graph=False,
            )[0]
            if gradient.shape != (
                local_values.numel(), layer_count, request_count
            ):
                raise ODEBFContractError("P2R2 batched VJP axes differ")
            flat = -gradient.detach().to(device="cpu", dtype=torch.float64).reshape(
                local_values.numel(), layer_count * request_count
            )
            for local, (ordinal, value, _count, _span) in enumerate(observed):
                if values[ordinal] is not None:
                    raise ODEBFContractError("P2R2 response duplicated a request")
                values[ordinal] = float(value.detach())
                response[ordinal] = flat[local]
            processed += int(batch_processed)
            vjp_count += 1
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFStateError("P2R2 physical response mutated model state")
    if (
        any(item is None for item in values)
        or processed != plan.processed_token_count
        or not bool(torch.isfinite(response).all())
        or tuple(overlay.fire_count) != tuple([len(plan.batches)] * layer_count)
    ):
        raise ODEBFContractError("P2R2 physical response coverage differs")
    payload = {
        "response_sha256": tensor_sha256(response),
        "current_nll_by_request": [float(item) for item in values if item is not None],
        "plan_sha256": plan.identity_sha256,
        "field_sha256": field.identity_sha256,
        "model_state_sha256": after,
        "model_forward_count": len(plan.batches),
        "batched_vjp_count": vjp_count,
        "processed_token_count": processed,
        "overlay_fire_count": overlay.fire_count,
    }
    return PhysicalResponseReceipt(
        response.contiguous(),
        tuple(float(item) for item in values if item is not None),
        len(plan.batches),
        vjp_count,
        processed,
        tuple(overlay.fire_count),
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class ProposalQuadratics:
    capacity_gram: torch.Tensor
    structural_p_gram: torch.Tensor
    structural_p_cross: torch.Tensor
    prior_structural_p: float
    prior_capacity_by_layer: tuple[float, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r2-proposal-quadratics/v1",
            "capacity_gram_sha256": tensor_sha256(self.capacity_gram),
            "structural_p_gram_sha256": tensor_sha256(self.structural_p_gram),
            "structural_p_cross_sha256": tensor_sha256(self.structural_p_cross),
            "prior_structural_p": self.prior_structural_p,
            "prior_capacity_by_layer": list(self.prior_capacity_by_layer),
            "identity_sha256": self.identity_sha256,
        }


def build_proposal_quadratics(
    field: P1DynamicField,
    prior_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    *,
    prior_structural_p: float,
) -> ProposalQuadratics:
    """Build exact candidate self terms and signed cross terms with prior accepted writes."""

    layer_count = len(field.layers)
    request_count = field.current_z.shape[1]
    alpha_count = layer_count * request_count
    capacity = torch.zeros((alpha_count, alpha_count), dtype=torch.float64)
    structural = torch.zeros_like(capacity)
    cross = torch.zeros(alpha_count, dtype=torch.float64)
    if not math.isfinite(prior_structural_p) or prior_structural_p < 0.0:
        raise ODEBFContractError("P2R2 prior Structural-P differs")
    prior_capacity: list[float] = []
    for layer_ordinal, layer in enumerate(field.layers):
        start = layer_ordinal * request_count
        end = start + request_count
        residual = layer.residual.detach().to(torch.float64)
        q = layer.q.detach().to(torch.float64)
        cq = layer.covariance_action.detach().to(torch.float64)
        residual_gram = residual.T @ residual
        capacity[start:end, start:end] = residual_gram * (q.T @ q)
        structural[start:end, start:end] = residual_gram * (q.T @ cq)
        previous = tuple(prior_factors_by_weight.get(layer.weight_name, ()))
        layer_capacity = 0.0
        for factor in previous:
            left = factor.left.detach().to(torch.float64)
            right = factor.right.detach().to(torch.float64)
            theta = float(factor.theta)
            cross[start:end] += theta * torch.sum(
                (left.T @ residual) * (right.T @ cq), dim=0
            )
        prior_capacity.append(layer_capacity)
        for left_index, left_factor in enumerate(previous):
            left_l = left_factor.left.detach().to(torch.float64)
            right_l = left_factor.right.detach().to(torch.float64)
            for right_factor in previous[left_index:]:
                left_r = right_factor.left.detach().to(torch.float64)
                right_r = right_factor.right.detach().to(torch.float64)
                value = (
                    float(left_factor.theta)
                    * float(right_factor.theta)
                    * float(torch.sum((left_l.T @ left_r) * (right_l.T @ right_r)))
                )
                layer_capacity += value if right_factor is left_factor else 2.0 * value
        prior_capacity[-1] = layer_capacity
    if not bool(
        torch.isfinite(capacity).all()
        and torch.isfinite(structural).all()
        and torch.isfinite(cross).all()
    ):
        raise ODEBFContractError("P2R2 proposal quadratics are nonfinite")
    payload = {
        "capacity_sha256": tensor_sha256(capacity),
        "structural_sha256": tensor_sha256(structural),
        "cross_sha256": tensor_sha256(cross),
        "prior_capacity_by_layer": prior_capacity,
        "prior_structural_p": prior_structural_p,
    }
    return ProposalQuadratics(
        capacity.contiguous(),
        structural.contiguous(),
        cross.contiguous(),
        prior_structural_p,
        tuple(prior_capacity),
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class P2R2RoutingResult:
    arm: str
    allocation: torch.Tensor
    predicted_progress: torch.Tensor
    neutral_predicted_progress: torch.Tensor
    active_requests: tuple[bool, ...]
    unreachable_requests: tuple[int, ...]
    status: str
    fallback_reason: str | None
    fairness_lambda: float
    total_normalized_progress: float
    marginal_structural_p: float
    capacity: float
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r2-routing/v1",
            "arm": self.arm,
            "allocation_shape": list(self.allocation.shape),
            "allocation_sha256": tensor_sha256(self.allocation),
            "allocation_by_layer_request": self.allocation.tolist(),
            "predicted_progress_by_request": self.predicted_progress.tolist(),
            "neutral_predicted_progress_by_request": self.neutral_predicted_progress.tolist(),
            "active_requests": list(self.active_requests),
            "unreachable_requests": list(self.unreachable_requests),
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "fairness_lambda": self.fairness_lambda,
            "total_normalized_progress": self.total_normalized_progress,
            "marginal_structural_p": self.marginal_structural_p,
            "capacity": self.capacity,
            "request_simplex_max_abs_residual": float(
                torch.max(
                    torch.abs(
                        torch.sum(self.allocation, dim=0)
                        - torch.tensor(self.active_requests, dtype=torch.float64)
                    )
                )
            ),
            "soft_requestwise_no_weaker_max_violation": float(
                torch.max(self.neutral_predicted_progress - self.predicted_progress)
            ),
            "p_budget_influence_count": 0,
            "functional_p_veto_count": 0,
            "strength_attenuation_count": 0,
            "candidate_forward_count": 0,
            "candidate_materialization_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def _bounds(active: np.ndarray, layer_count: int) -> list[tuple[float, float | None]]:
    return [
        (0.0, None) if bool(active[index % active.size]) else (0.0, 0.0)
        for index in range(layer_count * active.size)
    ]


def _constraints(
    response: np.ndarray,
    active: np.ndarray,
    layer_count: int,
    *,
    minimum_progress: np.ndarray | None = None,
    minimum_total_normalized: float | None = None,
    scale: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    request_count = active.size
    constraints: list[dict[str, Any]] = []
    for request in range(request_count):
        column = np.zeros(layer_count * request_count, dtype=np.float64)
        column[request::request_count] = 1.0
        expected = 1.0 if active[request] else 0.0
        constraints.append(
            {"type": "eq", "fun": lambda x, c=column, e=expected: float(c @ x - e)}
        )
    if minimum_progress is not None:
        for request in np.flatnonzero(active):
            row = response[request].copy()
            threshold = float(minimum_progress[request]) - P2R2_NUMERICAL_EPSILON
            constraints.append(
                {"type": "ineq", "fun": lambda x, r=row, t=threshold: float(r @ x - t)}
            )
    if minimum_total_normalized is not None:
        if scale is None:
            raise ODEBFContractError("P2R2 normalized progress scale is absent")
        normalized = np.sum(
            response[np.flatnonzero(active)] / scale[np.flatnonzero(active), None], axis=0
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, r=normalized, t=minimum_total_normalized: float(r @ x - t),
            }
        )
    return constraints


def _quadratic(matrix: np.ndarray, linear: np.ndarray | None = None):
    linear_value = np.zeros(matrix.shape[0], dtype=np.float64) if linear is None else linear
    return lambda x: float(x @ matrix @ x + 2.0 * linear_value @ x)


def _quadratic_gradient(
    matrix: np.ndarray, linear: np.ndarray | None = None
):
    linear_value = np.zeros(matrix.shape[0], dtype=np.float64) if linear is None else linear
    symmetric_sum = matrix + matrix.T
    return lambda x: symmetric_sum @ x + 2.0 * linear_value


def _reduced_quadratic_solve(
    start_full: np.ndarray,
    matrix: np.ndarray,
    linear: np.ndarray,
    response: np.ndarray,
    active: np.ndarray,
    *,
    minimum_progress: np.ndarray,
    minimum_total_normalized: float | None = None,
    scale: np.ndarray | None = None,
    p_matrix: np.ndarray | None = None,
    p_linear: np.ndarray | None = None,
    p_limit: float | None = None,
) -> Any:
    """Run SLSQP only on live request columns; fixed zero columns are exact."""

    request_count = active.size
    layer_count = start_full.size // request_count
    live = np.asarray(
        [
            layer * request_count + request
            for layer in range(layer_count)
            for request in range(request_count)
            if active[request]
        ],
        dtype=np.int64,
    )
    start = start_full[live]
    reduced_response = response[:, live]
    reduced_matrix = matrix[np.ix_(live, live)]
    reduced_linear = linear[live]
    constraints: list[dict[str, Any]] = []
    for request in np.flatnonzero(active):
        selector = np.asarray(
            [
                index
                for index, full in enumerate(live)
                if full % request_count == request
            ],
            dtype=np.int64,
        )
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, s=selector: float(np.sum(x[s]) - 1.0),
                "jac": lambda x, s=selector: np.asarray(
                    [1.0 if index in s else 0.0 for index in range(x.size)],
                    dtype=np.float64,
                ),
            }
        )
        row = reduced_response[request].copy()
        threshold = float(minimum_progress[request]) - P2R2_NUMERICAL_EPSILON
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, r=row, t=threshold: float(r @ x - t),
                "jac": lambda x, r=row: r,
            }
        )
    if minimum_total_normalized is not None:
        if scale is None:
            raise ODEBFContractError("P2R2 reduced normalized scale is absent")
        active_rows = np.flatnonzero(active)
        normalized = np.sum(
            reduced_response[active_rows] / scale[active_rows, None], axis=0
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, r=normalized, t=minimum_total_normalized: float(r @ x - t),
                "jac": lambda x, r=normalized: r,
            }
        )
    if p_limit is not None:
        if p_matrix is None or p_linear is None:
            raise ODEBFContractError("P2R2 reduced P tie is absent")
        reduced_p = p_matrix[np.ix_(live, live)]
        reduced_p_linear = p_linear[live]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, m=reduced_p, l=reduced_p_linear, p=p_limit: float(
                    p - _quadratic(m, l)(x)
                ),
                "jac": lambda x, m=reduced_p, l=reduced_p_linear: -_quadratic_gradient(
                    m, l
                )(x),
            }
        )
    result = minimize(
        _quadratic(reduced_matrix, reduced_linear),
        start,
        jac=_quadratic_gradient(reduced_matrix, reduced_linear),
        method="SLSQP",
        bounds=[(0.0, None)] * live.size,
        constraints=constraints,
        options={"ftol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE, "maxiter": 3000, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        expanded = np.zeros_like(start_full)
        expanded[live] = result.x
        result.x = expanded
    return result


def _solve_neutral(
    response: np.ndarray,
    capacity: np.ndarray,
    active: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    layer_count = capacity.shape[0] // active.size
    alpha_count = capacity.shape[0]
    bounds = _bounds(active, layer_count)
    active_rows = np.flatnonzero(active)
    if active_rows.size == 0:
        return np.zeros(alpha_count, dtype=np.float64), 0.0, 0.0

    # Stage N1: maximize minimum normalized progress.
    objective = np.zeros(alpha_count + 1, dtype=np.float64)
    objective[-1] = -1.0
    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for request in range(active.size):
        row = np.zeros(alpha_count + 1, dtype=np.float64)
        row[request::active.size] = 1.0
        a_eq.append(row)
        b_eq.append(1.0 if active[request] else 0.0)
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for request in active_rows:
        row = np.zeros(alpha_count + 1, dtype=np.float64)
        row[:alpha_count] = -response[request]
        row[-1] = scale[request]
        a_ub.append(row)
        b_ub.append(0.0)
    first = linprog(
        objective,
        A_ub=np.stack(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.stack(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=[*bounds, (None, None)],
        method="highs",
    )
    if not first.success or not np.all(np.isfinite(first.x)):
        raise ODEBFContractError("P2R2 Neutral max-min solve failed")
    fairness = float(first.x[-1])

    # Stage N2: maximize total normalized semantic progress at fixed fairness.
    normalized_total = np.sum(response[active_rows] / scale[active_rows, None], axis=0)
    second_a_ub = -response[active_rows]
    second_b_ub = -(
        fairness * scale[active_rows] - P2R2_NUMERICAL_EPSILON
    )
    second = linprog(
        -normalized_total,
        A_ub=second_a_ub,
        b_ub=second_b_ub,
        A_eq=np.stack([item[:alpha_count] for item in a_eq]),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not second.success or not np.all(np.isfinite(second.x)):
        raise ODEBFContractError("P2R2 Neutral total-semantic solve failed")
    total = float(normalized_total @ second.x)

    # Stage N3: capacity tie without weakening N1 or N2.
    third = _reduced_quadratic_solve(
        second.x,
        capacity,
        np.zeros(alpha_count, dtype=np.float64),
        response,
        active,
        minimum_progress=np.where(active, fairness * scale, 0.0),
        minimum_total_normalized=total - P2R2_NUMERICAL_EPSILON,
        scale=scale,
    )
    if not third.success or not np.all(np.isfinite(third.x)):
        raise ODEBFContractError("P2R2 Neutral capacity-tie solve failed")
    return third.x, fairness, total


def _certificate(
    allocation: np.ndarray,
    response: np.ndarray,
    active: np.ndarray,
    neutral_progress: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    request_count = active.size
    progress = response @ allocation
    simplex = max(
        abs(float(np.sum(allocation[request::request_count])) - (1.0 if active[request] else 0.0))
        for request in range(request_count)
    )
    negative = max(0.0, -float(np.min(allocation)))
    no_weaker = max(0.0, float(np.max(neutral_progress - progress)))
    if simplex > P2R2_NUMERICAL_EPSILON or negative > P2R2_NUMERICAL_EPSILON:
        raise ODEBFContractError("P2R2 request-conservation certificate failed")
    return progress, simplex, no_weaker


def solve_p2r2_routing(
    response: torch.Tensor,
    quadratics: ProposalQuadratics,
    *,
    arm: str,
) -> P2R2RoutingResult:
    if arm not in ("NEUTRAL", "SOFTP"):
        raise ODEBFContractError("P2R2 arm differs")
    observed = response.detach().to(torch.float64).cpu().numpy()
    request_count, alpha_count = observed.shape
    if (request_count, alpha_count) != (P2R2_REQUEST_COUNT, P2R2_ALPHA_COUNT):
        raise ODEBFContractError("P2R2 response shape differs")
    layer_count = alpha_count // request_count
    own = np.asarray(
        [observed[i, i::request_count] for i in range(request_count)], dtype=np.float64
    )
    scale = np.max(own, axis=1)
    active = scale > P2R2_NUMERICAL_EPSILON
    neutral, fairness, total = _solve_neutral(
        observed,
        quadratics.capacity_gram.numpy(),
        active,
        scale,
    )
    neutral_progress, _, _ = _certificate(
        neutral, observed, active, observed @ neutral
    )
    selected = neutral.copy()
    status = "NEUTRAL_CERTIFIED" if arm == "NEUTRAL" else "SOFT_CERTIFIED"
    fallback: str | None = None
    p_matrix = quadratics.structural_p_gram.numpy()
    p_cross = quadratics.structural_p_cross.numpy()
    capacity = quadratics.capacity_gram.numpy()
    if arm == "SOFTP" and np.any(active):
        first = _reduced_quadratic_solve(
            neutral,
            p_matrix,
            p_cross,
            observed,
            active,
            minimum_progress=neutral_progress,
        )
        if not first.success or not np.all(np.isfinite(first.x)):
            status = "SOFT_NEUTRAL_FALLBACK"
            fallback = "STRUCTURAL_P_SOLVE_UNCERTIFIED"
        else:
            p_star = float(_quadratic(p_matrix, p_cross)(first.x))
            p_scale = max(float(np.trace(p_matrix)), 1.0e-12)
            p_tie = SIMPLEX_XI_TIE_TOLERANCE * p_scale
            weighted = capacity.copy()
            for layer in range(layer_count):
                start = layer * request_count
                end = start + request_count
                weighted[start:end, start:end] *= 1.0 + quadratics.prior_capacity_by_layer[layer]
            second = _reduced_quadratic_solve(
                first.x,
                weighted,
                np.zeros(alpha_count, dtype=np.float64),
                observed,
                active,
                minimum_progress=neutral_progress,
                p_matrix=p_matrix,
                p_linear=p_cross,
                p_limit=p_star + p_tie,
            )
            if not second.success or not np.all(np.isfinite(second.x)):
                status = "SOFT_NEUTRAL_FALLBACK"
                fallback = "CAPACITY_TIE_SOLVE_UNCERTIFIED"
            else:
                progress, _, no_weaker = _certificate(
                    second.x, observed, active, neutral_progress
                )
                if no_weaker > P2R2_NUMERICAL_EPSILON:
                    status = "SOFT_NEUTRAL_FALLBACK"
                    fallback = "REQUESTWISE_NO_WEAKER_CERTIFICATE_FAILED"
                else:
                    selected = second.x
    progress, _, no_weaker = _certificate(
        selected, observed, active, neutral_progress
    )
    if arm == "SOFTP" and no_weaker > P2R2_NUMERICAL_EPSILON:
        raise ODEBFContractError("P2R2 Soft weakened a request")
    selected_tensor = torch.from_numpy(selected.reshape(layer_count, request_count)).to(
        torch.float64
    )
    progress_tensor = torch.from_numpy(progress).to(torch.float64)
    neutral_tensor = torch.from_numpy(neutral_progress).to(torch.float64)
    marginal_p = float(_quadratic(p_matrix, p_cross)(selected))
    selected_capacity = float(selected @ capacity @ selected)
    payload = {
        "arm": arm,
        "allocation_sha256": tensor_sha256(selected_tensor),
        "predicted_sha256": tensor_sha256(progress_tensor),
        "neutral_predicted_sha256": tensor_sha256(neutral_tensor),
        "active": active.tolist(),
        "status": status,
        "fallback": fallback,
        "fairness": fairness,
        "total": total,
        "marginal_p": marginal_p,
        "capacity": selected_capacity,
    }
    return P2R2RoutingResult(
        arm,
        selected_tensor,
        progress_tensor,
        neutral_tensor,
        tuple(bool(item) for item in active),
        tuple(int(item) for item in np.flatnonzero(~active)),
        status,
        fallback,
        fairness,
        total,
        marginal_p,
        selected_capacity,
        canonical_hash(payload),
    )


def p2r2_waypoint_factors(
    field: P1DynamicField,
    route: P2R2RoutingResult,
    *,
    outer_step: int,
) -> dict[str, WaypointFactor]:
    if route.allocation.shape != (len(field.layers), field.current_z.shape[1]):
        raise ODEBFContractError("P2R2 selected allocation geometry differs")
    result: dict[str, WaypointFactor] = {}
    for ordinal, layer in enumerate(field.layers):
        allocation = route.allocation[ordinal].to(torch.float32)
        left = layer.residual.detach().to(torch.float32) * allocation.unsqueeze(0)
        factor = WaypointFactor(
            layer.weight_name,
            layer.layer,
            0,
            outer_step,
            ordinal,
            1.0,
            left.contiguous(),
            layer.q.detach().to(torch.float32).contiguous(),
            joint_batch=True,
            global_batch_size=field.current_z.shape[1],
        )
        if factor.weight_name in result:
            raise ODEBFContractError("P2R2 layer factor duplicated")
        result[factor.weight_name] = factor
    return result


def p2r2_forbidden_influence_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p2r2-forbidden-influence/v1",
        "legacy_writer_access_count": 0,
        "legacy_writer_decision_influence_count": 0,
        "semantic_debt_input_count": 0,
        "remaining_horizon_division_count": 0,
        "hard_p_budget_influence_count": 0,
        "functional_p_veto_count": 0,
        "candidate_forward_count": 0,
        "candidate_materialization_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "writer_h_application_count": 0,
        "outer_joint_materialization_count_per_step": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P2R2_INSTRUCTION_ID",
    "P2R2_METHOD_ID",
    "P2R2RoutingResult",
    "PhysicalResponseReceipt",
    "ProposalQuadratics",
    "RequestLayerCoefficientOverlay",
    "build_proposal_quadratics",
    "measure_request_layer_response",
    "p2r2_forbidden_influence_receipt",
    "p2r2_waypoint_factors",
    "solve_p2r2_routing",
]
