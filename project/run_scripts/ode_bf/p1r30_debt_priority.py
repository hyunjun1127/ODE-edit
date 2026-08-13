"""P1R30 debt-priority routing around the exact P1R24 A0 writer.

The scientific reference write is the P1R24 Neutral/A0 coefficient vector.
Semantic debt is retained request-wise, but it may change only the second
progress constraint used by the Soft allocation.  It cannot change the A0
reference magnitude, the target trajectory, or the physical write clock.

The Soft program is expressed directly in applied coefficient ``c`` space.
There is intentionally no ``rho / slope`` transform and no coefficient
division by a physical slope in this module.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import minimize
import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_LAYER_ORDER,
    FIXED_E8_SOLVER_FTOL,
    FIXED_E8_SOLVER_MAXITER,
    FixedE8Arm,
)
from .functional import tensor_sha256
from .p1_backend import P1DynamicField, SignedProgressReceipt
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .routing import RoutingProblem
from .scalable_batched_model import (
    OrdinalTargetActivationOverlay,
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    _parameter_inventory_sha256,
    _score_prepared_target_new_batch,
)
from .scalable_batched_runtime import P1R23_CONTEXTS_PER_REQUEST


P1R30_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R30-DEBT-PRIORITY-A0-RELATIVE-BARRIER-V1"
)
P1R30_METHOD_ID = "P1R30-P1R24-A0-DEBT-PRIORITY-STRUCTURAL-P-SOFT-V1"
P1R30_K = 8
P1R30_H = 1.0 / P1R30_K
P1R30_EPSILON = 1.0e-12
P1R30_REDUCTION_TOLERANCE = 1.0e-7


def _finite_tuple(label: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ODEBFContractError(f"P1R30 {label} differs")
    observed = tuple(float(item) for item in values)
    if not observed or any(not math.isfinite(item) for item in observed):
        raise ODEBFContractError(f"P1R30 {label} differs")
    return observed


@dataclass(frozen=True, slots=True)
class P1R30TargetObjective:
    """P1R24-compatible target objective with raw request-column gradients."""

    objective: ScalableObjectiveResult
    per_request_gradient_fp64: torch.Tensor
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r30-request-gradient-objective/v1",
            "objective": self.objective.raw_free_payload(),
            "per_request_gradient_sha256": tensor_sha256(
                self.per_request_gradient_fp64
            ),
            "gradient_dtype": str(self.per_request_gradient_fp64.dtype),
            "gradient_shape": list(self.per_request_gradient_fp64.shape),
            "aggregation": "uniform-mean-over-requests-after-requestwise-full-six",
            "requestwise_gradient_reuses_objective_backward": True,
            "requestwise_gradient_added_model_forward_count": 0,
            "requestwise_gradient_added_backward_count": 0,
            "batch_mean_gradient_dot_substitution_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_p1r30_target_new_objective(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
) -> P1R30TargetObjective:
    """Retain unaveraged request gradients from the existing target backward."""

    if (
        target_state.ndim != 2
        or target_state.shape != current_terminal.shape
        or target_state.shape[1] != plan.request_count
        or not target_state.requires_grad
        or not bool(torch.isfinite(target_state).all())
        or not bool(torch.isfinite(current_terminal).all())
    ):
        raise ODEBFContractError("P1R30 target objective geometry differs")
    device = next(model.parameters()).device
    before = _parameter_inventory_sha256(model)
    values: list[torch.Tensor | None] = [None] * plan.request_count
    counts: list[int | None] = [None] * plan.request_count
    spans: list[str | None] = [None] * plan.request_count
    request_gradient = torch.zeros_like(
        target_state, dtype=torch.float64, device="cpu"
    )
    processed = 0
    overlay_fire_count = 0
    for batch in plan.batches:
        with ExitStack() as stack:
            residual = target_state - current_terminal.to(
                device=target_state.device, dtype=target_state.dtype
            )
            overlay = stack.enter_context(
                OrdinalTargetActivationOverlay(
                    model,
                    target_layer_name,
                    residual=residual,
                    row_request_ordinals=batch.prepared.row_request_ordinals,
                    padded_lookup_positions=batch.padded_lookup_positions,
                )
            )
            observed, batch_processed = _score_prepared_target_new_batch(
                model,
                batch.prepared,
                device=device,
                llama=plan.llama,
                context_sha256=plan.context_sha256,
            )
            request_sum = torch.stack([item[1] for item in observed]).sum()
            gradient = torch.autograd.grad(
                request_sum,
                target_state,
                retain_graph=False,
                create_graph=False,
            )[0]
            request_gradient.add_(
                gradient.detach().to(device="cpu", dtype=torch.float64)
            )
            overlay_fire_count += overlay.fire_count
            for ordinal, value, count, span in observed:
                if values[ordinal] is not None:
                    raise ODEBFContractError(
                        "P1R30 target objective duplicated a request"
                    )
                values[ordinal] = value.detach().to(device="cpu", dtype=torch.float64)
                counts[ordinal] = int(count)
                spans[ordinal] = str(span)
            processed += int(batch_processed)
    if (
        any(item is None for item in values)
        or any(item is None for item in counts)
        or any(item is None for item in spans)
        or processed != plan.processed_token_count
        or overlay_fire_count != len(plan.batches)
        or not bool(torch.isfinite(request_gradient).all())
    ):
        raise ODEBFContractError("P1R30 target objective coverage differs")
    mean_gradient = (
        request_gradient / plan.request_count
    ).to(dtype=torch.float32).contiguous()
    final_values = tuple(float(item) for item in values if item is not None)
    final_spans = tuple(str(item) for item in spans if item is not None)
    final_counts = tuple(int(item) for item in counts if item is not None)
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFStateError("P1R30 target objective mutated model")
    public_payload = {
        "loss": math.fsum(final_values) / plan.request_count,
        "per_request_values": list(final_values),
        "request_order_sha256": plan.request_order_sha256,
        "target_span_sha256": canonical_hash(list(final_spans)),
        "suffix_token_counts": list(final_counts),
        "model_forward_count": len(plan.batches),
        "backward_count": len(plan.batches),
        "processed_token_count": processed,
        "padded_token_count": plan.padded_token_count,
        "target_gradient_sha256": tensor_sha256(mean_gradient),
        "coefficient_gradient_sha256": None,
        "plan_sha256": plan.identity_sha256,
        "model_state_sha256": after,
    }
    objective = ScalableObjectiveResult(
        public_payload["loss"],
        final_values,
        plan.request_order_sha256,
        public_payload["target_span_sha256"],
        final_spans,
        final_counts,
        len(plan.batches),
        len(plan.batches),
        processed,
        plan.padded_token_count,
        mean_gradient,
        None,
        plan.identity_sha256,
        after,
        canonical_hash(public_payload),
    )
    payload = {
        "objective_sha256": objective.identity_sha256,
        "per_request_gradient_sha256": tensor_sha256(request_gradient),
        "request_count": plan.request_count,
        "request_order_sha256": plan.request_order_sha256,
        "full_six_contexts": True,
        "mean_of_means_count": 0,
    }
    return P1R30TargetObjective(
        objective,
        request_gradient.contiguous(),
        canonical_hash(payload),
    )


class _RequestwiseCoefficientOverlay(
    AbstractContextManager["_RequestwiseCoefficientOverlay"]
):
    """Give each request its own zero-entry coefficient for one joint VJP.

    At the zero coefficient entry the derivative of request ``i`` with
    respect to row ``i`` is exactly the derivative of that request with
    respect to the shared physical coefficient.  All request rows are
    collected in the same model forward/backward used for the mean slope.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[Any],
        coefficients: torch.Tensor,
        row_request_ordinals: Sequence[int],
    ) -> None:
        self.model = model
        self.layers = tuple(layers)
        self.coefficients = coefficients
        self.row_request_ordinals = tuple(int(item) for item in row_request_ordinals)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.fire_counts = [0 for _ in self.layers]

    def _hook(self, index: int):
        field = self.layers[index]

        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del module
            self.fire_counts[index] += 1
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise ODEBFContractError(
                    "P1R30 requestwise gradient overlay contract differs"
                )
            hidden = inputs[0]
            right = field.q.to(device=hidden.device, dtype=torch.float32)
            left = field.residual.to(device=hidden.device, dtype=torch.float32)
            perturbation = (hidden.float() @ right) @ left.T
            rows = perturbation.shape[0]
            if rows != len(self.row_request_ordinals):
                raise ODEBFContractError(
                    "P1R30 requestwise overlay row mapping differs"
                )
            row_index = torch.tensor(
                self.row_request_ordinals,
                device=self.coefficients.device,
                dtype=torch.long,
            )
            scale = self.coefficients[row_index, index].to(
                device=perturbation.device, dtype=perturbation.dtype
            )
            shape = (rows,) + (1,) * (perturbation.ndim - 1)
            result = output.float() + scale.reshape(shape) * perturbation
            return result.to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "_RequestwiseCoefficientOverlay":
        if (
            self.coefficients.ndim != 2
            or self.coefficients.shape[1] != len(self.layers)
            or not self.row_request_ordinals
            or min(self.row_request_ordinals) < 0
            or max(self.row_request_ordinals) >= self.coefficients.shape[0]
        ):
            raise ODEBFContractError(
                "P1R30 requestwise coefficient geometry differs"
            )
        for index, field in enumerate(self.layers):
            module_name = field.weight_name[: -len(".weight")]
            module = self.model.get_submodule(module_name)
            if type(module) is not torch.nn.Linear:
                raise ODEBFContractError(
                    "P1R30 requestwise overlay target is not exact Linear"
                )
            self._handles.append(module.register_forward_hook(self._hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if exc_type is None and any(item != 1 for item in self.fire_counts):
            raise ODEBFContractError(
                "P1R30 requestwise physical slope hook firing differs"
            )


@dataclass(frozen=True, slots=True)
class P1R30RequestwiseSlopeReceipt:
    signed_progress_by_request: tuple[tuple[float, ...], ...]
    signed_progress_mean: tuple[float, ...]
    reduction_max_abs_residual: float
    reduction_representation_rounding_max_abs: float
    model_forward_count: int
    backward_count: int
    processed_token_count: int
    objective_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r30-requestwise-physical-slope/v1",
            "request_count": len(self.signed_progress_by_request),
            "layer_order": list(FIXED_E8_LAYER_ORDER),
            "signed_progress_by_request": [
                list(item) for item in self.signed_progress_by_request
            ],
            "signed_progress_mean": list(self.signed_progress_mean),
            "signed_scalars_preserved": True,
            "reduction_max_abs_residual": self.reduction_max_abs_residual,
            "reduction_representation_rounding_max_abs": (
                self.reduction_representation_rounding_max_abs
            ),
            "reduction_certificate_coordinate": (
                "FP64_REQUEST_ACCUMULATION_THEN_MODEL_FACING_FP32"
            ),
            "reduction_identity": "mean_i(a_i_l)=a_mean_l",
            "same_authoritative_backward": True,
            "requestwise_extension_added_model_forward_count": 0,
            "requestwise_extension_added_backward_count": 0,
            "model_forward_count": self.model_forward_count,
            "backward_count": self.backward_count,
            "processed_token_count": self.processed_token_count,
            "objective_sha256": self.objective_sha256,
            "inverse_slope_operation_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_p1r30_requestwise_physical_slopes(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    field: P1DynamicField,
) -> tuple[SignedProgressReceipt, ScalableObjectiveResult, P1R30RequestwiseSlopeReceipt]:
    """Evaluate requestwise and uniform physical slopes in one F/B ledger."""

    if tuple(int(item.layer) for item in field.layers) != FIXED_E8_LAYER_ORDER:
        raise ODEBFContractError("P1R30 physical field layer order differs")
    device = next(model.parameters()).device
    coefficients = torch.zeros(
        (plan.request_count, len(FIXED_E8_LAYER_ORDER)),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    before = _parameter_inventory_sha256(model)
    request_gradient = torch.zeros_like(
        coefficients, device="cpu", dtype=torch.float64
    )
    values: list[torch.Tensor | None] = [None] * plan.request_count
    counts: list[int | None] = [None] * plan.request_count
    spans: list[str | None] = [None] * plan.request_count
    processed = 0
    backward_count = 0
    for batch in plan.batches:
        with _RequestwiseCoefficientOverlay(
            model,
            field.layers,
            coefficients,
            batch.prepared.row_request_ordinals,
        ):
            observed, batch_processed = _score_prepared_target_new_batch(
                model,
                batch.prepared,
                device=device,
                llama=plan.llama,
                context_sha256=plan.context_sha256,
            )
            request_sum = torch.stack([item[1] for item in observed]).sum()
            gradient = torch.autograd.grad(
                request_sum,
                coefficients,
                retain_graph=False,
                create_graph=False,
            )[0]
            request_gradient.add_(
                gradient.detach().to(device="cpu", dtype=torch.float64)
            )
            backward_count += 1
        for ordinal, value, count, span in observed:
            if values[ordinal] is not None:
                raise ODEBFContractError(
                    "P1R30 physical slope duplicated a request"
                )
            values[ordinal] = value.detach().to(device="cpu", dtype=torch.float64)
            counts[ordinal] = int(count)
            spans[ordinal] = str(span)
        processed += int(batch_processed)
    if (
        any(item is None for item in values)
        or any(item is None for item in counts)
        or any(item is None for item in spans)
        or processed != plan.processed_token_count
        or not bool(torch.isfinite(request_gradient).all())
    ):
        raise ODEBFContractError("P1R30 physical slope coverage differs")
    mean_gradient64 = request_gradient.mean(dim=0)
    mean_gradient = mean_gradient64.to(dtype=torch.float32).contiguous()
    signed_by_request64 = -request_gradient
    signed_mean64 = signed_by_request64.mean(dim=0)
    signed_mean_model_coordinate64 = signed_mean64.to(
        dtype=torch.float32
    ).to(dtype=torch.float64)
    signed_mean_from_public = -mean_gradient.to(dtype=torch.float64)
    reduction_residual = float(
        torch.max(
            torch.abs(
                signed_mean_model_coordinate64 - signed_mean_from_public
            )
        )
    )
    reduction_representation_rounding = float(
        torch.max(
            torch.abs(signed_mean64 - signed_mean_model_coordinate64)
        )
    )
    if reduction_residual > P1R30_REDUCTION_TOLERANCE:
        raise ODEBFContractError("P1R30 requestwise slope reduction differs")
    final_values = tuple(float(item) for item in values if item is not None)
    final_spans = tuple(str(item) for item in spans if item is not None)
    final_counts = tuple(int(item) for item in counts if item is not None)
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFStateError("P1R30 physical slope mutated model")
    objective_payload = {
        "loss": math.fsum(final_values) / plan.request_count,
        "per_request_values": list(final_values),
        "request_order_sha256": plan.request_order_sha256,
        "target_span_sha256": canonical_hash(list(final_spans)),
        "suffix_token_counts": list(final_counts),
        "model_forward_count": len(plan.batches),
        "backward_count": backward_count,
        "processed_token_count": processed,
        "padded_token_count": plan.padded_token_count,
        "target_gradient_sha256": None,
        "coefficient_gradient_sha256": tensor_sha256(mean_gradient),
        "plan_sha256": plan.identity_sha256,
        "model_state_sha256": after,
    }
    objective = ScalableObjectiveResult(
        objective_payload["loss"],
        final_values,
        plan.request_order_sha256,
        objective_payload["target_span_sha256"],
        final_spans,
        final_counts,
        len(plan.batches),
        backward_count,
        processed,
        plan.padded_token_count,
        None,
        mean_gradient,
        plan.identity_sha256,
        after,
        canonical_hash(objective_payload),
    )
    mean_signed = tuple(float(item) for item in signed_mean_from_public)
    signed_receipt = SignedProgressReceipt(
        field.identity_sha256,
        mean_signed,
        tuple(
            layer
            for layer, value in zip(
                FIXED_E8_LAYER_ORDER, mean_signed, strict=True
            )
            if value <= 0.0
        ),
        tensor_sha256(mean_gradient),
        objective.model_forward_count,
        objective.processed_token_count,
        True,
        "TARGET_NEW_NLL",
        0,
        objective.identity_sha256,
        plan.context_sha256,
        P1R23_CONTEXTS_PER_REQUEST,
        (1, 5),
        objective.backward_count,
        objective.loss,
    )
    rows = tuple(
        tuple(float(item) for item in row)
        for row in signed_by_request64
    )
    receipt_payload = {
        "field_sha256": field.identity_sha256,
        "request_order_sha256": plan.request_order_sha256,
        "signed_by_request": [list(item) for item in rows],
        "signed_mean": list(mean_signed),
        "reduction_max_abs_residual": reduction_residual,
        "reduction_representation_rounding_max_abs": (
            reduction_representation_rounding
        ),
        "objective_sha256": objective.identity_sha256,
        "model_forward_count": objective.model_forward_count,
        "backward_count": objective.backward_count,
    }
    requestwise = P1R30RequestwiseSlopeReceipt(
        rows,
        mean_signed,
        reduction_residual,
        reduction_representation_rounding,
        objective.model_forward_count,
        objective.backward_count,
        objective.processed_token_count,
        objective.identity_sha256,
        canonical_hash(receipt_payload),
    )
    return signed_receipt, objective, requestwise


@dataclass(frozen=True, slots=True)
class P1R30NominalDemand:
    per_request: tuple[float, ...]
    aggregate: float
    signed_per_request: tuple[float, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        values = np.asarray(self.per_request, dtype=np.float64)
        return {
            "schema": "ode-edit-s05-p1r30-nominal-demand/v1",
            "request_count": len(self.per_request),
            "per_request_sha256": canonical_hash(list(self.per_request)),
            "signed_per_request_sha256": canonical_hash(
                list(self.signed_per_request)
            ),
            "minimum": float(values.min()),
            "mean": self.aggregate,
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(values.max()),
            "definition": "positive_part(-g_new_i_dot_target_displacement_i)",
            "lag_influence_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def nominal_demand_from_target_displacement(
    per_request_gradient_fp64: torch.Tensor,
    target_displacement: torch.Tensor,
) -> P1R30NominalDemand:
    gradient = per_request_gradient_fp64.detach().to(
        device="cpu", dtype=torch.float64
    )
    displacement = target_displacement.detach().to(
        device="cpu", dtype=torch.float64
    )
    if (
        gradient.ndim != 2
        or displacement.shape != gradient.shape
        or gradient.shape[1] <= 0
        or not bool(torch.isfinite(gradient).all())
        or not bool(torch.isfinite(displacement).all())
    ):
        raise ODEBFContractError("P1R30 nominal demand geometry differs")
    signed_tensor = -torch.sum(gradient * displacement, dim=0)
    signed = tuple(float(item) for item in signed_tensor)
    values = tuple(max(item, 0.0) for item in signed)
    aggregate = math.fsum(values) / len(values)
    payload = {
        "gradient_sha256": tensor_sha256(gradient),
        "target_displacement_sha256": tensor_sha256(displacement),
        "signed": list(signed),
        "per_request": list(values),
        "aggregate": aggregate,
        "reduction": "mean_i_positive_part(-g_i_dot_delta_z_i)",
        "batch_mean_gradient_dot_substitution_count": 0,
        "lag_influence_count": 0,
    }
    return P1R30NominalDemand(
        values, aggregate, signed, canonical_hash(payload)
    )


@dataclass(frozen=True, slots=True)
class P1R30Priority:
    step_index: int
    remaining_steps: int
    nominal: tuple[float, ...]
    debt: tuple[float, ...]
    u: tuple[float, ...]
    omega: tuple[float, ...]
    all_u_zero: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r30-debt-priority/v1",
            "step_index": self.step_index,
            "remaining_steps": self.remaining_steps,
            "nominal_sha256": canonical_hash(list(self.nominal)),
            "debt_sha256": canonical_hash(list(self.debt)),
            "u": list(self.u),
            "omega": list(self.omega),
            "omega_sum": math.fsum(self.omega),
            "definition": "u_i=rho_nom_i+D_i/(K-k);omega_i=u_i/(sum_u+eps)",
            "epsilon": P1R30_EPSILON,
            "all_u_zero": self.all_u_zero,
            "uniform_branch": self.all_u_zero,
            "D_zero_nonuniform_nominal_preserved": True,
            "debt_total_update_magnitude_influence_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def debt_priority(
    nominal: Sequence[float],
    debt: Sequence[float],
    *,
    step_index: int,
) -> P1R30Priority:
    rho = np.asarray(_finite_tuple("nominal priority", nominal), dtype=np.float64)
    owed = np.asarray(_finite_tuple("debt priority", debt), dtype=np.float64)
    if (
        rho.shape != owed.shape
        or np.any(rho < 0.0)
        or np.any(owed < 0.0)
        or step_index < 0
        or step_index >= P1R30_K
    ):
        raise ODEBFContractError("P1R30 debt priority state differs")
    remaining = P1R30_K - step_index
    u = rho + owed / remaining
    all_zero = bool(np.all(u == 0.0))
    if all_zero:
        omega = np.full(u.size, 1.0 / u.size, dtype=np.float64)
    else:
        omega = u / (float(u.sum()) + P1R30_EPSILON)
    if not np.all(np.isfinite(omega)) or np.any(omega < 0.0):
        raise ODEBFContractError("P1R30 debt priority weights differ")
    payload = {
        "step_index": step_index,
        "remaining_steps": remaining,
        "nominal": rho.tolist(),
        "debt": owed.tolist(),
        "u": u.tolist(),
        "omega": omega.tolist(),
        "all_u_zero": all_zero,
        "epsilon": P1R30_EPSILON,
    }
    return P1R30Priority(
        step_index,
        remaining,
        tuple(float(item) for item in rho),
        tuple(float(item) for item in owed),
        tuple(float(item) for item in u),
        tuple(float(item) for item in omega),
        all_zero,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class P1R30DebtCompletion:
    step_index: int
    nominal: tuple[float, ...]
    w_only_before: tuple[float, ...]
    w_only_after: tuple[float, ...]
    actual_progress: tuple[float, ...]
    debt_before: tuple[float, ...]
    debt_after: tuple[float, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        actual = np.asarray(self.actual_progress, dtype=np.float64)
        debt = np.asarray(self.debt_after, dtype=np.float64)
        return {
            "schema": "ode-edit-s05-p1r30-debt-completion/v1",
            "step_index": self.step_index,
            "nominal_sha256": canonical_hash(list(self.nominal)),
            "w_only_before_sha256": canonical_hash(list(self.w_only_before)),
            "w_only_after_sha256": canonical_hash(list(self.w_only_after)),
            "actual_progress": list(self.actual_progress),
            "actual_progress_minimum": float(actual.min()),
            "actual_progress_median": float(np.median(actual)),
            "actual_progress_p90": float(np.quantile(actual, 0.9)),
            "actual_progress_mean": float(actual.mean()),
            "actual_progress_maximum": float(actual.max()),
            "negative_actual_count": int(np.count_nonzero(actual < 0.0)),
            "debt_before_sha256": canonical_hash(list(self.debt_before)),
            "debt_after": list(self.debt_after),
            "debt_after_minimum": float(debt.min()),
            "debt_after_median": float(np.median(debt)),
            "debt_after_p90": float(np.quantile(debt, 0.9)),
            "debt_after_mean": float(debt.mean()),
            "debt_after_maximum": float(debt.max()),
            "positive_debt_request_count": int(np.count_nonzero(debt > 0.0)),
            "negative_actual_increases_debt": True,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class P1R30DebtState:
    step_index: int
    debt: tuple[float, ...]
    current_w_only_values: tuple[float, ...]
    identity_sha256: str

    @classmethod
    def initial(cls, current_w_only_values: Sequence[float]) -> "P1R30DebtState":
        current = _finite_tuple("initial W-only values", current_w_only_values)
        debt = tuple(0.0 for _ in current)
        payload = {"step_index": 0, "debt": list(debt), "current": list(current)}
        return cls(0, debt, current, canonical_hash(payload))

    def priority(self, nominal: P1R30NominalDemand) -> P1R30Priority:
        if len(nominal.per_request) != len(self.debt):
            raise ODEBFContractError("P1R30 priority inventory differs")
        return debt_priority(
            nominal.per_request, self.debt, step_index=self.step_index
        )

    def complete(
        self,
        nominal: P1R30NominalDemand,
        next_w_only_values: Sequence[float],
    ) -> tuple["P1R30DebtState", P1R30DebtCompletion]:
        if len(nominal.per_request) != len(self.debt) or self.step_index >= P1R30_K:
            raise ODEBFContractError("P1R30 debt completion inventory differs")
        after = _finite_tuple("next W-only values", next_w_only_values)
        if len(after) != len(self.debt):
            raise ODEBFContractError("P1R30 next W-only request count differs")
        actual = tuple(
            float(self.current_w_only_values[index] - after[index])
            for index in range(len(after))
        )
        debt_after = tuple(
            max(
                float(
                    self.debt[index]
                    + nominal.per_request[index]
                    - actual[index]
                ),
                0.0,
            )
            for index in range(len(after))
        )
        completion_payload = {
            "step_index": self.step_index,
            "nominal": list(nominal.per_request),
            "before": list(self.current_w_only_values),
            "after": list(after),
            "actual": list(actual),
            "debt_before": list(self.debt),
            "debt_after": list(debt_after),
        }
        completion = P1R30DebtCompletion(
            self.step_index,
            nominal.per_request,
            self.current_w_only_values,
            after,
            actual,
            self.debt,
            debt_after,
            canonical_hash(completion_payload),
        )
        state_payload = {
            "step_index": self.step_index + 1,
            "debt": list(debt_after),
            "current": list(after),
        }
        return (
            P1R30DebtState(
                self.step_index + 1,
                debt_after,
                after,
                canonical_hash(state_payload),
            ),
            completion,
        )


class P1R30RoutingStatus(str, Enum):
    A0_REFERENCE = "A0_REFERENCE"
    SOFT_SELECTED = "SOFT_SELECTED"
    SOFT_FALLBACK_NEUTRAL = "SOFT_FALLBACK_NEUTRAL"
    NO_ROUTING_DOF = "NO_ROUTING_DOF"
    NO_POSITIVE_DIRECTION = "NO_POSITIVE_DIRECTION"


@dataclass(frozen=True, slots=True)
class P1R30SolverCertificate:
    phase: str
    success: bool
    status: int
    iterations: int
    finite: bool
    nonnegative_violation: float
    mean_progress_shortfall: float
    debt_progress_shortfall: float
    unnormalized_debt_shortfall: float
    energy_violation: float
    structural_p_tie_violation: float
    passed: bool
    message_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "backend": "scipy-slsqp-float64",
                "backend_version": SCIPY_VERSION,
                "solver_ftol": FIXED_E8_SOLVER_FTOL,
                "primal_tolerance": SIMPLEX_PRIMAL_TOLERANCE,
                "energy_relative_tolerance": SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
                "energy_absolute_tolerance": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
            }
        )
        return payload


def _signed_reference_ratio(selected: float, reference: float) -> float:
    """A ratio-like signed improvement score that stays ordered at ref <= 0."""

    return 1.0 + (selected - reference) / max(abs(reference), P1R30_EPSILON)


@dataclass(frozen=True, slots=True)
class P1R30RoutingResult:
    arm: FixedE8Arm
    status: P1R30RoutingStatus
    rho_reference: float
    a_mean: tuple[float, ...]
    a_debt: tuple[float, ...]
    a_debt_unnormalized: tuple[float, ...]
    active_mask: tuple[bool, ...]
    c0: tuple[float, ...]
    selected_c: tuple[float, ...]
    velocity: tuple[float, ...]
    reference_mean_progress: float
    selected_mean_progress: float
    reference_debt_progress: float
    selected_debt_progress: float
    reference_unnormalized_debt_progress: float
    selected_unnormalized_debt_progress: float
    mean_progress_ratio: float
    debt_progress_ratio: float
    reference_energy: float
    selected_energy: float
    energy_ratio: float
    reference_structural_p: float
    selected_structural_p: float
    structural_p_marginal: float
    reference_capacity: float
    selected_capacity: float
    coefficient_l1_distance: float
    coefficient_l2_distance: float
    coefficient_cosine: float
    reduction_max_abs_residual: float
    reduction_representation_rounding_max_abs: float
    linear_constraint_rank: int
    feasible_dimension: int
    certificates: tuple[P1R30SolverCertificate, ...]
    soft_fallback: bool
    identity_sha256: str

    @property
    def applied_coefficient(self) -> tuple[float, ...]:
        return self.selected_c

    @property
    def predicted_progress(self) -> float:
        return self.selected_mean_progress

    @property
    def alpha_req(self) -> float:
        return self.rho_reference

    @property
    def alpha_max(self) -> float:
        return self.reference_mean_progress

    @property
    def alpha_apply(self) -> float:
        return self.selected_mean_progress

    @property
    def coverage(self) -> float:
        if abs(self.reference_mean_progress) <= P1R30_EPSILON:
            return 1.0
        return self.selected_mean_progress / self.reference_mean_progress

    @property
    def equality_residual(self) -> float:
        return max(
            self.reference_mean_progress - self.selected_mean_progress,
            self.reference_debt_progress - self.selected_debt_progress,
            self.reference_unnormalized_debt_progress
            - self.selected_unnormalized_debt_progress,
            self.selected_energy
            - (
                self.reference_energy
                * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
                + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
            ),
            0.0,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "instruction_id": P1R30_INSTRUCTION_ID,
                "method_id": P1R30_METHOD_ID,
                "parameterization": "DIRECT_APPLIED_COEFFICIENT_C_SPACE",
                "reference_writer": "P1R24_A0_NEUTRAL",
                "constraints": [
                    "c>=0",
                    "a_mean_dot_c>=a_mean_dot_c0",
                    "a_debt_dot_c>=a_debt_dot_c0",
                    "E(c)<=E(c0)_WITH_NUMERICAL_TOLERANCE",
                ],
                "inverse_slope_operation_count": 0,
                "legacy_exact_strength_soft_hp_import_count": 0,
                "legacy_exact_strength_soft_hp_call_count": 0,
                "rho_over_slope_transform_count": 0,
                "coefficient_division_by_physical_slope_count": 0,
                "debt_total_update_magnitude_influence_count": 0,
                "adaptive_lambda_count": 0,
                "hidden_strength_cap_count": 0,
                "per_layer_cap_influence_count": 0,
                "hard_p_budget_influence_count": 0,
                "functional_p_veto_count": 0,
                "retry_backtracking_count": 0,
                "h": P1R30_H,
                "h_application_count": 1,
            }
        )
        return payload


def _certificate(
    phase: str,
    result: Any,
    value: np.ndarray,
    *,
    mean_delta: float,
    debt_delta: float,
    unnormalized_delta: float,
    energy_violation: float,
    p_tie_violation: float,
    require_unnormalized: bool,
) -> P1R30SolverCertificate:
    finite = bool(np.all(np.isfinite(value)))
    negative = max(float(-value.min(initial=0.0)), 0.0) if finite else math.inf
    mean_shortfall = max(-float(mean_delta), 0.0) if finite else math.inf
    debt_shortfall = max(-float(debt_delta), 0.0) if finite else math.inf
    unnormalized_shortfall = (
        max(-float(unnormalized_delta), 0.0)
        if finite and require_unnormalized
        else 0.0
    )
    passed = bool(
        result.success
        and finite
        and negative <= SIMPLEX_PRIMAL_TOLERANCE
        and mean_shortfall <= SIMPLEX_PRIMAL_TOLERANCE
        and debt_shortfall <= SIMPLEX_PRIMAL_TOLERANCE
        and unnormalized_shortfall <= SIMPLEX_PRIMAL_TOLERANCE
        and energy_violation <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
        and p_tie_violation <= SIMPLEX_PRIMAL_TOLERANCE
    )
    return P1R30SolverCertificate(
        phase,
        bool(result.success),
        int(result.status),
        int(getattr(result, "nit", 0)),
        finite,
        negative,
        mean_shortfall,
        debt_shortfall,
        unnormalized_shortfall,
        float(energy_violation),
        float(p_tie_violation),
        passed,
        canonical_hash({"message": str(result.message)}),
    )


def solve_p1r30_a0_relative_routing(
    problem: RoutingProblem,
    *,
    arm: FixedE8Arm | str,
    rho_reference: float,
    requestwise_signed_progress: Sequence[Sequence[float]],
    priority: P1R30Priority,
) -> P1R30RoutingResult:
    """Solve the P1R30 allocation directly in applied coefficient space."""

    selected_arm = FixedE8Arm(arm)
    if selected_arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("P1R30 routing arm differs")
    request_slopes = np.asarray(requestwise_signed_progress, dtype=np.float64)
    if (
        request_slopes.ndim != 2
        or request_slopes.shape[1] != len(FIXED_E8_LAYER_ORDER)
        or request_slopes.shape[0] != len(priority.omega)
        or not np.all(np.isfinite(request_slopes))
        or not math.isfinite(rho_reference)
        or rho_reference < 0.0
    ):
        raise ODEBFContractError("P1R30 routing input geometry differs")
    a_mean = request_slopes.mean(axis=0)
    applied_mean = np.asarray(problem.signed_progress, dtype=np.float64)
    raw_applied_mean = P1R30_H * a_mean
    model_coordinate_applied_mean = np.asarray(
        raw_applied_mean, dtype=np.float32
    ).astype(np.float64)
    reduction_residual = float(
        np.max(np.abs(model_coordinate_applied_mean - applied_mean))
    )
    reduction_representation_rounding = float(
        np.max(np.abs(raw_applied_mean - model_coordinate_applied_mean))
    )
    if reduction_residual > P1R30_REDUCTION_TOLERANCE:
        raise ODEBFContractError("P1R30 A0 mean slope reduction differs")
    omega = np.asarray(priority.omega, dtype=np.float64)
    u = np.asarray(priority.u, dtype=np.float64)
    a_debt = omega @ request_slopes
    a_debt_unnormalized = u @ request_slopes
    active = applied_mean > 0.0
    q = float(applied_mean[active].sum())
    c0 = np.zeros_like(applied_mean)
    if rho_reference > 0.0:
        if not np.any(active) or q <= P1R30_EPSILON:
            status = P1R30RoutingStatus.NO_POSITIVE_DIRECTION
            chosen = c0.copy()
            certificates: list[P1R30SolverCertificate] = []
        else:
            # Exact P1R24 A0/Neutral algebra without a per-layer inverse:
            # v0=rho/sum(h*a_l), c0=h*v0 on every positive direction.
            c0[active] = P1R30_H * rho_reference / q
            status = P1R30RoutingStatus.A0_REFERENCE
            chosen = c0.copy()
            certificates = []
    else:
        status = P1R30RoutingStatus.NO_ROUTING_DOF
        chosen = c0.copy()
        certificates = []

    energy_matrix = np.asarray(problem.trust_metric, dtype=np.float64) / (
        P1R30_H**2
    )
    capacity_matrix = np.asarray(problem.capacity_metric, dtype=np.float64) / (
        P1R30_H**2
    )
    p_linear = np.asarray(problem.pretrained.linear, dtype=np.float64) / P1R30_H
    p_gram = np.asarray(problem.pretrained.gram, dtype=np.float64) / (P1R30_H**2)

    def energy(value: np.ndarray) -> float:
        return float(value @ energy_matrix @ value)

    def energy_grad(value: np.ndarray) -> np.ndarray:
        return 2.0 * energy_matrix @ value

    def p_value(value: np.ndarray) -> float:
        return float(
            problem.pretrained.offset
            + 2.0 * p_linear @ value
            + value @ p_gram @ value
        )

    def p_grad(value: np.ndarray) -> np.ndarray:
        return 2.0 * p_linear + 2.0 * p_gram @ value

    def capacity(value: np.ndarray) -> float:
        return float(0.5 * value @ capacity_matrix @ value)

    def capacity_grad(value: np.ndarray) -> np.ndarray:
        return capacity_matrix @ value

    reference_energy = energy(c0)
    energy_limit = (
        reference_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
        + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    )
    reference_mean = float(a_mean @ c0)
    reference_debt = float(a_debt @ c0)
    reference_unnormalized = float(a_debt_unnormalized @ c0)
    require_unnormalized = not priority.all_u_zero

    linear_rows = [a_mean, a_debt]
    if require_unnormalized:
        linear_rows.append(a_debt_unnormalized)
    linear_rank = int(
        np.linalg.matrix_rank(
            np.stack(linear_rows), tol=SIMPLEX_PRIMAL_TOLERANCE
        )
    )
    feasible_dimension = max(len(c0) - linear_rank, 0)

    if (
        selected_arm is FixedE8Arm.SOFT
        and status is P1R30RoutingStatus.A0_REFERENCE
    ):
        constraints: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda value: float(a_mean @ (value - c0)),
                "jac": lambda value: a_mean.copy(),
            },
            {
                "type": "ineq",
                "fun": lambda value: float(a_debt @ (value - c0)),
                "jac": lambda value: a_debt.copy(),
            },
            {
                "type": "ineq",
                "fun": lambda value: float(energy_limit - energy(value)),
                "jac": lambda value: -energy_grad(value),
            },
        ]
        if require_unnormalized:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda value: float(
                        a_debt_unnormalized @ (value - c0)
                    ),
                    "jac": lambda value: a_debt_unnormalized.copy(),
                }
            )
        stage1 = minimize(
            p_value,
            c0.copy(),
            jac=p_grad,
            method="SLSQP",
            bounds=tuple((0.0, None) for _ in c0),
            constraints=tuple(constraints),
            options={
                "disp": False,
                "ftol": FIXED_E8_SOLVER_FTOL,
                "maxiter": FIXED_E8_SOLVER_MAXITER,
            },
        )
        c1 = np.asarray(stage1.x, dtype=np.float64)
        cert1 = _certificate(
            "minimum-marginal-structural-p",
            stage1,
            c1,
            mean_delta=float(a_mean @ (c1 - c0)),
            debt_delta=float(a_debt @ (c1 - c0)),
            unnormalized_delta=float(a_debt_unnormalized @ (c1 - c0)),
            energy_violation=max(energy(c1) - energy_limit, 0.0),
            p_tie_violation=0.0,
            require_unnormalized=require_unnormalized,
        )
        certificates.append(cert1)
        if cert1.passed:
            p_star = p_value(c1)
            p_scale = max(
                float(np.trace(problem.pretrained.gram)), P1R30_EPSILON
            )
            p_tie = SIMPLEX_XI_TIE_TOLERANCE * p_scale
            constraints2 = (
                *constraints,
                {
                    "type": "ineq",
                    "fun": lambda value: float(p_star + p_tie - p_value(value)),
                    "jac": lambda value: -p_grad(value),
                },
            )
            stage2 = minimize(
                capacity,
                c1,
                jac=capacity_grad,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in c0),
                constraints=constraints2,
                options={
                    "disp": False,
                    "ftol": FIXED_E8_SOLVER_FTOL,
                    "maxiter": FIXED_E8_SOLVER_MAXITER,
                },
            )
            c2 = np.asarray(stage2.x, dtype=np.float64)
            cert2 = _certificate(
                "minimum-capacity-inside-structural-p-tie",
                stage2,
                c2,
                mean_delta=float(a_mean @ (c2 - c0)),
                debt_delta=float(a_debt @ (c2 - c0)),
                unnormalized_delta=float(a_debt_unnormalized @ (c2 - c0)),
                energy_violation=max(energy(c2) - energy_limit, 0.0),
                p_tie_violation=max(p_value(c2) - (p_star + p_tie), 0.0),
                require_unnormalized=require_unnormalized,
            )
            certificates.append(cert2)
            if cert2.passed:
                chosen = c2
                if float(np.linalg.norm(chosen - c0)) <= SIMPLEX_PRIMAL_TOLERANCE:
                    status = P1R30RoutingStatus.NO_ROUTING_DOF
                else:
                    status = P1R30RoutingStatus.SOFT_SELECTED
            else:
                chosen = c0.copy()
                status = P1R30RoutingStatus.SOFT_FALLBACK_NEUTRAL
        else:
            chosen = c0.copy()
            status = P1R30RoutingStatus.SOFT_FALLBACK_NEUTRAL
    velocity = chosen / P1R30_H
    selected_mean = float(a_mean @ chosen)
    selected_debt = float(a_debt @ chosen)
    selected_unnormalized = float(a_debt_unnormalized @ chosen)
    selected_energy = energy(chosen)
    reference_p = p_value(c0)
    selected_p = p_value(chosen)
    reference_capacity = capacity(c0)
    selected_capacity = capacity(chosen)
    l1 = float(np.linalg.norm(chosen - c0, ord=1))
    l2 = float(np.linalg.norm(chosen - c0))
    denominator = float(np.linalg.norm(chosen) * np.linalg.norm(c0))
    cosine = (
        float(chosen @ c0 / denominator)
        if denominator > P1R30_EPSILON
        else (1.0 if l2 <= P1R30_EPSILON else 0.0)
    )
    payload = {
        "arm": selected_arm.value,
        "status": status.value,
        "rho_reference": rho_reference,
        "a_mean": a_mean.tolist(),
        "a_debt": a_debt.tolist(),
        "a_debt_unnormalized": a_debt_unnormalized.tolist(),
        "priority_sha256": priority.identity_sha256,
        "c0": c0.tolist(),
        "selected_c": chosen.tolist(),
        "velocity": velocity.tolist(),
        "reference_mean_progress": reference_mean,
        "selected_mean_progress": selected_mean,
        "reference_debt_progress": reference_debt,
        "selected_debt_progress": selected_debt,
        "reference_unnormalized_debt_progress": reference_unnormalized,
        "selected_unnormalized_debt_progress": selected_unnormalized,
        "reference_energy": reference_energy,
        "selected_energy": selected_energy,
        "reference_p": reference_p,
        "selected_p": selected_p,
        "reference_capacity": reference_capacity,
        "selected_capacity": selected_capacity,
        "reduction_residual": reduction_residual,
        "reduction_representation_rounding_max_abs": (
            reduction_representation_rounding
        ),
        "reduction_certificate_coordinate": (
            "FP64_REQUEST_ACCUMULATION_THEN_MODEL_FACING_FP32"
        ),
        "linear_constraint_rank": linear_rank,
        "feasible_dimension": feasible_dimension,
        "certificates": [item.raw_free_payload() for item in certificates],
        "inverse_slope_operation_count": 0,
    }
    return P1R30RoutingResult(
        selected_arm,
        status,
        rho_reference,
        tuple(float(item) for item in a_mean),
        tuple(float(item) for item in a_debt),
        tuple(float(item) for item in a_debt_unnormalized),
        tuple(bool(item) for item in active),
        tuple(float(item) for item in c0),
        tuple(float(item) for item in chosen),
        tuple(float(item) for item in velocity),
        reference_mean,
        selected_mean,
        reference_debt,
        selected_debt,
        reference_unnormalized,
        selected_unnormalized,
        _signed_reference_ratio(selected_mean, reference_mean),
        _signed_reference_ratio(selected_debt, reference_debt),
        reference_energy,
        selected_energy,
        (
            selected_energy / reference_energy
            if reference_energy > P1R30_EPSILON
            else (1.0 if selected_energy <= P1R30_EPSILON else math.inf)
        ),
        reference_p,
        selected_p,
        selected_p - reference_p,
        reference_capacity,
        selected_capacity,
        l1,
        l2,
        cosine,
        reduction_residual,
        reduction_representation_rounding,
        linear_rank,
        feasible_dimension,
        tuple(certificates),
        status is P1R30RoutingStatus.SOFT_FALLBACK_NEUTRAL,
        canonical_hash(payload),
    )


__all__ = [
    "P1R30DebtCompletion",
    "P1R30DebtState",
    "P1R30Priority",
    "P1R30RoutingResult",
    "P1R30RoutingStatus",
    "P1R30TargetObjective",
    "P1R30_INSTRUCTION_ID",
    "P1R30_METHOD_ID",
    "debt_priority",
    "evaluate_p1r30_requestwise_physical_slopes",
    "evaluate_p1r30_target_new_objective",
    "nominal_demand_from_target_displacement",
    "solve_p1r30_a0_relative_routing",
]
