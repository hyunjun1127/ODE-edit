"""Five-dimensional P1 shared writer for the frozen P2 target trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm, RoutingProblem
from .functional import tensor_sha256
from .p1_backend import P1DynamicField, SignedProgressReceipt
from .p1r43_full_strength_routing import (
    P1R43RoutingResult,
    P1R43SemanticNoPositiveDirection,
    solve_p1r43_full_strength_routing,
)
from .scalable_batched_model import (
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    evaluate_scalable_target_new_objective,
)


P2R7_INSTRUCTION_ID = "ODEEDIT-S05-P2R7-P2TARGET-P1DW-SHARED-WRITER-V1"
P2R7_METHOD_ID = "P2TARGET-P1DW-SHARED-WRITER-V1"
P2R7_ROUTING_DIMENSION = 5


class P2R7SharedWriterNoPositiveDirection(RuntimeError):
    """Scientific boundary: positive deficit demand but no positive shared slope."""


@dataclass(frozen=True, slots=True)
class P2R7DeficitWeighting:
    mode: str
    ell_w: tuple[float, ...]
    ell_z: tuple[float, ...]
    deficit: tuple[float, ...]
    omega: tuple[float, ...]
    deficit_sum: float
    rho_omega: float
    no_semantic_deficit: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        values = np.asarray(self.deficit, dtype=np.float64)
        weights = np.asarray(self.omega, dtype=np.float64)
        positive = weights[weights > 0.0]
        payload = {
            "schema": "ode-edit-s05-p2r7-deficit-weighting/v1",
            "instruction_id": P2R7_INSTRUCTION_ID,
            "method_id": P2R7_METHOD_ID,
            "mode": self.mode,
            "ell_w_by_request": list(self.ell_w),
            "ell_z_by_request": list(self.ell_z),
            "deficit_by_request": list(self.deficit),
            "omega_by_request": list(self.omega),
            "deficit_sum": self.deficit_sum,
            "omega_sum": float(weights.sum()),
            "rho_omega": self.rho_omega,
            "rho_sum_omega_d": float(weights @ values),
            "rho_sum_d2_over_sum_d": (
                None
                if self.deficit_sum == 0.0
                else float(values @ values / self.deficit_sum)
            ),
            "no_semantic_deficit": self.no_semantic_deficit,
            "maximum_omega": float(weights.max(initial=0.0)),
            "effective_request_count": (
                0.0 if not positive.size else float(1.0 / (weights @ weights))
            ),
            "deficit_median": float(np.median(values)),
            "deficit_p90": float(np.quantile(values, 0.9)),
            "deficit_maximum": float(values.max(initial=0.0)),
            "weight_temperature_access_count": 0,
            "weight_exponent_access_count": 0,
            "weight_topk_access_count": 0,
            "weight_clipping_count": 0,
            "weight_threshold_access_count": 0,
            "batch_size_demand_multiplier_count": 0,
            "weights_detached": True,
            "identity_sha256": self.identity_sha256,
        }
        return payload


def build_p2r7_deficit_weighting(
    ell_w: Sequence[float],
    ell_z: Sequence[float],
    *,
    mode: str,
) -> P2R7DeficitWeighting:
    if mode not in ("P1DW", "P1AGG"):
        raise ODEBFContractError("P2R7 weighting mode differs")
    w = np.asarray(tuple(float(item) for item in ell_w), dtype=np.float64)
    z = np.asarray(tuple(float(item) for item in ell_z), dtype=np.float64)
    if (
        w.shape != (BATCH_SIZE,)
        or z.shape != (BATCH_SIZE,)
        or not np.all(np.isfinite(w))
        or not np.all(np.isfinite(z))
    ):
        raise ODEBFContractError("P2R7 request objective geometry differs")
    deficit = np.maximum(w - z, 0.0)
    total = float(deficit.sum())
    no_deficit = total == 0.0
    if mode == "P1DW" and not no_deficit:
        omega = deficit / total
    else:
        omega = np.full(BATCH_SIZE, 1.0 / BATCH_SIZE, dtype=np.float64)
    rho = float(omega @ deficit)
    if (
        not math.isfinite(rho)
        or rho < 0.0
        or abs(float(omega.sum()) - 1.0) > 1.0e-12
    ):
        raise ODEBFContractError("P2R7 deficit weighting certificate differs")
    if mode == "P1DW" and not no_deficit:
        closed = float(deficit @ deficit / total)
        if abs(rho - closed) > 1.0e-12:
            raise ODEBFContractError("P2R7 weighted demand identity differs")
    payload = {
        "mode": mode,
        "ell_w": w.tolist(),
        "ell_z": z.tolist(),
        "deficit": deficit.tolist(),
        "omega": omega.tolist(),
        "rho_omega": rho,
        "no_semantic_deficit": no_deficit,
    }
    return P2R7DeficitWeighting(
        mode,
        tuple(float(item) for item in w),
        tuple(float(item) for item in z),
        tuple(float(item) for item in deficit),
        tuple(float(item) for item in omega),
        total,
        rho,
        no_deficit,
        canonical_hash(payload),
    )


def p2r7_weighted_physical_signed_progress(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    field: P1DynamicField,
    weights: Sequence[float],
) -> tuple[SignedProgressReceipt, ScalableObjectiveResult]:
    """One weighted scalar VJP producing exactly five shared-layer slopes."""

    device = next(model.parameters()).device
    coefficients = torch.zeros(
        P2R7_ROUTING_DIMENSION,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = evaluate_scalable_target_new_objective(
        model,
        plan,
        coefficient_layers=field.layers,
        coefficients=coefficients,
        request_weights=weights,
    )
    if observed.coefficient_gradient is None:
        raise ODEBFContractError("P2R7 weighted physical slope gradient is absent")
    gradient = observed.coefficient_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    progress = tuple(float(item) for item in -gradient)
    if len(progress) != P2R7_ROUTING_DIMENSION or any(
        not math.isfinite(item) for item in progress
    ):
        raise ODEBFContractError("P2R7 weighted shared slope differs")
    receipt = SignedProgressReceipt(
        field.identity_sha256,
        progress,
        tuple(
            int(item.layer)
            for item, value in zip(field.layers, progress, strict=True)
            if value <= 0.0
        ),
        tensor_sha256(gradient),
        observed.model_forward_count,
        observed.processed_token_count,
        True,
        "DEFICIT_WEIGHTED_TARGET_NEW_NLL",
        0,
        observed.identity_sha256,
        plan.context_sha256,
        6,
        (1, P2R7_ROUTING_DIMENSION),
        observed.backward_count,
        observed.loss,
    )
    return receipt, observed


def solve_p2r7_shared_routing(
    problem: RoutingProblem,
    *,
    arm: str,
    rho_omega: float,
) -> P1R43RoutingResult:
    requested = FixedE8Arm.NEUTRAL if arm.endswith("NEUTRAL") else FixedE8Arm.SOFT
    try:
        result = solve_p1r43_full_strength_routing(
            problem, arm=requested, alpha_req=rho_omega
        )
    except P1R43SemanticNoPositiveDirection as exc:
        raise P2R7SharedWriterNoPositiveDirection(
            "SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION"
        ) from exc
    if (
        len(result.velocity) != P2R7_ROUTING_DIMENSION
        or abs(result.predicted_progress - rho_omega) > 1.0e-8
    ):
        raise ODEBFContractError("P2R7 shared writer strength identity differs")
    return result


def p2r7_actual_bf16_common_metrics(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.nn.Parameter],
    field: P1DynamicField,
) -> dict[str, Any]:
    """Common observation metrics on the actual accepted BF16 increment."""

    if set(before) != set(after) or {item.weight_name for item in field.layers} != set(after):
        raise ODEBFContractError("P2R7 actual BF16 metric layer inventory differs")
    energy = 0.0
    structural_p = 0.0
    layers: list[dict[str, Any]] = []
    for item in field.layers:
        delta = (
            after[item.weight_name].detach().to(device="cpu", dtype=torch.float32)
            - before[item.weight_name].detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        layer_energy = float(torch.sum(delta.double() * delta.double()))
        left_gram = (
            item.residual.T.to(dtype=torch.float64)
            @ item.residual.to(dtype=torch.float64)
        )
        right_covariance_gram = (
            item.q.T.to(dtype=torch.float64)
            @ item.covariance_action.to(dtype=torch.float64)
        )
        nominal_covariance_quadratic = float(
            torch.sum(left_gram * right_covariance_gram)
        )
        if (
            not math.isfinite(nominal_covariance_quadratic)
            or nominal_covariance_quadratic < -1.0e-8
        ):
            raise ODEBFContractError("P2R7 common covariance factor metric differs")
        if item.factor_frobenius_sq <= 0.0:
            if layer_energy != 0.0:
                raise ODEBFContractError(
                    "P2R7 zero factor has nonzero accepted BF16 energy"
                )
            realized_theta_sq = 0.0
        else:
            realized_theta_sq = layer_energy / item.factor_frobenius_sq
        layer_p = realized_theta_sq * max(nominal_covariance_quadratic, 0.0)
        energy += layer_energy
        structural_p += layer_p
        layers.append(
            {
                "layer": item.layer,
                "weight_name_sha256": canonical_hash(item.weight_name),
                "actual_delta_sha256": tensor_sha256(delta),
                "actual_delta_frobenius_sq": layer_energy,
                "nominal_factor_frobenius_sq": item.factor_frobenius_sq,
                "nominal_factor_covariance_quadratic": max(
                    nominal_covariance_quadratic, 0.0
                ),
                "actual_bf16_energy_equivalent_theta_sq": realized_theta_sq,
                "actual_bf16_energy_calibrated_covariance_quadratic": layer_p,
                "covariance_source_sha256": item.covariance_receipt.source_sha256,
            }
        )
    payload = {
        "schema": "ode-edit-s05-p2r7-actual-bf16-common-metrics/v1",
        "actual_bf16_update_energy": energy,
        "actual_bf16_covariance_structural_p": structural_p,
        "actual_bf16_common_capacity": 0.5 * structural_p,
        "common_capacity_definition": (
            "0.5*SUM_LAYER[(ACTUAL_BF16_INCREMENT_FROBENIUS_SQ/"
            "NOMINAL_FACTOR_FROBENIUS_SQ)*NOMINAL_FACTOR_COVARIANCE_QUADRATIC]"
        ),
        "layers": layers,
        "model_forward_count": 0,
        "model_backward_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def p2r7_forbidden_influence_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p2r7-forbidden-influence/v1",
        "instruction_id": P2R7_INSTRUCTION_ID,
        "routing_variable_count": P2R7_ROUTING_DIMENSION,
        "request_layer_jacobian_variable_count": 0,
        "request_layer_response_matrix_count": 0,
        "p2r6_certified_qp_import_count": 0,
        "p2r6_shadow_solve_count": 0,
        "remaining_horizon_division_count": 0,
        "residual_presplit_count": 0,
        "semantic_debt_input_count": 0,
        "lag_readdition_count": 0,
        "target_hold_freeze_count": 0,
        "quarter_floor_count": 0,
        "layer_upper_cap_count": 0,
        "hard_structural_p_budget_count": 0,
        "functional_p_veto_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "candidate_materialization_count": 0,
        "historical_h_count": 0,
        "online_functional_p_probe_count": 0,
        "residual_inverse_h_count": 1,
        "physical_h_application_count": 1,
        "second_h_application_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P2R7DeficitWeighting",
    "P2R7SharedWriterNoPositiveDirection",
    "P2R7_INSTRUCTION_ID",
    "P2R7_METHOD_ID",
    "P2R7_ROUTING_DIMENSION",
    "build_p2r7_deficit_weighting",
    "p2r7_actual_bf16_common_metrics",
    "p2r7_forbidden_influence_receipt",
    "p2r7_weighted_physical_signed_progress",
    "solve_p2r7_shared_routing",
]
