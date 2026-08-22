"""Thin P4 semantic-objective binding for the raw Euler integrator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .p4_euler_integrator import (
    EulerObjectiveCallback,
    EulerObjectiveEvaluation,
    P4_EULER_INSTRUCTION_ID,
)
from .p4_semantic_barrier import (
    P4_CONTEXT_COUNT,
    P4TargetArm,
    smooth_semantic_logodds_potential,
)


@dataclass(frozen=True, slots=True)
class P4EulerObjectiveTerms:
    """All differentiable terms for one aggregate Euler field evaluation."""

    new_logprob: torch.Tensor
    old_logprob: torch.Tensor
    kl_per_request: torch.Tensor
    decay_per_request: torch.Tensor
    telemetry: Mapping[str, Any]


P4EulerTermsCallback = Callable[[torch.Tensor], P4EulerObjectiveTerms]


def _validate_terms(terms: P4EulerObjectiveTerms, target_state: torch.Tensor) -> None:
    request_count = target_state.shape[1]
    if (
        terms.new_logprob.dtype is not torch.float32
        or terms.old_logprob.dtype is not torch.float32
        or terms.new_logprob.shape != (request_count, P4_CONTEXT_COUNT)
        or terms.old_logprob.shape != terms.new_logprob.shape
        or terms.kl_per_request.dtype is not torch.float32
        or terms.decay_per_request.dtype is not torch.float32
        or terms.kl_per_request.shape != (request_count,)
        or terms.decay_per_request.shape != (request_count,)
        or not bool(torch.isfinite(terms.new_logprob).all())
        or not bool(torch.isfinite(terms.old_logprob).all())
        or not bool(torch.isfinite(terms.kl_per_request).all())
        or not bool(torch.isfinite(terms.decay_per_request).all())
    ):
        raise ODEBFContractError("P4 Euler objective terms differ")


def build_p4_euler_objective_callback(
    terms_callback: P4EulerTermsCallback,
    *,
    arm: P4TargetArm | str,
    kl_factor: float,
    decay_factor: float,
) -> EulerObjectiveCallback:
    """Compose the reused semantic potential with local KL and decay values.

    This function only builds values.  The integrator exclusively owns the
    single aggregate ``torch.autograd.grad`` call at each microstep.
    """

    selected = arm if isinstance(arm, P4TargetArm) else P4TargetArm(arm)
    kl = float(kl_factor)
    decay = float(decay_factor)
    if (
        isinstance(kl_factor, bool)
        or isinstance(decay_factor, bool)
        or not math.isfinite(kl)
        or not math.isfinite(decay)
        or kl < 0.0
        or decay < 0.0
    ):
        raise ODEBFContractError("P4 Euler regularizer coefficient differs")

    def evaluate(target_state: torch.Tensor) -> EulerObjectiveEvaluation:
        terms = terms_callback(target_state)
        _validate_terms(terms, target_state)
        semantic = smooth_semantic_logodds_potential(
            terms.new_logprob,
            terms.old_logprob,
            arm=selected,
        )
        per_request = (
            semantic.per_request_objective
            + kl * terms.kl_per_request
            + decay * terms.decay_per_request
        )
        if per_request.dtype is not torch.float32 or not bool(
            torch.isfinite(per_request).all()
        ):
            raise ODEBFContractError("P4 Euler aggregate objective differs")
        telemetry: dict[str, Any] = {
            "schema": "ode-edit-s05-p4-euler-objective-binding/v1",
            "instruction_id": P4_EULER_INSTRUCTION_ID,
            "arm": selected.value,
            "semantic_component": semantic.receipt,
            "semantic_component_reused_from_adam_independent_formula_module": True,
            "kl_factor": kl,
            "decay_factor": decay,
            "kl_by_request": [
                float(item) for item in terms.kl_per_request.detach().cpu()
            ],
            "decay_by_request": [
                float(item) for item in terms.decay_per_request.detach().cpu()
            ],
            "source_telemetry": dict(terms.telemetry),
            "historical_negative_access_count": 0,
            "heldout_access_count": 0,
        }
        telemetry["identity_sha256"] = canonical_hash(telemetry)
        return EulerObjectiveEvaluation(per_request, telemetry)

    return evaluate


def build_euler_nonsemantic_identity(
    *,
    request_order_sha256: str,
    context_sha256: str,
    teacher_sha256: str,
    origin_sha256: str,
    radius_sha256: str,
    target_layer_name: str,
    kl_factor: float,
    decay_factor: float,
    clamp_factor: float,
    target_horizon: float,
    microsteps: int,
) -> Mapping[str, Any]:
    """Arm-free identity that must be equal for Euler Z+ and Euler Z+- ."""

    for name, value in (
        ("request_order", request_order_sha256),
        ("context", context_sha256),
        ("teacher", teacher_sha256),
        ("origin", origin_sha256),
        ("radius", radius_sha256),
    ):
        if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
            raise ODEBFContractError(f"P4 Euler {name} identity differs")
    if (
        not target_layer_name
        or isinstance(microsteps, bool)
        or not isinstance(microsteps, int)
        or microsteps <= 0
        or not math.isfinite(float(target_horizon))
        or target_horizon <= 0.0
    ):
        raise ODEBFContractError("P4 Euler nonsemantic identity geometry differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-nonsemantic-arm-identity/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "request_order_sha256": request_order_sha256,
        "context_sha256": context_sha256,
        "teacher_sha256": teacher_sha256,
        "origin_sha256": origin_sha256,
        "radius_sha256": radius_sha256,
        "target_layer_name": target_layer_name,
        "kl_factor": float(kl_factor),
        "decay_factor": float(decay_factor),
        "clamp_factor": float(clamp_factor),
        "target_horizon": float(target_horizon),
        "microsteps": microsteps,
        "step_size": float(target_horizon) / microsteps,
        "optimizer": "NONE",
        "fresh_outer_reset": True,
        "final_iterate_only": True,
        "field_normalization_count": 0,
        "heldout_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P4EulerObjectiveTerms",
    "P4EulerTermsCallback",
    "build_euler_nonsemantic_identity",
    "build_p4_euler_objective_callback",
]
