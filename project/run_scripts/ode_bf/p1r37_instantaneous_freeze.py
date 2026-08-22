"""P1R37 current-state instantaneous target-freeze policy.

This module changes only the target-gradient freeze decision.  The inherited
P1R24 target construction remains authoritative and is called with an all-false
carry mask so that the current per-request phi decides the current step.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .p1r24_atomic_strength import (
    P1R24_FREEZE_THRESHOLD,
    P1R24AliasTargetLock,
    P1R24TargetStep,
    p1r24_target_step,
)
from .scalable_batched_field import (
    ScalableBatchGlobalMetric,
    ScalableRobustSharedMetric,
)
from .scalable_batched_model import ScalableObjectiveResult
from .p1r24_atomic_strength import P1R24KLResult


INSTRUCTION_ID = "ODEEDIT-S05-P1R37-P1R36-NO-PERSISTENT-FREEZE-INDEPENDENT-B10X10-V1"
METHOD_ID = "P1R36-P1R35-CURRENT-STATE-INSTANTANEOUS-NO-CARRY-FREEZE-V1"
FREEZE_POLICY = "CURRENT_STATE_INSTANTANEOUS_NO_CARRY"


def _norm_by_request(value: torch.Tensor) -> list[float]:
    value64 = value.detach().to(device="cpu", dtype=torch.float64)
    return [float(item) for item in torch.linalg.vector_norm(value64, dim=0)]


def p1r37_target_step(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    metric: ScalableBatchGlobalMetric | ScalableRobustSharedMetric,
    lock: P1R24AliasTargetLock,
    *,
    step_index: int,
    frozen_mask: Sequence[bool],
    alias: str,
    method: str,
    case_index: int,
) -> P1R24TargetStep:
    """Apply P1R24 using only the current phi threshold for this step."""

    prior = tuple(bool(item) for item in frozen_mask)
    if len(prior) != current_target.shape[1]:
        raise ODEBFContractError("P1R37 prior instantaneous mask geometry differs")
    target_step = p1r24_target_step(
        current_target,
        current_terminal,
        z0,
        nll,
        kl,
        metric,
        lock,
        step_index=step_index,
        frozen_mask=tuple(False for _ in prior),
    )
    phi = tuple(float(item) for item in target_step.receipt["combined_loss_by_request"])
    if not all(math.isfinite(item) for item in phi):
        raise ODEBFContractError("P1R37 current phi is nonfinite")
    instantaneous = tuple(item < P1R24_FREEZE_THRESHOLD for item in phi)
    if target_step.frozen_mask != instantaneous:
        raise ODEBFContractError("P1R37 instantaneous freeze decision differs")

    frozen_to_active = tuple(
        bool(was_frozen and not is_frozen)
        for was_frozen, is_frozen in zip(prior, instantaneous, strict=True)
    )
    active_to_frozen = tuple(
        bool(not was_frozen and is_frozen)
        for was_frozen, is_frozen in zip(prior, instantaneous, strict=True)
    )
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    next64 = target_step.target_next.detach().to(device="cpu", dtype=torch.float64)
    displacement64 = (next64 - current64).contiguous()
    lag64 = (current64 - terminal64).contiguous()
    full_writer_demand64 = (next64 - terminal64).contiguous()
    nll_gradient64 = target_step.nll_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    rho_full_signed_by_request = -torch.sum(
        nll_gradient64 * full_writer_demand64, dim=0
    )
    rho_full_signed = float(torch.sum(rho_full_signed_by_request))
    rho_full = max(rho_full_signed, 0.0)
    if not math.isfinite(rho_full_signed) or not math.isfinite(rho_full):
        raise ODEBFContractError("P1R37 full writer demand rho is nonfinite")

    accepted_k = step_index + 1
    sentinel = bool(
        alias == "llama3-8b-inst"
        and method == "RS-P1R35-SOFT"
        and case_index == 2
        and accepted_k == 7
    )
    payload: dict[str, Any] = {
        **dict(target_step.receipt),
        "schema": "ode-edit-s05-p1r37-current-state-instantaneous-freeze/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "freeze_policy": FREEZE_POLICY,
        "freeze_threshold": P1R24_FREEZE_THRESHOLD,
        "freeze_decision_source": "CURRENT_PHI_ONLY",
        "phi": list(phi),
        "instantaneous_mask": list(instantaneous),
        "prior_step_instantaneous_mask": list(prior),
        "frozen_to_active": list(frozen_to_active),
        "active_to_frozen": list(active_to_frozen),
        "reactivation_count": sum(frozen_to_active),
        "active_to_frozen_count": sum(active_to_frozen),
        "active_request_count": len(instantaneous) - sum(instantaneous),
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
        "carried_frozen_input_decision_influence_count": 0,
        "target_displacement_norm_by_request": _norm_by_request(displacement64),
        "z_y_lag_norm_by_request": _norm_by_request(lag64),
        "full_writer_demand_norm_by_request": _norm_by_request(
            full_writer_demand64
        ),
        "full_writer_rho_signed": rho_full_signed,
        "full_writer_rho_signed_by_request": [
            float(item) for item in rho_full_signed_by_request
        ],
        "full_writer_rho_by_request": [
            max(float(item), 0.0) for item in rho_full_signed_by_request
        ],
        "full_writer_rho": rho_full,
        "finite_rho": True,
        "alias": alias,
        "allocation_arm": method,
        "case_index": case_index,
        "accepted_k": accepted_k,
        "predeclared_sentinel_observation": sentinel,
        "predeclared_sentinel_identity": (
            "P1R36_LLAMA_RS_SOFT_CASE02_K7" if sentinel else None
        ),
        "sentinel_decision_influence_count": 0,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    return replace(target_step, receipt=payload)


__all__ = [
    "FREEZE_POLICY",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "p1r37_target_step",
]
