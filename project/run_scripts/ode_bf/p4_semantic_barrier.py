"""Tensor-only P4 smooth semantic log-odds barrier potential.

This module owns the scientific difference between the positive-only and
positive/negative target arms.  It deliberately owns no tokenizer, model,
optimizer, writer, evaluator, history, or held-out decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256


P4_INSTRUCTION_ID = (
    "ODEEDIT-S05-P4-TARGET-SIDE-SMOOTH-SEMANTIC-LOGODDS-BARRIER-V1"
)
P4_METHOD_ID = "P4-TARGET-SIDE-SMOOTH-SEMANTIC-LOGODDS-BARRIER-V1"
P4_CONTEXT_COUNT = 6


class P4TargetArm(str, Enum):
    POSITIVE = "Z+"
    POSITIVE_NEGATIVE = "Z±"


@dataclass(frozen=True, slots=True)
class P4SemanticPotential:
    objective: torch.Tensor
    per_request_objective: torch.Tensor
    new_logprob: torch.Tensor
    old_logprob: torch.Tensor
    pairwise_softplus: torch.Tensor
    sigma: torch.Tensor
    receipt: Mapping[str, Any]


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(device="cpu", dtype=torch.float32).flatten()
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ODEBFContractError("P4 summary values differ")
    return {
        "median": float(torch.quantile(flat, 0.5)),
        "p90": float(torch.quantile(flat, 0.9)),
        "max": float(torch.max(flat)),
        "mean": float(torch.mean(flat)),
    }


def semantic_gradient_coefficients(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    *,
    arm: P4TargetArm | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact per-context dV/ds+ and dV/ds- coefficients."""

    selected = arm if isinstance(arm, P4TargetArm) else P4TargetArm(arm)
    _validate_logprob_pair(new_logprob, old_logprob)
    sigma = torch.sigmoid(old_logprob - new_logprob)
    if selected is P4TargetArm.POSITIVE:
        return -torch.ones_like(new_logprob), torch.zeros_like(old_logprob)
    return -(torch.ones_like(new_logprob) + sigma), sigma


def _validate_logprob_pair(new_logprob: torch.Tensor, old_logprob: torch.Tensor) -> None:
    if (
        not isinstance(new_logprob, torch.Tensor)
        or not isinstance(old_logprob, torch.Tensor)
        or new_logprob.dtype != torch.float32
        or old_logprob.dtype != torch.float32
        or new_logprob.ndim != 2
        or new_logprob.shape != old_logprob.shape
        or new_logprob.shape[0] <= 0
        or new_logprob.shape[1] != P4_CONTEXT_COUNT
        or not bool(torch.isfinite(new_logprob).all())
        or not bool(torch.isfinite(old_logprob).all())
    ):
        raise ODEBFContractError("P4 semantic log-probability geometry differs")


def smooth_semantic_logodds_potential(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    *,
    arm: P4TargetArm | str,
) -> P4SemanticPotential:
    """Evaluate V+ or V+- with equal request and context weighting.

    Inputs are length-normalized teacher-forced log-likelihoods with shape
    ``[request, context]``.  Old scores remain telemetry-only for ``Z+``.
    """

    selected = arm if isinstance(arm, P4TargetArm) else P4TargetArm(arm)
    _validate_logprob_pair(new_logprob, old_logprob)
    gap = old_logprob - new_logprob
    pairwise = F.softplus(gap)
    sigma = torch.sigmoid(gap)
    per_context = -new_logprob
    if selected is P4TargetArm.POSITIVE_NEGATIVE:
        per_context = per_context + pairwise
    per_request = torch.mean(per_context, dim=1)
    objective = torch.mean(per_request)
    if objective.dtype != torch.float32 or not bool(torch.isfinite(objective.detach())):
        raise ODEBFContractError("P4 semantic potential is nonfinite or non-FP32")
    margin = new_logprob - old_logprob
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-smooth-semantic-logodds-potential/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "method_id": P4_METHOD_ID,
        "arm": selected.value,
        "request_count": int(new_logprob.shape[0]),
        "context_count": int(new_logprob.shape[1]),
        "formula": (
            "contexts_mean(-s_plus)"
            if selected is P4TargetArm.POSITIVE
            else "contexts_mean(-s_plus+softplus(s_minus-s_plus))"
        ),
        "barrier_weight": 1.0,
        "margin_threshold_access_count": 0,
        "historical_edit_negative_access_count": 0,
        "old_source": "CURRENT_REQUEST_TARGET_TRUE_ONLY",
        "new_logprob_sha256": tensor_sha256(new_logprob),
        "old_logprob_sha256": tensor_sha256(old_logprob),
        "objective": float(objective.detach()),
        "new_nll_summary": _summary(-new_logprob),
        "old_nll_summary": _summary(-old_logprob),
        "new_minus_old_logprob_margin_summary": _summary(margin),
        "pairwise_softplus_summary": _summary(pairwise),
        "sigma_summary": _summary(sigma),
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return P4SemanticPotential(
        objective,
        per_request,
        new_logprob,
        old_logprob,
        pairwise,
        sigma,
        receipt,
    )


def semantic_gradient_comparison(
    new_only_gradient: torch.Tensor,
    positive_negative_gradient: torch.Tensor,
) -> dict[str, Any]:
    """Record request-wise norms and cosine without creating authority."""

    if (
        new_only_gradient.dtype != torch.float32
        or positive_negative_gradient.dtype != torch.float32
        or new_only_gradient.ndim != 2
        or new_only_gradient.shape != positive_negative_gradient.shape
        or not bool(torch.isfinite(new_only_gradient).all())
        or not bool(torch.isfinite(positive_negative_gradient).all())
    ):
        raise ODEBFContractError("P4 semantic gradient comparison differs")
    plus_norm = torch.linalg.vector_norm(new_only_gradient, dim=0)
    pn_norm = torch.linalg.vector_norm(positive_negative_gradient, dim=0)
    denominator = plus_norm * pn_norm
    cosine = torch.zeros_like(denominator)
    nonzero = denominator > 0.0
    cosine[nonzero] = (
        torch.sum(new_only_gradient * positive_negative_gradient, dim=0)[nonzero]
        / denominator[nonzero]
    )
    if not all(math.isfinite(float(value)) for value in cosine):
        raise ODEBFContractError("P4 semantic gradient cosine is nonfinite")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-semantic-gradient-comparison/v1",
        "new_only_gradient_norm_by_request": [float(item) for item in plus_norm],
        "positive_negative_gradient_norm_by_request": [float(item) for item in pn_norm],
        "gradient_cosine_by_request": [float(item) for item in cosine],
        "observation_only": True,
        "decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P4_CONTEXT_COUNT",
    "P4_INSTRUCTION_ID",
    "P4_METHOD_ID",
    "P4SemanticPotential",
    "P4TargetArm",
    "semantic_gradient_coefficients",
    "semantic_gradient_comparison",
    "smooth_semantic_logodds_potential",
]
