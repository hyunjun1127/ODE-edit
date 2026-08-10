"""State-pure physical W-only endpoint measurements for P1R17."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_runtime import fixed_e8_waypoint_factors
from .fixed_e8_soft_routing import FIXED_E8_H, FIXED_E8_LAYER_ORDER
from .functional import WaypointFactor, tensor_sha256
from .p1_adaptive_runtime import _merge_factors, _parameter_contract_sha256
from .p1_backend import P1DynamicField, _virtual_context
from .p1_backend import SignedProgressReceipt
from .p1_state import P1HistoryLedger
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, evaluate_routing_objective


@dataclass(frozen=True, slots=True)
class PhysicalEditSlopeReceipt:
    field_sha256: str
    request_order_sha256: str
    context_sha256: str
    baseline_loss: float
    layer_endpoint_loss: tuple[float, ...]
    edit_slopes: tuple[float, ...]
    endpoint_factor_sha256: tuple[str, ...]
    model_forward_count: int
    processed_token_count: int
    backward_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r17-physical-w-only-edit-slope/v1",
            "field_sha256": self.field_sha256,
            "request_order_sha256": self.request_order_sha256,
            "context_sha256": self.context_sha256,
            "layer_order": list(FIXED_E8_LAYER_ORDER),
            "baseline_loss": self.baseline_loss,
            "layer_endpoint_loss": list(self.layer_endpoint_loss),
            "edit_slopes": list(self.edit_slopes),
            "edit_slope_definition": "Phi_edit_W(W_k)-Phi_edit_W(W_k+h*B_k_l)",
            "endpoint_factor_sha256": list(self.endpoint_factor_sha256),
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
            "backward_count": self.backward_count,
            "routing_objective": RoutingObjective.TARGET_NEW_NLL.value,
            "residual_overlay_access_count": 0,
            "heldout_access_count": 0,
            "target_old_access_count": 0,
            "h_application_count": 1,
            "same_bf16_authoritative_basis_for_measurement_and_write": True,
        }
        if canonical_hash(payload) != self.identity_sha256:
            raise ODEBFContractError("physical edit slope identity differs")
        return payload


def _factor_identity(factors: Mapping[str, WaypointFactor]) -> str:
    return canonical_hash(
        {
            name: {
                "layer": value.layer,
                "theta": value.theta,
                "residual_sha256": tensor_sha256(value.residual),
                "q_sha256": tensor_sha256(value.q),
            }
            for name, value in sorted(factors.items())
        }
    )


def measure_physical_edit_slopes(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    field: P1DynamicField,
    *,
    step_index: int,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    contexts: Sequence[Sequence[str]],
    ledger: ComputeLedger,
    touched: Mapping[str, torch.nn.Parameter],
    history: P1HistoryLedger,
    schedule: StatelessReplaySchedule,
    rng_identity: Any,
) -> PhysicalEditSlopeReceipt:
    """Measure the baseline and five unit-velocity physical endpoints.

    The endpoint is evaluated through the same cumulative BF16 virtual-factor
    path used by an authoritative write.  No residual activation hook or
    controller-unseen prompt is available to this function.
    """

    if (
        len(requests) != BATCH_SIZE
        or tuple(int(item.layer) for item in field.layers)
        != FIXED_E8_LAYER_ORDER
    ):
        raise ODEBFContractError("physical edit slope field geometry differs")
    before = (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        rng_identity(),
    )
    counter_before = dict(ledger.counters)
    with _virtual_context(model, cumulative_factors_by_weight):
        baseline = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
    endpoint_values: list[float] = []
    endpoint_hashes: list[str] = []
    endpoint_receipts: list[Any] = []
    for ordinal in range(len(FIXED_E8_LAYER_ORDER)):
        basis = [0.0] * len(FIXED_E8_LAYER_ORDER)
        basis[ordinal] = 1.0
        increment = fixed_e8_waypoint_factors(
            field, basis, step_index=step_index
        )
        candidate = _merge_factors(cumulative_factors_by_weight, increment)
        with _virtual_context(model, candidate):
            observed = evaluate_routing_objective(
                model,
                tokenizer,
                requests,
                objective=RoutingObjective.TARGET_NEW_NLL,
                contexts=contexts,
            )
        endpoint_values.append(float(observed.value))
        endpoint_hashes.append(_factor_identity(increment))
        endpoint_receipts.append(observed)
    after = (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        rng_identity(),
    )
    if after != before:
        raise ODEBFStateError("physical edit slope measurement mutated state")
    all_receipts = (baseline, *endpoint_receipts)
    if any(
        item.request_order_sha256 != baseline.request_order_sha256
        or item.context_sha256 != baseline.context_sha256
        or item.context_group_sizes != (1, 5)
        or item.context_count != 6
        or item.backward_count != 0
        or item.target_true_suffix_token_counts != (None,) * BATCH_SIZE
        for item in all_receipts
    ):
        raise ODEBFContractError("physical edit objective identity differs")
    slopes = tuple(float(baseline.value) - item for item in endpoint_values)
    if not all(math.isfinite(item) for item in (*endpoint_values, *slopes)):
        raise ODEBFContractError("physical edit endpoint is non-finite")
    model_forwards = sum(int(item.model_forward_count) for item in all_receipts)
    processed_tokens = sum(int(item.processed_token_count) for item in all_receipts)
    ledger.increment("model_forward", model_forwards)
    ledger.increment("processed_tokens", processed_tokens)
    if any(
        ledger.counters.get(name, 0) != counter_before.get(name, 0) + expected
        for name, expected in (
            ("model_forward", model_forwards),
            ("processed_tokens", processed_tokens),
        )
    ):
        raise ODEBFStateError("physical edit slope accounting differs")
    payload = {
        "schema": "ode-edit-s05-p1r17-physical-w-only-edit-slope/v1",
        "field_sha256": field.identity_sha256,
        "request_order_sha256": baseline.request_order_sha256,
        "context_sha256": baseline.context_sha256,
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "baseline_loss": float(baseline.value),
        "layer_endpoint_loss": endpoint_values,
        "edit_slopes": list(slopes),
        "edit_slope_definition": "Phi_edit_W(W_k)-Phi_edit_W(W_k+h*B_k_l)",
        "endpoint_factor_sha256": endpoint_hashes,
        "model_forward_count": model_forwards,
        "processed_token_count": processed_tokens,
        "backward_count": 0,
        "routing_objective": RoutingObjective.TARGET_NEW_NLL.value,
        "residual_overlay_access_count": 0,
        "heldout_access_count": 0,
        "target_old_access_count": 0,
        "h_application_count": 1,
        "same_bf16_authoritative_basis_for_measurement_and_write": True,
    }
    return PhysicalEditSlopeReceipt(
        field.identity_sha256,
        baseline.request_order_sha256,
        baseline.context_sha256,
        float(baseline.value),
        tuple(endpoint_values),
        slopes,
        tuple(endpoint_hashes),
        model_forwards,
        processed_tokens,
        0,
        canonical_hash(payload),
    )


def physical_edit_endpoint_contract() -> dict[str, Any]:
    payload = {
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "h": float(FIXED_E8_H),
        "endpoint_count": 1 + len(FIXED_E8_LAYER_ORDER),
        "objective": RoutingObjective.TARGET_NEW_NLL.value,
        "context_group_sizes": [1, 5],
        "residual_overlay_access_count": 0,
        "heldout_access_count": 0,
        "target_old_access_count": 0,
        "h_application_count": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def physical_slope_problem_adapter(
    receipt: PhysicalEditSlopeReceipt,
) -> SignedProgressReceipt:
    """Adapt applied-step endpoint losses to the inherited problem builder.

    ``_build_fixed_e8_problem`` multiplies its input slope by ``h``.  The
    physical measurement already contains that step because each endpoint is
    ``W+h B_l``.  Dividing exactly once here makes the resulting
    ``RoutingProblem.signed_progress`` identical to the measured endpoint
    difference and prevents a second ``h`` from entering the scientific
    objective.
    """

    if len(receipt.edit_slopes) != len(FIXED_E8_LAYER_ORDER):
        raise ODEBFContractError("physical slope adapter dimension differs")
    raw = tuple(float(item) / float(FIXED_E8_H) for item in receipt.edit_slopes)
    if not all(math.isfinite(item) for item in raw):
        raise ODEBFContractError("physical slope adapter is non-finite")
    payload = {
        "schema": "ode-edit-s05-p1r17-physical-slope-problem-adapter/v1",
        "field_sha256": receipt.field_sha256,
        "applied_endpoint_slope_sha256": canonical_hash(
            list(receipt.edit_slopes)
        ),
        "problem_builder_input_slope": list(raw),
        "problem_builder_internal_multiplier": float(FIXED_E8_H),
        "resulting_problem_slope": [
            float(FIXED_E8_H) * item for item in raw
        ],
        "result_exactly_equals_applied_endpoint_slope": True,
        "h_application_count": 1,
    }
    return SignedProgressReceipt(
        receipt.field_sha256,
        raw,
        tuple(
            int(layer)
            for layer, value in zip(
                FIXED_E8_LAYER_ORDER, receipt.edit_slopes, strict=True
            )
            if float(value) <= 0.0
        ),
        canonical_hash(payload),
        receipt.model_forward_count,
        receipt.processed_token_count,
        True,
        RoutingObjective.TARGET_NEW_NLL.value,
        0,
        receipt.identity_sha256,
        receipt.context_sha256,
        6,
        (1, 5),
        0,
    )


__all__ = [
    "PhysicalEditSlopeReceipt",
    "measure_physical_edit_slopes",
    "physical_edit_endpoint_contract",
    "physical_slope_problem_adapter",
]
