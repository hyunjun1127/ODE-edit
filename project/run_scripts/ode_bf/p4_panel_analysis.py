"""Paired ZA/ZB panel bindings and factual P4 arm deltas."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, canonical_hash


ZA_ARMS = ("Z+", "Z±", "Native-Z")
ZB_ARMS = ("A+", "A±", "Native")


def validate_paired_panel(
    rows: Sequence[Mapping[str, Any]], *, stage: str
) -> Mapping[str, Any]:
    expected = ZA_ARMS if stage == "ZA" else ZB_ARMS if stage == "ZB" else None
    if expected is None or len(rows) != len(expected):
        raise ODEBFContractError("P4 panel inventory differs")
    by_arm = {str(row.get("arm")): row for row in rows}
    if tuple(by_arm) != expected:
        raise ODEBFContractError("P4 panel arm order differs")
    paired_fields = (
        "model_alias",
        "model_revision",
        "sample_order_sha256",
        "evaluator_identity_sha256",
        "w0_sha256",
        "context_sha256",
    )
    for field in paired_fields:
        values = [row.get(field) for row in rows]
        if not isinstance(values[0], str) or len(values[0]) != 64 and field.endswith("sha256"):
            raise ODEBFContractError(f"P4 panel {field} identity differs")
        if len(set(values)) != 1:
            raise ODEBFContractError(f"P4 panel {field} pairing differs")
    payload: dict[str, Any] = {
        "schema": f"ode-edit-s05-p4-{stage.lower()}-paired-panel/v1",
        "stage": stage,
        "arms": list(expected),
        "paired_identity": {field: rows[0][field] for field in paired_fields},
        "heldout_decision_influence_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def factual_barrier_delta(
    positive: Mapping[str, float],
    positive_negative: Mapping[str, float],
    *,
    stage: str,
) -> Mapping[str, Any]:
    if stage not in ("ZA", "ZB") or set(positive) != set(positive_negative) or not positive:
        raise ODEBFContractError("P4 factual delta metric inventory differs")
    deltas: dict[str, float] = {}
    for metric in sorted(positive):
        left = float(positive[metric])
        right = float(positive_negative[metric])
        deltas[metric] = right - left
    payload: dict[str, Any] = {
        "schema": f"ode-edit-s05-p4-{stage.lower()}-barrier-delta/v1",
        "stage": stage,
        "estimand": "Z±-Z+" if stage == "ZA" else "A±-A+",
        "delta_positive_negative_minus_positive": deltas,
        "automatic_promotion": False,
        "margin_only_success_forbidden": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "ZA_ARMS",
    "ZB_ARMS",
    "factual_barrier_delta",
    "validate_paired_panel",
]
