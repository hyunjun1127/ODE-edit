"""Narrow PIR-U policy adapter for the P1R52 sequential runtime.

The adapter owns only the role/method identity and the writer-policy switch.
Target control, Alpha/Woodbury history, Structural-H routing, transactions and
evaluation remain in the existing P1R52 sequential runtime.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1_backend import P1DynamicField
from .p1r52_pir_writer import PIRWriterResult, P1R52PIRPolicy


P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-PIR-U-SEQUENTIAL-10XB100-STRUCTURALH-ON-V1"
)
P1R52_PIRU_SEQUENTIAL_METHOD_ID = (
    "P1R52-REPAIR-R1-SOFT-SEQUENTIAL-PIR-U-STRUCTURAL-H-ON-V1"
)
P1R52_PIRU_SEQUENTIAL_ROLE = "r52-pir-u-soft-sequential-structuralh-on"
P1R52_PIRU_SEQUENTIAL_REPAIR_REVISION = "TECH-R2"
P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX = "pir-u-structuralh-on-tech-r2-v1"
P1R52_PIRU_SEQUENTIAL_RESULT_NAME = (
    "s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-tech-r2-v1"
)
P1R52_PIRU_BATCH_ENTRY_EVALUATOR_ENABLED = False


def bind_piru_sequential_history(
    result: PIRWriterResult,
    entry_field: P1DynamicField,
) -> PIRWriterResult:
    """Rebind PIR q-only layer fields to the immutable entry H geometry."""

    current_fields = tuple(result.layer_fields)
    entry_fields = tuple(entry_field.layers)
    layer_receipts = [dict(value) for value in result.receipt["layers"]]
    if (
        len(current_fields) != len(entry_fields)
        or len(layer_receipts) != len(entry_fields)
    ):
        raise ODEBFContractError("P1R52 PIR-U sequential layer inventory differs")
    rebound = []
    for ordinal, (current, entry) in enumerate(
        zip(current_fields, entry_fields, strict=True)
    ):
        if (
            current.layer != entry.layer
            or current.history_action.ndim != 2
            or entry.history_action.ndim != 2
            or current.history_action.shape[0] != entry.history_action.shape[0]
            or (ordinal > 0 and current.history_action.shape[1] != 0)
            or not torch.isfinite(entry.history_action).all()
        ):
            raise ODEBFContractError(
                "P1R52 PIR-U sequential entry history geometry differs"
            )
        bound = replace(current, history_action=entry.history_action)
        rebound.append(bound)
        layer_receipts[ordinal].update(
            {
                "history_action_source": (
                    "ENTRY_FIELD_REUSE"
                    if ordinal == 0
                    else "ENTRY_STRUCTURAL_H_REBIND"
                ),
                "history_action_width": int(entry.history_action.shape[1]),
                "history_action_sha256": tensor_sha256(entry.history_action),
                "history_action_same_storage_as_entry": bool(
                    bound.history_action.data_ptr() == entry.history_action.data_ptr()
                ),
            }
        )
    receipt = dict(result.receipt)
    receipt.update(
        {
            "layers": layer_receipts,
            "entry_structural_h_history_rebind_count": len(entry_fields) - 1,
            "entry_structural_h_history_rebind_model_forward_count": 0,
            "entry_structural_h_history_rebind_backward_count": 0,
        }
    )
    receipt.pop("identity_sha256", None)
    receipt["identity_sha256"] = canonical_hash(receipt)
    return replace(result, layer_fields=tuple(rebound), receipt=receipt)


def pir_policy_for_role(role: str) -> P1R52PIRPolicy | None:
    """Return the frozen PIR-U writer policy only for the dedicated role."""

    return (
        P1R52PIRPolicy.PIR_U
        if role == P1R52_PIRU_SEQUENTIAL_ROLE
        else None
    )


def is_piru_structural_h_role(role: str) -> bool:
    return role == P1R52_PIRU_SEQUENTIAL_ROLE


def resolve_piru_batch_entry_evaluator_enabled(
    role: str,
    configured: bool | None,
) -> bool:
    """Resolve the PIR-U observation contract independently of run naming."""

    if role != P1R52_PIRU_SEQUENTIAL_ROLE:
        raise ODEBFContractError("P1R52 PIR-U evaluator role differs")
    if configured is not False:
        raise ODEBFContractError(
            "P1R52 PIR-U batch-entry evaluator must be explicitly disabled"
        )
    return False


__all__ = [
    "P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX",
    "P1R52_PIRU_BATCH_ENTRY_EVALUATOR_ENABLED",
    "P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID",
    "P1R52_PIRU_SEQUENTIAL_METHOD_ID",
    "P1R52_PIRU_SEQUENTIAL_REPAIR_REVISION",
    "P1R52_PIRU_SEQUENTIAL_RESULT_NAME",
    "P1R52_PIRU_SEQUENTIAL_ROLE",
    "bind_piru_sequential_history",
    "is_piru_structural_h_role",
    "pir_policy_for_role",
    "resolve_piru_batch_entry_evaluator_enabled",
]
