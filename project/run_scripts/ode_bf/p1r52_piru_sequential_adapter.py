"""Narrow PIR-U policy adapter for the P1R52 sequential runtime.

The adapter owns only the role/method identity and the writer-policy switch.
Target control, Alpha/Woodbury history, Structural-H routing, transactions and
evaluation remain in the existing P1R52 sequential runtime.
"""

from __future__ import annotations

from .p1r52_pir_writer import P1R52PIRPolicy


P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-PIR-U-SEQUENTIAL-10XB100-STRUCTURALH-ON-V1"
)
P1R52_PIRU_SEQUENTIAL_METHOD_ID = (
    "P1R52-REPAIR-R1-SOFT-SEQUENTIAL-PIR-U-STRUCTURAL-H-ON-V1"
)
P1R52_PIRU_SEQUENTIAL_ROLE = "r52-pir-u-soft-sequential-structuralh-on"
P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX = "pir-u-structuralh-on-v1"
P1R52_PIRU_SEQUENTIAL_RESULT_NAME = (
    "s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-v1"
)


def pir_policy_for_role(role: str) -> P1R52PIRPolicy | None:
    """Return the frozen PIR-U writer policy only for the dedicated role."""

    return (
        P1R52PIRPolicy.PIR_U
        if role == P1R52_PIRU_SEQUENTIAL_ROLE
        else None
    )


def is_piru_structural_h_role(role: str) -> bool:
    return role == P1R52_PIRU_SEQUENTIAL_ROLE


__all__ = [
    "P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX",
    "P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID",
    "P1R52_PIRU_SEQUENTIAL_METHOD_ID",
    "P1R52_PIRU_SEQUENTIAL_RESULT_NAME",
    "P1R52_PIRU_SEQUENTIAL_ROLE",
    "is_piru_structural_h_role",
    "pir_policy_for_role",
]
