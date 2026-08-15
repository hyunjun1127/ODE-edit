"""Receipt-only h-coordinate accounting for P2R4 Phase-B writer transitions."""

from __future__ import annotations

from typing import Any

from .contracts import ODEBFContractError, canonical_hash


def committed_outer_h_receipt(*, committed_joint_transition: bool) -> dict[str, Any]:
    """Separate one logical outer transition from zero extra numeric h multiplies."""

    if not committed_joint_transition:
        raise ODEBFContractError(
            "P2R4 outer-h receipt requires a committed joint writer transition"
        )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p2r4-phaseb-outer-h-receipt/v1",
        "outer_h_application_count": 1,
        "applied_coordinate_update_count": 1,
        "writer_h_numeric_multiplication_count": 0,
        "physical_h_application_count": 0,
        "second_h_application_count": 0,
        "residual_presplit_count": 0,
        "remaining_division_count": 0,
        "semantic_debt_input_count": 0,
        "definitions": {
            "outer_h_application_count": (
                "one committed logical outer ODE writer transition"
            ),
            "applied_coordinate_update_count": (
                "one selected applied-coordinate DeltaW passed to the joint materializer"
            ),
            "writer_h_numeric_multiplication_count": (
                "numeric multiplication of selected applied-coordinate DeltaW by h"
            ),
            "physical_h_application_count": (
                "legacy no-extra-h numeric multiplication counter"
            ),
            "second_h_application_count": (
                "a second numeric h multiplication after applied-coordinate selection"
            ),
        },
        "decision_influence_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return receipt


__all__ = ["committed_outer_h_receipt"]
