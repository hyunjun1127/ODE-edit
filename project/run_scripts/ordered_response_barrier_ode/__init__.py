"""Ordered Response-Barrier ODE-Edit.

The package deliberately contains only the experiment-specific orchestration.
Official MEMIT/AlphaEdit target construction and solves remain owned by the
stock EasyEdit source or by the already verified ODE-side bridges.
"""

from .contracts import (
    ArmId,
    NumericalMethodBoundary,
    ORBODEContractError,
    ORBODERuntimeLock,
    ScientificBoundary,
    StaleStateBoundary,
    TechnicalBoundary,
    canonical_arm_configs,
)

__all__ = [
    "ArmId",
    "NumericalMethodBoundary",
    "ORBODEContractError",
    "ORBODERuntimeLock",
    "ScientificBoundary",
    "StaleStateBoundary",
    "TechnicalBoundary",
    "canonical_arm_configs",
]
