"""BGODE-R3 fine-event, target-excluded CPU mathematical core."""

from .errors import (
    ActuatorBoundary,
    EqualityInfeasible,
    NumericalBoundary,
    R3ScientificBoundary,
)

__all__ = [
    "ActuatorBoundary",
    "EqualityInfeasible",
    "NumericalBoundary",
    "R3ScientificBoundary",
]
