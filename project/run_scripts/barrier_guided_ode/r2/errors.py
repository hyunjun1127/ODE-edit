"""Typed BGODE-R2 scientific and numerical boundaries."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BGODEScientificBoundary


class R2ScientificBoundary(BGODEScientificBoundary):
    """A locked BGODE-R2 scientific contract was violated."""


class PrefixEventBoundary(R2ScientificBoundary):
    """The termination-free source/target event space is not well-defined."""


class NumericalImplementationBoundary(R2ScientificBoundary):
    """A theorem-backed range identity failed numerically."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


class ActuatorEqualityInfeasible(R2ScientificBoundary):
    """The two actuator equalities are genuinely rank deficient."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


__all__ = [
    "ActuatorEqualityInfeasible",
    "NumericalImplementationBoundary",
    "PrefixEventBoundary",
    "R2ScientificBoundary",
]
