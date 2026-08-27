"""Typed BGODE-R3 scientific and numerical boundaries."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BGODEScientificBoundary


class R3ScientificBoundary(BGODEScientificBoundary):
    """A locked BGODE-R3 scientific contract was violated."""


class EventSemanticBoundary(R3ScientificBoundary):
    """The fixed fine-event partition is not semantically valid."""


class ActuatorBoundary(R3ScientificBoundary):
    """A native-derived actuator has zero or invalid physical norm."""


class NumericalBoundary(R3ScientificBoundary):
    """A deterministic numerical identity failed its locked bound."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


class EqualityInfeasible(R3ScientificBoundary):
    """The complete equality system is inconsistent."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


__all__ = [
    "ActuatorBoundary",
    "EqualityInfeasible",
    "EventSemanticBoundary",
    "NumericalBoundary",
    "R3ScientificBoundary",
]
