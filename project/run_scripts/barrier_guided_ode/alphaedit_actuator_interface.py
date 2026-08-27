"""Interface-only binding to the existing ordered genuine AlphaEdit adapter.

This module does not solve AlphaEdit's normal equation.  Production code must
obtain factors from ``ode_edit_motivation.AlphaEditProposalAdapter`` and pass
the resulting typed build through this guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from project.run_scripts.ode_edit_motivation.alphaedit_factors import (
    GENUINE_ISOLATED_ALPHAEDIT_DENSE_SOLVER_PREFIX,
    GENUINE_HISTORICAL_ALPHAEDIT_SOLVER_PREFIX,
    GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX,
)
from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import (
    AlphaProposalBuild,
)
from project.run_scripts.ode_edit_motivation.contracts import ProposalSemantics

from .errors import BGODEScientificBoundary


@runtime_checkable
class OrderedAlphaEditActuatorProvider(Protocol):
    """Minimal production interface; implementation remains in the old adapter."""

    def propose_ordered(
        self,
        *,
        origin_lineage: object,
        construction: str,
        solver_suffix: str,
    ) -> AlphaProposalBuild: ...


@dataclass(frozen=True, slots=True)
class AlphaEditActuatorDictionaryReceipt:
    factor_count: int
    construction: str
    semantics: str
    genuine_p_inside_solve: bool
    ordered_unit_prefix_predictor: bool
    model_mutation_count_on_return: int
    history_append_count: int

    def __post_init__(self) -> None:
        if self.factor_count <= 0:
            raise BGODEScientificBoundary("actuator dictionary must contain factors")
        if self.construction not in {
            "genuine-p-inside-solve",
            "genuine-p-inside-solve-history",
        }:
            raise BGODEScientificBoundary("BGODE rejects post-hoc projector factors")
        if self.semantics != ProposalSemantics.ORDERED_GAUSS_SEIDEL.value:
            raise BGODEScientificBoundary("BGODE requires native ordered rollout semantics")
        if not self.genuine_p_inside_solve or not self.ordered_unit_prefix_predictor:
            raise BGODEScientificBoundary("AlphaEdit actuator provenance is incomplete")
        if self.model_mutation_count_on_return != 0 or self.history_append_count != 0:
            raise BGODEScientificBoundary("dictionary construction leaked state/history")


def validate_ordered_alphaedit_build(build: AlphaProposalBuild) -> AlphaEditActuatorDictionaryReceipt:
    if not isinstance(build, AlphaProposalBuild):
        raise BGODEScientificBoundary("actuator build must come from AlphaEditProposalAdapter")
    proposal = build.proposal
    prefixes = (
        GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX,
        GENUINE_ISOLATED_ALPHAEDIT_DENSE_SOLVER_PREFIX,
        GENUINE_HISTORICAL_ALPHAEDIT_SOLVER_PREFIX,
    )
    if not proposal.solver_name.startswith(prefixes) or "/ordered/" not in proposal.solver_name:
        raise BGODEScientificBoundary("proposal is not a genuine ordered P-inside solve")
    names = tuple(factor.weight_name for factor in proposal.factors)
    if len(names) != len(set(names)):
        raise BGODEScientificBoundary("ordered actuator dictionary has duplicate weights")
    return AlphaEditActuatorDictionaryReceipt(
        factor_count=len(names),
        construction=build.construction,
        semantics=proposal.semantics.value,
        genuine_p_inside_solve=True,
        ordered_unit_prefix_predictor=True,
        model_mutation_count_on_return=0,
        history_append_count=0,
    )


@dataclass(frozen=True, slots=True)
class OrderedPredictorMismatch:
    unit_prefix_norm: float
    effective_prefix_norm: float
    difference_norm: float
    relative_to_unit_prefix: float

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(not isinstance(value, float) or not math.isfinite(value) or value < 0 for value in values):
            raise BGODEScientificBoundary("ordered predictor mismatch telemetry is invalid")
        if self.unit_prefix_norm <= 0.0:
            raise BGODEScientificBoundary("unit prefix norm must be positive")


def ordered_predictor_mismatch(
    unit_prefix_update: torch.Tensor,
    effective_prefix_update: torch.Tensor,
) -> OrderedPredictorMismatch:
    for name, value in (
        ("unit_prefix_update", unit_prefix_update),
        ("effective_prefix_update", effective_prefix_update),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.shape != unit_prefix_update.shape
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
        ):
            raise BGODEScientificBoundary(f"{name} is invalid")
    unit = float(torch.linalg.vector_norm(unit_prefix_update).cpu().item())
    effective = float(torch.linalg.vector_norm(effective_prefix_update).cpu().item())
    difference = float(
        torch.linalg.vector_norm(unit_prefix_update - effective_prefix_update).cpu().item()
    )
    if unit <= 0.0:
        raise BGODEScientificBoundary("unit prefix update collapsed")
    return OrderedPredictorMismatch(
        unit_prefix_norm=unit,
        effective_prefix_norm=effective,
        difference_norm=difference,
        relative_to_unit_prefix=difference / unit,
    )


__all__ = [
    "AlphaEditActuatorDictionaryReceipt",
    "OrderedAlphaEditActuatorProvider",
    "OrderedPredictorMismatch",
    "ordered_predictor_mismatch",
    "validate_ordered_alphaedit_build",
]
