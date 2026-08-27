"""Thin provenance guard around the approved ordered AlphaEdit factor adapter."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from project.run_scripts.barrier_guided_ode.alphaedit_actuator_interface import (
    AlphaEditActuatorDictionaryReceipt,
    validate_ordered_alphaedit_build,
)
from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import AlphaProposalBuild

from .errors import R2ScientificBoundary


class OrderedProposalProvider(Protocol):
    def propose_ordered(self, *args: object, **kwargs: object) -> AlphaProposalBuild: ...


@dataclass(frozen=True, slots=True)
class ValidatedProposalReceipt:
    build: AlphaProposalBuild
    dictionary: AlphaEditActuatorDictionaryReceipt
    factor_sha256: tuple[str, ...]
    validation_call_count: int

    def __post_init__(self) -> None:
        if (
            self.validation_call_count != 1
            or len(self.factor_sha256) != self.dictionary.factor_count
            or self.dictionary.factor_count != len(self.build.proposal.factors)
        ):
            raise R2ScientificBoundary("ordered factor validation receipt is incomplete")


def _factor_sha256(factor: object) -> str:
    digest = hashlib.sha256()
    for label in ("left", "right"):
        tensor = getattr(factor, label)
        if not isinstance(tensor, torch.Tensor):
            raise R2ScientificBoundary("ordered factor component is not a tensor")
        value = tensor.detach().contiguous().cpu()
        digest.update(label.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def propose_validated_ordered(
    provider: OrderedProposalProvider,
    *args: object,
    **kwargs: object,
) -> ValidatedProposalReceipt:
    proposal = provider.propose_ordered(*args, **kwargs)
    validated = validate_ordered_alphaedit_build(proposal)
    return ValidatedProposalReceipt(
        build=proposal,
        dictionary=validated,
        factor_sha256=tuple(_factor_sha256(factor) for factor in proposal.proposal.factors),
        validation_call_count=1,
    )


@dataclass(frozen=True, slots=True)
class OrderedPrefixMismatch:
    unit_prefix: tuple[float, ...]
    effective_prefix: tuple[float, ...]


def ordered_prefix_mismatch(
    factor_tensors: Sequence[torch.Tensor],
    coefficients: torch.Tensor,
) -> OrderedPrefixMismatch:
    factors = tuple(factor_tensors)
    if len(factors) < 2:
        raise R2ScientificBoundary("ordered mismatch needs at least two factors")
    if (
        coefficients.dtype != torch.float64
        or coefficients.shape != (len(factors),)
        or coefficients.requires_grad
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise R2ScientificBoundary("ordered coefficients must be detached finite FP64")
    if any(
        factor.dtype != torch.float32
        or factor.ndim != 2
        or factor.requires_grad
        or not bool(torch.isfinite(factor).all())
        for factor in factors
    ):
        raise R2ScientificBoundary("ordered factors must be detached finite FP32 matrices")
    unit: list[float] = []
    effective: list[float] = []
    for index, factor in enumerate(factors):
        denominator = max(float(torch.linalg.vector_norm(factor).cpu().item()), torch.finfo(torch.float32).tiny)
        prior = factors[:index]
        unit_value = sum(float(torch.linalg.vector_norm(value).cpu().item()) for value in prior) / denominator
        effective_value = sum(
            abs(float(coefficients[position].cpu().item()))
            * float(torch.linalg.vector_norm(value).cpu().item())
            for position, value in enumerate(prior)
        ) / denominator
        if not math.isfinite(unit_value) or not math.isfinite(effective_value):
            raise R2ScientificBoundary("ordered mismatch is non-finite")
        unit.append(unit_value)
        effective.append(effective_value)
    return OrderedPrefixMismatch(tuple(unit), tuple(effective))


__all__ = [
    "OrderedPrefixMismatch",
    "OrderedProposalProvider",
    "ValidatedProposalReceipt",
    "ordered_prefix_mismatch",
    "propose_validated_ordered",
]
