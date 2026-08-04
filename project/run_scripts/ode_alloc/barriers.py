"""Separate current-teacher history and theta0-teacher anchor barriers."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .contracts import ODEAllocContractError, finite, positive


_WHITESPACE = re.compile(r"\s+")


def normalize_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ODEAllocContractError("barrier identity text is empty")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


@dataclass(frozen=True, slots=True)
class RequestKey:
    subject: str
    relation: str
    template: str

    @property
    def normalized(self) -> tuple[str, str, str]:
        return (
            normalize_identity(self.subject),
            normalize_identity(self.relation),
            normalize_identity(self.template),
        )


@dataclass(frozen=True, slots=True)
class HistoryItem:
    request_key: RequestKey
    current_teacher_damage: float
    item_id: str

    def __post_init__(self) -> None:
        damage = finite("historical damage", self.current_teacher_damage)
        if damage < 0.0 or not self.item_id:
            raise ODEAllocContractError("historical barrier item is invalid")
        object.__setattr__(self, "current_teacher_damage", damage)


@dataclass(frozen=True, slots=True)
class BarrierSummary:
    raw: tuple[float, ...]
    mean: float
    smooth_max: float
    fixed_scale: float

    @property
    def normalized_mean(self) -> float:
        return self.mean / self.fixed_scale

    @property
    def normalized_smooth_max(self) -> float:
        return self.smooth_max / self.fixed_scale


@dataclass(frozen=True, slots=True)
class DualBarrierReading:
    """H and P remain separately attributed and separately normalized."""

    historical_current_teacher: BarrierSummary
    pretrained_theta0_teacher: BarrierSummary


def remove_obsolete_history(
    history: Iterable[HistoryItem],
    incoming: RequestKey,
) -> tuple[tuple[HistoryItem, ...], tuple[HistoryItem, ...]]:
    """Remove exact subject+relation+template conflicts from the hard H set."""

    target = incoming.normalized
    kept: list[HistoryItem] = []
    audit_only: list[HistoryItem] = []
    for item in history:
        (audit_only if item.request_key.normalized == target else kept).append(item)
    return tuple(kept), tuple(audit_only)


def summarize_small_buffer(
    values: Sequence[float],
    *,
    fixed_scale: float,
    smooth_temperature: float,
) -> BarrierSummary:
    """P1 aggregation for history size 0--3; CVaR is intentionally absent."""

    if len(values) > 3:
        raise ODEAllocContractError("P1 historical buffer exceeds the locked size")
    return _summarize(
        values,
        fixed_scale=fixed_scale,
        smooth_temperature=smooth_temperature,
    )


def _summarize(
    values: Sequence[float],
    *,
    fixed_scale: float,
    smooth_temperature: float,
) -> BarrierSummary:
    scale = positive("barrier fixed scale", fixed_scale)
    temperature = positive("smooth-max temperature", smooth_temperature)
    raw = tuple(finite("barrier damage", value) for value in values)
    if any(value < 0.0 for value in raw):
        raise ODEAllocContractError("barrier damage cannot be negative")
    if not raw:
        return BarrierSummary(raw=(), mean=0.0, smooth_max=0.0, fixed_scale=scale)
    tensor = torch.tensor(raw, dtype=torch.float64)
    # Mean-normalized logsumexp keeps an all-zero buffer at exactly zero.
    smooth = temperature * (
        torch.logsumexp(tensor / temperature, dim=0) - math.log(len(raw))
    )
    return BarrierSummary(
        raw=raw,
        mean=sum(raw) / len(raw),
        smooth_max=float(smooth),
        fixed_scale=scale,
    )


def summarize_pretrained_anchors(
    values: Sequence[float],
    *,
    fixed_scale: float,
    smooth_temperature: float,
    expected_count: int = 16,
) -> BarrierSummary:
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(values) != expected_count
    ):
        raise ODEAllocContractError("pretrained anchor panel count differs")
    return _summarize(
        values,
        fixed_scale=fixed_scale,
        smooth_temperature=smooth_temperature,
    )


def evaluate_dual_barriers(
    *,
    historical_damage: Sequence[float],
    pretrained_damage: Sequence[float],
    historical_fixed_scale: float,
    pretrained_fixed_scale: float,
    smooth_temperature: float,
) -> DualBarrierReading:
    return DualBarrierReading(
        historical_current_teacher=summarize_small_buffer(
            historical_damage,
            fixed_scale=historical_fixed_scale,
            smooth_temperature=smooth_temperature,
        ),
        pretrained_theta0_teacher=summarize_pretrained_anchors(
            pretrained_damage,
            fixed_scale=pretrained_fixed_scale,
            smooth_temperature=smooth_temperature,
        ),
    )


def assert_anchor_disjoint(
    edit_request_hashes: Iterable[str],
    anchor_request_hashes: Iterable[str],
) -> None:
    edits = tuple(edit_request_hashes)
    anchors = tuple(anchor_request_hashes)
    if len(edits) != len(set(edits)) or len(anchors) != len(set(anchors)):
        raise ODEAllocContractError("request or anchor identities repeat")
    overlap = set(edits).intersection(anchors)
    if overlap:
        raise ODEAllocContractError("theta0 anchors overlap edit requests")


def normalize_damage_with_fixed_floor(damage: float, fixed_floor: float) -> float:
    """Use an outcome-free absolute scale, never D(0)+tiny-delta."""

    value = finite("barrier damage", damage)
    floor = positive("barrier fixed floor", fixed_floor)
    if value < 0.0:
        raise ODEAllocContractError("barrier damage cannot be negative")
    return value / floor
