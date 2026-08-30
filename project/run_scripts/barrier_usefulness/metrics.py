"""Reference margin barrier, teacher KL, CVaR and preregistered selectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from .contracts import Selector, ScientificBoundary


def _cvar(values: torch.Tensor, alpha: float) -> tuple[float, list[int]]:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ScientificBoundary("CVaR input is empty/non-finite")
    count = max(1, math.ceil((1.0 - alpha) * values.numel()))
    ordered = torch.argsort(values, descending=True, stable=True)
    return float(values[ordered[:count]].mean().item()), [int(v) for v in ordered[:count].cpu()]


def teacher_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, alpha: float) -> dict[str, object]:
    if reference_logits.shape != candidate_logits.shape:
        raise ScientificBoundary("teacher KL shape mismatch")
    reference_log = torch.log_softmax(reference_logits.float(), dim=-1)
    candidate_log = torch.log_softmax(candidate_logits.float(), dim=-1)
    values = (reference_log.exp() * (reference_log - candidate_log)).sum(-1)
    cvar, tail = _cvar(values, alpha)
    return {
        "per_token": [float(v) for v in values.cpu()],
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "maximum": float(values.max().item()),
        "cvar_0_875": cvar,
        "tail_indices": tail,
        "count": int(values.numel()),
    }


def factual_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(labels.numel(), device=logits.device)
    target = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    return target - masked.max(dim=-1).values


def reference_barrier(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    numerical_floor: float,
) -> dict[str, object]:
    reference = factual_margins(reference_logits.float(), labels)
    candidate = factual_margins(candidate_logits.float(), labels)
    if bool((reference <= numerical_floor).any()):
        raise ScientificBoundary("inadmissible W0 factual margin reached barrier evaluator")
    violations = candidate <= 0
    ratios = candidate / reference
    finite = not bool(violations.any())
    score = float((-torch.log(ratios)).mean().item()) if finite else math.inf
    ordered, _ = torch.sort(ratios)
    lower_index = max(0, math.ceil(0.125 * ratios.numel()) - 1)
    return {
        "score": score,
        "finite": finite,
        "violation_count": int(violations.sum().item()),
        "margin_reference": [float(v) for v in reference.cpu()],
        "margin_candidate": [float(v) for v in candidate.cpu()],
        "slack_ratio": [float(v) for v in ratios.cpu()],
        "minimum_slack": float(ratios.min().item()),
        "lower_0_125_slack": float(ordered[lower_index].item()),
    }


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    barrier: float
    barrier_finite: bool
    kl_cvar: float


def _lexical_min(rows: Iterable[CandidateScore], key: str) -> CandidateScore:
    values = list(rows)
    if not values:
        raise ScientificBoundary("selector has no candidate")
    return min(values, key=lambda row: (getattr(row, key), row.candidate_id))


def select_ctrl(rows: list[CandidateScore], action_representative: str) -> dict[str, str | None]:
    safe = _lexical_min(rows, "barrier")
    finite = [row for row in rows if row.barrier_finite]
    adverse = max(finite, key=lambda row: (row.barrier, tuple(-ord(ch) for ch in row.candidate_id))) if finite else None
    kl = _lexical_min(rows, "kl_cvar")
    if action_representative not in {row.candidate_id for row in rows}:
        raise ScientificBoundary("preregistered action representative absent")
    return {
        Selector.BARRIER_SAFE.value: safe.candidate_id,
        Selector.BARRIER_ADVERSE.value: None if adverse is None else adverse.candidate_id,
        Selector.KL_ONLY.value: kl.candidate_id,
        Selector.ACTION_REPRESENTATIVE.value: action_representative,
        Selector.OFFICIAL.value: "official",
    }
