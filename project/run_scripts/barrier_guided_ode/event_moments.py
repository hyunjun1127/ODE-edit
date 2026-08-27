"""Event sensitivity aggregation for one BGODE-R1 request."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .errors import BGODEScientificBoundary, UnsupportedBatchBoundary
from .event_trie import EventDistribution
from .fisher_pullback import FisherPullback, categorical_fisher_pullback


@dataclass(frozen=True, slots=True)
class EventScoreMoments:
    scores: torch.Tensor
    score_expectation: torch.Tensor
    fisher: FisherPullback
    progress_sensitivity: torch.Tensor
    pair_mass_sensitivity: torch.Tensor
    moving_gradient: torch.Tensor
    anchor_gradient: torch.Tensor
    moving_minus_anchor_parallel_residual: float
    moving_minus_anchor_scale: float

    def __post_init__(self) -> None:
        actuator_count = self.scores.shape[1]
        for name in (
            "score_expectation",
            "progress_sensitivity",
            "pair_mass_sensitivity",
            "moving_gradient",
            "anchor_gradient",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != (actuator_count,)
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise BGODEScientificBoundary(f"{name} is invalid")
        for name in ("moving_minus_anchor_parallel_residual", "moving_minus_anchor_scale"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise BGODEScientificBoundary(f"{name} must be finite")
        if self.moving_minus_anchor_parallel_residual < 0.0:
            raise BGODEScientificBoundary("parallel residual must be non-negative")


def aggregate_event_score_moments(
    current: EventDistribution,
    *,
    initial: EventDistribution,
    reference_log_probabilities: torch.Tensor,
    scores: torch.Tensor,
    batch_size: int = 1,
) -> EventScoreMoments:
    """Aggregate ``S``, ``G``, ``a``, ``b`` and moving/anchor gradients."""

    if batch_size != 1:
        raise UnsupportedBatchBoundary("one shared coefficient vector is undefined for B>1")
    if current.events != initial.events:
        raise BGODEScientificBoundary("current and initial event spaces differ")
    if (
        not isinstance(reference_log_probabilities, torch.Tensor)
        or reference_log_probabilities.dtype != torch.float32
        or reference_log_probabilities.shape != current.log_probabilities.shape
        or reference_log_probabilities.requires_grad
        or not bool(torch.isfinite(reference_log_probabilities).all())
    ):
        raise BGODEScientificBoundary("reference log probabilities are invalid")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or scores.shape[0] != len(current.events)
        or scores.shape[1] == 0
        or scores.requires_grad
        or not bool(torch.isfinite(scores).all())
    ):
        raise BGODEScientificBoundary("scores must be finite FP32 [event,actuator]")
    pi = current.probabilities
    pi0 = initial.probabilities
    pi_star = reference_log_probabilities.exp()
    normalization_tolerance = 64.0 * torch.finfo(torch.float32).eps * len(current.events)
    if abs(float(pi_star.sum().cpu().item()) - 1.0) > normalization_tolerance:
        raise BGODEScientificBoundary("moving reference is not normalized")
    expectation = (pi[:, None] * scores).sum(dim=0)
    fisher = categorical_fisher_pullback(pi, scores)
    iy, is_ = current.target_index, current.source_index
    progress = scores[iy] - scores[is_]
    pair_mass = pi[iy] * scores[iy] + pi[is_] * scores[is_]
    moving = -(pi_star[:, None] * scores).sum(dim=0)
    anchor = -(pi0[:, None] * scores).sum(dim=0)
    difference = moving - anchor
    denominator = torch.dot(progress, progress)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise BGODEScientificBoundary("progress sensitivity collapsed")
    scale = torch.dot(difference, progress) / denominator
    parallel_residual = torch.linalg.vector_norm(difference - scale * progress)
    return EventScoreMoments(
        scores=scores.detach().contiguous(),
        score_expectation=expectation.detach().contiguous(),
        fisher=fisher,
        progress_sensitivity=progress.detach().contiguous(),
        pair_mass_sensitivity=pair_mass.detach().contiguous(),
        moving_gradient=moving.detach().contiguous(),
        anchor_gradient=anchor.detach().contiguous(),
        moving_minus_anchor_parallel_residual=float(parallel_residual.cpu().item()),
        moving_minus_anchor_scale=float(scale.cpu().item()),
    )


__all__ = ["EventScoreMoments", "aggregate_event_score_moments"]
