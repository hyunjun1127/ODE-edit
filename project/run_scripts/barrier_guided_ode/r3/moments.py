"""Fine-event sufficient statistics and target-excluded equalities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .errors import NumericalBoundary, R3ScientificBoundary
from .events import FineEventEvaluation
from .reference import W0ConditionalSeal


@dataclass(frozen=True, slots=True)
class FineEventMoments:
    factor: torch.Tensor
    objective_offset: torch.Tensor
    fisher_diagnostic: torch.Tensor
    moving_gradient: torch.Tensor
    equality_directions: torch.Tensor
    equality_rates: torch.Tensor
    target_score: torch.Tensor
    source_score: torch.Tensor
    non_target_score: torch.Tensor
    score_expectation: torch.Tensor
    target_probability: float
    source_probability: float
    minimum_log_probability: float
    maximum_log_probability: float

    def __post_init__(self) -> None:
        event_count, dimension = self.factor.shape
        if (
            self.factor.dtype != torch.float64
            or self.objective_offset.dtype != torch.float64
            or self.objective_offset.shape != (event_count,)
            or self.fisher_diagnostic.shape != (dimension, dimension)
            or self.equality_directions.shape != (dimension, 2)
            or self.equality_rates.tolist() != [1.0, 0.0]
        ):
            raise R3ScientificBoundary("fine-event moment shapes differ")
        tensors = (
            self.factor,
            self.objective_offset,
            self.fisher_diagnostic,
            self.moving_gradient,
            self.equality_directions,
            self.target_score,
            self.source_score,
            self.non_target_score,
            self.score_expectation,
        )
        if any(value.requires_grad or not bool(torch.isfinite(value).all()) for value in tensors):
            raise R3ScientificBoundary("fine-event moments are not detached finite tensors")


def aggregate_fine_event_moments(
    current: FineEventEvaluation,
    *,
    seal: W0ConditionalSeal,
    reference_time: float,
) -> FineEventMoments:
    if current.layout.identity != seal.layout_identity or current.scores is None:
        raise R3ScientificBoundary("moment event identity/scores differ from sealed q0")
    seal.assert_immutable()
    logs = current.log_probabilities
    probabilities = logs.exp()
    if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities > 0.0).all()):
        raise NumericalBoundary(
            "event probability conversion underflowed",
            receipt={"minimum_log_probability": float(logs.min())},
        )
    reference_logs = seal.reference_log_probabilities(reference_time)
    reference_probabilities = reference_logs.exp()
    if not bool(torch.isfinite(reference_probabilities).all()) or not bool((reference_probabilities > 0.0).all()):
        raise NumericalBoundary(
            "reference probability conversion underflowed",
            receipt={"minimum_log_probability": float(reference_logs.min())},
        )
    scores = current.scores
    expectation = (probabilities[:, None] * scores).sum(dim=0)
    target_index = seal.target_index
    target_probability = probabilities[target_index]
    non_target_indices = list(seal.non_target_indices)
    non_target_mass = probabilities[non_target_indices].sum()
    non_target_score = (
        probabilities[non_target_indices, None] * scores[non_target_indices]
    ).sum(dim=0) / non_target_mass
    target_score = scores[target_index]
    source_indices = list(current.layout.source_event_indices)
    source_probability = probabilities[source_indices].sum()
    source_score = (
        probabilities[source_indices, None] * scores[source_indices]
    ).sum(dim=0) / source_probability
    directions = torch.stack(
        (target_score - non_target_score, source_score - non_target_score), dim=1
    ).detach().contiguous()
    rates = torch.tensor([1.0, 0.0], dtype=torch.float64)
    factor = (torch.sqrt(probabilities)[:, None] * scores).detach().contiguous()
    exponent = reference_logs - 0.5 * logs
    offset = -torch.exp(exponent)
    if not bool(torch.isfinite(offset).all()) or not bool((offset < 0.0).all()):
        raise NumericalBoundary(
            "log-space objective factor conversion failed",
            receipt={"minimum_exponent": float(exponent.min()), "maximum_exponent": float(exponent.max())},
        )
    fisher = (factor.transpose(0, 1) @ factor).detach().contiguous()
    fisher = (0.5 * (fisher + fisher.transpose(0, 1))).contiguous()
    gradient = (factor.transpose(0, 1) @ offset).detach().contiguous()
    return FineEventMoments(
        factor=factor,
        objective_offset=offset.detach().contiguous(),
        fisher_diagnostic=fisher,
        moving_gradient=gradient,
        equality_directions=directions,
        equality_rates=rates,
        target_score=target_score.detach().contiguous(),
        source_score=source_score.detach().contiguous(),
        non_target_score=non_target_score.detach().contiguous(),
        score_expectation=expectation.detach().contiguous(),
        target_probability=float(target_probability.cpu().item()),
        source_probability=float(source_probability.cpu().item()),
        minimum_log_probability=float(logs.min().cpu().item()),
        maximum_log_probability=float(logs.max().cpu().item()),
    )


def differential_identity(moments: FineEventMoments, velocity: torch.Tensor) -> dict[str, float]:
    if velocity.dtype != torch.float64 or velocity.shape != (moments.factor.shape[1],):
        raise R3ScientificBoundary("velocity differs from moment coordinates")
    p_y = moments.target_probability
    p_s = moments.source_probability
    s_y = float(torch.dot(moments.target_score, velocity))
    s_s = float(torch.dot(moments.source_score, velocity))
    mu = float(torch.dot(moments.non_target_score, velocity))
    values = {
        "target_score_rate": s_y,
        "source_score_rate": s_s,
        "non_target_score_rate": mu,
        "target_probability_rate": p_y * s_y,
        "source_probability_rate": p_s * s_s,
        "target_closed_form_rate": p_y * (1.0 - p_y),
        "source_closed_form_rate": -p_y * p_s,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise R3ScientificBoundary("differential identity is non-finite")
    return values


__all__ = ["FineEventMoments", "aggregate_fine_event_moments", "differential_identity"]
