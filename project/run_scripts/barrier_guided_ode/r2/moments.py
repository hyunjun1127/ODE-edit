"""FP64 reference path, event moments, and barrier decomposition for R2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .errors import R2ScientificBoundary
from .events import PrefixEventEvaluation


def _finite_scalar(value: torch.Tensor, *, name: str) -> float:
    if value.dtype != torch.float64 or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise R2ScientificBoundary(f"{name} must be a finite FP64 scalar")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise R2ScientificBoundary(f"{name} must be finite")
    return result


def _normalization_tolerance(log_probabilities: torch.Tensor) -> float:
    scale = max(
        1.0,
        float(torch.max(torch.abs(log_probabilities)).cpu().item()),
        math.log(max(2, log_probabilities.numel())),
    )
    return 256.0 * torch.finfo(torch.float64).eps * scale


def _validate_log_distribution(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.ndim != 1
        or value.numel() < 3
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise R2ScientificBoundary(f"{name} must be a detached finite FP64 log distribution")
    residual = abs(float(torch.logsumexp(value, dim=0).cpu().item()))
    if residual > _normalization_tolerance(value):
        raise R2ScientificBoundary(f"{name} is not normalized")
    return value


def forward_kl_fp64(log_reference: torch.Tensor, log_current: torch.Tensor) -> torch.Tensor:
    reference = _validate_log_distribution(log_reference, name="log_reference")
    current = _validate_log_distribution(log_current, name="log_current")
    if reference.shape != current.shape:
        raise R2ScientificBoundary("KL event spaces differ")
    value = torch.sum(reference.exp() * (reference - current))
    tolerance = 512.0 * torch.finfo(torch.float64).eps * max(
        1.0, float(torch.max(torch.abs(reference - current)).cpu().item())
    )
    if not bool(torch.isfinite(value)) or float(value) < -tolerance:
        raise R2ScientificBoundary("forward KL is invalid")
    return torch.clamp_min(value, 0.0)


@dataclass(frozen=True, slots=True)
class PrefixReferencePath:
    log_probabilities: torch.Tensor
    time: float
    initial_log_odds: float
    initial_pair_mass: float


def prefix_reference_path(
    initial: PrefixEventEvaluation,
    *,
    time: float,
) -> PrefixReferencePath:
    if isinstance(time, bool) or not isinstance(time, (int, float)):
        raise R2ScientificBoundary("reference time must be numeric")
    value = float(time)
    if not math.isfinite(value) or value < 0.0:
        raise R2ScientificBoundary("reference time must be finite and nonnegative")
    logs = _validate_log_distribution(initial.log_probabilities, name="initial")
    iy, is_ = initial.layout.target_index, initial.layout.source_index
    log_y, log_s = logs[iy], logs[is_]
    log_pair = torch.logaddexp(log_y, log_s)
    initial_log_odds = log_y - log_s
    expected = initial_log_odds + value
    result = logs.clone()
    result[iy] = log_pair + functional.logsigmoid(expected)
    result[is_] = log_pair + functional.logsigmoid(-expected)
    _validate_log_distribution(result, name="reference")
    return PrefixReferencePath(
        log_probabilities=result.detach().contiguous(),
        time=value,
        initial_log_odds=_finite_scalar(initial_log_odds, name="initial log odds"),
        initial_pair_mass=_finite_scalar(log_pair.exp(), name="initial pair mass"),
    )


@dataclass(frozen=True, slots=True)
class BarrierDecomposition:
    moving_reference_kl: float
    anchored_kl: float
    unavoidable_pair_cost: float
    pair_mass_drift_kl: float
    outside_conditional_event_kl: float
    target_probability: float
    source_probability: float
    pair_mass: float
    log_odds: float
    signed_progress_error: float


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if not bool(((p > 0.0) & (p < 1.0) & (q > 0.0) & (q < 1.0)).all()):
        raise R2ScientificBoundary("Bernoulli KL needs open-interval probabilities")
    return p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def barrier_decomposition(
    initial: PrefixEventEvaluation,
    current: PrefixEventEvaluation,
    *,
    time: float,
) -> BarrierDecomposition:
    if initial.layout.events != current.layout.events:
        raise R2ScientificBoundary("barrier event spaces differ")
    reference = prefix_reference_path(initial, time=time)
    p0 = initial.probabilities
    pw = current.probabilities
    iy, is_ = initial.layout.target_index, initial.layout.source_index
    c0 = p0[iy] + p0[is_]
    cw = pw[iy] + pw[is_]
    s0 = p0[iy] / c0
    st = reference.log_probabilities[iy].exp() / c0
    outside = torch.ones_like(p0, dtype=torch.bool)
    outside[iy] = False
    outside[is_] = False
    if not bool(outside.any()):
        raise R2ScientificBoundary("barrier needs collateral events")
    q0 = p0[outside] / (1.0 - c0)
    qw = pw[outside] / (1.0 - cw)
    outside_kl = torch.sum(q0 * (torch.log(q0) - torch.log(qw)))
    log_odds = current.log_odds
    expected = torch.tensor(reference.initial_log_odds + float(time), dtype=torch.float64)
    return BarrierDecomposition(
        moving_reference_kl=_finite_scalar(
            forward_kl_fp64(reference.log_probabilities, current.log_probabilities),
            name="moving reference KL",
        ),
        anchored_kl=_finite_scalar(
            forward_kl_fp64(initial.log_probabilities, current.log_probabilities),
            name="anchored KL",
        ),
        unavoidable_pair_cost=_finite_scalar(c0 * _bernoulli_kl(s0, st), name="pair cost"),
        pair_mass_drift_kl=_finite_scalar(_bernoulli_kl(c0, cw), name="pair drift KL"),
        outside_conditional_event_kl=_finite_scalar(
            (1.0 - c0) * outside_kl,
            name="outside conditional KL",
        ),
        target_probability=_finite_scalar(pw[iy], name="target probability"),
        source_probability=_finite_scalar(pw[is_], name="source probability"),
        pair_mass=_finite_scalar(cw, name="pair mass"),
        log_odds=_finite_scalar(log_odds, name="log odds"),
        signed_progress_error=_finite_scalar(log_odds - expected, name="progress error"),
    )


@dataclass(frozen=True, slots=True)
class R2EventMoments:
    fisher: torch.Tensor
    moving_gradient: torch.Tensor
    anchor_gradient: torch.Tensor
    progress_direction: torch.Tensor
    normalized_pair_mass_direction: torch.Tensor
    equality_directions: torch.Tensor
    equality_rates: torch.Tensor
    score_expectation: torch.Tensor
    moving_minus_anchor_parallel_residual: float
    target_probability: float
    source_probability: float
    pair_mass: float
    minimum_event_log_probability: float
    maximum_event_log_probability: float

    def __post_init__(self) -> None:
        dimension = self.fisher.shape[0]
        if (
            self.fisher.dtype != torch.float64
            or self.fisher.shape != (dimension, dimension)
            or not bool(torch.isfinite(self.fisher).all())
        ):
            raise R2ScientificBoundary("FP64 Fisher matrix is invalid")
        for name in (
            "moving_gradient",
            "anchor_gradient",
            "progress_direction",
            "normalized_pair_mass_direction",
            "score_expectation",
        ):
            tensor = getattr(self, name)
            if tensor.dtype != torch.float64 or tensor.shape != (dimension,) or not bool(torch.isfinite(tensor).all()):
                raise R2ScientificBoundary(f"{name} is invalid")
        if self.equality_directions.dtype != torch.float64 or self.equality_directions.shape != (dimension, 2):
            raise R2ScientificBoundary("two-equality matrix is invalid")
        if self.equality_rates.dtype != torch.float64 or self.equality_rates.tolist() != [1.0, 0.0]:
            raise R2ScientificBoundary("two-equality rate must be [1,0]")


def aggregate_r2_event_moments(
    current: PrefixEventEvaluation,
    *,
    initial: PrefixEventEvaluation,
    reference_time: float,
) -> R2EventMoments:
    if current.layout.events != initial.layout.events:
        raise R2ScientificBoundary("moment event spaces differ")
    if current.scores is None:
        raise R2ScientificBoundary("current event scores are required")
    scores = current.scores
    probabilities = current.probabilities
    initial_probabilities = initial.probabilities
    reference = prefix_reference_path(initial, time=reference_time)
    reference_probabilities = reference.log_probabilities.exp()
    expectation = (probabilities[:, None] * scores).sum(dim=0)
    fisher_raw = scores.transpose(0, 1) @ (probabilities[:, None] * scores)
    fisher = (0.5 * (fisher_raw + fisher_raw.transpose(0, 1))).detach().contiguous()
    iy, is_ = current.layout.target_index, current.layout.source_index
    target_probability = probabilities[iy]
    source_probability = probabilities[is_]
    pair_mass = target_probability + source_probability
    if not bool(torch.isfinite(pair_mass)) or float(pair_mass) <= 0.0:
        raise R2ScientificBoundary("distinguished pair mass collapsed")
    progress = scores[iy] - scores[is_]
    pair_direction = (
        (target_probability / pair_mass) * scores[iy]
        + (source_probability / pair_mass) * scores[is_]
    )
    moving = -(reference_probabilities[:, None] * scores).sum(dim=0)
    anchor = -(initial_probabilities[:, None] * scores).sum(dim=0)
    difference = moving - anchor
    denominator = torch.dot(progress, progress)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise R2ScientificBoundary("progress direction collapsed")
    parallel_scale = torch.dot(difference, progress) / denominator
    parallel_residual = torch.linalg.vector_norm(difference - parallel_scale * progress)
    directions = torch.stack((progress, pair_direction), dim=1)
    return R2EventMoments(
        fisher=fisher,
        moving_gradient=moving.detach().contiguous(),
        anchor_gradient=anchor.detach().contiguous(),
        progress_direction=progress.detach().contiguous(),
        normalized_pair_mass_direction=pair_direction.detach().contiguous(),
        equality_directions=directions.detach().contiguous(),
        equality_rates=torch.tensor([1.0, 0.0], dtype=torch.float64),
        score_expectation=expectation.detach().contiguous(),
        moving_minus_anchor_parallel_residual=float(parallel_residual.cpu().item()),
        target_probability=_finite_scalar(target_probability, name="target probability"),
        source_probability=_finite_scalar(source_probability, name="source probability"),
        pair_mass=_finite_scalar(pair_mass, name="pair mass"),
        minimum_event_log_probability=float(current.log_probabilities.min().cpu().item()),
        maximum_event_log_probability=float(current.log_probabilities.max().cpu().item()),
    )


__all__ = [
    "BarrierDecomposition",
    "PrefixReferencePath",
    "R2EventMoments",
    "aggregate_r2_event_moments",
    "barrier_decomposition",
    "forward_kl_fp64",
    "prefix_reference_path",
]
