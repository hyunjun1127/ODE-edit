"""Single-coordinate forward-KL reference path and diagnostic identities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .errors import BGODEScientificBoundary
from .event_trie import EventDistribution


def _scalar(value: torch.Tensor, *, name: str) -> float:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise BGODEScientificBoundary(f"{name} must be a finite scalar")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise BGODEScientificBoundary(f"{name} must be finite")
    return result


def _validate_log_distribution(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 1
        or value.numel() < 2
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise BGODEScientificBoundary(f"{name} must be a finite FP32 log distribution")
    residual = abs(_scalar(torch.logsumexp(value, dim=0), name=f"{name} normalization"))
    tolerance = 64.0 * torch.finfo(torch.float32).eps * value.numel()
    if residual > tolerance:
        raise BGODEScientificBoundary(f"{name} is not normalized")
    return value


def forward_kl(log_reference: torch.Tensor, log_current: torch.Tensor) -> torch.Tensor:
    """Return ``KL(reference || current)`` without length normalization."""

    reference = _validate_log_distribution(log_reference, name="log_reference")
    current = _validate_log_distribution(log_current, name="log_current")
    if reference.shape != current.shape:
        raise BGODEScientificBoundary("KL distributions have different event spaces")
    value = torch.sum(reference.exp() * (reference - current))
    if not bool(torch.isfinite(value)):
        raise BGODEScientificBoundary("forward KL is non-finite")
    tolerance = 128.0 * torch.finfo(torch.float32).eps * reference.numel()
    if float(value) < -tolerance:
        raise BGODEScientificBoundary("forward KL is materially negative")
    return torch.clamp_min(value, 0.0)


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    one = torch.ones((), dtype=torch.float32, device=p.device)
    if not bool(((p > 0) & (p < 1) & (q > 0) & (q < 1)).all()):
        raise BGODEScientificBoundary("Bernoulli KL requires probabilities in (0,1)")
    return p * (torch.log(p) - torch.log(q)) + (one - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


@dataclass(frozen=True, slots=True)
class SingleCoordinateReference:
    log_probabilities: torch.Tensor
    time: float
    initial_log_odds: float
    pair_mass: float
    target_fraction: float

    def __post_init__(self) -> None:
        _validate_log_distribution(self.log_probabilities, name="reference log probabilities")
        for name in ("time", "initial_log_odds", "pair_mass", "target_fraction"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise BGODEScientificBoundary(f"{name} must be finite")
        if self.time < 0.0 or not 0.0 < self.pair_mass < 1.0 or not 0.0 < self.target_fraction < 1.0:
            raise BGODEScientificBoundary("single-coordinate reference scalar boundary failed")


def single_coordinate_reference(
    initial: EventDistribution,
    *,
    time: float,
) -> SingleCoordinateReference:
    if not isinstance(initial, EventDistribution):
        raise BGODEScientificBoundary("initial must be an EventDistribution")
    if isinstance(time, bool) or not isinstance(time, (int, float)):
        raise BGODEScientificBoundary("time must be a non-negative finite scalar")
    t = float(time)
    if not math.isfinite(t) or t < 0.0:
        raise BGODEScientificBoundary("time must be a non-negative finite scalar")
    log_initial = initial.log_probabilities
    log_y = log_initial[initial.target_index]
    log_s = log_initial[initial.source_index]
    log_c0 = torch.logaddexp(log_y, log_s)
    r0 = log_y - log_s
    r_t = r0 + t
    result = log_initial.clone()
    result[initial.target_index] = log_c0 + F.logsigmoid(r_t)
    result[initial.source_index] = log_c0 + F.logsigmoid(-r_t)
    return SingleCoordinateReference(
        log_probabilities=result.detach().contiguous(),
        time=t,
        initial_log_odds=_scalar(r0, name="initial log odds"),
        pair_mass=_scalar(log_c0.exp(), name="initial pair mass"),
        target_fraction=_scalar(torch.sigmoid(r_t), name="target fraction"),
    )


@dataclass(frozen=True, slots=True)
class SingleCoordinateDecomposition:
    anchored_kl: float
    unavoidable_pair_cost: float
    pair_mass_drift: float
    outside_conditional_drift: float
    moving_reference_kl: float
    exact_progress_residual: float

    @property
    def anchored_excess(self) -> float:
        return self.anchored_kl - self.unavoidable_pair_cost


def single_coordinate_decomposition(
    initial: EventDistribution,
    current_log_probabilities: torch.Tensor,
    *,
    time: float,
) -> SingleCoordinateDecomposition:
    current = _validate_log_distribution(current_log_probabilities, name="current")
    if current.shape != initial.log_probabilities.shape:
        raise BGODEScientificBoundary("current event space differs from initial")
    reference = single_coordinate_reference(initial, time=time)
    p0 = initial.probabilities
    pw = current.exp()
    iy, is_ = initial.target_index, initial.source_index
    c0 = p0[iy] + p0[is_]
    cw = pw[iy] + pw[is_]
    s0 = p0[iy] / c0
    st = torch.tensor(reference.target_fraction, dtype=torch.float32)
    outside = torch.ones_like(p0, dtype=torch.bool)
    outside[iy] = False
    outside[is_] = False
    q0 = p0[outside] / (1.0 - c0)
    qw = pw[outside] / (1.0 - cw)
    outside_kl = torch.sum(q0 * (torch.log(q0) - torch.log(qw)))
    r_current = current[iy] - current[is_]
    expected_r = torch.tensor(reference.initial_log_odds + float(time), dtype=torch.float32)
    return SingleCoordinateDecomposition(
        anchored_kl=_scalar(forward_kl(initial.log_probabilities, current), name="anchored KL"),
        unavoidable_pair_cost=_scalar(c0 * _bernoulli_kl(s0, st), name="unavoidable pair cost"),
        pair_mass_drift=_scalar(_bernoulli_kl(c0, cw), name="pair-mass drift"),
        outside_conditional_drift=_scalar((1.0 - c0) * outside_kl, name="outside drift"),
        moving_reference_kl=_scalar(
            forward_kl(reference.log_probabilities, current), name="moving-reference KL"
        ),
        exact_progress_residual=abs(_scalar(r_current - expected_r, name="progress residual")),
    )


@dataclass(frozen=True, slots=True)
class PairMassCeilingDiagnostic:
    initial_pair_mass: float
    native_pair_mass: float
    pair_mass_ratio: float
    native_log_odds: float
    matched_target_probability: float
    native_target_probability: float
    matched_source_probability: float
    native_source_probability: float


def pair_mass_ceiling_diagnostic(
    initial: EventDistribution,
    native_endpoint: EventDistribution,
) -> PairMassCeilingDiagnostic:
    if initial.events != native_endpoint.events:
        raise BGODEScientificBoundary("pair-mass diagnostic event spaces differ")
    p0 = initial.probabilities
    pae = native_endpoint.probabilities
    iy, is_ = initial.target_index, initial.source_index
    c0 = p0[iy] + p0[is_]
    cae = pae[iy] + pae[is_]
    r_ae = native_endpoint.log_probabilities[iy] - native_endpoint.log_probabilities[is_]
    target = c0 * torch.sigmoid(r_ae)
    source = c0 * torch.sigmoid(-r_ae)
    return PairMassCeilingDiagnostic(
        initial_pair_mass=_scalar(c0, name="c0"),
        native_pair_mass=_scalar(cae, name="c_AE"),
        pair_mass_ratio=_scalar(c0 / cae, name="c0/c_AE"),
        native_log_odds=_scalar(r_ae, name="native log odds"),
        matched_target_probability=_scalar(target, name="matched target probability"),
        native_target_probability=_scalar(pae[iy], name="native target probability"),
        matched_source_probability=_scalar(source, name="matched source probability"),
        native_source_probability=_scalar(pae[is_], name="native source probability"),
    )


__all__ = [
    "PairMassCeilingDiagnostic",
    "SingleCoordinateDecomposition",
    "SingleCoordinateReference",
    "forward_kl",
    "pair_mass_ceiling_diagnostic",
    "single_coordinate_decomposition",
    "single_coordinate_reference",
]
