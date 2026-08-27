"""Immutable W0 conditional, target-excluded tilt, and KL identities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .errors import NumericalBoundary, R3ScientificBoundary
from .events import FineEventEvaluation


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _check_log_distribution(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.ndim != 1
        or value.numel() < 3
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise R3ScientificBoundary(f"{name} must be detached finite FP64 log probabilities")
    residual = abs(float(torch.logsumexp(value, dim=0).cpu().item()))
    scale = 1.0 + float(torch.max(torch.abs(value)).cpu().item()) + math.log(value.numel())
    tolerance = 128.0 * torch.finfo(torch.float64).eps * scale
    if residual > tolerance:
        raise R3ScientificBoundary(f"{name} is not normalized")
    return value


def forward_kl(log_reference: torch.Tensor, log_current: torch.Tensor) -> torch.Tensor:
    reference = _check_log_distribution(log_reference, name="log_reference")
    current = _check_log_distribution(log_current, name="log_current")
    if reference.shape != current.shape:
        raise R3ScientificBoundary("KL event inventories differ")
    weights = reference.exp()
    if not bool((weights > 0.0).all()) or not bool(torch.isfinite(weights).all()):
        raise NumericalBoundary(
            "reference probability conversion underflowed",
            receipt={"minimum_log_probability": float(reference.min())},
        )
    value = torch.sum(weights * (reference - current))
    tolerance = 256.0 * torch.finfo(torch.float64).eps * (
        1.0 + float(torch.max(torch.abs(reference - current)).cpu().item())
    )
    if not bool(torch.isfinite(value)) or float(value) < -tolerance:
        raise NumericalBoundary("forward KL is invalid", receipt={"value": float(value)})
    return torch.clamp_min(value, 0.0)


@dataclass(frozen=True, slots=True)
class W0ConditionalSeal:
    layout_identity: str
    initial_log_probabilities: torch.Tensor
    target_index: int
    non_target_indices: tuple[int, ...]
    log_q0: torch.Tensor
    initial_target_probability: float
    tensor_identity: str

    @classmethod
    def capture(cls, initial: FineEventEvaluation) -> "W0ConditionalSeal":
        if len(initial.layout.target_event_indices) != 1:
            raise R3ScientificBoundary("R3 target partition requires one collapsed target event")
        logs = _check_log_distribution(initial.log_probabilities, name="initial event")
        target = initial.layout.target_event_indices[0]
        non_target = tuple(index for index in range(logs.numel()) if index != target)
        log_p = logs[target]
        log_one_minus = torch.logsumexp(logs[list(non_target)], dim=0)
        log_q0 = (logs[list(non_target)] - log_one_minus).detach().clone().contiguous()
        p0 = float(log_p.exp().cpu().item())
        if not math.isfinite(p0) or not 0.0 < p0 < 1.0:
            raise NumericalBoundary("initial target probability is not interior", receipt={"p0": p0})
        joined = torch.cat((logs.detach().clone(), log_q0)).contiguous()
        return cls(
            layout_identity=initial.layout.identity,
            initial_log_probabilities=logs.detach().clone().contiguous(),
            target_index=target,
            non_target_indices=non_target,
            log_q0=log_q0,
            initial_target_probability=p0,
            tensor_identity=_tensor_sha(joined),
        )

    def assert_immutable(self) -> None:
        joined = torch.cat((self.initial_log_probabilities, self.log_q0)).contiguous()
        if _tensor_sha(joined) != self.tensor_identity:
            raise R3ScientificBoundary("sealed W0 q0 bytes changed")

    def reference_log_probabilities(self, time: float) -> torch.Tensor:
        self.assert_immutable()
        value = float(time)
        if not math.isfinite(value) or value < 0.0:
            raise R3ScientificBoundary("reference time must be finite and nonnegative")
        log_p0 = self.initial_log_probabilities[self.target_index]
        log_non_target0 = torch.logsumexp(
            self.initial_log_probabilities[list(self.non_target_indices)], dim=0
        )
        log_denominator = torch.logaddexp(log_non_target0, log_p0 + value)
        log_target = log_p0 + value - log_denominator
        log_non_target_mass = log_non_target0 - log_denominator
        result = torch.empty_like(self.initial_log_probabilities)
        result[self.target_index] = log_target
        result[list(self.non_target_indices)] = log_non_target_mass + self.log_q0
        return _check_log_distribution(result.detach().contiguous(), name="target-excluded reference")


@dataclass(frozen=True, slots=True)
class R3BarrierDecomposition:
    moving_kl: float
    bernoulli_target_kl: float
    conditional_kl_weighted: float
    anchored_kl: float
    anchored_bernoulli_kl: float
    anchored_excess: float
    target_probability: float
    source_probability: float


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if not bool(((p > 0.0) & (p < 1.0) & (q > 0.0) & (q < 1.0)).all()):
        raise NumericalBoundary("Bernoulli KL requires interior probabilities", receipt={})
    return p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def barrier_decomposition(
    seal: W0ConditionalSeal,
    current: FineEventEvaluation,
    *,
    time: float,
) -> R3BarrierDecomposition:
    if current.layout.identity != seal.layout_identity:
        raise R3ScientificBoundary("current event identity differs from sealed W0 q0")
    logs = _check_log_distribution(current.log_probabilities, name="current event")
    reference = seal.reference_log_probabilities(time)
    probabilities = logs.exp()
    if not bool((probabilities > 0.0).all()):
        raise NumericalBoundary("current event probability conversion underflowed", receipt={})
    p = probabilities[seal.target_index]
    p_ref = reference[seal.target_index].exp()
    p0 = torch.tensor(seal.initial_target_probability, dtype=torch.float64)
    log_non_target = torch.logsumexp(logs[list(seal.non_target_indices)], dim=0)
    log_q = logs[list(seal.non_target_indices)] - log_non_target
    q0 = seal.log_q0.exp()
    if not bool((q0 > 0.0).all()):
        raise NumericalBoundary("q0 conversion underflowed", receipt={})
    conditional = torch.sum(q0 * (seal.log_q0 - log_q))
    moving = forward_kl(reference, logs)
    bernoulli = _bernoulli_kl(p_ref, p)
    weighted = (1.0 - p_ref) * conditional
    anchored = forward_kl(seal.initial_log_probabilities, logs)
    anchored_bernoulli = _bernoulli_kl(p0, p)
    source_probability = float(
        torch.logsumexp(logs[list(current.layout.source_event_indices)], dim=0).exp().cpu().item()
    )
    return R3BarrierDecomposition(
        moving_kl=float(moving.cpu().item()),
        bernoulli_target_kl=float(bernoulli.cpu().item()),
        conditional_kl_weighted=float(weighted.cpu().item()),
        anchored_kl=float(anchored.cpu().item()),
        anchored_bernoulli_kl=float(anchored_bernoulli.cpu().item()),
        anchored_excess=float((anchored - anchored_bernoulli).cpu().item()),
        target_probability=float(p.cpu().item()),
        source_probability=source_probability,
    )


__all__ = [
    "R3BarrierDecomposition",
    "W0ConditionalSeal",
    "barrier_decomposition",
    "forward_kl",
]
