"""Target-excluded first-departure distribution and KL gradient."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from project.run_scripts.barrier_guided_ode.r3.events import FineEventEvaluation


class ConditionalEventBoundary(RuntimeError):
    """The target-excluded event distribution is invalid."""


def _indices(value: FineEventEvaluation) -> tuple[int, ...]:
    target = set(value.layout.target_event_indices)
    result = tuple(index for index in range(value.layout.event_count) if index not in target)
    if not result:
        raise ConditionalEventBoundary("target-excluded event set is empty")
    return result


@dataclass(frozen=True, slots=True)
class TargetExcludedReference:
    layout_identity: str
    non_target_indices: tuple[int, ...]
    log_q0: torch.Tensor

    @classmethod
    def capture(cls, value: FineEventEvaluation) -> "TargetExcludedReference":
        indices = _indices(value)
        logs = value.log_probabilities[list(indices)]
        log_q0 = (logs - torch.logsumexp(logs, dim=0)).detach().to(dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(log_q0).all()):
            raise ConditionalEventBoundary("W0 conditional event reference is non-finite")
        return cls(value.layout.identity, indices, log_q0)

    def evaluate(self, value: FineEventEvaluation) -> tuple[float, torch.Tensor]:
        if value.layout.identity != self.layout_identity or _indices(value) != self.non_target_indices:
            raise ConditionalEventBoundary("event layout drifted from fixed q0")
        if value.scores is None:
            raise ConditionalEventBoundary("barrier gradient requires event directional scores")
        logs = value.log_probabilities[list(self.non_target_indices)]
        log_q = logs - torch.logsumexp(logs, dim=0)
        q = log_q.exp()
        scores = value.scores[list(self.non_target_indices)]
        score_q = scores - (q[:, None] * scores).sum(dim=0, keepdim=True)
        gradient = -(self.log_q0.exp()[:, None] * score_q).sum(dim=0).detach().to(dtype=torch.float64)
        barrier = float((self.log_q0.exp() * (self.log_q0 - log_q)).sum().cpu().item())
        if not torch.isfinite(gradient).all() or not torch.isfinite(torch.tensor(barrier)):
            raise ConditionalEventBoundary("conditional event KL or gradient is non-finite")
        return max(0.0, barrier), gradient.contiguous()

    def observe(self, value: FineEventEvaluation) -> dict[str, float]:
        if value.layout.identity != self.layout_identity:
            raise ConditionalEventBoundary("event layout drifted from fixed q0")
        logs = value.log_probabilities[list(self.non_target_indices)]
        log_q = logs - torch.logsumexp(logs, dim=0)
        barrier = float((self.log_q0.exp() * (self.log_q0 - log_q)).sum().cpu().item())
        return {
            "event_q_kl_from_w0": max(0.0, barrier),
            "source_probability": float(value.source_log_probability.exp()),
            "target_probability": float(value.target_log_probability.exp()),
            "normalization_log_residual": value.normalization_log_residual,
        }
