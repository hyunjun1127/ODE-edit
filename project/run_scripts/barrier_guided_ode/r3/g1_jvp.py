"""All-prefix normalized-actuator serial JVP/central-FD gate for G1."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import (
    SerialForwardJVPBackend,
    _forward_ad_attention,
)
from project.run_scripts.barrier_guided_ode.s1_streaming_events import PrefixDirectionalLogits
from project.run_scripts.ode_edit_motivation.contracts import MemitFactorProposal

from .errors import NumericalBoundary, R3ScientificBoundary
from .events import FineEventLayout
from .solver import FactorSpaceSolution, validate_retained_direction_scores


@dataclass(frozen=True, slots=True)
class PrefixFDReceipt:
    prefix: tuple[int, ...]
    direction_count: int
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float
    maximum_absolute_error: float
    allclose: bool


class R3SerialForwardJVPBackend(SerialForwardJVPBackend):
    """Retain the R1 reference backend while validating every internal prefix."""

    def __init__(self, model: torch.nn.Module, tokenization: Any) -> None:
        super().__init__(model, tokenization)
        self.prefix_fd_receipts: list[PrefixFDReceipt] = []
        self._jvp_scores: dict[tuple[int, ...], torch.Tensor] = {}
        self._fd_scores: dict[tuple[int, ...], torch.Tensor] = {}

    @staticmethod
    def _score(logits: torch.Tensor, tangents: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=0)
        return tangents - (logp.exp()[:, None] * tangents).sum(dim=0, keepdim=True)

    def observe_all_prefix_fd(
        self,
        *,
        layout: FineEventLayout,
        proposal: MemitFactorProposal,
    ) -> dict[tuple[int, ...], PrefixDirectionalLogits]:
        if self.prefix_fd_receipts or self._jvp_scores or self._fd_scores:
            raise R3ScientificBoundary("G1 all-prefix JVP backend is single-use")
        started = time.perf_counter()
        result: dict[tuple[int, ...], PrefixDirectionalLogits] = {}
        direction_count = len(proposal.factors)
        device = next(self.model.parameters()).device
        with _forward_ad_attention(self.model) as switched:
            self.ledger.attention_backend_switch_count += int(switched)
            for prefix in layout.internal_prefixes:
                function = self._function(proposal, prefix)
                primals: list[torch.Tensor] = []
                tangents: list[torch.Tensor] = []
                for index in range(direction_count):
                    direction = torch.zeros(direction_count, dtype=torch.float32, device=device)
                    direction[index] = 1.0
                    primal, tangent = self._directional(function, direction)
                    primals.append(primal)
                    tangents.append(tangent)
                primal = primals[0]
                if any(not torch.equal(primal, candidate) for candidate in primals[1:]):
                    raise R3ScientificBoundary("serial JVP primal bytes differ by direction")
                tangent_matrix = torch.stack(tangents, dim=1).to(dtype=torch.float32)
                jvp_scores = self._score(primal, tangent_matrix)
                observed: list[torch.Tensor] = []
                for index in range(direction_count):
                    direction = torch.zeros(direction_count, dtype=torch.float32, device=device)
                    direction[index] = self.FD_EPSILON
                    plus = torch.log_softmax(function(direction), dim=0)
                    minus = torch.log_softmax(function(-direction), dim=0)
                    self.ledger.finite_difference_forward_count += 2
                    observed.append((plus - minus) / (2.0 * self.FD_EPSILON))
                fd_scores = torch.stack(observed, dim=1)
                allclose = bool(
                    torch.allclose(
                        jvp_scores,
                        fd_scores,
                        atol=self.FD_ABSOLUTE_TOLERANCE,
                        rtol=self.FD_RELATIVE_TOLERANCE,
                    )
                )
                maximum = float(torch.max(torch.abs(jvp_scores - fd_scores)).cpu().item())
                receipt = PrefixFDReceipt(
                    prefix=prefix,
                    direction_count=direction_count,
                    epsilon=self.FD_EPSILON,
                    absolute_tolerance=self.FD_ABSOLUTE_TOLERANCE,
                    relative_tolerance=self.FD_RELATIVE_TOLERANCE,
                    maximum_absolute_error=maximum,
                    allclose=allclose,
                )
                self.prefix_fd_receipts.append(receipt)
                if not allclose:
                    raise NumericalBoundary(
                        "all-prefix normalized JVP/central-FD identity failed",
                        receipt=asdict(receipt),
                    )
                self._jvp_scores[prefix] = jvp_scores.detach().cpu().to(dtype=torch.float64).contiguous()
                self._fd_scores[prefix] = fd_scores.detach().cpu().to(dtype=torch.float64).contiguous()
                result[prefix] = PrefixDirectionalLogits(
                    logits=primal.detach().cpu().contiguous(),
                    tangent_logits=tangent_matrix.detach().cpu().contiguous(),
                )
                self.ledger.primal_prefix_count += 1
                self.ledger.score_buffer_peak_bytes = max(
                    self.ledger.score_buffer_peak_bytes,
                    int(tangent_matrix.numel() * tangent_matrix.element_size()),
                )
        torch.cuda.synchronize(device)
        self.ledger.wall_seconds += time.perf_counter() - started
        return result

    def validate_retained(
        self,
        solution: FactorSpaceSolution,
    ) -> tuple[dict[str, object], ...]:
        directions = solution.retained_directions
        rows: list[dict[str, object]] = []
        for prefix in self._jvp_scores:
            receipt = validate_retained_direction_scores(
                self._jvp_scores[prefix],
                self._fd_scores[prefix],
                directions,
                absolute_tolerance=self.FD_ABSOLUTE_TOLERANCE,
                relative_tolerance=self.FD_RELATIVE_TOLERANCE,
            )
            row = asdict(receipt)
            row["prefix"] = list(prefix)
            rows.append(row)
        if len(rows) != len(self.prefix_fd_receipts) or any(
            not math.isfinite(float(row["maximum_absolute_error"])) for row in rows
        ):
            raise R3ScientificBoundary("retained-mode validation receipt is incomplete")
        return tuple(rows)


__all__ = ["PrefixFDReceipt", "R3SerialForwardJVPBackend"]
