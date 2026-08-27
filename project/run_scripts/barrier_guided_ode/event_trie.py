"""Joint source/target first-departure event partition for BGODE-R1.

The S0 implementation intentionally materializes tiny-vocabulary departure
events for exhaustive correctness tests.  A production vocabulary reducer can
implement the same :class:`JointFirstDepartureTrie` node interface without
changing event semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch

from .errors import EventPartitionBoundary, UnsupportedBatchBoundary


class EventKind(str, Enum):
    TARGET = "exact-target-leaf"
    SOURCE = "exact-source-leaf"
    DEPARTURE = "first-departure"


@dataclass(frozen=True, slots=True)
class TerminationConvention:
    """A semantically frozen single-token termination convention."""

    name: str
    boundary_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EventPartitionBoundary("termination convention name is empty")
        if isinstance(self.boundary_token, bool) or not isinstance(self.boundary_token, int):
            raise EventPartitionBoundary("boundary token must be an integer token id")
        if self.boundary_token < 0:
            raise EventPartitionBoundary("boundary token must be non-negative")


@dataclass(frozen=True, slots=True)
class EventSpec:
    kind: EventKind
    prefix: tuple[int, ...]
    token: int | None

    @property
    def identity(self) -> str:
        if self.kind is EventKind.DEPARTURE:
            return f"departure:{','.join(map(str, self.prefix))}|{self.token}"
        return f"{self.kind.value}:{','.join(map(str, self.prefix))}"


@dataclass(frozen=True, slots=True)
class EventDistribution:
    """Normalized log probabilities for one ordered event partition."""

    events: tuple[EventSpec, ...]
    log_probabilities: torch.Tensor
    target_index: int
    source_index: int
    normalization_log_residual: float

    def __post_init__(self) -> None:
        if (
            not self.events
            or not isinstance(self.log_probabilities, torch.Tensor)
            or self.log_probabilities.dtype != torch.float32
            or self.log_probabilities.ndim != 1
            or self.log_probabilities.shape[0] != len(self.events)
            or self.log_probabilities.requires_grad
            or not bool(torch.isfinite(self.log_probabilities).all())
        ):
            raise EventPartitionBoundary("event distribution log probabilities are invalid")
        if not (0 <= self.target_index < len(self.events)) or not (
            0 <= self.source_index < len(self.events)
        ):
            raise EventPartitionBoundary("distinguished event index is invalid")
        if self.target_index == self.source_index:
            raise EventPartitionBoundary("target and source events must differ")
        if self.events[self.target_index].kind is not EventKind.TARGET:
            raise EventPartitionBoundary("target index does not bind the target leaf")
        if self.events[self.source_index].kind is not EventKind.SOURCE:
            raise EventPartitionBoundary("source index does not bind the source leaf")
        if not math.isfinite(self.normalization_log_residual) or self.normalization_log_residual < 0:
            raise EventPartitionBoundary("normalization residual is invalid")
        tolerance = 64.0 * torch.finfo(torch.float32).eps * max(1, len(self.events))
        if self.normalization_log_residual > tolerance:
            raise EventPartitionBoundary("event partition does not normalize in log space")

    @property
    def probabilities(self) -> torch.Tensor:
        return self.log_probabilities.exp()

    @property
    def target_log_probability(self) -> torch.Tensor:
        return self.log_probabilities[self.target_index]

    @property
    def source_log_probability(self) -> torch.Tensor:
        return self.log_probabilities[self.source_index]


def _tokens(value: Sequence[int], *, name: str, vocabulary_size: int) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise EventPartitionBoundary(f"{name} must be a token sequence") from exc
    if not result:
        raise EventPartitionBoundary(f"{name} must not be empty")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= vocabulary_size
        for token in result
    ):
        raise EventPartitionBoundary(f"{name} contains an invalid token id")
    return tuple(int(token) for token in result)


class JointFirstDepartureTrie:
    """Two-sequence trie defining a disjoint/exhaustive completion partition."""

    def __init__(
        self,
        *,
        source_tokens: Sequence[int],
        target_tokens: Sequence[int],
        termination: TerminationConvention,
        vocabulary_size: int,
        batch_size: int = 1,
    ) -> None:
        if batch_size != 1:
            raise UnsupportedBatchBoundary("BGODE-R1 event controller requires B=1")
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= 1
        ):
            raise EventPartitionBoundary("vocabulary size must exceed one")
        if not isinstance(termination, TerminationConvention):
            raise EventPartitionBoundary("termination must be typed")
        if termination.boundary_token >= vocabulary_size:
            raise EventPartitionBoundary("boundary token is outside the vocabulary")
        source = _tokens(source_tokens, name="source_tokens", vocabulary_size=vocabulary_size)
        target = _tokens(target_tokens, name="target_tokens", vocabulary_size=vocabulary_size)
        if source == target:
            raise EventPartitionBoundary(
                "source and target have equal tokenization and cannot define two leaves"
            )
        if termination.boundary_token in source or termination.boundary_token in target:
            raise EventPartitionBoundary(
                "the termination token cannot occur inside a distinguished sequence"
            )
        self.source_tokens = source
        self.target_tokens = target
        self.termination = termination
        self.vocabulary_size = vocabulary_size
        self.source_path = source + (termination.boundary_token,)
        self.target_path = target + (termination.boundary_token,)

        children: dict[tuple[int, ...], set[int]] = {}
        for path in (self.source_path, self.target_path):
            for index, token in enumerate(path):
                children.setdefault(path[:index], set()).add(token)
        self._children = {
            prefix: tuple(sorted(tokens))
            for prefix, tokens in sorted(children.items(), key=lambda item: (len(item[0]), item[0]))
        }
        departures = tuple(
            EventSpec(EventKind.DEPARTURE, prefix, token)
            for prefix, child_tokens in self._children.items()
            for token in range(vocabulary_size)
            if token not in child_tokens
        )
        self.events = (
            EventSpec(EventKind.TARGET, self.target_path, None),
            EventSpec(EventKind.SOURCE, self.source_path, None),
            *departures,
        )

    @property
    def internal_prefixes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._children)

    @property
    def child_tokens_by_prefix(self) -> Mapping[tuple[int, ...], tuple[int, ...]]:
        return dict(self._children)

    @property
    def prefix_relation(self) -> str:
        if self.target_tokens[: len(self.source_tokens)] == self.source_tokens:
            return "source-prefix-of-target"
        if self.source_tokens[: len(self.target_tokens)] == self.target_tokens:
            return "target-prefix-of-source"
        return "non-prefix"

    def _normalized_row(
        self,
        logits_by_prefix: Mapping[tuple[int, ...], torch.Tensor],
        prefix: tuple[int, ...],
    ) -> torch.Tensor:
        try:
            logits = logits_by_prefix[prefix]
        except KeyError as exc:
            raise EventPartitionBoundary(f"missing logits for trie prefix {prefix}") from exc
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise EventPartitionBoundary("prefix logits must be floating tensors")
        if logits.ndim == 2:
            if logits.shape[0] != 1:
                raise UnsupportedBatchBoundary("BGODE-R1 logits must have batch dimension one")
            logits = logits[0]
        if (
            logits.ndim != 1
            or logits.shape[0] != self.vocabulary_size
            or logits.requires_grad
            or not bool(torch.isfinite(logits).all())
        ):
            raise EventPartitionBoundary("prefix logits shape/finite contract failed")
        return torch.log_softmax(logits.detach().to(dtype=torch.float32), dim=0)

    def evaluate_bruteforce(
        self,
        logits_by_prefix: Mapping[tuple[int, ...], torch.Tensor],
    ) -> EventDistribution:
        """Materialize tiny-vocabulary event log probabilities exactly once."""

        prefix_mass: dict[tuple[int, ...], torch.Tensor] = {
            (): torch.zeros((), dtype=torch.float32)
        }
        normalized: dict[tuple[int, ...], torch.Tensor] = {}
        for prefix in self.internal_prefixes:
            if prefix not in prefix_mass:
                raise EventPartitionBoundary("trie prefix mass was not reached")
            row = self._normalized_row(logits_by_prefix, prefix)
            normalized[prefix] = row
            for token in self._children[prefix]:
                prefix_mass[prefix + (token,)] = prefix_mass[prefix] + row[token]

        values: list[torch.Tensor] = []
        for event in self.events:
            if event.kind is EventKind.TARGET or event.kind is EventKind.SOURCE:
                try:
                    values.append(prefix_mass[event.prefix])
                except KeyError as exc:
                    raise EventPartitionBoundary("distinguished leaf was not reached") from exc
            else:
                assert event.token is not None
                values.append(prefix_mass[event.prefix] + normalized[event.prefix][event.token])
        log_probabilities = torch.stack(values).detach().contiguous()
        log_total = torch.logsumexp(log_probabilities, dim=0)
        if not bool(torch.isfinite(log_total)):
            raise EventPartitionBoundary("event normalization is non-finite")
        return EventDistribution(
            events=self.events,
            log_probabilities=log_probabilities,
            target_index=0,
            source_index=1,
            normalization_log_residual=abs(float(log_total.cpu().item())),
        )


__all__ = [
    "EventDistribution",
    "EventKind",
    "EventSpec",
    "JointFirstDepartureTrie",
    "TerminationConvention",
]
