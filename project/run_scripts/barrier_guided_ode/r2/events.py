"""Termination-free prefix-resolving event partition for BGODE-R2.

The production-shaped reducer stores one aggregate departure event per trie
node.  It never materializes one Python event per vocabulary token.  A second
tiny-vocabulary implementation is retained as an independent correctness
reference for Stage A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch

from .errors import PrefixEventBoundary


class PrefixEventKind(str, Enum):
    TARGET = "target"
    SOURCE = "source"
    COLLATERAL = "collateral-first-departure"


class PrefixEventMode(str, Enum):
    CYLINDER = "cylinder-leaf"
    EXCLUDED_NEXT_TOKEN_AGGREGATE = "excluded-next-token-aggregate"


@dataclass(frozen=True, slots=True)
class PrefixEventSpec:
    kind: PrefixEventKind
    mode: PrefixEventMode
    prefix: tuple[int, ...]
    continuation_tokens: tuple[int, ...] = ()

    @property
    def identity(self) -> str:
        continuation = ",".join(str(token) for token in self.continuation_tokens)
        prefix = ",".join(str(token) for token in self.prefix)
        return f"{self.kind.value}|{self.mode.value}|{prefix}|{continuation}"


def _token_path(
    values: Sequence[int],
    *,
    name: str,
    output_vocabulary_size: int,
    tokenizer_vocabulary_size: int,
) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise PrefixEventBoundary(f"{name} must be a token sequence") from exc
    if not result:
        raise PrefixEventBoundary(f"{name} must not be empty")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= output_vocabulary_size
        or token >= tokenizer_vocabulary_size
        for token in result
    ):
        raise PrefixEventBoundary(f"{name} contains an invalid output-head token")
    return tuple(int(token) for token in result)


@dataclass(frozen=True, slots=True)
class PrefixEventLayout:
    source_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    internal_prefixes: tuple[tuple[int, ...], ...]
    children_by_prefix: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    events: tuple[PrefixEventSpec, ...]
    target_index: int
    source_index: int
    output_vocabulary_size: int
    tokenizer_vocabulary_size: int
    prefix_relation: str

    @classmethod
    def build(
        cls,
        *,
        source_tokens: Sequence[int],
        target_tokens: Sequence[int],
        output_vocabulary_size: int,
        tokenizer_vocabulary_size: int,
        batch_size: int = 1,
    ) -> "PrefixEventLayout":
        if batch_size != 1:
            raise PrefixEventBoundary("BGODE-R2 supports exactly one request")
        for name, value in (
            ("output_vocabulary_size", output_vocabulary_size),
            ("tokenizer_vocabulary_size", tokenizer_vocabulary_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
                raise PrefixEventBoundary(f"{name} must exceed one")
        source = _token_path(
            source_tokens,
            name="source_tokens",
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        )
        target = _token_path(
            target_tokens,
            name="target_tokens",
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        )
        if source == target:
            raise PrefixEventBoundary("source and target tokenizations are equal")

        source_prefix = len(source) < len(target) and target[: len(source)] == source
        target_prefix = len(target) < len(source) and source[: len(target)] == target
        relation = (
            "source-prefix-of-target"
            if source_prefix
            else "target-prefix-of-source"
            if target_prefix
            else "non-prefix"
        )

        children: dict[tuple[int, ...], set[int]] = {}
        for path in (source, target):
            for index, token in enumerate(path):
                children.setdefault(path[:index], set()).add(token)
        ordered_children = tuple(
            (prefix, tuple(sorted(tokens)))
            for prefix, tokens in sorted(children.items(), key=lambda item: (len(item[0]), item[0]))
        )
        children_map = dict(ordered_children)

        if source_prefix:
            source_event = PrefixEventSpec(
                PrefixEventKind.SOURCE,
                PrefixEventMode.EXCLUDED_NEXT_TOKEN_AGGREGATE,
                source,
                children_map[source],
            )
        else:
            source_event = PrefixEventSpec(
                PrefixEventKind.SOURCE,
                PrefixEventMode.CYLINDER,
                source,
            )
        if target_prefix:
            target_event = PrefixEventSpec(
                PrefixEventKind.TARGET,
                PrefixEventMode.EXCLUDED_NEXT_TOKEN_AGGREGATE,
                target,
                children_map[target],
            )
        else:
            target_event = PrefixEventSpec(
                PrefixEventKind.TARGET,
                PrefixEventMode.CYLINDER,
                target,
            )

        distinguished_aggregate_prefixes = {
            event.prefix
            for event in (target_event, source_event)
            if event.mode is PrefixEventMode.EXCLUDED_NEXT_TOKEN_AGGREGATE
        }
        collateral = tuple(
            PrefixEventSpec(
                PrefixEventKind.COLLATERAL,
                PrefixEventMode.EXCLUDED_NEXT_TOKEN_AGGREGATE,
                prefix,
                next_tokens,
            )
            for prefix, next_tokens in ordered_children
            if prefix not in distinguished_aggregate_prefixes
        )
        events = (target_event, source_event, *collateral)
        if len({event.identity for event in events}) != len(events):
            raise PrefixEventBoundary("event identities are not unique")
        return cls(
            source_tokens=source,
            target_tokens=target,
            internal_prefixes=tuple(prefix for prefix, _ in ordered_children),
            children_by_prefix=ordered_children,
            events=events,
            target_index=0,
            source_index=1,
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
            prefix_relation=relation,
        )

    def children(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        for candidate, children in self.children_by_prefix:
            if candidate == prefix:
                return children
        raise PrefixEventBoundary(f"unknown trie prefix: {prefix}")


@dataclass(frozen=True, slots=True)
class PrefixDirectionalObservation:
    logits: torch.Tensor
    tangent_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logits, torch.Tensor)
            or self.logits.dtype != torch.float32
            or self.logits.ndim != 1
            or self.logits.requires_grad
            or not bool(torch.isfinite(self.logits).all())
        ):
            raise PrefixEventBoundary("logits must be detached finite FP32 [V]")
        tangent = self.tangent_logits
        if tangent is not None and (
            not isinstance(tangent, torch.Tensor)
            or tangent.dtype != torch.float32
            or tangent.ndim != 2
            or tangent.shape[0] != self.logits.numel()
            or tangent.shape[1] <= 0
            or tangent.requires_grad
            or not bool(torch.isfinite(tangent).all())
        ):
            raise PrefixEventBoundary("tangent logits must be detached finite FP32 [V,L]")


@dataclass(frozen=True, slots=True)
class PrefixEventEvaluation:
    layout: PrefixEventLayout
    log_probabilities: torch.Tensor
    scores: torch.Tensor | None
    normalization_log_residual: float
    normalization_tolerance: float
    score_centering_norm: float | None

    def __post_init__(self) -> None:
        event_count = len(self.layout.events)
        if (
            not isinstance(self.log_probabilities, torch.Tensor)
            or self.log_probabilities.dtype != torch.float64
            or self.log_probabilities.shape != (event_count,)
            or self.log_probabilities.requires_grad
            or not bool(torch.isfinite(self.log_probabilities).all())
        ):
            raise PrefixEventBoundary("event log probabilities must be detached finite FP64")
        if self.scores is not None and (
            self.scores.dtype != torch.float64
            or self.scores.ndim != 2
            or self.scores.shape[0] != event_count
            or self.scores.requires_grad
            or not bool(torch.isfinite(self.scores).all())
        ):
            raise PrefixEventBoundary("event scores must be detached finite FP64")
        for value in (self.normalization_log_residual, self.normalization_tolerance):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
                raise PrefixEventBoundary("normalization diagnostics must be finite")
        if self.normalization_log_residual > self.normalization_tolerance:
            raise PrefixEventBoundary("prefix event partition does not normalize")
        if self.score_centering_norm is not None and (
            not math.isfinite(self.score_centering_norm) or self.score_centering_norm < 0.0
        ):
            raise PrefixEventBoundary("score centering norm is invalid")

    @property
    def probabilities(self) -> torch.Tensor:
        return self.log_probabilities.exp()

    @property
    def target_log_probability(self) -> torch.Tensor:
        return self.log_probabilities[self.layout.target_index]

    @property
    def source_log_probability(self) -> torch.Tensor:
        return self.log_probabilities[self.layout.source_index]

    @property
    def log_odds(self) -> torch.Tensor:
        return self.target_log_probability - self.source_log_probability

    @property
    def pair_mass(self) -> torch.Tensor:
        return torch.logaddexp(self.target_log_probability, self.source_log_probability).exp()


def _conditional(
    observation: PrefixDirectionalObservation,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    logits = observation.logits.to(dtype=torch.float64)
    log_probability = logits - torch.logsumexp(logits, dim=0)
    probability = log_probability.exp()
    if observation.tangent_logits is None:
        return log_probability, None, logits
    tangent = observation.tangent_logits.to(dtype=torch.float64)
    scores = tangent - (probability[:, None] * tangent).sum(dim=0, keepdim=True)
    return log_probability, scores, logits


def _excluded_aggregate(
    *,
    log_probability: torch.Tensor,
    scores: torch.Tensor | None,
    logits: torch.Tensor,
    continuation_tokens: tuple[int, ...],
    implementation: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = torch.ones(log_probability.numel(), dtype=torch.bool, device=log_probability.device)
    mask[list(continuation_tokens)] = False
    if not bool(mask.any()):
        raise PrefixEventBoundary("excluded-token aggregate is empty")
    if implementation == "bruteforce":
        aggregate_log = torch.logsumexp(log_probability[mask], dim=0)
    elif implementation == "streaming":
        aggregate_log = torch.logsumexp(logits[mask], dim=0) - torch.logsumexp(logits, dim=0)
    else:
        raise PrefixEventBoundary("unknown event reducer implementation")
    if scores is None:
        return aggregate_log, None
    conditional_weights = torch.softmax(logits[mask], dim=0)
    aggregate_score = (conditional_weights[:, None] * scores[mask]).sum(dim=0)
    return aggregate_log, aggregate_score


def evaluate_prefix_events(
    layout: PrefixEventLayout,
    observations: Mapping[tuple[int, ...], PrefixDirectionalObservation],
    *,
    implementation: str = "streaming",
) -> PrefixEventEvaluation:
    """Evaluate the fixed event space with FP64 log-space reductions."""

    if set(observations) != set(layout.internal_prefixes):
        raise PrefixEventBoundary("observation prefix inventory differs from layout")
    tangent_presence = {observation.tangent_logits is not None for observation in observations.values()}
    if len(tangent_presence) != 1:
        raise PrefixEventBoundary("all prefixes must consistently include or omit tangents")
    actuator_count = None
    prefix_log_mass: dict[tuple[int, ...], torch.Tensor] = {
        (): torch.zeros((), dtype=torch.float64)
    }
    prefix_score: dict[tuple[int, ...], torch.Tensor] = {}
    conditionals: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = {}
    if True in tangent_presence:
        first = next(iter(observations.values())).tangent_logits
        assert first is not None
        actuator_count = int(first.shape[1])
        prefix_score[()] = torch.zeros(actuator_count, dtype=torch.float64)

    for prefix in layout.internal_prefixes:
        if prefix not in prefix_log_mass:
            raise PrefixEventBoundary("reachable prefix mass is missing")
        observation = observations[prefix]
        if observation.logits.numel() != layout.output_vocabulary_size:
            raise PrefixEventBoundary("output-head vocabulary differs from sealed layout")
        if observation.tangent_logits is not None and observation.tangent_logits.shape[1] != actuator_count:
            raise PrefixEventBoundary("actuator width drifted across prefixes")
        logp, score, logits = _conditional(observation)
        conditionals[prefix] = (logp, score, logits)
        for token in layout.children(prefix):
            child = prefix + (token,)
            prefix_log_mass[child] = prefix_log_mass[prefix] + logp[token]
            if score is not None:
                prefix_score[child] = prefix_score[prefix] + score[token]

    values: list[torch.Tensor] = []
    score_values: list[torch.Tensor] = []
    for event in layout.events:
        if event.mode is PrefixEventMode.CYLINDER:
            if event.prefix not in prefix_log_mass:
                raise PrefixEventBoundary("distinguished cylinder was not reached")
            values.append(prefix_log_mass[event.prefix])
            if actuator_count is not None:
                score_values.append(prefix_score[event.prefix])
            continue
        logp, conditional_scores, logits = conditionals[event.prefix]
        aggregate_log, aggregate_score = _excluded_aggregate(
            log_probability=logp,
            scores=conditional_scores,
            logits=logits,
            continuation_tokens=event.continuation_tokens,
            implementation=implementation,
        )
        values.append(prefix_log_mass[event.prefix] + aggregate_log)
        if actuator_count is not None:
            assert aggregate_score is not None
            score_values.append(prefix_score[event.prefix] + aggregate_score)

    log_probabilities = torch.stack(values).detach().to(dtype=torch.float64).contiguous()
    log_total = torch.logsumexp(log_probabilities, dim=0)
    residual = abs(float(log_total.cpu().item()))
    eps = torch.finfo(torch.float64).eps
    scale = max(
        1.0,
        float(torch.max(torch.abs(log_probabilities)).cpu().item()),
        math.log(max(2, len(values))),
    )
    tolerance = 256.0 * eps * scale
    scores_out = None
    centering = None
    if actuator_count is not None:
        scores_out = torch.stack(score_values).detach().to(dtype=torch.float64).contiguous()
        expectation = (log_probabilities.exp()[:, None] * scores_out).sum(dim=0)
        centering = float(torch.linalg.vector_norm(expectation).cpu().item())
    return PrefixEventEvaluation(
        layout=layout,
        log_probabilities=log_probabilities,
        scores=scores_out,
        normalization_log_residual=residual,
        normalization_tolerance=tolerance,
        score_centering_norm=centering,
    )


__all__ = [
    "PrefixDirectionalObservation",
    "PrefixEventEvaluation",
    "PrefixEventKind",
    "PrefixEventLayout",
    "PrefixEventMode",
    "PrefixEventSpec",
    "evaluate_prefix_events",
]
