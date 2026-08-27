"""Canonical fine first-departure partitions and FP64 tensor reducer.

The production-shaped layout uses numeric tensor rows.  It does not allocate a
Python event object for every vocabulary token.  A tiny-vocabulary reference
path is exposed only as an independent G0 correctness oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import torch

from .errors import EventSemanticBoundary


class EventRowMode(IntEnum):
    ATOMIC_DEPARTURE = 0
    LEAF_CYLINDER = 1
    COMPLEMENT_MACRO = 2


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path(
    values: Sequence[int],
    *,
    name: str,
    output_vocabulary_size: int,
    tokenizer_vocabulary_size: int,
) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise EventSemanticBoundary(f"{name} must be a token sequence") from exc
    if not result:
        raise EventSemanticBoundary(f"{name} must not be empty")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= output_vocabulary_size
        or token >= tokenizer_vocabulary_size
        for token in result
    ):
        raise EventSemanticBoundary(f"{name} contains an invalid distinguished token")
    return tuple(int(token) for token in result)


@dataclass(frozen=True, slots=True)
class FineEventLayout:
    source_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    internal_prefixes: tuple[tuple[int, ...], ...]
    children_by_prefix: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    event_prefix_ordinals: torch.Tensor
    event_token_ids: torch.Tensor
    event_modes: torch.Tensor
    macro_excluded_tokens: tuple[tuple[int, ...], ...]
    target_event_indices: tuple[int, ...]
    source_event_indices: tuple[int, ...]
    output_vocabulary_size: int
    tokenizer_vocabulary_size: int
    topology: str
    identity: str

    def __post_init__(self) -> None:
        count = int(self.event_modes.numel())
        if (
            self.event_prefix_ordinals.dtype != torch.int64
            or self.event_token_ids.dtype != torch.int64
            or self.event_modes.dtype != torch.int64
            or self.event_prefix_ordinals.shape != (count,)
            or self.event_token_ids.shape != (count,)
            or len(self.macro_excluded_tokens) != count
            or count < 3
        ):
            raise EventSemanticBoundary("fine-event numeric row inventory is invalid")
        if not self.target_event_indices or not self.source_event_indices:
            raise EventSemanticBoundary("distinguished event membership is empty")
        target = set(self.target_event_indices)
        source = set(self.source_event_indices)
        if target & source or any(index < 0 or index >= count for index in target | source):
            raise EventSemanticBoundary("distinguished event membership overlaps")
        if self.topology == "source-prefix-target" and any(
            EventRowMode(int(self.event_modes[index])) is not EventRowMode.ATOMIC_DEPARTURE
            for index in self.source_event_indices
        ):
            raise EventSemanticBoundary("source-exclusive equality macro lost atomic identity")
        if self.topology == "target-prefix-source" and len(self.target_event_indices) != 1:
            raise EventSemanticBoundary("target-exclusive barrier macro was not collapsed")

    @property
    def event_count(self) -> int:
        return int(self.event_modes.numel())

    def children(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        for candidate, children in self.children_by_prefix:
            if candidate == prefix:
                return children
        raise EventSemanticBoundary(f"unknown trie prefix: {prefix}")

    @classmethod
    def build(
        cls,
        *,
        source_tokens: Sequence[int],
        target_tokens: Sequence[int],
        output_vocabulary_size: int,
        tokenizer_vocabulary_size: int,
        batch_size: int = 1,
    ) -> "FineEventLayout":
        if batch_size != 1:
            raise EventSemanticBoundary("BGODE-R3 supports exactly one request")
        for name, value in (
            ("output_vocabulary_size", output_vocabulary_size),
            ("tokenizer_vocabulary_size", tokenizer_vocabulary_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
                raise EventSemanticBoundary(f"{name} must exceed one")
        source = _path(
            source_tokens,
            name="source_tokens",
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        )
        target = _path(
            target_tokens,
            name="target_tokens",
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        )
        if source == target:
            raise EventSemanticBoundary("source and target tokenizations are equal")

        source_prefix = len(source) < len(target) and target[: len(source)] == source
        target_prefix = len(target) < len(source) and source[: len(target)] == target
        shared = 0
        for left, right in zip(source, target):
            if left != right:
                break
            shared += 1
        topology = (
            "source-prefix-target"
            if source_prefix
            else "target-prefix-source"
            if target_prefix
            else "common-prefix-divergence"
            if shared > 0
            else "unequal-non-prefix"
        )

        children: dict[tuple[int, ...], set[int]] = {}
        for candidate in (source, target):
            for ordinal, token in enumerate(candidate):
                children.setdefault(candidate[:ordinal], set()).add(token)
        ordered_children = tuple(
            (prefix, tuple(sorted(tokens)))
            for prefix, tokens in sorted(children.items(), key=lambda item: (len(item[0]), item[0]))
        )
        prefix_ordinal = {prefix: index for index, (prefix, _) in enumerate(ordered_children)}

        row_prefixes: list[int] = []
        row_tokens: list[int] = []
        row_modes: list[int] = []
        excluded: list[tuple[int, ...]] = []
        target_indices: list[int] = []
        source_indices: list[int] = []

        def add(prefix: tuple[int, ...], token: int, mode: EventRowMode, blocked: tuple[int, ...] = ()) -> int:
            row_prefixes.append(prefix_ordinal.get(prefix, -1))
            row_tokens.append(token)
            row_modes.append(int(mode))
            excluded.append(blocked)
            return len(row_modes) - 1

        for prefix, next_tokens in ordered_children:
            if target_prefix and prefix == target:
                # One target-exclusive macro in the barrier partition.
                index = add(prefix, -1, EventRowMode.COMPLEMENT_MACRO, next_tokens)
                target_indices.append(index)
                continue
            blocked = set(next_tokens)
            indices = [
                add(prefix, token, EventRowMode.ATOMIC_DEPARTURE)
                for token in range(output_vocabulary_size)
                if token not in blocked
            ]
            if source_prefix and prefix == source:
                source_indices.extend(indices)

        if not source_prefix:
            source_indices.append(add(source, -1, EventRowMode.LEAF_CYLINDER))
        if not target_prefix:
            target_indices.append(add(target, -1, EventRowMode.LEAF_CYLINDER))

        payload = {
            "source_tokens": source,
            "target_tokens": target,
            "prefixes": ordered_children,
            "row_prefixes": row_prefixes,
            "row_tokens": row_tokens,
            "row_modes": row_modes,
            "excluded": excluded,
            "target_group": target_indices,
            "source_group": source_indices,
            "output_vocabulary_size": output_vocabulary_size,
            "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
            "topology": topology,
            "termination_convention": "NONE_PREFIX_RESOLVING",
        }
        return cls(
            source_tokens=source,
            target_tokens=target,
            internal_prefixes=tuple(prefix for prefix, _ in ordered_children),
            children_by_prefix=ordered_children,
            event_prefix_ordinals=torch.tensor(row_prefixes, dtype=torch.int64),
            event_token_ids=torch.tensor(row_tokens, dtype=torch.int64),
            event_modes=torch.tensor(row_modes, dtype=torch.int64),
            macro_excluded_tokens=tuple(excluded),
            target_event_indices=tuple(target_indices),
            source_event_indices=tuple(source_indices),
            output_vocabulary_size=output_vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
            topology=topology,
            identity=_canonical_sha(payload),
        )


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
            raise EventSemanticBoundary("logits must be detached finite FP32 [V]")
        tangent = self.tangent_logits
        if tangent is not None and (
            tangent.dtype != torch.float32
            or tangent.ndim != 2
            or tangent.shape[0] != self.logits.numel()
            or tangent.shape[1] <= 0
            or tangent.requires_grad
            or not bool(torch.isfinite(tangent).all())
        ):
            raise EventSemanticBoundary("tangent logits must be detached finite FP32 [V,L]")


@dataclass(frozen=True, slots=True)
class FineEventEvaluation:
    layout: FineEventLayout
    log_probabilities: torch.Tensor
    scores: torch.Tensor | None
    normalization_log_residual: float
    normalization_tolerance: float
    score_centering_norm: float | None
    reducer: str

    def __post_init__(self) -> None:
        if (
            self.log_probabilities.dtype != torch.float64
            or self.log_probabilities.shape != (self.layout.event_count,)
            or self.log_probabilities.requires_grad
            or not bool(torch.isfinite(self.log_probabilities).all())
        ):
            raise EventSemanticBoundary("event log probabilities must be detached finite FP64")
        if self.scores is not None and (
            self.scores.dtype != torch.float64
            or self.scores.ndim != 2
            or self.scores.shape[0] != self.layout.event_count
            or self.scores.requires_grad
            or not bool(torch.isfinite(self.scores).all())
        ):
            raise EventSemanticBoundary("event scores must be detached finite FP64")
        if self.normalization_log_residual > self.normalization_tolerance:
            raise EventSemanticBoundary("fine-event partition does not normalize")

    @property
    def probabilities(self) -> torch.Tensor:
        return self.log_probabilities.exp()

    def group_log_probability(self, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.logsumexp(self.log_probabilities[list(indices)], dim=0)

    def group_score(self, indices: tuple[int, ...]) -> torch.Tensor:
        if self.scores is None:
            raise EventSemanticBoundary("event scores are not available")
        logs = self.log_probabilities[list(indices)]
        weights = torch.softmax(logs, dim=0)
        return (weights[:, None] * self.scores[list(indices)]).sum(dim=0)

    @property
    def target_log_probability(self) -> torch.Tensor:
        return self.group_log_probability(self.layout.target_event_indices)

    @property
    def source_log_probability(self) -> torch.Tensor:
        return self.group_log_probability(self.layout.source_event_indices)


def _conditional(observation: PrefixDirectionalObservation) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    logits = observation.logits.to(dtype=torch.float64)
    log_probability = logits - torch.logsumexp(logits, dim=0)
    if observation.tangent_logits is None:
        return log_probability, None, logits
    tangent = observation.tangent_logits.to(dtype=torch.float64)
    probability = log_probability.exp()
    score = tangent - (probability[:, None] * tangent).sum(dim=0, keepdim=True)
    return log_probability, score, logits


def _complement(
    *,
    log_probability: torch.Tensor,
    score: torch.Tensor | None,
    logits: torch.Tensor,
    blocked: tuple[int, ...],
    reducer: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = torch.ones(log_probability.numel(), dtype=torch.bool)
    mask[list(blocked)] = False
    if not bool(mask.any()):
        raise EventSemanticBoundary("exclusive macro is empty")
    if reducer == "streaming":
        log_mass = torch.logsumexp(logits[mask], dim=0) - torch.logsumexp(logits, dim=0)
    elif reducer == "bruteforce":
        log_mass = torch.logsumexp(log_probability[mask], dim=0)
    else:
        raise EventSemanticBoundary("unknown reducer")
    if score is None:
        return log_mass, None
    weights = torch.softmax(logits[mask], dim=0)
    return log_mass, (weights[:, None] * score[mask]).sum(dim=0)


def evaluate_fine_events(
    layout: FineEventLayout,
    observations: Mapping[tuple[int, ...], PrefixDirectionalObservation],
    *,
    reducer: str = "streaming",
) -> FineEventEvaluation:
    if set(observations) != set(layout.internal_prefixes):
        raise EventSemanticBoundary("observation prefix inventory differs from fixed layout")
    tangent_presence = {value.tangent_logits is not None for value in observations.values()}
    if len(tangent_presence) != 1:
        raise EventSemanticBoundary("all prefixes must consistently include tangents")
    actuator_count = None
    if True in tangent_presence:
        first = next(iter(observations.values())).tangent_logits
        assert first is not None
        actuator_count = int(first.shape[1])

    prefix_log: dict[tuple[int, ...], torch.Tensor] = {(): torch.zeros((), dtype=torch.float64)}
    prefix_score: dict[tuple[int, ...], torch.Tensor] = {}
    if actuator_count is not None:
        prefix_score[()] = torch.zeros(actuator_count, dtype=torch.float64)
    conditional: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    for prefix in layout.internal_prefixes:
        if prefix not in prefix_log:
            raise EventSemanticBoundary("reachable prefix mass is missing")
        observation = observations[prefix]
        if observation.logits.numel() != layout.output_vocabulary_size:
            raise EventSemanticBoundary("output-head vocabulary drifted")
        if observation.tangent_logits is not None and observation.tangent_logits.shape[1] != actuator_count:
            raise EventSemanticBoundary("actuator width drifted across prefixes")
        logp, score, logits = _conditional(observation)
        conditional.append((logp, score, logits))
        for token in layout.children(prefix):
            child = prefix + (token,)
            prefix_log[child] = prefix_log[prefix] + logp[token]
            if score is not None:
                prefix_score[child] = prefix_score[prefix] + score[token]

    values: list[torch.Tensor] = []
    score_values: list[torch.Tensor] = []
    for index in range(layout.event_count):
        mode = EventRowMode(int(layout.event_modes[index]))
        prefix_index = int(layout.event_prefix_ordinals[index])
        if mode is EventRowMode.LEAF_CYLINDER:
            # Leaf rows use -1 because the completed path is not internal.
            prefix = layout.source_tokens if index in layout.source_event_indices else layout.target_tokens
            values.append(prefix_log[prefix])
            if actuator_count is not None:
                score_values.append(prefix_score[prefix])
            continue
        prefix = layout.internal_prefixes[prefix_index]
        logp, score, logits = conditional[prefix_index]
        if mode is EventRowMode.ATOMIC_DEPARTURE:
            token = int(layout.event_token_ids[index])
            values.append(prefix_log[prefix] + logp[token])
            if actuator_count is not None:
                assert score is not None
                score_values.append(prefix_score[prefix] + score[token])
        else:
            log_mass, aggregate_score = _complement(
                log_probability=logp,
                score=score,
                logits=logits,
                blocked=layout.macro_excluded_tokens[index],
                reducer=reducer,
            )
            values.append(prefix_log[prefix] + log_mass)
            if actuator_count is not None:
                assert aggregate_score is not None
                score_values.append(prefix_score[prefix] + aggregate_score)

    logs = torch.stack(values).detach().to(dtype=torch.float64).contiguous()
    log_residual = abs(float(torch.logsumexp(logs, dim=0).cpu().item()))
    eps = torch.finfo(torch.float64).eps
    scale = 1.0 + float(torch.max(torch.abs(logs)).cpu().item()) + math.log(layout.event_count)
    normalization_tolerance = 128.0 * eps * scale
    scores_out = None
    centering = None
    if actuator_count is not None:
        scores_out = torch.stack(score_values).detach().to(dtype=torch.float64).contiguous()
        expectation = (logs.exp()[:, None] * scores_out).sum(dim=0)
        centering = float(torch.linalg.vector_norm(expectation).cpu().item())
        score_scale = 1.0 + float(torch.linalg.matrix_norm(scores_out).cpu().item())
        if centering > 512.0 * eps * score_scale:
            raise EventSemanticBoundary("fine-event score centering failed")
    return FineEventEvaluation(
        layout=layout,
        log_probabilities=logs,
        scores=scores_out,
        normalization_log_residual=log_residual,
        normalization_tolerance=normalization_tolerance,
        score_centering_norm=centering,
        reducer=reducer,
    )


__all__ = [
    "EventRowMode",
    "FineEventEvaluation",
    "FineEventLayout",
    "PrefixDirectionalObservation",
    "evaluate_fine_events",
]
