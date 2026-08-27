"""Vocabulary-streaming first-departure reductions for BGODE-R1 S1.

Only one ``V x 5`` score buffer exists at a time.  Distinguished leaves are
stored as scalars; departure events are reduced node-by-node without creating
Python ``EventSpec`` objects for the full vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .errors import BGODEScientificBoundary, EventPartitionBoundary
from .fisher_pullback import FisherPullback


@dataclass(frozen=True, slots=True)
class StreamingTrieLayout:
    source_path: tuple[int, ...]
    target_path: tuple[int, ...]
    internal_prefixes: tuple[tuple[int, ...], ...]
    children_by_prefix: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    vocabulary_size: int
    boundary_token: int

    @classmethod
    def build(
        cls,
        *,
        source_tokens: Sequence[int],
        target_tokens: Sequence[int],
        boundary_token: int,
        vocabulary_size: int,
        batch_size: int = 1,
    ) -> "StreamingTrieLayout":
        if batch_size != 1:
            raise EventPartitionBoundary("BGODE-R1 S1 requires B=1")
        source = tuple(int(v) for v in source_tokens)
        target = tuple(int(v) for v in target_tokens)
        if not source or not target or source == target:
            raise EventPartitionBoundary("source/target token paths must be nonempty and distinct")
        if boundary_token in source or boundary_token in target:
            raise EventPartitionBoundary("boundary token occurs inside a distinguished sequence")
        paths = (source + (int(boundary_token),), target + (int(boundary_token),))
        if any(token < 0 or token >= vocabulary_size for path in paths for token in path):
            raise EventPartitionBoundary("trie token lies outside vocabulary")
        children: dict[tuple[int, ...], set[int]] = {}
        for path in paths:
            for index, token in enumerate(path):
                children.setdefault(path[:index], set()).add(token)
        ordered = tuple(
            (prefix, tuple(sorted(tokens)))
            for prefix, tokens in sorted(children.items(), key=lambda item: (len(item[0]), item[0]))
        )
        return cls(
            source_path=paths[0],
            target_path=paths[1],
            internal_prefixes=tuple(prefix for prefix, _ in ordered),
            children_by_prefix=ordered,
            vocabulary_size=int(vocabulary_size),
            boundary_token=int(boundary_token),
        )

    def children(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        for candidate, children in self.children_by_prefix:
            if candidate == prefix:
                return children
        raise EventPartitionBoundary("unknown trie prefix")


@dataclass(frozen=True, slots=True)
class PrefixDirectionalLogits:
    logits: torch.Tensor
    tangent_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            self.logits.dtype != torch.float32
            or self.logits.ndim != 1
            or self.logits.requires_grad
            or not bool(torch.isfinite(self.logits).all())
        ):
            raise EventPartitionBoundary("prefix logits must be detached finite FP32 [V]")
        if self.tangent_logits is not None and (
            self.tangent_logits.dtype != torch.float32
            or self.tangent_logits.ndim != 2
            or self.tangent_logits.shape[0] != self.logits.numel()
            or self.tangent_logits.shape[1] <= 0
            or self.tangent_logits.requires_grad
            or not bool(torch.isfinite(self.tangent_logits).all())
        ):
            raise EventPartitionBoundary("prefix tangent logits must be detached finite FP32 [V,L]")


@dataclass(frozen=True, slots=True)
class StreamingEventState:
    layout: StreamingTrieLayout
    target_log_probability: float
    source_log_probability: float
    departure_log_probabilities: tuple[torch.Tensor, ...]
    departure_token_ids: tuple[torch.Tensor, ...]
    normalization_log_residual: float

    @property
    def log_odds(self) -> float:
        return self.target_log_probability - self.source_log_probability

    @property
    def pair_mass(self) -> float:
        return math.exp(self.target_log_probability) + math.exp(self.source_log_probability)


@dataclass(frozen=True, slots=True)
class StreamingEventMoments:
    current: StreamingEventState
    fisher: FisherPullback
    progress_sensitivity: torch.Tensor
    pair_mass_sensitivity: torch.Tensor
    moving_gradient: torch.Tensor
    anchor_gradient: torch.Tensor
    score_expectation: torch.Tensor
    moving_reference_kl: float
    anchored_kl: float
    reference_target_log_probability: float
    reference_source_log_probability: float
    event_score_buffer_peak_bytes: int


def _conditional(observation: PrefixDirectionalLogits) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits = observation.logits
    logp = torch.log_softmax(logits, dim=0)
    if observation.tangent_logits is None:
        return logp, None
    tangent = observation.tangent_logits
    score = tangent - (logp.exp()[:, None] * tangent).sum(dim=0, keepdim=True)
    return logp, score


def evaluate_streaming_state(
    layout: StreamingTrieLayout,
    observations: Mapping[tuple[int, ...], PrefixDirectionalLogits],
) -> StreamingEventState:
    prefix_log_mass: dict[tuple[int, ...], torch.Tensor] = {(): torch.zeros((), dtype=torch.float32)}
    departure_logs: list[torch.Tensor] = []
    departure_ids: list[torch.Tensor] = []
    for prefix in layout.internal_prefixes:
        if prefix not in prefix_log_mass or prefix not in observations:
            raise EventPartitionBoundary("missing reachable streaming trie prefix")
        logp, _ = _conditional(observations[prefix])
        if logp.numel() != layout.vocabulary_size:
            raise EventPartitionBoundary("streaming prefix vocabulary differs")
        children = layout.children(prefix)
        mask = torch.ones(layout.vocabulary_size, dtype=torch.bool, device=logp.device)
        mask[list(children)] = False
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        departure_ids.append(ids.detach().cpu().contiguous())
        departure_logs.append((prefix_log_mass[prefix] + logp[ids]).detach().cpu().contiguous())
        for token in children:
            prefix_log_mass[prefix + (token,)] = prefix_log_mass[prefix] + logp[token]
    try:
        target_log = prefix_log_mass[layout.target_path]
        source_log = prefix_log_mass[layout.source_path]
    except KeyError as exc:
        raise EventPartitionBoundary("distinguished leaf was not reached") from exc
    chunks = [target_log.reshape(1), source_log.reshape(1), *[value.to(target_log.device) for value in departure_logs]]
    log_total = torch.logsumexp(torch.cat(chunks), dim=0)
    residual = abs(float(log_total.detach().cpu().item()))
    event_count = sum(value.numel() for value in chunks)
    tolerance = 64.0 * torch.finfo(torch.float32).eps * event_count
    if not math.isfinite(residual) or residual > tolerance:
        raise EventPartitionBoundary("streaming first-departure partition does not normalize")
    return StreamingEventState(
        layout=layout,
        target_log_probability=float(target_log.detach().cpu().item()),
        source_log_probability=float(source_log.detach().cpu().item()),
        departure_log_probabilities=tuple(departure_logs),
        departure_token_ids=tuple(departure_ids),
        normalization_log_residual=residual,
    )


def _reference_leaf_logs(initial: StreamingEventState, time: float) -> tuple[float, float]:
    if not math.isfinite(time) or time < 0.0:
        raise BGODEScientificBoundary("reference time must be finite and nonnegative")
    ly = torch.tensor(initial.target_log_probability, dtype=torch.float32)
    ls = torch.tensor(initial.source_log_probability, dtype=torch.float32)
    log_c = torch.logaddexp(ly, ls)
    r = ly - ls + float(time)
    return (
        float((log_c + torch.nn.functional.logsigmoid(r)).item()),
        float((log_c + torch.nn.functional.logsigmoid(-r)).item()),
    )


def aggregate_streaming_moments(
    *,
    initial: StreamingEventState,
    observations: Mapping[tuple[int, ...], PrefixDirectionalLogits],
    reference_time: float,
) -> StreamingEventMoments:
    layout = initial.layout
    prefix_log_mass: dict[tuple[int, ...], torch.Tensor] = {(): torch.zeros((), dtype=torch.float32)}
    first = next(iter(observations.values()))
    if first.tangent_logits is None:
        raise BGODEScientificBoundary("streaming event moments require directional logits")
    actuator_count = int(first.tangent_logits.shape[1])
    prefix_score: dict[tuple[int, ...], torch.Tensor] = {
        (): torch.zeros(actuator_count, dtype=torch.float32, device=first.logits.device)
    }
    fisher = torch.zeros((actuator_count, actuator_count), dtype=torch.float32, device=first.logits.device)
    expectation = torch.zeros(actuator_count, dtype=torch.float32, device=first.logits.device)
    moving = torch.zeros_like(expectation)
    anchor = torch.zeros_like(expectation)
    kl_moving = torch.zeros((), dtype=torch.float32, device=first.logits.device)
    kl_anchor = torch.zeros_like(kl_moving)
    peak = 0
    ref_y, ref_s = _reference_leaf_logs(initial, reference_time)
    departure_current: list[torch.Tensor] = []
    departure_ids: list[torch.Tensor] = []
    for node_index, prefix in enumerate(layout.internal_prefixes):
        observation = observations.get(prefix)
        if observation is None or observation.tangent_logits is None:
            raise EventPartitionBoundary("directional observation is missing")
        logp, conditional_score = _conditional(observation)
        assert conditional_score is not None
        if conditional_score.shape[1] != actuator_count:
            raise EventPartitionBoundary("actuator count changed across trie prefixes")
        children = layout.children(prefix)
        mask = torch.ones(layout.vocabulary_size, dtype=torch.bool, device=logp.device)
        mask[list(children)] = False
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        score = prefix_score[prefix][None, :] + conditional_score[ids]
        current_log = prefix_log_mass[prefix] + logp[ids]
        initial_log = initial.departure_log_probabilities[node_index].to(logp.device)
        if not torch.equal(initial.departure_token_ids[node_index], ids.detach().cpu()):
            raise EventPartitionBoundary("departure token order changed")
        current_prob = current_log.exp()
        initial_prob = initial_log.exp()
        fisher += score.transpose(0, 1) @ (current_prob[:, None] * score)
        expectation += (current_prob[:, None] * score).sum(dim=0)
        moving -= (initial_prob[:, None] * score).sum(dim=0)
        anchor -= (initial_prob[:, None] * score).sum(dim=0)
        kl_moving += torch.sum(initial_prob * (initial_log - current_log))
        kl_anchor += torch.sum(initial_prob * (initial_log - current_log))
        peak = max(peak, int(score.numel() * score.element_size()))
        departure_current.append(current_log.detach().cpu().contiguous())
        departure_ids.append(ids.detach().cpu().contiguous())
        for token in children:
            prefix_log_mass[prefix + (token,)] = prefix_log_mass[prefix] + logp[token]
            prefix_score[prefix + (token,)] = prefix_score[prefix] + conditional_score[token]
    target_log = prefix_log_mass[layout.target_path]
    source_log = prefix_log_mass[layout.source_path]
    target_score = prefix_score[layout.target_path]
    source_score = prefix_score[layout.source_path]
    current_leaf_logs = (target_log, source_log)
    initial_leaf_logs = (
        torch.tensor(initial.target_log_probability, dtype=torch.float32, device=target_log.device),
        torch.tensor(initial.source_log_probability, dtype=torch.float32, device=target_log.device),
    )
    reference_leaf_logs = (
        torch.tensor(ref_y, dtype=torch.float32, device=target_log.device),
        torch.tensor(ref_s, dtype=torch.float32, device=target_log.device),
    )
    for current_log, initial_log, reference_log, score in zip(
        current_leaf_logs, initial_leaf_logs, reference_leaf_logs, (target_score, source_score), strict=True
    ):
        p_current = current_log.exp()
        p_initial = initial_log.exp()
        p_reference = reference_log.exp()
        fisher += p_current * torch.outer(score, score)
        expectation += p_current * score
        moving -= p_reference * score
        anchor -= p_initial * score
        kl_moving += p_reference * (reference_log - current_log)
        kl_anchor += p_initial * (initial_log - current_log)
    matrix = 0.5 * (fisher + fisher.transpose(0, 1))
    symmetry = float(torch.linalg.matrix_norm(matrix - matrix.transpose(0, 1)).detach().cpu().item())
    minimum = float(torch.linalg.eigvalsh(matrix).amin().detach().cpu().item())
    pullback = FisherPullback(matrix.detach().cpu().contiguous(), symmetry, minimum)
    current = StreamingEventState(
        layout=layout,
        target_log_probability=float(target_log.detach().cpu().item()),
        source_log_probability=float(source_log.detach().cpu().item()),
        departure_log_probabilities=tuple(departure_current),
        departure_token_ids=tuple(departure_ids),
        normalization_log_residual=evaluate_streaming_state(layout, observations).normalization_log_residual,
    )
    return StreamingEventMoments(
        current=current,
        fisher=pullback,
        progress_sensitivity=(target_score - source_score).detach().cpu().contiguous(),
        pair_mass_sensitivity=(target_log.exp() * target_score + source_log.exp() * source_score).detach().cpu().contiguous(),
        moving_gradient=moving.detach().cpu().contiguous(),
        anchor_gradient=anchor.detach().cpu().contiguous(),
        score_expectation=expectation.detach().cpu().contiguous(),
        moving_reference_kl=float(kl_moving.detach().cpu().item()),
        anchored_kl=float(kl_anchor.detach().cpu().item()),
        reference_target_log_probability=ref_y,
        reference_source_log_probability=ref_s,
        event_score_buffer_peak_bytes=peak,
    )


__all__ = [
    "PrefixDirectionalLogits",
    "StreamingEventMoments",
    "StreamingEventState",
    "StreamingTrieLayout",
    "aggregate_streaming_moments",
    "evaluate_streaming_state",
]
