"""Canonical-only functional H replay and disjoint theta0 P calibration."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .barriers import FunctionalHPVerdict, ReplayRiskReceipt, functional_replay_risk
from .contracts import ODEBFContractError, canonical_hash
from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
from .functional import CumulativeBF16FunctionalTrial, WaypointFactor, tensor_sha256


@dataclass(frozen=True, slots=True)
class VectorEvaluationReceipt:
    request_order_sha256: str
    value_sha256: str
    item_count: int
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int

    def __post_init__(self) -> None:
        if len(self.request_order_sha256) != 64 or len(self.value_sha256) != 64:
            raise ODEBFContractError("functional replay evaluation digest differs")
        if self.item_count < 0 or self.model_forward_count < 0 or self.processed_token_count < 0:
            raise ODEBFContractError("functional replay evaluation count differs")
        if self.generation_call_count != 0:
            raise ODEBFContractError("functional replay called generation")


def _target_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("str")
    if not isinstance(value, str) or not value:
        raise ODEBFContractError("functional replay target is empty")
    return value


def _validate_canonical_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool,
) -> tuple[Mapping[str, Any], ...]:
    batch = tuple(requests)
    if not batch and not allow_empty:
        raise ODEBFContractError("functional replay request batch is empty")
    identities = [str(item.get("request_sha256", "")) for item in batch]
    if any(len(value) != 64 for value in identities) or len(set(identities)) != len(identities):
        raise ODEBFContractError("functional replay request identities differ")
    forbidden = ("paraphrase", "neighborhood", "locality", "generation", "heldout")
    if any(any(fragment in str(key).casefold() for fragment in forbidden) for item in batch for key in item):
        raise ODEBFContractError("held-out field entered functional replay")
    return batch


def _request_order(requests: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash([str(item["request_sha256"]) for item in requests])


def evaluate_target_new_nlls(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    model_alias: str,
) -> tuple[torch.Tensor, VectorEvaluationReceipt]:
    batch = _validate_canonical_requests(requests, allow_empty=True)
    if not batch:
        values = torch.empty((0,), dtype=torch.float64)
        return values, VectorEvaluationReceipt(
            canonical_hash([]), tensor_sha256(values), 0, 0, 0, 0
        )
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    prefixes = [str(item["prompt"]).format(str(item["subject"])) for item in batch]
    targets = [_target_text(item["target_new"]) for item in batch]
    prefix_lengths = [len(tokens) for tokens in tokenizer(prefixes)["input_ids"]]
    encoded = tokenizer(
        [f"{prefix} {target}" for prefix, target in zip(prefixes, targets)],
        padding=True,
        return_tensors="pt",
    ).to(device)
    target_tokens = [tokenizer(f" {target}")["input_ids"] for target in targets]
    if llama:
        prefix_lengths = [value - 1 for value in prefix_lengths]
        target_tokens = [value[1:] for value in target_tokens]
    if any(not value for value in target_tokens):
        raise ODEBFContractError("functional H target suffix is empty")
    with _EvaluationStateGuard(model), torch.no_grad():
        logits = model(**encoded).logits
        if llama:
            logits = logits[:, 1:, :]
        values: list[torch.Tensor] = []
        for row, tokens in enumerate(target_tokens):
            per_token = []
            for offset, token in enumerate(tokens):
                position = prefix_lengths[row] + offset - 1
                if position < 0 or position >= logits.shape[1]:
                    raise ODEBFContractError("functional H target span is out of bounds")
                per_token.append(
                    -torch.log_softmax(logits[row, position, :].float(), dim=0)[int(token)]
                )
            values.append(torch.stack(per_token).mean())
        result = torch.stack(values).detach().to(device="cpu", dtype=torch.float64)
    attention = encoded.get("attention_mask")
    processed = int(attention.sum()) if attention is not None else int(encoded["input_ids"].numel())
    del logits, encoded
    return result, VectorEvaluationReceipt(
        _request_order(batch),
        tensor_sha256(result),
        len(batch),
        1,
        processed,
        0,
    )


def evaluate_next_token_log_probs(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, VectorEvaluationReceipt]:
    batch = _validate_canonical_requests(requests, allow_empty=False)
    device = _model_device(model)
    prompts = [str(item["prompt"]).format(str(item["subject"])) for item in batch]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with _EvaluationStateGuard(model), torch.no_grad():
        logits = model(**encoded).logits
        attention = encoded.get("attention_mask")
        if attention is None:
            positions = torch.full(
                (logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device
            )
        else:
            positions = attention.sum(dim=1) - 1
        selected = logits[
            torch.arange(logits.shape[0], device=logits.device), positions, :
        ]
        log_probs = torch.log_softmax(selected.float(), dim=-1).detach().to(device="cpu")
    processed = int(attention.sum()) if attention is not None else int(encoded["input_ids"].numel())
    del logits, selected, encoded
    return log_probs, VectorEvaluationReceipt(
        _request_order(batch),
        tensor_sha256(log_probs),
        len(batch),
        1,
        processed,
        0,
    )


@dataclass(slots=True)
class Theta0TeacherCache:
    request_order: tuple[str, ...]
    log_probs_by_request: dict[str, torch.Tensor]
    population_sha256: str
    receipt_sha256: str
    model_forward_count: int
    processed_token_count: int

    def select(self, identities: Sequence[str]) -> torch.Tensor:
        order = tuple(identities)
        if len(set(order)) != len(order) or any(item not in self.log_probs_by_request for item in order):
            raise ODEBFContractError("theta0 teacher selection differs")
        return torch.stack([self.log_probs_by_request[item] for item in order])


@dataclass(frozen=True, slots=True)
class FixedEntryPretrainedBaselineReceipt:
    baseline_kind: str
    outer_entry_snapshot_sha256: str
    anchor_population_sha256: str
    sample_order_sha256: str
    entry_kl_identity_sha256: str
    cache_receipt_sha256: str
    item_count: int

    def __post_init__(self) -> None:
        if self.baseline_kind != "outer_entry":
            raise ODEBFContractError("functional P baseline kind differs")
        for value in (
            self.outer_entry_snapshot_sha256,
            self.anchor_population_sha256,
            self.sample_order_sha256,
            self.entry_kl_identity_sha256,
            self.cache_receipt_sha256,
        ):
            if len(value) != 64:
                raise ODEBFContractError("functional P baseline digest differs")
        if self.item_count != 10:
            raise ODEBFContractError("functional P baseline sample count differs")


@dataclass(slots=True)
class OuterEntryPretrainedCache:
    request_order: tuple[str, ...]
    entry_kl_by_request: dict[str, torch.Tensor]
    entry_kl_sha256_by_request: dict[str, str]
    outer_entry_snapshot_sha256: str
    population_sha256: str
    receipt_sha256: str
    model_forward_count: int
    processed_token_count: int

    def __post_init__(self) -> None:
        if (
            len(self.request_order) != 160
            or len(set(self.request_order)) != 160
            or set(self.entry_kl_by_request) != set(self.request_order)
            or set(self.entry_kl_sha256_by_request) != set(self.request_order)
        ):
            raise ODEBFContractError("outer-entry P cache population differs")
        for value in (
            self.outer_entry_snapshot_sha256,
            self.population_sha256,
            self.receipt_sha256,
        ):
            if len(value) != 64:
                raise ODEBFContractError("outer-entry P cache digest differs")
        if self.population_sha256 != canonical_hash(list(self.request_order)):
            raise ODEBFContractError("outer-entry P cache order differs")
        if self.model_forward_count != 16 or self.processed_token_count <= 0:
            raise ODEBFContractError("outer-entry P cache accounting differs")

    def select(
        self,
        identities: Sequence[str],
    ) -> tuple[
        torch.Tensor,
        VectorEvaluationReceipt,
        FixedEntryPretrainedBaselineReceipt,
    ]:
        order = tuple(identities)
        if (
            len(order) != 10
            or len(set(order)) != 10
            or any(item not in self.entry_kl_by_request for item in order)
        ):
            raise ODEBFContractError("outer-entry P cache selection differs")
        for item in order:
            value = self.entry_kl_by_request[item]
            if (
                value.shape != ()
                or value.dtype is not torch.float64
                or not torch.isfinite(value)
                or tensor_sha256(value) != self.entry_kl_sha256_by_request[item]
            ):
                raise ODEBFContractError("outer-entry P cache value mutated")
        values = torch.stack(
            [self.entry_kl_by_request[item].clone() for item in order]
        ).contiguous()
        order_sha256 = _request_order(
            tuple({"request_sha256": item} for item in order)
        )
        value_sha256 = tensor_sha256(values)
        evaluation = VectorEvaluationReceipt(
            order_sha256,
            value_sha256,
            len(order),
            0,
            0,
            0,
        )
        baseline = FixedEntryPretrainedBaselineReceipt(
            "outer_entry",
            self.outer_entry_snapshot_sha256,
            self.population_sha256,
            order_sha256,
            value_sha256,
            self.receipt_sha256,
            len(order),
        )
        return values, evaluation, baseline


def build_theta0_teacher_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    population_requests: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = 10,
) -> Theta0TeacherCache:
    population = _validate_canonical_requests(population_requests, allow_empty=False)
    if len(population) != 160 or chunk_size != 10:
        raise ODEBFContractError("theta0 P population/chunk contract differs")
    result: dict[str, torch.Tensor] = {}
    receipts: list[dict[str, Any]] = []
    forwards = 0
    tokens = 0
    for start in range(0, len(population), chunk_size):
        chunk = population[start : start + chunk_size]
        values, receipt = evaluate_next_token_log_probs(model, tokenizer, chunk)
        for index, request in enumerate(chunk):
            result[str(request["request_sha256"])] = values[index].clone()
        forwards += receipt.model_forward_count
        tokens += receipt.processed_token_count
        receipts.append(
            {
                "order": receipt.request_order_sha256,
                "value": receipt.value_sha256,
                "count": receipt.item_count,
            }
        )
    order = tuple(str(item["request_sha256"]) for item in population)
    population_sha = canonical_hash(list(order))
    return Theta0TeacherCache(
        order,
        result,
        population_sha,
        canonical_hash(receipts),
        forwards,
        tokens,
    )


def samplewise_teacher_kl(
    theta0_log_probs: torch.Tensor,
    observed_log_probs: torch.Tensor,
) -> torch.Tensor:
    if theta0_log_probs.shape != observed_log_probs.shape or theta0_log_probs.ndim != 2:
        raise ODEBFContractError("functional P teacher/observed geometry differs")
    if not torch.isfinite(theta0_log_probs).all() or not torch.isfinite(observed_log_probs).all():
        raise ODEBFContractError("functional P log probabilities are non-finite")
    teacher = theta0_log_probs.to(dtype=torch.float64)
    observed = observed_log_probs.to(dtype=torch.float64)
    values = torch.sum(torch.exp(teacher) * (teacher - observed), dim=1)
    # Numerical roundoff may make an exact self-KL a few ulps negative.
    return torch.clamp(values, min=0.0)


@dataclass(frozen=True, slots=True)
class FunctionalReplayPair:
    historical: ReplayRiskReceipt
    pretrained: ReplayRiskReceipt
    historical_entry_receipt: VectorEvaluationReceipt
    historical_trial_receipt: VectorEvaluationReceipt
    pretrained_entry_receipt: VectorEvaluationReceipt
    pretrained_trial_receipt: VectorEvaluationReceipt
    functional_identity_sha256: str


def evaluate_functional_replay_pair(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_alias: str,
    history_requests: Sequence[Mapping[str, Any]],
    history_entry_nll: torch.Tensor,
    pretrained_requests: Sequence[Mapping[str, Any]],
    pretrained_entry_kl: torch.Tensor,
    theta0_cache: Theta0TeacherCache,
    trial_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    historical_budget: float,
    pretrained_budget: float,
    smoothmax_temperature: float,
) -> FunctionalReplayPair:
    history = _validate_canonical_requests(history_requests, allow_empty=True)
    pretrained = _validate_canonical_requests(pretrained_requests, allow_empty=False)
    if len(pretrained) != 10:
        raise ODEBFContractError("functional P replay must use its sealed B10")
    if history_entry_nll.shape != (len(history),) or pretrained_entry_kl.shape != (len(pretrained),):
        raise ODEBFContractError("functional replay entry baseline shape differs")
    empty_history_receipt = VectorEvaluationReceipt(
        canonical_hash([]), tensor_sha256(torch.empty((0,), dtype=torch.float64)), 0, 0, 0, 0
    )
    history_entry_receipt = VectorEvaluationReceipt(
        _request_order(history),
        tensor_sha256(history_entry_nll),
        len(history),
        0,
        0,
        0,
    ) if history else empty_history_receipt
    pretrained_entry_receipt = VectorEvaluationReceipt(
        _request_order(pretrained),
        tensor_sha256(pretrained_entry_kl),
        len(pretrained),
        0,
        0,
        0,
    )
    trial_context = (
        CumulativeBF16FunctionalTrial(model, trial_factors_by_weight, row_block=64)
        if trial_factors_by_weight
        else nullcontext()
    )
    with trial_context:
        history_trial, history_trial_receipt = evaluate_target_new_nlls(
            model, tokenizer, history, model_alias=model_alias
        )
        observed_log_probs, pretrained_trial_receipt = evaluate_next_token_log_probs(
            model, tokenizer, pretrained
        )
    identities = [str(item["request_sha256"]) for item in pretrained]
    theta0 = theta0_cache.select(identities)
    trial_kl = samplewise_teacher_kl(theta0, observed_log_probs)
    pretrained_trial_receipt = VectorEvaluationReceipt(
        pretrained_trial_receipt.request_order_sha256,
        tensor_sha256(trial_kl),
        pretrained_trial_receipt.item_count,
        pretrained_trial_receipt.model_forward_count,
        pretrained_trial_receipt.processed_token_count,
        0,
    )
    historical = functional_replay_risk(
        barrier="historical-current-teacher",
        entry_values=history_entry_nll.tolist(),
        trial_values=history_trial.tolist(),
        sample_sha256=[str(item["request_sha256"]) for item in history],
        budget=historical_budget,
        smooth_max_temperature=smoothmax_temperature,
    )
    pretrained_risk = functional_replay_risk(
        barrier="pretrained-theta0-teacher",
        entry_values=pretrained_entry_kl.tolist(),
        trial_values=trial_kl.tolist(),
        sample_sha256=identities,
        budget=pretrained_budget,
        smooth_max_temperature=smoothmax_temperature,
    )
    identity = canonical_hash(
        {
            "historical": historical.sample_order_sha256,
            "pretrained": pretrained_risk.sample_order_sha256,
            "historical_entry": history_entry_receipt.value_sha256,
            "historical_trial": history_trial_receipt.value_sha256,
            "pretrained_entry": pretrained_entry_receipt.value_sha256,
            "pretrained_trial": pretrained_trial_receipt.value_sha256,
        }
    )
    return FunctionalReplayPair(
        historical,
        pretrained_risk,
        history_entry_receipt,
        history_trial_receipt,
        pretrained_entry_receipt,
        pretrained_trial_receipt,
        identity,
    )


def capture_pretrained_entry_kl(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    theta0_cache: Theta0TeacherCache,
) -> tuple[torch.Tensor, VectorEvaluationReceipt]:
    batch = _validate_canonical_requests(requests, allow_empty=False)
    observed, receipt = evaluate_next_token_log_probs(model, tokenizer, batch)
    identities = [str(item["request_sha256"]) for item in batch]
    values = samplewise_teacher_kl(theta0_cache.select(identities), observed)
    return values, VectorEvaluationReceipt(
        receipt.request_order_sha256,
        tensor_sha256(values),
        receipt.item_count,
        receipt.model_forward_count,
        receipt.processed_token_count,
        0,
    )


def build_outer_entry_pretrained_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    population_requests: Sequence[Mapping[str, Any]],
    theta0_cache: Theta0TeacherCache,
    *,
    outer_entry_snapshot_sha256: str,
    chunk_size: int = 10,
) -> OuterEntryPretrainedCache:
    """Capture scalar theta0 KL once at the immutable outer-batch entry."""

    population = _validate_canonical_requests(
        population_requests,
        allow_empty=False,
    )
    order = tuple(str(item["request_sha256"]) for item in population)
    if (
        len(population) != 160
        or chunk_size != 10
        or order != theta0_cache.request_order
        or len(outer_entry_snapshot_sha256) != 64
    ):
        raise ODEBFContractError("outer-entry P cache build contract differs")
    values_by_request: dict[str, torch.Tensor] = {}
    hashes_by_request: dict[str, str] = {}
    chunk_receipts: list[dict[str, Any]] = []
    forwards = 0
    tokens = 0
    for start in range(0, len(population), chunk_size):
        chunk = population[start : start + chunk_size]
        values, receipt = capture_pretrained_entry_kl(
            model,
            tokenizer,
            chunk,
            theta0_cache,
        )
        if values.shape != (chunk_size,) or values.dtype is not torch.float64:
            raise ODEBFContractError("outer-entry P cache chunk geometry differs")
        for index, request in enumerate(chunk):
            identity = str(request["request_sha256"])
            scalar = values[index].detach().to(device="cpu", dtype=torch.float64).clone()
            values_by_request[identity] = scalar
            hashes_by_request[identity] = tensor_sha256(scalar)
        forwards += receipt.model_forward_count
        tokens += receipt.processed_token_count
        chunk_receipts.append(
            {
                "sample_order_sha256": receipt.request_order_sha256,
                "entry_kl_identity_sha256": receipt.value_sha256,
                "item_count": receipt.item_count,
            }
        )
    population_sha256 = canonical_hash(list(order))
    cache_receipt_sha256 = canonical_hash(
        {
            "schema": "ode-edit-s04-ode-bf-fixed-entry-p-cache/v1",
            "baseline_kind": "outer_entry",
            "outer_entry_snapshot_sha256": outer_entry_snapshot_sha256,
            "anchor_population_sha256": population_sha256,
            "chunks": chunk_receipts,
        }
    )
    return OuterEntryPretrainedCache(
        order,
        values_by_request,
        hashes_by_request,
        outer_entry_snapshot_sha256,
        population_sha256,
        cache_receipt_sha256,
        forwards,
        tokens,
    )
