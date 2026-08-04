"""Teacher-forced pinned CounterFact/zsRE canonical efficacy adapters."""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .benchmark import (
    BatchSuccessReceipt,
    CounterFactRequestScore,
    counterfact_batch_receipt,
    counterfact_score_from_token_nlls,
    zsre_batch_receipt,
)
from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError
from .firewall import evaluator_request_hash


def _hash_tensor_sequence(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        logical = tensor.detach().to(device="cpu").contiguous()
        header = f"{tuple(logical.shape)}|{logical.dtype}|".encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = logical.view(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class _EvaluationStateGuard(AbstractContextManager["_EvaluationStateGuard"]):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool, torch.Tensor | None]] = []
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}

    def __enter__(self) -> "_EvaluationStateGuard":
        self._cpu_rng = torch.get_rng_state().clone()
        for name, parameter in self.model.named_parameters():
            self._parameters.append(
                (
                    name,
                    parameter,
                    parameter.data_ptr(),
                    parameter._version,
                    parameter.requires_grad,
                    None if parameter.grad is None else parameter.grad.detach().clone(),
                )
            )
        devices = {parameter.device for _, parameter, *_ in self._parameters if parameter.device.type == "cuda"}
        self._cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in devices}
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        violations: list[str] = []
        for name, parameter, pointer, version, requires_grad, grad in self._parameters:
            if parameter.data_ptr() != pointer:
                violations.append(f"{name}:pointer")
            if parameter._version != version:
                violations.append(f"{name}:version")
            if parameter.requires_grad != requires_grad:
                violations.append(f"{name}:requires_grad")
            if grad is None and parameter.grad is not None:
                violations.append(f"{name}:grad")
            elif grad is not None and (parameter.grad is None or not torch.equal(parameter.grad, grad)):
                violations.append(f"{name}:grad")
        self._parameters.clear()
        if self._cpu_rng is not None:
            torch.set_rng_state(self._cpu_rng)
            self._cpu_rng = None
        for device, state in self._cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self._cuda_rng.clear()
        if violations:
            error = ODEBFContractError("canonical evaluator mutated model state: " + ",".join(violations))
            if exc is not None:
                raise error from exc
            raise error
        return False


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ODEBFContractError("canonical evaluator model has no parameters") from exc


def _is_llama(model: torch.nn.Module, model_alias: str) -> bool:
    if model_alias not in MODEL_ALIASES:
        raise ODEBFContractError("canonical evaluator model alias is not locked")
    path = str(getattr(getattr(model, "config", object()), "_name_or_path", "")).lower()
    observed = "llama" in path
    if observed != (model_alias == "llama3-8b-inst"):
        raise ODEBFContractError("model config and evaluator alias differ")
    return observed


def _target_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("str")
    if not isinstance(value, str) or not value:
        raise ODEBFContractError("benchmark target text is empty")
    return value


def _validate_requests(requests: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    batch = tuple(requests)
    if len(batch) != BATCH_SIZE:
        raise ODEBFContractError("canonical evaluator requires ten requests")
    identities: list[str] = []
    cases: list[int] = []
    for request in batch:
        observed = evaluator_request_hash(request)
        if observed != request["request_sha256"]:
            raise ODEBFContractError("canonical evaluator request hash differs")
        identities.append(observed)
        cases.append(int(request["case_id"]))
    if len(set(identities)) != BATCH_SIZE or len(set(cases)) != BATCH_SIZE:
        raise ODEBFContractError("canonical evaluator batch is not distinct")
    return batch


@dataclass(frozen=True, slots=True)
class ModelEvaluationReceipt:
    batch_success: BatchSuccessReceipt
    target_full_vocabulary_logits_sha256: str
    request_order_sha256: str
    target_span_lengths: tuple[int, ...]
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int
    counterfact_scores: tuple[CounterFactRequestScore, ...] | None

    def __post_init__(self) -> None:
        if len(self.target_span_lengths) != BATCH_SIZE or any(length <= 0 for length in self.target_span_lengths):
            raise ODEBFContractError("canonical evaluator target spans are invalid")
        if self.model_forward_count != BATCH_SIZE:
            raise ODEBFContractError("canonical evaluator forward count differs")
        if self.processed_token_count <= 0 or self.generation_call_count != 0:
            raise ODEBFContractError("canonical evaluator accounting differs")
        if len(self.target_full_vocabulary_logits_sha256) != 64 or len(self.request_order_sha256) != 64:
            raise ODEBFContractError("canonical evaluator receipt digest is invalid")


def evaluate_counterfact_rewrite_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    model_alias: str,
) -> ModelEvaluationReceipt:
    batch = _validate_requests(requests)
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    scores: list[CounterFactRequestScore] = []
    logit_slices: list[torch.Tensor] = []
    span_lengths: list[int] = []
    processed_tokens = 0
    with _EvaluationStateGuard(model), torch.no_grad():
        for request in batch:
            prefix = str(request["prompt"]).format(str(request["subject"]))
            target_new = _target_text(request["target_new"])
            target_true = _target_text(request["target_true"])
            raw_prefix_length = len(tokenizer([prefix])["input_ids"][0])
            encoded = tokenizer(
                [f"{prefix} {target_new}", f"{prefix} {target_true}"],
                padding=True,
                return_tensors="pt",
            ).to(device)
            new_tokens = tokenizer(f" {target_new}")["input_ids"]
            true_tokens = tokenizer(f" {target_true}")["input_ids"]
            prefix_length = raw_prefix_length
            if llama:
                new_tokens = new_tokens[1:]
                true_tokens = true_tokens[1:]
                prefix_length -= 1
            if not new_tokens or not true_tokens:
                raise ODEBFContractError("CounterFact target suffix is empty")
            logits = model(**encoded).logits
            if llama:
                logits = logits[:, 1:, :]
            nll_by_choice: list[list[float]] = [[], []]
            for row, target_tokens in enumerate((new_tokens, true_tokens)):
                for offset, token in enumerate(target_tokens):
                    position = prefix_length + offset - 1
                    if position < 0 or position >= logits.shape[1]:
                        raise ODEBFContractError("CounterFact target span is out of bounds")
                    target_logits = logits[row, position, :]
                    logit_slices.append(target_logits.detach().cpu())
                    nll_by_choice[row].append(
                        -torch.log_softmax(target_logits.float(), dim=0)[int(token)].item()
                    )
            scores.append(counterfact_score_from_token_nlls(nll_by_choice[0], nll_by_choice[1]))
            span_lengths.append(len(new_tokens) + len(true_tokens))
            attention = encoded.get("attention_mask")
            processed_tokens += int(attention.sum()) if attention is not None else int(encoded["input_ids"].numel())
            del logits, encoded
    request_order = hashlib.sha256("".join(str(request["request_sha256"]) for request in batch).encode("ascii")).hexdigest()
    return ModelEvaluationReceipt(
        counterfact_batch_receipt(scores),
        _hash_tensor_sequence(logit_slices),
        request_order,
        tuple(span_lengths),
        BATCH_SIZE,
        processed_tokens,
        0,
        tuple(scores),
    )


def evaluate_zsre_rewrite_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    model_alias: str,
) -> ModelEvaluationReceipt:
    batch = _validate_requests(requests)
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    position_bits: list[list[int]] = []
    logit_slices: list[torch.Tensor] = []
    span_lengths: list[int] = []
    processed_tokens = 0
    with _EvaluationStateGuard(model), torch.no_grad():
        for request in batch:
            prefix = str(request["prompt"]).format(str(request["subject"]))
            target = _target_text(request["target_new"])
            target_tokens = tokenizer(" " + target)["input_ids"]
            if llama:
                target_tokens = target_tokens[1:]
            if not target_tokens:
                raise ODEBFContractError("zsRE target suffix is empty")
            prompts = [
                prefix + tokenizer.decode(target_tokens[:index])
                if not llama or index == 0
                else prefix + " " + tokenizer.decode(target_tokens[:index])
                for index in range(len(target_tokens))
            ]
            targets = [tokenizer.decode(target_tokens[index]) for index in range(len(target_tokens))]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            logits = model(**encoded).logits
            last_non_masked = encoded["attention_mask"].sum(1) - 1
            gathered = logits[
                torch.arange(logits.shape[0], device=logits.device),
                last_non_masked,
                :,
            ]
            correct_ids = tokenizer(targets, padding=True, return_tensors="pt").to(device)["input_ids"]
            correct_ids = correct_ids[:, 1] if llama else correct_ids[:, 0]
            bits = (torch.argmax(gathered, dim=1) == correct_ids).to(dtype=torch.int64).cpu().tolist()
            position_bits.append([int(value) for value in bits])
            logit_slices.extend(row.detach().cpu() for row in gathered)
            span_lengths.append(len(target_tokens))
            processed_tokens += int(encoded["attention_mask"].sum())
            del logits, gathered, encoded, correct_ids
    request_order = hashlib.sha256("".join(str(request["request_sha256"]) for request in batch).encode("ascii")).hexdigest()
    return ModelEvaluationReceipt(
        zsre_batch_receipt(position_bits),
        _hash_tensor_sequence(logit_slices),
        request_order,
        tuple(span_lengths),
        BATCH_SIZE,
        processed_tokens,
        0,
        None,
    )
