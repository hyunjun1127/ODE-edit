"""Stock-compute-z target-context inventory and strict first-hit predicate.

Only ``prompt``, ``subject`` and ``target_new`` are consumed.  Rephrase,
neighborhood, target_true and endpoint evaluators are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .contracts import TechnicalBoundary


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().to(device="cpu").tolist()


def normalize_official_requests(requests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for ordinal, item in enumerate(requests):
        for key in ("case_id", "prompt", "subject", "target_new"):
            if key not in item:
                raise TechnicalBoundary(f"request {ordinal} lacks {key}")
        prompt = str(item["prompt"])
        subject = str(item["subject"])
        target_new = str(item["target_new"])
        if not prompt or not subject or not target_new:
            raise TechnicalBoundary("Official request strings must be nonempty")
        if "{}" not in prompt:
            if subject not in prompt:
                raise TechnicalBoundary("request subject is absent from prompt")
            prompt = prompt.replace(subject, "{}")
        if not target_new.startswith(" "):
            target_new = " " + target_new
        normalized.append(
            {
                "case_id": str(item["case_id"]),
                "prompt": prompt,
                "subject": subject,
                "target_new": target_new,
            }
        )
    if not normalized:
        raise TechnicalBoundary("semantic inventory requires at least one request")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SemanticEventRow:
    request_ordinal: int
    context_ordinal: int
    target_token_ordinal: int
    batch_row: int
    logit_position: int
    target_token_id: int


@dataclass(frozen=True, slots=True)
class RequestSemanticBatch:
    request_ordinal: int
    case_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_ids: torch.Tensor
    rows: tuple[SemanticEventRow, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.input_ids.ndim != 2
            or self.input_ids.dtype is not torch.long
            or self.attention_mask.shape != self.input_ids.shape
            or self.position_ids.shape != self.input_ids.shape
            or self.attention_mask.dtype is not torch.long
            or self.position_ids.dtype is not torch.long
            or self.target_ids.ndim != 1
            or self.target_ids.dtype is not torch.long
            or len(self.rows) != self.input_ids.shape[0] * self.target_ids.numel()
            or len(self.identity_sha256) != 64
        ):
            raise TechnicalBoundary("semantic request batch contract differs")


@dataclass(frozen=True, slots=True)
class SemanticInventory:
    batches: tuple[RequestSemanticBatch, ...]
    request_order_sha256: str
    context_templates_sha256: str
    identity_sha256: str
    kl_only_context_count: int

    @property
    def request_count(self) -> int:
        return len(self.batches)

    @property
    def event_count(self) -> int:
        return sum(len(batch.rows) for batch in self.batches)

    def __post_init__(self) -> None:
        if (
            not self.batches
            or tuple(batch.request_ordinal for batch in self.batches) != tuple(range(len(self.batches)))
            or any(len(value) != 64 for value in (
                self.request_order_sha256,
                self.context_templates_sha256,
                self.identity_sha256,
            ))
            or self.kl_only_context_count != len(self.batches)
        ):
            raise TechnicalBoundary("semantic inventory identity/order differs")


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    all_strict: bool
    strict_event_count: int
    event_count: int
    request_strict: tuple[bool, ...]
    request_strict_count: int
    target_logit_mean: float
    target_logit_min: float
    maximum_other_logit_mean: float
    strict_tie_count: int


def _target_ids(tokenizer: Any, target_new: str) -> torch.Tensor:
    raw = tokenizer.encode(target_new, return_tensors="pt", add_special_tokens=False)
    ids = raw[0] if isinstance(raw, torch.Tensor) and raw.ndim == 2 else torch.tensor(raw, dtype=torch.long)
    ids = ids.detach().to(device="cpu", dtype=torch.long).flatten()
    if ids.numel() and int(ids[0]) in {getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "unk_token_id", None)}:
        ids = ids[1:]
    if not ids.numel():
        raise TechnicalBoundary("stock compute-z target token sequence is empty")
    return ids.contiguous()


def build_compute_z_semantic_inventory(
    *,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    context_templates: Sequence[Sequence[str]],
    request_order_sha256: str,
) -> SemanticInventory:
    """Reproduce the target-bearing portion of stock ``compute_z`` exactly."""

    if getattr(tokenizer, "padding_side", None) != "right":
        raise TechnicalBoundary("Official/semantic tokenizer padding_side must be right")
    normalized = normalize_official_requests(requests)
    flattened = tuple(str(context) for group in context_templates for context in group)
    if not flattened or any(context.count("{}") < 1 for context in flattened):
        raise TechnicalBoundary("stock compute-z context templates differ")
    if len(request_order_sha256) != 64:
        raise TechnicalBoundary("request-order identity is not SHA-256")
    context_identity = _canonical_hash(flattened)
    batches: list[RequestSemanticBatch] = []
    for request_ordinal, request in enumerate(normalized):
        target_ids = _target_ids(tokenizer, request["target_new"])
        suffix = tokenizer.decode(target_ids[:-1])
        rewriting_prompts = [
            context.format(request["prompt"]) + suffix
            for context in flattened
        ]
        rendered = [prompt.format(request["subject"]) for prompt in rewriting_prompts]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].detach().to(device="cpu", dtype=torch.long).contiguous()
        attention = encoded["attention_mask"].detach().to(device="cpu", dtype=torch.long).contiguous()
        if input_ids.shape != attention.shape or input_ids.shape[0] != len(flattened):
            raise TechnicalBoundary("stock compute-z target context tokenization differs")
        position = attention.cumsum(dim=1).sub(1).clamp_min(0).mul(attention).long().contiguous()
        rows: list[SemanticEventRow] = []
        target_count = int(target_ids.numel())
        for context_ordinal in range(input_ids.shape[0]):
            length = int(attention[context_ordinal].sum().item())
            if length < target_count:
                raise TechnicalBoundary("compute-z target labels precede context start")
            for target_ordinal, target_id in enumerate(target_ids.tolist()):
                logit_position = length - target_count + target_ordinal
                rows.append(
                    SemanticEventRow(
                        request_ordinal=request_ordinal,
                        context_ordinal=context_ordinal,
                        target_token_ordinal=target_ordinal,
                        batch_row=context_ordinal,
                        logit_position=logit_position,
                        target_token_id=int(target_id),
                    )
                )
        payload = {
            "request_ordinal": request_ordinal,
            "case_id": request["case_id"],
            "input_ids": _tensor_list(input_ids),
            "attention_mask": _tensor_list(attention),
            "position_ids": _tensor_list(position),
            "target_ids": _tensor_list(target_ids),
            "rows": [
                [row.request_ordinal, row.context_ordinal, row.target_token_ordinal,
                 row.batch_row, row.logit_position, row.target_token_id]
                for row in rows
            ],
        }
        batches.append(
            RequestSemanticBatch(
                request_ordinal=request_ordinal,
                case_id=request["case_id"],
                input_ids=input_ids,
                attention_mask=attention,
                position_ids=position,
                target_ids=target_ids,
                rows=tuple(rows),
                identity_sha256=_canonical_hash(payload),
            )
        )
    inventory_payload = {
        "request_order_sha256": request_order_sha256,
        "context_templates_sha256": context_identity,
        "batch_identities": [batch.identity_sha256 for batch in batches],
        "kl_only_context_count": len(batches),  # stock compute-z has one KL prompt/request
    }
    return SemanticInventory(
        batches=tuple(batches),
        request_order_sha256=request_order_sha256,
        context_templates_sha256=context_identity,
        identity_sha256=_canonical_hash(inventory_payload),
        kl_only_context_count=len(batches),
    )


def observe_semantic_predicate(
    model: torch.nn.Module,
    inventory: SemanticInventory,
    *,
    microbatch_size: int = 64,
) -> SemanticObservation:
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
        raise TechnicalBoundary("semantic microbatch size is invalid")
    device = next(model.parameters()).device
    observations: dict[tuple[int, int, int], tuple[bool, float, float, bool]] = {}
    flattened: list[tuple[torch.Tensor, tuple[SemanticEventRow, ...]]] = []
    for batch in inventory.batches:
        by_context: dict[int, list[SemanticEventRow]] = {}
        for row in batch.rows:
            by_context.setdefault(row.context_ordinal, []).append(row)
        for context_ordinal in range(batch.input_ids.shape[0]):
            length = int(batch.attention_mask[context_ordinal].sum().item())
            flattened.append((
                batch.input_ids[context_ordinal, :length].contiguous(),
                tuple(by_context[context_ordinal]),
            ))
    if not flattened:
        raise TechnicalBoundary("semantic inventory has no target-bearing contexts")
    tie_count = 0
    with torch.inference_mode():
        for offset in range(0, len(flattened), microbatch_size):
            group = flattened[offset : offset + microbatch_size]
            maximum = max(int(ids.numel()) for ids, _ in group)
            pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            input_ids = torch.full((len(group), maximum), int(pad_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            position_ids = torch.zeros_like(input_ids)
            for row_index, (ids, _) in enumerate(group):
                length = int(ids.numel())
                input_ids[row_index, :length] = ids.to(device)
                attention_mask[row_index, :length] = 1
                position_ids[row_index, :length] = torch.arange(length, device=device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            if output.dtype is not torch.float32 or output.shape[:2] != input_ids.shape:
                raise TechnicalBoundary("semantic forward left FULL_FP32 or changed token geometry")
            for row_index, (_, rows) in enumerate(group):
                for row in rows:
                    logits = output[row_index, row.logit_position]
                    target = logits[row.target_token_id]
                    masked = logits.clone()
                    masked[row.target_token_id] = -torch.inf
                    other = masked.max()
                    if not bool(torch.isfinite(target)) or not bool(torch.isfinite(other)):
                        raise TechnicalBoundary("semantic event logit is non-finite")
                    strict = bool(target > other)
                    tied = bool(target == other)
                    key = (row.request_ordinal, row.context_ordinal, row.target_token_ordinal)
                    if key in observations:
                        raise TechnicalBoundary("semantic event was observed twice")
                    observations[key] = (strict, float(target.item()), float(other.item()), tied)
                    tie_count += int(tied)
    event_strict: list[bool] = []
    target_logits: list[float] = []
    other_logits: list[float] = []
    per_request: list[bool] = []
    for batch in inventory.batches:
        request_values: list[bool] = []
        for row in batch.rows:
            key = (row.request_ordinal, row.context_ordinal, row.target_token_ordinal)
            try:
                strict, target, other, _ = observations[key]
            except KeyError as exc:
                raise TechnicalBoundary("semantic event was not observed") from exc
            request_values.append(strict)
            event_strict.append(strict)
            target_logits.append(target)
            other_logits.append(other)
        per_request.append(all(request_values))
    if len(event_strict) != inventory.event_count or len(per_request) != inventory.request_count:
        raise TechnicalBoundary("semantic event inventory was not consumed exactly once")
    target_tensor = torch.tensor(target_logits, dtype=torch.float64)
    other_tensor = torch.tensor(other_logits, dtype=torch.float64)
    return SemanticObservation(
        all_strict=all(event_strict),
        strict_event_count=sum(event_strict),
        event_count=len(event_strict),
        request_strict=tuple(per_request),
        request_strict_count=sum(per_request),
        target_logit_mean=float(target_tensor.mean().item()),
        target_logit_min=float(target_tensor.min().item()),
        maximum_other_logit_mean=float(other_tensor.mean().item()),
        strict_tie_count=tie_count,
    )


__all__ = [
    "RequestSemanticBatch",
    "SemanticEventRow",
    "SemanticInventory",
    "SemanticObservation",
    "build_compute_z_semantic_inventory",
    "normalize_official_requests",
    "observe_semantic_predicate",
]
