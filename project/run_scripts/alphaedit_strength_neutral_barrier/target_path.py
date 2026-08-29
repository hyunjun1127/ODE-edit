"""Multi-token target-path observations for the strength-neutral barrier.

Only canonical rewrite prompts and their target prefixes are accepted here.
Locality, rephrase, held-out, and generation prompts have no controller path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Sequence

import torch

from .contracts import prompt_token_ids, target_token_ids


class TargetPathBoundary(RuntimeError):
    """The target-prefix observation contract cannot be satisfied."""


@dataclass(frozen=True)
class TargetEvent:
    event_index: int
    request_index: int
    token_index: int
    target_token_id: int
    context_length: int


@dataclass(frozen=True)
class TargetEventBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    event_weights: torch.Tensor
    events: tuple[TargetEvent, ...]
    request_count: int
    identity_sha256: str

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class TargetPathReference:
    q0: torch.Tensor
    log_q0_safe: torch.Tensor
    target_ids: torch.Tensor
    identity_sha256: str


@dataclass(frozen=True)
class TargetPathValues:
    barrier: torch.Tensor
    per_event_kl: torch.Tensor
    target_log_odds: torch.Tensor
    target_nll: torch.Tensor


def build_target_event_batch(
    tok: Any,
    requests: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
) -> TargetEventBatch:
    """Create one teacher-forced event for every target token.

    The target IDs are appended as IDs rather than re-tokenized strings, so
    every event is byte/order deterministic at the prompt/target boundary.
    """

    if not requests:
        raise TargetPathBoundary("request batch is empty")
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    if pad_id is None:
        raise TargetPathBoundary("tokenizer has neither pad nor eos token")

    contexts: List[List[int]] = []
    events: List[TargetEvent] = []
    token_counts: List[int] = []
    digest = sha256()
    for request_index, request in enumerate(requests):
        prompt = str(request["prompt"]).format(str(request["subject"]))
        prompt_ids = prompt_token_ids(tok, prompt)
        target_ids = target_token_ids(tok, str(request["target_new"]))
        token_counts.append(len(target_ids))
        digest.update(request_index.to_bytes(8, "big"))
        digest.update(str(request.get("case_id", request_index)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(request["target_new"]).encode("utf-8"))
        digest.update(b"\0")
        for token_index, token_id in enumerate(target_ids):
            context = prompt_ids + target_ids[:token_index]
            contexts.append(context)
            events.append(
                TargetEvent(
                    event_index=len(events),
                    request_index=request_index,
                    token_index=token_index,
                    target_token_id=token_id,
                    context_length=len(context),
                )
            )
            digest.update(token_id.to_bytes(8, "big", signed=False))

    max_length = max(len(context) for context in contexts)
    input_ids = torch.full(
        (len(contexts), max_length), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, context in enumerate(contexts):
        values = torch.tensor(context, dtype=torch.long, device=device)
        input_ids[row, -len(context) :] = values
        attention_mask[row, -len(context) :] = 1

    target_tensor = torch.tensor(
        [event.target_token_id for event in events], dtype=torch.long, device=device
    )
    weights = torch.tensor(
        [
            1.0 / (len(requests) * token_counts[event.request_index])
            for event in events
        ],
        dtype=torch.float32,
        device=device,
    )
    return TargetEventBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_tensor,
        event_weights=weights,
        events=tuple(events),
        request_count=len(requests),
        identity_sha256=digest.hexdigest(),
    )


def build_sequence_event_batch(
    tok: Any,
    requests: Sequence[Dict[str, Any]],
    *,
    target_key: str,
    device: torch.device,
) -> TargetEventBatch:
    """Build observation-only teacher-forced events for an alternate target."""

    alternate = []
    for request in requests:
        if target_key not in request:
            raise TargetPathBoundary(f"request is missing {target_key}")
        item = dict(request)
        item["target_new"] = request[target_key]
        alternate.append(item)
    return build_target_event_batch(tok, alternate, device=device)


def _log_q_excluding_target(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] != target_ids.numel():
        raise TargetPathBoundary("logit/event shape mismatch")
    masked = logits.clone()
    masked.scatter_(1, target_ids[:, None], -torch.inf)
    log_z_non_target = torch.logsumexp(masked, dim=-1)
    if not torch.isfinite(log_z_non_target).all():
        raise TargetPathBoundary("non-target log partition is nonfinite")
    log_q = logits - log_z_non_target[:, None]
    log_q = log_q.scatter(1, target_ids[:, None], -torch.inf)
    return log_q, log_z_non_target


def capture_reference(
    logits: torch.Tensor, batch: TargetEventBatch
) -> TargetPathReference:
    log_q0, _ = _log_q_excluding_target(logits.float(), batch.target_ids)
    q0 = torch.exp(log_q0)
    q0 = q0.scatter(1, batch.target_ids[:, None], 0.0)
    log_q0_safe = log_q0.scatter(1, batch.target_ids[:, None], 0.0)
    normalization = q0.sum(dim=-1)
    if not torch.allclose(
        normalization,
        torch.ones_like(normalization),
        atol=8 * torch.finfo(torch.float32).eps,
        rtol=8 * torch.finfo(torch.float32).eps,
    ):
        raise TargetPathBoundary("target-excluded reference is not normalized")
    payload = q0.detach().cpu().contiguous().numpy().tobytes()
    return TargetPathReference(
        q0=q0.detach(),
        log_q0_safe=log_q0_safe.detach(),
        target_ids=batch.target_ids.detach(),
        identity_sha256=sha256(payload).hexdigest(),
    )


def evaluate_values(
    logits: torch.Tensor,
    batch: TargetEventBatch,
    reference: TargetPathReference,
) -> TargetPathValues:
    logits = logits.float()
    log_q, log_z_non_target = _log_q_excluding_target(logits, batch.target_ids)
    log_q_safe = log_q.scatter(1, batch.target_ids[:, None], 0.0)
    per_event_kl = torch.sum(
        reference.q0 * (reference.log_q0_safe - log_q_safe), dim=-1
    )
    barrier = torch.sum(batch.event_weights * per_event_kl)
    target_logits = logits.gather(1, batch.target_ids[:, None]).squeeze(1)
    odds = target_logits - log_z_non_target
    nll = torch.nn.functional.softplus(-odds)
    tensors = (barrier, per_event_kl, odds, nll)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise TargetPathBoundary("target-path observation is nonfinite")
    return TargetPathValues(
        barrier=barrier,
        per_event_kl=per_event_kl,
        target_log_odds=odds,
        target_nll=nll,
    )


def sequence_nll_by_request(
    per_event_nll: torch.Tensor, batch: TargetEventBatch
) -> torch.Tensor:
    """Return token-mean NLL for each request in deterministic request order."""

    if per_event_nll.ndim != 1 or per_event_nll.numel() != batch.event_count:
        raise TargetPathBoundary("sequence NLL/event shape mismatch")
    result = per_event_nll.new_zeros((batch.request_count,))
    counts = per_event_nll.new_zeros((batch.request_count,))
    for event in batch.events:
        result[event.request_index] = (
            result[event.request_index] + per_event_nll[event.event_index]
        )
        counts[event.request_index] = counts[event.request_index] + 1
    if torch.any(counts == 0):
        raise TargetPathBoundary("request has no target-token events")
    result = result / counts
    if not torch.isfinite(result).all():
        raise TargetPathBoundary("sequence NLL is nonfinite")
    return result


def evaluate_token_nll(
    logits: torch.Tensor, batch: TargetEventBatch
) -> torch.Tensor:
    """Evaluate observation-only teacher-forced token NLLs."""

    logits = logits.float()
    if logits.ndim != 2 or logits.shape[0] != batch.event_count:
        raise TargetPathBoundary("token NLL logit/event shape mismatch")
    values = -torch.log_softmax(logits, dim=-1).gather(
        1, batch.target_ids[:, None]
    ).squeeze(1)
    if not torch.isfinite(values).all():
        raise TargetPathBoundary("token NLL is nonfinite")
    return values
