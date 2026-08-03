"""Length-normalized rewrite event and controller information firewall."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .contracts import EventReading, MethodContractError, canonical_hash


@dataclass(frozen=True, slots=True)
class ControllerRequest:
    """The complete request visible before a controller action is frozen."""

    case_id: str
    prompt: str
    subject: str
    target_new: str
    target_old: str

    def __post_init__(self) -> None:
        for name in ("case_id", "prompt", "subject", "target_new", "target_old"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MethodContractError(f"controller request {name} is empty")
        if self.prompt.count("{}") != 1:
            raise MethodContractError("controller rewrite prompt must contain one '{}' field")

    @classmethod
    def from_counterfact_row(cls, row: Mapping[str, Any]) -> "ControllerRequest":
        """Project one CounterFact row onto rewrite-only fields.

        Paraphrases, neighborhood prompts, locality answers, and any other
        evaluation fields are neither returned nor retained.
        """

        try:
            case_id = str(row["case_id"])
            rewrite = row["requested_rewrite"]
            return cls(
                case_id=case_id,
                prompt=str(rewrite["prompt"]),
                subject=str(rewrite["subject"]),
                target_new=str(rewrite["target_new"]["str"]),
                target_old=str(rewrite["target_true"]["str"]),
            )
        except (KeyError, TypeError) as exc:
            raise MethodContractError(
                "CounterFact row lacks the controller rewrite schema"
            ) from exc


class InformationFirewall:
    """Hold evaluation payload closed until the controller action is frozen."""

    def __init__(
        self,
        request: ControllerRequest,
        evaluation_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(request, ControllerRequest):
            raise MethodContractError("firewall requires a ControllerRequest")
        self._request = request
        self.__evaluation_payload = dict(evaluation_payload or {})
        self._action_hash: str | None = None

    @property
    def controller_request(self) -> ControllerRequest:
        return self._request

    @property
    def action_frozen(self) -> bool:
        return self._action_hash is not None

    def freeze_action(self, action_payload: Mapping[str, Any]) -> str:
        if self._action_hash is not None:
            raise MethodContractError("controller action may be frozen only once")
        self._action_hash = canonical_hash(dict(action_payload))
        return self._action_hash

    def open_evaluation(self) -> dict[str, Any]:
        if self._action_hash is None:
            raise MethodContractError("evaluation payload is closed before action freeze")
        return dict(self.__evaluation_payload)


@dataclass(frozen=True, slots=True)
class TokenizationPolicy:
    target_prefix_rule: str = "single-leading-space-if-absent"
    target_add_special_tokens: bool = False
    full_add_special_tokens: bool = True
    padding_side: str = "right"
    normalization: str = "mean-log-likelihood-per-object-token"
    bos_unk_rule: str = "drop-one-leading-bos-or-unk-from-target"

    @property
    def policy_id(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


def normalize_object_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MethodContractError("event object text is empty")
    return value if value.startswith(" ") else f" {value}"


def build_allowed_contexts(
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    if not context_templates:
        raise MethodContractError("allowed context manifest is empty")
    contexts: list[str] = []
    for group in context_templates:
        if not group:
            raise MethodContractError("allowed context group is empty")
        for template in group:
            if not isinstance(template, str) or template.count("{}") != 1:
                raise MethodContractError("allowed context template requires one '{}' field")
            prefix = template.format(request.prompt)
            if prefix.count("{}") != 1:
                raise MethodContractError("rewrite/context composition lost the subject field")
            contexts.append(prefix.format(request.subject))
    if len(contexts) != len(set(contexts)):
        raise MethodContractError("allowed controller contexts must be unique")
    return tuple(contexts)


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    input_lengths: tuple[int, ...]
    pad_token_id: int

    def __post_init__(self) -> None:
        if (
            self.input_ids.ndim != 2
            or self.attention_mask.shape != self.input_ids.shape
            or self.target_ids.ndim != 1
            or self.target_ids.numel() <= 0
            or len(self.input_lengths) != self.input_ids.shape[0]
            or isinstance(self.pad_token_id, bool)
            or not isinstance(self.pad_token_id, int)
            or self.pad_token_id < 0
        ):
            raise MethodContractError("teacher-forced batch shapes are invalid")


@dataclass(frozen=True, slots=True)
class DifferentiableEvent:
    """One combined event forward plus its grad-enabled smooth scalar."""

    reading: EventReading
    smooth_phi: torch.Tensor
    target_new_log_likelihoods: tuple[float, ...]
    target_old_log_likelihoods: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.smooth_phi.ndim != 0 or not self.smooth_phi.requires_grad:
            raise MethodContractError("differentiable event scalar has no gradient graph")
        if self.reading.nfe != 1:
            raise MethodContractError("combined differentiable event must use one forward")


def _target_ids(tokenizer: Any, target_text: str) -> tuple[int, ...]:
    values = tuple(
        int(value)
        for value in tokenizer.encode(
            normalize_object_text(target_text),
            add_special_tokens=False,
        )
    )
    bos = getattr(tokenizer, "bos_token_id", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if values and values[0] in {bos, unk}:
        values = values[1:]
    if not values:
        raise MethodContractError("event object tokenization is empty")
    return values


def build_teacher_batch(
    tokenizer: Any,
    contexts: Sequence[str],
    target_text: str,
) -> TeacherBatch:
    if getattr(tokenizer, "padding_side", None) != "right":
        raise MethodContractError("event scoring requires right padding")
    target = _target_ids(tokenizer, target_text)
    normalized_text = normalize_object_text(target_text)
    rows: list[tuple[int, ...]] = []
    for context in contexts:
        if not isinstance(context, str) or not context:
            raise MethodContractError("event context is empty")
        full = tuple(
            int(value)
            for value in tokenizer.encode(
                context + normalized_text,
                add_special_tokens=True,
            )
        )
        if len(full) <= len(target) or full[-len(target) :] != target:
            raise MethodContractError("prefix+object tokenization changed the object suffix")
        model_input = full[:-1]
        if len(model_input) < len(target):
            raise MethodContractError("teacher input is shorter than its object tokens")
        rows.append(model_input)
    if not rows:
        raise MethodContractError("event scorer has no contexts")
    pad = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise MethodContractError("event tokenizer has no valid pad token")
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    return TeacherBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=torch.tensor(target, dtype=torch.long),
        input_lengths=tuple(len(row) for row in rows),
        pad_token_id=pad,
    )


def _selected_log_likelihoods(
    logits: torch.Tensor,
    batch: TeacherBatch,
    *,
    row_offset: int = 0,
) -> torch.Tensor:
    target_count = int(batch.target_ids.numel())
    rows: list[torch.Tensor] = []
    for row_index, length in enumerate(batch.input_lengths):
        start = length - target_count
        rows.append(logits[row_offset + row_index, start:length, :])
    selected = torch.stack(rows, dim=0)
    targets = batch.target_ids.to(logits.device).expand(selected.shape[0], -1)
    log_probs = torch.log_softmax(selected, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    # Mean over each object's own token count is the canonical normalization.
    return token_log_probs.mean(dim=1)


def score_teacher_batch_tensor(
    model: torch.nn.Module,
    batch: TeacherBatch,
    *,
    differentiable: bool,
) -> torch.Tensor:
    device = next(model.parameters()).device
    grad_context = nullcontext() if differentiable else torch.inference_mode()
    with grad_context:
        logits = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
        ).logits.float()
        return _selected_log_likelihoods(logits, batch)


def score_teacher_batch(model: torch.nn.Module, batch: TeacherBatch) -> tuple[float, ...]:
    values = score_teacher_batch_tensor(model, batch, differentiable=False)
    return tuple(float(value) for value in values.cpu())


def score_combined_teacher_batches(
    model: torch.nn.Module,
    batches: Sequence[TeacherBatch],
    *,
    differentiable: bool,
) -> tuple[torch.Tensor, ...]:
    """Score target-specific panels in one right-padded model invocation."""

    locked = tuple(batches)
    if not locked:
        raise MethodContractError("combined teacher scorer has no panels")
    pad_values = {batch.pad_token_id for batch in locked}
    if len(pad_values) != 1:
        raise MethodContractError("combined teacher panels use different pad IDs")
    pad = next(iter(pad_values))
    width = max(int(batch.input_ids.shape[1]) for batch in locked)
    rows = sum(int(batch.input_ids.shape[0]) for batch in locked)
    input_ids = torch.full((rows, width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    offset = 0
    for batch in locked:
        count, batch_width = batch.input_ids.shape
        input_ids[offset : offset + count, :batch_width] = batch.input_ids
        attention_mask[offset : offset + count, :batch_width] = batch.attention_mask
        offset += int(count)
    device = next(model.parameters()).device
    grad_context = nullcontext() if differentiable else torch.inference_mode()
    with grad_context:
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits.float()
        results: list[torch.Tensor] = []
        offset = 0
        for batch in locked:
            results.append(_selected_log_likelihoods(logits, batch, row_offset=offset))
            offset += int(batch.input_ids.shape[0])
    return tuple(results)


def differentiable_event_from_log_likelihoods(
    target_new: torch.Tensor,
    target_old: torch.Tensor,
    *,
    tau: float,
) -> DifferentiableEvent:
    if (
        target_new.ndim != 1
        or target_old.shape != target_new.shape
        or target_new.numel() <= 0
    ):
        raise MethodContractError("differentiable event panels differ")
    temperature = float(tau)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise MethodContractError("event smooth temperature must be positive")
    margins = target_new - target_old
    deficits = -margins
    smooth = temperature * (
        torch.logsumexp(deficits / temperature, dim=0)
        - math.log(int(deficits.numel()))
    )
    new_values = tuple(float(value) for value in target_new.detach().cpu())
    old_values = tuple(float(value) for value in target_old.detach().cpu())
    reading = event_from_log_likelihoods(
        new_values,
        old_values,
        tau=temperature,
        nfe=1,
    )
    return DifferentiableEvent(
        reading=reading,
        smooth_phi=smooth,
        target_new_log_likelihoods=new_values,
        target_old_log_likelihoods=old_values,
    )


def event_from_log_likelihoods(
    target_new: Sequence[float],
    target_old: Sequence[float],
    *,
    tau: float,
    nfe: int = 0,
) -> EventReading:
    new = tuple(float(value) for value in target_new)
    old = tuple(float(value) for value in target_old)
    if not new or len(new) != len(old) or any(not math.isfinite(v) for v in new + old):
        raise MethodContractError("event log-likelihood panels differ or are non-finite")
    temperature = float(tau)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise MethodContractError("event smooth temperature must be positive")
    margins = tuple(new_value - old_value for new_value, old_value in zip(new, old, strict=True))
    deficits = tuple(-margin for margin in margins)
    maximum = max(deficits)
    smooth = maximum + temperature * (
        math.log(
            math.fsum(math.exp((value - maximum) / temperature) for value in deficits)
        )
        - math.log(len(deficits))
    )
    return EventReading(
        hard_phi=maximum,
        smooth_phi=smooth,
        context_margins=margins,
        nfe=nfe,
    )


def measure_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    tau: float,
) -> EventReading:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new, old = score_combined_teacher_batches(
        model,
        (new_batch, old_batch),
        differentiable=False,
    )
    return event_from_log_likelihoods(
        tuple(float(value) for value in new.cpu()),
        tuple(float(value) for value in old.cpu()),
        tau=tau,
        nfe=1,
    )


def measure_differentiable_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    tau: float,
) -> DifferentiableEvent:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new, old = score_combined_teacher_batches(
        model,
        (new_batch, old_batch),
        differentiable=True,
    )
    return differentiable_event_from_log_likelihoods(new, old, tau=tau)


def context_manifest_payload(
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    tokenizer: Any,
) -> dict[str, Any]:
    contexts = build_allowed_contexts(request, context_templates)
    policy = TokenizationPolicy()
    new_ids = _target_ids(tokenizer, request.target_new)
    old_ids = _target_ids(tokenizer, request.target_old)
    return {
        "context_templates": [list(group) for group in context_templates],
        "context_template_hash": canonical_hash(
            [list(group) for group in context_templates]
        ),
        "rendered_context_hash": canonical_hash(list(contexts)),
        "tokenization_policy": policy.to_dict(),
        "tokenization_policy_id": policy.policy_id,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unavailable")),
        "target_suffix_identity": {
            "new": {
                "token_count": len(new_ids),
                "token_ids_sha256": canonical_hash(list(new_ids)),
            },
            "old": {
                "token_count": len(old_ids),
                "token_ids_sha256": canonical_hash(list(old_ids)),
            },
        },
    }
