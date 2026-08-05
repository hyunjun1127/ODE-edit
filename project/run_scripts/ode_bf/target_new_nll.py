"""Locked, teacher-forced B10 routing objectives.

The controller's historical objective is the mean ``new_nll - true_nll``.
``TARGET_NEW_NLL`` deliberately evaluates only the new target so its value and
autograd graph cannot depend on an old target.  It renders the locked 1+5
controller contexts for every request and takes uniform context, request, and
suffix-token means.  Both objectives retain the controller's tokenization
convention: ``prefix + " " + target``, a separately tokenized leading-space
suffix, and Llama's BOS/logit shift.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

import torch

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .request_digest import ordered_request_digest_v1


class RoutingObjective(str, Enum):
    """The only routing objectives accepted by the locked selector."""

    MARGIN = "MARGIN"
    TARGET_NEW_NLL = "TARGET_NEW_NLL"


def select_locked_routing_objective(value: RoutingObjective | str) -> RoutingObjective:
    """Return one exact objective name; aliases and fallbacks are forbidden."""

    if isinstance(value, RoutingObjective):
        return value
    if type(value) is str:
        try:
            return RoutingObjective(value)
        except ValueError as exc:
            raise ODEBFContractError("routing objective is not locked") from exc
    raise ODEBFContractError("routing objective is not locked")


@dataclass(frozen=True, slots=True)
class RoutingObjectiveResult:
    """Differentiable B10 objective plus raw-free receipt accounting."""

    objective: RoutingObjective
    loss: torch.Tensor
    per_request_values: torch.Tensor
    suffix_token_counts: tuple[int, ...]
    target_true_suffix_token_counts: tuple[int | None, ...]
    request_sha256: tuple[str, ...]
    request_order_sha256: str
    target_span_identities: tuple[str, ...]
    target_span_sha256: str
    context_group_sizes: tuple[int, ...]
    context_count: int
    context_sha256: str
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int
    input_gradient: torch.Tensor | None = None
    backward_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0:
            raise ODEBFContractError("routing objective loss is not scalar")
        if (
            not isinstance(self.per_request_values, torch.Tensor)
            or self.per_request_values.ndim != 1
            or self.per_request_values.numel() != BATCH_SIZE
        ):
            raise ODEBFContractError("routing objective request values differ")
        if not bool(torch.isfinite(self.loss.detach()).all()) or not bool(
            torch.isfinite(self.per_request_values.detach()).all()
        ):
            raise ODEBFContractError("routing objective is non-finite")
        if (
            len(self.suffix_token_counts) != BATCH_SIZE
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in self.suffix_token_counts
            )
        ):
            raise ODEBFContractError("routing objective target-new spans are invalid")
        if len(self.target_true_suffix_token_counts) != BATCH_SIZE:
            raise ODEBFContractError("routing objective target-true spans differ")
        if self.objective is RoutingObjective.TARGET_NEW_NLL:
            if any(value is not None for value in self.target_true_suffix_token_counts):
                raise ODEBFContractError("target-new objective observed an old target")
        elif self.objective is RoutingObjective.MARGIN:
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in self.target_true_suffix_token_counts
            ):
                raise ODEBFContractError("routing objective target-true spans are invalid")
        else:
            raise ODEBFContractError("routing objective is not locked")
        if (
            len(self.request_sha256) != BATCH_SIZE
            or len(self.target_span_identities) != BATCH_SIZE
            or self.request_order_sha256
            != ordered_request_digest_v1(self.request_sha256)
            or any(len(value) != 64 for value in self.target_span_identities)
            or len(self.target_span_sha256) != 64
        ):
            raise ODEBFContractError("routing objective receipt identities differ")
        expected_context_geometry = (
            ((1, 5), 6)
            if self.objective is RoutingObjective.TARGET_NEW_NLL
            else ((1,), 1)
        )
        if (
            (self.context_group_sizes, self.context_count)
            != expected_context_geometry
            or len(self.context_sha256) != 64
        ):
            raise ODEBFContractError("routing objective context geometry differs")
        if (
            self.model_forward_count != BATCH_SIZE * self.context_count
            or self.processed_token_count <= 0
            or self.generation_call_count != 0
        ):
            raise ODEBFContractError("routing objective accounting differs")
        if self.input_gradient is None:
            if self.backward_count != 0:
                raise ODEBFContractError("routing objective backward accounting differs")
        elif (
            self.objective is not RoutingObjective.TARGET_NEW_NLL
            or not isinstance(self.input_gradient, torch.Tensor)
            or self.input_gradient.ndim != 1
            or not bool(torch.isfinite(self.input_gradient).all())
            or self.input_gradient.requires_grad
            or self.backward_count != BATCH_SIZE
            or self.loss.requires_grad
            or self.per_request_values.requires_grad
        ):
            raise ODEBFContractError("routing objective streamed gradient differs")

    @property
    def value(self) -> torch.Tensor:
        """Compatibility spelling for the differentiable scalar loss."""

        return self.loss

    @property
    def graph_requires_grad(self) -> bool:
        return bool(self.loss.requires_grad and self.per_request_values.requires_grad)


@dataclass(frozen=True, slots=True)
class _SuffixScore:
    value: torch.Tensor
    token_count: int
    span_identity: str


@dataclass(frozen=True, slots=True)
class _ModelState:
    parameters: tuple[tuple[str, int, int, bool, int | None, int | None], ...]
    buffers: tuple[tuple[str, int, int], ...]
    training: tuple[tuple[str, bool], ...]


def _model_state(model: torch.nn.Module) -> _ModelState:
    parameters: list[tuple[str, int, int, bool, int | None, int | None]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        parameters.append(
            (
                name,
                parameter.data_ptr(),
                parameter._version,
                parameter.requires_grad,
                None if gradient is None else gradient.data_ptr(),
                None if gradient is None else gradient._version,
            )
        )
    buffers = tuple(
        (name, buffer.data_ptr(), buffer._version)
        for name, buffer in model.named_buffers()
    )
    training = tuple((name, module.training) for name, module in model.named_modules())
    return _ModelState(tuple(parameters), buffers, training)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ODEBFContractError("routing objective model has no parameters") from exc


@contextmanager
def _temporary_left_padding(tokenizer: Any) -> Iterator[None]:
    """Set the controller's left-padding convention without leaking state."""

    try:
        original = tokenizer.padding_side
    except AttributeError as exc:
        raise ODEBFContractError("routing objective tokenizer lacks padding_side") from exc
    if original not in ("left", "right"):
        raise ODEBFContractError("routing objective tokenizer padding side differs")
    try:
        tokenizer.padding_side = "left"
    except Exception as exc:
        raise ODEBFContractError("routing objective cannot lock left padding") from exc
    if getattr(tokenizer, "padding_side", None) != "left":
        try:
            tokenizer.padding_side = original
        except Exception:
            pass
        raise ODEBFContractError("routing objective tokenizer rejected left padding")
    try:
        yield
    finally:
        try:
            tokenizer.padding_side = original
        except Exception as exc:
            raise ODEBFContractError("routing objective tokenizer padding restoration failed") from exc
        if getattr(tokenizer, "padding_side", None) != original:
            raise ODEBFContractError("routing objective tokenizer padding leaked")


def _tokenize_left(tokenizer: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a tokenizer only while its locked padding convention is intact."""

    if getattr(tokenizer, "padding_side", None) != "left":
        raise ODEBFContractError("routing objective tokenizer padding leaked")
    encoded = tokenizer(*args, **kwargs)
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ODEBFContractError("routing objective tokenizer padding leaked")
    return encoded


def _ordered_batch(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...], str]:
    batch = tuple(requests)
    if len(batch) != BATCH_SIZE:
        raise ODEBFContractError("routing objective requires one joint B10")
    identities: list[str] = []
    for request in batch:
        if not isinstance(request, Mapping):
            raise ODEBFContractError("routing objective request is not a mapping")
        try:
            identity = request["request_sha256"]
        except KeyError as exc:
            raise ODEBFContractError("routing objective request identity is absent") from exc
        if not isinstance(identity, str):
            raise ODEBFContractError("routing objective request identity differs")
        identities.append(identity)
    ordered = tuple(identities)
    return batch, ordered, ordered_request_digest_v1(ordered)


def _surface(
    request: Mapping[str, Any],
    target_key: str,
    *,
    context_template: str | None,
) -> tuple[str, str]:
    try:
        prompt = str(request["prompt"])
        subject = str(request["subject"])
        prefix = (
            prompt.format(subject)
            if context_template is None
            else context_template.format(prompt).format(subject)
        )
        target = str(request[target_key])
    except (KeyError, IndexError, ValueError) as exc:
        raise ODEBFContractError("routing objective request surface differs") from exc
    if not prefix or not target:
        raise ODEBFContractError("routing objective request surface is empty")
    return prefix, target


def _locked_context_templates(
    objective: RoutingObjective,
    contexts: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str | None, ...], tuple[int, ...], str]:
    if objective is RoutingObjective.MARGIN:
        if contexts is not None:
            raise ODEBFContractError("historical margin received context expansion")
        return (
            (None,),
            (1,),
            canonical_hash(
                {
                    "schema": "ode-edit-s05-routing-contexts/v1",
                    "surface": "canonical-request-only",
                }
            ),
        )
    if contexts is None:
        raise ODEBFContractError("target-new routing contexts are absent")
    groups = tuple(tuple(group) for group in contexts)
    sizes = tuple(len(group) for group in groups)
    if sizes != (1, 5):
        raise ODEBFContractError("target-new routing context groups differ")
    flattened: list[str] = []
    for group in groups:
        for template in group:
            if not isinstance(template, str) or not template or template.count("{}") != 1:
                raise ODEBFContractError("target-new routing context template differs")
            flattened.append(template)
    return (
        tuple(flattened),
        sizes,
        canonical_hash(
            {
                "schema": "ode-edit-s05-routing-contexts/v1",
                "group_sizes": list(sizes),
                "templates": flattened,
            }
        ),
    )


def _input_ids(value: Any, *, label: str, one_row: bool) -> tuple[tuple[int, ...], ...]:
    try:
        ids = value["input_ids"]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError(f"routing objective {label} has no input_ids") from exc
    if isinstance(ids, torch.Tensor):
        if ids.ndim == 1:
            rows = (tuple(ids.detach().to(device="cpu").tolist()),)
        elif ids.ndim == 2:
            rows = tuple(
                tuple(row.detach().to(device="cpu").tolist()) for row in ids
            )
        else:
            raise ODEBFContractError(f"routing objective {label} input_ids rank differs")
    else:
        try:
            raw = tuple(ids)
        except TypeError as exc:
            raise ODEBFContractError(f"routing objective {label} input_ids differ") from exc
        if raw and isinstance(raw[0], (list, tuple, torch.Tensor)):
            rows = tuple(tuple(row) for row in raw)
        else:
            rows = (raw,)
    if one_row and len(rows) != 1:
        raise ODEBFContractError(f"routing objective {label} row count differs")
    normalized: list[tuple[int, ...]] = []
    for row in rows:
        checked: list[int] = []
        for token in row:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ODEBFContractError(f"routing objective {label} token differs")
            checked.append(int(token))
        normalized.append(tuple(checked))
    return tuple(normalized)


def _suffix_tokens(tokenizer: Any, target: str, *, llama: bool) -> tuple[int, ...]:
    tokens = _input_ids(
        _tokenize_left(tokenizer, f" {target}"), label="target suffix", one_row=True
    )[0]
    if llama:
        tokens = tokens[1:]
    if not tokens:
        raise ODEBFContractError("routing objective target suffix is empty")
    return tokens


def _move_encoding(encoding: Any, device: torch.device) -> Mapping[str, Any]:
    try:
        moved = encoding.to(device)
    except AttributeError:
        if not isinstance(encoding, Mapping):
            raise ODEBFContractError("routing objective tokenizer encoding differs")
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoding.items()
        }
    if not isinstance(moved, Mapping):
        raise ODEBFContractError("routing objective tokenizer encoding differs")
    return moved


def _left_padding_offsets(
    encoded: Mapping[str, Any],
    *,
    rows: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    input_ids = encoded.get("input_ids")
    attention = encoded.get("attention_mask")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != rows
        or input_ids.dtype
        not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        or not isinstance(attention, torch.Tensor)
        or attention.shape != input_ids.shape
    ):
        raise ODEBFContractError("routing objective left-padded encoding differs")
    if not bool(torch.isfinite(attention.float()).all()):
        raise ODEBFContractError("routing objective attention mask is non-finite")
    offsets: list[int] = []
    for row in range(rows):
        mask = attention[row].detach().to(device="cpu")
        raw_values = tuple(mask.tolist())
        if not raw_values or any(value not in (0, 1) for value in raw_values):
            raise ODEBFContractError("routing objective attention mask differs")
        values = tuple(int(value) for value in raw_values)
        try:
            offset = values.index(1)
        except ValueError as exc:
            raise ODEBFContractError("routing objective attention row is empty") from exc
        if any(value != 0 for value in values[:offset]) or any(
            value != 1 for value in values[offset:]
        ):
            raise ODEBFContractError("routing objective encoding is not left padded")
        offsets.append(offset)
    return input_ids, tuple(offsets)


def _is_llama(model: torch.nn.Module) -> bool:
    return "llama" in str(
        getattr(getattr(model, "config", object()), "_name_or_path", "")
    ).casefold()


def _score_suffix(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    row: int,
    prefix_length: int,
    left_padding: int,
    tokens: tuple[int, ...],
    llama: bool,
    request_sha256: str,
    ordinal: int,
    context_ordinal: int,
    context_sha256: str,
    target_label: str,
) -> _SuffixScore:
    if logits.ndim != 3 or logits.shape[0] <= row:
        raise ODEBFContractError("routing objective logits differ")
    if llama:
        logits = logits[:, 1:, :]
        prefix_length -= 1
    if prefix_length <= 0:
        raise ODEBFContractError("routing objective prefix span is invalid")
    # ``prefix_length`` is shifted after slicing Llama logits; its BOS is
    # restored only when locating the suffix in the original input tensor.
    input_start = left_padding + prefix_length + int(llama)
    input_stop = input_start + len(tokens)
    if (
        input_start < 1
        or input_stop > input_ids.shape[1]
        or tuple(int(value) for value in input_ids[row, input_start:input_stop].detach().to(device="cpu").tolist())
        != tokens
    ):
        raise ODEBFContractError("routing objective target suffix is misaligned")
    token_losses: list[torch.Tensor] = []
    logit_indices: list[int] = []
    for offset, token in enumerate(tokens):
        position = left_padding + prefix_length + offset - 1
        if position < 0 or position >= logits.shape[1] or token >= logits.shape[2]:
            raise ODEBFContractError("routing objective target span is out of bounds")
        value = -torch.log_softmax(logits[row, position, :].float(), dim=0)[token]
        if not bool(torch.isfinite(value.detach())):
            raise ODEBFContractError("routing objective token NLL is non-finite")
        token_losses.append(value)
        logit_indices.append(position)
    value = torch.stack(token_losses).mean()
    if not bool(torch.isfinite(value.detach())):
        raise ODEBFContractError("routing objective suffix NLL is non-finite")
    return _SuffixScore(
        value,
        len(tokens),
        canonical_hash(
            {
                "schema": "ode-edit-s05-target-new-nll-span/v1",
                "request_sha256": request_sha256,
                "ordinal": ordinal,
                "context_ordinal": context_ordinal,
                "context_sha256": context_sha256,
                "target": target_label,
                "padding_side": "left",
                "llama_logit_shift": llama,
                "suffix_token_count": len(tokens),
                "input_start": input_start,
                "input_stop": input_stop,
                "logit_start": logit_indices[0],
                "logit_stop": logit_indices[-1] + 1,
            }
        ),
    )


def _score_request(
    model: torch.nn.Module,
    tokenizer: Any,
    request: Mapping[str, Any],
    *,
    objective: RoutingObjective,
    request_sha256: str,
    ordinal: int,
    device: torch.device,
    llama: bool,
    context_template: str | None,
    context_ordinal: int,
    context_sha256: str,
) -> tuple[torch.Tensor, _SuffixScore, _SuffixScore | None, int]:
    prefix, target_new = _surface(
        request, "target_new", context_template=context_template
    )
    prefix_length = len(
        _input_ids(_tokenize_left(tokenizer, [prefix]), label="prefix", one_row=True)[0]
    )
    new_tokens = _suffix_tokens(tokenizer, target_new, llama=llama)
    texts = [f"{prefix} {target_new}"]
    true_tokens: tuple[int, ...] | None = None
    if objective is RoutingObjective.MARGIN:
        _, target_true = _surface(
            request, "target_true", context_template=context_template
        )
        true_tokens = _suffix_tokens(tokenizer, target_true, llama=llama)
        texts.append(f"{prefix} {target_true}")
    encoded = _move_encoding(
        _tokenize_left(tokenizer, texts, padding=True, return_tensors="pt"), device
    )
    input_ids, left_padding = _left_padding_offsets(encoded, rows=len(texts))
    logits = model(**encoded).logits
    if not isinstance(logits, torch.Tensor):
        raise ODEBFContractError("routing objective model logits differ")
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ODEBFContractError("routing objective tokenizer padding leaked")
    new_score = _score_suffix(
        logits=logits,
        input_ids=input_ids,
        row=0,
        prefix_length=prefix_length,
        left_padding=left_padding[0],
        tokens=new_tokens,
        llama=llama,
        request_sha256=request_sha256,
        ordinal=ordinal,
        context_ordinal=context_ordinal,
        context_sha256=context_sha256,
        target_label="target_new",
    )
    attention = encoded["attention_mask"]
    processed = int(attention.detach().to(device="cpu").sum().item())
    if objective is RoutingObjective.TARGET_NEW_NLL:
        return new_score.value, new_score, None, processed
    if true_tokens is None:
        raise ODEBFContractError("routing margin lacks old target tokens")
    true_score = _score_suffix(
        logits=logits,
        input_ids=input_ids,
        row=1,
        prefix_length=prefix_length,
        left_padding=left_padding[1],
        tokens=true_tokens,
        llama=llama,
        request_sha256=request_sha256,
        ordinal=ordinal,
        context_ordinal=context_ordinal,
        context_sha256=context_sha256,
        target_label="target_true",
    )
    return new_score.value - true_score.value, new_score, true_score, processed


def evaluate_routing_objective(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    objective: RoutingObjective | str,
    contexts: Sequence[Sequence[str]] | None = None,
    gradient_input: torch.Tensor | None = None,
) -> RoutingObjectiveResult:
    """Evaluate one locked routing objective without mutating the model.

    TARGET_NEW_NLL deliberately reaches only ``prompt``, ``subject``,
    ``target_new``, and ``request_sha256`` on each request.  In particular it
    never reads or tokenizes ``target_true``.
    """

    selected = select_locked_routing_objective(objective)
    if gradient_input is not None and (
        selected is not RoutingObjective.TARGET_NEW_NLL
        or not isinstance(gradient_input, torch.Tensor)
        or gradient_input.ndim != 1
        or not gradient_input.requires_grad
    ):
        raise ODEBFContractError("routing objective streamed gradient input differs")
    batch, request_sha256, request_order_sha256 = _ordered_batch(requests)
    device = _model_device(model)
    llama = _is_llama(model)
    context_templates, context_group_sizes, context_sha256 = (
        _locked_context_templates(selected, contexts)
    )
    before = _model_state(model)
    failure: BaseException | None = None
    result: RoutingObjectiveResult | None = None
    try:
        with _temporary_left_padding(tokenizer):
            values: list[torch.Tensor] = []
            new_counts: list[int] = []
            true_counts: list[int | None] = []
            span_identities: list[str] = []
            processed_tokens = 0
            streamed_gradient = (
                None
                if gradient_input is None
                else torch.zeros_like(gradient_input, memory_format=torch.preserve_format)
            )
            for ordinal, (request, identity) in enumerate(zip(batch, request_sha256)):
                context_values: list[torch.Tensor] = []
                context_new_counts: list[int] = []
                context_true_counts: list[int | None] = []
                context_spans: list[str] = []
                for context_ordinal, context_template in enumerate(context_templates):
                    value, new_score, true_score, processed = _score_request(
                        model,
                        tokenizer,
                        request,
                        objective=selected,
                        request_sha256=identity,
                        ordinal=ordinal,
                        device=device,
                        llama=llama,
                        context_template=context_template,
                        context_ordinal=context_ordinal,
                        context_sha256=context_sha256,
                    )
                    context_values.append(value)
                    context_new_counts.append(new_score.token_count)
                    context_true_counts.append(
                        None if true_score is None else true_score.token_count
                    )
                    context_spans.append(
                        canonical_hash(
                            {
                                "context_ordinal": context_ordinal,
                                "target_new": new_score.span_identity,
                                "target_true": None
                                if true_score is None
                                else true_score.span_identity,
                            }
                        )
                    )
                    processed_tokens += processed
                if len(set(context_new_counts)) != 1 or len(
                    set(context_true_counts)
                ) != 1:
                    raise ODEBFContractError(
                        "routing objective target span differs across contexts"
                    )
                request_value = torch.stack(context_values).mean()
                if gradient_input is None:
                    values.append(request_value)
                else:
                    request_gradient = torch.autograd.grad(
                        request_value,
                        gradient_input,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    if not bool(torch.isfinite(request_gradient.detach()).all()):
                        raise ODEBFContractError(
                            "routing objective streamed gradient is non-finite"
                        )
                    assert streamed_gradient is not None
                    streamed_gradient.add_(request_gradient.detach())
                    values.append(request_value.detach())
                new_counts.append(context_new_counts[0])
                true_counts.append(context_true_counts[0])
                span_identities.append(
                    canonical_hash(
                        {
                            "schema": "ode-edit-s05-target-new-nll-request-span/v1",
                            "context_sha256": context_sha256,
                            "contexts": context_spans,
                        }
                    )
                )
            per_request = torch.stack(values)
            loss = per_request.mean()
            if streamed_gradient is not None:
                streamed_gradient.div_(BATCH_SIZE)
            result = RoutingObjectiveResult(
                selected,
                loss,
                per_request,
                tuple(new_counts),
                tuple(true_counts),
                request_sha256,
                request_order_sha256,
                tuple(span_identities),
                canonical_hash(
                    {
                        "schema": "ode-edit-s05-target-new-nll-span-vector/v1",
                        "request_order_sha256": request_order_sha256,
                        "spans": span_identities,
                    }
                ),
                context_group_sizes,
                len(context_templates),
                context_sha256,
                BATCH_SIZE * len(context_templates),
                processed_tokens,
                0,
                streamed_gradient,
                0 if streamed_gradient is None else BATCH_SIZE,
            )
    except BaseException as exc:
        failure = exc
    after = _model_state(model)
    if after != before:
        mutation = ODEBFContractError("routing objective mutated model state")
        if failure is not None:
            raise mutation from failure
        raise mutation
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if result is None:
        raise ODEBFContractError("routing objective did not produce a result")
    return result
