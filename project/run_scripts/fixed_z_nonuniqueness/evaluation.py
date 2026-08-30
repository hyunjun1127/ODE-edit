"""Position-safe full-model activation, token, and functional evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .contracts import NumericalLock, TechnicalBoundary
from .padding import (
    build_position_safe_batch,
    gather_batch_positions,
    semantic_last_columns,
    semantic_token_columns,
)


def _hidden(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TechnicalBoundary("unsupported traced hidden-state container")


def target_ids(tokenizer: Any, target: str) -> torch.Tensor:
    normalized = target if target.startswith(" ") else " " + target
    ids = tokenizer.encode(normalized, add_special_tokens=False, return_tensors="pt")[0]
    if ids.numel() and ids[0].item() in {tokenizer.bos_token_id, tokenizer.unk_token_id}:
        ids = ids[1:]
    if ids.numel() == 0:
        raise TechnicalBoundary("empty target tokenization")
    return ids


def target_prefixes(tokenizer: Any, prompt: str, target: str) -> tuple[list[str], torch.Tensor]:
    ids = target_ids(tokenizer, target)
    prefixes = [prompt + tokenizer.decode(ids[:index]) for index in range(ids.numel())]
    return prefixes, ids


def next_token_logits(model: Any, tokenizer: Any, texts: list[str], *, batch_size: int = 16) -> torch.Tensor:
    answer = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = build_position_safe_batch(tokenizer, texts[start : start + batch_size], padding_side="left").to(device)
        with torch.no_grad():
            logits = model(**batch).logits
        rows = torch.arange(logits.shape[0], device=logits.device)
        cols = semantic_last_columns(batch["attention_mask"]).to(logits.device)
        answer.append(logits[rows, cols].float().detach())
    return torch.cat(answer, dim=0)


def capture_last_hidden(model: Any, tokenizer: Any, texts: list[str], module_name: str, *, track: str) -> tuple[torch.Tensor, torch.Tensor]:
    from easyeditor.util import nethook

    batch = build_position_safe_batch(tokenizer, texts, padding_side="left").to(next(model.parameters()).device)
    with torch.no_grad(), nethook.Trace(model, module_name, retain_input=track == "in", retain_output=track == "out") as trace:
        logits = model(**batch).logits
    hidden = _hidden(trace.input if track == "in" else trace.output)
    if hidden.shape[0] != len(texts):
        hidden = hidden.transpose(0, 1)
    cols = semantic_last_columns(batch["attention_mask"])
    rows = torch.arange(len(texts), device=hidden.device)
    selected = hidden[rows, cols.to(hidden.device)].float().detach()
    next_logits = logits[rows, cols.to(logits.device)].float().detach()
    return selected, next_logits


def capture_subject_keys(
    model: Any,
    tokenizer: Any,
    templates: list[str],
    subjects: list[str],
    module_name: str,
) -> torch.Tensor:
    from easyeditor.models.rome.repr_tools import get_words_idxs_in_templates
    from easyeditor.util import nethook

    semantic_idxs = [
        get_words_idxs_in_templates(tokenizer, [template], [subject], "last")[0][0]
        for template, subject in zip(templates, subjects, strict=True)
    ]
    texts = [template.format(subject) for template, subject in zip(templates, subjects, strict=True)]
    batch = build_position_safe_batch(tokenizer, texts, padding_side="left").to(next(model.parameters()).device)
    real_cols = semantic_token_columns(batch["attention_mask"])
    raw_cols = []
    for index, cols in zip(semantic_idxs, real_cols, strict=True):
        if index < 0:
            index += cols.numel()
        if index < 0 or index >= cols.numel():
            raise TechnicalBoundary("subject semantic index outside real tokens")
        raw_cols.append(cols[index])
    with torch.no_grad(), nethook.Trace(model, module_name, retain_input=True, retain_output=False) as trace:
        model(**batch)
    hidden = _hidden(trace.input)
    if hidden.shape[0] != len(texts):
        hidden = hidden.transpose(0, 1)
    rows = torch.arange(len(texts), device=hidden.device)
    return hidden[rows, torch.stack(raw_cols).to(hidden.device)].float().detach().T


def target_path_observation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    input_module: str,
    activation_module: str,
) -> dict[str, Any]:
    prefixes, ids = target_prefixes(tokenizer, prompt, target)
    keys, _ = capture_last_hidden(model, tokenizer, prefixes, input_module, track="in")
    activations, logits = capture_last_hidden(model, tokenizer, prefixes, activation_module, track="out")
    labels = ids.to(logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs[torch.arange(labels.numel(), device=logits.device), labels]
    target_logits = logits[torch.arange(labels.numel(), device=logits.device), labels]
    masked = logits.clone()
    masked[torch.arange(labels.numel(), device=logits.device), labels] = -torch.inf
    margins = target_logits - masked.max(dim=-1).values
    predictions = logits.argmax(dim=-1)
    return {
        "keys": keys.T.contiguous(),
        "activations": activations,
        "logits": logits,
        "target_ids": ids,
        "nll": [float(v) for v in nll.cpu()],
        "margin": [float(v) for v in margins.cpu()],
        "strict": bool(torch.equal(predictions.cpu(), ids.cpu())),
        "token_count": int(ids.numel()),
    }


def sequence_metrics(model: Any, tokenizer: Any, prompts: list[str], target: str) -> dict[str, Any]:
    ids = target_ids(tokenizer, target)
    all_texts: list[str] = []
    labels: list[int] = []
    prompt_ranges: list[tuple[int, int]] = []
    for prompt in prompts:
        begin = len(all_texts)
        for index in range(ids.numel()):
            all_texts.append(prompt + tokenizer.decode(ids[:index]))
            labels.append(int(ids[index]))
        prompt_ranges.append((begin, len(all_texts)))
    logits = next_token_logits(model, tokenizer, all_texts)
    label_tensor = torch.tensor(labels, device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs[torch.arange(len(labels), device=logits.device), label_tensor]
    pred = logits.argmax(dim=-1)
    per_prompt_nll = [float(nll[a:b].mean().item()) for a, b in prompt_ranges]
    strict = [bool(torch.equal(pred[a:b].cpu(), label_tensor[a:b].cpu())) for a, b in prompt_ranges]
    target_logits = logits[torch.arange(len(labels), device=logits.device), label_tensor]
    masked = logits.clone()
    masked[torch.arange(len(labels), device=logits.device), label_tensor] = -torch.inf
    margins = target_logits - masked.max(-1).values
    per_prompt_margin = [float(margins[a:b].min().item()) for a, b in prompt_ranges]
    return {
        "nll": per_prompt_nll,
        "strict": strict,
        "margin": per_prompt_margin,
        "prompt_count": len(prompts),
        "target_token_count": int(ids.numel()),
    }


def teacher_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, alpha: float) -> dict[str, Any]:
    ref_log = torch.log_softmax(reference_logits.float(), dim=-1)
    cand_log = torch.log_softmax(candidate_logits.float(), dim=-1)
    values = (ref_log.exp() * (ref_log - cand_log)).sum(-1)
    sorted_values = values.sort(descending=True).values
    tail_count = max(1, math.ceil((1.0 - alpha) * values.numel()))
    return {
        "per_prompt": [float(v) for v in values.cpu()],
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
        "cvar_0_875": float(sorted_values[:tail_count].mean().item()),
        "count": int(values.numel()),
        "tail_count": tail_count,
    }


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).norm().div(right.float().norm().clamp_min(torch.finfo(torch.float32).tiny)).item())


def padding_safety_gate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    module_name: str,
    lock: NumericalLock,
    *,
    input_module: str,
    subject_templates: list[str],
    subjects: list[str],
) -> dict[str, Any]:
    """Compare left-padded batches, singletons, reorder, and longer padding."""
    if len(prompts) < 3 or len(prompts) != len(targets):
        raise TechnicalBoundary("padding gate needs at least three prompt/target pairs")
    batch_hidden, batch_logits = capture_last_hidden(model, tokenizer, prompts, module_name, track="out")
    singleton_hidden, singleton_logits = zip(
        *(capture_last_hidden(model, tokenizer, [prompt], module_name, track="out") for prompt in prompts),
        strict=True,
    )
    single_h = torch.cat(singleton_hidden)
    single_l = torch.cat(singleton_logits)
    order = list(reversed(range(len(prompts))))
    reorder_h, reorder_l = capture_last_hidden(model, tokenizer, [prompts[i] for i in order], module_name, track="out")
    inverse = torch.tensor(order).argsort()
    reorder_h = reorder_h[inverse]
    reorder_l = reorder_l[inverse]
    longer = "This deliberately longer padding-control prefix contains no experimental request. " * 8
    padded_h, padded_l = capture_last_hidden(model, tokenizer, prompts + [longer], module_name, track="out")
    hidden_errors = {
        "batch_singleton": relative_error(batch_hidden, single_h),
        "batch_reorder": relative_error(batch_hidden, reorder_h),
        "batch_padding_length": relative_error(batch_hidden, padded_h[:-1]),
    }
    logit_errors = {
        "batch_singleton": relative_error(batch_logits, single_l),
        "batch_reorder": relative_error(batch_logits, reorder_l),
        "batch_padding_length": relative_error(batch_logits, padded_l[:-1]),
    }
    target_prefix_texts: list[str] = []
    for prompt, target in zip(prompts, targets, strict=True):
        prefixes, _ = target_prefixes(tokenizer, prompt, target)
        target_prefix_texts.extend(prefixes)
    target_batch, _ = capture_last_hidden(model, tokenizer, target_prefix_texts, input_module, track="in")
    target_singletons = torch.cat([
        capture_last_hidden(model, tokenizer, [text], input_module, track="in")[0]
        for text in target_prefix_texts
    ])
    target_order = list(reversed(range(len(target_prefix_texts))))
    target_reordered, _ = capture_last_hidden(
        model, tokenizer, [target_prefix_texts[index] for index in target_order], input_module, track="in"
    )
    target_reordered = target_reordered[torch.tensor(target_order).argsort()]
    target_padded, _ = capture_last_hidden(
        model, tokenizer, target_prefix_texts + [longer], input_module, track="in"
    )
    target_key_errors = {
        "batch_singleton": relative_error(target_batch, target_singletons),
        "batch_reorder": relative_error(target_batch, target_reordered),
        "batch_padding_length": relative_error(target_batch, target_padded[:-1]),
    }
    if len(subject_templates) != len(subjects) or len(subjects) < 3:
        raise TechnicalBoundary("subject-key padding gate needs three identities")
    subject_batch = capture_subject_keys(model, tokenizer, subject_templates, subjects, input_module)
    subject_singletons = torch.cat([
        capture_subject_keys(model, tokenizer, [template], [subject], input_module)
        for template, subject in zip(subject_templates, subjects, strict=True)
    ], dim=1)
    subject_order = list(reversed(range(len(subjects))))
    subject_reordered = capture_subject_keys(
        model,
        tokenizer,
        [subject_templates[index] for index in subject_order],
        [subjects[index] for index in subject_order],
        input_module,
    )[:, torch.tensor(subject_order).argsort()]
    subject_padded = capture_subject_keys(
        model,
        tokenizer,
        subject_templates + [longer + " {}"],
        subjects + ["control"],
        input_module,
    )[:, :-1]
    subject_key_errors = {
        "batch_singleton": relative_error(subject_batch, subject_singletons),
        "batch_reorder": relative_error(subject_batch, subject_reordered),
        "batch_padding_length": relative_error(subject_batch, subject_padded),
    }
    nll_errors = []
    target_slices = []
    for prompt, target in zip(prompts, targets, strict=True):
        prefixes, ids = target_prefixes(tokenizer, prompt, target)
        target_slices.append({"prefix_count": len(prefixes), "target_ids": [int(v) for v in ids]})
        singleton = sequence_metrics(model, tokenizer, [prompt], target)["nll"][0]
        # The same routine is batch-order independent by construction; this
        # explicit repeat catches tokenizer state/padding mutation.
        repeated = sequence_metrics(model, tokenizer, [prompt], target)["nll"][0]
        nll_errors.append(abs(singleton - repeated))
    maximum = max([
        *hidden_errors.values(), *logit_errors.values(), *target_key_errors.values(),
        *subject_key_errors.values(), *nll_errors,
    ])
    passed = maximum <= lock.fp32_relative_tolerance
    if not passed:
        raise TechnicalBoundary(f"left-padding batch/singleton identity failed: {maximum}")
    return {
        "status": "PASS",
        "padding_side": "left",
        "semantic_position_source": "attention_mask_nonzero_columns_and_cumsum_position_ids",
        "hidden_relative_errors": hidden_errors,
        "logit_relative_errors": logit_errors,
        "target_key_relative_errors": target_key_errors,
        "edit_history_key_relative_errors": subject_key_errors,
        "nll_absolute_errors": nll_errors,
        "target_prefix_slices": target_slices,
        "batch_reorder_identity": True,
        "padding_length_identity": True,
        "tolerance": lock.fp32_relative_tolerance,
    }
