"""Native rewrite-context receipts, A* keys, and all-context realized-z."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from project.run_scripts.fixed_z_nonuniqueness.evaluation import (
    capture_last_hidden,
    capture_subject_keys,
    target_ids,
)
from project.run_scripts.fixed_z_nonuniqueness.padding import (
    build_position_safe_batch,
    semantic_token_columns,
)

from .contracts import Method, ScientificBoundary, TechnicalBoundary
from .hashing import canonical_hash, tensor_sha256


def _hidden(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TechnicalBoundary("unsupported hidden container")


@dataclass(frozen=True)
class NativeContext:
    ordinal: int
    context_template: str
    rewrite_prompt_template: str
    text: str
    lookup_semantic_index: int
    input_ids: list[int]
    attention_mask: list[int]
    target_ids: list[int]
    identity: str


def build_native_contexts(
    *, tokenizer: Any, request: dict[str, Any], context_templates: list[list[str]], fact_token: str
) -> list[NativeContext]:
    from easyeditor.models.alphaedit.compute_z import find_fact_lookup_idx

    ids = target_ids(tokenizer, request["target_new"])
    suffix = tokenizer.decode(ids[:-1])
    rows = []
    ordinal = 0
    for group in context_templates:
        for context in group:
            prompt_template = context.format(request["prompt"])
            rewrite_template = prompt_template + suffix
            text = rewrite_template.format(request["subject"])
            lookup = int(find_fact_lookup_idx(rewrite_template, request["subject"], tokenizer, fact_token, verbose=False))
            encoded = tokenizer(text, add_special_tokens=True)
            payload = {
                "ordinal": ordinal,
                "context_template": context,
                "rewrite_prompt_template": rewrite_template,
                "text": text,
                "lookup_semantic_index": lookup,
                "input_ids": [int(v) for v in encoded["input_ids"]],
                "attention_mask": [int(v) for v in encoded["attention_mask"]],
                "target_ids": [int(v) for v in ids],
            }
            rows.append(NativeContext(**payload, identity=canonical_hash(payload)))
            ordinal += 1
    if not rows:
        raise ScientificBoundary("empty native rewrite context bank")
    return rows


def capture_at_semantic_indices(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    semantic_indices: list[int],
    module_name: str,
    *,
    track: str,
) -> torch.Tensor:
    from easyeditor.util import nethook

    batch = build_position_safe_batch(tokenizer, texts, padding_side="left").to(next(model.parameters()).device)
    columns = semantic_token_columns(batch["attention_mask"])
    raw = []
    for index, valid in zip(semantic_indices, columns, strict=True):
        adjusted = index + valid.numel() if index < 0 else index
        if adjusted < 0 or adjusted >= valid.numel():
            raise TechnicalBoundary("native semantic lookup outside real tokens")
        raw.append(valid[adjusted])
    with torch.no_grad(), nethook.Trace(
        model, module_name, retain_input=track == "in", retain_output=track == "out"
    ) as trace:
        model(**batch)
    hidden = _hidden(trace.input if track == "in" else trace.output)
    if hidden.shape[0] != len(texts):
        hidden = hidden.transpose(0, 1)
    rows = torch.arange(len(texts), device=hidden.device)
    return hidden[rows, torch.stack(raw).to(hidden.device)].float().detach()


def native_edit_keys(model: Any, tokenizer: Any, contexts: list[NativeContext], input_module: str) -> torch.Tensor:
    values = capture_at_semantic_indices(
        model, tokenizer, [row.text for row in contexts],
        [row.lookup_semantic_index for row in contexts], input_module, track="in"
    )
    return values.T.contiguous()


def all_target_prefix_keys(
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    contexts: list[NativeContext],
    input_module: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    ids = target_ids(tokenizer, request["target_new"])
    bases = [request["prompt"].format(request["subject"])]
    bases.extend(row.context_template.format(request["prompt"]).format(request["subject"]) for row in contexts)
    texts: list[str] = []
    receipts = []
    for context_ordinal, base in enumerate(bases):
        for token_ordinal in range(ids.numel()):
            text = base + tokenizer.decode(ids[:token_ordinal])
            texts.append(text)
            receipts.append({
                "context_ordinal": context_ordinal - 1,
                "canonical": context_ordinal == 0,
                "target_token_ordinal": token_ordinal,
                "target_token_id": int(ids[token_ordinal]),
                "prefix_sha256": canonical_hash(text),
            })
    keys, _ = capture_last_hidden(model, tokenizer, texts, input_module, track="in")
    for row, key in zip(receipts, keys, strict=True):
        row["key_sha256"] = tensor_sha256(key)
    return keys.T.contiguous(), receipts


def build_constraint_bank(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    contexts: list[NativeContext],
    history_templates: list[str],
    history_subjects: list[str],
    input_module: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    edit = native_edit_keys(model, tokenizer, contexts, input_module)
    history = capture_subject_keys(model, tokenizer, history_templates, history_subjects, input_module)
    target, target_receipts = all_target_prefix_keys(model, tokenizer, request, contexts, input_module)
    bank = torch.cat([edit, history, target], dim=1)
    return bank, {
        "K_E_all_count": int(edit.shape[1]),
        "K_H_count": int(history.shape[1]),
        "K_T_all_count": int(target.shape[1]),
        "A_star_sha256": tensor_sha256(bank),
        "native_contexts": [row.__dict__ for row in contexts],
        "target_prefix_receipts": target_receipts,
    }


def realized_z_by_context(
    model: Any,
    tokenizer: Any,
    contexts: list[NativeContext],
    activation_module: str,
    z: torch.Tensor,
) -> dict[str, Any]:
    values = capture_at_semantic_indices(
        model, tokenizer, [row.text for row in contexts],
        [row.lookup_semantic_index for row in contexts], activation_module, track="out"
    )
    target = z.to(values.device, values.dtype).reshape(1, -1)
    residuals = torch.linalg.vector_norm(values - target, dim=1)
    return {
        "activation_sha256": tensor_sha256(values),
        "context_residual": [float(v) for v in residuals.cpu()],
        "context_count": len(contexts),
        "finite": bool(torch.isfinite(values).all()),
        "values": values,
    }


def compare_realized_z(candidate: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    left = candidate["values"]
    right = official["values"]
    if left.shape != right.shape:
        raise ScientificBoundary("realized-z context shape mismatch")
    degradation = torch.linalg.vector_norm(left - right, dim=1)
    denominator = torch.linalg.vector_norm(right, dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    relative = degradation / denominator
    return {
        "candidate_absolute_residual": candidate["context_residual"],
        "official_absolute_residual": official["context_residual"],
        "candidate_official_degradation": [float(v) for v in degradation.cpu()],
        "candidate_official_relative": [float(v) for v in relative.cpu()],
        "maximum_relative": float(relative.max().item()),
        "context_count": int(relative.numel()),
    }


def public_realized_z(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "values"}
