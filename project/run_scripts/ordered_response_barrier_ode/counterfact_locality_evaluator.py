"""Reusable observation-only CounterFact continuation evaluator.

The runtime deliberately keeps this module independent of EasyEdit imports so
an evaluation-only W0 backfill cannot initialize writer/controller state.
Prompts are manually left padded within each microbatch while the shared
tokenizer remains in its stock writer-compatible ``padding_side='right'``
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


class LocalityEvaluationBoundary(RuntimeError):
    """Fail-closed tokenizer or evaluation boundary."""


@dataclass(frozen=True, slots=True)
class LocalityPromptTarget:
    case_id: int
    kind: str
    prompt_index: int
    prompt: str
    target: str


def counterfact_locality_target_new_pairs(
    records: Sequence[Mapping[str, Any]],
) -> list[Any]:
    """Build target-new neighborhood rows in canonical request/prompt order.

    The returned values use the stock observation evaluator's ``PromptTarget``
    type so endpoint runs retain exactly the same prompt tokenization and raw
    row schema as rewrite, rephrase, and locality-target-true observations.
    """

    from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import (
        PromptTarget,
    )

    pairs: list[Any] = []
    for record in records:
        rewrite = record.get("requested_rewrite")
        prompts = record.get("neighborhood_prompts")
        if not isinstance(rewrite, Mapping) or not isinstance(prompts, list):
            raise LocalityEvaluationBoundary("CounterFact locality input schema differs")
        target_new = rewrite.get("target_new")
        if not isinstance(target_new, Mapping) or not isinstance(target_new.get("str"), str):
            raise LocalityEvaluationBoundary("CounterFact target-new schema differs")
        case_id = int(record["case_id"])
        for prompt_index, prompt in enumerate(prompts):
            pairs.append(
                PromptTarget(
                    case_id=case_id,
                    kind="locality_target_new",
                    prompt_index=prompt_index,
                    prompt=str(prompt),
                    target=str(target_new["str"]),
                )
            )
    if not pairs:
        raise LocalityEvaluationBoundary("CounterFact locality target-new inventory is empty")
    return pairs


def evaluate_counterfact_with_canonical_ns(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    microbatch_size: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate the stock endpoint panel plus canonical NS target-new rows."""

    from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import (
        evaluate_counterfact,
        evaluate_pairs,
    )

    if tokenizer.padding_side != "right":
        raise LocalityEvaluationBoundary("shared tokenizer padding must remain right")
    result = dict(
        evaluate_counterfact(
            model,
            tokenizer,
            records,
            device=device,
            microbatch_size=microbatch_size,
        )
    )
    if "locality_target_new" in result:
        raise LocalityEvaluationBoundary("duplicate locality target-new evaluator kind")
    result["locality_target_new"] = evaluate_pairs(
        model,
        tokenizer,
        counterfact_locality_target_new_pairs(records),
        device=device,
        microbatch_size=microbatch_size,
    )
    return result


def _target_token_ids(tokenizer: Any, target: str) -> list[int]:
    text = str(target)
    if not text:
        raise LocalityEvaluationBoundary("target text is empty")
    normalized = text if text.startswith(" ") else " " + text
    values = list(tokenizer.encode(normalized, add_special_tokens=False))
    while values and values[0] in {tokenizer.bos_token_id, tokenizer.unk_token_id}:
        values = values[1:]
    if not values:
        raise LocalityEvaluationBoundary("target tokenization is empty")
    return [int(value) for value in values]


def _prompt_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    values = list(tokenizer(str(prompt), add_special_tokens=True)["input_ids"])
    if not values:
        raise LocalityEvaluationBoundary("prompt tokenization is empty")
    return [int(value) for value in values]


def evaluate_prompt_targets(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[LocalityPromptTarget],
    *,
    device: torch.device,
    microbatch_size: int = 16,
) -> list[dict[str, Any]]:
    """Return teacher-forced NLL/strict rows without mutating model/tokenizer."""

    if tokenizer.padding_side != "right":
        raise LocalityEvaluationBoundary("shared tokenizer padding must remain right")
    if microbatch_size <= 0 or not pairs:
        raise LocalityEvaluationBoundary("evaluation batch contract differs")
    rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise LocalityEvaluationBoundary("tokenizer has no pad/eos token")
    for offset in range(0, len(pairs), microbatch_size):
        group = pairs[offset : offset + microbatch_size]
        encoded = [
            ((_prompt_token_ids(tokenizer, pair.prompt), _target_token_ids(tokenizer, pair.target)), pair)
            for pair in group
        ]
        full_ids = [prompt_ids + target_ids for (prompt_ids, target_ids), _pair in encoded]
        max_length = max(len(values) - 1 for values in full_ids)
        input_ids = torch.full((len(group), max_length), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        target_positions: list[list[int]] = []
        for row_index, ((prompt_ids, target_ids), _pair) in enumerate(encoded):
            inputs = (prompt_ids + target_ids)[:-1]
            next_tokens = (prompt_ids + target_ids)[1:]
            start = max_length - len(inputs)
            input_ids[row_index, start:] = torch.tensor(inputs, dtype=torch.long, device=device)
            attention_mask[row_index, start:] = 1
            target_start = start + len(prompt_ids) - 1
            target_positions.append(list(range(target_start, max_length)))
            if next_tokens[-len(target_ids) :] != target_ids:
                raise LocalityEvaluationBoundary("teacher-forced target alignment failed")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
        for row_index, ((unused_prompt_ids, target_ids), pair) in enumerate(encoded):
            del unused_prompt_ids
            positions = target_positions[row_index]
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
            selected = log_probs[row_index, positions, :].gather(1, target_tensor[:, None])
            predicted = predictions[row_index, positions]
            rows.append(
                {
                    "case_id": pair.case_id,
                    "kind": pair.kind,
                    "prompt_index": pair.prompt_index,
                    "prompt": pair.prompt,
                    "target": pair.target,
                    "target_token_ids": target_ids,
                    "nll": float(-selected.mean().item()),
                    "all_tokens_correct": bool(torch.all(predicted == target_tensor).item()),
                }
            )
        del logits, log_probs, predictions, input_ids, attention_mask
    return rows


__all__ = [
    "LocalityEvaluationBoundary",
    "LocalityPromptTarget",
    "counterfact_locality_target_new_pairs",
    "evaluate_counterfact_with_canonical_ns",
    "evaluate_prompt_targets",
]
