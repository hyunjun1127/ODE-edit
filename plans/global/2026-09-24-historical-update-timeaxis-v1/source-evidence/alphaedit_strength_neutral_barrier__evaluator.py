"""Observation-only CounterFact evaluator for barrier experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch

from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import (
    prompt_token_ids,
    target_token_ids,
)


@dataclass(frozen=True)
class PromptTarget:
    case_id: int
    kind: str
    prompt_index: int
    prompt: str
    target: str


def counterfact_pairs(records: Sequence[Dict[str, Any]]) -> Dict[str, List[PromptTarget]]:
    pairs: Dict[str, List[PromptTarget]] = {
        "rewrite_target_new": [],
        "rewrite_target_true": [],
        "rephrase_target_new": [],
        "rephrase_target_true": [],
        "locality_target_true": [],
    }
    for record in records:
        rewrite = record["requested_rewrite"]
        case_id = int(record["case_id"])
        prompt = rewrite["prompt"].format(rewrite["subject"])
        for suffix, target in (
            ("target_new", rewrite["target_new"]["str"]),
            ("target_true", rewrite["target_true"]["str"]),
        ):
            pairs[f"rewrite_{suffix}"].append(
                PromptTarget(case_id, f"rewrite_{suffix}", 0, prompt, target)
            )
        for prompt_index, rephrase in enumerate(record["paraphrase_prompts"]):
            for suffix, target in (
                ("target_new", rewrite["target_new"]["str"]),
                ("target_true", rewrite["target_true"]["str"]),
            ):
                pairs[f"rephrase_{suffix}"].append(
                    PromptTarget(
                        case_id,
                        f"rephrase_{suffix}",
                        prompt_index,
                        rephrase,
                        target,
                    )
                )
        for prompt_index, locality in enumerate(record["neighborhood_prompts"]):
            pairs["locality_target_true"].append(
                PromptTarget(
                    case_id,
                    "locality_target_true",
                    prompt_index,
                    locality,
                    rewrite["target_true"]["str"],
                )
            )
    return pairs


def _encode_pair(tok: Any, pair: PromptTarget) -> tuple[List[int], List[int]]:
    return prompt_token_ids(tok, pair.prompt), target_token_ids(tok, pair.target)


def evaluate_pairs(
    model: Any,
    tok: Any,
    pairs: Sequence[PromptTarget],
    *,
    device: torch.device,
    microbatch_size: int = 16,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for offset in range(0, len(pairs), microbatch_size):
        group = pairs[offset : offset + microbatch_size]
        encoded = [(_encode_pair(tok, pair), pair) for pair in group]
        full_ids = [prompt_ids + target_ids for (prompt_ids, target_ids), _ in encoded]
        max_length = max(len(ids) - 1 for ids in full_ids)
        input_ids = torch.full(
            (len(group), max_length), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        target_positions: List[List[int]] = []
        for row_index, ((prompt_ids, target_ids), _) in enumerate(encoded):
            inputs = (prompt_ids + target_ids)[:-1]
            next_tokens = (prompt_ids + target_ids)[1:]
            start = max_length - len(inputs)
            input_ids[row_index, start:] = torch.tensor(inputs, device=device)
            attention_mask[row_index, start:] = 1
            target_start = start + len(prompt_ids) - 1
            labels[row_index, target_start:] = torch.tensor(target_ids, device=device)
            target_positions.append(list(range(target_start, max_length)))
            if next_tokens[-len(target_ids) :] != target_ids:
                raise RuntimeError("teacher-forced target alignment failed")
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
        for row_index, ((_, target_ids), pair) in enumerate(encoded):
            positions = target_positions[row_index]
            target_tensor = torch.tensor(target_ids, device=device)
            selected = log_probs[row_index, positions, :].gather(
                1, target_tensor[:, None]
            )
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
                    "token_predictions": [int(value) for value in predicted.tolist()],
                    "token_correct": [
                        bool(value)
                        for value in (predicted == target_tensor).detach().cpu().tolist()
                    ],
                    "all_tokens_correct": bool(torch.all(predicted == target_tensor).item()),
                }
            )
    return rows


def evaluate_counterfact(
    model: Any,
    tok: Any,
    records: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    microbatch_size: int = 16,
) -> Dict[str, List[Dict[str, Any]]]:
    pairs = counterfact_pairs(records)
    return {
        kind: evaluate_pairs(
            model, tok, values, device=device, microbatch_size=microbatch_size
        )
        for kind, values in pairs.items()
    }


def locality_preservation(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    if len(before) != len(after):
        raise RuntimeError("locality before/after denominator mismatch")
    numerator = 0
    denominator = 0
    for left, right in zip(before, after):
        identity_left = (left["case_id"], left["prompt_index"], left["prompt"])
        identity_right = (right["case_id"], right["prompt_index"], right["prompt"])
        if identity_left != identity_right:
            raise RuntimeError("locality row identity mismatch")
        left_tokens = left["token_predictions"]
        right_tokens = right["token_predictions"]
        if len(left_tokens) != len(right_tokens):
            raise RuntimeError("locality token denominator mismatch")
        numerator += sum(a == b for a, b in zip(left_tokens, right_tokens))
        denominator += len(left_tokens)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }
