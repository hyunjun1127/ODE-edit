"""Endpoint-only PRE_EDIT-relative metrics for the atomic-B10 pilot."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch

from project.run_scripts.fixed_z_nonuniqueness.evaluation import next_token_logits, teacher_kl

from .evaluation import evaluate_rows
from .hashing import canonical_hash, tensor_sha256


@dataclass(slots=True)
class EvaluationReference:
    prompts: dict[str, list[str]]
    logits: dict[str, torch.Tensor]
    pre_edit: dict[str, Any]
    identity: str


def _prompt_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "rewrite": [
            row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"])
            for row in rows
        ],
        "rephrase": [prompt for row in rows for prompt in row.get("paraphrase_prompts", [])],
        "locality_unrelated": [prompt for row in rows for prompt in row.get("neighborhood_prompts", [])],
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def capture_pre_edit(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> EvaluationReference:
    prompts = _prompt_groups(rows)
    logits = {
        name: next_token_logits(model, tokenizer, values).detach().to("cpu", torch.float32)
        for name, values in prompts.items()
        if values
    }
    pre_edit = evaluate_rows(model, tokenizer, rows)
    identity = canonical_hash({
        "prompts": prompts,
        "logits": {name: tensor_sha256(value) for name, value in logits.items()},
        "request_count": len(rows),
    })
    return EvaluationReference(prompts=prompts, logits=logits, pre_edit=pre_edit, identity=identity)


def evaluate_endpoint(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    reference: EvaluationReference,
) -> dict[str, Any]:
    target = evaluate_rows(model, tokenizer, rows)
    output_kl: dict[str, Any] = {}
    entropy: dict[str, Any] = {}
    for name, prompts in reference.prompts.items():
        if not prompts:
            output_kl[name] = {"status": "NOT_RECORDED_EMPTY_PROMPT_SET", "count": 0}
            continue
        candidate = next_token_logits(model, tokenizer, prompts).float()
        teacher = reference.logits[name].to(candidate.device)
        output_kl[name] = teacher_kl(teacher, candidate, 0.875)
        probabilities = torch.softmax(candidate, dim=-1)
        values = -(probabilities * torch.log(probabilities.clamp_min(torch.finfo(torch.float32).tiny))).sum(-1)
        entropy[name] = {
            "count": int(values.numel()),
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "p90": _quantile([float(value) for value in values.cpu()], 0.9),
            "max": float(values.max().item()),
            "finite_fraction": float(torch.isfinite(candidate).float().mean().item()),
        }
    aggregate = target["aggregate"]
    rewrite_nll = [
        float(value)
        for row in target["records"]
        for value in row["rewrite_target_new"]["nll"]
    ]
    rephrase_strict = [
        bool(value)
        for row in target["records"]
        for value in row["rephrase_target_new"]["strict"]
    ]
    rewrite_strict = [
        bool(value)
        for row in target["records"]
        for value in row["rewrite_target_new"]["strict"]
    ]
    locality_strict = [
        bool(value)
        for row in target["records"]
        for value in row["locality_target_true"]["strict"]
    ]
    headline = {
        "efficacy_strict_numerator": sum(rewrite_strict),
        "efficacy_strict_denominator": len(rewrite_strict),
        "efficacy_strict_rate": sum(rewrite_strict) / len(rewrite_strict) if rewrite_strict else math.nan,
        "paraphrase_strict_numerator": sum(rephrase_strict),
        "paraphrase_strict_denominator": len(rephrase_strict),
        "paraphrase_strict_rate": sum(rephrase_strict) / len(rephrase_strict) if rephrase_strict else math.nan,
        "locality_strict_numerator": sum(locality_strict),
        "locality_strict_denominator": len(locality_strict),
        "locality_strict_rate": sum(locality_strict) / len(locality_strict) if locality_strict else math.nan,
        "target_probability_geometric_mean": statistics.fmean(math.exp(-value) for value in rewrite_nll) if rewrite_nll else math.nan,
        "worst_request_target_nll": max(rewrite_nll) if rewrite_nll else math.nan,
        "rewrite_target_new_nll": aggregate["rewrite_target_new"],
        "rephrase_target_new_nll": aggregate["rephrase_target_new"],
        "rewrite_target_true_nll": aggregate["rewrite_target_true"],
        "rephrase_target_true_nll": aggregate["rephrase_target_true"],
    }
    return {
        **target,
        "pre_edit_reference_identity": reference.identity,
        "output_kl_to_pre_edit": output_kl,
        "fluency_generation_degeneration_indicator": {
            "type": "NEXT_TOKEN_ENTROPY_AND_FINITE_LOGITS_ONLY",
            "generation_run_count": 0,
            "by_prompt_group": entropy,
        },
        "headline": headline,
        "controller_selection_influence_count": 0,
    }
