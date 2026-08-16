"""Action-frozen arbitrary-B CounterFact evaluation for P1R23.

The scoring kernel is the pinned R13 evaluator.  Only its B10 receipt shell is
generalized: held-out rows are opened in sealed B10 chunks after action freeze,
then all cases are scored exactly once and aggregated uniformly by request.
No held-out value is available to target construction or routing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .analysis import PrimaryMetric
from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
from .p1_evaluator import (
    EndpointActionFreeze,
    PINNED_SOURCE_SHA256,
    CounterFactEvaluationCase,
    PrefixNLLPair,
    _evaluate_prefixes_pinned,
    load_counterfact_cases_after_freeze,
)
from .request_digest import ordered_request_digest_v1
from .scalable_batched_runtime import scalable_ordered_request_digest


@dataclass(frozen=True, slots=True)
class ScalablePrimaryMetricReceipt:
    metric: str
    case_count: int
    prompt_count: int
    correct_count: int
    aggregate: float
    per_case_correct: tuple[int, ...]
    per_case_required: tuple[int, ...]
    per_case_mean_target_new_nll: tuple[float, ...]
    per_case_mean_target_old_nll: tuple[float, ...]
    per_case_mean_margin: tuple[float, ...]
    bit_vector_sha256: str
    score_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "case_count": self.case_count,
            "prompt_count": self.prompt_count,
            "correct_count": self.correct_count,
            "aggregate": self.aggregate,
            "mean_target_new_nll": float(
                math.fsum(self.per_case_mean_target_new_nll) / self.case_count
            ),
            "mean_target_old_nll": float(
                math.fsum(self.per_case_mean_target_old_nll) / self.case_count
            ),
            "mean_margin": float(
                math.fsum(self.per_case_mean_margin) / self.case_count
            ),
            "per_case_correct": list(self.per_case_correct),
            "per_case_required": list(self.per_case_required),
            "per_case_mean_target_new_nll": list(
                self.per_case_mean_target_new_nll
            ),
            "per_case_mean_target_old_nll": list(
                self.per_case_mean_target_old_nll
            ),
            "per_case_mean_margin": list(self.per_case_mean_margin),
            "bit_vector_sha256": self.bit_vector_sha256,
            "score_sha256": self.score_sha256,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScalablePrimaryReceipt:
    request_count: int
    request_order_sha256: str
    evaluation_case_identity_sha256: str
    target_span_sha256: str
    endpoint_freeze_sha256: str
    chunk_freeze_sha256: tuple[str, ...]
    efficacy: ScalablePrimaryMetricReceipt
    generalization: ScalablePrimaryMetricReceipt
    locality: ScalablePrimaryMetricReceipt
    model_forward_count: int
    processed_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-scalable-primary-evaluator/v1",
            "request_count": self.request_count,
            "request_order_sha256": self.request_order_sha256,
            "evaluation_case_identity_sha256": (
                self.evaluation_case_identity_sha256
            ),
            "target_span_sha256": self.target_span_sha256,
            "endpoint_freeze_sha256": self.endpoint_freeze_sha256,
            "chunk_freeze_sha256": list(self.chunk_freeze_sha256),
            "evaluator_source_sha256": PINNED_SOURCE_SHA256[
                "experiments/py/eval_utils_counterfact.py"
            ],
            "aggregator_source_sha256": PINNED_SOURCE_SHA256[
                "experiments/summarize.py"
            ],
            "metrics": {
                "efficacy": self.efficacy.raw_free_payload(),
                "generalization": self.generalization.raw_free_payload(),
                "locality-preservation": self.locality.raw_free_payload(),
            },
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
            "controller_or_routing_influence_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def _metric_receipt(
    metric: PrimaryMetric,
    scores_by_case: Sequence[Sequence[PrefixNLLPair]],
) -> ScalablePrimaryMetricReceipt:
    case_count = len(scores_by_case)
    if case_count <= 0:
        raise ODEBFContractError("P1R23 primary score cases are empty")
    bits: list[tuple[int, ...]] = []
    new_means: list[float] = []
    old_means: list[float] = []
    margins: list[float] = []
    score_payload: list[list[tuple[float, float]]] = []
    for scores in scores_by_case:
        if not scores:
            raise ODEBFContractError("P1R23 primary score case is empty")
        pairs = tuple(
            (float(item.target_new_nll), float(item.target_true_nll))
            for item in scores
        )
        if any(not math.isfinite(value) for pair in pairs for value in pair):
            raise ODEBFContractError("P1R23 primary score is non-finite")
        local_bits = tuple(
            int(old < new)
            if metric is PrimaryMetric.LOCALITY
            else int(new < old)
            for new, old in pairs
        )
        mean_new = math.fsum(new for new, _old in pairs) / len(pairs)
        mean_old = math.fsum(old for _new, old in pairs) / len(pairs)
        bits.append(local_bits)
        new_means.append(mean_new)
        old_means.append(mean_old)
        margins.append(
            mean_new - mean_old
            if metric is PrimaryMetric.LOCALITY
            else mean_old - mean_new
        )
        score_payload.append(list(pairs))
    per_correct = tuple(sum(item) for item in bits)
    per_required = tuple(len(item) for item in bits)
    aggregate = float(
        np.mean(
            np.asarray(
                [np.mean(np.asarray(item, dtype=np.bool_)) for item in bits],
                dtype=np.float64,
            )
        )
    )
    payload = {
        "metric": metric.value,
        "case_count": case_count,
        "prompt_count": sum(per_required),
        "correct_count": sum(per_correct),
        "aggregate": aggregate,
        "per_case_correct": list(per_correct),
        "per_case_required": list(per_required),
        "per_case_mean_target_new_nll": new_means,
        "per_case_mean_target_old_nll": old_means,
        "per_case_mean_margin": margins,
        "bit_vector_sha256": canonical_hash([list(item) for item in bits]),
        "score_sha256": canonical_hash(score_payload),
    }
    return ScalablePrimaryMetricReceipt(
        metric.value,
        case_count,
        payload["prompt_count"],
        payload["correct_count"],
        aggregate,
        per_correct,
        per_required,
        tuple(new_means),
        tuple(old_means),
        tuple(margins),
        payload["bit_vector_sha256"],
        payload["score_sha256"],
        canonical_hash(payload),
    )


def load_scalable_cases_after_freeze(
    dataset_path: str | Path,
    canonical_requests: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    selected_snapshot_sha256: str,
    fixed_budget_slots_completed: int,
) -> tuple[tuple[CounterFactEvaluationCase, ...], dict[str, Any]]:
    requests = tuple(canonical_requests)
    if len(requests) not in (10, 100) or len(requests) % BATCH_SIZE:
        raise ODEBFContractError("P1R23 evaluator batch must be B10 or B100")
    order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    endpoint_payload = {
        "schema": "ode-edit-s05-p1r23-endpoint-action-freeze/v1",
        "arm": arm,
        "request_count": len(requests),
        "request_order_sha256": order,
        "selected_snapshot_sha256": selected_snapshot_sha256,
        "fixed_budget_slots_completed": fixed_budget_slots_completed,
        "action_frozen": True,
    }
    endpoint_sha = canonical_hash(endpoint_payload)
    chunks: list[CounterFactEvaluationCase] = []
    chunk_shas: list[str] = []
    for start in range(0, len(requests), BATCH_SIZE):
        chunk = requests[start : start + BATCH_SIZE]
        chunk_order = ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in chunk]
        )
        freeze = EndpointActionFreeze(
            arm=f"{arm}-chunk-{start // BATCH_SIZE}",
            sequential_batch=0,
            request_order_sha256=chunk_order,
            selected_snapshot_sha256=selected_snapshot_sha256,
            fixed_budget_slots_completed=fixed_budget_slots_completed,
        )
        chunks.extend(
            load_counterfact_cases_after_freeze(dataset_path, chunk, freeze)
        )
        chunk_shas.append(freeze.identity())
    if len(chunks) != len(requests) or [item.request_sha256 for item in chunks] != [
        str(item["request_sha256"]) for item in requests
    ]:
        raise ODEBFContractError("P1R23 held-out case order differs")
    endpoint_payload["chunk_freeze_sha256"] = chunk_shas
    endpoint_payload["identity_sha256"] = endpoint_sha
    return tuple(chunks), endpoint_payload


def evaluate_scalable_primary(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[CounterFactEvaluationCase],
    *,
    model_alias: str,
    freeze_payload: Mapping[str, Any],
) -> ScalablePrimaryReceipt:
    batch = tuple(cases)
    if len(batch) not in (10, 100) or len(
        {item.request_sha256 for item in batch}
    ) != len(batch):
        raise ODEBFContractError("P1R23 primary evaluator batch differs")
    order = scalable_ordered_request_digest(
        [item.request_sha256 for item in batch]
    )
    if (
        freeze_payload.get("request_order_sha256") != order
        or freeze_payload.get("action_frozen") is not True
    ):
        raise ODEBFContractError("P1R23 evaluator preceded action freeze")
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    efficacy: list[tuple[PrefixNLLPair, ...]] = []
    generalization: list[tuple[PrefixNLLPair, ...]] = []
    locality: list[tuple[PrefixNLLPair, ...]] = []
    spans: list[dict[str, Any]] = []
    processed = 0
    with _EvaluationStateGuard(model), torch.no_grad():
        for case in batch:
            prefixes = (
                (case.rewrite_prompt,)
                + case.paraphrase_prompts
                + case.neighborhood_prompts
            )
            pairs, tokens, span = _evaluate_prefixes_pinned(
                model,
                tokenizer,
                prefixes,
                case.target_new,
                case.target_true,
                llama=llama,
                device=device,
            )
            paraphrase_end = 1 + len(case.paraphrase_prompts)
            efficacy.append(pairs[:1])
            generalization.append(pairs[1:paraphrase_end])
            locality.append(pairs[paraphrase_end:])
            processed += tokens
            spans.append(
                {
                    "request_sha256": case.request_sha256,
                    "rewrite_count": 1,
                    "paraphrase_count": len(case.paraphrase_prompts),
                    "neighborhood_count": len(case.neighborhood_prompts),
                    **span,
                }
            )
    eff = _metric_receipt(PrimaryMetric.EFFICACY, efficacy)
    gen = _metric_receipt(PrimaryMetric.GENERALIZATION, generalization)
    loc = _metric_receipt(PrimaryMetric.LOCALITY, locality)
    payload = {
        "request_count": len(batch),
        "request_order_sha256": order,
        "evaluation_case_identity_sha256": canonical_hash(
            [item.raw_free_identity() for item in batch]
        ),
        "target_span_sha256": canonical_hash(spans),
        "endpoint_freeze_sha256": str(freeze_payload["identity_sha256"]),
        "chunk_freeze_sha256": list(freeze_payload["chunk_freeze_sha256"]),
        "efficacy_sha256": eff.identity_sha256,
        "generalization_sha256": gen.identity_sha256,
        "locality_sha256": loc.identity_sha256,
        "model_forward_count": len(batch),
        "processed_token_count": processed,
    }
    return ScalablePrimaryReceipt(
        len(batch),
        order,
        payload["evaluation_case_identity_sha256"],
        payload["target_span_sha256"],
        payload["endpoint_freeze_sha256"],
        tuple(payload["chunk_freeze_sha256"]),
        eff,
        gen,
        loc,
        len(batch),
        processed,
        canonical_hash(payload),
    )


def added_ninety_payload(receipt: ScalablePrimaryReceipt) -> dict[str, Any] | None:
    """Return the A1-mandated added-90 view without rerunning evaluation."""

    if receipt.request_count == 10:
        return None
    if receipt.request_count != 100:
        raise ODEBFContractError("P1R23 added-90 view requires B100")
    result: dict[str, Any] = {"request_count": 90}
    for label, metric in (
        ("efficacy", receipt.efficacy),
        ("generalization", receipt.generalization),
        ("locality-preservation", receipt.locality),
    ):
        correct = metric.per_case_correct[10:]
        required = metric.per_case_required[10:]
        new = metric.per_case_mean_target_new_nll[10:]
        old = metric.per_case_mean_target_old_nll[10:]
        margin = metric.per_case_mean_margin[10:]
        result[label] = {
            "correct_count": sum(correct),
            "prompt_count": sum(required),
            "aggregate": float(
                np.mean(
                    np.asarray(
                        [a / b for a, b in zip(correct, required, strict=True)],
                        dtype=np.float64,
                    )
                )
            ),
            "mean_target_new_nll": math.fsum(new) / 90,
            "mean_target_old_nll": math.fsum(old) / 90,
            "mean_margin": math.fsum(margin) / 90,
        }
    result["identity_sha256"] = canonical_hash(result)
    return result


__all__ = [
    "ScalablePrimaryMetricReceipt",
    "ScalablePrimaryReceipt",
    "added_ninety_payload",
    "evaluate_scalable_primary",
    "load_scalable_cases_after_freeze",
]
