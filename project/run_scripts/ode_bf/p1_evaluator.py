"""Post-action CounterFact primary evaluator and matched-Native floors.

Held-out paraphrase and neighborhood strings are decoded only by
``load_counterfact_cases_after_freeze``.  They never enter a controller
request, receipt, log, or digest payload as plaintext.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from project.run_scripts.ode_alloc.selection import _iter_top_level_objects

from .analysis import (
    CanonicalMetricReceipt,
    NativeFloorLock,
    PrimaryMetric,
    PrimaryNativeFloorTable,
    build_native_floor_table,
)
from .benchmark import PINNED_SOURCE_SHA256
from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash, finite
from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
from .p1_selection import _case_id_from_row, project_p1_request_identity
from .request_digest import ordered_request_digest_v1


PRIMARY_SOURCE_SHA256 = canonical_hash(
    {
        "evaluator": PINNED_SOURCE_SHA256["experiments/py/eval_utils_counterfact.py"],
        "aggregator": PINNED_SOURCE_SHA256["experiments/summarize.py"],
    }
)


@dataclass(frozen=True, slots=True)
class EndpointActionFreeze:
    arm: str
    sequential_batch: int
    request_order_sha256: str
    selected_snapshot_sha256: str
    fixed_budget_slots_completed: int
    action_frozen: bool = True

    def __post_init__(self) -> None:
        if not self.arm or self.sequential_batch < 0 or self.sequential_batch >= 10:
            raise ODEBFContractError("P1 endpoint action-freeze identity differs")
        for value in (self.request_order_sha256, self.selected_snapshot_sha256):
            if len(value) != 64:
                raise ODEBFContractError("P1 endpoint action-freeze digest differs")
        if self.fixed_budget_slots_completed not in (0, 8):
            raise ODEBFContractError("P1 endpoint freeze does not follow Native/K8 policy")
        if not self.action_frozen:
            raise ODEBFContractError("P1 held-out evaluator was not action frozen")

    def identity(self) -> str:
        return canonical_hash(
            {
                "arm": self.arm,
                "sequential_batch": self.sequential_batch,
                "request_order_sha256": self.request_order_sha256,
                "selected_snapshot_sha256": self.selected_snapshot_sha256,
                "fixed_budget_slots_completed": self.fixed_budget_slots_completed,
                "action_frozen": self.action_frozen,
            }
        )


@dataclass(frozen=True, slots=True)
class BatchEntryObservationSeal:
    """Seal a sequential batch-entry held-out observation away from actions.

    The evaluator may decode held-out surfaces only after the physical entry
    bytes and the action-independent request order have been sealed.  The
    resulting metrics are never returned to the controller, router, history,
    or anchor paths.
    """

    arm: str
    sequential_batch: int
    request_order_sha256: str
    selected_snapshot_sha256: str
    parameter_byte_sha256: str
    fixed_budget_slots_completed: int = 0
    action_frozen: bool = True
    observation_only: bool = True
    controller_influence_count: int = 0
    routing_influence_count: int = 0
    history_influence_count: int = 0
    anchor_influence_count: int = 0

    def __post_init__(self) -> None:
        if not self.arm or self.sequential_batch < 0 or self.sequential_batch >= 10:
            raise ODEBFContractError("P1 batch-entry observation identity differs")
        for value in (
            self.request_order_sha256,
            self.selected_snapshot_sha256,
            self.parameter_byte_sha256,
        ):
            if len(value) != 64:
                raise ODEBFContractError("P1 batch-entry observation digest differs")
        if self.fixed_budget_slots_completed != 0:
            raise ODEBFContractError("P1 batch-entry observation followed a current-B10 action")
        if not self.action_frozen or not self.observation_only:
            raise ODEBFContractError("P1 batch-entry observation is not sealed")
        if any(
            (
                self.controller_influence_count,
                self.routing_influence_count,
                self.history_influence_count,
                self.anchor_influence_count,
            )
        ):
            raise ODEBFContractError("P1 batch-entry observation influenced an action")

    def identity(self) -> str:
        return canonical_hash(
            {
                "arm": self.arm,
                "sequential_batch": self.sequential_batch,
                "request_order_sha256": self.request_order_sha256,
                "selected_snapshot_sha256": self.selected_snapshot_sha256,
                "parameter_byte_sha256": self.parameter_byte_sha256,
                "fixed_budget_slots_completed": self.fixed_budget_slots_completed,
                "action_frozen": self.action_frozen,
                "observation_only": self.observation_only,
                "controller_influence_count": self.controller_influence_count,
                "routing_influence_count": self.routing_influence_count,
                "history_influence_count": self.history_influence_count,
                "anchor_influence_count": self.anchor_influence_count,
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class CounterFactEvaluationCase:
    case_id: int
    request_sha256: str
    rewrite_prompt: str
    paraphrase_prompts: tuple[str, ...]
    neighborhood_prompts: tuple[str, ...]
    target_new: str
    target_true: str

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 0:
            raise ODEBFContractError("CounterFact evaluation case identity differs")
        if len(self.request_sha256) != 64:
            raise ODEBFContractError("CounterFact evaluation request digest differs")
        if not self.rewrite_prompt or not self.target_new or not self.target_true:
            raise ODEBFContractError("CounterFact evaluation rewrite surface is empty")
        if not self.paraphrase_prompts or not self.neighborhood_prompts:
            raise ODEBFContractError("CounterFact held-out primary prompt set is empty")
        if any(not isinstance(item, str) or not item for item in self.paraphrase_prompts):
            raise ODEBFContractError("CounterFact paraphrase prompt differs")
        if any(not isinstance(item, str) or not item for item in self.neighborhood_prompts):
            raise ODEBFContractError("CounterFact neighborhood prompt differs")

    def raw_free_identity(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "request_sha256": self.request_sha256,
            "rewrite_prompt_sha256": hashlib.sha256(self.rewrite_prompt.encode("utf-8")).hexdigest(),
            "paraphrase_prompt_count": len(self.paraphrase_prompts),
            "paraphrase_order_sha256": canonical_hash(
                [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in self.paraphrase_prompts]
            ),
            "neighborhood_prompt_count": len(self.neighborhood_prompts),
            "neighborhood_order_sha256": canonical_hash(
                [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in self.neighborhood_prompts]
            ),
            "target_new_sha256": hashlib.sha256(self.target_new.encode("utf-8")).hexdigest(),
            "target_true_sha256": hashlib.sha256(self.target_true.encode("utf-8")).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class PrefixNLLPair:
    target_new_nll: float
    target_true_nll: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_new_nll", finite("target_new_nll", self.target_new_nll))
        object.__setattr__(self, "target_true_nll", finite("target_true_nll", self.target_true_nll))


@dataclass(frozen=True, slots=True)
class PrefixSuccessAccuracy:
    """One-pass CounterFact prompt score.

    ``success`` is the pinned CounterFact NLL-preference metric.  ``accuracy``
    is the strict teacher-forced target-new suffix-token argmax metric used by
    the pinned AlphaEdit ``*_prompts_correct`` fields.  Both are derived from
    the same logits.
    """

    nll: PrefixNLLPair
    success: int
    accuracy: int
    target_new_suffix_token_count: int

    def __post_init__(self) -> None:
        if self.success not in (0, 1) or self.accuracy not in (0, 1):
            raise ODEBFContractError("CounterFact success/accuracy bit differs")
        if self.target_new_suffix_token_count <= 0:
            raise ODEBFContractError("CounterFact accuracy target span is empty")


@dataclass(frozen=True, slots=True)
class PromptMetricReceipt:
    name: str
    per_request_bits: tuple[tuple[int, ...], ...]
    per_request_correct: tuple[int, ...]
    per_request_required: tuple[int, ...]
    prompt_numerator: int
    prompt_denominator: int
    prompt_rate: float
    prompt_bit_vector_sha256: str
    strict_all_prompt_bits: tuple[int, ...]
    strict_request_numerator: int
    strict_request_denominator: int
    strict_request_rate: float
    strict_request_bit_vector_sha256: str
    target_new_nll_by_request: tuple[tuple[float, ...], ...]
    target_true_nll_by_request: tuple[tuple[float, ...], ...]
    target_new_minus_true_margin_by_request: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.per_request_bits) != BATCH_SIZE:
            raise ODEBFContractError("CounterFact prompt metric lacks B10")
        if len(self.per_request_correct) != BATCH_SIZE or len(self.per_request_required) != BATCH_SIZE:
            raise ODEBFContractError("CounterFact prompt metric request counts differ")
        for bits, correct, required in zip(
            self.per_request_bits,
            self.per_request_correct,
            self.per_request_required,
            strict=True,
        ):
            if not bits or any(bit not in (0, 1) for bit in bits):
                raise ODEBFContractError("CounterFact prompt metric bits differ")
            if len(bits) != required or sum(bits) != correct:
                raise ODEBFContractError("CounterFact prompt metric arithmetic differs")
        if self.prompt_numerator != sum(self.per_request_correct) or self.prompt_denominator != sum(self.per_request_required):
            raise ODEBFContractError("CounterFact prompt metric aggregate differs")
        if self.strict_all_prompt_bits != tuple(int(all(bits)) for bits in self.per_request_bits):
            raise ODEBFContractError("CounterFact strict request bits differ")
        if self.strict_request_numerator != sum(self.strict_all_prompt_bits) or self.strict_request_denominator != BATCH_SIZE:
            raise ODEBFContractError("CounterFact strict request aggregate differs")
        if self.prompt_rate != self.prompt_numerator / self.prompt_denominator:
            raise ODEBFContractError("CounterFact prompt rate differs")
        if self.strict_request_rate != self.strict_request_numerator / self.strict_request_denominator:
            raise ODEBFContractError("CounterFact strict request rate differs")
        if len(self.prompt_bit_vector_sha256) != 64 or len(self.strict_request_bit_vector_sha256) != 64:
            raise ODEBFContractError("CounterFact prompt metric digest differs")
        if not (
            len(self.target_new_nll_by_request)
            == len(self.target_true_nll_by_request)
            == len(self.target_new_minus_true_margin_by_request)
            == BATCH_SIZE
        ):
            raise ODEBFContractError("CounterFact prompt NLL request count differs")
        for bits, new, true, margin in zip(
            self.per_request_bits,
            self.target_new_nll_by_request,
            self.target_true_nll_by_request,
            self.target_new_minus_true_margin_by_request,
            strict=True,
        ):
            if not (len(bits) == len(new) == len(true) == len(margin)):
                raise ODEBFContractError("CounterFact prompt NLL span differs")
            if any(not np.isfinite(value) for value in (*new, *true, *margin)):
                raise ODEBFContractError("CounterFact prompt NLL is non-finite")
            if any(observed != left - right for observed, left, right in zip(margin, new, true, strict=True)):
                raise ODEBFContractError("CounterFact prompt margin arithmetic differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "per_request_bits": [list(bits) for bits in self.per_request_bits],
            "per_request_correct": list(self.per_request_correct),
            "per_request_required": list(self.per_request_required),
            "prompt_numerator": self.prompt_numerator,
            "prompt_denominator": self.prompt_denominator,
            "prompt_rate": self.prompt_rate,
            "prompt_bit_vector_sha256": self.prompt_bit_vector_sha256,
            "strict_all_prompt_bits": list(self.strict_all_prompt_bits),
            "strict_request_numerator": self.strict_request_numerator,
            "strict_request_denominator": self.strict_request_denominator,
            "strict_request_rate": self.strict_request_rate,
            "strict_request_bit_vector_sha256": self.strict_request_bit_vector_sha256,
            "target_new_nll_by_request": [list(row) for row in self.target_new_nll_by_request],
            "target_true_nll_by_request": [list(row) for row in self.target_true_nll_by_request],
            "target_new_minus_true_margin_by_request": [list(row) for row in self.target_new_minus_true_margin_by_request],
        }


@dataclass(frozen=True, slots=True)
class CounterFactSuccessAccuracyReceipt:
    primary: "CounterFactPrimaryReceipt"
    rewrite_success: PromptMetricReceipt
    rewrite_accuracy: PromptMetricReceipt
    paraphrase_success: PromptMetricReceipt
    paraphrase_accuracy: PromptMetricReceipt
    added_model_forward_count: int = 0
    added_backward_count: int = 0
    added_generation_call_count: int = 0
    controller_influence_count: int = 0

    def __post_init__(self) -> None:
        if self.rewrite_success.per_request_bits != self.primary.efficacy.per_case_bits:
            raise ODEBFContractError("Eff is not rewrite_success")
        if self.paraphrase_success.per_request_bits != self.primary.generalization.per_case_bits:
            raise ODEBFContractError("Gen is not paraphrase_success")
        if any((self.added_model_forward_count, self.added_backward_count, self.added_generation_call_count, self.controller_influence_count)):
            raise ODEBFContractError("CounterFact accuracy extension changed execution")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r52-sequential-counterfact-success-accuracy/v1",
            "legacy_primary": self.primary.raw_free_payload(),
            "canonical_semantics": {
                "Eff": "rewrite_success",
                "Gen": "paraphrase_success",
                "rewrite_success": "target_new_mean_nll_lt_target_true_mean_nll",
                "rewrite_acc": "all_target_new_suffix_tokens_teacher_forced_argmax",
                "paraphrase_success": "per_paraphrase_target_new_mean_nll_lt_target_true_mean_nll",
                "rephrase_success": "alias_of_paraphrase_success",
                "paraphrase_acc": "per_paraphrase_all_target_new_suffix_tokens_teacher_forced_argmax",
                "rephrase_acc": "alias_of_paraphrase_acc",
            },
            "rewrite_success": self.rewrite_success.raw_free_payload(),
            "rewrite_acc": self.rewrite_accuracy.raw_free_payload(),
            "paraphrase_success": self.paraphrase_success.raw_free_payload(),
            "rephrase_success": self.paraphrase_success.raw_free_payload(),
            "paraphrase_acc": self.paraphrase_accuracy.raw_free_payload(),
            "rephrase_acc": self.paraphrase_accuracy.raw_free_payload(),
            "added_model_forward_count": self.added_model_forward_count,
            "added_backward_count": self.added_backward_count,
            "added_generation_call_count": self.added_generation_call_count,
            "controller_influence_count": self.controller_influence_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class OfficialMetricBatchReceipt:
    metric: PrimaryMetric
    per_case_bits: tuple[tuple[int, ...], ...]
    per_case_correct: tuple[int, ...]
    per_case_required: tuple[int, ...]
    numerator: int
    denominator: int
    official_aggregate: float
    bit_vector_sha256: str
    comparison_score_sha256: str

    def __post_init__(self) -> None:
        if len(self.per_case_bits) != BATCH_SIZE:
            raise ODEBFContractError("official CounterFact metric lacks ten cases")
        if len(self.per_case_correct) != BATCH_SIZE or len(self.per_case_required) != BATCH_SIZE:
            raise ODEBFContractError("official CounterFact per-case counts differ")
        for bits, correct, required in zip(
            self.per_case_bits, self.per_case_correct, self.per_case_required
        ):
            if not bits or any(bit not in (0, 1) for bit in bits):
                raise ODEBFContractError("official CounterFact metric bits differ")
            if len(bits) != required or sum(bits) != correct:
                raise ODEBFContractError("official CounterFact metric count differs")
        if self.numerator != sum(self.per_case_correct) or self.denominator != sum(
            self.per_case_required
        ):
            raise ODEBFContractError("official CounterFact batch counts differ")
        expected = float(
            np.mean(
                np.asarray(
                    [np.mean(np.asarray(bits, dtype=np.bool_)) for bits in self.per_case_bits],
                    dtype=np.float64,
                )
            )
        )
        if self.official_aggregate != expected:
            raise ODEBFContractError("official CounterFact arithmetic differs")
        if len(self.bit_vector_sha256) != 64 or len(self.comparison_score_sha256) != 64:
            raise ODEBFContractError("official CounterFact metric digest differs")

    def canonical_metric(self) -> CanonicalMetricReceipt:
        return CanonicalMetricReceipt(
            self.metric,
            self.official_aggregate,
            PRIMARY_SOURCE_SHA256,
            self.numerator,
            self.denominator,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "per_case_bits": [list(bits) for bits in self.per_case_bits],
            "per_case_correct": list(self.per_case_correct),
            "per_case_required": list(self.per_case_required),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "official_aggregate": self.official_aggregate,
            "bit_vector_sha256": self.bit_vector_sha256,
            "comparison_score_sha256": self.comparison_score_sha256,
        }


@dataclass(frozen=True, slots=True)
class CounterFactPrimaryReceipt:
    efficacy: OfficialMetricBatchReceipt
    generalization: OfficialMetricBatchReceipt
    locality: OfficialMetricBatchReceipt
    request_order_sha256: str
    evaluation_case_identity_sha256: str
    target_span_sha256: str
    evaluator_source_sha256: str
    aggregator_source_sha256: str
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int
    endpoint_freeze_sha256: str
    boundary_touched: bool

    def __post_init__(self) -> None:
        if (self.efficacy.metric, self.generalization.metric, self.locality.metric) != tuple(
            PrimaryMetric
        ):
            raise ODEBFContractError("CounterFact primary metric order differs")
        if any(
            len(value) != 64
            for value in (
                self.request_order_sha256,
                self.evaluation_case_identity_sha256,
                self.target_span_sha256,
                self.evaluator_source_sha256,
                self.aggregator_source_sha256,
                self.endpoint_freeze_sha256,
            )
        ):
            raise ODEBFContractError("CounterFact primary receipt digest differs")
        if self.model_forward_count != BATCH_SIZE or self.processed_token_count <= 0:
            raise ODEBFContractError("CounterFact primary evaluator accounting differs")
        if self.generation_call_count != 0:
            raise ODEBFContractError("CounterFact P1 primary evaluator called generation")

    def metrics(self) -> dict[PrimaryMetric, OfficialMetricBatchReceipt]:
        return {
            PrimaryMetric.EFFICACY: self.efficacy,
            PrimaryMetric.GENERALIZATION: self.generalization,
            PrimaryMetric.LOCALITY: self.locality,
        }

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "metrics": {
                metric.value: receipt.raw_free_payload()
                for metric, receipt in self.metrics().items()
            },
            "request_order_sha256": self.request_order_sha256,
            "evaluation_case_identity_sha256": self.evaluation_case_identity_sha256,
            "target_span_sha256": self.target_span_sha256,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "aggregator_source_sha256": self.aggregator_source_sha256,
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
            "generation_call_count": self.generation_call_count,
            "endpoint_freeze_sha256": self.endpoint_freeze_sha256,
            "boundary_touched": self.boundary_touched,
        }


@dataclass(frozen=True, slots=True)
class PairedPrimaryMetricReceipt:
    metric: PrimaryMetric
    floor_passed: bool
    native_numerator: int
    ours_numerator: int
    denominator: int
    native_aggregate: float
    ours_aggregate: float
    win_bits_sha256: str
    loss_bits_sha256: str
    win_count: int
    loss_count: int
    pairing_sha256: str


@dataclass(frozen=True, slots=True)
class PairedPrimaryFloorReceipt:
    table: PrimaryNativeFloorTable
    paired: tuple[PairedPrimaryMetricReceipt, ...]
    request_span_parity: bool

    @property
    def all_primary_pass(self) -> bool:
        return self.request_span_parity and self.table.all_primary_pass

    @property
    def pairwise_miss_review(self) -> bool:
        return any(item.loss_count > 0 for item in self.paired)


def load_counterfact_cases_after_freeze(
    dataset_path: str | Path,
    canonical_requests: Sequence[Mapping[str, Any]],
    freeze: EndpointActionFreeze,
) -> tuple[CounterFactEvaluationCase, ...]:
    if len(canonical_requests) != BATCH_SIZE:
        raise ODEBFContractError("held-out CounterFact loader requires B10")
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in canonical_requests]
    )
    if request_order != freeze.request_order_sha256 or not freeze.action_frozen:
        raise ODEBFContractError("held-out CounterFact access preceded action freeze")
    approved = {
        int(item["case_id"]): str(item["request_sha256"])
        for item in canonical_requests
    }
    if len(approved) != BATCH_SIZE:
        raise ODEBFContractError("held-out CounterFact cases repeat")
    loaded: dict[int, CounterFactEvaluationCase] = {}
    for blob in _iter_top_level_objects(Path(dataset_path).resolve(strict=True)):
        case_id = _case_id_from_row(blob)
        if case_id not in approved:
            continue
        identity = project_p1_request_identity(blob)
        if identity.request_sha256 != approved[case_id]:
            raise ODEBFContractError("held-out CounterFact request identity differs")
        try:
            row = json.loads(blob)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEBFContractError("held-out CounterFact row is invalid") from exc
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise ODEBFContractError("held-out CounterFact rewrite schema differs")
        subject = str(rewrite["subject"])
        rewrite_prompt = str(rewrite["prompt"]).format(subject)
        paraphrase = tuple(row.get("paraphrase_prompts", ()))
        neighborhood = tuple(row.get("neighborhood_prompts", ()))
        # generation_prompts and attribute_prompts are deliberately not copied.
        loaded[case_id] = CounterFactEvaluationCase(
            case_id,
            identity.request_sha256,
            rewrite_prompt,
            paraphrase,
            neighborhood,
            str(rewrite["target_new"]["str"]),
            str(rewrite["target_true"]["str"]),
        )
    if set(loaded) != set(approved):
        raise ODEBFContractError("held-out CounterFact load is incomplete")
    return tuple(loaded[int(item["case_id"])] for item in canonical_requests)


def _metric_receipt(
    metric: PrimaryMetric,
    scores_by_case: Sequence[Sequence[PrefixNLLPair]],
) -> OfficialMetricBatchReceipt:
    if len(scores_by_case) != BATCH_SIZE:
        raise ODEBFContractError("official CounterFact score cases differ")
    bits_by_case: list[tuple[int, ...]] = []
    score_payload: list[list[tuple[float, float]]] = []
    for scores in scores_by_case:
        if not scores:
            raise ODEBFContractError("official CounterFact score case is empty")
        if metric is PrimaryMetric.LOCALITY:
            bits = tuple(int(score.target_true_nll < score.target_new_nll) for score in scores)
        else:
            bits = tuple(int(score.target_new_nll < score.target_true_nll) for score in scores)
        bits_by_case.append(bits)
        score_payload.append(
            [(score.target_new_nll, score.target_true_nll) for score in scores]
        )
    per_correct = tuple(sum(bits) for bits in bits_by_case)
    per_required = tuple(len(bits) for bits in bits_by_case)
    numerator = sum(per_correct)
    denominator = sum(per_required)
    aggregate = float(
        np.mean(
            np.asarray(
                [np.mean(np.asarray(bits, dtype=np.bool_)) for bits in bits_by_case],
                dtype=np.float64,
            )
        )
    )
    return OfficialMetricBatchReceipt(
        metric,
        tuple(bits_by_case),
        per_correct,
        per_required,
        numerator,
        denominator,
        aggregate,
        canonical_hash([list(bits) for bits in bits_by_case]),
        canonical_hash(score_payload),
    )


def counterfact_primary_receipt_from_scores(
    *,
    efficacy: Sequence[Sequence[PrefixNLLPair]],
    generalization: Sequence[Sequence[PrefixNLLPair]],
    locality: Sequence[Sequence[PrefixNLLPair]],
    request_order_sha256: str,
    evaluation_case_identity_sha256: str,
    target_span_sha256: str,
    model_forward_count: int,
    processed_token_count: int,
    endpoint_freeze_sha256: str,
) -> CounterFactPrimaryReceipt:
    flattened = [score for group in (efficacy, generalization, locality) for case in group for score in case]
    return CounterFactPrimaryReceipt(
        _metric_receipt(PrimaryMetric.EFFICACY, efficacy),
        _metric_receipt(PrimaryMetric.GENERALIZATION, generalization),
        _metric_receipt(PrimaryMetric.LOCALITY, locality),
        request_order_sha256,
        evaluation_case_identity_sha256,
        target_span_sha256,
        PINNED_SOURCE_SHA256["experiments/py/eval_utils_counterfact.py"],
        PINNED_SOURCE_SHA256["experiments/summarize.py"],
        model_forward_count,
        processed_token_count,
        0,
        endpoint_freeze_sha256,
        any(score.target_new_nll == score.target_true_nll for score in flattened),
    )


def _evaluate_prefixes_pinned(
    model: torch.nn.Module,
    tokenizer: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    *,
    llama: bool,
    device: torch.device,
) -> tuple[tuple[PrefixNLLPair, ...], int, dict[str, Any]]:
    extended, processed, receipt = _evaluate_prefixes_pinned_with_accuracy(
        model,
        tokenizer,
        prefixes,
        target_new,
        target_true,
        llama=llama,
        device=device,
    )
    return tuple(item.nll for item in extended), processed, receipt


def _evaluate_prefixes_pinned_with_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    *,
    llama: bool,
    device: torch.device,
) -> tuple[tuple[PrefixSuccessAccuracy, ...], int, dict[str, Any]]:
    """Pinned CounterFact NLL and strict suffix accuracy from one logits tensor."""

    prefix_lens = [len(tokens) for tokens in tokenizer(list(prefixes))["input_ids"]]
    encoded = tokenizer(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in (target_new, target_true)
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)
    new_tokens, true_tokens = (
        tokenizer(f" {target}")["input_ids"] for target in (target_new, target_true)
    )
    if llama:
        new_tokens = new_tokens[1:]
        true_tokens = true_tokens[1:]
        prefix_lens = [length - 1 for length in prefix_lens]
    if not new_tokens or not true_tokens or any(length <= 0 for length in prefix_lens):
        raise ODEBFContractError("CounterFact primary target/prefix span differs")
    logits = model(**encoded).logits
    if llama:
        logits = logits[:, 1:, :]
    probabilities = np.zeros((logits.size(0),), dtype=np.float32)
    exact_new_bits: list[int] = []
    span_payload: list[dict[str, int]] = []
    for row in range(logits.size(0)):
        tokens = new_tokens if row % 2 == 0 else true_tokens
        token_exact: list[bool] = []
        for offset, token in enumerate(tokens):
            position = prefix_lens[row // 2] + offset - 1
            if position < 0 or position >= logits.shape[1]:
                raise ODEBFContractError("CounterFact primary target span is out of bounds")
            # Keep the pinned evaluator's dtype and expression order exactly.
            probabilities[row] += -torch.nn.functional.log_softmax(
                logits[row, position, :], dim=0
            )[int(token)].item()
            if row % 2 == 0:
                token_exact.append(
                    int(torch.argmax(logits[row, position, :]).item()) == int(token)
                )
        probabilities[row] /= len(tokens)
        if row % 2 == 0:
            exact_new_bits.append(int(all(token_exact)))
        span_payload.append(
            {
                "prefix_tokens": prefix_lens[row // 2],
                "suffix_tokens": len(tokens),
                "choice": row % 2,
            }
        )
    pairs = tuple(
        PrefixSuccessAccuracy(
            PrefixNLLPair(float(probabilities[index]), float(probabilities[index + 1])),
            int(probabilities[index] < probabilities[index + 1]),
            exact_new_bits[index // 2],
            len(new_tokens),
        )
        for index in range(0, len(probabilities), 2)
    )
    attention = encoded.get("attention_mask")
    processed = int(attention.sum()) if attention is not None else int(encoded["input_ids"].numel())
    del logits, encoded
    return pairs, processed, {"spans": span_payload}


def _prompt_metric_receipt(
    name: str,
    bits_by_request: Sequence[Sequence[int]],
    scores_by_request: Sequence[Sequence[PrefixNLLPair]],
) -> PromptMetricReceipt:
    bits = tuple(tuple(int(value) for value in row) for row in bits_by_request)
    scores = tuple(tuple(row) for row in scores_by_request)
    correct = tuple(sum(row) for row in bits)
    required = tuple(len(row) for row in bits)
    numerator = sum(correct)
    denominator = sum(required)
    strict = tuple(int(all(row)) for row in bits)
    return PromptMetricReceipt(
        name,
        bits,
        correct,
        required,
        numerator,
        denominator,
        numerator / denominator,
        canonical_hash([list(row) for row in bits]),
        strict,
        sum(strict),
        len(strict),
        sum(strict) / len(strict),
        canonical_hash(list(strict)),
        tuple(tuple(item.target_new_nll for item in row) for row in scores),
        tuple(tuple(item.target_true_nll for item in row) for row in scores),
        tuple(tuple(item.target_new_nll - item.target_true_nll for item in row) for row in scores),
    )


def evaluate_counterfact_success_accuracy_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[CounterFactEvaluationCase],
    *,
    model_alias: str,
    freeze: EndpointActionFreeze,
) -> CounterFactSuccessAccuracyReceipt:
    """One-pass terminal evaluator for NLL success and strict token accuracy."""

    batch = tuple(cases)
    if len(batch) != BATCH_SIZE or len({item.request_sha256 for item in batch}) != BATCH_SIZE:
        raise ODEBFContractError("CounterFact success/accuracy evaluator requires distinct B10")
    request_order = ordered_request_digest_v1([item.request_sha256 for item in batch])
    if request_order != freeze.request_order_sha256:
        raise ODEBFContractError("CounterFact success/accuracy request order differs")
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    efficacy: list[tuple[PrefixNLLPair, ...]] = []
    generalization: list[tuple[PrefixNLLPair, ...]] = []
    locality: list[tuple[PrefixNLLPair, ...]] = []
    rewrite_success_bits: list[tuple[int, ...]] = []
    rewrite_accuracy_bits: list[tuple[int, ...]] = []
    paraphrase_success_bits: list[tuple[int, ...]] = []
    paraphrase_accuracy_bits: list[tuple[int, ...]] = []
    spans: list[dict[str, Any]] = []
    processed = 0
    with _EvaluationStateGuard(model), torch.no_grad():
        for case in batch:
            prefixes = (case.rewrite_prompt,) + case.paraphrase_prompts + case.neighborhood_prompts
            scores, tokens, span = _evaluate_prefixes_pinned_with_accuracy(
                model,
                tokenizer,
                prefixes,
                case.target_new,
                case.target_true,
                llama=llama,
                device=device,
            )
            rewrite_end = 1
            paraphrase_end = rewrite_end + len(case.paraphrase_prompts)
            efficacy.append(tuple(item.nll for item in scores[:rewrite_end]))
            generalization.append(tuple(item.nll for item in scores[rewrite_end:paraphrase_end]))
            locality.append(tuple(item.nll for item in scores[paraphrase_end:]))
            rewrite_success_bits.append(tuple(item.success for item in scores[:rewrite_end]))
            rewrite_accuracy_bits.append(tuple(item.accuracy for item in scores[:rewrite_end]))
            paraphrase_success_bits.append(tuple(item.success for item in scores[rewrite_end:paraphrase_end]))
            paraphrase_accuracy_bits.append(tuple(item.accuracy for item in scores[rewrite_end:paraphrase_end]))
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
    primary = counterfact_primary_receipt_from_scores(
        efficacy=efficacy,
        generalization=generalization,
        locality=locality,
        request_order_sha256=request_order,
        evaluation_case_identity_sha256=canonical_hash([item.raw_free_identity() for item in batch]),
        target_span_sha256=canonical_hash(spans),
        model_forward_count=BATCH_SIZE,
        processed_token_count=processed,
        endpoint_freeze_sha256=freeze.identity(),
    )
    return CounterFactSuccessAccuracyReceipt(
        primary,
        _prompt_metric_receipt("rewrite_success", rewrite_success_bits, efficacy),
        _prompt_metric_receipt("rewrite_acc", rewrite_accuracy_bits, efficacy),
        _prompt_metric_receipt("paraphrase_success", paraphrase_success_bits, generalization),
        _prompt_metric_receipt("paraphrase_acc", paraphrase_accuracy_bits, generalization),
    )


def evaluate_counterfact_primary_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[CounterFactEvaluationCase],
    *,
    model_alias: str,
    freeze: EndpointActionFreeze,
) -> CounterFactPrimaryReceipt:
    batch = tuple(cases)
    if len(batch) != BATCH_SIZE or len({item.request_sha256 for item in batch}) != BATCH_SIZE:
        raise ODEBFContractError("CounterFact primary evaluator requires distinct B10")
    request_order = ordered_request_digest_v1([item.request_sha256 for item in batch])
    if request_order != freeze.request_order_sha256:
        raise ODEBFContractError("CounterFact primary evaluator request order differs")
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
            rewrite_end = 1
            paraphrase_end = rewrite_end + len(case.paraphrase_prompts)
            efficacy.append(pairs[:rewrite_end])
            generalization.append(pairs[rewrite_end:paraphrase_end])
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
    return counterfact_primary_receipt_from_scores(
        efficacy=efficacy,
        generalization=generalization,
        locality=locality,
        request_order_sha256=request_order,
        evaluation_case_identity_sha256=canonical_hash(
            [item.raw_free_identity() for item in batch]
        ),
        target_span_sha256=canonical_hash(spans),
        model_forward_count=BATCH_SIZE,
        processed_token_count=processed,
        endpoint_freeze_sha256=freeze.identity(),
    )


def pair_primary_native_floor(
    native: CounterFactPrimaryReceipt,
    ours: CounterFactPrimaryReceipt,
) -> PairedPrimaryFloorReceipt:
    request_span_parity = (
        native.request_order_sha256 == ours.request_order_sha256
        and native.evaluation_case_identity_sha256 == ours.evaluation_case_identity_sha256
        and native.target_span_sha256 == ours.target_span_sha256
        and native.evaluator_source_sha256 == ours.evaluator_source_sha256
        and native.aggregator_source_sha256 == ours.aggregator_source_sha256
    )
    native_metrics = native.metrics()
    ours_metrics = ours.metrics()
    canonical_native = {
        metric: receipt.canonical_metric() for metric, receipt in native_metrics.items()
    }
    canonical_ours = {
        metric: receipt.canonical_metric() for metric, receipt in ours_metrics.items()
    }
    locks = {
        metric: NativeFloorLock(
            metric,
            0,
            0.0,
            1.0 / max(native_metrics[metric].denominator, 1),
            True,
        )
        for metric in PrimaryMetric
    }
    table = build_native_floor_table(canonical_native, canonical_ours, locks)
    verdicts = {item.metric: item for item in table.verdicts}
    paired: list[PairedPrimaryMetricReceipt] = []
    for metric in PrimaryMetric:
        n_receipt = native_metrics[metric]
        o_receipt = ours_metrics[metric]
        if n_receipt.per_case_required != o_receipt.per_case_required:
            raise ODEBFContractError("paired CounterFact primary denominators differ")
        wins = tuple(
            tuple(int(o == 1 and n == 0) for n, o in zip(n_bits, o_bits))
            for n_bits, o_bits in zip(n_receipt.per_case_bits, o_receipt.per_case_bits)
        )
        losses = tuple(
            tuple(int(n == 1 and o == 0) for n, o in zip(n_bits, o_bits))
            for n_bits, o_bits in zip(n_receipt.per_case_bits, o_receipt.per_case_bits)
        )
        paired.append(
            PairedPrimaryMetricReceipt(
                metric,
                verdicts[metric].passed,
                n_receipt.numerator,
                o_receipt.numerator,
                n_receipt.denominator,
                n_receipt.official_aggregate,
                o_receipt.official_aggregate,
                canonical_hash([list(item) for item in wins]),
                canonical_hash([list(item) for item in losses]),
                sum(map(sum, wins)),
                sum(map(sum, losses)),
                canonical_hash(
                    {
                        "metric": metric.value,
                        "request": native.request_order_sha256,
                        "span": native.target_span_sha256,
                        "required": list(n_receipt.per_case_required),
                    }
                ),
            )
        )
    return PairedPrimaryFloorReceipt(table, tuple(paired), request_span_parity)
