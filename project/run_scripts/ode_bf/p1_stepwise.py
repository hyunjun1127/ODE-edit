"""Post-freeze numeric CounterFact receipts for the adaptive P1R4 panel.

This module reuses the pinned P1 evaluator's exact prefix scorer and official
aggregator.  It adds numeric, index-keyed diagnostics only after every rollout
action has been frozen; none of its outputs are available to the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .analysis import PrimaryMetric
from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
from .p1_evaluator import (
    CounterFactEvaluationCase,
    CounterFactPrimaryReceipt,
    PrefixNLLPair,
    _evaluate_prefixes_pinned,
    counterfact_primary_receipt_from_scores,
)
from .request_digest import ordered_request_digest_v1


STEPWISE_SCHEMA = "ode-edit-s04-ode-bf-p1r4-adaptive-stepwise/v1"
R_NEW_DENOMINATOR_EPSILON = 1.0e-8
PREDECLARED_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class StepwiseActionFreeze:
    variant: str
    request_order_sha256: str
    rollout_sha256: str
    snapshot_sha256: str
    snapshot_index: int
    accepted_snapshot_count: int
    rejected_retry_count: int
    trajectory_status: str
    action_frozen: bool = True

    def __post_init__(self) -> None:
        if not self.variant or not self.trajectory_status:
            raise ODEBFContractError("stepwise action-freeze label differs")
        if any(
            len(value) != 64
            for value in (
                self.request_order_sha256,
                self.rollout_sha256,
                self.snapshot_sha256,
            )
        ):
            raise ODEBFContractError("stepwise action-freeze digest differs")
        if (
            isinstance(self.accepted_snapshot_count, bool)
            or self.accepted_snapshot_count < 0
            or isinstance(self.snapshot_index, bool)
            or self.snapshot_index < 0
            or isinstance(self.rejected_retry_count, bool)
            or self.rejected_retry_count < 0
        ):
            raise ODEBFContractError("stepwise action-freeze counts differ")
        if not self.action_frozen:
            raise ODEBFContractError("stepwise held-out access preceded action freeze")

    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": STEPWISE_SCHEMA,
                "variant": self.variant,
                "request_order_sha256": self.request_order_sha256,
                "rollout_sha256": self.rollout_sha256,
                "snapshot_sha256": self.snapshot_sha256,
                "snapshot_index": self.snapshot_index,
                "accepted_snapshot_count": self.accepted_snapshot_count,
                "rejected_retry_count": self.rejected_retry_count,
                "trajectory_status": self.trajectory_status,
                "action_frozen": self.action_frozen,
            }
        )


@dataclass(frozen=True, slots=True)
class StepwisePrimaryReceipt:
    primary: CounterFactPrimaryReceipt
    efficacy_scores: tuple[tuple[PrefixNLLPair, ...], ...]
    generalization_scores: tuple[tuple[PrefixNLLPair, ...], ...]
    locality_scores: tuple[tuple[PrefixNLLPair, ...], ...]
    action_freeze_sha256: str
    numeric_vectors_sha256: str

    def __post_init__(self) -> None:
        groups = (
            self.efficacy_scores,
            self.generalization_scores,
            self.locality_scores,
        )
        if any(len(group) != BATCH_SIZE for group in groups):
            raise ODEBFContractError("stepwise score case count differs")
        if any(not values for group in groups for values in group):
            raise ODEBFContractError("stepwise score prompt count differs")
        if self.action_freeze_sha256 != self.primary.endpoint_freeze_sha256:
            raise ODEBFContractError("stepwise score/freeze identity differs")
        if self.numeric_vectors_sha256 != canonical_hash(self.numeric_payload()):
            raise ODEBFContractError("stepwise numeric-vector digest differs")

    def scores(self) -> dict[PrimaryMetric, tuple[tuple[PrefixNLLPair, ...], ...]]:
        return {
            PrimaryMetric.EFFICACY: self.efficacy_scores,
            PrimaryMetric.GENERALIZATION: self.generalization_scores,
            PrimaryMetric.LOCALITY: self.locality_scores,
        }

    def numeric_payload(self) -> dict[str, Any]:
        return {
            metric.value: [
                [
                    {
                        "request_index": case_index,
                        "prompt_index": prompt_index,
                        "nll_new": pair.target_new_nll,
                        "nll_old": pair.target_true_nll,
                        "margin": (
                            pair.target_new_nll - pair.target_true_nll
                            if metric is PrimaryMetric.LOCALITY
                            else pair.target_true_nll - pair.target_new_nll
                        ),
                    }
                    for prompt_index, pair in enumerate(case)
                ]
                for case_index, case in enumerate(self.scores()[metric])
            ]
            for metric in PrimaryMetric
        }

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": STEPWISE_SCHEMA,
            "primary": self.primary.raw_free_payload(),
            "numeric_vectors": self.numeric_payload(),
            "numeric_vectors_sha256": self.numeric_vectors_sha256,
            "action_freeze_sha256": self.action_freeze_sha256,
        }


def evaluate_counterfact_stepwise_primary(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[CounterFactEvaluationCase],
    *,
    model_alias: str,
    freeze: StepwiseActionFreeze,
) -> StepwisePrimaryReceipt:
    """Run the pinned evaluator once and retain its numeric NLL pairs."""

    batch = tuple(cases)
    if len(batch) != BATCH_SIZE or len({item.request_sha256 for item in batch}) != BATCH_SIZE:
        raise ODEBFContractError("stepwise CounterFact evaluator requires distinct B10")
    request_order = ordered_request_digest_v1([item.request_sha256 for item in batch])
    if request_order != freeze.request_order_sha256 or not freeze.action_frozen:
        raise ODEBFContractError("stepwise evaluator request/freeze identity differs")
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
    primary = counterfact_primary_receipt_from_scores(
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
    frozen_groups = (
        tuple(efficacy),
        tuple(generalization),
        tuple(locality),
    )
    temporary = object.__new__(StepwisePrimaryReceipt)
    object.__setattr__(temporary, "primary", primary)
    object.__setattr__(temporary, "efficacy_scores", frozen_groups[0])
    object.__setattr__(temporary, "generalization_scores", frozen_groups[1])
    object.__setattr__(temporary, "locality_scores", frozen_groups[2])
    object.__setattr__(temporary, "action_freeze_sha256", freeze.identity())
    object.__setattr__(temporary, "numeric_vectors_sha256", "0" * 64)
    numeric_sha256 = canonical_hash(temporary.numeric_payload())
    return StepwisePrimaryReceipt(
        primary,
        frozen_groups[0],
        frozen_groups[1],
        frozen_groups[2],
        freeze.identity(),
        numeric_sha256,
    )


def _flatten(
    values: Sequence[Sequence[PrefixNLLPair]],
) -> tuple[PrefixNLLPair, ...]:
    return tuple(item for case in values for item in case)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0 or not np.isfinite(observed).all():
        raise ODEBFContractError("stepwise comparison distribution differs")
    return {
        "count": int(observed.size),
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
        "quantiles": {
            f"q{int(100 * quantile):02d}": float(
                np.quantile(observed, quantile, method="linear")
            )
            for quantile in PREDECLARED_QUANTILES
        },
    }


def _success(metric: PrimaryMetric, value: PrefixNLLPair) -> int:
    return int(
        value.target_true_nll < value.target_new_nll
        if metric is PrimaryMetric.LOCALITY
        else value.target_new_nll < value.target_true_nll
    )


def compare_stepwise_primary(
    w0: StepwisePrimaryReceipt,
    native: StepwisePrimaryReceipt,
    observed: StepwisePrimaryReceipt,
) -> dict[str, Any]:
    """Return predeclared W0/Native numeric and paired-bit comparisons."""

    identities = (
        "request_order_sha256",
        "evaluation_case_identity_sha256",
        "target_span_sha256",
        "evaluator_source_sha256",
        "aggregator_source_sha256",
    )
    if any(
        len({getattr(item.primary, name) for item in (w0, native, observed)}) != 1
        for name in identities
    ):
        raise ODEBFContractError("stepwise W0/Native request-span parity differs")
    result: dict[str, Any] = {}
    for metric in PrimaryMetric:
        w_values = _flatten(w0.scores()[metric])
        n_values = _flatten(native.scores()[metric])
        o_values = _flatten(observed.scores()[metric])
        if not (len(w_values) == len(n_values) == len(o_values)):
            raise ODEBFContractError("stepwise comparison denominator differs")
        w_bits = tuple(_success(metric, item) for item in w_values)
        n_bits = tuple(_success(metric, item) for item in n_values)
        o_bits = tuple(_success(metric, item) for item in o_values)
        wins = tuple(int(o == 1 and n == 0) for n, o in zip(n_bits, o_bits))
        losses = tuple(int(n == 1 and o == 0) for n, o in zip(n_bits, o_bits))
        ties = tuple(int(n == o) for n, o in zip(n_bits, o_bits))
        delta_w0_new = tuple(
            o.target_new_nll - w.target_new_nll for w, o in zip(w_values, o_values)
        )
        delta_w0_old = tuple(
            o.target_true_nll - w.target_true_nll for w, o in zip(w_values, o_values)
        )
        delta_native_new = tuple(
            o.target_new_nll - n.target_new_nll for n, o in zip(n_values, o_values)
        )
        delta_native_old = tuple(
            o.target_true_nll - n.target_true_nll for n, o in zip(n_values, o_values)
        )
        old_degradation_only = tuple(
            int(
                metric is not PrimaryMetric.LOCALITY
                and o_bit == 1
                and o.target_new_nll >= w.target_new_nll
                and o.target_true_nll > w.target_true_nll
            )
            for w, o, o_bit in zip(w_values, o_values, o_bits)
        )
        r_new: list[float] = []
        r_new_eligible: list[int] = []
        for w, n, o in zip(w_values, n_values, o_values):
            denominator = w.target_new_nll - n.target_new_nll
            eligible = int(
                np.isfinite(denominator)
                and denominator > R_NEW_DENOMINATOR_EPSILON
            )
            r_new_eligible.append(eligible)
            if eligible:
                r_new.append(
                    (w.target_new_nll - o.target_new_nll) / denominator
                )
        official = observed.primary.metrics()[metric]
        result[metric.value] = {
            "official": official.raw_free_payload(),
            "paired_native": {
                "win_count": sum(wins),
                "loss_count": sum(losses),
                "tie_count": sum(ties),
                "win_bits_sha256": canonical_hash(list(wins)),
                "loss_bits_sha256": canonical_hash(list(losses)),
                "tie_bits_sha256": canonical_hash(list(ties)),
                "native_success_loss_count": sum(losses),
                "ours_only_win_count": sum(wins),
            },
            "delta_vs_w0": {
                "nll_new": _distribution(delta_w0_new),
                "nll_old": _distribution(delta_w0_old),
                "nll_new_sha256": canonical_hash(list(delta_w0_new)),
                "nll_old_sha256": canonical_hash(list(delta_w0_old)),
            },
            "delta_vs_native": {
                "nll_new": _distribution(delta_native_new),
                "nll_old": _distribution(delta_native_old),
                "nll_new_sha256": canonical_hash(list(delta_native_new)),
                "nll_old_sha256": canonical_hash(list(delta_native_old)),
            },
            "old_degradation_only_success_count": sum(old_degradation_only),
            "old_degradation_only_sha256": canonical_hash(
                list(old_degradation_only)
            ),
            "r_new": {
                "denominator_epsilon": R_NEW_DENOMINATOR_EPSILON,
                "eligible_count": sum(r_new_eligible),
                "ineligible_count": len(r_new_eligible) - sum(r_new_eligible),
                "eligibility_sha256": canonical_hash(r_new_eligible),
                "eligible_values_sha256": canonical_hash(r_new),
                "eligible_distribution": (
                    None if not r_new else _distribution(r_new)
                ),
            },
        }
    return {
        "schema": STEPWISE_SCHEMA,
        "request_span_parity": True,
        "metrics": result,
        "comparison_sha256": canonical_hash(result),
    }
