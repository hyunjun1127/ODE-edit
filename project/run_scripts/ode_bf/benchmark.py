"""Pinned benchmark-aware success predicates with raw-free receipts.

CounterFact parity follows ``test_batch_prediction``: token NLLs are added to
a NumPy float32 accumulator in suffix order, divided by suffix length, and the
new target succeeds only under a strict ``new_nll < true_nll`` comparison.

zsRE parity follows ``test_batch_prediction_acc``: each teacher-forced target
suffix position is a full-vocabulary argmax bit.  The official run aggregate
is the arithmetic mean of per-case position means.  No generation is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .contracts import BATCH_SIZE, ODEBFContractError, finite


COUNTERFACT_EVALUATOR_RELATIVE = "experiments/py/eval_utils_counterfact.py"
ZSRE_EVALUATOR_RELATIVE = "experiments/py/eval_utils_zsre.py"
OFFICIAL_AGGREGATOR_RELATIVE = "experiments/summarize.py"
PINNED_SOURCE_SHA256 = {
    COUNTERFACT_EVALUATOR_RELATIVE: (
        "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
    ),
    ZSRE_EVALUATOR_RELATIVE: (
        "4370a3b2c7eff88bfeffd6412be485c0195bae794f7c7a6d39b9dd9e01299d97"
    ),
    OFFICIAL_AGGREGATOR_RELATIVE: (
        "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0"
    ),
}


class BenchmarkAdapter(str, Enum):
    COUNTERFACT = "counterfact"
    ZSRE = "zsre"


def sha256_file(path: Path, *, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_evaluator_sources(alphaedit_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = alphaedit_root / relative
        if not path.is_file() or path.is_symlink():
            raise ODEBFContractError("pinned evaluator source is absent or a symlink")
        actual = sha256_file(path)
        if actual != expected:
            raise ODEBFContractError("pinned evaluator source digest differs")
        observed[relative] = actual
    return observed


def _binary_bit(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise ODEBFContractError("benchmark correctness must be a binary bit")


def _float32_suffix_mean(token_nlls: Iterable[float]) -> float:
    values = tuple(token_nlls)
    if not values:
        raise ODEBFContractError("benchmark target suffix is empty")
    accumulator = np.float32(0.0)
    for value in values:
        accumulator = np.float32(accumulator + np.float32(finite("token NLL", value)))
    accumulator = np.float32(accumulator / len(values))
    if not np.isfinite(accumulator):
        raise ODEBFContractError("benchmark suffix NLL is non-finite")
    return float(accumulator)


@dataclass(frozen=True, slots=True)
class CounterFactRequestScore:
    target_new_nll: float
    target_true_nll: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_new_nll", finite("target_new_nll", self.target_new_nll))
        object.__setattr__(self, "target_true_nll", finite("target_true_nll", self.target_true_nll))

    @property
    def success_bit(self) -> int:
        return int(self.target_new_nll < self.target_true_nll)


def counterfact_score_from_token_nlls(
    target_new_token_nlls: Iterable[float],
    target_true_token_nlls: Iterable[float],
) -> CounterFactRequestScore:
    return CounterFactRequestScore(
        _float32_suffix_mean(target_new_token_nlls),
        _float32_suffix_mean(target_true_token_nlls),
    )


@dataclass(frozen=True, slots=True)
class BatchSuccessReceipt:
    adapter: BenchmarkAdapter
    request_success_vector: tuple[int, ...]
    official_aggregate: float
    numerator: int
    denominator: int
    per_case_correct: tuple[int, ...]
    per_case_required: tuple[int, ...]
    per_case_position_bits: tuple[tuple[int, ...], ...]
    joint_exact_success: bool

    def __post_init__(self) -> None:
        if len(self.request_success_vector) != BATCH_SIZE:
            raise ODEBFContractError("success vector must contain ten request bits")
        if any(value not in (0, 1) for value in self.request_success_vector):
            raise ODEBFContractError("request success vector is not binary")
        if self.denominator <= 0 or self.numerator < 0 or self.numerator > self.denominator:
            raise ODEBFContractError("benchmark integer aggregate is invalid")
        if len(self.per_case_correct) != BATCH_SIZE or len(self.per_case_required) != BATCH_SIZE:
            raise ODEBFContractError("benchmark per-case counts must contain ten entries")
        if len(self.per_case_position_bits) != BATCH_SIZE:
            raise ODEBFContractError("benchmark position bits must contain ten cases")
        if any(
            required <= 0 or correct < 0 or correct > required
            for correct, required in zip(self.per_case_correct, self.per_case_required)
        ):
            raise ODEBFContractError("benchmark per-case counts are invalid")
        for correct, required, bits in zip(
            self.per_case_correct,
            self.per_case_required,
            self.per_case_position_bits,
        ):
            if len(bits) != required or any(bit not in (0, 1) for bit in bits):
                raise ODEBFContractError("benchmark position-bit shape is invalid")
            if sum(bits) != correct:
                raise ODEBFContractError("benchmark position bits disagree with counts")
        aggregate = finite("official aggregate", self.official_aggregate)
        if aggregate < 0.0 or aggregate > 1.0:
            raise ODEBFContractError("official aggregate is outside [0,1]")
        integer_exact = self.numerator == self.denominator
        if self.joint_exact_success != integer_exact:
            raise ODEBFContractError("joint success disagrees with integer counts")
        if self.joint_exact_success and aggregate != 1.0:
            raise ODEBFContractError("joint exact success requires canonical aggregate 1.0")

    def raw_free_payload(self) -> dict[str, object]:
        return {
            "adapter": self.adapter.value,
            "request_success_vector": list(self.request_success_vector),
            "official_aggregate": self.official_aggregate,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "per_case_correct": list(self.per_case_correct),
            "per_case_required": list(self.per_case_required),
            "per_case_position_bits": [list(bits) for bits in self.per_case_position_bits],
            "joint_exact_success": self.joint_exact_success,
        }


def counterfact_batch_receipt(
    scores: Sequence[CounterFactRequestScore],
) -> BatchSuccessReceipt:
    if len(scores) != BATCH_SIZE:
        raise ODEBFContractError("CounterFact batch must contain ten request scores")
    bits = tuple(score.success_bit for score in scores)
    numerator = sum(bits)
    # The pinned summarizer takes a NumPy arithmetic mean of request-level bits.
    aggregate = float(np.mean(np.asarray(bits, dtype=np.bool_)))
    return BatchSuccessReceipt(
        adapter=BenchmarkAdapter.COUNTERFACT,
        request_success_vector=bits,
        official_aggregate=aggregate,
        numerator=numerator,
        denominator=BATCH_SIZE,
        per_case_correct=bits,
        per_case_required=(1,) * BATCH_SIZE,
        per_case_position_bits=tuple((bit,) for bit in bits),
        joint_exact_success=numerator == BATCH_SIZE,
    )


def zsre_batch_receipt(
    per_request_position_bits: Sequence[Sequence[object]],
) -> BatchSuccessReceipt:
    if len(per_request_position_bits) != BATCH_SIZE:
        raise ODEBFContractError("zsRE batch must contain ten request score vectors")
    normalized: list[tuple[int, ...]] = []
    for values in per_request_position_bits:
        bits = tuple(_binary_bit(value) for value in values)
        if not bits:
            raise ODEBFContractError("zsRE target suffix has no required positions")
        normalized.append(bits)
    per_correct = tuple(sum(bits) for bits in normalized)
    per_required = tuple(len(bits) for bits in normalized)
    request_bits = tuple(
        int(correct == required)
        for correct, required in zip(per_correct, per_required)
    )
    # Pinned summarize.py first averages positions within each case, then cases.
    per_case_means = np.asarray(
        [np.mean(np.asarray(bits, dtype=np.bool_)) for bits in normalized],
        dtype=np.float64,
    )
    aggregate = float(np.mean(per_case_means))
    numerator = sum(per_correct)
    denominator = sum(per_required)
    return BatchSuccessReceipt(
        adapter=BenchmarkAdapter.ZSRE,
        request_success_vector=request_bits,
        official_aggregate=aggregate,
        numerator=numerator,
        denominator=denominator,
        per_case_correct=per_correct,
        per_case_required=per_required,
        per_case_position_bits=tuple(normalized),
        joint_exact_success=numerator == denominator,
    )


@dataclass(frozen=True, slots=True)
class PairedNativeEfficacyReceipt:
    adapter: BenchmarkAdapter
    native_numerator: int
    ours_numerator: int
    denominator: int
    native_official_aggregate: float
    ours_official_aggregate: float
    paired_win_bits: tuple[tuple[int, ...], ...]
    paired_loss_bits: tuple[tuple[int, ...], ...]
    native_level_efficacy_pass: bool

    @property
    def mechanistic_miss_count(self) -> int:
        return sum(sum(bits) for bits in self.paired_loss_bits)

    @property
    def compensated_win_count(self) -> int:
        return sum(sum(bits) for bits in self.paired_win_bits)


def paired_native_efficacy_gate(
    native: BatchSuccessReceipt,
    ours: BatchSuccessReceipt,
) -> PairedNativeEfficacyReceipt:
    """Apply the delta_eff=0 small-batch prerequisite without hiding losses."""

    if native.adapter is not ours.adapter:
        raise ODEBFContractError("paired efficacy adapters differ")
    if native.per_case_required != ours.per_case_required:
        raise ODEBFContractError("paired efficacy denominators differ")
    wins: list[tuple[int, ...]] = []
    losses: list[tuple[int, ...]] = []
    for native_bits, ours_bits in zip(
        native.per_case_position_bits,
        ours.per_case_position_bits,
    ):
        wins.append(
            tuple(int(ours_bit == 1 and native_bit == 0) for native_bit, ours_bit in zip(native_bits, ours_bits))
        )
        losses.append(
            tuple(int(native_bit == 1 and ours_bit == 0) for native_bit, ours_bit in zip(native_bits, ours_bits))
        )
    passed = (
        ours.numerator >= native.numerator
        and ours.official_aggregate >= native.official_aggregate
    )
    return PairedNativeEfficacyReceipt(
        native.adapter,
        native.numerator,
        ours.numerator,
        native.denominator,
        native.official_aggregate,
        ours.official_aggregate,
        tuple(wins),
        tuple(losses),
        passed,
    )
