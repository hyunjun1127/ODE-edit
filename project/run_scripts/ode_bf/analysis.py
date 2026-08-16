"""Non-compensating matched-Native floor and future promotion schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .contracts import ODEBFContractError, finite


class PrimaryMetric(str, Enum):
    EFFICACY = "efficacy"
    GENERALIZATION = "generalization"
    LOCALITY = "locality-preservation"


@dataclass(frozen=True, slots=True)
class CanonicalMetricReceipt:
    metric: PrimaryMetric
    official_aggregate: float
    source_sha256: str
    numerator: int | None = None
    denominator: int | None = None

    def __post_init__(self) -> None:
        aggregate = finite("official aggregate", self.official_aggregate)
        if aggregate < 0.0 or aggregate > 1.0:
            raise ODEBFContractError("official metric aggregate is outside [0,1]")
        if len(self.source_sha256) != 64:
            raise ODEBFContractError("metric source identity is not SHA-256")
        count_fields = (self.numerator, self.denominator)
        if (count_fields[0] is None) != (count_fields[1] is None):
            raise ODEBFContractError("count metric numerator/denominator are incomplete")
        if self.denominator is not None:
            assert self.numerator is not None
            if self.denominator <= 0 or self.numerator < 0 or self.numerator > self.denominator:
                raise ODEBFContractError("count metric fields are invalid")

    @property
    def count_based(self) -> bool:
        return self.denominator is not None


@dataclass(frozen=True, slots=True)
class NativeFloorLock:
    metric: PrimaryMetric
    count_loss_margin: int | None
    aggregate_margin: float
    minimal_resolution: float
    gh_approved: bool

    def __post_init__(self) -> None:
        aggregate_margin = finite("aggregate margin", self.aggregate_margin)
        resolution = finite("minimal resolution", self.minimal_resolution)
        if aggregate_margin < 0.0 or resolution <= 0.0:
            raise ODEBFContractError("native-floor resolution/margin is invalid")
        if aggregate_margin > resolution:
            raise ODEBFContractError("native-floor margin exceeds minimal resolution")
        if self.count_loss_margin is not None and (
            isinstance(self.count_loss_margin, bool)
            or not isinstance(self.count_loss_margin, int)
            or self.count_loss_margin < 0
        ):
            raise ODEBFContractError("count loss margin is invalid")
        if not self.gh_approved:
            raise ODEBFContractError("native-floor margin is not GH-approved")


@dataclass(frozen=True, slots=True)
class NativeFloorVerdict:
    metric: PrimaryMetric
    passed: bool
    native_aggregate: float
    ours_aggregate: float
    native_numerator: int | None
    ours_numerator: int | None
    denominator: int | None
    aggregate_margin: float
    count_loss_margin: int | None


def evaluate_native_floor(
    native: CanonicalMetricReceipt,
    ours: CanonicalMetricReceipt,
    lock: NativeFloorLock,
) -> NativeFloorVerdict:
    if native.metric is not ours.metric or native.metric is not lock.metric:
        raise ODEBFContractError("native-floor metric identities differ")
    if native.source_sha256 != ours.source_sha256:
        raise ODEBFContractError("native/ours evaluator source identities differ")
    if native.count_based != ours.count_based:
        raise ODEBFContractError("native/ours canonical metric types differ")
    count_pass = True
    denominator: int | None = None
    if native.count_based:
        if native.denominator != ours.denominator:
            raise ODEBFContractError("native/ours count denominators differ")
        if lock.count_loss_margin is None:
            raise ODEBFContractError("count-based native floor lacks an integer margin")
        assert native.numerator is not None and ours.numerator is not None
        count_pass = ours.numerator >= native.numerator - lock.count_loss_margin
        denominator = native.denominator
    elif lock.count_loss_margin is not None:
        raise ODEBFContractError("continuous native floor supplied an integer margin")
    aggregate_pass = ours.official_aggregate >= native.official_aggregate - lock.aggregate_margin
    return NativeFloorVerdict(
        native.metric,
        count_pass and aggregate_pass,
        native.official_aggregate,
        ours.official_aggregate,
        native.numerator,
        ours.numerator,
        denominator,
        lock.aggregate_margin,
        lock.count_loss_margin,
    )


@dataclass(frozen=True, slots=True)
class PrimaryNativeFloorTable:
    verdicts: tuple[NativeFloorVerdict, ...]

    def __post_init__(self) -> None:
        if {verdict.metric for verdict in self.verdicts} != set(PrimaryMetric):
            raise ODEBFContractError("native-floor table must contain all primary metrics")
        if len(self.verdicts) != len(PrimaryMetric):
            raise ODEBFContractError("native-floor table repeats a primary metric")

    @property
    def all_primary_pass(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def failed_metrics(self) -> tuple[str, ...]:
        return tuple(verdict.metric.value for verdict in self.verdicts if not verdict.passed)


def build_native_floor_table(
    native: Mapping[PrimaryMetric, CanonicalMetricReceipt],
    ours: Mapping[PrimaryMetric, CanonicalMetricReceipt],
    locks: Mapping[PrimaryMetric, NativeFloorLock],
) -> PrimaryNativeFloorTable:
    if set(native) != set(PrimaryMetric) or set(ours) != set(PrimaryMetric) or set(locks) != set(PrimaryMetric):
        raise ODEBFContractError("native-floor inputs omit a primary metric")
    return PrimaryNativeFloorTable(
        tuple(evaluate_native_floor(native[metric], ours[metric], locks[metric]) for metric in PrimaryMetric)
    )


@dataclass(frozen=True, slots=True)
class LongHorizonPromotionVerdict:
    strict_technical_pass: bool
    native_floor_pass: bool
    lenient_continuation: bool
    claim_promoted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.claim_promoted and not (
            self.strict_technical_pass and self.native_floor_pass
        ):
            raise ODEBFContractError("claim promotion bypassed a strict prerequisite")
        if self.lenient_continuation and not (
            self.strict_technical_pass and self.native_floor_pass
        ):
            raise ODEBFContractError("lenient continuation bypassed Native-level primary metrics")
        if not self.reasons:
            raise ODEBFContractError("promotion verdict lacks an explicit reason")
