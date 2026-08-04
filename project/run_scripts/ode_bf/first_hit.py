"""Fixed-budget exact-first-hit selection for a genuine joint batch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .benchmark import BatchSuccessReceipt
from .contracts import (
    BATCH_SIZE,
    AcceptedStepClock,
    ODEBFContractError,
    ODEBFStateError,
    RolloutBudget,
)


class EndpointStatus(str, Enum):
    EXACT_BATCH_HIT = "EXACT_BATCH_HIT"
    NO_EXACT_BATCH_HIT = "NO_EXACT_BATCH_HIT"


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    structural_h: bool
    structural_p: bool
    trust: bool
    functional_h: bool
    functional_p: bool
    authoritative_bf16: bool

    @property
    def all_pass(self) -> bool:
        return all(
            (
                self.structural_h,
                self.structural_p,
                self.trust,
                self.functional_h,
                self.functional_p,
                self.authoritative_bf16,
            )
        )


@dataclass(frozen=True, slots=True)
class WaypointRecord:
    clock: AcceptedStepClock
    accepted: bool
    success: BatchSuccessReceipt | None
    feasibility: FeasibilityVerdict | None
    snapshot_digest: str | None

    def __post_init__(self) -> None:
        if self.accepted:
            if self.clock.accepted_beta <= 0.0:
                raise ODEBFContractError("accepted waypoint requires positive beta")
            if self.success is None or self.feasibility is None:
                raise ODEBFContractError("accepted waypoint lacks an exact verdict")
            if self.snapshot_digest is None or len(self.snapshot_digest) != 64:
                raise ODEBFContractError("accepted waypoint lacks a snapshot digest")
        else:
            if self.clock.accepted_beta != 0.0:
                raise ODEBFContractError("rejected step must contribute zero accepted time")
            if any(value is not None for value in (self.success, self.feasibility, self.snapshot_digest)):
                raise ODEBFContractError("rejected step cannot expose an endpoint verdict")

    @property
    def exact_feasible_hit(self) -> bool:
        return bool(
            self.accepted
            and self.success is not None
            and self.success.joint_exact_success
            and self.feasibility is not None
            and self.feasibility.all_pass
        )


@dataclass(frozen=True, slots=True)
class FirstHitResult:
    status: EndpointStatus
    joint_first_exact_hit_step: int | None
    joint_first_exact_hit_cycle: int | None
    selected_snapshot_digest: str | None
    all_exact_hit_steps: tuple[int, ...]
    per_request_first_hit_step: tuple[int | None, ...]
    persistence_after_first_hit: tuple[bool, ...]
    reversal_after_first_hit: bool
    fixed_budget_terminal_exact: bool
    max_official_aggregate: float
    earliest_max_aggregate_step: int | None
    executed_k_total: int
    accepted_waypoint_count: int
    final_t_acc: float

    @property
    def scientific_commit_permitted(self) -> bool:
        return self.status is EndpointStatus.EXACT_BATCH_HIT


class FixedBudgetFirstHitRollout:
    """Records all locked steps and selects only after the budget is exhausted."""

    def __init__(self, budget: RolloutBudget) -> None:
        self.budget = budget
        self._records: list[WaypointRecord] = []
        self._result: FirstHitResult | None = None

    @property
    def records(self) -> tuple[WaypointRecord, ...]:
        return tuple(self._records)

    def append(self, record: WaypointRecord) -> None:
        if self._result is not None:
            raise ODEBFStateError("fixed-budget rollout is already finalized")
        expected = len(self._records)
        if expected >= self.budget.k_total:
            raise ODEBFStateError("fixed-budget rollout received an extra step")
        if record.clock.k_total_index != expected:
            raise ODEBFContractError("rollout step/cycle ordering is not contiguous")
        if record.clock.k_resolution != self.budget.k_resolution:
            raise ODEBFContractError("rollout clock resolution differs from budget")
        if record.clock.cycle_count != self.budget.correction_cycles:
            raise ODEBFContractError("rollout clock cycle count differs from budget")
        if record.clock.nominal_t_per_cycle != self.budget.nominal_t_per_cycle:
            raise ODEBFContractError("rollout clock nominal T differs from budget")
        expected_t = self._records[-1].clock.t_acc_after if self._records else 0.0
        if record.clock.t_acc_before != expected_t:
            raise ODEBFContractError("accepted-time ledger is discontinuous")
        self._records.append(record)

    def finalize(self) -> FirstHitResult:
        if self._result is not None:
            return self._result
        if len(self._records) != self.budget.k_total:
            raise ODEBFStateError("endpoint selection before fixed K_total is forbidden")

        accepted = [record for record in self._records if record.accepted]
        exact_hits = tuple(
            record.clock.k_total_index
            for record in accepted
            if record.exact_feasible_hit
        )
        first_hit = exact_hits[0] if exact_hits else None
        hit_record = self._records[first_hit] if first_hit is not None else None

        per_request: list[int | None] = [None] * BATCH_SIZE
        for record in accepted:
            assert record.success is not None
            assert record.feasibility is not None
            if not record.feasibility.all_pass:
                continue
            for index, bit in enumerate(record.success.request_success_vector):
                if bit and per_request[index] is None:
                    per_request[index] = record.clock.k_total_index

        persistence: tuple[bool, ...] = ()
        if first_hit is not None:
            persistence = tuple(
                record.exact_feasible_hit
                for record in accepted
                if record.clock.k_total_index > first_hit
            )

        aggregates = [
            (record.success.official_aggregate, record.clock.k_total_index)
            for record in accepted
            if record.success is not None
        ]
        if aggregates:
            maximum = max(value for value, _ in aggregates)
            earliest_max = min(step for value, step in aggregates if value == maximum)
        else:
            maximum = 0.0
            earliest_max = None

        terminal = self._records[-1]
        terminal_exact = terminal.exact_feasible_hit
        final_t_acc = terminal.clock.t_acc_after
        self._result = FirstHitResult(
            status=(
                EndpointStatus.EXACT_BATCH_HIT
                if first_hit is not None
                else EndpointStatus.NO_EXACT_BATCH_HIT
            ),
            joint_first_exact_hit_step=first_hit,
            joint_first_exact_hit_cycle=(
                hit_record.clock.correction_cycle if hit_record is not None else None
            ),
            selected_snapshot_digest=(
                hit_record.snapshot_digest if hit_record is not None else None
            ),
            all_exact_hit_steps=exact_hits,
            per_request_first_hit_step=tuple(per_request),
            persistence_after_first_hit=persistence,
            reversal_after_first_hit=any(not value for value in persistence),
            fixed_budget_terminal_exact=terminal_exact,
            max_official_aggregate=maximum,
            earliest_max_aggregate_step=earliest_max,
            executed_k_total=len(self._records),
            accepted_waypoint_count=len(accepted),
            final_t_acc=final_t_acc,
        )
        return self._result

    def verify_post_commit(
        self,
        *,
        committed_snapshot_digest: str,
        success: BatchSuccessReceipt,
        feasibility: FeasibilityVerdict,
    ) -> None:
        result = self.finalize()
        if not result.scientific_commit_permitted:
            raise ODEBFStateError("NO_EXACT_BATCH_HIT cannot be scientifically committed")
        if committed_snapshot_digest != result.selected_snapshot_digest:
            raise ODEBFContractError("committed endpoint is not the selected first hit")
        if not success.joint_exact_success or not feasibility.all_pass:
            raise ODEBFContractError("post-commit all-ten exact verification failed")


def complete_synthetic_rollout(
    budget: RolloutBudget,
    records: Sequence[WaypointRecord],
) -> FirstHitResult:
    """Small helper used by deterministic CPU fixtures."""

    rollout = FixedBudgetFirstHitRollout(budget)
    for record in records:
        rollout.append(record)
    return rollout.finalize()
