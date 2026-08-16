from __future__ import annotations

import hashlib
import unittest

import numpy as np

from project.run_scripts.ode_bf.benchmark import (
    BatchSuccessReceipt,
    BenchmarkAdapter,
    CounterFactRequestScore,
    counterfact_batch_receipt,
    counterfact_score_from_token_nlls,
    paired_native_efficacy_gate,
    zsre_batch_receipt,
)
from project.run_scripts.ode_bf.contracts import (
    BATCH_SIZE,
    AcceptedStepClock,
    DirectZDiagnostics,
    JointBatchSeal,
    ODEBFContractError,
    ODEBFStateError,
    RolloutBudget,
    SealedRequest,
    assert_joint_request_batch,
)
from project.run_scripts.ode_bf.first_hit import (
    EndpointStatus,
    FeasibilityVerdict,
    FixedBudgetFirstHitRollout,
    WaypointRecord,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cf_receipt(success_count: int) -> BatchSuccessReceipt:
    scores = [
        CounterFactRequestScore(0.25, 0.5) if index < success_count
        else CounterFactRequestScore(0.5, 0.25)
        for index in range(BATCH_SIZE)
    ]
    return counterfact_batch_receipt(scores)


def _feasible(value: bool = True) -> FeasibilityVerdict:
    return FeasibilityVerdict(value, value, value, value, value, value)


class BenchmarkParityTests(unittest.TestCase):
    def test_counterfact_float32_suffix_arithmetic_and_strict_tie(self) -> None:
        values = [0.1, 0.2, 0.3]
        observed = counterfact_score_from_token_nlls(values, [0.4, 0.5])
        accumulator = np.float32(0.0)
        for value in values:
            accumulator = np.float32(accumulator + np.float32(value))
        expected = float(np.float32(accumulator / len(values)))
        self.assertEqual(observed.target_new_nll, expected)
        self.assertEqual(observed.success_bit, 1)
        self.assertEqual(CounterFactRequestScore(1.0, 1.0).success_bit, 0)

    def test_counterfact_exact_joint_aggregate_uses_integer_counts(self) -> None:
        exact = _cf_receipt(10)
        self.assertEqual(exact.request_success_vector, (1,) * 10)
        self.assertEqual((exact.numerator, exact.denominator), (10, 10))
        self.assertEqual(exact.official_aggregate, 1.0)
        self.assertTrue(exact.joint_exact_success)
        partial = _cf_receipt(9)
        self.assertEqual(partial.official_aggregate, 0.9)
        self.assertFalse(partial.joint_exact_success)

    def test_zsre_per_case_then_batch_aggregation(self) -> None:
        position_bits = [[1, 1, 1], [1]] + [[1, 1]] * 8
        exact = zsre_batch_receipt(position_bits)
        self.assertEqual(exact.official_aggregate, 1.0)
        self.assertEqual(exact.numerator, exact.denominator)
        self.assertTrue(exact.joint_exact_success)
        position_bits[0] = [1, 0, 1]
        partial = zsre_batch_receipt(position_bits)
        expected = float(np.mean([2.0 / 3.0] + [1.0] * 9))
        self.assertEqual(partial.official_aggregate, expected)
        self.assertEqual(partial.request_success_vector[0], 0)
        self.assertFalse(partial.joint_exact_success)

    def test_receipt_rejects_float_only_claim(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "integer counts"):
            BatchSuccessReceipt(
                BenchmarkAdapter.COUNTERFACT,
                (1,) * 10,
                1.0,
                9,
                10,
                (1,) * 10,
                (1,) * 10,
                tuple((1,) for _ in range(10)),
                True,
            )

    def test_native_level_gate_and_paired_mechanistic_losses(self) -> None:
        native = _cf_receipt(10)
        ours = _cf_receipt(9)
        failed = paired_native_efficacy_gate(native, ours)
        self.assertFalse(failed.native_level_efficacy_pass)
        self.assertEqual(failed.mechanistic_miss_count, 1)
        native_partial = _cf_receipt(8)
        ours_partial = _cf_receipt(9)
        passed = paired_native_efficacy_gate(native_partial, ours_partial)
        self.assertTrue(passed.native_level_efficacy_pass)
        self.assertEqual(passed.ours_numerator, 9)


class BatchAndCycleContractTests(unittest.TestCase):
    def test_joint_batch_is_exactly_ten_and_not_singletons(self) -> None:
        requests = [
            {"case_id": index, "request_sha256": _digest(f"request-{index}")}
            for index in range(10)
        ]
        assert_joint_request_batch(requests)
        with self.assertRaisesRegex(ODEBFContractError, "joint ten-request"):
            assert_joint_request_batch(requests[:1])
        assert_joint_request_batch(requests[:1], allow_singleton_algebra_fixture=True)
        duplicated = list(requests)
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ODEBFContractError, "not distinct"):
            assert_joint_request_batch(duplicated)

    def test_seal_order_distinctness_and_exclusions(self) -> None:
        requests = tuple(
            SealedRequest(index, _digest(f"request-{index}"), index)
            for index in range(10)
        )
        seal = JointBatchSeal(_digest("dataset"), "odebf-p0-b10-v1", requests, (99,))
        self.assertEqual(len(seal.root_digest()), 64)
        with self.assertRaisesRegex(ODEBFContractError, "collides"):
            JointBatchSeal(_digest("dataset"), "odebf-p0-b10-v1", requests, (9,))

    def test_k_resolution_cycle_horizon_and_oracle_residual_are_separate(self) -> None:
        p0 = RolloutBudget()
        p0.assert_current_p0()
        self.assertEqual(p0.k_resolution, 8)
        self.assertEqual(p0.correction_cycles, 1)
        self.assertEqual(p0.k_total, 8)
        self.assertEqual(p0.eta, 0.125)
        future = RolloutBudget(correction_cycles=3)
        self.assertEqual(future.k_total, 24)
        with self.assertRaisesRegex(ODEBFContractError, "technical P0"):
            future.assert_current_p0()
        diagnostics = DirectZDiagnostics(4.0, 2.0, 0.5, 1.0)
        self.assertEqual(diagnostics.local_progress_fraction, 1.0)
        self.assertGreater(diagnostics.oracle_residual_norm, 0.0)


class FixedBudgetFirstHitTests(unittest.TestCase):
    def _clock(self, step: int, beta: float, t_acc: float) -> AcceptedStepClock:
        return AcceptedStepClock(0, 1, step, 8, 1.0, beta, t_acc)

    def test_first_hit_is_selected_only_after_all_k8_with_reversal(self) -> None:
        rollout = FixedBudgetFirstHitRollout(RolloutBudget())
        t_acc = 0.0
        success_counts = [4, 7, 10, 9, 10, 10, 8, 10]
        for step, count in enumerate(success_counts):
            clock = self._clock(step, 1.0, t_acc)
            t_acc = clock.t_acc_after
            rollout.append(
                WaypointRecord(
                    clock,
                    True,
                    _cf_receipt(count),
                    _feasible(),
                    _digest(f"snapshot-{step}"),
                )
            )
            if step < 7:
                with self.assertRaisesRegex(ODEBFStateError, "before fixed K_total"):
                    rollout.finalize()
        result = rollout.finalize()
        self.assertEqual(result.status, EndpointStatus.EXACT_BATCH_HIT)
        self.assertEqual(result.joint_first_exact_hit_step, 2)
        self.assertEqual(result.selected_snapshot_digest, _digest("snapshot-2"))
        self.assertEqual(result.all_exact_hit_steps, (2, 4, 5, 7))
        self.assertTrue(result.reversal_after_first_hit)
        self.assertTrue(result.fixed_budget_terminal_exact)
        self.assertEqual(result.executed_k_total, 8)
        self.assertEqual(result.final_t_acc, 1.0)
        rollout.verify_post_commit(
            committed_snapshot_digest=_digest("snapshot-2"),
            success=_cf_receipt(10),
            feasibility=_feasible(),
        )

    def test_no_exact_hit_is_diagnostic_only_and_cannot_commit(self) -> None:
        rollout = FixedBudgetFirstHitRollout(RolloutBudget())
        t_acc = 0.0
        for step, count in enumerate([1, 5, 8, 9, 8, 7, 9, 6]):
            clock = self._clock(step, 1.0, t_acc)
            t_acc = clock.t_acc_after
            rollout.append(
                WaypointRecord(clock, True, _cf_receipt(count), _feasible(), _digest(str(step)))
            )
        result = rollout.finalize()
        self.assertEqual(result.status, EndpointStatus.NO_EXACT_BATCH_HIT)
        self.assertEqual(result.max_official_aggregate, 0.9)
        self.assertEqual(result.earliest_max_aggregate_step, 3)
        self.assertFalse(result.scientific_commit_permitted)
        with self.assertRaisesRegex(ODEBFStateError, "cannot be scientifically committed"):
            rollout.verify_post_commit(
                committed_snapshot_digest=_digest("3"),
                success=_cf_receipt(10),
                feasibility=_feasible(),
            )

    def test_rejected_step_advances_neither_state_time_nor_endpoint(self) -> None:
        rollout = FixedBudgetFirstHitRollout(RolloutBudget())
        t_acc = 0.0
        for step in range(8):
            accepted = step != 3
            beta = 1.0 if accepted else 0.0
            clock = self._clock(step, beta, t_acc)
            t_acc = clock.t_acc_after
            rollout.append(
                WaypointRecord(
                    clock,
                    accepted,
                    _cf_receipt(10) if accepted else None,
                    _feasible() if accepted else None,
                    _digest(str(step)) if accepted else None,
                )
            )
        result = rollout.finalize()
        self.assertEqual(result.accepted_waypoint_count, 7)
        self.assertEqual(result.final_t_acc, 0.875)
        self.assertEqual(result.joint_first_exact_hit_step, 0)


if __name__ == "__main__":
    unittest.main()
