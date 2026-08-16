from __future__ import annotations

import hashlib
import unittest

import torch

from project.run_scripts.ode_bf.accounting import (
    ComputeLedger,
    JointInitializationReceipt,
    LayerFactorReceipt,
    LongitudinalBatchPoint,
    assert_matched_rollout_accounting,
    assert_sequential_b10_trajectory,
)
from project.run_scripts.ode_bf.analysis import (
    CanonicalMetricReceipt,
    NativeFloorLock,
    PrimaryMetric,
    build_native_floor_table,
)
from project.run_scripts.ode_bf.contracts import (
    DirectZDiagnostics,
    ODEBFContractError,
    ODEBFStateError,
    RolloutBudget,
)
from project.run_scripts.ode_bf.history import HistoryLedger, HistoryRecord
from project.run_scripts.ode_bf.sampling import (
    LineageSeal,
    PopulationItem,
    SampleLineage,
    SamplingSeal,
    StatelessReplaySchedule,
    assert_cross_arm_common_random_numbers,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _records(batch: int, version: int, *, collide_first: bool = False) -> tuple[HistoryRecord, ...]:
    records = []
    for index in range(10):
        collision = (
            "collision-shared"
            if index == 0 and (batch == 0 or collide_first)
            else f"collision-{batch}-{index}"
        )
        records.append(
            HistoryRecord(
                _digest(f"request-{batch}-{index}"),
                batch * 100 + index,
                _digest(collision),
                _digest(f"target-{batch}-{index}"),
                version,
                _digest(f"event-{batch}-{index}"),
            )
        )
    return tuple(records)


def _interacting_keys(batch: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + batch)
    result: dict[int, torch.Tensor] = {}
    for layer, dimension in ((4, 7), (5, 8), (6, 9)):
        base = torch.randn((dimension, 10), generator=generator, dtype=torch.float64)
        # Shared low-rank interaction prevents this fixture from being ten
        # independent repeated singleton columns.
        shared = torch.linspace(0.1, 1.0, 10, dtype=torch.float64).unsqueeze(0)
        base[0:1] += shared
        result[layer] = base
    return result


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = HistoryLedger(max_active_records=30, layer_order=(4, 5, 6))

    def test_reject_and_faults_leave_all_history_state_byte_identical(self) -> None:
        before = self.ledger.snapshot()
        prospective = self.ledger.prospective(
            transaction_id="batch-0",
            expected_version=0,
            records=_records(0, 1),
            projected_keys=_interacting_keys(0),
        )
        self.assertEqual(self.ledger.snapshot().digest, before.digest)
        for phase in ("early", "middle", "late"):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.ledger.finalize(
                    prospective,
                    post_commit_verified=True,
                    load_increment_by_layer={4: 1.0, 5: 2.0, 6: 3.0},
                    fault_phase=phase,
                )
            self.assertEqual(self.ledger.snapshot().digest, before.digest)

    def test_all_ten_append_once_and_finalization_retry_is_idempotent(self) -> None:
        prospective = self.ledger.prospective(
            transaction_id="batch-0",
            expected_version=0,
            records=_records(0, 1),
            projected_keys=_interacting_keys(0),
        )
        receipt = self.ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0, 6: 3.0},
        )
        self.assertEqual(receipt.appended_count, 10)
        self.assertEqual(receipt.after_version, 1)
        snapshot = self.ledger.snapshot()
        self.assertEqual(len(snapshot.active_records), 10)
        for layer in (4, 5, 6):
            self.assertEqual(tuple(self.ledger.solve_key_view(layer).shape)[1], 10)
        retry = self.ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0, 6: 3.0},
        )
        self.assertTrue(retry.idempotent_replay)
        self.assertEqual(retry.appended_count, 0)
        self.assertEqual(self.ledger.snapshot().digest, snapshot.digest)

    def test_obsolete_removal_is_prospective_and_batch_atomic(self) -> None:
        first = self.ledger.prospective(
            transaction_id="batch-0",
            expected_version=0,
            records=_records(0, 1),
            projected_keys=_interacting_keys(0),
        )
        self.ledger.finalize(
            first,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 1.0, 6: 1.0},
        )
        before = self.ledger.snapshot()
        second = self.ledger.prospective(
            transaction_id="batch-1",
            expected_version=1,
            records=_records(1, 2, collide_first=True),
            projected_keys=_interacting_keys(1),
        )
        self.assertEqual(len(second.obsolete_request_sha256), 1)
        self.assertEqual(self.ledger.snapshot().digest, before.digest)
        receipt = self.ledger.finalize(
            second,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 1.0, 6: 1.0},
        )
        self.assertEqual(receipt.obsolete_count, 1)
        after = self.ledger.snapshot()
        self.assertEqual(len(after.active_records), 19)
        self.assertEqual(len(after.audit_obsolete_records), 1)

    def test_partial_invalid_and_mixed_collision_batches_fail_closed(self) -> None:
        bad_keys = _interacting_keys(0)
        bad_keys.pop(5)
        with self.assertRaisesRegex(ODEBFContractError, "incomplete"):
            self.ledger.prospective(
                transaction_id="bad-layers",
                expected_version=0,
                records=_records(0, 1),
                projected_keys=bad_keys,
            )
        records = list(_records(0, 1))
        records[1] = HistoryRecord(
            records[1].request_sha256,
            records[1].case_id,
            records[0].collision_key_sha256,
            records[1].target_sha256,
            1,
            records[1].terminal_event_sha256,
        )
        with self.assertRaisesRegex(ODEBFContractError, "ambiguous"):
            self.ledger.prospective(
                transaction_id="bad-collision",
                expected_version=0,
                records=records,
                projected_keys=_interacting_keys(0),
            )

    def test_weight_view_is_deterministic_and_persistent_keys_stay_unweighted(self) -> None:
        prospective = self.ledger.prospective(
            transaction_id="batch-0",
            expected_version=0,
            records=_records(0, 1),
            projected_keys=_interacting_keys(0),
        )
        self.ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 0.0, 5: 0.0, 6: 0.0},
        )
        before = self.ledger.snapshot().digest
        weights = {record.request_sha256: float(index + 1) for index, record in enumerate(_records(0, 1))}
        first = self.ledger.deterministic_weighted_view(4, weights)
        second = self.ledger.deterministic_weighted_view(4, weights)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(self.ledger.snapshot().digest, before)


class SamplingTests(unittest.TestCase):
    def _seal(self) -> SamplingSeal:
        items = tuple(
            PopulationItem(_digest(f"population-{index}"), f"stratum-{index % 2}", "wikipedia-calibration")
            for index in range(30)
        )
        lineages = []
        for offset, lineage in enumerate(SampleLineage):
            pool = tuple(item.item_sha256 for item in items[offset * 10 : (offset + 1) * 10])
            lineages.append(LineageSeal(lineage, _digest("population"), 41 + offset, 4, pool, ("stratum-0", "stratum-1")))
        return SamplingSeal(_digest("population"), items, tuple(lineages))

    def test_lineages_are_separate_stateless_and_common_across_arms(self) -> None:
        schedule = StatelessReplaySchedule(self._seal())
        before = schedule.state_digest
        generic = schedule.batch(
            SampleLineage.CONTROLLER,
            outer_batch_index=2,
            correction_cycle=0,
            waypoint=3,
            replay_batch_id=0,
        )
        bf = schedule.batch(
            SampleLineage.CONTROLLER,
            outer_batch_index=2,
            correction_cycle=0,
            waypoint=3,
            replay_batch_id=0,
        )
        self.assertEqual(assert_cross_arm_common_random_numbers({"generic": generic, "bf": bf}), generic.schedule_digest)
        self.assertEqual(schedule.state_digest, before)
        terminal = schedule.batch(
            SampleLineage.TERMINAL_CONFIRMATION,
            outer_batch_index=2,
            correction_cycle=0,
            waypoint=3,
            replay_batch_id=0,
        )
        self.assertTrue(set(generic.item_sha256).isdisjoint(terminal.item_sha256))

    def test_heldout_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "held-out"):
            PopulationItem(_digest("x"), "stratum", "counterfact-paraphrase")


class AccountingAndNativeFloorTests(unittest.TestCase):
    def test_genuine_joint_initialization_receipt_and_rank_one_kill(self) -> None:
        factors = tuple(
            LayerFactorReceipt(layer, (16 + layer, 10), (12 + layer, 10), 10, "torch.float64", "cpu")
            for layer in (4, 5, 6)
        )
        receipt = JointInitializationReceipt(1, 10, 10, 10, 0, 3, 0, 3, 3, False, factors, _digest("order"))
        self.assertEqual(receipt.editor_invocations, 1)
        with self.assertRaisesRegex(ODEBFContractError, "collapsed"):
            LayerFactorReceipt(4, (20, 10), (16, 10), 1, "torch.float64", "cpu")
        with self.assertRaisesRegex(ODEBFContractError, "decomposed"):
            JointInitializationReceipt(10, 10, 10, 10, 10, 3, 0, 3, 3, False, factors, _digest("order"))

    def test_k_cycle_compute_accounting_and_matched_arms(self) -> None:
        budget = RolloutBudget()
        generic = ComputeLedger()
        bf = ComputeLedger()
        for ledger in (generic, bf):
            for _ in range(8):
                ledger.record_accepted_step(accepted_dt=0.125, completed_k_total=ledger.completed_k_total + 1)
            ledger.finish_cycle(0)
            for name, count in (
                ("model_forward", 24),
                ("processed_tokens", 2400),
                ("backward", 8),
                ("target_backward", 8),
                ("trial", 24),
                ("evaluator_forward", 8),
                ("evaluator_tokens", 800),
                ("effective_bf16_weight_peak_live", 1),
            ):
                ledger.increment(name, count)
            ledger.add_direct_z_diagnostics(DirectZDiagnostics(3.0, 1.5, 0.6, 1.0))
        bf.increment("constraint_vjp", 8)
        bf.increment("qp_solve", 8)
        assert_matched_rollout_accounting(generic, bf, budget=budget)

    def test_native_floor_has_three_noncompensating_primary_rows(self) -> None:
        source = _digest("official-evaluator")
        native = {
            metric: CanonicalMetricReceipt(metric, 0.9, source, 9, 10)
            for metric in PrimaryMetric
        }
        ours = dict(native)
        ours[PrimaryMetric.LOCALITY] = CanonicalMetricReceipt(PrimaryMetric.LOCALITY, 0.8, source, 8, 10)
        locks = {
            metric: NativeFloorLock(metric, 0, 0.0, 0.1, True)
            for metric in PrimaryMetric
        }
        table = build_native_floor_table(native, ours, locks)
        self.assertFalse(table.all_primary_pass)
        self.assertEqual(table.failed_metrics, (PrimaryMetric.LOCALITY.value,))

    def test_longitudinal_units_are_ordered_joint_batches(self) -> None:
        points = tuple(
            LongitudinalBatchPoint(
                "full-ode-bf",
                index,
                (index + 1) * 10,
                1.0,
                0.9,
                0.95,
                0.9,
                0.85,
                0.01,
                0.02,
                index + 1,
                0,
                0,
                0,
                100.0 + index,
                0.25,
                _digest(f"compute-{index}"),
            )
            for index in range(3)
        )
        assert_sequential_b10_trajectory(points)
        with self.assertRaisesRegex(ODEBFContractError, "gap or reset"):
            assert_sequential_b10_trajectory((points[0], points[2]))


if __name__ == "__main__":
    unittest.main()
