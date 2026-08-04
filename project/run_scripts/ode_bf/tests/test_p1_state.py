from __future__ import annotations

import hashlib
import unittest

import torch

from project.run_scripts.ode_bf.benchmark import (
    CounterFactRequestScore,
    counterfact_batch_receipt,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.first_hit import FeasibilityVerdict
from project.run_scripts.ode_bf.p1_state import (
    FixedK8P1Selector,
    P1Arm,
    P1HistoryLedger,
    P1HistoryRecord,
    P1Waypoint,
    evaluate_p1a_resource_gate,
    restore_arm_snapshot,
    snapshot_touched_weights,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _success(count: int):
    return counterfact_batch_receipt(
        tuple(
            CounterFactRequestScore(0.0 if index < count else 2.0, 1.0)
            for index in range(10)
        )
    )


PASS = FeasibilityVerdict(True, True, True, True, True, True)


def _records(batch: int, version: int, *, collision: str | None = None):
    return tuple(
        P1HistoryRecord(
            _digest(f"request-{batch}-{index}"),
            batch * 10 + index,
            _digest(collision if index == 0 and collision is not None else f"collision-{batch}-{index}"),
            _digest(f"target-{batch}-{index}"),
            _digest(f"event-{batch}-{index}"),
            version,
        )
        for index in range(10)
    )


def _keys(batch: int):
    generator = torch.Generator().manual_seed(1000 + batch)
    return {
        layer: torch.randn((12 + layer, 10), generator=generator, dtype=torch.float32)
        for layer in (4, 5, 6, 7, 8)
    }


class P1EndpointTests(unittest.TestCase):
    def test_first_exact_hit_is_selected_only_after_all_k8_and_reversal_recorded(self) -> None:
        selector = FixedK8P1Selector()
        state = _digest("entry")
        selector.append(P1Waypoint(0, False, _success(4), PASS, state, _digest("field-0"), _digest("raw"), 0.0, False))
        for stage in range(1, 9):
            accepted = stage != 6
            if accepted:
                state = _digest(f"state-{stage}")
            count = 10 if stage in (4, 5, 7, 8) else 9
            selector.append(
                P1Waypoint(
                    stage,
                    accepted,
                    _success(count),
                    PASS,
                    state,
                    _digest(f"field-{stage if accepted else 5}"),
                    _digest("raw"),
                    1.0 if accepted else 0.0,
                    not accepted,
                )
            )
        selection = selector.finalize()
        self.assertEqual(selection.selected_stage, 4)
        self.assertEqual(selection.joint_first_exact_hit_step, 4)
        self.assertEqual(selection.executed_slots, 8)
        self.assertTrue(selection.reversal_after_first_hit)
        selector.verify_post_commit(
            committed_snapshot_sha256=selection.selected_snapshot_sha256,
            success=_success(10),
            feasibility=PASS,
        )

    def test_no_exact_hit_selects_feasible_fixed_k8_terminal_without_relabeling(self) -> None:
        selector = FixedK8P1Selector()
        state = _digest("entry")
        selector.append(P1Waypoint(0, False, _success(2), PASS, state, _digest("f0"), _digest("raw"), 0.0, False))
        for stage in range(1, 9):
            state = _digest(f"state-{stage}")
            selector.append(P1Waypoint(stage, True, _success(min(9, stage + 2)), PASS, state, _digest("field"), _digest("raw"), 1.0, False))
        selection = selector.finalize()
        self.assertEqual(selection.status, "NO_EXACT_BATCH_HIT")
        self.assertEqual(selection.selected_stage, 8)
        self.assertEqual(selection.max_official_success_count, 9)
        selector.verify_post_commit(
            committed_snapshot_sha256=selection.selected_snapshot_sha256,
            success=_success(9),
            feasibility=PASS,
        )

    def test_rejected_slot_must_leave_snapshot_exact(self) -> None:
        selector = FixedK8P1Selector()
        selector.append(P1Waypoint(0, False, _success(0), PASS, _digest("entry"), _digest("field"), _digest("raw"), 0.0, False))
        with self.assertRaisesRegex(ODEBFStateError, "changed"):
            selector.append(P1Waypoint(1, False, _success(0), PASS, _digest("mutated"), _digest("field"), _digest("raw"), 0.0, True))


class P1HistoryAndIsolationTests(unittest.TestCase):
    def test_actual_four_by_b10_history_chain_and_reject_fault_purity(self) -> None:
        ledger = P1HistoryLedger(layer_order=(4, 5, 6, 7, 8))
        for batch in range(4):
            before = ledger.snapshot()
            solve = _keys(batch)
            risk = {layer: value * 0.5 for layer, value in solve.items()}
            prospective = ledger.prospective(
                transaction_id=f"transaction-{batch}",
                expected_version=batch,
                records=_records(batch, batch + 1),
                solve_keys_by_layer=solve,
                risk_keys_by_layer=risk,
            )
            self.assertEqual(ledger.snapshot().digest, before.digest)
            if batch == 0:
                for phase in ("early", "middle", "late"):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        ledger.finalize(
                            prospective,
                            post_commit_verified=True,
                            load_increment_by_layer={layer: 1.0 for layer in solve},
                            fault_phase=phase,
                        )
                    self.assertEqual(ledger.snapshot().digest, before.digest)
            receipt = ledger.finalize(
                prospective,
                post_commit_verified=True,
                load_increment_by_layer={layer: float(batch + 1) for layer in solve},
            )
            self.assertEqual(receipt.appended_count, 10)
            self.assertEqual(ledger.version, batch + 1)
            self.assertEqual(len(ledger.snapshot().active_records), (batch + 1) * 10)
            self.assertEqual(ledger.solve_keys(4).shape[1], (batch + 1) * 10)
            self.assertEqual(ledger.risk_keys(4).shape[1], (batch + 1) * 10)
            self.assertLessEqual(len(ledger.rotating_records(sequential_batch=batch, waypoint=3)), 10)

    def test_obsolete_transition_is_prospective_and_atomic(self) -> None:
        ledger = P1HistoryLedger(layer_order=(4, 5, 6, 7, 8))
        first = ledger.prospective(
            transaction_id="first",
            expected_version=0,
            records=_records(0, 1, collision="shared"),
            solve_keys_by_layer=_keys(0),
            risk_keys_by_layer=_keys(0),
        )
        ledger.finalize(first, post_commit_verified=True, load_increment_by_layer={layer: 0.0 for layer in (4, 5, 6, 7, 8)})
        before = ledger.snapshot()
        records = list(_records(1, 2, collision="shared"))
        second = ledger.prospective(
            transaction_id="second",
            expected_version=1,
            records=records,
            solve_keys_by_layer=_keys(1),
            risk_keys_by_layer=_keys(1),
        )
        self.assertEqual(len(second.obsolete_request_sha256), 1)
        self.assertEqual(ledger.snapshot().digest, before.digest)
        receipt = ledger.finalize(second, post_commit_verified=True, load_increment_by_layer={layer: 0.0 for layer in (4, 5, 6, 7, 8)})
        self.assertEqual(receipt.obsolete_count, 1)
        self.assertEqual(len(ledger.snapshot().active_records), 19)
        self.assertEqual(len(ledger.snapshot().audit_obsolete_records), 1)

    def test_arm_snapshots_chain_and_restore_without_pointer_changes(self) -> None:
        parameters = {
            f"layer-{index}.weight": torch.nn.Parameter(
                torch.full((4, 5), float(index), dtype=torch.bfloat16),
                requires_grad=False,
            )
            for index in range(5)
        }
        baseline_receipt, baseline = snapshot_touched_weights(P1Arm.N32_NATIVE, 0, parameters)
        pointers = {name: value.data_ptr() for name, value in parameters.items()}
        with torch.no_grad():
            for parameter in parameters.values():
                parameter.add_(1)
        restore_arm_snapshot(parameters, baseline, baseline_receipt)
        self.assertTrue(all(parameters[name].data_ptr() == pointers[name] for name in parameters))
        self.assertEqual(
            tuple((name, hashlib.sha256(value.view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()) for name, value in baseline.items()),
            baseline_receipt.parameter_sha256,
        )

    def test_partial_invalid_joint_batch_is_rejected(self) -> None:
        ledger = P1HistoryLedger(layer_order=(4, 5, 6, 7, 8))
        bad = _keys(0)
        bad.pop(8)
        with self.assertRaisesRegex(ODEBFContractError, "incomplete"):
            ledger.prospective(
                transaction_id="bad",
                expected_version=0,
                records=_records(0, 1),
                solve_keys_by_layer=bad,
                risk_keys_by_layer=_keys(0),
            )


class P1AResourceTests(unittest.TestCase):
    def test_locked_first_batch_forecast_gate(self) -> None:
        passed = evaluate_p1a_resource_gate(
            peak_reserved_bytes=51_000 * 1024**2,
            process_maxrss_kib=47_000 * 1024,
            elapsed_first_batch_all_arms_seconds=2 * 3600,
        )
        self.assertTrue(passed.passed)
        self.assertEqual(passed.status, "P1A_CONTINUE")
        for kwargs in (
            {"peak_reserved_bytes": 52_000 * 1024**2, "process_maxrss_kib": 1, "elapsed_first_batch_all_arms_seconds": 1},
            {"peak_reserved_bytes": 1, "process_maxrss_kib": 48_000 * 1024, "elapsed_first_batch_all_arms_seconds": 1},
            {"peak_reserved_bytes": 1, "process_maxrss_kib": 1, "elapsed_first_batch_all_arms_seconds": 6 * 3600},
        ):
            self.assertEqual(evaluate_p1a_resource_gate(**kwargs).status, "P1A_RESOURCE_HOLD")


if __name__ == "__main__":
    unittest.main()
