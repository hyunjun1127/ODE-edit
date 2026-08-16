from __future__ import annotations

import hashlib
import threading
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.barriers import functional_replay_risk
from project.run_scripts.ode_bf.benchmark import (
    CounterFactRequestScore,
    counterfact_batch_receipt,
)
from project.run_scripts.ode_bf.contracts import AcceptedStepClock, RolloutBudget
from project.run_scripts.ode_bf.first_hit import (
    FeasibilityVerdict,
    FixedBudgetFirstHitRollout,
    WaypointRecord,
)
from project.run_scripts.ode_bf.history import HistoryLedger, HistoryRecord
from project.run_scripts.ode_bf.sampling import (
    SampleLineage,
    StatelessReplaySchedule,
    load_cpu_sampling_seal,
)
from project.run_scripts.ode_bf.transaction import AtomicBatchTransaction


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class JointB10IntegrationTests(unittest.TestCase):
    def test_k8_first_hit_transaction_replay_and_history_are_one_joint_batch(self) -> None:
        budget = RolloutBudget()
        rollout = FixedBudgetFirstHitRollout(budget)
        t_acc = 0.0
        request_ids = tuple(_digest(f"request-{index}") for index in range(10))
        for step in range(8):
            bits = [True] * 10
            if step < 4:
                bits[7] = False
            if step == 5:
                bits[7] = False
            scores = tuple(
                CounterFactRequestScore(0.0 if bit else 2.0, 1.0)
                for bit in bits
            )
            history = functional_replay_risk(
                barrier="historical-current-teacher",
                entry_values=[0.0] * 10,
                trial_values=[0.0] * 10,
                sample_sha256=request_ids,
                budget=1.0e-3,
            )
            pretrained_ids = tuple(_digest(f"pretrained-{index}") for index in range(10))
            pretrained = functional_replay_risk(
                barrier="pretrained-theta0-teacher",
                entry_values=[0.0] * 10,
                trial_values=[0.0] * 10,
                sample_sha256=pretrained_ids,
                budget=1.0e-3,
            )
            feasibility = FeasibilityVerdict(
                True,
                True,
                True,
                history.passed,
                pretrained.passed,
                True,
            )
            clock = AcceptedStepClock(0, 1, step, 8, 1.0, 1.0, t_acc)
            rollout.append(
                WaypointRecord(
                    clock,
                    True,
                    counterfact_batch_receipt(scores),
                    feasibility,
                    _digest(f"snapshot-{step}"),
                )
            )
            t_acc = clock.t_acc_after

        result = rollout.finalize()
        self.assertEqual(result.executed_k_total, 8)
        self.assertEqual(result.joint_first_exact_hit_step, 4)
        self.assertTrue(result.reversal_after_first_hit)
        self.assertEqual(result.per_request_first_hit_step[7], 4)

        parameters = {
            f"layer-{layer}.weight": torch.nn.Parameter(
                torch.zeros((12, 12), dtype=torch.bfloat16),
                requires_grad=False,
            )
            for layer in (4, 5, 6, 7, 8)
        }
        transaction = AtomicBatchTransaction(
            parameters,
            transaction_id="joint-b10-endpoint",
            mutation_lock=threading.RLock(),
        )
        for name, parameter in parameters.items():
            transaction.stage(name, torch.ones_like(parameter))
        commit = transaction.commit(post_commit_verify=lambda: True)
        self.assertEqual(commit.commit_count, 1)
        final_success = counterfact_batch_receipt(
            tuple(CounterFactRequestScore(0.0, 1.0) for _ in range(10))
        )
        final_feasibility = FeasibilityVerdict(True, True, True, True, True, True)
        rollout.verify_post_commit(
            committed_snapshot_digest=result.selected_snapshot_digest or "",
            success=final_success,
            feasibility=final_feasibility,
        )

        ledger = HistoryLedger(max_active_records=20, layer_order=(4, 5, 6, 7, 8))
        records = tuple(
            HistoryRecord(
                request_ids[index],
                index,
                _digest(f"collision-{index}"),
                _digest(f"target-{index}"),
                1,
                _digest(f"event-{index}"),
            )
            for index in range(10)
        )
        projected = {
            layer: torch.arange(120, dtype=torch.float64).reshape(12, 10) + layer
            for layer in (4, 5, 6, 7, 8)
        }
        prospective = ledger.prospective(
            transaction_id="joint-b10-endpoint",
            expected_version=0,
            records=records,
            projected_keys=projected,
        )
        finalized = ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={layer: 1.0 for layer in projected},
        )
        self.assertEqual(finalized.appended_count, 10)
        self.assertEqual(len(ledger.snapshot().active_records), 10)

        seal = load_cpu_sampling_seal(
            str(Path(__file__).resolve().parents[1] / "locks/p0_cpu_sampling_seal.json")
        )
        schedule = StatelessReplaySchedule(seal)
        before = schedule.state_digest
        first = schedule.batch(
            SampleLineage.CONTROLLER,
            outer_batch_index=0,
            correction_cycle=0,
            waypoint=4,
            replay_batch_id=0,
        )
        second = schedule.batch(
            SampleLineage.CONTROLLER,
            outer_batch_index=0,
            correction_cycle=0,
            waypoint=4,
            replay_batch_id=0,
        )
        self.assertEqual(first, second)
        self.assertEqual(schedule.state_digest, before)


if __name__ == "__main__":
    unittest.main()
