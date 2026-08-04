from __future__ import annotations

import hashlib
import inspect
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf.contracts import BATCH_SIZE, FIXED_K, ODEBFContractError
from project.run_scripts.ode_bf.functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from project.run_scripts.ode_bf.p1_runtime import (
    ArmRuntimeState,
    P1StageRecorder,
    _assert_arm_batch_transition,
    _assert_matched_frozen_pair,
    _run_native_batch,
    _run_nonnative_rollout,
)
from project.run_scripts.ode_bf.p1_state import (
    P1_ARM_ORDER,
    P1Arm,
    P1HistoryLedger,
    P1HistoryRecord,
    restore_arm_snapshot,
    snapshot_touched_weights,
)
from project.run_scripts.ode_bf.resource import forecast_p1_b10_memory
from project.run_scripts.ode_bf.transaction import AtomicBatchTransaction


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _records(batch: int, version: int) -> tuple[P1HistoryRecord, ...]:
    return tuple(
        P1HistoryRecord(
            _digest(f"request-{batch}-{index}"),
            batch * BATCH_SIZE + index,
            _digest(f"collision-{batch}-{index}"),
            _digest(f"target-{batch}-{index}"),
            _digest(f"event-{batch}"),
            version,
        )
        for index in range(BATCH_SIZE)
    )


def _joint_candidates(
    parameters: dict[str, torch.nn.Parameter],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    result: dict[str, torch.Tensor] = {}
    for ordinal, (name, parameter) in enumerate(sorted(parameters.items())):
        left = torch.randn(
            (parameter.shape[0], BATCH_SIZE), generator=generator, dtype=torch.float64
        ) * 0.01
        right = torch.randn(
            (parameter.shape[1], BATCH_SIZE), generator=generator, dtype=torch.float64
        ) * 0.01
        # Every column has a distinct contribution to one shared rank-10 product.
        left[0] += torch.linspace(0.001, 0.01, BATCH_SIZE, dtype=torch.float64)
        right[0] += torch.linspace(0.01, 0.001, BATCH_SIZE, dtype=torch.float64)
        factor = WaypointFactor(
            name,
            ordinal + 4,
            0,
            0,
            ordinal,
            1.0,
            left,
            right,
        )
        candidate, stats = assemble_effective_bf16(parameter, (factor,), row_block=3)
        if stats.rank_columns_total != BATCH_SIZE:
            raise AssertionError("joint fixture did not retain rank 10")
        result[name] = candidate
    return result


def _history_keys(batch: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(9000 + batch)
    return {
        layer: torch.randn((11, BATCH_SIZE), generator=generator, dtype=torch.float32)
        for layer in range(4, 9)
    }


class SequentialPanelIntegrationTests(unittest.TestCase):
    def test_four_isolated_arms_retain_four_joint_b10_transactions(self) -> None:
        torch.manual_seed(4200)
        parameters = {
            f"layers.{layer}.weight": torch.nn.Parameter(
                torch.randn((12, 11), dtype=torch.bfloat16), requires_grad=False
            )
            for layer in range(4, 9)
        }
        pointers = {name: value.data_ptr() for name, value in parameters.items()}
        base_receipt, base_values = snapshot_touched_weights(
            P1Arm.N32_NATIVE, 0, parameters
        )
        states: dict[P1Arm, ArmRuntimeState] = {}
        for arm in P1_ARM_ORDER:
            receipt, values = snapshot_touched_weights(arm, 0, parameters)
            states[arm] = ArmRuntimeState(
                arm,
                P1HistoryLedger(layer_order=range(4, 9), maximum_records=40),
                ComputeLedger(),
                receipt,
                values,
            )
        shared_lock = threading.RLock()
        for batch in range(4):
            for arm_index, arm in enumerate(P1_ARM_ORDER):
                state = states[arm]
                restore_arm_snapshot(
                    parameters, state.snapshot_values, state.snapshot_receipt
                )
                before_history = state.history.snapshot().digest
                candidates = _joint_candidates(
                    parameters, seed=10000 + batch * 100 + arm_index
                )
                transaction = AtomicBatchTransaction(
                    parameters,
                    transaction_id=f"{arm.value}-batch-{batch}",
                    mutation_lock=shared_lock,
                )
                for name, candidate in candidates.items():
                    transaction.stage(name, candidate)
                transaction.commit(
                    post_commit_verify=lambda: all(
                        tensor_sha256(parameters[name]) == tensor_sha256(candidate)
                        for name, candidate in candidates.items()
                    )
                )
                solve = _history_keys(batch * len(P1_ARM_ORDER) + arm_index)
                prospective = state.history.prospective(
                    transaction_id=f"{arm.value}-history-{batch}",
                    expected_version=batch,
                    records=_records(batch, batch + 1),
                    solve_keys_by_layer=solve,
                    risk_keys_by_layer={layer: value * 0.5 for layer, value in solve.items()},
                )
                state.history.finalize(
                    prospective,
                    post_commit_verified=True,
                    load_increment_by_layer={layer: 1.0 for layer in range(4, 9)},
                )
                receipt, values = snapshot_touched_weights(arm, batch + 1, parameters)
                state.snapshot_receipt = receipt
                state.snapshot_values = values
                payload = {"status": "COMMITTED"}
                state.completed_batches.append(payload)
                _assert_arm_batch_transition(
                    state,
                    payload,
                    sequential_batch=batch,
                    history_before_sha256=before_history,
                )
        for state in states.values():
            self.assertEqual(state.snapshot_receipt.sequential_batch_completed, 4)
            self.assertEqual(state.history.version, 4)
            self.assertEqual(len(state.history.snapshot().active_records), 40)
            self.assertEqual(len(state.completed_batches), 4)
        restore_arm_snapshot(parameters, base_values, base_receipt)
        self.assertEqual(
            {name: value.data_ptr() for name, value in parameters.items()}, pointers
        )

    def test_joint_faults_restore_every_layer_and_append_no_history(self) -> None:
        parameters = {
            f"layers.{layer}.weight": torch.nn.Parameter(
                torch.full((12, 11), float(layer), dtype=torch.bfloat16),
                requires_grad=False,
            )
            for layer in range(4, 9)
        }
        entry = {name: value.detach().clone() for name, value in parameters.items()}
        pointers = {name: value.data_ptr() for name, value in parameters.items()}
        history = P1HistoryLedger(layer_order=range(4, 9), maximum_records=40)
        before_history = history.snapshot().digest
        candidates = _joint_candidates(parameters, seed=21000)
        for fault_after in (0, 2, 4):
            transaction = AtomicBatchTransaction(
                parameters,
                transaction_id=f"fault-{fault_after}",
                mutation_lock=threading.RLock(),
            )
            for name, candidate in candidates.items():
                transaction.stage(name, candidate)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                transaction.commit(
                    post_commit_verify=lambda: True,
                    fault_after_writes=fault_after,
                )
            self.assertEqual(history.snapshot().digest, before_history)
            for name, parameter in parameters.items():
                self.assertTrue(torch.equal(parameter, entry[name]))
                self.assertEqual(parameter.data_ptr(), pointers[name])

    def test_history_finalize_fault_inside_postcommit_rolls_back_all_weights(self) -> None:
        parameters = {
            f"layers.{layer}.weight": torch.nn.Parameter(
                torch.full((12, 11), float(layer), dtype=torch.bfloat16),
                requires_grad=False,
            )
            for layer in range(4, 9)
        }
        entry = {name: value.detach().clone() for name, value in parameters.items()}
        history = P1HistoryLedger(layer_order=range(4, 9), maximum_records=40)
        solve = _history_keys(88)
        prospective = history.prospective(
            transaction_id="combined-transaction",
            expected_version=0,
            records=_records(0, 1),
            solve_keys_by_layer=solve,
            risk_keys_by_layer={layer: value * 0.5 for layer, value in solve.items()},
        )
        before_history = history.snapshot().digest
        candidates = _joint_candidates(parameters, seed=22000)
        for phase in ("early", "middle", "late"):
            transaction = AtomicBatchTransaction(
                parameters,
                transaction_id=f"combined-{phase}",
                mutation_lock=threading.RLock(),
            )
            for name, candidate in candidates.items():
                transaction.stage(name, candidate)

            def verify(current_phase: str = phase) -> bool:
                history.finalize(
                    prospective,
                    post_commit_verified=True,
                    load_increment_by_layer={layer: 1.0 for layer in range(4, 9)},
                    fault_phase=current_phase,
                )
                return True

            with self.assertRaisesRegex(RuntimeError, "injected"):
                transaction.commit(post_commit_verify=verify)
            self.assertEqual(history.snapshot().digest, before_history)
            for name, parameter in parameters.items():
                self.assertTrue(torch.equal(parameter, entry[name]))


class RuntimeSourceAndReceiptTests(unittest.TestCase):
    @staticmethod
    def _matched_payload(projector: str) -> dict[str, object]:
        waypoints: list[dict[str, object]] = [
            {
                "stage": 0,
                "field_sha256": "f" * 64,
                "raw_velocity_sha256": "r" * 64,
            }
        ]
        for stage in range(1, FIXED_K + 1):
            waypoints.append(
                {
                    "stage": stage,
                    "field_sha256": "f" * 64,
                    "raw_velocity_sha256": "r" * 64,
                    "controller_schedule_sha256": _digest(f"schedule-{stage}"),
                    "projection": {"barrier_projector": projector},
                    "trials": [
                        {
                            "beta": beta,
                            "historical": {"sample_order_sha256": _digest(f"h-{stage}")},
                            "pretrained": {"sample_order_sha256": _digest(f"p-{stage}")},
                        }
                        for beta in (1.0, 0.5, 0.25)
                    ],
                }
            )
        counters = {
            name: 1
            for name in (
                "model_forward",
                "processed_tokens",
                "backward",
                "target_backward",
                "trial",
                "functional_h_replay",
                "functional_p_replay",
                "evaluator_forward",
                "evaluator_tokens",
            )
        }
        return {
            "arm": P1Arm.F_G.value if projector == "identity" else P1Arm.F_BF.value,
            "request_order_sha256": "q" * 64,
            "field_sha256": "f" * 64,
            "raw_velocity_sha256": "r" * 64,
            "shared_source_compute": {
                "actual_owner": P1Arm.F_G.value,
                "normalized_counter_delta": {"backward": 1},
                "actual_owner_wall_seconds": 0.5,
            },
            "waypoints": waypoints,
            "rollout_compute_delta": counters,
        }

    def test_projector_only_pair_gate_and_negative_raw_mismatch(self) -> None:
        generic = self._matched_payload("identity")
        bf = self._matched_payload("cbf")
        observed = _assert_matched_frozen_pair(generic, bf)
        self.assertTrue(observed["projector_only_difference"])
        bf["raw_velocity_sha256"] = "x" * 64
        with self.assertRaisesRegex(ODEBFContractError, "field/raw/request"):
            _assert_matched_frozen_pair(generic, bf)

    def test_commit_paths_use_one_shared_lock_and_fixed_k8_source(self) -> None:
        native = inspect.getsource(_run_native_batch)
        nonnative = inspect.getsource(_run_nonnative_rollout)
        self.assertIn("mutation_lock=mutation_lock", native)
        self.assertIn("mutation_lock=mutation_lock", nonnative)
        self.assertNotIn("mutation_lock=threading.RLock()", native + nonnative)
        self.assertIn("range(1, FIXED_K + 1)", nonnative)
        self.assertNotIn("model.generate", nonnative)
        self.assertNotIn("session03", nonnative.lower())
        for component in (
            "barrier_projection_qcqp",
            "target_velocity_backward",
            "functional_trial_replay",
            "authoritative_bf16_rewrite_verdict",
            "heldout_primary_native_and_ours",
            "atomic_commit_and_postverify",
        ):
            self.assertIn(component, nonnative)

    def test_stage_receipt_is_create_once_and_raw_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = P1StageRecorder(Path(directory))
            digest = recorder.record("post_fixture", {"count": 10, "digest": "a" * 64})
            self.assertEqual(len(digest), 64)
            recorder.sequence = 0
            with self.assertRaises(FileExistsError):
                recorder.record("post_fixture", {"count": 10})

    def test_both_aliases_use_the_same_b10_panel_memory_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifact = root / "locks/p0_artifact_lock.json"
        base = root.parent / "ode_alloc/p0_artifact_lock_r1.json"
        forecasts = [
            forecast_p1_b10_memory(artifact, base, alias)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
        ]
        for forecast in forecasts:
            self.assertEqual(forecast.edit_batch_size, 10)
            self.assertEqual(forecast.sequential_batch_count, 4)
            self.assertEqual(forecast.arm_count, 4)
            self.assertLessEqual(forecast.forecast_gpu_peak_mib, 65_000)
            self.assertLessEqual(forecast.forecast_host_peak_mib, 65_000)
            self.assertFalse(forecast.dense_fp64_full_delta)
        self.assertEqual(
            forecasts[0].runtime_reserved_hold_limit_mib,
            forecasts[1].runtime_reserved_hold_limit_mib,
        )


if __name__ == "__main__":
    unittest.main()
