from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_state import P1HistoryRecord
from project.run_scripts.ode_bf.p1r29_sequential_preparation import (
    FIXED_H,
    MAXIMUM_HISTORY_RECORDS,
    CumulativeStructuralPState,
    LowRankUpdate,
    SequentialArmState,
    SequentialOuterRuntimeSkeleton,
    assert_round1_empty_history_equivalence,
    build_b10x10_outer_plan,
    build_history_factorization,
    incremental_structural_h,
    solve_alpha_woodbury_cached,
    solve_exact_strength_soft_hp,
    stage_a_dry_plan,
)
from project.run_scripts.ode_bf.woodbury import ProjectorCertificate


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def history_records(version: int, prefix: str, collision_offset: int = 0) -> tuple[P1HistoryRecord, ...]:
    return tuple(
        P1HistoryRecord(
            digest(f"{prefix}-request-{index}"),
            (version - 1) * 10 + index,
            digest(f"collision-{collision_offset + index}"),
            digest(f"{prefix}-target-{index}"),
            digest(f"{prefix}-terminal-{index}"),
            version,
        )
        for index in range(10)
    )


class HistoricalWoodburyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.dimension = 32
        diagonal = torch.linspace(0.2, 0.9, self.dimension, dtype=torch.float64)
        self.projector = torch.diag(diagonal)
        self.certificate = ProjectorCertificate(
            digest("general-projector"),
            0.0,
            0.4,
            "artifact-unverified",
            1.0e-10,
        )

    def test_cached_exact_backend_parity_history_0_10_50_90(self) -> None:
        for columns in (0, 10, 50, 90):
            with self.subTest(columns=columns):
                current = torch.randn(self.dimension, 10, dtype=torch.float64)
                raw = torch.randn(self.dimension, columns, dtype=torch.float64)
                projected = self.projector @ raw
                q_miss, factor, miss = solve_alpha_woodbury_cached(
                    self.projector,
                    current,
                    raw,
                    projected,
                    history_version=columns // 10,
                    regularization=0.25,
                    projector_certificate=self.certificate,
                    cached_factorization=None,
                )
                q_hit, reused, hit = solve_alpha_woodbury_cached(
                    self.projector,
                    current,
                    raw,
                    projected,
                    history_version=columns // 10,
                    regularization=0.25,
                    projector_certificate=self.certificate,
                    cached_factorization=factor,
                )
                self.assertTrue(torch.equal(q_miss, q_hit))
                self.assertIs(reused, factor)
                self.assertFalse(miss.history_cache_hit)
                self.assertTrue(hit.history_cache_hit)
                self.assertTrue(hit.parity_passed)
                self.assertEqual(hit.small_dimension, columns + 10)
                self.assertEqual(hit.model_forward_count + hit.backward_count, 0)

    def test_projected_key_drift_disables_cache(self) -> None:
        raw = torch.randn(self.dimension, 10, dtype=torch.float64)
        projected = self.projector @ raw
        projected[0, 0] += torch.finfo(torch.float64).eps
        with self.assertRaisesRegex(ODEBFContractError, "semantic drift"):
            build_history_factorization(
                self.projector,
                raw,
                projected,
                history_version=1,
                regularization=0.25,
            )

    def test_alias_layer_dimension_shape_fixtures(self) -> None:
        for alias, dimension in (("llama", 24), ("qwen", 40)):
            with self.subTest(alias=alias):
                projector = torch.diag(torch.linspace(0.3, 0.8, dimension, dtype=torch.float64))
                current = torch.randn(dimension, 10, dtype=torch.float64)
                raw = torch.randn(dimension, 10, dtype=torch.float64)
                projected = projector @ raw
                q, _, receipt = solve_alpha_woodbury_cached(
                    projector,
                    current,
                    raw,
                    projected,
                    history_version=1,
                    regularization=0.5,
                    projector_certificate=self.certificate,
                    cached_factorization=None,
                )
                self.assertEqual(q.shape, (dimension, 10))
                self.assertTrue(receipt.parity_passed)


class StructuralRiskTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(30)
        self.layers = (4, 5)
        self.covariance = torch.diag(torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64))

    def update(self, label: str) -> LowRankUpdate:
        seed = int(hashlib.sha256(label.encode()).hexdigest()[:7], 16)
        generator = torch.Generator().manual_seed(seed)
        left = torch.randn(2, 2, dtype=torch.float64, generator=generator)
        right = torch.randn(3, 2, dtype=torch.float64, generator=generator)
        return LowRankUpdate.build(
            left,
            right,
            coefficient=1.0,
            covariance_action=lambda value: self.covariance @ value,
        )

    def dense_value(self, updates: list[LowRankUpdate]) -> float:
        dense = sum(
            (item.coefficient * item.left @ item.right.T for item in updates),
            start=torch.zeros((2, 3), dtype=torch.float64),
        )
        return float(torch.trace(dense @ self.covariance @ dense.T))

    def test_cumulative_p_has_prior_cross_and_dense_parity(self) -> None:
        state = CumulativeStructuralPState(self.layers)
        first = {layer: self.update(f"first-{layer}") for layer in self.layers}
        state.finalize(first, (1.0, 0.5))
        second = {layer: self.update(f"second-{layer}") for layer in self.layers}
        state.finalize(second, (0.25, 0.75))
        candidate = {layer: self.update(f"candidate-{layer}") for layer in self.layers}
        risk = state.candidate_risk(candidate)
        velocity = np.asarray((0.7, 1.2), dtype=np.float64)
        predicted = risk.offset + risk.linear @ velocity + velocity @ risk.gram @ velocity
        dense = 0.0
        for index, layer in enumerate(self.layers):
            dense += self.dense_value(
                state.contributions[layer]
                + [candidate[layer].scaled(FIXED_H * float(velocity[index]))]
            )
        self.assertAlmostEqual(float(predicted), dense, places=10)
        self.assertTrue(np.any(np.abs(risk.linear) > 0.0))
        self.assertNotEqual(state.value(), sum(state.committed_load.values()))

    def test_incremental_h_is_only_candidate_movement(self) -> None:
        factors = {
            4: torch.tensor(((1.0, 2.0),), dtype=torch.float64),
            5: torch.tensor(((2.0, -1.0),), dtype=torch.float64),
        }
        keys = {
            4: torch.tensor(((1.0, 0.0), (0.5, 1.0)), dtype=torch.float64),
            5: torch.tensor(((0.0, 1.0), (1.0, 1.0)), dtype=torch.float64),
        }
        risk = incremental_structural_h(factors, keys, layer_order=self.layers)
        velocity = np.asarray((0.2, 0.8))
        direct = 0.0
        for index, layer in enumerate(self.layers):
            movement = FIXED_H * velocity[index] * (factors[layer] @ keys[layer])
            direct += float(torch.sum(movement * movement))
        self.assertEqual(risk.offset, 0.0)
        self.assertTrue(np.array_equal(risk.linear, np.zeros(2)))
        self.assertAlmostEqual(float(velocity @ risk.gram @ velocity), direct, places=14)


class ExactStrengthRouterTest(unittest.TestCase):
    def risks(self) -> tuple:
        from project.run_scripts.ode_bf.p1r29_sequential_preparation import QuadraticRoutingRisk

        h = QuadraticRoutingRisk(0.0, np.zeros(3), np.diag((10.0, 0.0, 0.0)), 10.0, "H")
        p = QuadraticRoutingRisk(0.0, np.zeros(3), np.diag((0.0, 10.0, 0.0)), 10.0, "P")
        return h, p

    def test_soft_changes_allocation_without_changing_strength(self) -> None:
        h, p = self.risks()
        neutral = solve_exact_strength_soft_hp(
            (1.0, 2.0, 3.0), 3.0, np.eye(3), np.eye(3), h, p, arm="NEUTRAL"
        )
        soft = solve_exact_strength_soft_hp(
            (1.0, 2.0, 3.0), 3.0, np.eye(3), np.eye(3), h, p, arm="SOFT"
        )
        self.assertLessEqual(soft.exact_strength_residual, 1.0e-8)
        self.assertEqual(neutral.requested_strength, soft.requested_strength)
        self.assertAlmostEqual(neutral.predicted_strength, soft.predicted_strength, places=8)
        self.assertTrue(soft.allocation_influence)
        self.assertEqual(soft.routing_dof, 2)
        self.assertLessEqual(soft.selected_energy, neutral.neutral_energy * (1 + 1e-8) + 1e-12)

    def test_numerical_failure_is_same_strength_neutral_fallback(self) -> None:
        h, p = self.risks()
        result = solve_exact_strength_soft_hp(
            (1.0, 2.0, 3.0),
            3.0,
            np.eye(3),
            np.eye(3),
            h,
            p,
            arm="SOFT",
            inject_numerical_failure=True,
        )
        self.assertEqual(result.status, "SOFT_NUMERICAL_NEUTRAL_FALLBACK")
        self.assertTrue(result.fallback_to_neutral)
        self.assertEqual(result.retry_count + result.backtracking_count, 0)
        self.assertLessEqual(result.exact_strength_residual, 1.0e-8)

    def test_one_active_direction_is_no_dof_not_forced_influence(self) -> None:
        h, p = self.risks()
        result = solve_exact_strength_soft_hp(
            (0.0, 2.0, -1.0), 1.0, np.eye(3), np.eye(3), h, p, arm="SOFT"
        )
        self.assertEqual(result.status, "NO_ROUTING_DOF")
        self.assertFalse(result.allocation_influence)


class SequentialTransactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = (4, 5)
        self.arm = SequentialArmState("neutral", self.layers)
        weights = {layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in self.layers}
        self.arm.set_initial_weights(weights)

    def prospective(self, *, version: int, prefix: str, collision_offset: int = 0):
        keys = {layer: torch.full((3, 10), float(version), dtype=torch.float32) for layer in self.layers}
        return self.arm.ledger.prospective(
            transaction_id=f"tx-{prefix}",
            expected_version=version - 1,
            records=history_records(version, prefix, collision_offset),
            solve_keys_by_layer=keys,
            risk_keys_by_layer={layer: keys[layer] * 2 for layer in self.layers},
        )

    def test_append_after_verified_commit_once_and_rollback_zero(self) -> None:
        prospective = self.prospective(version=1, prefix="one")
        self.assertEqual(self.arm.history_entry_count(), 0)
        self.assertIsNone(
            self.arm.finalize_history_after_commit(
                prospective,
                post_commit_verified=False,
                load_increment_by_layer={4: 1.0, 5: 1.0},
            )
        )
        self.assertEqual(self.arm.history_entry_count(), 0)
        receipt = self.arm.finalize_history_after_commit(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0},
        )
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.appended_count, 10)
        replay = self.arm.finalize_history_after_commit(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0},
        )
        self.assertEqual(replay.appended_count, 0)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(self.arm.history_entry_count(), 10)

    def test_collision_removes_registry_but_cumulative_load_never_decrements(self) -> None:
        first = self.prospective(version=1, prefix="one")
        self.arm.finalize_history_after_commit(
            first, post_commit_verified=True, load_increment_by_layer={4: 3.0, 5: 4.0}
        )
        before = self.arm.ledger.cumulative_load()
        second = self.prospective(version=2, prefix="two", collision_offset=0)
        receipt = self.arm.finalize_history_after_commit(
            second, post_commit_verified=True, load_increment_by_layer={4: 1.0, 5: 1.0}
        )
        self.assertEqual(receipt.obsolete_count, 10)
        self.assertEqual(self.arm.history_entry_count(), 10)
        after = self.arm.ledger.cumulative_load()
        self.assertEqual(after, {4: before[4] + 1.0, 5: before[5] + 1.0})

    def test_terminal_key_reuse_and_explicit_recapture(self) -> None:
        keys = {layer: torch.ones((3, 10), dtype=torch.float32) for layer in self.layers}
        projectors = {layer: torch.eye(3, dtype=torch.float32) for layer in self.layers}
        raw, projected, receipt = self.arm.stage_terminal_keys(
            keys,
            projectors,
            terminal_virtual_weight_sha256=digest("state"),
            endpoint_weight_sha256=digest("state"),
        )
        self.assertTrue(receipt.reused_terminal_keys)
        self.assertEqual(receipt.recapture_count, 0)
        self.assertTrue(torch.equal(raw[4], projected[4]))
        _, _, recaptured = self.arm.stage_terminal_keys(
            keys,
            projectors,
            terminal_virtual_weight_sha256=digest("virtual"),
            endpoint_weight_sha256=digest("physical"),
            recapture=lambda: {layer: keys[layer] * 2 for layer in self.layers},
        )
        self.assertTrue(recaptured.explicit_recapture_fallback)
        self.assertEqual(recaptured.recapture_count, 1)

    def test_arm_weight_states_are_independent_and_unverified_commit_is_noop(self) -> None:
        other = SequentialArmState("soft", self.layers)
        other.set_initial_weights({layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in self.layers})
        changed = {layer: torch.ones((2, 2), dtype=torch.bfloat16) for layer in self.layers}
        self.arm.replace_persistent_weights_after_verified_commit(changed, verified=False)
        self.assertEqual(float(self.arm.persistent_weights[4].sum()), 0.0)
        self.arm.replace_persistent_weights_after_verified_commit(changed, verified=True)
        self.assertEqual(float(self.arm.persistent_weights[4].sum()), 4.0)
        self.assertEqual(float(other.persistent_weights[4].sum()), 0.0)


class OuterPlanTest(unittest.TestCase):
    def test_plan_round1_equivalence_and_cpu_firewall(self) -> None:
        plan = build_b10x10_outer_plan()
        self.assertEqual([item.history_count_at_entry for item in plan], list(range(0, 100, 10)))
        self.assertEqual(sum(item.round_index for item in plan), 55)
        self.assertTrue(all(item.debt_reset_value == 0.0 for item in plan))
        payload = stage_a_dry_plan()
        self.assertEqual(payload["status"], "PREPARED_WAITING_P1R29_ATOMIC_GATE")
        self.assertEqual(
            payload["models"] + payload["gpu_allocations"] + payload["slurm_jobs"] + payload["result_roots"],
            0,
        )
        self.assertEqual(payload["server1_janghj_gpu_cap"], 4)
        self.assertEqual(payload["server2_janghj_gpu_cap_independent"], 4)
        atomic = {"scientific": 1, "endpoint": digest("endpoint")}
        sequential = {**atomic, "wrapper": "sequential"}
        identity = assert_round1_empty_history_equivalence(
            atomic, sequential, wrapper_only_keys=("wrapper",)
        )
        self.assertEqual(identity, digest('{"endpoint":"' + digest("endpoint") + '","scientific":1}'))

    def test_outer_skeleton_rejects_inner_evaluator_and_requires_k8(self) -> None:
        arm = SequentialArmState("neutral", (4, 5))
        arm.set_initial_weights({layer: torch.zeros((1, 1), dtype=torch.bfloat16) for layer in (4, 5)})
        skeleton = SequentialOuterRuntimeSkeleton(arm)
        round_plan = skeleton.begin_round(1)
        self.assertEqual(round_plan.history_count_at_entry, 0)
        with self.assertRaisesRegex(ODEBFContractError, "inaccessible"):
            skeleton.record_inner_evaluator_access()
        for _ in range(8):
            skeleton.record_accepted_step()
        keys = {layer: torch.zeros((2, 10), dtype=torch.float32) for layer in (4, 5)}
        prospective = arm.ledger.prospective(
            transaction_id="round1",
            expected_version=0,
            records=history_records(1, "round1"),
            solve_keys_by_layer=keys,
            risk_keys_by_layer=keys,
        )
        receipt = arm.finalize_history_after_commit(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={4: 0.0, 5: 0.0},
        )
        self.assertEqual(skeleton.close_round_after_history_finalize(receipt), 1)
        self.assertEqual(arm.completed_rounds, 1)

    def test_history_bound_is_100(self) -> None:
        self.assertEqual(MAXIMUM_HISTORY_RECORDS, 100)


if __name__ == "__main__":
    unittest.main()
