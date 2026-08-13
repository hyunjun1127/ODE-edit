from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_state import P1HistoryRecord
from project.run_scripts.ode_bf.p1r29_sequential_preparation import (
    FIXED_H,
    MAXIMUM_HISTORY_RECORDS,
    AtomicAdapterPreprocessingIdentity,
    AtomicSequentialAdapterFrame,
    AtomicSequentialTerminalFrame,
    CumulativeStructuralPState,
    LowRankUpdate,
    RoundStructuralHState,
    SequentialArmState,
    SequentialOuterRuntimeSkeleton,
    TerminalPhysicalKeyIdentity,
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
                    active_record_sha256=digest(f"active-{columns}"),
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
                    active_record_sha256=digest(f"active-{columns}"),
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
                active_record_sha256=digest("active-drift"),
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
                    active_record_sha256=digest(f"active-{alias}"),
                    regularization=0.5,
                    projector_certificate=self.certificate,
                    cached_factorization=None,
                )
                self.assertEqual(q.shape, (dimension, 10))
                self.assertTrue(receipt.parity_passed)

    def test_collision_version_uses_active_count_and_cache_key_digest(self) -> None:
        current = torch.randn(self.dimension, 10, dtype=torch.float64)
        raw = torch.randn(self.dimension, 10, dtype=torch.float64)
        projected = self.projector @ raw
        _, factor, receipt = solve_alpha_woodbury_cached(
            self.projector,
            current,
            raw,
            projected,
            history_version=2,
            active_record_sha256=digest("version2-active10-after-collision"),
            regularization=0.25,
            projector_certificate=self.certificate,
            cached_factorization=None,
        )
        self.assertEqual(factor.history_version, 2)
        self.assertEqual(factor.history_columns, 10)
        self.assertEqual(factor.active_record_sha256, digest("version2-active10-after-collision"))
        self.assertEqual(len(receipt.cache_key_sha256), 64)

    def test_stale_cache_falls_back_to_exact_backend(self) -> None:
        current = torch.randn(self.dimension, 10, dtype=torch.float64)
        raw = torch.randn(self.dimension, 10, dtype=torch.float64)
        projected = self.projector @ raw
        stale = build_history_factorization(
            self.projector,
            raw,
            projected,
            history_version=1,
            active_record_sha256=digest("stale-active"),
            regularization=0.25,
        )
        q, refreshed, receipt = solve_alpha_woodbury_cached(
            self.projector,
            current,
            raw,
            projected,
            history_version=1,
            active_record_sha256=digest("current-active"),
            regularization=0.25,
            projector_certificate=self.certificate,
            cached_factorization=stale,
        )
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(receipt.cache_mismatch_exact_backend_fallback)
        self.assertFalse(receipt.history_cache_hit)
        self.assertEqual(refreshed.active_record_sha256, digest("current-active"))


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
            4: torch.tensor(((1.0, 0.0), (0.5, 1.0)), dtype=torch.float64).repeat(1, 5),
            5: torch.tensor(((0.0, 1.0), (1.0, 1.0)), dtype=torch.float64).repeat(1, 5),
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

    def test_round_cumulative_h_dense_parity_with_signed_cross(self) -> None:
        keys = {
            layer: torch.randn(3, 10, dtype=torch.float64) for layer in self.layers
        }
        first = {
            layer: torch.randn(2, 3, dtype=torch.float64) for layer in self.layers
        }
        second = {layer: -first[layer] for layer in self.layers}
        state = RoundStructuralHState(self.layers, 1, 10)
        first_velocity = np.asarray((0.8, 0.5))
        state.accept(first, keys, first_velocity)
        risk = state.candidate_risk(second, keys)
        second_velocity = np.asarray((0.3, 0.9))
        self.assertLess(float(risk.linear @ second_velocity), 0.0)
        predicted = risk.raw_value(second_velocity)
        dense = 0.0
        for index, layer in enumerate(self.layers):
            direct = (
                FIXED_H * first_velocity[index] * (first[layer] @ keys[layer])
                + FIXED_H * second_velocity[index] * (second[layer] @ keys[layer])
            )
            dense += float(torch.sum(direct * direct))
        self.assertAlmostEqual(predicted, dense, places=13)
        self.assertGreater(risk.offset, 0.0)
        self.assertGreater(float(second_velocity @ risk.gram @ second_velocity), 0.0)

    def test_round1_empty_history_h_is_exact_zero(self) -> None:
        state = RoundStructuralHState(self.layers, 0, 0)
        keys = {layer: torch.empty((3, 0), dtype=torch.float64) for layer in self.layers}
        factors = {layer: torch.randn(2, 3, dtype=torch.float64) for layer in self.layers}
        risk = state.candidate_risk(factors, keys)
        self.assertEqual(risk.raw_value(np.asarray((100.0, 200.0))), 0.0)

    def test_p_prior_scalar_avoids_pairwise_replay(self) -> None:
        state = CumulativeStructuralPState(self.layers)
        for step in range(50):
            state.finalize(
                {layer: self.update(f"prior-{step}-{layer}") for layer in self.layers},
                (0.5, 0.75),
            )
        before_calls = state.batched_cross_call_count
        before_replay = state.prior_pairwise_replay_count
        risk = state.candidate_risk(
            {layer: self.update(f"query-{layer}") for layer in self.layers}
        )
        self.assertTrue(math.isfinite(risk.offset))
        self.assertEqual(state.batched_cross_call_count - before_calls, len(self.layers))
        self.assertEqual(state.prior_pairwise_replay_count, before_replay)
        self.assertEqual(before_replay, 0)


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

    def test_constant_p_prior_is_telemetry_not_minimax_input(self) -> None:
        from project.run_scripts.ode_bf.p1r29_sequential_preparation import QuadraticRoutingRisk

        h, p = self.risks()
        p_large = QuadraticRoutingRisk(
            1.0e9, p.linear.copy(), p.gram.copy(), p.normalization, "P-large-prior"
        )
        base = solve_exact_strength_soft_hp(
            (1.0, 2.0, 3.0), 3.0, np.eye(3), np.eye(3), h, p, arm="SOFT"
        )
        large = solve_exact_strength_soft_hp(
            (1.0, 2.0, 3.0), 3.0, np.eye(3), np.eye(3), h, p_large, arm="SOFT"
        )
        self.assertTrue(np.allclose(base.velocity, large.velocity, rtol=0.0, atol=1e-8))
        self.assertAlmostEqual(base.p_marginal_delta, large.p_marginal_delta, places=8)
        self.assertNotEqual(base.p_raw_cumulative, large.p_raw_cumulative)
        self.assertAlmostEqual(base.predicted_strength, large.predicted_strength, places=8)


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

    def staging(self):
        projected = {
            layer: self.arm.ledger.risk_keys(layer) for layer in self.layers
        }
        staging = self.arm.begin_round_staging(projected)
        for step in range(8):
            candidates = {
                layer: LowRankUpdate.build(
                    torch.full((2, 2), 0.1 * (step + 1), dtype=torch.float64),
                    torch.full((3, 2), 0.05 * (layer + step), dtype=torch.float64),
                    coefficient=1.0,
                    covariance_action=lambda value: value,
                )
                for layer in self.layers
            }
            factors = {
                layer: torch.full((2, 3), 0.01 * (layer + step), dtype=torch.float64)
                for layer in self.layers
            }
            staging.stage_accepted_step(candidates, factors, (0.5, 0.75))
        return staging

    def commit(self, *, version: int, prefix: str, collision_offset: int = 0, fault_phase=None):
        staging = self.staging()
        prospective = self.prospective(
            version=version, prefix=prefix, collision_offset=collision_offset
        )
        weights = {
            layer: torch.full((2, 2), float(version), dtype=torch.bfloat16)
            for layer in self.layers
        }
        return self.arm.commit_verified_round(
            f"tx-{prefix}",
            prospective,
            staging,
            weights,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0},
            fault_phase=fault_phase,
        ), staging, prospective, weights

    def test_append_after_verified_commit_once_and_rollback_zero(self) -> None:
        self.assertEqual(self.arm.history_entry_count(), 0)
        self.arm.factorization_cache["precommit-cache"] = None  # opaque cache sentinel
        staging = self.staging()
        prospective = self.prospective(version=1, prefix="one")
        weights = {layer: torch.ones((2, 2), dtype=torch.bfloat16) for layer in self.layers}
        before = self.arm.state_identity()
        self.assertIsNone(
            self.arm.commit_verified_round(
                "tx-one",
                prospective,
                staging,
                weights,
                post_commit_verified=False,
                load_increment_by_layer={4: 1.0, 5: 1.0},
            )
        )
        self.assertEqual(self.arm.state_identity(), before)
        self.assertEqual(self.arm.history_entry_count(), 0)
        receipt = self.arm.commit_verified_round(
            "tx-one",
            prospective,
            staging,
            weights,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0},
        )
        self.assertEqual(receipt.appended_count, 10)
        self.assertTrue(receipt.cache_invalidated)
        self.assertEqual(self.arm.factorization_cache, {})
        replay = self.arm.commit_verified_round(
            "tx-one",
            prospective,
            staging,
            weights,
            post_commit_verified=True,
            load_increment_by_layer={4: 1.0, 5: 2.0},
        )
        self.assertEqual(replay.appended_count, 0)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(self.arm.history_entry_count(), 10)

    def test_all_component_fault_injection_rolls_back_exactly(self) -> None:
        for phase in (
            "after_weights",
            "after_history",
            "after_p",
            "after_h",
            "after_cache",
            "after_round",
        ):
            with self.subTest(phase=phase):
                arm = SequentialArmState(f"neutral-{phase}", self.layers)
                arm.set_initial_weights(
                    {layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in self.layers}
                )
                arm.factorization_cache["rollback-cache"] = None  # opaque cache sentinel
                self.arm = arm
                before = arm.state_identity()
                with self.assertRaisesRegex(RuntimeError, "injected backend transaction"):
                    self.commit(version=1, prefix=phase, fault_phase=phase)
                self.assertEqual(arm.state_identity(), before)
                self.assertEqual(arm.completed_rounds, 0)
                self.assertEqual(arm.history_entry_count(), 0)
                self.assertEqual(arm.cumulative_p.value(), 0.0)

    def test_collision_removes_registry_but_cumulative_load_never_decrements(self) -> None:
        first, _, _, _ = self.commit(version=1, prefix="one")
        self.assertEqual(first.appended_count, 10)
        before = self.arm.ledger.cumulative_load()
        before_p = self.arm.cumulative_p.value()
        second, _, _, _ = self.commit(version=2, prefix="two", collision_offset=0)
        self.assertEqual(second.appended_count, 10)
        self.assertEqual(self.arm.history_entry_count(), 10)
        self.assertEqual(self.arm.ledger.version, 2)
        after = self.arm.ledger.cumulative_load()
        self.assertEqual(after, {4: before[4] + 1.0, 5: before[5] + 2.0})
        self.assertGreater(self.arm.cumulative_p.value(), before_p)
        projector = 2.0 * torch.eye(3, dtype=torch.float32)
        certificate = ProjectorCertificate(
            digest("collision-projector"), 0.0, 2.0, "artifact-unverified", 1e-10
        )
        _, solve_receipt = self.arm.solve_historical_alpha(
            4,
            projector,
            torch.ones((3, 10), dtype=torch.float32),
            regularization=0.5,
            projector_certificate=certificate,
        )
        self.assertEqual(solve_receipt.history_version, 2)
        self.assertEqual(solve_receipt.history_columns, 10)

    def identity(self, suffix: str = "same") -> TerminalPhysicalKeyIdentity:
        return TerminalPhysicalKeyIdentity(
            digest(f"weight-{suffix}"),
            digest(f"order-{suffix}"),
            digest(f"normalization-{suffix}"),
            digest(f"layers-{suffix}"),
            digest(f"capture-{suffix}"),
            digest(f"physical-{suffix}"),
        )

    def test_terminal_key_reuse_requires_all_six_identities(self) -> None:
        keys = {layer: torch.ones((3, 10), dtype=torch.float32) for layer in self.layers}
        projectors = {layer: torch.eye(3, dtype=torch.float32) for layer in self.layers}
        same = self.identity()
        raw, projected, receipt = self.arm.stage_terminal_keys(
            keys,
            projectors,
            captured_identity=same,
            committed_identity=same,
        )
        self.assertTrue(receipt.reused_terminal_keys)
        self.assertEqual(receipt.recapture_count, 0)
        self.assertTrue(torch.equal(raw[4], projected[4]))
        fields = tuple(TerminalPhysicalKeyIdentity.__dataclass_fields__)
        for count, field_name in enumerate(fields, start=1):
            values = same.__dict__ if hasattr(same, "__dict__") else {
                field: getattr(same, field) for field in fields
            }
            changed = dict(values)
            changed[field_name] = digest(f"mismatch-{field_name}")
            _, _, recaptured = self.arm.stage_terminal_keys(
                keys,
                projectors,
                captured_identity=TerminalPhysicalKeyIdentity(**changed),
                committed_identity=same,
                recapture=lambda: {layer: keys[layer] * 2 for layer in self.layers},
            )
            self.assertTrue(recaptured.explicit_recapture_fallback)
            self.assertEqual(recaptured.recapture_count, count)

    def test_arm_weight_states_are_independent(self) -> None:
        other = SequentialArmState("soft", self.layers)
        other.set_initial_weights({layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in self.layers})
        self.commit(version=1, prefix="independent")
        self.assertEqual(float(self.arm.persistent_weights[4].sum()), 4.0)
        self.assertEqual(float(other.persistent_weights[4].sum()), 0.0)


class AdapterFirewallTest(unittest.TestCase):
    def test_adapter_owns_preprocessing_and_backend_adds_no_model_calls(self) -> None:
        layers = (4, 5)
        preprocessing = AtomicAdapterPreprocessingIdentity(
            "future-atomic-adapter",
            digest("preprocess"),
            digest("order"),
            digest("tokenizer-target-normalization"),
        )
        frame = AtomicSequentialAdapterFrame(
            preprocessing,
            layers,
            {layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in layers},
            torch.zeros((10, 3), dtype=torch.float32),
            {layer: torch.ones((2, 3), dtype=torch.float32) for layer in layers},
            (1.0, 2.0),
            2.5,
            {layer: torch.ones((3, 10), dtype=torch.float32) for layer in layers},
            digest("accepted-state"),
        )
        receipt = frame.raw_free_receipt()
        self.assertFalse(receipt["target_or_debt_preprocessing_implemented_by_backend"])
        self.assertTrue(receipt["shared_preprocessing_owned_by_selected_adapter"])
        self.assertEqual(
            receipt["backend_preprocessing_model_forward_count"]
            + receipt["backend_preprocessing_backward_count"]
            + receipt["backend_h_p_model_forward_count"]
            + receipt["backend_h_p_backward_count"],
            0,
        )
        self.assertEqual(receipt["semantic_rho"], 2.5)

        physical = TerminalPhysicalKeyIdentity(
            digest("weight"),
            preprocessing.request_order_sha256,
            preprocessing.tokenizer_target_normalization_sha256,
            digest("layers"),
            digest("capture"),
            digest("hook-disabled-physical"),
        )
        terminal = AtomicSequentialTerminalFrame(
            preprocessing,
            {layer: torch.ones((3, 10), dtype=torch.float32) for layer in layers},
            physical,
        )
        arm = SequentialArmState("adapter", layers)
        arm.set_initial_weights(
            {layer: torch.zeros((2, 2), dtype=torch.bfloat16) for layer in layers}
        )
        _, _, terminal_receipt = arm.stage_adapter_terminal_frame(
            terminal,
            {layer: torch.eye(3, dtype=torch.float32) for layer in layers},
            committed_identity=physical,
        )
        self.assertTrue(terminal_receipt.reused_terminal_keys)

    def test_adapter_rejects_missing_layer_and_invalid_rho(self) -> None:
        preprocessing = AtomicAdapterPreprocessingIdentity(
            "future-atomic-adapter",
            digest("preprocess"),
            digest("order"),
            digest("normalization"),
        )
        with self.assertRaises(ODEBFContractError):
            AtomicSequentialAdapterFrame(
                preprocessing,
                (4, 5),
                {4: torch.zeros((2, 2), dtype=torch.bfloat16)},
                torch.zeros((10, 3), dtype=torch.float32),
                {4: torch.ones((2, 3), dtype=torch.float32)},
                (1.0, 2.0),
                -1.0,
                {4: torch.ones((3, 10), dtype=torch.float32)},
                digest("state"),
            )


class OuterPlanTest(unittest.TestCase):
    def test_plan_round1_equivalence_and_cpu_firewall(self) -> None:
        plan = build_b10x10_outer_plan()
        self.assertEqual([item.history_count_at_entry for item in plan], list(range(0, 100, 10)))
        self.assertEqual(sum(item.round_index for item in plan), 55)
        self.assertTrue(all(item.debt_reset_value == 0.0 for item in plan))
        payload = stage_a_dry_plan()
        self.assertEqual(payload["status"], "BACKEND_HARDENED_WAITING_VIABLE_ATOMIC_ADAPTER")
        self.assertEqual(payload["sequential_backend_implementation"], "CONTINUE_COMPLETE_MODEL_FREE")
        self.assertEqual(
            payload["scientific_sequential_execution"], "HOLD_UNTIL_VIABLE_ATOMIC_ADAPTER"
        )
        self.assertEqual(
            payload["actual_b10x2_model_smoke_receipt"],
            "NOT_RECORDED_NO_ADAPTER_SELECTED",
        )
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
        staging = arm.begin_round_staging(
            {layer: torch.empty((2, 0), dtype=torch.float32) for layer in (4, 5)}
        )
        for step in range(8):
            candidates = {
                layer: LowRankUpdate.build(
                    torch.ones((1, 1), dtype=torch.float64),
                    torch.ones((2, 1), dtype=torch.float64),
                    coefficient=1.0,
                    covariance_action=lambda value: value,
                )
                for layer in (4, 5)
            }
            staging.stage_accepted_step(
                candidates,
                {layer: torch.ones((1, 2), dtype=torch.float64) for layer in (4, 5)},
                (0.5, 0.5),
            )
        receipt = arm.commit_verified_round(
            "round1",
            prospective,
            staging,
            {layer: torch.ones((1, 1), dtype=torch.bfloat16) for layer in (4, 5)},
            post_commit_verified=True,
            load_increment_by_layer={4: 0.0, 5: 0.0},
        )
        self.assertEqual(skeleton.close_round_after_history_finalize(receipt), 1)
        self.assertEqual(arm.completed_rounds, 1)

    def test_history_bound_is_100(self) -> None:
        self.assertEqual(MAXIMUM_HISTORY_RECORDS, 100)


if __name__ == "__main__":
    unittest.main()
