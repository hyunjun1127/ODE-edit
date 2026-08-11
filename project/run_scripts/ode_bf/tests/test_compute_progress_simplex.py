from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.compute_progress_simplex_runtime import (
    COMPUTE_FIELD_MODEL_FORWARD_CEILING,
    COMPUTE_TOKEN_BUDGET,
    FixedRankHistoricalSketch,
    empty_historical_sketch_receipt,
    rotating_context_ordinals,
    validate_field_forward_count,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.progress_simplex_routing import (
    StructuralOnlyRoutingInventory,
    solve_progress_simplex_routing,
)
from project.run_scripts.ode_bf import progress_simplex_routing
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    build_token_budget_streaming_batch_plan,
)
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    P1R23_COMPUTE_A1_TECH_R2_ATTEMPT,
    P1R23_COMPUTE_A1_TECH_R3_ATTEMPT,
    expected_p1r23_result_name,
)


ROOT = Path(__file__).resolve().parents[4]


def _problem() -> RoutingProblem:
    zeros = np.zeros(5, dtype=np.float64)
    return RoutingProblem(
        np.asarray((0.5, 0.3, 0.2, 0.1, 0.05)),
        np.diag((1.0, 1.1, 1.2, 1.3, 1.4)),
        np.zeros((5, 5)),
        1.0,
        np.ones(5),
        1.0e-6,
        1.0e-8,
        QuadraticBarrier("historical", 0.0, zeros, np.zeros((5, 5)), 1.0, "layer-local-diagonal"),
        QuadraticBarrier(
            "pretrained",
            0.0,
            np.asarray((2.0, 0.1, 0.1, 0.1, 0.1)),
            np.diag((0.8, 0.01, 0.01, 0.01, 0.01)),
            1.0,
            "layer-local-diagonal",
        ),
    )


class ComputeProgressSimplexTests(unittest.TestCase):
    @staticmethod
    def _boundary_problem() -> RoutingProblem:
        zeros = np.zeros(5, dtype=np.float64)
        return RoutingProblem(
            np.asarray((0.5, 0.3, 0.2, 0.0, -0.1)),
            np.diag((0.7, 0.9, 1.1, 1.3, 1.5)),
            np.diag((0.5, 0.6, 0.7, 0.8, 0.9)),
            1.0,
            np.ones(5),
            1.0e-6,
            1.0e-8,
            QuadraticBarrier("historical", 0.0, zeros, np.zeros((5, 5)), 1.0, "layer-local-diagonal"),
            QuadraticBarrier(
                "pretrained",
                0.0,
                np.asarray((2.0, 0.01, 0.01, 0.0, 0.0)),
                np.diag((0.8, 0.01, 0.01, 0.01, 0.01)),
                1.0,
                "layer-local-diagonal",
            ),
        )

    def test_structural_only_soft_changes_allocation_without_functional_score(self) -> None:
        inventory = StructuralOnlyRoutingInventory(0, "a" * 64, "b" * 64, "c" * 64)
        neutral = solve_progress_simplex_routing(
            _problem(), inventory, arm=FixedE8Arm.NEUTRAL, alpha_req=2.0
        )
        soft = solve_progress_simplex_routing(
            _problem(), inventory, arm=FixedE8Arm.SOFT, alpha_req=2.0
        )
        self.assertAlmostEqual(neutral.predicted_progress, soft.predicted_progress, places=8)
        self.assertGreater(soft.neutral_soft_coefficient_l2, 1.0e-7)
        self.assertLessEqual(soft.soft_energy_ratio, 1.0 + 1.0e-8 + 1.0e-10)
        payload = soft.raw_free_payload()
        self.assertEqual(payload["functional_routing_influence_count"], 0)
        self.assertEqual(
            [item["label"] for item in payload["scores"]],
            ["structural_historical", "structural_pretrained"],
        )

    def test_rotating_context_schedule_and_field_forward_ceiling(self) -> None:
        self.assertEqual([rotating_context_ordinals(k) for k in range(6)], [(0, 3), (1, 4), (2, 5)] * 2)
        self.assertEqual(validate_field_forward_count(target_forwards=5, slope_forwards=5, capture_forwards=5), 15)
        with self.assertRaises(ODEBFContractError):
            validate_field_forward_count(target_forwards=9, slope_forwards=8, capture_forwards=8)
        self.assertEqual(COMPUTE_FIELD_MODEL_FORWARD_CEILING, 24)

    def test_token_budget_partition_is_length_bucketed_complete_and_bounded(self) -> None:
        identities = tuple(f"{item:064x}" for item in range(1, 11))
        lengths = (120, 400, 200, 380, 180, 360, 140, 340, 160, 320)
        plan = build_token_budget_streaming_batch_plan(
            identities, lengths, token_budget=COMPUTE_TOKEN_BUDGET
        )
        flattened = [ordinal for batch in plan.batches for ordinal in batch.request_ordinals]
        self.assertEqual(set(flattened), set(range(10)))
        self.assertEqual(plan.token_budget, COMPUTE_TOKEN_BUDGET)
        for batch in plan.batches:
            self.assertLessEqual(
                len(batch.request_ordinals) * 6 * batch.maximum_unpadded_length,
                COMPUTE_TOKEN_BUDGET,
            )

    def test_h_sketch_stages_once_and_rolls_back_without_update(self) -> None:
        sketch = FixedRankHistoricalSketch(2, {4: 3, 5: 3})
        first = {4: np.eye(3), 5: np.ones((2, 3))}
        sketch.stage(first)
        sketch.finalize(transaction_committed=False)
        self.assertEqual(sketch.commit_count, 0)
        self.assertEqual(float(np.linalg.norm(sketch.gram(4))), 0.0)
        sketch.stage(first)
        sketch.finalize(transaction_committed=True)
        self.assertEqual(sketch.commit_count, 1)
        self.assertLessEqual(sketch.raw_free_payload()["layers"]["4"]["row_count"], 2)
        self.assertEqual(empty_historical_sketch_receipt((4, 5, 6, 7, 8))["decision_influence_count"], 0)

    def test_energy_boundary_recertifies_without_relaxing_frozen_contract(self) -> None:
        inventory = StructuralOnlyRoutingInventory(0, "a" * 64, "b" * 64, "c" * 64)
        routed = solve_progress_simplex_routing(
            self._boundary_problem(), inventory, arm=FixedE8Arm.SOFT, alpha_req=1.5
        )
        stage1 = routed.certificates[1]
        self.assertTrue(stage1.passed)
        self.assertEqual(stage1.fallback_invocation_count, 1)
        self.assertEqual(
            stage1.backend,
            "deterministic-convex-feasibility-restoration-float64",
        )
        self.assertEqual(
            stage1.failed_primary_certificate["first_false_component"],
            "neutral_relative_global_energy",
        )
        self.assertGreater(stage1.failed_primary_certificate["energy_violation"], 1.0e-12)
        self.assertGreater(stage1.feasibility_restoration_linf, 0.0)
        self.assertLessEqual(routed.soft_energy_ratio, 1.0 + 1.0e-8 + 1.0e-10)

        real_minimize = progress_simplex_routing.minimize

        def nonauthoritative_exit_flag(*args: object, **kwargs: object) -> object:
            result = real_minimize(*args, **kwargs)
            if kwargs.get("method") == "SLSQP":
                result.success = False
                result.status = 8
                result.message = "positive directional derivative telemetry"
            return result

        with mock.patch.object(
            progress_simplex_routing,
            "minimize",
            side_effect=nonauthoritative_exit_flag,
        ):
            flagged = solve_progress_simplex_routing(
                self._boundary_problem(),
                inventory,
                arm=FixedE8Arm.SOFT,
                alpha_req=1.5,
            )
        flagged_stage1 = flagged.certificates[1]
        self.assertTrue(flagged_stage1.passed)
        self.assertEqual(flagged_stage1.fallback_invocation_count, 1)
        self.assertFalse(flagged_stage1.failed_primary_certificate["success"])
        self.assertEqual(
            flagged_stage1.failed_primary_certificate["first_false_component"],
            "neutral_relative_global_energy",
        )

        def malformed_primary(*args: object, **kwargs: object) -> object:
            result = real_minimize(*args, **kwargs)
            if kwargs.get("method") == "SLSQP":
                result.x = np.asarray(result.x, dtype=np.float64)
                result.x[0] = -1.0
            return result

        with mock.patch.object(
            progress_simplex_routing, "minimize", side_effect=malformed_primary
        ):
            with self.assertRaises(ODEBFContractError):
                solve_progress_simplex_routing(
                    self._boundary_problem(),
                    inventory,
                    arm=FixedE8Arm.SOFT,
                    alpha_req=1.5,
                )

    def test_lock_roles_and_static_compute_boundary(self) -> None:
        lock, _ = load_rooted_json(
            ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_compute_progress_simplex.json",
            expected_schema="ode-edit-s05-p1r23-compute-progress-simplex-lock/v1",
        )
        self.assertEqual(lock["routing"]["functional_probe_per_step_count"], 0)
        self.assertEqual(lock["scientific_grid"]["early_stop_threshold_status"], "NOT_DEFINED")
        self.assertEqual(lock["streaming"]["field_model_forward_ceiling"], 24)
        names = {
            expected_p1r23_result_name(alias, batch_size=10, role=role)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for role in ("COMPUTE_PROGRESS_SIMPLEX_BG_PAIR", "COMPUTE_PROGRESS_SIMPLEX_RS_PAIR")
        }
        self.assertEqual(len(names), 4)
        source = inspect.getsource(solve_progress_simplex_routing)
        self.assertNotIn("model(", source)

    def test_attempt_namespace_is_exact_and_role_bound(self) -> None:
        expected = expected_p1r23_result_name(
            "llama3-8b-inst",
            batch_size=10,
            role="COMPUTE_PROGRESS_SIMPLEX_BG_PAIR",
            attempt_namespace=P1R23_COMPUTE_A1_TECH_R2_ATTEMPT,
        )
        self.assertEqual(
            expected,
            "s05-p1r23-b10-llama3-8b-inst-compute-progress-simplex-"
            "bg-neutral-soft-pair-compute-a1-tech-r2-v1",
        )
        for stale in ("compute-a1-v1", "compute-a1-tech-r1-v1", "tech-r2"):
            with self.assertRaisesRegex(ODEBFContractError, "attempt namespace"):
                expected_p1r23_result_name(
                    "llama3-8b-inst",
                    batch_size=10,
                    role="COMPUTE_PROGRESS_SIMPLEX_BG_PAIR",
                    attempt_namespace=stale,
                )
        with self.assertRaisesRegex(ODEBFContractError, "attempt namespace"):
            expected_p1r23_result_name(
                "qwen2.5-7b-inst",
                batch_size=10,
                role="PROGRESS_SIMPLEX_BG_PAIR",
                attempt_namespace=P1R23_COMPUTE_A1_TECH_R2_ATTEMPT,
            )
        self.assertTrue(
            expected_p1r23_result_name(
                "llama3-8b-inst",
                batch_size=10,
                role="COMPUTE_PROGRESS_SIMPLEX_BG_PAIR",
                attempt_namespace=P1R23_COMPUTE_A1_TECH_R3_ATTEMPT,
            ).endswith("compute-a1-tech-r3-v1")
        )


if __name__ == "__main__":
    unittest.main()
