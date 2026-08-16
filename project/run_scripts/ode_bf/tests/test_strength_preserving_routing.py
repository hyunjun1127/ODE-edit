from __future__ import annotations

import inspect
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Clock,
    FixedE8Arm,
    FixedE8SoftInventory,
    FunctionalBasisMetric,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.strength_preserving_routing import (
    STRENGTH_EQUALITY_TOLERANCE,
    StrengthPreservingStatus,
    route_independent_target_write_identity,
    solve_strength_preserving_routing,
    target_probe_requested_strength,
)


def _problem(
    slopes: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1, -0.1),
) -> RoutingProblem:
    size = len(slopes)
    return RoutingProblem(
        np.asarray(slopes, dtype=np.float64),
        np.diag(np.asarray((1.0, 1.2, 1.4, 1.6, 1.8))),
        np.eye(size),
        1.0,
        np.ones(size),
        1.0e-6,
        1.0e-8,
        QuadraticBarrier(
            "historical",
            0.0,
            np.zeros(size),
            np.diag(np.asarray((0.01, 0.02, 0.03, 0.04, 0.05))),
            1.0,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            0.0,
            np.asarray((0.20, 0.01, 0.01, 0.01, 0.01)),
            np.diag(np.asarray((0.20, 0.01, 0.01, 0.01, 0.01))),
            1.0,
            "layer-local-diagonal",
        ),
    )


def _inventory() -> FixedE8SoftInventory:
    reason = "INACTIVE_EMPTY_HISTORY"
    return FixedE8SoftInventory(
        FunctionalBasisMetric(
            "functional_p", 0.0, (0.3, 0.01, 0.01, 0.01, 0.01), True
        ),
        FunctionalBasisMetric(
            "functional_h_mean", 0.0, (0.0,) * 5, False, reason
        ),
        FunctionalBasisMetric(
            "functional_h_smoothmax", 0.0, (0.0,) * 5, False, reason
        ),
        0,
        "a" * 64,
        "b" * 64,
    )


def _target_receipt(*, allocation: str) -> dict[str, object]:
    scale = {
        "scale_identity_sha256": "c" * 64,
        "allocation": allocation,
        "shared_speed": 2.0,
        "gradient_norm_by_request": [1.0] * 10,
        "gradient_frobenius_norm": 4.0,
    }
    alpha = (
        2.5
        if allocation == "PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED"
        else 0.25 * np.sqrt(10.0) * 4.0
    )
    return {
        "velocity_coefficient": [0.0] * 5,
        "target_probe_applied_coefficient": [0.0] * 5,
        "scale_velocity": scale,
        "identity_sha256": "d" * 64,
        "velocity_sha256": "e" * 64,
        "target_new_nll_applied_step_reduction": alpha,
    }


class StrengthPreservingRoutingTests(unittest.TestCase):
    def test_common_zero_probe_requested_strength_has_one_h(self) -> None:
        rs, rs_receipt = target_probe_requested_strength(
            _target_receipt(allocation="PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED")
        )
        bg, bg_receipt = target_probe_requested_strength(
            _target_receipt(allocation="BATCH_GLOBAL_MATCHED_TOTAL_FROBENIUS_SPEED")
        )
        self.assertEqual(rs, 2.5)
        self.assertAlmostEqual(bg, 0.25 * np.sqrt(10.0) * 4.0)
        self.assertEqual(rs_receipt["h_application_count"], 1)
        self.assertEqual(rs_receipt["additional_target_graph_count"], 0)
        self.assertEqual(bg_receipt["target_probe_arm_dependent_influence_count"], 0)

    def test_route_dependent_target_probe_fails_closed(self) -> None:
        receipt = _target_receipt(
            allocation="PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED"
        )
        receipt["velocity_coefficient"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ODEBFContractError):
            target_probe_requested_strength(receipt)

    def test_neutral_and_soft_preserve_identical_strength(self) -> None:
        problem = _problem()
        inventory = _inventory()
        neutral = solve_strength_preserving_routing(
            problem, inventory, arm=FixedE8Arm.NEUTRAL, alpha_req=0.35
        )
        soft = solve_strength_preserving_routing(
            problem, inventory, arm=FixedE8Arm.SOFT, alpha_req=0.35
        )
        for observed in (neutral, soft):
            self.assertLessEqual(observed.equality_residual, STRENGTH_EQUALITY_TOLERANCE)
            self.assertAlmostEqual(
                np.dot(observed.signed_slopes, observed.velocity), 0.35, places=8
            )
            self.assertEqual(observed.active_direction_mask[-1], False)
            self.assertEqual(observed.velocity[-1], 0.0)
        self.assertAlmostEqual(
            np.dot(neutral.signed_slopes, neutral.velocity),
            np.dot(soft.signed_slopes, soft.velocity),
            places=8,
        )
        self.assertGreater(soft.neutral_soft_coefficient_distance, 0.0)
        self.assertLessEqual(max(score.score for score in soft.scores), max(score.score for score in neutral.scores) + 1.0e-7)

    def test_soft_influence_is_allocation_only(self) -> None:
        soft = solve_strength_preserving_routing(
            _problem(), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=0.25
        )
        payload = soft.raw_free_payload()
        self.assertEqual(payload["hard_h_p_budget_influence_count"], 0)
        self.assertEqual(payload["functional_veto_count"], 0)
        self.assertEqual(payload["retry_count"], 0)
        self.assertEqual(payload["overlay_authoritative_access_count"], 0)
        self.assertEqual(payload["target_probe_arm_dependent_influence_count"], 0)
        self.assertTrue(any(score.influence_count == 1 for score in soft.scores))

    def test_writer_unreachable_records_coverage_without_relaxation(self) -> None:
        observed = solve_strength_preserving_routing(
            _problem((0.1, 0.0, 0.0, 0.0, 0.0)),
            _inventory(),
            arm=FixedE8Arm.NEUTRAL,
            alpha_req=1.0,
        )
        self.assertEqual(observed.status, StrengthPreservingStatus.WRITER_UNREACHABLE)
        self.assertEqual(observed.alpha_apply, observed.alpha_max)
        self.assertLess(observed.coverage, 1.0)
        self.assertEqual(observed.feasible_allocation_dimension, 0)

    def test_unique_strength_manifold_is_typed_no_dof(self) -> None:
        problem = _problem((0.5, 0.0, 0.0, 0.0, 0.0))
        neutral = solve_strength_preserving_routing(
            problem, _inventory(), arm=FixedE8Arm.NEUTRAL, alpha_req=0.25
        )
        soft = solve_strength_preserving_routing(
            problem, _inventory(), arm=FixedE8Arm.SOFT, alpha_req=0.25
        )
        self.assertEqual(neutral.status, StrengthPreservingStatus.NO_ROUTING_DOF)
        self.assertEqual(soft.status, StrengthPreservingStatus.NO_ROUTING_DOF)
        self.assertEqual(neutral.velocity, soft.velocity)

    def test_zero_target_demand_is_total_and_target_only(self) -> None:
        observed = solve_strength_preserving_routing(
            _problem(), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=0.0
        )
        self.assertEqual(
            observed.status, StrengthPreservingStatus.ZERO_DEMAND_TARGET_ONLY
        )
        self.assertEqual(observed.velocity, (0.0,) * 5)
        self.assertEqual(observed.coverage, 1.0)

    def test_tiny_positive_demand_is_not_attenuated_to_zero(self) -> None:
        observed = solve_strength_preserving_routing(
            _problem(), _inventory(), arm=FixedE8Arm.NEUTRAL, alpha_req=1.0e-13
        )
        self.assertGreater(observed.alpha_apply, 0.0)
        self.assertGreater(np.dot(observed.signed_slopes, observed.velocity), 0.0)
        self.assertNotEqual(
            observed.status, StrengthPreservingStatus.ZERO_DEMAND_TARGET_ONLY
        )

    def test_positive_demand_with_no_active_direction_is_typed_unreachable(self) -> None:
        observed = solve_strength_preserving_routing(
            _problem((-1.0, -0.5, 0.0, -0.2, -0.1)),
            _inventory(),
            arm=FixedE8Arm.NEUTRAL,
            alpha_req=0.2,
        )
        self.assertEqual(observed.status, StrengthPreservingStatus.WRITER_UNREACHABLE)
        self.assertEqual(observed.alpha_apply, 0.0)
        self.assertEqual(observed.coverage, 0.0)
        self.assertEqual(observed.velocity, (0.0,) * 5)

    def test_soft_shadow_failure_never_stops_neutral(self) -> None:
        from project.run_scripts.ode_bf import strength_preserving_routing

        with mock.patch.object(
            strength_preserving_routing,
            "_solve_soft",
            side_effect=ODEBFContractError("detached shadow failed"),
        ):
            observed = solve_strength_preserving_routing(
                _problem(),
                _inventory(),
                arm=FixedE8Arm.NEUTRAL,
                alpha_req=0.2,
            )
        self.assertTrue(observed.soft_shadow_status.startswith("DIAGNOSTIC_UNAVAILABLE_"))
        self.assertAlmostEqual(
            np.dot(observed.signed_slopes, observed.velocity), 0.2, places=8
        )

    def test_source_has_no_legacy_floor_or_trust_feasible_constraint(self) -> None:
        source = inspect.getsource(solve_strength_preserving_routing)
        self.assertNotIn("FIXED_E8_KAPPA", source)
        self.assertNotIn("trust_radius", source)
        self.assertNotIn("problem.requested_progress", source)
        self.assertIn("slopes > 0.0", source)

    def test_route_independent_probe_and_physical_write_are_separate(self) -> None:
        routing = solve_strength_preserving_routing(
            _problem(), _inventory(), arm=FixedE8Arm.NEUTRAL, alpha_req=0.2
        )
        receipt = _target_receipt(
            allocation="PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED"
        )
        observed = route_independent_target_write_identity(
            routing, routing.applied_coefficient, receipt
        )
        self.assertEqual(observed["target_probe_h"], 0.125)
        self.assertEqual(observed["physical_write_h"], 0.125)
        self.assertEqual(observed["target_probe_arm_dependent_influence_count"], 0)

    def test_fixed_clock_has_exactly_eight_updates_and_no_retry(self) -> None:
        clock = FixedE8Clock()
        for index in range(8):
            point = clock.begin_field()
            self.assertEqual(point.step_index, index)
            clock.advance(point)
        receipt = clock.terminal_receipt()
        self.assertEqual(clock.tau, Fraction(1, 1))
        self.assertEqual(receipt["grid_count"], 8)
        self.assertEqual(receipt["scientific_retry_count"], 0)
        self.assertEqual(receipt["backtracking_count"], 0)

    def test_structural_p_is_soft_only_and_uses_existing_problem_geometry(self) -> None:
        problem = _problem()
        changed_budget = RoutingProblem(
            problem.signed_progress,
            problem.capacity_metric,
            problem.trust_metric * 1000.0,
            1.0e-6,
            problem.layer_caps,
            problem.requested_progress,
            problem.minimum_progress,
            problem.historical,
            QuadraticBarrier(
                "pretrained",
                problem.pretrained.offset,
                problem.pretrained.linear,
                problem.pretrained.gram,
                1000.0,
                "layer-local-diagonal",
            ),
        )
        first = solve_strength_preserving_routing(
            problem, _inventory(), arm=FixedE8Arm.SOFT, alpha_req=0.2
        )
        second = solve_strength_preserving_routing(
            changed_budget, _inventory(), arm=FixedE8Arm.SOFT, alpha_req=0.2
        )
        self.assertEqual(first.velocity, second.velocity)
        self.assertEqual(first.alpha_apply, second.alpha_apply)

    def test_runtime_source_reuses_nohook_and_zero_target_probe(self) -> None:
        from project.run_scripts.ode_bf import common_coldcoord_fixed_e8_runtime

        source = inspect.getsource(common_coldcoord_fixed_e8_runtime._build_common_field)
        self.assertIn("legacy._fixed_e8_signed_progress_gradient", source)
        self.assertIn("velocity_coefficients=(0.0,) * len(COMMON_COLD_LAYER_ORDER)", source)
        self.assertIn("router_added_model_forward_count", source)
        self.assertIn('router_counter_delta.get(\n            "processed_tokens", 0', source)
        self.assertNotIn("strength_preserving_endpoint_probe", source)

    def test_runtime_uses_postfreeze_w_only_and_z_oracle_panel(self) -> None:
        from project.run_scripts.ode_bf import common_coldcoord_fixed_e8_runtime

        source = inspect.getsource(
            common_coldcoord_fixed_e8_runtime.run_common_coldcoord_fixed_e8_diagnostic
        )
        self.assertIn("_postfreeze_bg_soft_prefix_panel", source)
        self.assertIn("PHYSICAL_BF16_WONLY_TARGET_NEW_NLL_PROGRESS", source)
        rollout_source = inspect.getsource(
            common_coldcoord_fixed_e8_runtime._run_common_arm
        )
        self.assertIn(
            "omega_changed != bool(authoritative_weight_write_count)",
            rollout_source,
        )


if __name__ == "__main__":
    unittest.main()
