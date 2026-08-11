from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Arm,
    FixedE8SoftInventory,
    FunctionalBasisMetric,
)
from project.run_scripts.ode_bf.progress_simplex_routing import (
    ProgressSimplexStatus,
    SIMPLEX_PRIMAL_TOLERANCE,
    progress_simplex_waypoint_factors,
    solve_progress_simplex_routing,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _problem(
    slopes: tuple[float, ...] = (0.5, 0.3, 0.2, 0.0, -0.1),
    *,
    caps: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    symmetric_risk: bool = False,
) -> RoutingProblem:
    size = len(slopes)
    linear = np.zeros(size) if symmetric_risk else np.asarray((2.0, 0.01, 0.01, 0.0, 0.0))
    gram = np.diag(np.asarray((0.8, 0.01, 0.01, 0.01, 0.01)))
    if symmetric_risk:
        gram = np.zeros((size, size))
    capacity = (
        np.eye(size)
        if symmetric_risk
        else np.diag(np.asarray((0.7, 0.9, 1.1, 1.3, 1.5)))
    )
    energy = (
        np.eye(size)
        if symmetric_risk
        else np.diag(np.asarray((0.5, 0.6, 0.7, 0.8, 0.9)))
    )
    return RoutingProblem(
        np.asarray(slopes, dtype=np.float64),
        capacity,
        energy,
        1.0,
        np.asarray(caps, dtype=np.float64),
        1.0e-6,
        1.0e-8,
        QuadraticBarrier(
            "historical", 0.0, np.zeros(size), np.zeros((size, size)), 1.0, "layer-local-diagonal"
        ),
        QuadraticBarrier("pretrained", 0.0, linear, gram, 1.0, "layer-local-diagonal"),
    )


def _inventory(*, symmetric: bool = False) -> FixedE8SoftInventory:
    function = (0.0, 0.0, 0.0, 0.0, 0.0) if symmetric else (1.0, 0.01, 0.01, 0.0, 0.0)
    reason = "INACTIVE_EMPTY_HISTORY"
    return FixedE8SoftInventory(
        FunctionalBasisMetric("functional_p", 0.0, function, True),
        FunctionalBasisMetric("functional_h_mean", 0.0, (0.0,) * 5, False, reason),
        FunctionalBasisMetric("functional_h_smoothmax", 0.0, (0.0,) * 5, False, reason),
        0,
        "a" * 64,
        "b" * 64,
    )


class ProgressSimplexRoutingTests(unittest.TestCase):
    def test_active_q_simplex_neutral_identity_and_applied_progress(self) -> None:
        result = solve_progress_simplex_routing(
            _problem(), _inventory(), arm=FixedE8Arm.NEUTRAL, alpha_req=2.0
        )
        self.assertEqual(result.active_direction_mask, (True, True, True, False, False))
        self.assertAlmostEqual(result.q, 1.0)
        self.assertEqual(result.velocity, (1.0, 1.0, 1.0, 0.0, 0.0))
        self.assertEqual(result.neutral_pi, (0.5, 0.3, 0.2, 0.0, 0.0))
        self.assertAlmostEqual(result.predicted_progress, result.q, places=12)
        self.assertLessEqual(result.equality_residual, SIMPLEX_PRIMAL_TOLERANCE)
        self.assertAlmostEqual(result.coverage, 0.5)
        self.assertEqual(result.velocity_above_one_count, 0)
        self.assertLessEqual(result.neutral_mapping_max_abs, SIMPLEX_PRIMAL_TOLERANCE)
        self.assertAlmostEqual(result.neutral_capacity, 1.35)
        self.assertAlmostEqual(result.selected_capacity, result.neutral_capacity)
        self.assertEqual(result.selected_pi_top1_index, 0)
        self.assertEqual(result.selected_pi_top1_layer_id, 4)
        self.assertGreater(result.selected_pi_entropy, 0.0)
        self.assertGreater(result.selected_pi_effective_layer_count, 1.0)
        payload = result.raw_free_payload()
        self.assertEqual(payload["selected_pi_top1_layer_id"], 4)
        self.assertIn("risk_difference_soft_minus_neutral", payload)

    def test_soft_changes_allocation_at_same_progress_without_upper_cap(self) -> None:
        neutral = solve_progress_simplex_routing(
            _problem(), _inventory(), arm=FixedE8Arm.NEUTRAL, alpha_req=1.5
        )
        soft = solve_progress_simplex_routing(
            _problem(), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=1.5
        )
        self.assertAlmostEqual(soft.predicted_progress, neutral.q, places=8)
        self.assertLessEqual(soft.selected_energy_ratio, 1.0 + 1.0e-8 + 1.0e-10)
        self.assertGreater(soft.neutral_soft_coefficient_l2, 1.0e-7)
        self.assertGreater(soft.velocity_above_one_count, 0)
        self.assertEqual(soft.status, ProgressSimplexStatus.JOINT_WRITE)
        self.assertEqual(soft.soft_objective_decision_influence_count, 1)
        self.assertAlmostEqual(
            soft.global_energy_soft_neutral_ratio, soft.soft_energy_ratio
        )
        self.assertLessEqual(
            soft.risk_difference_soft_minus_neutral, SIMPLEX_PRIMAL_TOLERANCE
        )
        self.assertNotEqual(soft.neutral_capacity, soft.soft_capacity)

    def test_layer_caps_do_not_change_simplex_solution(self) -> None:
        first = solve_progress_simplex_routing(
            _problem(caps=(1.0,) * 5), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=1.0
        )
        second = solve_progress_simplex_routing(
            _problem(caps=(0.05, 0.07, 0.09, 0.11, 0.13)), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=1.0
        )
        self.assertTrue(np.allclose(first.velocity, second.velocity, rtol=0.0, atol=1.0e-8))
        source = inspect.getsource(solve_progress_simplex_routing)
        self.assertNotIn("layer_caps", source)
        self.assertNotIn("min(float(alpha_req)", source)

    def test_zero_and_one_positive_direction_are_typed_without_rescue(self) -> None:
        zero = solve_progress_simplex_routing(
            _problem((-0.1, 0.0, -0.2, 0.0, -0.3)), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=1.0
        )
        self.assertEqual(zero.status, ProgressSimplexStatus.NO_POSITIVE_DIRECTION)
        self.assertEqual(zero.velocity, (0.0,) * 5)
        one = solve_progress_simplex_routing(
            _problem((0.4, 0.0, -0.2, 0.0, -0.3)), _inventory(), arm=FixedE8Arm.SOFT, alpha_req=1.0
        )
        self.assertEqual(one.status, ProgressSimplexStatus.NO_ROUTING_DOF)
        self.assertEqual(one.velocity, (1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(one.feasible_allocation_dimension, 0)

    def test_symmetric_geometry_is_no_safe_redistribution(self) -> None:
        result = solve_progress_simplex_routing(
            _problem((0.5, 0.5, 0.0, 0.0, 0.0), symmetric_risk=True),
            _inventory(symmetric=True),
            arm=FixedE8Arm.SOFT,
            alpha_req=1.0,
        )
        self.assertEqual(result.status, ProgressSimplexStatus.NO_SAFE_REDISTRIBUTION)
        self.assertLessEqual(result.neutral_soft_coefficient_l2, SIMPLEX_PRIMAL_TOLERANCE)

    def test_factor_builder_allows_v_above_one_and_applies_h_once(self) -> None:
        layers = []
        for ordinal, layer in enumerate((4, 5, 6, 7, 8)):
            del ordinal
            layers.append(
                SimpleNamespace(
                    layer=layer,
                    weight_name=f"model.layers.{layer}.weight",
                    residual=torch.ones((2, 10), dtype=torch.float32),
                    q=torch.ones((2, 10), dtype=torch.float32),
                    factor=SimpleNamespace(global_batch_size=10),
                )
            )
        factors = progress_simplex_waypoint_factors(
            SimpleNamespace(layers=tuple(layers)), (2.5, 1.0, 0.0, 0.0, 0.0), step_index=0
        )
        self.assertAlmostEqual(factors["model.layers.4.weight"].theta, 2.5 / 8.0)
        self.assertAlmostEqual(factors["model.layers.5.weight"].theta, 1.0 / 8.0)
        with self.assertRaises(ODEBFContractError):
            progress_simplex_waypoint_factors(
                SimpleNamespace(layers=tuple(layers)), (-1.0, 0.0, 0.0, 0.0, 0.0), step_index=0
            )


if __name__ == "__main__":
    unittest.main()
