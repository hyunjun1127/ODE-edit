from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.fixed_e8_soft_routing import RoutingProblem
from project.run_scripts.ode_bf.p2r7_shared_writer import (
    P2R7_ROUTING_DIMENSION,
    P2R7SharedWriterNoPositiveDirection,
    build_p2r7_deficit_weighting,
    p2r7_forbidden_influence_receipt,
    solve_p2r7_shared_routing,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier


def _problem(slopes: np.ndarray) -> RoutingProblem:
    dimension = slopes.size
    zero = np.zeros(dimension, dtype=np.float64)
    gram = np.diag(np.linspace(1.0, 2.0, dimension))
    return RoutingProblem(
        slopes.astype(np.float64),
        np.diag(np.linspace(2.0, 3.0, dimension)),
        np.eye(dimension, dtype=np.float64),
        100.0,
        np.full(dimension, 1.0e9, dtype=np.float64),
        0.1,
        1.0e-8,
        QuadraticBarrier(
            "historical", 0.0, zero, np.zeros_like(gram), 0.0, "layer-local-diagonal"
        ),
        QuadraticBarrier(
            "pretrained", 0.0, zero, gram, 0.0, "layer-local-diagonal"
        ),
    )


class P2R7SharedWriterTest(unittest.TestCase):
    def test_deficit_weighted_identity(self) -> None:
        ell_w = [float(index + 1) for index in range(10)]
        ell_z = [0.5 for _ in range(10)]
        result = build_p2r7_deficit_weighting(ell_w, ell_z, mode="P1DW")
        deficit = np.maximum(np.asarray(ell_w) - np.asarray(ell_z), 0.0)
        self.assertAlmostEqual(sum(result.omega), 1.0, places=14)
        self.assertAlmostEqual(
            result.rho_omega,
            float(deficit @ deficit / deficit.sum()),
            places=14,
        )
        self.assertFalse(result.no_semantic_deficit)

    def test_aggregate_control_is_uniform(self) -> None:
        result = build_p2r7_deficit_weighting(
            [1.0, 2.0, *([0.0] * 8)], [0.0] * 10, mode="P1AGG"
        )
        self.assertEqual(result.omega, tuple([0.1] * 10))
        self.assertAlmostEqual(result.rho_omega, 0.3)

    def test_all_zero_deficit_is_total(self) -> None:
        result = build_p2r7_deficit_weighting([0.1] * 10, [0.2] * 10, mode="P1DW")
        self.assertTrue(result.no_semantic_deficit)
        self.assertEqual(result.rho_omega, 0.0)
        self.assertAlmostEqual(sum(result.omega), 1.0)

    def test_neutral_is_five_dimensional_exact_strength(self) -> None:
        slopes = np.asarray([0.2, 0.0, 0.3, -0.1, 0.5], dtype=np.float64)
        result = solve_p2r7_shared_routing(
            _problem(slopes), arm="P2TARGET-P1DW-NEUTRAL", rho_omega=0.4
        )
        self.assertEqual(len(result.velocity), P2R7_ROUTING_DIMENSION)
        self.assertAlmostEqual(result.predicted_progress, 0.4, places=8)
        active_velocity = [
            value for value, slope in zip(result.velocity, slopes, strict=True) if slope > 0
        ]
        self.assertLess(max(active_velocity) - min(active_velocity), 1.0e-12)

    def test_positive_demand_without_direction_is_scientific(self) -> None:
        with self.assertRaises(P2R7SharedWriterNoPositiveDirection):
            solve_p2r7_shared_routing(
                _problem(np.asarray([-1.0, 0.0, -2.0, 0.0, -3.0])),
                arm="NEUTRAL",
                rho_omega=0.1,
            )

    def test_forbidden_paths_are_zero(self) -> None:
        receipt = p2r7_forbidden_influence_receipt()
        self.assertEqual(receipt["routing_variable_count"], 5)
        for key in (
            "request_layer_jacobian_variable_count",
            "request_layer_response_matrix_count",
            "p2r6_certified_qp_import_count",
            "p2r6_shadow_solve_count",
            "retry_count",
            "backtracking_count",
            "candidate_materialization_count",
        ):
            self.assertEqual(receipt[key], 0)
        self.assertEqual(receipt["residual_inverse_h_count"], 1)
        self.assertEqual(receipt["physical_h_application_count"], 1)

    def test_runtime_has_no_p2r6_import(self) -> None:
        from project.run_scripts.ode_bf import p2r7_atomic_runtime

        source = inspect.getsource(p2r7_atomic_runtime)
        self.assertNotIn("p2r6_", source.lower())
        self.assertNotIn("measure_request_layer_response(", source)


if __name__ == "__main__":
    unittest.main()
