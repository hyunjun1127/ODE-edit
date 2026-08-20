from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.p1r52_residual_reserve_geometry import (
    build_residual_reserve_geometry,
)
from project.run_scripts.ode_bf import p1r52_residual_reserve_pc_router as router_module
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_router import (
    QuadraticProxy,
    solve_residual_reserve_pc_router,
)


class ResidualReservePCRouterTests(unittest.TestCase):
    @staticmethod
    def squared_distance_proxy(name: str, layer: int) -> QuadraticProxy:
        linear = torch.zeros(5, dtype=torch.float32)
        linear[layer] = -2.0
        return QuadraticProxy(
            name,
            1.0,
            linear,
            torch.eye(5, dtype=torch.float32),
        )

    @staticmethod
    def constant_proxy(name: str, value: float = 0.0) -> QuadraticProxy:
        return QuadraticProxy(
            name,
            value,
            torch.zeros(5, dtype=torch.float32),
            torch.zeros((5, 5), dtype=torch.float32),
        )

    def antagonistic(self):
        return solve_residual_reserve_pc_router(
            self.squared_distance_proxy("P", 0),
            self.squared_distance_proxy("C", 4),
        )

    def test_simplex_nonnegative_and_deterministic_repeat(self) -> None:
        left = self.antagonistic()
        right = self.antagonistic()
        for value in (left.pi_p, left.pi_c, left.pi_balanced, left.pi_uniform):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(bool((value >= 0.0).all()))
            self.assertAlmostEqual(float(value.sum()), 1.0, places=6)
        self.assertTrue(torch.equal(left.pi_balanced, right.pi_balanced))
        self.assertEqual(
            left.receipt.raw_free_payload(), right.receipt.raw_free_payload()
        )

    def test_antagonistic_convex_ordering(self) -> None:
        result = self.antagonistic()
        candidates = {
            item.role: item for item in result.receipt.candidates
        }
        self.assertLess(candidates["P_ONLY"].p_value, candidates["BALANCED"].p_value)
        self.assertLess(candidates["BALANCED"].p_value, candidates["C_ONLY"].p_value)
        self.assertLess(candidates["C_ONLY"].c_value, candidates["BALANCED"].c_value)
        self.assertLess(candidates["BALANCED"].c_value, candidates["P_ONLY"].c_value)
        torch.testing.assert_close(
            result.pi_balanced,
            torch.tensor([0.5, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32),
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_flat_axis_behaviors(self) -> None:
        p_active = self.squared_distance_proxy("P", 0)
        c_active = self.squared_distance_proxy("C", 4)
        p_flat = solve_residual_reserve_pc_router(
            self.constant_proxy("P-flat", 3.0), c_active
        )
        self.assertTrue(p_flat.receipt.p_flat)
        self.assertFalse(p_flat.receipt.c_flat)
        self.assertTrue(torch.equal(p_flat.pi_balanced, p_flat.pi_c))
        c_flat = solve_residual_reserve_pc_router(
            p_active, self.constant_proxy("C-flat", -2.0)
        )
        self.assertFalse(c_flat.receipt.p_flat)
        self.assertTrue(c_flat.receipt.c_flat)
        self.assertTrue(torch.equal(c_flat.pi_balanced, c_flat.pi_p))
        both = solve_residual_reserve_pc_router(
            self.constant_proxy("P-flat"), self.constant_proxy("C-flat")
        )
        self.assertTrue(both.receipt.p_flat and both.receipt.c_flat)
        self.assertTrue(torch.equal(both.pi_balanced, both.pi_uniform))
        self.assertEqual(both.receipt.selected_status, "BOTH_AXES_FLAT_UNIFORM")

    def test_positive_affine_scaling_is_invariant(self) -> None:
        p = self.squared_distance_proxy("P", 0)
        c = self.squared_distance_proxy("C", 4)
        base = solve_residual_reserve_pc_router(p, c)
        scaled_p = QuadraticProxy(
            "scaled-P",
            7.5 * p.constant + 13.0,
            7.5 * p.linear,
            7.5 * p.quadratic,
        )
        scaled_c = QuadraticProxy(
            "scaled-C",
            0.25 * c.constant - 5.0,
            0.25 * c.linear,
            0.25 * c.quadratic,
        )
        for scaled in (
            solve_residual_reserve_pc_router(scaled_p, c),
            solve_residual_reserve_pc_router(p, scaled_c),
            solve_residual_reserve_pc_router(scaled_p, scaled_c),
        ):
            torch.testing.assert_close(
                scaled.pi_balanced,
                base.pi_balanced,
                rtol=0.0,
                atol=torch.finfo(torch.float32).eps,
            )

    def test_nonuniform_route_preserves_m1_geometry(self) -> None:
        route = self.antagonistic().pi_balanced
        self.assertFalse(torch.equal(route, torch.full((5,), 0.2, dtype=torch.float32)))
        geometry = build_residual_reserve_geometry(route)
        self.assertLessEqual(
            abs(float(geometry.omega.sum() - geometry.mass)),
            router_module.FP32_SIMPLEX_SUM_TOLERANCE,
        )
        torch.testing.assert_close(
            geometry.beta * geometry.suffix_retention,
            geometry.omega,
            rtol=0.0,
            atol=torch.finfo(torch.float32).eps,
        )

    def test_exact_tie_determinism_has_no_epsilon_objective(self) -> None:
        p = QuadraticProxy(
            "symmetric-P",
            0.0,
            torch.zeros(5, dtype=torch.float32),
            torch.eye(5, dtype=torch.float32),
        )
        c = QuadraticProxy(
            "symmetric-C",
            4.0,
            torch.zeros(5, dtype=torch.float32),
            torch.eye(5, dtype=torch.float32),
        )
        first = solve_residual_reserve_pc_router(p, c)
        second = solve_residual_reserve_pc_router(p, c)
        self.assertTrue(torch.equal(first.pi_balanced, first.pi_uniform))
        self.assertTrue(torch.equal(first.pi_balanced, second.pi_balanced))
        self.assertEqual(
            first.receipt.tie_break_order,
            router_module.TIE_BREAK_ORDER,
        )
        source = inspect.getsource(router_module)
        self.assertNotIn("epsilon *", source)

    def test_fp32_inputs_are_cloned_and_immutable(self) -> None:
        linear = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        quadratic = torch.eye(5, dtype=torch.float32)
        linear_before = linear.clone()
        quadratic_before = quadratic.clone()
        p = QuadraticProxy("P", 1.0, linear, quadratic)
        c = self.squared_distance_proxy("C", 4)
        result = solve_residual_reserve_pc_router(p, c)
        torch.testing.assert_close(linear, linear_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(quadratic, quadratic_before, rtol=0.0, atol=0.0)
        self.assertNotEqual(linear.data_ptr(), p.linear.data_ptr())
        self.assertNotEqual(quadratic.data_ptr(), p.quadratic.data_ptr())
        self.assertEqual(result.pi_balanced.dtype, torch.float32)
        self.assertNotEqual(result.pi_balanced.data_ptr(), result.pi_p.data_ptr())

    def test_invalid_quadratics_fail_closed(self) -> None:
        valid_linear = torch.zeros(5, dtype=torch.float32)
        nonsymmetric = torch.eye(5, dtype=torch.float32)
        nonsymmetric[0, 1] = 0.5
        non_psd = torch.eye(5, dtype=torch.float32)
        non_psd[2, 2] = -1.0
        invalid = (
            (float("nan"), valid_linear, torch.eye(5, dtype=torch.float32)),
            (0.0, torch.full((5,), float("inf")), torch.eye(5, dtype=torch.float32)),
            (0.0, valid_linear, nonsymmetric),
            (0.0, valid_linear, non_psd),
            (0.0, valid_linear.double(), torch.eye(5, dtype=torch.float32)),
        )
        for constant, linear, quadratic in invalid:
            with self.subTest(constant=constant, dtype=linear.dtype):
                with self.assertRaises(ODEBFContractError):
                    QuadraticProxy("invalid", constant, linear, quadratic)

    def test_solver_failure_is_typed(self) -> None:
        failed = SimpleNamespace(
            success=False,
            x=torch.full((5,), 0.2, dtype=torch.float64).numpy(),
            message="injected solver failure",
            nit=1,
        )
        with mock.patch.object(router_module, "_scipy_minimize", return_value=failed):
            with self.assertRaises(ODEBFStateError):
                solve_residual_reserve_pc_router(
                    self.squared_distance_proxy("P", 0),
                    self.squared_distance_proxy("C", 4),
                )
        impossible = SimpleNamespace(
            success=True,
            x=torch.zeros(5, dtype=torch.float64).numpy(),
            message="injected impossible simplex",
            nit=1,
        )
        with mock.patch.object(router_module, "_scipy_minimize", return_value=impossible):
            with self.assertRaises(ODEBFStateError):
                solve_residual_reserve_pc_router(
                    self.squared_distance_proxy("P", 0),
                    self.squared_distance_proxy("C", 4),
                )

    def test_prohibited_imports_and_symbols_are_absent(self) -> None:
        source = inspect.getsource(router_module)
        prohibited = (
            "progress_simplex_routing",
            "fixed_e8",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_pir_writer",
            "p1r52_frozen_pi_quota_writer",
            "p1r43_full_strength_routing",
            "torch.nn",
        )
        for symbol in prohibited:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)
        payload = self.antagonistic().receipt.raw_free_payload()
        self.assertEqual(payload["model_forward_backward_materialization"], [0, 0, 0])
        self.assertEqual(payload["manual_weighted_sum_count"], 0)
        self.assertEqual(payload["fallback_retry_backtracking_count"], 0)


if __name__ == "__main__":
    unittest.main()
