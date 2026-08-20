from __future__ import annotations

import inspect
import math
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf import p1r52_residual_reserve_geometry as geometry_module
from project.run_scripts.ode_bf.p1r52_residual_reserve_geometry import (
    RESIDUAL_RESERVE_H_REF,
    RESIDUAL_RESERVE_RHO_REF,
    build_residual_reserve_geometry,
    current_prefix_residual,
    ideal_final_residual,
    ideal_layer_contributions,
)


class ResidualReserveGeometryTests(unittest.TestCase):
    @staticmethod
    def uniform() -> torch.Tensor:
        return torch.full((5,), 0.2, dtype=torch.float32)

    def test_simplex_total_contribution_and_no_last_dump(self) -> None:
        result = build_residual_reserve_geometry(self.uniform())
        self.assertEqual(result.pi.dtype, torch.float32)
        self.assertEqual(result.omega.dtype, torch.float32)
        self.assertEqual(result.suffix_retention.dtype, torch.float32)
        self.assertEqual(result.beta.dtype, torch.float32)
        self.assertAlmostEqual(float(result.pi.sum()), 1.0, places=7)
        self.assertLessEqual(
            abs(float(result.omega.sum()) - float(result.mass)),
            geometry_module.FP32_SIMPLEX_SUM_TOLERANCE,
        )
        permitted_maximum = torch.nextafter(
            result.mass,
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        self.assertLessEqual(float(result.beta.max()), float(permitted_maximum))
        self.assertLess(float(result.beta[-1]), 1.0)
        self.assertLess(float(result.mass), 1.0)

    def test_uniform_expected_beta(self) -> None:
        result = build_residual_reserve_geometry(self.uniform())
        expected = torch.tensor(
            [
                0.134464,
                0.15535345673561096,
                0.18392717838287354,
                0.2253808230161667,
                0.29095694422721863,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result.beta, expected, rtol=0.0, atol=2.0e-7)
        self.assertAlmostEqual(result.receipt.rho, RESIDUAL_RESERVE_RHO_REF, places=7)

    def test_ideal_contribution_and_final_residual(self) -> None:
        result = build_residual_reserve_geometry(self.uniform())
        entry = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float32)
        contributions = ideal_layer_contributions(entry, result)
        expected = result.omega[:, None, None] * entry[None, :, :]
        torch.testing.assert_close(contributions, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            contributions.sum(dim=0) + ideal_final_residual(entry, result),
            entry,
            rtol=0.0,
            atol=4.0e-7,
        )

    def test_nonuniform_direct_layer_identity(self) -> None:
        pi = torch.tensor([0.05, 0.1, 0.2, 0.25, 0.4], dtype=torch.float32)
        result = build_residual_reserve_geometry(pi)
        direct = result.beta * result.suffix_retention
        direct_residual = torch.abs(direct - result.omega)
        omega_ulp = torch.maximum(
            torch.nextafter(
                result.omega,
                torch.full_like(result.omega, float("inf")),
            )
            - result.omega,
            result.omega
            - torch.nextafter(
                result.omega,
                torch.full_like(result.omega, float("-inf")),
            ),
        )
        self.assertTrue(bool((direct_residual <= omega_ulp).all()))

        entry = torch.tensor([[1.25, -0.75], [0.5, 3.0]], dtype=torch.float32)
        left = result.beta[:, None, None] * (
            result.suffix_retention[:, None, None] * entry[None, :, :]
        )
        right = result.omega[:, None, None] * entry[None, :, :]
        realized_residual = torch.abs(left - right)
        machine_bound = (
            3.0
            * torch.finfo(torch.float32).eps
            * torch.maximum(torch.abs(right), torch.ones_like(right))
        )
        self.assertTrue(bool((realized_residual <= machine_bound).all()))

    def test_previous_writer_defect_is_in_next_residual(self) -> None:
        target = torch.tensor([[3.0, -1.0]], dtype=torch.float32)
        current_after_prefix = torch.tensor([[2.25, -1.5]], dtype=torch.float32)
        observed = current_prefix_residual(target, current_after_prefix)
        torch.testing.assert_close(
            observed,
            torch.tensor([[0.75, 0.5]], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
        changed_prefix = current_after_prefix + torch.tensor(
            [[0.125, -0.25]], dtype=torch.float32
        )
        torch.testing.assert_close(
            current_prefix_residual(target, changed_prefix),
            target - changed_prefix,
            rtol=0.0,
            atol=0.0,
        )

    def test_h_grid_is_finite_and_scaled(self) -> None:
        previous_rho = 1.0
        for k in (16, 8, 4):
            result = build_residual_reserve_geometry(self.uniform(), h=1.0 / k)
            self.assertTrue(torch.isfinite(result.beta).all())
            self.assertTrue(torch.isfinite(result.omega).all())
            self.assertLess(result.receipt.rho, previous_rho)
            self.assertAlmostEqual(
                result.receipt.rho,
                math.exp(-result.receipt.kappa / k),
                places=7,
            )
            previous_rho = result.receipt.rho
        reference = build_residual_reserve_geometry(
            self.uniform(), h=RESIDUAL_RESERVE_H_REF
        )
        self.assertAlmostEqual(reference.receipt.rho, RESIDUAL_RESERVE_RHO_REF, places=7)

    def test_zero_pi_components_remain_zero(self) -> None:
        pi = torch.tensor([0.5, 0.0, 0.25, 0.0, 0.25], dtype=torch.float32)
        result = build_residual_reserve_geometry(pi)
        self.assertEqual(float(result.omega[1]), 0.0)
        self.assertEqual(float(result.omega[3]), 0.0)
        self.assertEqual(float(result.beta[1]), 0.0)
        self.assertEqual(float(result.beta[3]), 0.0)

    def test_input_tensors_are_not_mutated_or_aliased(self) -> None:
        pi = self.uniform()
        pi_before = pi.clone()
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        terminal = torch.tensor([[0.25, 1.5]], dtype=torch.float32)
        target_before = target.clone()
        terminal_before = terminal.clone()
        result = build_residual_reserve_geometry(pi)
        residual = current_prefix_residual(target, terminal)
        torch.testing.assert_close(pi, pi_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target, target_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(terminal, terminal_before, rtol=0.0, atol=0.0)
        self.assertNotEqual(pi.data_ptr(), result.pi.data_ptr())
        self.assertNotEqual(target.data_ptr(), residual.data_ptr())

    def test_invalid_geometry_is_typed(self) -> None:
        invalid = (
            torch.tensor([0.2, 0.2, 0.2, 0.2, -0.2], dtype=torch.float32),
            torch.ones(5, dtype=torch.float64) / 5.0,
            torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float32),
            torch.tensor([0.2, 0.2, float("nan"), 0.2, 0.4], dtype=torch.float32),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ODEBFContractError):
                    build_residual_reserve_geometry(value)
        with self.assertRaises(ODEBFContractError):
            build_residual_reserve_geometry(self.uniform(), h=0.0)

    def test_receipt_is_stable_and_raw_free(self) -> None:
        left = build_residual_reserve_geometry(self.uniform()).receipt.raw_free_payload()
        right = build_residual_reserve_geometry(self.uniform()).receipt.raw_free_payload()
        self.assertEqual(left, right)
        self.assertEqual(left["tensor_dtype"], "torch.float32")
        self.assertTrue(left["no_debt_lag_catch_up_state"])
        self.assertEqual(len(left["identity_sha256"]), 64)

    def test_prohibited_imports_are_absent(self) -> None:
        source = inspect.getsource(geometry_module)
        prohibited = (
            "assemble_effective_bf16",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "p1r52_pir_writer",
            "p1r52_frozen_pi_quota_writer",
            "torch.nn",
        )
        for symbol in prohibited:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)


if __name__ == "__main__":
    unittest.main()
