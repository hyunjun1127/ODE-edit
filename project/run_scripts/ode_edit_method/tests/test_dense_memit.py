from __future__ import annotations

import gc
import inspect
import unittest
import weakref
from unittest import mock

import torch

from project.run_scripts.ode_edit_method.dense_memit import (
    TransientDenseMemitSolver,
)
from project.run_scripts.ode_edit_motivation.diagnostic_math import NativeMemitSolver


def _fixture(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    panel = torch.randn(12, 12, generator=generator, dtype=dtype)
    covariance = panel @ panel.transpose(0, 1) + 0.75 * torch.eye(12, dtype=dtype)
    keys = torch.randn(12, 3, generator=generator, dtype=dtype)
    return covariance, keys


class TransientDenseMemitSolverTests(unittest.TestCase):
    def test_float64_matches_native_reference_tightly(self) -> None:
        covariance, keys = _fixture(torch.float64)
        pointer = covariance.data_ptr()
        version = covariance._version
        expected = NativeMemitSolver().adjusted_keys(covariance, keys, 0.65)
        observed = TransientDenseMemitSolver().adjusted_keys(
            covariance, keys, 0.65
        )
        self.assertTrue(torch.allclose(observed, expected, atol=1e-12, rtol=1e-12))
        self.assertEqual(covariance.data_ptr(), pointer)
        self.assertEqual(covariance._version, version)

    def test_float32_matches_native_reference_at_locked_tolerance(self) -> None:
        covariance, keys = _fixture(torch.float32)
        expected = NativeMemitSolver().adjusted_keys(covariance, keys, 0.65)
        observed = TransientDenseMemitSolver().adjusted_keys(
            covariance, keys, 0.65
        )
        self.assertTrue(torch.allclose(observed, expected, atol=2e-5, rtol=2e-5))

    def test_one_owned_system_exact_equation_and_no_retained_cache(self) -> None:
        covariance, keys = _fixture(torch.float64)
        expected_system = 0.65 * covariance + keys @ keys.transpose(0, 1)
        observed_system: torch.Tensor | None = None
        system_reference: weakref.ReferenceType[torch.Tensor] | None = None

        def solve(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            nonlocal observed_system, system_reference
            observed_system = system.detach().clone()
            system_reference = weakref.ref(system)
            return torch.zeros_like(rhs)

        with mock.patch("torch.linalg.solve", side_effect=solve) as solve_mock:
            result = TransientDenseMemitSolver().adjusted_keys(
                covariance, keys, 0.65
            )
        solve_mock.assert_called_once()
        self.assertTrue(torch.equal(result, torch.zeros_like(keys)))
        self.assertIsNotNone(observed_system)
        self.assertTrue(torch.allclose(observed_system, expected_system, atol=0, rtol=0))
        solve_mock.reset_mock()
        del solve_mock
        del result
        gc.collect()
        assert system_reference is not None
        self.assertIsNone(system_reference())
        self.assertFalse(hasattr(TransientDenseMemitSolver(), "__dict__"))

    def test_source_contains_only_in_place_dense_assembly(self) -> None:
        source = inspect.getsource(TransientDenseMemitSolver.adjusted_keys)
        self.assertIn("copy=True", source)
        self.assertIn("system.mul_", source)
        self.assertIn("system.addmm_", source)
        self.assertIn("torch.linalg.solve(system, keys)", source)
        self.assertNotIn("keys @ keys", source)
        self.assertNotIn("covariance_weight * covariance", source)


if __name__ == "__main__":
    unittest.main()
