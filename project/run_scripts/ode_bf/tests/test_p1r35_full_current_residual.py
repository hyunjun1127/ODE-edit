from __future__ import annotations

from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24TargetStep
from project.run_scripts.ode_bf.p1r35_full_current_residual import (
    apply_p1r35_full_current_residual,
)
from project.run_scripts.ode_bf.p1r35_full_current_residual_panel import (
    P1R35_ROLES,
    expected_p1r35_result_name,
)


def _target_step() -> P1R24TargetStep:
    shape = (3, 2)
    return P1R24TargetStep(
        target_next=torch.tensor([[3.0, 5.0], [2.0, 1.0], [4.0, 4.0]]),
        target_displacement=torch.tensor(
            [[1.0, 1.0], [1.0, -1.0], [1.0, 2.0]]
        ),
        required_displacement=torch.full(shape, 99.0),
        write_velocity=torch.full(shape, 99.0),
        nll_gradient=torch.full(shape, -0.5),
        combined_gradient=torch.full(shape, -0.25),
        rho_write_signed=123.0,
        rho_write=123.0,
        alpha_target_signed=0.5,
        frozen_mask=(False, False),
        receipt={"identity_sha256": "a" * 64, "remaining_steps": 8},
    )


class P1R35FullCurrentResidualTests(unittest.TestCase):
    def test_full_current_residual_has_no_remaining_division(self) -> None:
        current = torch.tensor([[2.0, 4.0], [1.0, 2.0], [3.0, 2.0]])
        terminal = torch.tensor([[1.5, 2.0], [0.0, 3.0], [2.5, 1.0]])
        result = apply_p1r35_full_current_residual(
            _target_step(),
            current_target=current,
            current_terminal=terminal,
            step_index=0,
        )
        expected = _target_step().target_next - terminal
        self.assertTrue(torch.equal(result.required_displacement, expected))
        self.assertTrue(torch.equal(0.125 * result.write_velocity, expected))
        self.assertEqual(result.receipt["remaining_horizon_division_count"], 0)
        self.assertEqual(result.receipt["fresh_component_division_count"], 0)
        self.assertEqual(result.receipt["lag_component_division_count"], 0)
        self.assertEqual(result.receipt["semantic_debt_input_count"], 0)
        self.assertEqual(result.receipt["physical_h_application_count"], 1)
        self.assertEqual(result.receipt["second_h_application_count"], 0)

    def test_full_residual_differs_from_remaining_horizon_coordinate(self) -> None:
        current = torch.ones((3, 2))
        terminal = torch.zeros((3, 2))
        step = P1R24TargetStep(
            target_next=torch.full((3, 2), 2.0),
            target_displacement=torch.ones((3, 2)),
            required_displacement=torch.full((3, 2), 1.125),
            write_velocity=torch.full((3, 2), 9.0),
            nll_gradient=torch.full((3, 2), -1.0),
            combined_gradient=torch.full((3, 2), -1.0),
            rho_write_signed=0.0,
            rho_write=0.0,
            alpha_target_signed=0.0,
            frozen_mask=(False, False),
            receipt={"identity_sha256": "b" * 64},
        )
        result = apply_p1r35_full_current_residual(
            step,
            current_target=current,
            current_terminal=terminal,
            step_index=0,
        )
        self.assertTrue(torch.equal(result.required_displacement, torch.full((3, 2), 2.0)))
        self.assertNotEqual(
            result.receipt["required_displacement_sha256"],
            "b" * 64,
        )

    def test_matrix_is_two_models_times_two_phases(self) -> None:
        self.assertEqual(P1R35_ROLES, ("P1R35_B1_RS_PAIR", "P1R35_B10_RS_PAIR"))
        names = {
            expected_p1r35_result_name(alias, role)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for role in P1R35_ROLES
        }
        self.assertEqual(len(names), 4)

    def test_helper_has_no_scientific_import_from_r32_r33(self) -> None:
        source = (
            Path(__file__).parents[1] / "p1r35_full_current_residual.py"
        ).read_text()
        self.assertNotIn("p1r32", source.casefold())
        self.assertNotIn("p1r33", source.casefold())
        self.assertIn("full64 = (d64 + lag64)", source)
        self.assertNotIn("lag64 /", source)


if __name__ == "__main__":
    unittest.main()
