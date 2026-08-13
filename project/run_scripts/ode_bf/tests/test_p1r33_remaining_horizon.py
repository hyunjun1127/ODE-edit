from __future__ import annotations

from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24_H,
    p1r24_remaining_lag_coordinate,
)
from project.run_scripts.ode_bf.p1r33_remaining_horizon import (
    P1R33_METHOD_ID,
    p1r33_full_residual_remaining_horizon_coordinate,
)
from project.run_scripts.ode_bf.p1r33_remaining_horizon_panel import (
    P1R33_ROLES,
    expected_p1r33_result_name,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    terminal = torch.tensor([[0.5, 1.5], [2.0, 3.5]], dtype=torch.float64)
    displacement = torch.tensor([[0.8, -0.4], [0.2, 0.6]], dtype=torch.float64)
    target_next = current + displacement
    return target_next, current, terminal, displacement


class P1R33RemainingHorizonTests(unittest.TestCase):
    def test_k0_forms_full_residual_before_single_remaining_division(self) -> None:
        target_next, current, terminal, displacement = _inputs()
        result = p1r33_full_residual_remaining_horizon_coordinate(
            target_next, current, terminal, displacement, 0
        )
        full = target_next - terminal
        torch.testing.assert_close(
            result.required_analytic, full / 8.0, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            P1R24_H * result.write_velocity,
            result.required_model,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(result.receipt["remaining_steps"], 8)
        self.assertEqual(result.receipt["full_current_residual_count"], 1)
        self.assertEqual(result.receipt["remaining_horizon_division_count"], 1)
        self.assertEqual(result.receipt["physical_h_application_count"], 1)

    def test_k7_has_remaining_one_and_no_hidden_attenuation(self) -> None:
        values = _inputs()
        result = p1r33_full_residual_remaining_horizon_coordinate(*values, 7)
        torch.testing.assert_close(
            result.required_analytic, values[0] - values[2], rtol=0.0, atol=0.0
        )
        for name in (
            "semantic_debt_input_count",
            "explicit_lag_readdition_count",
            "second_h_application_count",
            "second_remaining_division_count",
            "residual_attenuation_by_barrier_count",
        ):
            self.assertEqual(result.receipt[name], 0)

    def test_new_policy_is_not_old_displacement_plus_lag_over_remaining(self) -> None:
        values = _inputs()
        new = p1r33_full_residual_remaining_horizon_coordinate(*values, 0)
        old = p1r24_remaining_lag_coordinate(*values, 0)
        self.assertFalse(torch.equal(new.required_analytic, old.required_analytic))
        expected_delta = values[3] / 8.0 - values[3]
        torch.testing.assert_close(
            new.required_analytic - old.required_analytic,
            expected_delta,
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_receipt_method_and_identities_are_total(self) -> None:
        result = p1r33_full_residual_remaining_horizon_coordinate(*_inputs(), 3)
        self.assertEqual(result.receipt["method_id"], P1R33_METHOD_ID)
        self.assertLessEqual(result.receipt["full_residual_identity_max_abs"], 1e-8)
        self.assertLessEqual(result.receipt["remaining_horizon_identity_max_abs"], 1e-8)
        self.assertLessEqual(result.receipt["h_write_velocity_identity_max_abs"], 1e-8)
        self.assertEqual(len(result.receipt["coordinate_receipt_sha256"]), 64)

    def test_execution_roles_are_exactly_two_paired_rs_cells(self) -> None:
        self.assertEqual(P1R33_ROLES, ("P1R33_B1_RS_PAIR", "P1R33_B10_RS_PAIR"))
        names = {
            expected_p1r33_result_name(alias, role)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for role in P1R33_ROLES
        }
        self.assertEqual(len(names), 4)

    def test_source_firewall_excludes_p1r32_and_old_expression(self) -> None:
        root = Path(__file__).parents[1]
        source = (root / "p1r33_remaining_horizon.py").read_text(encoding="utf-8")
        self.assertNotIn("p1r32", source.casefold())
        self.assertNotIn("displacement + lag /", source)
        self.assertIn("target_next64 - current_terminal64", source)
        self.assertIn("full_current_residual / float(remaining_steps)", source)


if __name__ == "__main__":
    unittest.main()
