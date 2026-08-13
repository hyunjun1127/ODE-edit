from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.p1r32_dynamic_z5 import (
    DynamicZ5RoutingStatus,
    P1R32_H,
    solve_dynamic_z5_direct_routing,
    solve_dynamic_z5_target,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _problem(slopes: np.ndarray) -> RoutingProblem:
    dimension = slopes.size
    zero = np.zeros(dimension, dtype=np.float64)
    pretrained = QuadraticBarrier(
        "pretrained",
        0.0,
        np.asarray((0.5, 0.2, 0.1, 0.3, 0.4), dtype=np.float64),
        np.diag(np.asarray((3.0, 1.0, 0.5, 2.0, 1.5))),
        1.0e6,
        "layer-local-diagonal",
    )
    historical = QuadraticBarrier(
        "historical", 0.0, zero, np.zeros((dimension, dimension)), 1.0e6,
        "layer-local-diagonal",
    )
    return RoutingProblem(
        slopes,
        np.diag(np.asarray((1.0, 2.0, 3.0, 4.0, 5.0))),
        np.eye(dimension),
        1.0e6,
        np.full(dimension, 1.0e6),
        1.0,
        0.5,
        historical,
        pretrained,
    )


class DynamicZ5RouterTest(unittest.TestCase):
    def test_neutral_is_direct_nominal_c0(self) -> None:
        slopes = np.asarray((2.0, 1.0, -0.5, 0.25, 3.0))
        result = solve_dynamic_z5_direct_routing(
            _problem(slopes), arm=FixedE8Arm.NEUTRAL, rho_write=0.75
        )
        self.assertEqual(result.status, DynamicZ5RoutingStatus.JOINT_WRITE)
        q = float(np.sum(P1R32_H * slopes[slopes > 0.0]))
        expected_velocity = tuple(
            0.75 / q if item > 0.0 else 0.0 for item in slopes
        )
        self.assertEqual(result.velocity, expected_velocity)
        self.assertEqual(
            result.selected_applied_coefficient,
            tuple(P1R32_H * item for item in expected_velocity),
        )
        self.assertAlmostEqual(
            result.predicted_progress,
            0.75,
            places=14,
        )
        self.assertEqual(result.raw_free_payload()["inverse_slope_operation_count"], 0)

    def test_soft_preserves_reference_strength_and_energy(self) -> None:
        result = solve_dynamic_z5_direct_routing(
            _problem(np.asarray((2.0, 1.0, 0.5, 0.25, 3.0))),
            arm=FixedE8Arm.SOFT,
            rho_write=0.75,
        )
        self.assertEqual(result.status, DynamicZ5RoutingStatus.JOINT_WRITE)
        self.assertGreaterEqual(result.predicted_progress + 1.0e-8, result.reference_progress)
        self.assertLessEqual(result.selected_energy, result.reference_energy * (1 + 1e-8) + 1e-12 + 1e-10)
        self.assertLessEqual(result.selected_p, result.reference_p + 1e-10)

    def test_no_positive_direction_is_typed(self) -> None:
        result = solve_dynamic_z5_direct_routing(
            _problem(-np.ones(5)), arm=FixedE8Arm.SOFT, rho_write=0.75
        )
        self.assertEqual(result.status, DynamicZ5RoutingStatus.NO_POSITIVE_DIRECTION)


class DynamicZ5TargetTest(unittest.TestCase):
    def test_exact_five_adam_updates_and_full_residual(self) -> None:
        model = torch.nn.Linear(3, 3, bias=False)
        objective_plan = SimpleNamespace(request_count=2)
        kl_plan = SimpleNamespace(request_count=2)
        terminal = torch.tensor(
            [[2.0, 3.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
        )
        current = terminal.clone()
        calls = {"nll": 0, "kl": 0, "teacher": 0}

        def nll(*args, target_state=None, **kwargs):
            del args, kwargs
            calls["nll"] += 1
            return SimpleNamespace(
                per_request_values=(1.0, 2.0),
                target_gradient=torch.full_like(target_state.detach().cpu(), -0.25),
                model_forward_count=1,
                backward_count=1,
                processed_token_count=12,
                identity_sha256="1" * 64,
            )

        def kl(*args, teacher_log_probs=None, target_state=None, **kwargs):
            del args, kwargs
            if teacher_log_probs is None:
                calls["teacher"] += 1
                result = SimpleNamespace(
                    per_request_values=(0.0, 0.0),
                    gradient=None,
                    model_forward_count=1,
                    backward_count=0,
                    processed_token_count=2,
                    identity_sha256="2" * 64,
                )
                return result, (torch.zeros(3), torch.zeros(3))
            calls["kl"] += 1
            result = SimpleNamespace(
                per_request_values=(0.0, 0.0),
                gradient=torch.zeros_like(target_state.detach().cpu()),
                model_forward_count=1,
                backward_count=1,
                processed_token_count=2,
                identity_sha256="3" * 64,
            )
            return result, teacher_log_probs

        with mock.patch(
            "project.run_scripts.ode_bf.p1r32_dynamic_z5.evaluate_scalable_target_new_objective",
            side_effect=nll,
        ), mock.patch(
            "project.run_scripts.ode_bf.p1r32_dynamic_z5.evaluate_p1r24_kl",
            side_effect=kl,
        ):
            result = solve_dynamic_z5_target(
                model,
                objective_plan,
                kl_plan,
                alias="llama3-8b-inst",
                current_target=current,
                current_terminal=terminal,
                step_index=0,
                target_layer_name="unused",
            )
        receipt = result.receipt
        self.assertEqual(receipt["adam_optimizer_step_count"], 5)
        self.assertEqual(receipt["loss_backward_count"], 5)
        self.assertEqual(receipt["inner_observation_count"], 6)
        self.assertEqual(receipt["teacher_refresh_count"], 1)
        self.assertEqual(receipt["remaining_step_division_count"], 0)
        self.assertEqual(receipt["semantic_debt_input_count"], 0)
        self.assertEqual(receipt["physical_h_application_count"], 1)
        self.assertEqual(receipt["second_h_application_count"], 0)
        self.assertGreaterEqual(result.rho_write, 0.0)
        self.assertTrue(torch.equal(P1R32_H * result.field_velocity, result.full_residual))
        self.assertTrue(torch.allclose(result.target_next - terminal, result.full_residual))
        self.assertEqual(calls, {"nll": 6, "kl": 6, "teacher": 1})


if __name__ == "__main__":
    unittest.main()
