from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from project.run_scripts.ode_bf.p1r39_normalized_gradient_target import (
    P1R39ControllerState,
    P1R39_DIRECTION_POLICY,
    prepare_p1r39_target_proposal,
    select_p1r39_target_proposal,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(values: tuple[float, ...], gradient: torch.Tensor | None, *, backward: int) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, backward, 10, 10, gradient, None, "4" * 64, "5" * 64, "6" * 64,
    )


def kl_result(count: int, gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(
        0.0, tuple(0.0 for _ in range(count)), gradient,
        1, 1, count, count, "7" * 64,
    )


class P1R39NormalizedGradientTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.tensor([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        self.terminal = self.current.clone()
        self.z0 = self.current.clone()
        self.gradient = torch.tensor([[0.3, 0.2, -0.4], [0.1, -0.1, 0.2]])
        self.lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")

    def test_normalized_columns_no_adam_cap_or_velocity_decay(self) -> None:
        state = P1R39ControllerState((True, False, False))
        proposal = prepare_p1r39_target_proposal(
            self.current, self.terminal, self.z0,
            objective((1.0, 0.01, 2.0), self.gradient, backward=1),
            kl_result(3, torch.zeros_like(self.gradient)), self.lock, state,
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        self.assertEqual(proposal.receipt["target_direction_policy"], P1R39_DIRECTION_POLICY)
        self.assertEqual(proposal.active_mask, (True, False, True))
        self.assertEqual(proposal.reactivated_mask, (True, False, False))
        self.assertEqual(proposal.receipt["hold_mode"], "CURRENT_STATE_INSTANTANEOUS_NO_CARRY")
        self.assertEqual(proposal.receipt["persistent_mask_decision_influence_count"], 0)
        self.assertEqual(proposal.receipt["carried_frozen_input_count"], 0)
        for key in (
            "adam_state_access_count", "adam_learning_rate_access_count",
            "adam_bias_correction_access_count", "request_raw_cap_access_count",
            "request_raw_cap_application_count", "semantic_velocity_decay_access_count",
        ):
            self.assertEqual(proposal.receipt[key], 0)
        expected = 0.125 * 0.4
        norms = proposal.receipt["nominal_displacement_norm"]
        self.assertAlmostEqual(norms[0], expected, places=8)
        self.assertEqual(norms[1], 0.0)
        self.assertAlmostEqual(norms[2], expected, places=8)

    def test_semantic_selection_reuses_endpoint_and_full_residual(self) -> None:
        state = P1R39ControllerState.zero(self.current)
        current = objective((1.0, 0.01, 2.0), self.gradient, backward=1)
        proposal = prepare_p1r39_target_proposal(
            self.current, self.terminal, self.z0, current,
            kl_result(3, torch.zeros_like(self.gradient)), self.lock, state,
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        selected = select_p1r39_target_proposal(
            proposal, self.current, self.terminal, current,
            objective((0.9, 0.01, 2.1), None, backward=0), step_index=0,
        )
        self.assertEqual(selected.receipt["proposal_accept_mask"], [True, False, False])
        self.assertEqual(selected.receipt["selection_added_model_forward_count"], 0)
        self.assertEqual(selected.receipt["selection_added_backward_count"], 0)
        self.assertEqual(selected.target_step.receipt["remaining_horizon_division_count"], 0)
        self.assertEqual(selected.target_step.receipt["semantic_debt_input_count"], 0)
        self.assertEqual(selected.target_step.receipt["physical_h_application_count"], 1)
        self.assertEqual(selected.target_step.receipt["second_h_application_count"], 0)
        expected = selected.target_step.target_next - self.terminal
        self.assertTrue(torch.equal(selected.target_step.required_displacement, expected))


if __name__ == "__main__":
    unittest.main()
