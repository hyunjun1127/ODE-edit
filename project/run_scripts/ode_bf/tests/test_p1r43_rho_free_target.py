from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r43_rho_free_target import (
    P1R43ControllerState,
    prepare_p1r43_rescue_proposal,
    prepare_p1r43_target_proposal,
    select_p1r43_target_proposal,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(
    values: tuple[float, ...], gradient: torch.Tensor | None, *, backward: int
) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values),
        values,
        "1" * 64,
        "2" * 64,
        tuple("3" * 64 for _ in values),
        tuple(1 for _ in values),
        1,
        backward,
        10,
        10,
        gradient,
        None,
        "4" * 64,
        "5" * 64,
        "6" * 64,
    )


class P1R43RhoFreeTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor([[2.0, 3.0], [1.0, 1.0]])
        self.terminal = self.target.clone()
        self.gradient = torch.tensor([[0.5, 0.0], [0.0, 1.0]]) / 2.0

    def proposal(self, state: P1R43ControllerState, step: int = 0):
        return prepare_p1r43_target_proposal(
            self.target,
            self.terminal,
            objective((1.0, 2.0), self.gradient, backward=1),
            state,
            alias="llama3-8b-inst",
            step_index=step,
            shared_speed=0.4,
        )

    def test_entry_norm_freezes_and_current_gradient_controls_magnitude(self) -> None:
        first = self.proposal(P1R43ControllerState.zero(self.target))
        second = prepare_p1r43_target_proposal(
            self.target,
            self.terminal,
            objective((1.0, 2.0), 0.5 * self.gradient, backward=1),
            first.next_state,
            alias="llama3-8b-inst",
            step_index=1,
            shared_speed=0.4,
        )
        self.assertEqual(
            first.next_state.entry_semantic_gradient_norm,
            second.next_state.entry_semantic_gradient_norm,
        )
        first_norm = torch.linalg.vector_norm(first.primary_delta, dim=0)
        second_norm = torch.linalg.vector_norm(second.primary_delta, dim=0)
        self.assertTrue(torch.allclose(second_norm, 0.5 * first_norm))
        self.assertEqual(first.receipt["trust_rho_decision_influence_count"], 0)
        self.assertEqual(first.receipt["semantic_hold_threshold_access_count"], 0)
        self.assertEqual(first.receipt["target_kl_decision_influence_count"], 0)

    def test_primary_improving_is_accepted_without_rho(self) -> None:
        proposal = self.proposal(P1R43ControllerState.zero(self.target))
        current = objective((1.0, 2.0), self.gradient, backward=1)
        primary = objective((0.9, 1.9), None, backward=0)
        rescue = prepare_p1r43_rescue_proposal(
            proposal, self.target, self.terminal, current, primary, step_index=0
        )
        selected = select_p1r43_target_proposal(
            proposal,
            rescue,
            self.target,
            self.terminal,
            current,
            primary,
            None,
            step_index=0,
        )
        self.assertEqual(selected.receipt["selection_by_request"], ["PRIMARY", "PRIMARY"])
        self.assertEqual(selected.receipt["trust_rho_decision_influence_count"], 0)
        self.assertEqual(selected.target_step.receipt["remaining_horizon_division_count"], 0)

    def test_worsening_primary_can_use_exactly_one_rescue_forward(self) -> None:
        proposal = self.proposal(P1R43ControllerState.zero(self.target))
        current = objective((1.0, 2.0), self.gradient, backward=1)
        primary = objective((1.2, 2.2), None, backward=0)
        rescue = prepare_p1r43_rescue_proposal(
            proposal, self.target, self.terminal, current, primary, step_index=0
        )
        self.assertLessEqual(rescue.receipt["maximum_rescue_forward_count"], 1)
        if rescue.rescue_step is None:
            self.assertEqual(rescue.receipt["rescue_eligible_count"], 0)
            return
        rescue_endpoint = objective((0.95, 1.95), None, backward=0)
        selected = select_p1r43_target_proposal(
            proposal,
            rescue,
            self.target,
            self.terminal,
            current,
            primary,
            rescue_endpoint,
            step_index=0,
        )
        self.assertEqual(selected.receipt["same_step_scalar_corrector_physical_k_count"], 0)
        self.assertEqual(selected.receipt["same_step_scalar_corrector_backward_count"], 0)


if __name__ == "__main__":
    unittest.main()
