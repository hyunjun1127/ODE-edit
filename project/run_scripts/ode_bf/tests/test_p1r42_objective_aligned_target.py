from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p1r42_objective_aligned_target import (
    P1R42ControllerState,
    P1R42_DIRECTION_POLICY,
    prepare_p1r42_target_proposal,
    select_p1r42_target_proposal,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    ScalableObjectiveResult,
)


def objective(
    values: tuple[float, ...],
    gradient: torch.Tensor | None,
    *,
    backward: int,
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


def kl_result(
    values: tuple[float, ...], gradient: torch.Tensor
) -> P1R24KLResult:
    return P1R24KLResult(
        sum(values) / len(values),
        values,
        gradient,
        1,
        1,
        len(values),
        len(values),
        "7" * 64,
    )


class P1R42ObjectiveAlignedTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
        self.terminal = self.current.clone()
        self.z0 = self.current.clone()
        self.lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")

    def proposal(
        self,
        *,
        values: tuple[float, float] = (1.0, 2.0),
        semantic: torch.Tensor | None = None,
        preservation: torch.Tensor | None = None,
        kl_values: tuple[float, float] = (0.0, 0.0),
        previous: tuple[bool, bool] = (False, False),
    ):
        semantic = (
            torch.tensor([[0.3, -0.4], [0.1, 0.2]])
            if semantic is None
            else semantic
        )
        preservation = (
            torch.zeros_like(semantic) if preservation is None else preservation
        )
        return prepare_p1r42_target_proposal(
            self.current,
            self.terminal,
            self.z0,
            objective(values, semantic, backward=1),
            kl_result(kl_values, preservation),
            self.lock,
            P1R42ControllerState(previous),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
        )

    def test_conflicting_preservation_projection_and_fixed_normalization(self) -> None:
        semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        preservation = torch.tensor(
            [[-1.0 / self.lock.kl_factor, 0.0], [0.0, 1.0 / self.lock.kl_factor]]
        )
        proposal = self.proposal(
            semantic=semantic,
            preservation=preservation,
        )
        receipt = proposal.receipt
        self.assertEqual(receipt["target_direction_policy"], P1R42_DIRECTION_POLICY)
        self.assertGreater(receipt["raw_preservation_conflict_inner_product"][0], 0.0)
        self.assertLess(receipt["raw_preservation_conflict_inner_product"][1], 0.0)
        self.assertGreater(receipt["removed_conflicting_component_norm"][0], 0.0)
        self.assertEqual(receipt["removed_conflicting_component_norm"][1], 0.0)
        self.assertLessEqual(receipt["safe_inner_product_max"], 1.0e-8)
        expected = 0.125 * 0.4
        for value in receipt["nominal_displacement_norm"]:
            self.assertAlmostEqual(value, expected, places=8)
        self.assertEqual(receipt["global_frobenius_normalization_count"], 0)

    def test_semantic_hold_old_composite_counterfactual_and_reactivation(self) -> None:
        proposal = self.proposal(
            values=(0.01, 0.2),
            kl_values=(100.0, 0.0),
            previous=(False, True),
        )
        self.assertEqual(proposal.semantic_held_mask, (True, False))
        self.assertEqual(proposal.active_mask, (False, True))
        self.assertEqual(proposal.reactivated_mask, (False, True))
        self.assertEqual(
            proposal.receipt["old_active_but_semantic_held_mask"],
            [True, False],
        )
        self.assertEqual(proposal.receipt["persistent_mask_input_count"], 0)
        self.assertEqual(
            proposal.receipt["persistent_mask_decision_influence_count"], 0
        )

    def test_zero_semantic_gradient_forbids_preservation_only_move(self) -> None:
        semantic = torch.zeros((2, 2))
        preservation = torch.ones((2, 2))
        proposal = self.proposal(
            semantic=semantic,
            preservation=preservation,
        )
        self.assertEqual(proposal.active_mask, (False, False))
        self.assertEqual(
            proposal.receipt["semantic_zero_gradient_mask"], [True, True]
        )
        self.assertEqual(proposal.receipt["nominal_displacement_norm"], [0.0, 0.0])
        self.assertEqual(
            proposal.receipt["preservation_only_target_movement_count"], 0
        )

    def test_finite_semantic_accept_and_held_writer_full_residual(self) -> None:
        current = objective(
            (1.0, 0.01),
            torch.tensor([[0.3, 0.2], [0.1, -0.1]]),
            backward=1,
        )
        proposal = prepare_p1r42_target_proposal(
            self.current,
            self.terminal,
            self.z0,
            current,
            kl_result((0.0, 0.0), torch.zeros((2, 2))),
            self.lock,
            P1R42ControllerState.zero(self.current),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
        )
        selected = select_p1r42_target_proposal(
            proposal,
            self.current,
            self.terminal,
            current,
            objective((0.9, 0.0), None, backward=0),
            step_index=0,
        )
        self.assertEqual(selected.receipt["proposal_accept_mask"], [True, False])
        self.assertTrue(selected.receipt["held_writer_continues_full_residual"])
        self.assertIsNotNone(
            selected.receipt["held_writer_residual_norm_by_request"][1]
        )
        expected = selected.target_step.target_next - self.terminal
        self.assertTrue(torch.equal(selected.target_step.required_displacement, expected))
        self.assertEqual(selected.target_step.receipt["physical_h_application_count"], 1)
        self.assertEqual(selected.target_step.receipt["second_h_application_count"], 0)
        self.assertEqual(selected.receipt["selection_added_model_forward_count"], 0)
        self.assertEqual(selected.receipt["selection_added_backward_count"], 0)


if __name__ == "__main__":
    unittest.main()
