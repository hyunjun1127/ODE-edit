from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p1r41_trust_clipped_target import (
    P1R41ControllerState,
    P1R41_DIRECTION_POLICY,
    prepare_p1r41_target_proposal,
    select_p1r41_target_proposal,
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


def kl_result(count: int, gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(
        0.0,
        tuple(0.0 for _ in range(count)),
        gradient,
        1,
        1,
        count,
        count,
        "7" * 64,
    )


class P1R41TrustClippedTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.tensor([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        self.terminal = self.current.clone()
        self.z0 = self.current.clone()
        self.gradient = torch.tensor([[0.3, 0.2, -0.4], [0.1, -0.1, 0.2]])
        self.lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        self.speed = 0.4

    def _prepare(
        self,
        state: P1R41ControllerState,
        *,
        step_index: int,
        values: tuple[float, ...] = (1.0, 0.01, 2.0),
    ):
        current = objective(values, self.gradient, backward=1)
        proposal = prepare_p1r41_target_proposal(
            self.current,
            self.terminal,
            self.z0,
            current,
            kl_result(3, torch.zeros_like(self.gradient)),
            self.lock,
            state,
            alias="llama3-8b-inst",
            step_index=step_index,
            shared_speed=self.speed,
        )
        return current, proposal

    def test_entry_normalization_trust_clip_and_forbidden_access(self) -> None:
        state = P1R41ControllerState(
            (True, False, False), (), (), (0, 0, 0)
        )
        _, proposal = self._prepare(state, step_index=0)
        receipt = proposal.receipt
        self.assertEqual(receipt["target_direction_policy"], P1R41_DIRECTION_POLICY)
        self.assertEqual(proposal.active_mask, (True, False, True))
        self.assertEqual(proposal.reactivated_mask, (True, False, False))
        self.assertEqual(receipt["hold_mode"], "CURRENT_STATE_INSTANTANEOUS_NO_CARRY")
        nominal = 0.125 * self.speed
        self.assertAlmostEqual(receipt["trust_radius_before"][0], nominal)
        self.assertAlmostEqual(receipt["trust_radius_min"][0], nominal / 16.0)
        self.assertAlmostEqual(receipt["trust_radius_max"][0], 2.0 * nominal)
        for index in (0, 2):
            self.assertLessEqual(
                receipt["post_trust_displacement_norm"][index],
                receipt["trust_radius_before"][index] + 1.0e-8,
            )
        self.assertEqual(receipt["post_trust_displacement_norm"][1], 0.0)
        for key in (
            "adam_state_access_count",
            "adam_learning_rate_access_count",
            "adam_bias_correction_access_count",
            "request_raw_cap_access_count",
            "semantic_velocity_decay_access_count",
            "objective_alignment_access_count",
            "added_model_forward_count",
            "added_backward_count",
            "added_materialization_count",
            "radius_same_step_retry_count",
        ):
            self.assertEqual(receipt[key], 0)

    def test_radius_truth_table_and_next_step_only(self) -> None:
        state = P1R41ControllerState.zero(self.current)
        current, proposal = self._prepare(
            state, step_index=0, values=(1.0, 1.0, 1.0)
        )
        q = []
        trial = proposal.trial_step.target_next.to(torch.float64) - self.current.to(
            torch.float64
        )
        pure = proposal.pure_request_gradient
        for index in range(3):
            q.append(float(-torch.sum(pure[:, index] * trial[:, index])))
        trial_values = (
            1.1,
            1.0 - 0.1 * q[1],
            1.0 - 0.9 * q[2],
        )
        selected = select_p1r41_target_proposal(
            proposal,
            self.current,
            self.terminal,
            current,
            objective(trial_values, None, backward=0),
            state,
            step_index=0,
        )
        nominal = 0.125 * self.speed
        self.assertEqual(
            selected.receipt["radius_update_reason"],
            [
                "SEMANTIC_REJECT_SHRINK",
                "LOW_TRUST_RATIO_SHRINK",
                "ACCEPT_BOUNDARY_HIGH_RATIO_EXPAND",
            ],
        )
        self.assertEqual(
            selected.receipt["trust_radius_after"],
            [0.5 * nominal, 0.5 * nominal, 2.0 * nominal],
        )
        self.assertEqual(selected.receipt["same_step_retry_count"], 0)
        self.assertTrue(selected.receipt["radius_update_applies_next_global_step_only"])
        self.assertEqual(selected.next_state.consecutive_reject_count, (1, 0, 0))

    def test_held_radius_is_stable_and_reactivation_reuses_it(self) -> None:
        nominal = 0.125 * self.speed
        held_state = P1R41ControllerState(
            (False, False, False),
            (1.0, 1.0, 1.0),
            (nominal, 0.5 * nominal, nominal),
            (0, 2, 0),
        )
        held_current, held_proposal = self._prepare(
            held_state, step_index=1, values=(1.0, 0.01, 2.0)
        )
        held_selected = select_p1r41_target_proposal(
            held_proposal,
            self.current,
            self.terminal,
            held_current,
            objective((0.9, 0.01, 1.9), None, backward=0),
            held_state,
            step_index=1,
        )
        self.assertEqual(held_selected.receipt["radius_update_reason"][1], "HELD_KEEP")
        self.assertEqual(held_selected.next_state.trust_radius[1], 0.5 * nominal)
        self.assertEqual(held_selected.next_state.consecutive_reject_count[1], 2)

        reactivated_current, reactivated = self._prepare(
            held_selected.next_state, step_index=2, values=(1.0, 1.0, 2.0)
        )
        self.assertTrue(reactivated.reactivated_mask[1])
        self.assertEqual(reactivated.radius_before[1], 0.5 * nominal)
        self.assertEqual(reactivated_current.backward_count, 1)

    def test_selection_reuses_endpoint_and_full_current_residual(self) -> None:
        state = P1R41ControllerState.zero(self.current)
        current, proposal = self._prepare(state, step_index=0)
        selected = select_p1r41_target_proposal(
            proposal,
            self.current,
            self.terminal,
            current,
            objective((0.9, 0.01, 2.1), None, backward=0),
            state,
            step_index=0,
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

    def test_case_state_reset(self) -> None:
        first = P1R41ControllerState.zero(self.current)
        second = P1R41ControllerState.zero(self.current)
        self.assertEqual(first, second)
        self.assertEqual(first.entry_gradient_norm, ())
        self.assertEqual(first.trust_radius, ())
        self.assertEqual(first.consecutive_reject_count, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
