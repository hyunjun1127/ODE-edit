from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24_H, P1R24TargetStep
from project.run_scripts.ode_bf.p1r35_full_current_residual import (
    apply_p1r35_full_current_residual,
)
from project.run_scripts.ode_bf.p1r38_perrequest_target import (
    P1R38AdamState,
    P1R38TargetProposal,
)
from project.run_scripts.ode_bf.p1r40_semantic_deficit_velocity_decay import (
    P1R40_METHOD_ID,
    apply_p1r40_semantic_deficit_velocity_decay,
    select_p1r40_target_proposal,
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


def nominal_proposal(
    current: torch.Tensor, nominal_next: torch.Tensor, gradient: torch.Tensor
) -> P1R38TargetProposal:
    delta = nominal_next - current
    required = delta.clone()
    velocity = required / P1R24_H
    receipt = {
        "schema": "test-p1r38-proposal/v1",
        "identity_sha256": canonical_hash({"nominal": nominal_next.tolist()}),
    }
    step = P1R24TargetStep(
        nominal_next.contiguous(),
        delta.contiguous(),
        required.contiguous(),
        velocity.contiguous(),
        gradient.contiguous(),
        gradient.contiguous(),
        0.0,
        0.0,
        0.0,
        tuple(False for _ in range(current.shape[1])),
        receipt,
    )
    zeros = torch.zeros_like(current, dtype=torch.float64)
    return P1R38TargetProposal(
        step,
        zeros,
        zeros,
        torch.zeros(current.shape[1], dtype=torch.int64),
        tuple(True for _ in range(current.shape[1])),
        tuple(False for _ in range(current.shape[1])),
        tuple(False for _ in range(current.shape[1])),
        tuple(1.0 for _ in range(current.shape[1])),
        tuple(float(torch.linalg.vector_norm(delta[:, index])) for index in range(current.shape[1])),
        tuple(float(torch.linalg.vector_norm(delta[:, index])) for index in range(current.shape[1])),
        tuple(False for _ in range(current.shape[1])),
        tuple(False for _ in range(current.shape[1])),
        receipt,
    )


class P1R40SemanticDeficitVelocityDecayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.zeros((2, 3), dtype=torch.float32)
        self.terminal = self.current.clone()
        self.nominal = torch.tensor(
            [[-2.0, -0.1, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        # Batched columns include the exact 1/B factor.  Request-local gains
        # are 3.0, 0.06, and -0.3 after multiplying by B=3.
        self.batched_gradient = torch.tensor(
            [[0.5, 0.2, 0.1], [0.0, 0.0, 0.0]], dtype=torch.float32
        )

    def test_request_demeaning_and_piecewise_gamma(self) -> None:
        proposal = apply_p1r40_semantic_deficit_velocity_decay(
            nominal_proposal(self.current, self.nominal, self.batched_gradient),
            self.current,
            self.terminal,
            objective((0.3, 1.0, 2.0), self.batched_gradient, backward=1),
            step_index=0,
        )
        receipt = proposal.receipt
        self.assertEqual(receipt["batch_demean_factor"], 3)
        self.assertEqual(receipt["q_batch_demean_identity_max_abs"], 0.0)
        self.assertAlmostEqual(receipt["q_signed_by_request"][0], 3.0)
        self.assertAlmostEqual(receipt["q_signed_by_request"][1], 0.06, places=7)
        self.assertAlmostEqual(receipt["q_signed_by_request"][2], -0.3, places=7)
        self.assertAlmostEqual(receipt["gamma_by_request"][0], 0.1, places=7)
        self.assertEqual(receipt["gamma_by_request"][1:], [1.0, 1.0])
        self.assertEqual(receipt["gamma_lt_one_count"], 1)
        self.assertNotEqual(
            receipt["nominal_target_column_sha256"][0],
            receipt["scaled_target_column_sha256"][0],
        )

    def test_gamma_is_target_only_and_writer_uses_full_current_residual(self) -> None:
        proposal = apply_p1r40_semantic_deficit_velocity_decay(
            nominal_proposal(self.current, self.nominal, self.batched_gradient),
            self.current,
            self.terminal,
            objective((0.3, 1.0, 2.0), self.batched_gradient, backward=1),
            step_index=3,
        )
        writer = apply_p1r35_full_current_residual(
            proposal.trial_step,
            current_target=self.current,
            current_terminal=self.terminal,
            step_index=3,
        )
        torch.testing.assert_close(
            writer.required_displacement,
            writer.target_next - self.terminal,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(proposal.receipt["gamma_target_displacement_application_count"], 1)
        self.assertEqual(proposal.receipt["gamma_target_to_writer_application_count"], 0)
        self.assertEqual(writer.receipt["remaining_horizon_division_count"], 0)
        self.assertEqual(writer.receipt["second_h_application_count"], 0)

    def test_existing_semantic_selector_is_reused_without_model_compute(self) -> None:
        base = nominal_proposal(self.current, self.nominal, self.batched_gradient)
        proposal = apply_p1r40_semantic_deficit_velocity_decay(
            base,
            self.current,
            self.terminal,
            objective((0.3, 1.0, 2.0), self.batched_gradient, backward=1),
            step_index=0,
        )
        state = P1R38AdamState.zero(self.current)
        selected = select_p1r40_target_proposal(
            proposal,
            state,
            self.current,
            self.terminal,
            objective((0.3, 1.0, 2.0), self.batched_gradient, backward=1),
            objective((0.2, 0.9, 1.9), None, backward=0),
            step_index=0,
        )
        self.assertEqual(selected.receipt["method_id"], P1R40_METHOD_ID)
        self.assertEqual(selected.receipt["accept_count"], 3)
        self.assertEqual(selected.receipt["selection_added_model_forward_count"], 0)
        self.assertEqual(selected.receipt["selection_added_backward_count"], 0)

    def test_controller_totality_for_zero_and_negative_gain(self) -> None:
        zero_nominal = self.current.clone()
        proposal = apply_p1r40_semantic_deficit_velocity_decay(
            nominal_proposal(self.current, zero_nominal, self.batched_gradient),
            self.current,
            self.terminal,
            objective((0.0, 0.2, 1.0), self.batched_gradient, backward=1),
            step_index=7,
        )
        self.assertEqual(proposal.receipt["gamma_by_request"], [1.0, 1.0, 1.0])
        self.assertEqual(proposal.receipt["gamma_lt_one_count"], 0)


if __name__ == "__main__":
    unittest.main()
