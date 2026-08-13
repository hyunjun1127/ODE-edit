from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from project.run_scripts.ode_bf.p1r38_perrequest_target import (
    P1R38AdamState,
    P1R38_SEMANTIC_EPSILON,
    prepare_p1r38_target_proposal,
    select_p1r38_target_proposal,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(values: tuple[float, ...], gradient: torch.Tensor | None, *, backward: int) -> ScalableObjectiveResult:
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


class P1R38PerRequestTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.tensor(
            [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
        self.terminal = self.current.clone()
        self.z0 = self.current.clone()
        self.gradient = torch.tensor(
            [[0.3, 0.2, -0.4], [0.1, -0.1, 0.2]], dtype=torch.float32
        )
        self.lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")

    def test_reversible_hold_cap_and_no_carry(self) -> None:
        state = P1R38AdamState.zero(self.current)
        state = P1R38AdamState(state.m, state.v, state.t, (True, False, False))
        proposal = prepare_p1r38_target_proposal(
            self.current,
            self.terminal,
            self.z0,
            objective((1.0, 0.01, 2.0), self.gradient, backward=1),
            kl_result(3, torch.zeros_like(self.gradient)),
            self.lock,
            state,
            alias="llama3-8b-inst",
            step_index=0,
            request_cap_radius=0.05,
        )
        self.assertEqual(proposal.active_mask, (True, False, True))
        self.assertEqual(proposal.reactivated_mask, (True, False, False))
        self.assertTrue(proposal.receipt["current_state_instantaneous_no_carry"] if "current_state_instantaneous_no_carry" in proposal.receipt else proposal.receipt["hold_mode"] == "CURRENT_STATE_INSTANTANEOUS_NO_CARRY")
        self.assertEqual(proposal.receipt["persistent_mask_input_count"], 0)
        self.assertEqual(proposal.receipt["global_frobenius_normalization_count"], 0)
        self.assertLessEqual(max(proposal.applied_delta_norm), 0.05 + 1e-8)
        self.assertEqual(proposal.applied_delta_norm[1], 0.0)
        self.assertEqual(proposal.receipt["added_model_forward_count"], 0)
        self.assertEqual(proposal.receipt["added_backward_count"], 0)

    def test_semantic_selection_commits_only_accepted_and_resets_others(self) -> None:
        state = P1R38AdamState.zero(self.current)
        proposal = prepare_p1r38_target_proposal(
            self.current,
            self.terminal,
            self.z0,
            objective((1.0, 0.01, 2.0), self.gradient, backward=1),
            kl_result(3, torch.zeros_like(self.gradient)),
            self.lock,
            state,
            alias="llama3-8b-inst",
            step_index=0,
            request_cap_radius=10.0,
        )
        selected = select_p1r38_target_proposal(
            proposal,
            state,
            self.current,
            self.terminal,
            objective((1.0, 0.01, 2.0), self.gradient, backward=1),
            objective((0.9, 0.01, 2.1), None, backward=0),
            step_index=0,
        )
        self.assertEqual(selected.receipt["proposal_accept_mask"], [True, False, False])
        self.assertEqual(selected.receipt["accept_count"], 1)
        self.assertEqual(selected.receipt["reject_count"], 1)
        self.assertEqual(tuple(int(item) for item in selected.next_state.t), (1, 0, 0))
        self.assertTrue(torch.count_nonzero(selected.next_state.m[:, 1:]) == 0)
        self.assertEqual(selected.selected_endpoint.per_request_values, (0.9, 0.01, 2.0))
        self.assertEqual(selected.receipt["selection_added_model_forward_count"], 0)
        self.assertEqual(selected.receipt["selection_added_backward_count"], 0)

    def test_semantic_epsilon_is_inherited_and_upper_cap_has_no_floor(self) -> None:
        self.assertEqual(P1R38_SEMANTIC_EPSILON, 1.0e-8)
        state = P1R38AdamState.zero(self.current[:, :1])
        gradient = self.gradient[:, :1]
        proposal = prepare_p1r38_target_proposal(
            self.current[:, :1],
            self.terminal[:, :1],
            self.z0[:, :1],
            objective((1.0,), gradient, backward=1),
            kl_result(1, torch.zeros_like(gradient)),
            self.lock,
            state,
            alias="llama3-8b-inst",
            step_index=0,
            request_cap_radius=100.0,
        )
        self.assertFalse(proposal.cap_hit[0])
        self.assertAlmostEqual(proposal.raw_delta_norm[0], proposal.applied_delta_norm[0], places=12)


if __name__ == "__main__":
    unittest.main()
