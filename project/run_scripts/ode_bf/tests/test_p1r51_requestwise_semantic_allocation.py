from __future__ import annotations

import math
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r43_rho_free_target import (
    P1R43ControllerState,
    prepare_p1r43_rescue_proposal,
    prepare_p1r43_target_proposal,
    select_p1r43_target_proposal,
)
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
    prepare_p1r51_rescue_proposal,
    prepare_p1r51_target_proposal,
    select_p1r51_target_proposal,
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


class P1R51RequestwiseSemanticAllocationTests(unittest.TestCase):
    def test_b1_equivalence_to_parent_field_and_selection(self) -> None:
        target = torch.tensor([[2.0], [1.0]], dtype=torch.float32)
        terminal = target.clone()
        gradient = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
        current = objective((1.25,), gradient, backward=1)
        parent = prepare_p1r43_target_proposal(
            target,
            terminal,
            current,
            P1R43ControllerState.zero(target),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
        )
        rsa = prepare_p1r51_target_proposal(
            target,
            terminal,
            current,
            P1R51ControllerState.zero(target),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
        )
        cosine = torch.nn.functional.cosine_similarity(
            parent.primary_delta.flatten(), rsa.primary_delta.flatten(), dim=0
        )
        self.assertAlmostEqual(float(cosine), 1.0, places=7)
        self.assertTrue(torch.allclose(parent.primary_delta, rsa.primary_delta, atol=1e-8, rtol=1e-7))
        parent_endpoint = objective((1.0,), None, backward=0)
        parent_rescue = prepare_p1r43_rescue_proposal(
            parent, target, terminal, current, parent_endpoint, step_index=0
        )
        rsa_rescue = prepare_p1r51_rescue_proposal(
            rsa, target, terminal, current, parent_endpoint, step_index=0
        )
        parent_selected = select_p1r43_target_proposal(
            parent, parent_rescue, target, terminal, current, parent_endpoint, None, step_index=0
        )
        rsa_selected = select_p1r51_target_proposal(
            rsa, rsa_rescue, target, terminal, current, parent_endpoint, None, step_index=0
        )
        self.assertEqual(
            parent_selected.receipt["selection_by_request"],
            rsa_selected.receipt["selection_by_request"],
        )
        self.assertTrue(torch.allclose(parent_selected.target_step.target_next, rsa_selected.target_step.target_next))

    def test_b10_energy_conservation_and_urgency_allocation(self) -> None:
        target = torch.arange(40, dtype=torch.float32).reshape(4, 10) + 1.0
        terminal = target.clone()
        gradient = torch.arange(1, 41, dtype=torch.float32).reshape(4, 10) / 10.0
        values = tuple(float(index + 1) for index in range(10))
        proposal = prepare_p1r51_target_proposal(
            target,
            terminal,
            objective(values, gradient / 10.0, backward=1),
            P1R51ControllerState.zero(target),
            alias="qwen2.5-7b-inst",
            step_index=0,
            shared_speed=0.5,
        )
        self.assertLessEqual(proposal.receipt["energy_relative_error"], 1e-8)
        self.assertAlmostEqual(
            sum(proposal.receipt["allocation_energy_share_by_request"]), 1.0, places=9
        )
        self.assertGreater(
            proposal.receipt["allocation_amplitude_by_request"][-1],
            proposal.receipt["allocation_amplitude_by_request"][0],
        )

    def test_permutation_equivariance(self) -> None:
        target = torch.arange(30, dtype=torch.float32).reshape(3, 10) + 1.0
        terminal = target.clone()
        gradient = torch.arange(1, 31, dtype=torch.float32).reshape(3, 10) / 50.0
        values = tuple(0.25 + index for index in range(10))
        base = prepare_p1r51_target_proposal(
            target,
            terminal,
            objective(values, gradient / 10.0, backward=1),
            P1R51ControllerState.zero(target),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.75,
        )
        permutation = torch.tensor([7, 2, 9, 0, 4, 1, 8, 5, 3, 6])
        permuted = prepare_p1r51_target_proposal(
            target[:, permutation],
            terminal[:, permutation],
            objective(tuple(values[i] for i in permutation), gradient[:, permutation] / 10.0, backward=1),
            P1R51ControllerState.zero(target[:, permutation]),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.75,
        )
        self.assertTrue(torch.allclose(base.primary_delta[:, permutation], permuted.primary_delta))
        expected = torch.tensor(base.receipt["allocation_amplitude_by_request"])[permutation]
        observed = torch.tensor(permuted.receipt["allocation_amplitude_by_request"])
        self.assertTrue(torch.allclose(expected, observed))

    def test_flat_zero_and_nonfinite_states_are_distinct(self) -> None:
        target = torch.ones((3, 2), dtype=torch.float32)
        zero = prepare_p1r51_target_proposal(
            target,
            target,
            objective((1.0, 2.0), torch.zeros_like(target), backward=1),
            P1R51ControllerState.zero(target),
            alias="qwen2.5-7b-inst",
            step_index=0,
            shared_speed=0.5,
        )
        self.assertEqual(zero.receipt["allocation_status"], "ZERO_NOMINAL_ENERGY")
        self.assertEqual(zero.receipt["flat_gradient_count"], 2)
        self.assertEqual(zero.receipt["rsa_total_velocity_energy"], 0.0)
        bad_gradient = torch.zeros_like(target)
        bad_gradient[0, 0] = math.nan
        with self.assertRaises(ODEBFContractError):
            prepare_p1r51_target_proposal(
                target,
                target,
                objective((1.0, 2.0), bad_gradient, backward=1),
                P1R51ControllerState.zero(target),
                alias="qwen2.5-7b-inst",
                step_index=0,
                shared_speed=0.5,
            )
        with self.assertRaises(ODEBFContractError):
            prepare_p1r51_target_proposal(
                target,
                target,
                objective((1.0, math.inf), torch.ones_like(target), backward=1),
                P1R51ControllerState.zero(target),
                alias="qwen2.5-7b-inst",
                step_index=0,
                shared_speed=0.5,
            )

    def test_selection_updates_cumulative_accepted_path(self) -> None:
        target = torch.tensor([[2.0, 3.0], [1.0, 1.0]])
        gradient = torch.tensor([[0.5, 0.0], [0.0, 1.0]]) / 2.0
        current = objective((1.0, 2.0), gradient, backward=1)
        proposal = prepare_p1r51_target_proposal(
            target,
            target,
            current,
            P1R51ControllerState.zero(target),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
        )
        endpoint = objective((0.9, 1.9), None, backward=0)
        rescue = prepare_p1r51_rescue_proposal(
            proposal, target, target, current, endpoint, step_index=0
        )
        selected = select_p1r51_target_proposal(
            proposal, rescue, target, target, current, endpoint, None, step_index=0
        )
        self.assertTrue(all(value > 0.0 for value in selected.next_state.cumulative_accepted_activation_path))
        self.assertEqual(selected.receipt["selection_by_request"], ["PRIMARY", "PRIMARY"])
        self.assertEqual(selected.receipt["allocation_only_decision_influence_count"], 1)


if __name__ == "__main__":
    unittest.main()
