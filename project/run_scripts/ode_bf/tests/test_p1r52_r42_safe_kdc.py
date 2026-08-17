from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
    prepare_p1r51_target_proposal,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    _origin_relative_clamp,
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(values: tuple[float, ...], gradient: torch.Tensor | None, *, backward: int) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, backward, 10, 10, gradient, None, "4" * 64, "5" * 64, "6" * 64,
    )


def kl_result(values: tuple[float, ...], gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(
        sum(values) / len(values), values, gradient, 1, 1,
        len(values), len(values), "7" * 64,
    )


class P1R52R42SafeKDCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float32)
        self.terminal = self.current.clone()
        self.origin = self.current.clone()
        self.gradient = torch.tensor([[0.25, -0.5], [0.5, 0.25]], dtype=torch.float32) / 2.0
        self.nll = objective((1.0, 2.0), self.gradient, backward=1)

    def proposal(self, *, lock: P1R24AliasTargetLock | None = None, kl_gradient: torch.Tensor | None = None, current: torch.Tensor | None = None):
        lock = lock or P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        current = self.current if current is None else current
        return prepare_p1r52_target_proposal(
            current, self.terminal, self.origin, self.nll,
            kl_result((0.0, 0.0), torch.zeros_like(self.gradient) if kl_gradient is None else kl_gradient),
            lock, P1R51ControllerState.zero(current), alias="llama3-8b-inst",
            step_index=0, shared_speed=0.4,
        )

    def test_zero_kl_decay_clamp_off_equivalence_to_p1r51(self) -> None:
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9)
        p52 = self.proposal(lock=lock)
        p51 = prepare_p1r51_target_proposal(
            self.current, self.terminal, self.nll, P1R51ControllerState.zero(self.current),
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        self.assertTrue(torch.allclose(p52.primary_delta, p51.primary_delta, atol=1e-8, rtol=1e-7))
        self.assertTrue(torch.equal(p52.primary_step.target_next, p51.primary_step.target_next))

    def test_k0_native_lock_equivalence_and_zero_decay(self) -> None:
        p52 = self.proposal()
        p51 = prepare_p1r51_target_proposal(
            self.current, self.terminal, self.nll, P1R51ControllerState.zero(self.current),
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        self.assertTrue(torch.allclose(p52.primary_delta, p51.primary_delta, atol=1e-8, rtol=1e-7))
        self.assertEqual(p52.receipt["decay_by_request"], [0.0, 0.0])
        self.assertEqual(p52.receipt["target_decay_decision_influence_count"], 0)

    def test_one_sided_certificate_and_nonconflicting_preservation(self) -> None:
        # Column 0 conflicts and is projected; column 1 is nonconflicting.
        preservation = torch.tensor([[-8.0, 8.0], [-16.0, -4.0]], dtype=torch.float32)
        proposal = self.proposal(kl_gradient=preservation)
        receipt = proposal.receipt
        self.assertLessEqual(receipt["safe_preservation_certificate_max"], 1e-8)
        self.assertGreaterEqual(receipt["removed_conflicting_component_norm_by_request"][0], 0.0)
        self.assertEqual(receipt["added_kl_model_forward_count"], 0)
        self.assertEqual(receipt["added_kl_backward_count"], 0)

    def test_request_permutation_equivariance(self) -> None:
        direct = self.proposal()
        permutation = torch.tensor([1, 0])
        permuted = prepare_p1r52_target_proposal(
            self.current[:, permutation], self.terminal[:, permutation], self.origin[:, permutation],
            objective((2.0, 1.0), self.gradient[:, permutation], backward=1),
            kl_result((0.0, 0.0), torch.zeros_like(self.gradient[:, permutation])),
            P1R24AliasTargetLock.for_alias("llama3-8b-inst"),
            P1R51ControllerState.zero(self.current[:, permutation]),
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        self.assertTrue(torch.allclose(direct.primary_delta[:, permutation], permuted.primary_delta, atol=1e-8, rtol=1e-7))

    def test_origin_clamp_bound_and_idempotence(self) -> None:
        origin = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
        candidate = torch.tensor([[4.0, 2.1], [4.0, 0.1]], dtype=torch.float64)
        once, ratio, maximum = _origin_relative_clamp(candidate, origin, 0.5)
        twice, _, _ = _origin_relative_clamp(once, origin, 0.5)
        self.assertTrue(torch.allclose(once, twice, atol=1e-12, rtol=0.0))
        self.assertTrue(torch.all(torch.linalg.vector_norm(once - origin, dim=0) <= maximum + 1e-12))
        self.assertLess(float(ratio[0]), 1.0)

    def test_rescue_uses_post_clamp_post_cast_actual_delta(self) -> None:
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 0.01)
        proposal = self.proposal(lock=lock)
        endpoint = objective((1.2, 2.2), None, backward=0)
        rescue = prepare_p1r52_rescue_proposal(
            proposal, self.current, self.terminal, self.nll, endpoint, step_index=0
        )
        expected = torch.sum(proposal.semantic_gradient * proposal.primary_delta, dim=0)
        self.assertTrue(torch.allclose(torch.tensor(rescue.directional_derivative_by_request, dtype=expected.dtype), expected))
        self.assertEqual(rescue.receipt["directional_derivative_delta_source"], "POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA")
        self.assertEqual(proposal.primary_step.target_next.dtype, torch.float32)
        self.assertTrue(torch.isfinite(proposal.primary_delta).all())


if __name__ == "__main__":
    unittest.main()
