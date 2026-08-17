from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
    prepare_p1r51_target_proposal,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    P1R52ActiveDirectionContractError,
    P1R52SemanticDescentContractError,
    _cast_origin_clamped_target_fp32,
    _origin_relative_clamp,
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
    select_p1r52_target_proposal,
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
    TEACHER_SHA256 = "8" * 64

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
            kl_teacher_input_sha256=self.TEACHER_SHA256,
        )

    def test_zero_kl_decay_clamp_off_equivalence_to_p1r51(self) -> None:
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9)
        p52 = self.proposal(lock=lock)
        p51 = prepare_p1r51_target_proposal(
            self.current, self.terminal, self.nll, P1R51ControllerState.zero(self.current),
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
        )
        semantic = self.gradient.to(dtype=torch.float64) * self.current.shape[1]
        expected_direction = -semantic / torch.linalg.vector_norm(semantic, dim=0).unsqueeze(0)
        expected_velocity = expected_direction * torch.tensor(
            p51.receipt["allocation_amplitude_by_request"], dtype=torch.float64
        ).unsqueeze(0)
        self.assertTrue(torch.equal(p52.kdc_direction, expected_direction))
        self.assertTrue(torch.equal(p52.raw_velocity, expected_velocity))
        self.assertEqual(
            p52.receipt["active_gradient_mask"], p51.receipt["active_gradient_mask"]
        )
        self.assertEqual(
            p52.receipt["entry_semantic_gradient_norm_by_request"],
            p51.receipt["entry_semantic_gradient_norm_by_request"],
        )
        self.assertEqual(
            p52.receipt["allocation_amplitude_by_request"],
            p51.receipt["allocation_amplitude_by_request"],
        )
        self.assertLessEqual(p52.receipt["raw_energy_relative_error"], 1e-8)
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
        preservation = torch.tensor([[-8.0, -8.0], [-16.0, 4.0]], dtype=torch.float32)
        proposal = self.proposal(kl_gradient=preservation)
        receipt = proposal.receipt
        self.assertLessEqual(receipt["safe_preservation_certificate_max"], 1e-8)
        self.assertGreaterEqual(receipt["removed_conflicting_component_norm_by_request"][0], 0.0)
        self.assertEqual(receipt["removed_conflicting_component_norm_by_request"][1], 0.0)
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
            kl_teacher_input_sha256=self.TEACHER_SHA256,
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

    def test_b100_fp32_cast_closes_the_unchanged_origin_clamp(self) -> None:
        generator = torch.Generator().manual_seed(0)
        origin = torch.randn(4096, 100, generator=generator, dtype=torch.float32).double()
        candidate = origin + 100.0 * torch.randn(
            4096, 100, generator=generator, dtype=torch.float64
        )
        clamped, _, maximum = _origin_relative_clamp(candidate, origin, 0.75)
        naive_excess = torch.clamp(
            torch.linalg.vector_norm(clamped.float().double() - origin, dim=0) - maximum,
            min=0.0,
        )
        self.assertGreater(float(torch.max(naive_excess)), 1.0e-8)
        repaired, before, after, reprojected, nextafter = (
            _cast_origin_clamped_target_fp32(clamped, origin, maximum)
        )
        self.assertEqual(repaired.dtype, torch.float32)
        self.assertTrue(torch.equal(before, naive_excess))
        self.assertLessEqual(float(torch.max(after)), 1.0e-8)
        self.assertGreater(int(torch.count_nonzero(reprojected)), 0)
        self.assertGreater(int(torch.count_nonzero(nextafter)), 0)
        self.assertLessEqual(
            float(
                torch.max(
                    torch.linalg.vector_norm(repaired.double() - origin, dim=0)
                    - maximum
                )
            ),
            1.0e-8,
        )

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

    def test_exact_active_unit_norm_and_energy_certificate(self) -> None:
        proposal = self.proposal(lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9))
        active = torch.tensor(proposal.receipt["active_gradient_mask"], dtype=torch.bool)
        norms = torch.linalg.vector_norm(proposal.kdc_direction, dim=0)
        self.assertTrue(
            torch.all(torch.abs(norms[active] - 1.0) <= 1.0e-8)
        )
        self.assertLessEqual(proposal.receipt["kdc_unit_norm_max_abs_residual"], 1.0e-8)
        self.assertLessEqual(proposal.receipt["raw_energy_relative_error"], 1e-8)
        self.assertTrue(all(proposal.receipt["final_semantic_descent_pass_by_request"]))

    def test_active_small_combined_direction_is_typed_failure(self) -> None:
        current = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        tiny = torch.tensor([[2.0e-12], [0.0]], dtype=torch.float32)
        with self.assertRaises(P1R52ActiveDirectionContractError):
            prepare_p1r52_target_proposal(
                current,
                current,
                current,
                objective((1.0,), tiny, backward=1),
                kl_result((0.0,), -tiny),
                P1R24AliasTargetLock("llama3-8b-inst", 1.0, 0.0, 1.0e9),
                P1R51ControllerState.zero(current),
                alias="llama3-8b-inst",
                step_index=0,
                shared_speed=0.4,
                kl_teacher_input_sha256=self.TEACHER_SHA256,
            )

    def test_non_descent_combined_direction_is_typed_failure(self) -> None:
        current = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        semantic_gradient = torch.tensor([[1.0e-6], [0.0]], dtype=torch.float32)
        preservation_gradient = torch.tensor([[-3.0e-6], [0.0]], dtype=torch.float32)
        with self.assertRaises(P1R52SemanticDescentContractError):
            prepare_p1r52_target_proposal(
                current,
                current,
                current,
                objective((1.0,), semantic_gradient, backward=1),
                kl_result((0.0,), preservation_gradient),
                P1R24AliasTargetLock("llama3-8b-inst", 1.0, 0.0, 1.0e9),
                P1R51ControllerState.zero(current),
                alias="llama3-8b-inst",
                step_index=0,
                shared_speed=0.4,
                kl_teacher_input_sha256=self.TEACHER_SHA256,
            )

    def test_batch_and_coefficient_scaling_are_applied_once(self) -> None:
        proposal = self.proposal(
            lock=P1R24AliasTargetLock("llama3-8b-inst", 0.5, 0.25, 1.0e9),
            kl_gradient=torch.tensor([[0.5, 0.25], [-0.25, 0.5]], dtype=torch.float32) / 2.0,
            current=self.current + torch.tensor([[0.1, -0.2], [0.2, 0.1]], dtype=torch.float32),
        )
        receipt = proposal.receipt
        expected_kl = torch.linalg.vector_norm(
            torch.tensor([[0.5, 0.25], [-0.25, 0.5]], dtype=torch.float64), dim=0
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(receipt["kl_gradient_norm_by_request"], dtype=torch.float64),
                expected_kl,
            )
        )
        self.assertTrue(all(value >= 0.0 for value in receipt["decay_gradient_norm_by_request"]))
        self.assertEqual(receipt["added_kl_model_forward_count"], 0)
        self.assertEqual(receipt["added_kl_backward_count"], 0)

    def test_batch_one_and_replicated_batch_restore_request_gradients_once(self) -> None:
        one_current = torch.tensor([[2.0], [1.0]], dtype=torch.float32)
        one_gradient = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
        one_kl = torch.tensor([[0.125], [-0.25]], dtype=torch.float32)
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.5, 0.0, 1.0e9)
        one = prepare_p1r52_target_proposal(
            one_current,
            one_current,
            one_current,
            objective((1.0,), one_gradient, backward=1),
            kl_result((0.0,), one_kl),
            lock,
            P1R51ControllerState.zero(one_current),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
            kl_teacher_input_sha256=self.TEACHER_SHA256,
        )
        two_current = one_current.repeat(1, 2)
        two = prepare_p1r52_target_proposal(
            two_current,
            two_current,
            two_current,
            objective((1.0, 1.0), one_gradient.repeat(1, 2) / 2.0, backward=1),
            kl_result((0.0, 0.0), one_kl.repeat(1, 2) / 2.0),
            lock,
            P1R51ControllerState.zero(two_current),
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
            kl_teacher_input_sha256=self.TEACHER_SHA256,
        )
        self.assertEqual(
            one.receipt["semantic_gradient_norm_by_request"][0],
            two.receipt["semantic_gradient_norm_by_request"][0],
        )
        self.assertEqual(
            one.receipt["kl_gradient_norm_by_request"][0],
            two.receipt["kl_gradient_norm_by_request"][0],
        )
        self.assertTrue(torch.equal(one.kdc_direction[:, 0], two.kdc_direction[:, 0]))

    def test_clamp_cast_and_selection_energy_are_separate(self) -> None:
        proposal = self.proposal(lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 0.01))
        endpoint = objective((0.9, 1.9), None, backward=0)
        rescue = prepare_p1r52_rescue_proposal(
            proposal, self.current, self.terminal, self.nll, endpoint, step_index=0
        )
        selected = select_p1r52_target_proposal(
            proposal,
            rescue,
            self.current,
            self.terminal,
            self.nll,
            endpoint,
            None,
            step_index=0,
        )
        receipt = selected.receipt
        self.assertIn("clamp_only_removed_energy", receipt)
        self.assertIn("fp32_cast_energy_delta", receipt)
        self.assertIn("accepted_energy", receipt)
        self.assertIn("total_unused_energy", receipt)
        self.assertLessEqual(receipt["energy_decomposition_max_abs_residual"], 1e-12)
        self.assertEqual(receipt["kl_teacher_input_sha256"], self.TEACHER_SHA256)


if __name__ == "__main__":
    unittest.main()
