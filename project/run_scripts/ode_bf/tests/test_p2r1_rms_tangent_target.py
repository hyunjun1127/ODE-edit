from __future__ import annotations

import math
import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24_NUMERICAL_EPSILON,
)
from project.run_scripts.ode_bf.p2r1_rms_tangent_target import (
    P2R1RMSState,
    p2r1_preservation_gradient,
    p2r1_target_update,
)


class P2R1RMSTangentTargetTests(unittest.TestCase):
    def test_semantic_and_tangent_certificates_and_running_average(self) -> None:
        torch.manual_seed(48)
        origin = torch.ones(7, 3)
        target = origin.clone()
        semantic0 = torch.randn(7, 3) * 0.02
        preservation0 = torch.randn(7, 3) * 0.01
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        first = p2r1_target_update(
            target,
            origin,
            semantic0,
            preservation0,
            P2R1RMSState.zero(),
            alias=lock.alias,
            microstep_index=0,
            lock=lock,
        )
        self.assertLessEqual(first.receipt["semantic_tangent_max_abs_residual"], 1e-8)
        self.assertLessEqual(first.receipt["semantic_rate_max_abs_residual"], 1e-8)
        self.assertTrue(
            torch.allclose(
                first.state_next.running_squared_gradient,
                semantic0.to(torch.float64).square(),
                atol=0.0,
                rtol=1e-12,
            )
        )
        semantic1 = torch.randn(7, 3) * 0.03
        second = p2r1_target_update(
            first.target_next,
            origin,
            semantic1,
            preservation0,
            first.state_next,
            alias=lock.alias,
            microstep_index=1,
            lock=lock,
        )
        expected = (semantic0.to(torch.float64).square() + semantic1.to(torch.float64).square()) / 2
        self.assertTrue(
            torch.allclose(
                second.state_next.running_squared_gradient,
                expected,
                atol=0.0,
                rtol=1e-12,
            )
        )
        expected_eps = math.sqrt(torch.finfo(torch.float32).eps) * torch.sqrt(
            torch.mean(semantic0.to(torch.float64).square(), dim=0)
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    first.receipt["epsilon_by_request"], dtype=torch.float64
                ),
                expected_eps,
                rtol=1e-6,
                atol=0.0,
            )
        )

    def test_zero_semantic_gradient_has_zero_field(self) -> None:
        origin = torch.ones(5, 2)
        update = p2r1_target_update(
            origin,
            origin,
            torch.zeros_like(origin),
            torch.randn_like(origin),
            P2R1RMSState.zero(),
            alias="qwen2.5-7b-inst",
            microstep_index=0,
            lock=P1R24AliasTargetLock.for_alias("qwen2.5-7b-inst"),
        )
        self.assertEqual(torch.count_nonzero(update.field), 0)
        self.assertTrue(torch.equal(update.target_next, origin))
        self.assertEqual(update.receipt["writer_materialization_count"], 0)

    def test_decay_subgradient_is_zero_at_entry(self) -> None:
        origin = torch.randn(11, 2)
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        preservation, decay, decay_gradient = p2r1_preservation_gradient(
            origin, origin, torch.zeros_like(origin), lock
        )
        self.assertEqual(torch.count_nonzero(preservation), 0)
        self.assertEqual(torch.count_nonzero(decay), 0)
        self.assertEqual(torch.count_nonzero(decay_gradient), 0)

    def test_decay_uses_source_bound_numerical_denominator(self) -> None:
        origin = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        current = origin + torch.tensor([[0.3], [0.4]], dtype=torch.float64)
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        _, decay, decay_gradient = p2r1_preservation_gradient(
            current, origin, torch.zeros_like(origin), lock
        )
        denominator = origin.square().sum(dim=0) + P1R24_NUMERICAL_EPSILON
        displacement = current - origin
        displacement_norm = torch.linalg.vector_norm(displacement, dim=0)
        self.assertTrue(
            torch.allclose(
                decay,
                lock.decay_factor * displacement_norm / denominator,
                atol=0.0,
                rtol=1e-14,
            )
        )
        self.assertTrue(
            torch.allclose(
                decay_gradient,
                lock.decay_factor
                * displacement
                / displacement_norm.unsqueeze(0)
                / denominator.unsqueeze(0),
                atol=0.0,
                rtol=1e-14,
            )
        )

    def test_exact_24_step_state_chain(self) -> None:
        origin = torch.ones(4, 1)
        target = origin.clone()
        state = P2R1RMSState.zero()
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        for n in range(24):
            update = p2r1_target_update(
                target,
                origin,
                torch.full_like(origin, 0.001 + n * 1e-6),
                torch.zeros_like(origin),
                state,
                alias=lock.alias,
                microstep_index=n,
                lock=lock,
            )
            target, state = update.target_next, update.state_next
        self.assertEqual(state.completed_microsteps, 24)


if __name__ == "__main__":
    unittest.main()
