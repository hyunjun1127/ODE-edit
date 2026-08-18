from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r52_frozen_pi_quota_writer import (
    P1R52WriterPolicy,
    _factor_for_velocity,
    _merge_factors,
    _velocity_residual,
    post_commit_identity,
    sequential_quota_decision,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import P1R23_H


class FrozenPiQuotaAlgebraTests(unittest.TestCase):
    def test_empty_future_layers_are_absent_from_virtual_prefix(self) -> None:
        factor = _factor_for_velocity
        self.assertEqual(
            _merge_factors({"a.weight": (), "b.weight": ()}, {}), {}
        )
        self.assertIsNotNone(factor)

    def test_layer4_fpiq_matches_entry_velocity(self) -> None:
        alpha = 0.8
        pi = (0.2, 0.1, 0.3, 0.15, 0.25)
        applied = (0.4, 0.5, 0.6, 0.7, 0.8)
        entry_velocity = tuple(alpha * p / a for p, a in zip(pi, applied))
        result = sequential_quota_decision(
            policy=P1R52WriterPolicy.FPIQ,
            alpha_star=alpha,
            current_nll=1.2,
            endpoint_nll=0.4,
            entry_pi=pi,
            entry_velocity=entry_velocity,
            applied_slope=applied[0],
            layer_ordinal=0,
        )
        self.assertAlmostEqual(result["velocity"], entry_velocity[0], places=15)
        self.assertAlmostEqual(result["quota"], alpha * pi[0], places=15)
        self.assertAlmostEqual(result["suffix_mass"], 1.0, places=15)

    def test_later_quota_uses_remaining_deficit_and_suffix_mass(self) -> None:
        result = sequential_quota_decision(
            policy="FPIQ",
            alpha_star=1.0,
            current_nll=0.7,
            endpoint_nll=0.2,
            entry_pi=(0.1, 0.2, 0.3, 0.2, 0.2),
            entry_velocity=(1.0, 1.0, 1.0, 1.0, 1.0),
            applied_slope=0.25,
            layer_ordinal=2,
        )
        self.assertAlmostEqual(result["alpha_remaining"], 0.5, places=15)
        self.assertAlmostEqual(result["suffix_mass"], 0.7, places=15)
        self.assertAlmostEqual(result["relative_share"], 3.0 / 7.0, places=15)
        self.assertAlmostEqual(result["quota"], 3.0 / 14.0, places=15)
        self.assertAlmostEqual(result["velocity"], 6.0 / 7.0, places=15)

    def test_sv_freezes_entry_velocity_and_nonpositive_fpiq_is_zero(self) -> None:
        common = {
            "alpha_star": 1.0,
            "current_nll": 1.1,
            "endpoint_nll": 0.1,
            "entry_pi": (0.2, 0.2, 0.2, 0.2, 0.2),
            "entry_velocity": (0.5, 0.4, 0.3, 0.2, 0.1),
            "layer_ordinal": 3,
        }
        sv = sequential_quota_decision(
            policy="SV", applied_slope=-2.0, **common
        )
        fpiq = sequential_quota_decision(
            policy="FPIQ", applied_slope=0.0, **common
        )
        self.assertEqual(sv["velocity"], 0.2)
        self.assertEqual(fpiq["velocity"], 0.0)
        self.assertTrue(fpiq["nonpositive_slope_zero_velocity"])

    def test_full_residual_and_one_h_identity(self) -> None:
        target = torch.tensor([[3.0, 5.0], [7.0, 11.0]], dtype=torch.float32)
        current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        residual, velocity_residual, identity = _velocity_residual(target, current)
        self.assertEqual(identity, 0.0)
        self.assertTrue(
            torch.equal(float(P1R23_H) * velocity_residual, residual)
        )

    def test_virtual_physical_receipt_requires_exact_hashes(self) -> None:
        class Result:
            policy = P1R52WriterPolicy.FPIQ
            expected_bf16_sha256 = {"a.weight": "0" * 64}

        value = post_commit_identity(
            Result(),
            {"effective_bf16_sha256": {"a.weight": "0" * 64}},
        )
        self.assertTrue(value["exact_hash_identity"])
        self.assertEqual(value["physical_commit_count"], 1)


if __name__ == "__main__":
    unittest.main()
