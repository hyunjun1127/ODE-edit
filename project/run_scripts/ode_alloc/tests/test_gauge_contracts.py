from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_alloc.contracts import (
    MODEL_ALIASES,
    ODEAllocContractError,
    assert_common_model_policy,
    reject_global_strength_fields,
)
from project.run_scripts.ode_alloc.gauge import (
    BasisEligibilityError,
    FactorPair,
    FixedEnergyGauge,
    QDomainError,
    factor_gram_energy,
)


def _pairs() -> dict[int, FactorPair]:
    return {
        4: FactorPair(
            4,
            torch.tensor([[1.0, 0.2], [-0.5, 0.3]], dtype=torch.float32),
            torch.tensor([[0.2, -0.7], [0.5, 0.1], [0.4, 0.9]]),
        ),
        5: FactorPair(
            5,
            torch.tensor([[0.4], [0.8], [-0.2]]),
            torch.tensor([[0.7], [-0.1]]),
        ),
        6: FactorPair(
            6,
            torch.tensor([[0.3], [-0.6]]),
            torch.tensor([[0.2], [0.5], [-0.4]]),
        ),
        7: FactorPair(7, torch.zeros(2, 1), torch.ones(3, 1)),
    }


class GaugeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = _pairs()
        self.gauge = FixedEnergyGauge(
            self.pairs,
            basis_energy_epsilon=1.0e-24,
            max_abs_centered_q=1.3862943611198906,
            quantized_zero_by_layer={7: True},
        )

    def test_factor_gram_matches_dense_fixture(self) -> None:
        for pair in self.pairs.values():
            dense = pair.left.double() @ pair.right.double().transpose(0, 1)
            self.assertAlmostEqual(
                factor_gram_energy(pair), float(torch.sum(dense.square())), places=12
            )

    def test_q_zero_is_exact_native_and_energy_is_fixed(self) -> None:
        reading = self.gauge.evaluate(self.gauge.zeros())
        self.assertTrue(torch.equal(reading.ratios, torch.ones_like(reading.ratios)))
        self.assertEqual(reading.excluded_layers, (7,))
        self.assertEqual(self.gauge.dimension, 2)
        self.assertTrue(
            torch.allclose(
                reading.energy_before,
                reading.energy_after,
                rtol=2.0e-15,
                atol=2.0e-15,
            )
        )

    def test_constant_shift_and_layer_order_are_invariant(self) -> None:
        q = {4: -0.3, 5: 0.4, 6: 0.9}
        base = self.gauge.evaluate(q)
        shifted = self.gauge.evaluate({layer: value + 7.5 for layer, value in q.items()})
        permuted_gauge = FixedEnergyGauge(
            {6: self.pairs[6], 4: self.pairs[4], 7: self.pairs[7], 5: self.pairs[5]},
            basis_energy_epsilon=1.0e-24,
            max_abs_centered_q=1.3862943611198906,
            quantized_zero_by_layer={7: True},
        )
        permuted = permuted_gauge.evaluate({6: q[6], 5: q[5], 4: q[4]})
        self.assertTrue(torch.allclose(base.ratios, shifted.ratios, atol=2.0e-15, rtol=2.0e-15))
        self.assertTrue(torch.equal(base.ratios, permuted.ratios))

    def test_zero_mean_chart_retains_nontrivial_gradient(self) -> None:
        q = {
            layer: torch.zeros((), dtype=torch.float64, requires_grad=True)
            for layer in self.gauge.layers
        }
        reading = self.gauge.evaluate(q)
        reading.ratios[0].backward()
        gradients = torch.stack([q[layer].grad for layer in self.gauge.layers])
        self.assertGreater(float(torch.linalg.vector_norm(gradients)), 0.0)
        self.assertAlmostEqual(float(gradients.sum()), 0.0, places=14)

    def test_global_scalar_confound_is_detected_and_unrepresentable(self) -> None:
        reading = self.gauge.evaluate({4: -0.3, 5: 0.4, 6: 0.9})
        scaled_energy = torch.sum(reading.energies * (1.1 * reading.ratios).square())
        self.assertFalse(torch.isclose(scaled_energy, reading.energy_before))
        with self.assertRaises(ODEAllocContractError):
            reject_global_strength_fields({"global_scale": 1.1})

    def test_degenerate_basis_fails_closed(self) -> None:
        with self.assertRaises(BasisEligibilityError) as observed:
            FixedEnergyGauge(
                {4: self.pairs[4], 7: self.pairs[7]},
                basis_energy_epsilon=factor_gram_energy(self.pairs[4]) * 0.9,
                max_abs_centered_q=1.3862943611198906,
                quantized_zero_by_layer={7: True},
            )
        self.assertEqual(observed.exception.code, "INELIGIBLE_BASIS_ENERGY")

    def test_positive_near_zero_and_unattested_exact_zero_are_ineligible(self) -> None:
        near = FactorPair(
            8,
            torch.tensor([[1.0e-7]], dtype=torch.float64),
            torch.tensor([[1.0e-7]], dtype=torch.float64),
        )
        with self.assertRaises(BasisEligibilityError):
            FixedEnergyGauge(
                {4: self.pairs[4], 5: self.pairs[5], 8: near},
                basis_energy_epsilon=1.0e-24,
                max_abs_centered_q=1.3862943611198906,
            )
        with self.assertRaises(BasisEligibilityError):
            FixedEnergyGauge(
                self.pairs,
                basis_energy_epsilon=1.0e-24,
                max_abs_centered_q=1.3862943611198906,
            )

    def test_q_cap_hit_is_explicit_and_not_clamped(self) -> None:
        with self.assertRaises(QDomainError) as observed:
            self.gauge.evaluate({4: -2.0, 5: 0.0, 6: 2.0})
        self.assertEqual(observed.exception.code, "Q_CAP_HIT")

    def test_both_aliases_share_one_policy(self) -> None:
        policy = {"rho": 0.5, "fixed_k": 8}
        identity = assert_common_model_policy({alias: policy for alias in MODEL_ALIASES})
        self.assertEqual(len(identity), 64)
        with self.assertRaises(ODEAllocContractError):
            assert_common_model_policy(
                {MODEL_ALIASES[0]: policy, MODEL_ALIASES[1]: {"rho": 0.6, "fixed_k": 8}}
            )


if __name__ == "__main__":
    unittest.main()
