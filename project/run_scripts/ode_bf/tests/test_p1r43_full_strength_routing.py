from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Arm,
    QuadraticBarrier,
    RoutingProblem,
)
from project.run_scripts.ode_bf.p1r43_full_strength_routing import (
    solve_p1r43_full_strength_routing,
)


def problem() -> RoutingProblem:
    zero = np.zeros(2, dtype=np.float64)
    return RoutingProblem(
        np.asarray([2.0, 1.0]),
        np.eye(2),
        np.eye(2),
        10.0,
        np.asarray([10.0, 10.0]),
        0.1,
        0.01,
        QuadraticBarrier(
            "historical", 0.0, zero, np.diag(zero), 10.0, "layer-local-diagonal"
        ),
        QuadraticBarrier(
            "pretrained", 0.0, zero, np.eye(2), 10.0, "layer-local-diagonal"
        ),
    )


class P1R43FullStrengthRoutingTests(unittest.TestCase):
    def test_neutral_and_soft_apply_identical_requested_strength(self) -> None:
        neutral = solve_p1r43_full_strength_routing(
            problem(), arm=FixedE8Arm.NEUTRAL, alpha_req=0.5
        )
        soft = solve_p1r43_full_strength_routing(
            problem(), arm=FixedE8Arm.SOFT, alpha_req=0.5
        )
        self.assertAlmostEqual(neutral.alpha_apply, 0.5)
        self.assertAlmostEqual(soft.alpha_apply, 0.5)
        self.assertEqual(neutral.raw_free_payload()["p_capacity_strength_attenuation_count"], 0)
        self.assertEqual(soft.raw_free_payload()["writer_demand_name"], "alpha_req")

    def test_soft_certificate_failure_falls_back_to_neutral(self) -> None:
        from project.run_scripts.ode_bf import p1r43_full_strength_routing as module

        original = module.solve_p1r24_matched_routing

        def side_effect(value, *, arm, rho_write):
            if FixedE8Arm(arm) is FixedE8Arm.SOFT:
                raise ODEBFContractError("P1R24 Structural-P certificate failed: fixture")
            return original(value, arm=arm, rho_write=rho_write)

        with patch.object(module, "solve_p1r24_matched_routing", side_effect=side_effect):
            result = solve_p1r43_full_strength_routing(
                problem(), arm=FixedE8Arm.SOFT, alpha_req=0.5
            )
        self.assertTrue(result.fallback_to_neutral)
        self.assertAlmostEqual(result.alpha_apply, 0.5)


if __name__ == "__main__":
    unittest.main()
