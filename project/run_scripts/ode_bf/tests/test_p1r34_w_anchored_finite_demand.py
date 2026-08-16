from __future__ import annotations

from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24TargetStep
from project.run_scripts.ode_bf.p1r34_w_anchored_finite_demand import (
    P1R34DemandStatus,
    build_p1r34_finite_demand,
)
from project.run_scripts.ode_bf.p1r34_w_anchored_finite_demand_panel import (
    P1R34_ROLES,
    expected_p1r34_result_name,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def _target_step(old: float = 0.25) -> P1R24TargetStep:
    shape = (3, 1)
    return P1R24TargetStep(
        torch.ones(shape),
        torch.full(shape, 0.1),
        torch.full(shape, 0.2),
        torch.full(shape, 1.6),
        torch.full(shape, -0.5),
        torch.full(shape, -0.5),
        old,
        max(old, 0.0),
        0.1,
        (False,),
        {"identity_sha256": "a" * 64},
    )


def _endpoint(loss: float) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        loss,
        (loss,),
        "a" * 64,
        "b" * 64,
        ("c" * 64,),
        (1,),
        1,
        0,
        6,
        6,
        None,
        None,
        "d" * 64,
        "e" * 64,
        "f" * 64,
    )


class P1R34FiniteDemandTests(unittest.TestCase):
    def test_positive_finite_decrease_is_not_a_rate(self) -> None:
        result = build_p1r34_finite_demand(
            step_index=0,
            l_base=1.0,
            endpoint=_endpoint(0.6),
            target_step=_target_step(),
            current_target=torch.zeros((3, 1)),
            current_terminal=torch.zeros((3, 1)),
        )
        self.assertEqual(result.status, P1R34DemandStatus.FINITE_SEMANTIC_WRITE)
        self.assertAlmostEqual(result.rho_write, 0.4)
        self.assertEqual(result.receipt["finite_demand_backward_count"], 0)
        self.assertEqual(result.receipt["physical_h_application_count"], 1)
        self.assertEqual(result.receipt["second_remaining_division_count"], 0)

    def test_numerical_negative_clamps_but_material_negative_is_typed(self) -> None:
        common = dict(
            step_index=1,
            target_step=_target_step(),
            current_target=torch.zeros((3, 1)),
            current_terminal=torch.zeros((3, 1)),
        )
        near = build_p1r34_finite_demand(
            l_base=1.0, endpoint=_endpoint(1.0 + 5e-9), **common
        )
        self.assertEqual(near.status, P1R34DemandStatus.NUMERICAL_ZERO_WRITE)
        self.assertEqual(near.rho_write, 0.0)
        material = build_p1r34_finite_demand(
            l_base=1.0, endpoint=_endpoint(1.0 + 1e-5), **common
        )
        self.assertEqual(
            material.status, P1R34DemandStatus.NON_SEMANTIC_TARGET_MOVE
        )
        self.assertIsNone(material.rho_write)

    def test_old_zero_denominator_is_typed(self) -> None:
        result = build_p1r34_finite_demand(
            step_index=2,
            l_base=1.0,
            endpoint=_endpoint(0.75),
            target_step=_target_step(0.0),
            current_target=torch.zeros((3, 1)),
            current_terminal=torch.zeros((3, 1)),
        )
        self.assertIsNone(result.receipt["finite_over_old_ratio"])
        self.assertEqual(
            result.receipt["finite_over_old_ratio_status"],
            "OLD_DEMAND_ZERO_DENOMINATOR",
        )

    def test_matrix_is_two_models_times_paired_smoke_and_b10(self) -> None:
        self.assertEqual(P1R34_ROLES, ("P1R34_B1_RS_PAIR", "P1R34_B10_RS_PAIR"))
        names = {
            expected_p1r34_result_name(alias, role)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for role in P1R34_ROLES
        }
        self.assertEqual(len(names), 4)

    def test_scientific_helper_excludes_r32_r33_and_velocity_demand(self) -> None:
        root = Path(__file__).parents[1]
        source = (root / "p1r34_w_anchored_finite_demand.py").read_text()
        self.assertNotIn("p1r32", source.casefold())
        self.assertNotIn("p1r33", source.casefold())
        self.assertIn("l_base - endpoint.loss", source)
        self.assertIn("target_step.required_displacement", source)
        self.assertNotIn("target_step.write_velocity", source)


if __name__ == "__main__":
    unittest.main()
