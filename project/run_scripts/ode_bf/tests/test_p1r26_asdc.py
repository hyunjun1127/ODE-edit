from __future__ import annotations

import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock
from project.run_scripts.ode_bf.p1r26_asdc import (
    P1R26_EPSILON_Z,
    p1r26_target_step,
    project_absolute_semantic_deficit,
)
from project.run_scripts.ode_bf.scalable_batched_field import ScalableRobustSharedMetric
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts import session05_ode_bf_p1r26_asdc_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1r26_asdc_panel import (
    LOCK_FILE,
    load_and_validate_p1r26_lock,
)


class _Objective:
    def __init__(self, gradient: torch.Tensor, values: tuple[float, ...]) -> None:
        self.target_gradient = gradient
        self.per_request_values = values
        self.loss = sum(values) / len(values)


class _KL:
    def __init__(self, gradient: torch.Tensor, values: tuple[float, ...]) -> None:
        self.gradient = gradient
        self.per_request_values = values
        self.loss = sum(values) / len(values)


class P1R26ASDCTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[4]

    def test_requestwise_deadline_and_nominal_no_weakening(self) -> None:
        nominal = torch.tensor([[-0.1, -0.4], [0.0, 0.0]], dtype=torch.float64)
        gradient = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        result = project_absolute_semantic_deficit(
            nominal,
            torch.zeros_like(nominal),
            torch.zeros_like(nominal),
            gradient,
            (0.85, 0.04),
            (10.0, 10.0),
            remaining_steps=4,
            observation_mask=(False, True),
        )
        self.assertAlmostEqual(result.required_progress[0], 0.2, places=12)
        self.assertAlmostEqual(result.required_progress[1], 0.4, places=12)
        self.assertGreaterEqual(result.achieved_progress[0] + 1.0e-8, 0.2)
        self.assertGreaterEqual(result.achieved_progress[1] + 1.0e-8, 0.4)
        self.assertEqual(result.receipt["combined_freeze_decision_influence_count"], 0)
        self.assertEqual(result.receipt["additional_model_forward_count"], 0)

    def test_clamp_limited_is_typed_not_silently_relaxed(self) -> None:
        result = project_absolute_semantic_deficit(
            torch.zeros(2, 1, dtype=torch.float64),
            torch.zeros(2, 1, dtype=torch.float64),
            torch.zeros(2, 1, dtype=torch.float64),
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            (1.05,),
            (0.1,),
            remaining_steps=1,
            observation_mask=(False,),
        )
        self.assertEqual(result.clamp_limited, (True,))
        self.assertEqual(result.receipt["clamp_limited_count"], 1)
        self.assertEqual(result.receipt["silent_understrength_count"], 0)

    def test_unclamped_numerical_understrength_fails_closed(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "silently under-strength"):
            project_absolute_semantic_deficit(
                torch.zeros(1, 1, dtype=torch.float64),
                torch.zeros(1, 1, dtype=torch.float64),
                torch.zeros(1, 1, dtype=torch.float64),
                torch.tensor([[1.0e-5]], dtype=torch.float64),
                (1.05,),
                (1.0e9,),
                remaining_steps=1,
                observation_mask=(False,),
            )

    def test_fullspeed_observation_does_not_freeze_nominal_field(self) -> None:
        z0 = torch.ones(4, 1)
        metric = ScalableRobustSharedMetric.from_z0(z0, "a" * 64)
        gradient = -torch.ones(4, 1)
        result = p1r26_target_step(
            z0,
            z0,
            z0,
            _Objective(gradient, (0.01,)),
            _KL(torch.zeros_like(gradient), (0.0,)),
            metric,
            P1R24AliasTargetLock.for_alias("llama3-8b-inst"),
            step_index=0,
            frozen_mask=(False,),
        )
        self.assertEqual(result.frozen_mask, (True,))
        self.assertGreater(float(torch.linalg.vector_norm(result.target_displacement)), 0.0)
        self.assertEqual(result.receipt["combined_freeze_decision_influence_count"], 0)
        self.assertEqual(result.receipt["physical_h_application_count"], 1)
        self.assertEqual(result.receipt["second_remaining_division_count"], 0)
        torch.testing.assert_close(
            0.125 * result.write_velocity,
            result.required_displacement,
            rtol=0.0,
            atol=0.0,
        )

    def test_remaining_eight_and_one_use_selected_displacement(self) -> None:
        z0 = torch.ones(3, 1)
        metric = ScalableRobustSharedMetric.from_z0(z0, "b" * 64)
        nll = _Objective(-torch.ones(3, 1), (0.8,))
        kl = _KL(torch.zeros(3, 1), (0.0,))
        lock = P1R24AliasTargetLock.for_alias("qwen2.5-7b-inst")
        step0 = p1r26_target_step(z0, z0, z0, nll, kl, metric, lock, step_index=0, frozen_mask=(False,))
        self.assertEqual(step0.receipt["remaining_steps"], 8)
        terminal = step0.target_next - 0.25
        step7 = p1r26_target_step(step0.target_next, terminal, z0, nll, kl, metric, lock, step_index=7, frozen_mask=step0.frozen_mask)
        self.assertEqual(step7.receipt["remaining_steps"], 1)
        expected = step7.target_displacement + (step0.target_next - terminal)
        torch.testing.assert_close(step7.required_displacement, expected)

    def test_controller_threshold_is_alias_independent_and_history_free(self) -> None:
        self.assertEqual(P1R26_EPSILON_Z, 0.05)
        self.assertNotEqual(
            P1R24AliasTargetLock.for_alias("llama3-8b-inst").decay_factor,
            P1R24AliasTargetLock.for_alias("qwen2.5-7b-inst").decay_factor,
        )
        # Alias-native target geometry may differ; ASDC itself has no alias input.
        self.assertNotIn("alias", project_absolute_semantic_deficit.__annotations__)

    def test_a1_matrix_dry_plan_and_cap4_are_exact(self) -> None:
        for phase, expected_trajectories in (("smoke", 8), ("production", 80)):
            plan = dry.build_plan("child", phase, repository_root=self.ROOT)
            self.assertEqual(plan["job_count"], 4)
            self.assertEqual(plan["array_max_concurrent_gpu"], 4)
            self.assertEqual(plan["project_gpu_cap_server1"], 4)
            self.assertEqual(plan["trajectory_count"], expected_trajectories)
            self.assertEqual(
                [(item["alias"], item["allocation"]) for item in plan["jobs"]],
                [
                    ("llama3-8b-inst", "RS"),
                    ("llama3-8b-inst", "BG"),
                    ("qwen2.5-7b-inst", "RS"),
                    ("qwen2.5-7b-inst", "BG"),
                ],
            )

    def test_lock_and_protected_p1r24_controller_source(self) -> None:
        lock, _ = load_and_validate_p1r26_lock(
            self.ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(lock["epsilon_z"], 0.05)
        self.assertEqual(
            sha256_file(self.ROOT / "project/run_scripts/ode_bf/p1r24_atomic_strength.py"),
            "95ef3aff1e0b2d82df08453492a01be4d69d7e1c8795edd72c7d098979f54b0f",
        )

    def test_runtime_closure_excludes_pacing_history_and_callbacks(self) -> None:
        asdc_source = (self.ROOT / "project/run_scripts/ode_bf/p1r26_asdc.py").read_text()
        independent = (
            self.ROOT / "project/run_scripts/ode_bf/p1r26_independent_b10x10_runtime.py"
        ).read_text()
        submitter = (
            self.ROOT / "project/run_scripts/session05_ode_bf_submit_p1r26_asdc.py"
        ).read_text()
        self.assertNotIn("S64", asdc_source)
        self.assertIn('"p1r25_scalar_pacing_access_count": 0', asdc_source)
        self.assertIn('"history_mode": _history_off_receipt()', independent)
        self.assertIn('"callback_job_count": 0', submitter)
        self.assertNotIn("--dependency", submitter)


if __name__ == "__main__":
    unittest.main()
