from __future__ import annotations

import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p3r1_finite_horizon import finite_horizon_waypoint
from project.run_scripts.ode_bf.p3r1_fixed_m_target import run_fixed_m_final_iterate
from project.run_scripts.ode_bf.p3r1_runtime import (
    case_role,
    expected_result_name,
    expected_result_parent,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


class P3R1FastTargetTest(unittest.TestCase):
    def _objective(self, value: torch.Tensor, required: bool) -> ScalableObjectiveResult:
        gradient = value.detach().clone() / value.shape[1] if required else None
        per_request = tuple(float(torch.sum(value[:, index].square())) for index in range(value.shape[1]))
        payload = {
            "value": per_request,
            "gradient": required,
        }
        return ScalableObjectiveResult(
            sum(per_request) / len(per_request),
            per_request,
            "order",
            "span",
            tuple("span" for _ in per_request),
            tuple(1 for _ in per_request),
            1,
            1 if required else 0,
            2,
            2,
            gradient,
            None,
            "plan",
            "state",
            canonical_hash(payload),
        )

    def _kl(self, value: torch.Tensor, required: bool) -> P1R24KLResult:
        gradient = torch.zeros_like(value) if required else None
        values = tuple(0.0 for _ in range(value.shape[1]))
        return P1R24KLResult(
            0.0,
            values,
            gradient,
            1,
            1 if required else 0,
            2,
            2,
            canonical_hash({"kl": required}),
        )

    def test_fixed_m_has_exact_updates_no_selector_and_resets_moments(self) -> None:
        current = torch.tensor([[1.0, 2.0], [0.5, 1.0]], dtype=torch.float32)
        identity = canonical_hash({"W": "fixed"})
        result = run_fixed_m_final_iterate(
            current_target=current,
            current_terminal=torch.zeros_like(current),
            target_origin=current,
            lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 10.0),
            alias="llama3-8b-inst",
            inner_count=5,
            outer_step_index=0,
            evaluate_target=self._objective,
            evaluate_kl=self._kl,
            fixed_state_identity=lambda: identity,
        )
        self.assertEqual(result.receipt["executed_update_count"], 5)
        self.assertEqual(result.receipt["iterate_observation_count"], 6)
        self.assertEqual(result.receipt["adam_moment_reset_count"], 1)
        self.assertEqual(result.receipt["primary_count"], 0)
        self.assertEqual(result.receipt["retry_count"], 0)
        self.assertFalse(result.inner_receipts[-1]["gradient_required"])
        self.assertEqual(result.inner_receipts[-1]["model_backward_count"], 0)
        self.assertFalse(torch.equal(result.final_target, current))

    def test_m25_is_exact_and_has_no_writer(self) -> None:
        current = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
        identity = canonical_hash({"W": "fixed"})
        result = run_fixed_m_final_iterate(
            current_target=current,
            current_terminal=torch.zeros_like(current),
            target_origin=current,
            lock=P1R24AliasTargetLock("qwen2.5-7b-inst", 0.0, 0.0, 100.0),
            alias="qwen2.5-7b-inst",
            inner_count=25,
            outer_step_index=None,
            evaluate_target=self._objective,
            evaluate_kl=self._kl,
            fixed_state_identity=lambda: identity,
        )
        self.assertEqual(result.receipt["executed_update_count"], 25)
        self.assertEqual(result.receipt["inner_writer_call_count"], 0)
        self.assertEqual(len(result.inner_receipts), 26)


class P3R1FiniteHorizonTest(unittest.TestCase):
    def test_exact_lambda_and_no_carry(self) -> None:
        current = torch.zeros((2, 2), dtype=torch.float32)
        oracle = torch.ones((2, 2), dtype=torch.float32)
        first = finite_horizon_waypoint(current, oracle, outer_step_index=0)
        last = finite_horizon_waypoint(current, oracle, outer_step_index=7)
        self.assertEqual(first.lambda_k, 1.0 / 8.0)
        self.assertTrue(torch.equal(first.waypoint, torch.full_like(oracle, 0.125)))
        self.assertEqual(last.lambda_k, 1.0)
        self.assertTrue(torch.equal(last.waypoint, oracle))
        self.assertEqual(first.receipt["debt_input_count"], 0)
        self.assertEqual(first.receipt["lag_carry_count"], 0)
        self.assertEqual(first.receipt["second_remaining_division_count"], 0)


class P3R1SourceLockTest(unittest.TestCase):
    def test_tech_r1_namespace_is_shared_by_dispatcher_and_runtime(self) -> None:
        root = Path(__file__).resolve().parents[4]
        role = case_role(1)
        parent = expected_result_parent(root)
        self.assertEqual(
            parent,
            (root / "local/odebf/results/p3r1-two-timescale-fastz-fh-c013-fp32-tech-r1").resolve(
                strict=False
            ),
        )
        self.assertEqual(
            expected_result_name("llama3-8b-inst", role),
            "s05-p3r1-two-timescale-fastz-fh-c013-fp32-tech-r1-llama3-8b-inst-case-01-v1",
        )
        dispatcher = (root / "project/run_scripts/session05_ode_bf_p3r1_fastz_fh.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("p3r1-two-timescale-fastz-fh-c013-fp32-tech-r1-${ALIAS}", dispatcher)

    def test_runtime_binds_frozen_kernels_and_forbidden_counts(self) -> None:
        root = Path(__file__).resolve().parents[4]
        text = (root / "project/run_scripts/ode_bf/p3r1_runtime.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "run_official_native_apply",
            "run_official_memit_apply",
            "solve_joint_pc_router",
            "solve_p1r43_full_strength_routing",
            "remaining_pi_beta",
            '"fallback_count": 0',
            '"retry_count": 0',
            '"bf16_materializer_call_count": 0',
            'expected_native_compute_z_call_count=0',
        ):
            self.assertIn(required, text)
        self.assertNotIn("max(alpha", text)
        self.assertNotIn("current_slope", text)


if __name__ == "__main__":
    unittest.main()
