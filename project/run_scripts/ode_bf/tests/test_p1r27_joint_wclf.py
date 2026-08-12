from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from project.run_scripts.session05_ode_bf_p1r27_joint_wclf_dry_plan import build_plan
from project.run_scripts.ode_bf.contracts import ODEBFStateError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock
from project.run_scripts.ode_bf.p1r27_joint_wclf import (
    AcceptedRealizationLedger,
    P1R27_H,
    P1R27RoutingStatus,
    WriterCLFStatus,
    apply_z_writer_backpressure,
    build_writer_clf_entry,
    build_zero_seeking_z_entry,
    build_zero_seeking_z_step,
    semantic_writer_action,
    solve_semantic_first_routing,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.scalable_batched_field import (
    ScalableBatchGlobalMetric,
    ScalableRobustSharedMetric,
)


def _barrier(label: str, diagonal: tuple[float, ...]) -> QuadraticBarrier:
    size = len(diagonal)
    return QuadraticBarrier(
        label,
        0.0,
        np.zeros(size, dtype=np.float64),
        np.diag(np.asarray(diagonal, dtype=np.float64)),
        100.0,
        "layer-local-diagonal",
    )


def _problem(
    slopes: tuple[float, ...] = (4.0, 3.0, 2.0, 1.0, 0.5),
    *,
    trust: np.ndarray | None = None,
    pretrained_diagonal: tuple[float, ...] | None = None,
) -> RoutingProblem:
    size = len(slopes)
    return RoutingProblem(
        np.asarray(slopes, dtype=np.float64),
        np.diag(np.arange(1.0, size + 1.0)),
        np.eye(size, dtype=np.float64) if trust is None else trust,
        10.0,
        np.ones(size, dtype=np.float64),
        1.0,
        0.1,
        _barrier("historical", (0.0,) * size),
        _barrier(
            "pretrained",
            (0.0,) * size
            if pretrained_diagonal is None
            else pretrained_diagonal,
        ),
    )


class P1R27JointWCLFTest(unittest.TestCase):
    def test_zero_seeking_rs_and_bg_freeze_entry_denominator(self) -> None:
        z0 = torch.tensor([[3.0, 4.0], [4.0, 3.0]], dtype=torch.float32)
        g0 = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
        order = "a" * 64
        for allocation, metric in (
            ("RS", ScalableRobustSharedMetric.from_z0(z0, order)),
            ("BG", ScalableBatchGlobalMetric.from_z0(z0, order)),
        ):
            entry = build_zero_seeking_z_entry(metric, g0, allocation=allocation)
            nll = SimpleNamespace(
                target_gradient=g0 * 1.0e-6,
                loss=1.0e-6,
                per_request_values=(1.0e-6, 1.0e-6),
            )
            kl = SimpleNamespace(gradient=torch.zeros_like(g0), loss=0.0)
            step = build_zero_seeking_z_step(
                z0,
                z0,
                nll,
                kl,
                entry,
                P1R24AliasTargetLock("llama3-8b-inst", 0.0625, 0.5, 100.0),
                step_index=1,
            )
            self.assertLess(float(torch.linalg.vector_norm(step.nominal_displacement)), 1.0e-4)
            self.assertEqual(step.receipt["absolute_0p05_decision_influence_count"], 0)
            self.assertEqual(step.receipt["remaining_steps_decision_influence_count"], 0)

    def test_regularizer_projection_never_increases_target_nll(self) -> None:
        z0 = torch.ones((3, 2), dtype=torch.float32)
        g = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        metric = ScalableRobustSharedMetric.from_z0(z0, "b" * 64)
        entry = build_zero_seeking_z_entry(metric, g, allocation="RS")
        step = build_zero_seeking_z_step(
            z0,
            z0,
            SimpleNamespace(target_gradient=g, loss=2.0, per_request_values=(2.0, 2.0)),
            SimpleNamespace(gradient=-g, loss=1.0),
            entry,
            P1R24AliasTargetLock("llama3-8b-inst", 0.0625, 0.5, 100.0),
            step_index=0,
        )
        self.assertTrue(all(item >= 0.0 for item in step.receipt["regularizer_increasing_component_removed"]))

    def test_applied_coordinate_energy_identity_eta_kappa_and_freeze(self) -> None:
        problem = _problem()
        entry, action = build_writer_clf_entry(problem, v_w0=8.0)
        c = np.asarray(entry.c_ref0)
        g = problem.trust_metric / (P1R27_H**2)
        self.assertAlmostEqual(float(c @ g @ c), float((c / P1R27_H) @ problem.trust_metric @ (c / P1R27_H)))
        self.assertEqual(entry.status, WriterCLFStatus.ACTIVE)
        self.assertGreater(entry.eta_w, 0.0)
        self.assertAlmostEqual(entry.kappa_w, entry.r_w0 / (P1R27_H * 8.0))
        smaller = _problem(tuple(item * 1.0e-6 for item in problem.signed_progress))
        later = semantic_writer_action(smaller, entry)
        self.assertLess(later.r_w, action.r_w * 1.0e-5)
        self.assertEqual(entry.beta_h, -np.expm1(-entry.kappa_w * P1R27_H))

    def test_active_mask_refreshes_without_per_layer_cap(self) -> None:
        entry, _ = build_writer_clf_entry(_problem(), v_w0=5.0)
        changed = semantic_writer_action(_problem((4.0, -3.0, 2.0, -1.0, 0.5)), entry)
        self.assertEqual(changed.active_mask, (True, False, True, False, True))
        self.assertEqual(changed.coefficients[1], 0.0)
        self.assertEqual(changed.coefficients[3], 0.0)

    def test_entry_zero_is_scientific_stall(self) -> None:
        entry, action = build_writer_clf_entry(
            _problem((-1.0, 0.0, -2.0, -3.0, -4.0)), v_w0=9.0
        )
        self.assertEqual(entry.status, WriterCLFStatus.SEMANTIC_STALL)
        self.assertEqual(entry.eta_w, 0.0)
        self.assertEqual(entry.kappa_w, 0.0)
        self.assertEqual(action.r_w, 0.0)

    def test_invalid_metric_fails_technical(self) -> None:
        singular = np.diag([1.0, 1.0, 0.0, 1.0, 1.0])
        with self.assertRaisesRegex(ODEBFStateError, "NUMERICAL_ACTION_METRIC_INVALID"):
            build_writer_clf_entry(_problem(trust=singular), v_w0=2.0)

    def test_soft_preserves_neutral_predicted_strength(self) -> None:
        problem = _problem()
        entry, semantic = build_writer_clf_entry(problem, v_w0=10.0)
        neutral = solve_semantic_first_routing(
            problem,
            entry,
            arm=FixedE8Arm.NEUTRAL,
            requested_progress=semantic.r_w,
            semantic_reference=semantic.coefficients,
        )
        soft = solve_semantic_first_routing(
            problem,
            entry,
            arm=FixedE8Arm.SOFT,
            requested_progress=semantic.r_w,
            semantic_reference=semantic.coefficients,
        )
        self.assertLessEqual(abs(neutral.predicted_progress - soft.predicted_progress), 1.0e-8)
        self.assertEqual(soft.hard_p_h_budget_influence_count if hasattr(soft, "hard_p_h_budget_influence_count") else 0, 0)
        self.assertLessEqual(soft.selected_energy, entry.energy_limit + 1.0e-12)

    def test_soft_no_dof_equals_neutral(self) -> None:
        problem = _problem((1.0, -1.0, -1.0, -1.0, -1.0))
        entry, semantic = build_writer_clf_entry(problem, v_w0=3.0)
        soft = solve_semantic_first_routing(
            problem,
            entry,
            arm=FixedE8Arm.SOFT,
            requested_progress=semantic.r_w,
            semantic_reference=semantic.coefficients,
        )
        self.assertEqual(soft.status, P1R27RoutingStatus.SOFT_NO_ROUTING_DOF)
        self.assertEqual(soft.soft_coefficients, soft.neutral_coefficients)

    def test_backpressure_never_accelerates_and_p_h_absent(self) -> None:
        d0 = torch.tensor([[2.0], [0.0]])
        gradient = torch.tensor([[-1.0], [0.0]])
        tracking = apply_z_writer_backpressure(
            d0, gradient, torch.ones_like(d0), r_w=0.5, beta_h=0.2
        )
        self.assertLessEqual(tracking.gamma_z, 1.0)
        self.assertEqual(tracking.raw_free_payload()["p_h_gamma_influence_count"], 0)
        self.assertEqual(tracking.raw_free_payload()["remaining_steps_decision_influence_count"], 0)

    def test_realization_ledger_is_accepted_state_only(self) -> None:
        ledger = AcceptedRealizationLedger()
        value = ledger.complete(
            source_v_w=3.0,
            next_v_w=2.5,
            requested_progress=0.4,
            predicted_progress=0.4,
            transition_index=1,
        )
        self.assertAlmostEqual(value["actual_progress"], 0.5)
        self.assertEqual(value["candidate_trial_count"], 0)
        self.assertEqual(ledger.raw_free_payload()["entry_count"], 1)

    def test_historical_and_functional_decision_paths_are_absent(self) -> None:
        source = Path(
            __file__.replace("tests/test_p1r27_joint_wclf.py", "p1r27_joint_wclf.py")
        ).read_text(encoding="utf-8")
        self.assertNotIn("P1R26_FREEZE_THRESHOLD", source)
        self.assertNotIn("deficit / remaining", source)
        self.assertNotIn("functional_trial", source)
        self.assertNotIn("history_solve", source)
        self.assertIn('"remaining_steps_decision_influence_count": 0', source)

    def test_full_b1_and_b10_matrix_is_cap4_and_callback_free(self) -> None:
        for phase, request_count in (("smoke", 1), ("production", 10)):
            plan = build_plan("c" * 40, phase)
            self.assertEqual(plan["job_count"], 4)
            self.assertEqual(plan["trajectory_count"], 8)
            self.assertEqual(plan["project_gpu_cap"], 4)
            self.assertEqual(plan["array_max_concurrent_gpu"], 4)
            self.assertEqual(plan["callback_job_count"], 0)
            self.assertEqual(
                {(item["alias"], item["role"].split("_")[2]) for item in plan["jobs"]},
                {
                    ("llama3-8b-inst", "RS"),
                    ("llama3-8b-inst", "BG"),
                    ("qwen2.5-7b-inst", "RS"),
                    ("qwen2.5-7b-inst", "BG"),
                },
            )
            self.assertTrue(
                all(item["request_count"] == request_count for item in plan["jobs"])
            )


if __name__ == "__main__":
    unittest.main()
