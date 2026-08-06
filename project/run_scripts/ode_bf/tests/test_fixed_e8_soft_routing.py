from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf import cold_start_target
from project.run_scripts.ode_bf import fixed_e8_runtime
from project.run_scripts.ode_bf import p1_backend
from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FIXED_E8_GRID_COUNT,
    FIXED_E8_H,
    FIXED_E8_KAPPA,
    FIXED_E8_LAYER_ORDER,
    FixedE8Arm,
    FixedE8Clock,
    FixedE8OperationCeiling,
    FixedE8SoftInventory,
    FixedE8StepMode,
    FunctionalBasisMetric,
    direct_low_rank_delta,
    expanded_quadratic_delta,
    expanded_terms_from_low_rank,
    fixed_e8_semantic_receipt,
    solve_fixed_e8_routing,
    structural_contribution_vector,
    structural_soft_score,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_fixed_e8_soft_panel import (
    FIXED_E8_INSTRUCTION_ID,
    FIXED_E8_PARENT_HEAD,
    FIXED_E8_RESULT_TOKEN,
    expected_fixed_e8_result_name,
    fixed_e8_schedule,
    forecast_fixed_e8_panel,
    load_and_validate_fixed_e8_lock,
    load_fixed_e8_case_provenance,
    validate_fixed_e8_lock,
    validate_fixed_e8_runtime_gpu_capacity,
    verify_fixed_e8_case_provenance,
)
from project.run_scripts.ode_bf.p1_cold_structp_softp_noveto_panel import (
    verify_cold_case_seal,
)
from project.run_scripts.ode_bf.p1_diagnostics import _validate_raw_free
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal
from project.run_scripts import (
    session05_ode_bf_fixed_e8_structfunc_soft_dry_plan as fixed_e8_dry,
)
from project.run_scripts import (
    session05_ode_bf_submit_fixed_e8_structfunc_soft as fixed_e8_submit,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.tests.test_cold_start_target import (
    _cold_semantic_field,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _problem(
    *,
    progress: tuple[float, ...] = (0.40, 0.32, 0.24, 0.16, 0.08),
    p_linear: tuple[float, ...] = (0.08, 0.01, 0.01, 0.01, 0.01),
    h_linear: tuple[float, ...] = (0.01, 0.01, 0.02, 0.01, 0.01),
    capacity: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.4),
    budget: float = 0.5,
) -> RoutingProblem:
    dimension = len(FIXED_E8_LAYER_ORDER)
    return RoutingProblem(
        np.asarray(progress, dtype=np.float64),
        np.diag(np.asarray(capacity, dtype=np.float64)),
        np.diag(np.full(dimension, 0.01, dtype=np.float64)),
        0.25,
        np.ones(dimension, dtype=np.float64),
        0.01,
        1.0e-8,
        QuadraticBarrier(
            "historical",
            0.0,
            np.asarray(h_linear, dtype=np.float64),
            np.diag(np.full(dimension, 0.02, dtype=np.float64)),
            budget,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            0.0,
            np.asarray(p_linear, dtype=np.float64),
            np.diag(np.full(dimension, 0.025, dtype=np.float64)),
            budget,
            "layer-local-diagonal",
        ),
    )


def _inventory(*, history_item_count: int = 0) -> FixedE8SoftInventory:
    h_active = history_item_count > 0
    reason = None if h_active else "INACTIVE_EMPTY_HISTORY"
    return FixedE8SoftInventory(
        FunctionalBasisMetric(
            "functional_p",
            0.02,
            (0.10, 0.025, 0.022, 0.021, 0.020),
            True,
        ),
        FunctionalBasisMetric(
            "functional_h_mean",
            0.0,
            (0.01, 0.01, 0.02, 0.01, 0.01),
            h_active,
            reason,
        ),
        FunctionalBasisMetric(
            "functional_h_smoothmax",
            0.0,
            (0.02, 0.01, 0.01, 0.01, 0.01),
            h_active,
            reason,
        ),
        history_item_count,
        "a" * 64,
        "b" * 64,
    )


class _ForbiddenObservation(dict[str, object]):
    def __iter__(self):  # pragma: no cover - any access is a failure
        raise AssertionError("scientific observation entered fixed-grid control")

    def __len__(self):
        raise AssertionError("scientific observation entered fixed-grid control")

    def __getitem__(self, key: str) -> object:
        raise AssertionError("scientific observation entered fixed-grid control")


class FixedE8SoftRoutingTests(unittest.TestCase):
    def test_zero_capacity_is_fixed_e8_explicit_and_legacy_fail_closed(self) -> None:
        signature = inspect.signature(p1_backend.build_p1_dynamic_field)
        self.assertFalse(signature.parameters["allow_zero_capacity"].default)

        self.assertEqual(
            p1_backend._validate_dynamic_factor_capacity(
                0.0, allow_zero_capacity=True
            ),
            0.0,
        )
        self.assertEqual(
            p1_backend._validate_dynamic_factor_capacity(
                1.0, allow_zero_capacity=False
            ),
            1.0,
        )
        for value, policy in (
            (0.0, False),
            (-1.0e-12, True),
            (float("nan"), True),
            (float("inf"), True),
        ):
            with self.subTest(value=value, policy=policy):
                with self.assertRaises(ODEBFContractError):
                    p1_backend._validate_dynamic_factor_capacity(
                        value, allow_zero_capacity=policy
                    )
        with self.assertRaises(ODEBFContractError):
            p1_backend._validate_dynamic_factor_capacity(
                0.0, allow_zero_capacity=1  # type: ignore[arg-type]
            )

        backend_source = inspect.getsource(p1_backend.build_p1_dynamic_field)
        self.assertIn("_validate_dynamic_factor_capacity", backend_source)

        fixed_source = inspect.getsource(fixed_e8_runtime._build_fixed_field_with_metric)
        tree = ast.parse(fixed_source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_p1_dynamic_field"
        ]
        self.assertEqual(len(calls), 1)
        policy = next(
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "allow_zero_capacity"
        )
        self.assertIsInstance(policy, ast.Constant)
        self.assertIs(policy.value, True)

        other_calls: list[ast.Call] = []
        for name, value in vars(p1_backend).items():
            if not inspect.isfunction(value) or name == "build_p1_dynamic_field":
                continue
            try:
                source = inspect.getsource(value)
            except (OSError, TypeError):
                continue
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "build_p1_dynamic_field"
                ):
                    other_calls.append(node)
        self.assertTrue(
            all(
                not any(
                    keyword.arg == "allow_zero_capacity"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in call.keywords
                )
                for call in other_calls
            )
        )

    def test_clock_is_exact_eight_without_scientific_decision_path(self) -> None:
        clock = FixedE8Clock()
        receipts = []
        for index in range(FIXED_E8_GRID_COUNT):
            point = clock.begin_field()
            self.assertEqual(point.step_index, index)
            self.assertEqual(point.h, Fraction(1, 8))
            receipts.append(
                clock.advance(
                    point,
                    scientific_observation=_ForbiddenObservation(),
                )
            )
        terminal = clock.terminal_receipt()
        self.assertEqual(clock.tau, Fraction(1, 1))
        self.assertEqual(terminal["grid_count"], 8)
        self.assertEqual(terminal["field_count"], 8)
        self.assertEqual(terminal["scientific_retry_count"], 0)
        self.assertTrue(all(item["adaptive_clock_access_count"] == 0 for item in receipts))
        with self.assertRaises(ODEBFStateError):
            clock.begin_field()

    def test_new_clock_has_no_retry_or_adaptive_control_surface(self) -> None:
        source = inspect.getsource(FixedE8Clock)
        tree = ast.parse(source)
        methods = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertFalse({"reject", "retry", "contract", "backtrack"} & methods)
        decision_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("rho", decision_names)
        self.assertNotIn("first_hit", decision_names)

    def test_no_positive_direction_selects_zero_write_recovery(self) -> None:
        result = solve_fixed_e8_routing(
            _problem(progress=(-0.4, 0.0, -0.2, -0.1, -0.01)),
            _inventory(),
            arm=FixedE8Arm.SOFT,
        )
        self.assertEqual(result.mode, FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY)
        self.assertEqual(result.velocity, (0.0,) * 5)
        self.assertEqual(result.applied_coefficient, (0.0,) * 5)
        self.assertEqual(result.p_max, 0.0)
        payload = result.raw_free_payload()
        self.assertFalse(payload["matched_solver_schedule"])
        self.assertEqual(payload["actual_logical_qp_count"], 0)

    def test_factual_failed_h_p_observation_does_not_block_grid(self) -> None:
        observed = {
            "historical": {"passed": True},
            "pretrained": {"passed": False},
            "trust": {"passed": True},
            "functional_h": {"passed": True},
            "functional_p": {"passed": False},
        }
        verdict, receipt = fixed_e8_runtime._fixed_e8_factual_online_feasibility(
            observed, history_item_count=0
        )
        self.assertFalse(verdict.structural_p)
        self.assertFalse(verdict.functional_p)
        self.assertFalse(verdict.all_pass)
        self.assertFalse(receipt["all_observed_components_pass"])
        self.assertEqual(
            receipt["components"]["structural_h"]["status"],
            "INACTIVE_EMPTY_HISTORY",
        )
        self.assertEqual(receipt["clock_decision_influence_count"], 0)

        ledger = ComputeLedger()
        clock = FixedE8Clock()
        point = clock.begin_field()
        transition = fixed_e8_runtime._fixed_e8_advance_grid_transition(
            ledger,
            clock,
            point,
            scientific_observation={
                "structural_p": False,
                "functional_p": False,
            },
        )
        self.assertEqual(clock.tau, Fraction(1, 8))
        self.assertEqual(ledger.counters["trial"], 1)
        self.assertEqual(transition["scientific_rejection_count"], 0)

    def test_actual_normal_and_zero_write_solver_accounting(self) -> None:
        normal = solve_fixed_e8_routing(
            _problem(), _inventory(), arm=FixedE8Arm.SOFT
        )
        actual = fixed_e8_runtime._fixed_e8_solver_accounting(normal)
        self.assertEqual(actual["actual_logical_qp_certificate_count"], 4)
        self.assertEqual(actual["actual_optimizer_backend_invocation_count"], 5)
        self.assertEqual(actual["actual_numerical_backend_continuation_count"], 1)
        self.assertTrue(actual["solver_schedule_matches_static_maximum"])
        stage2 = normal.certificates[-1].raw_free_payload()
        self.assertEqual(stage2["numerical_backend_continuation_count"], 1)
        self.assertEqual(
            stage2["numerical_backend_continuation_role"],
            "FIXED_NUMERICAL_BACKEND_CONTINUATION_SAME_QP",
        )
        self.assertEqual(stage2["scientific_retry_count"], 0)

        zero = solve_fixed_e8_routing(
            _problem(progress=(-0.4, 0.0, -0.2, -0.1, -0.01)),
            _inventory(),
            arm=FixedE8Arm.NEUTRAL,
        )
        zero_actual = fixed_e8_runtime._fixed_e8_solver_accounting(zero)
        self.assertEqual(zero_actual["actual_logical_qp_certificate_count"], 0)
        self.assertEqual(
            zero_actual["actual_optimizer_backend_invocation_count"], 0
        )
        self.assertEqual(
            zero_actual["actual_numerical_backend_continuation_count"], 0
        )
        self.assertFalse(
            zero_actual["solver_schedule_matches_static_maximum"]
        )
        self.assertTrue(zero_actual["static_operation_counts_are_maximum_ceiling"])

    def test_empty_history_is_inactive_and_adds_no_probe_endpoints(self) -> None:
        inventory = _inventory(history_item_count=0)
        payload = inventory.raw_free_payload()
        self.assertEqual(payload["basis_endpoint_count"], 6)
        self.assertEqual(payload["functional_h_probe_endpoint_count"], 0)
        h_metrics = [
            item for item in payload["metrics"] if item["label"].startswith("functional_h")
        ]
        self.assertEqual(len(h_metrics), 2)
        self.assertTrue(all(not item["active"] for item in h_metrics))
        self.assertTrue(
            all(item["inactive_reason"] == "INACTIVE_EMPTY_HISTORY" for item in h_metrics)
        )
        neutral = solve_fixed_e8_routing(_problem(), inventory, arm=FixedE8Arm.NEUTRAL)
        soft = solve_fixed_e8_routing(_problem(), inventory, arm=FixedE8Arm.SOFT)
        self.assertTrue(all(score.influence_count == 0 for score in neutral.scores))
        self.assertEqual(
            [score.active for score in soft.scores],
            [False, True, True, False, False],
        )

    def test_soft_lexicographic_routing_reduces_worst_score(self) -> None:
        problem = _problem()
        inventory = _inventory()
        neutral = solve_fixed_e8_routing(problem, inventory, arm=FixedE8Arm.NEUTRAL)
        soft = solve_fixed_e8_routing(problem, inventory, arm=FixedE8Arm.SOFT)
        neutral_max = max(item.score for item in neutral.scores if item.active)
        soft_max = max(item.score for item in soft.scores if item.active)
        self.assertEqual(neutral.pre_soft_velocity, soft.pre_soft_velocity)
        self.assertEqual(neutral.soft_velocity, soft.soft_velocity)
        self.assertEqual(len(neutral.certificates), 4)
        self.assertEqual(len(soft.certificates), 4)
        self.assertEqual(
            [item.optimizer_pass_count for item in soft.certificates],
            [1, 1, 1, 2],
        )
        self.assertAlmostEqual(
            soft.certificates[2].xi, soft.xi_star, places=12
        )
        self.assertEqual(neutral.velocity, neutral.pre_soft_velocity)
        self.assertEqual(soft.velocity, soft.soft_velocity)
        self.assertLessEqual(soft_max, neutral_max + 1.1e-8)
        self.assertAlmostEqual(
            float(np.dot(problem.signed_progress, np.asarray(soft.velocity))),
            FIXED_E8_KAPPA * soft.p_max,
            places=7,
        )
        self.assertEqual(soft.mode, FixedE8StepMode.JOINT_WRITE)
        self.assertTrue(all(value >= -1.0e-10 for value in soft.velocity))
        self.assertTrue(all(value <= 1.0 + 1.0e-10 for value in soft.velocity))

    def test_nonempty_history_activates_both_functional_h_scores(self) -> None:
        problem = _problem()
        inventory = _inventory(history_item_count=3)
        neutral = solve_fixed_e8_routing(
            problem, inventory, arm=FixedE8Arm.NEUTRAL
        )
        soft = solve_fixed_e8_routing(problem, inventory, arm=FixedE8Arm.SOFT)
        labels = {item.label: item for item in soft.scores}
        self.assertTrue(labels["structural_historical"].active)
        self.assertTrue(labels["functional_h_mean"].active)
        self.assertTrue(labels["functional_h_smoothmax"].active)
        self.assertEqual(
            inventory.raw_free_payload()["basis_endpoint_count"], 6
        )
        self.assertEqual(
            inventory.raw_free_payload()["functional_h_probe_endpoint_count"],
            0,
        )
        self.assertLessEqual(
            max(item.score for item in soft.scores if item.active),
            max(item.score for item in neutral.scores if item.active) + 1.1e-8,
        )

    def test_structural_h_p_budgets_have_zero_new_method_influence(self) -> None:
        low = solve_fixed_e8_routing(
            _problem(budget=0.01), _inventory(), arm=FixedE8Arm.SOFT
        )
        high = solve_fixed_e8_routing(
            _problem(budget=100.0), _inventory(), arm=FixedE8Arm.SOFT
        )
        np.testing.assert_allclose(low.velocity, high.velocity, rtol=0.0, atol=1.0e-12)
        self.assertEqual(low.scores, high.scores)
        # The full receipt identity deliberately retains the legacy budget as
        # comparison telemetry even though it has no decision influence.
        self.assertNotEqual(low.identity_sha256, high.identity_sha256)
        self.assertTrue(
            all(
                score.influence_count == 0 or score.active
                for score in low.scores
            )
        )

    def test_signed_functional_receipt_and_positive_clipping_are_separate(self) -> None:
        metric = FunctionalBasisMetric(
            "functional_p",
            1.0,
            (0.75, 1.5, 1.0, 0.5, 1.25),
            True,
        )
        payload = metric.raw_free_payload()
        self.assertEqual(payload["raw_increment"], [-0.25, 0.5, 0.0, -0.5, 0.25])
        self.assertEqual(payload["positive_clipped_increment"], [0.0, 0.5, 0.0, 0.0, 0.25])
        self.assertEqual(payload["old_1e_3_hinge_access_count"], 0)

    def test_velocity_theta_and_quadratic_units_apply_h_exactly_once(self) -> None:
        rng = np.random.default_rng(20260806)
        baseline = rng.normal(size=13)
        updates = rng.normal(size=(5, 13))
        velocity = np.asarray((0.9, 0.7, 0.4, 0.2, 0.1), dtype=np.float64)
        linear, gram = expanded_terms_from_low_rank(
            baseline=baseline,
            layer_updates=updates,
        )
        direct = direct_low_rank_delta(
            baseline=baseline,
            layer_updates=updates,
            velocity=velocity,
        )
        expanded = expanded_quadratic_delta(
            linear_scaled_by_h=linear,
            gram_scaled_by_h2=gram,
            velocity=velocity,
        )
        self.assertAlmostEqual(direct, expanded, places=12)
        barrier = SimpleNamespace(
            label="pretrained", linear=linear, gram=gram
        )
        self.assertAlmostEqual(
            direct,
            float(structural_contribution_vector(barrier, velocity).sum()),
            places=12,
        )
        double_h = expanded_quadratic_delta(
            linear_scaled_by_h=float(FIXED_E8_H) * linear,
            gram_scaled_by_h2=float(FIXED_E8_H) ** 2 * gram,
            velocity=velocity,
        )
        self.assertNotAlmostEqual(direct, double_h, places=6)
        score = structural_soft_score(
            _problem().pretrained,
            velocity,
            active=True,
            influence_count=1,
        )
        self.assertAlmostEqual(
            score.raw_delta,
            expanded_quadratic_delta(
                linear_scaled_by_h=_problem().pretrained.linear,
                gram_scaled_by_h2=_problem().pretrained.gram,
                velocity=velocity,
            ),
            places=14,
        )

    def test_operation_ceiling_is_exact_and_matched(self) -> None:
        ceiling = FixedE8OperationCeiling().raw_free_payload()
        self.assertEqual(ceiling["per_arm"]["field_count"], 8)
        self.assertEqual(ceiling["per_arm"]["candidate_count"], 8)
        self.assertEqual(ceiling["per_arm"]["functional_basis_endpoint_count"], 48)
        self.assertEqual(ceiling["two_arm"]["field_count"], 16)
        self.assertEqual(ceiling["two_arm"]["functional_basis_endpoint_count"], 96)
        self.assertEqual(ceiling["per_arm"]["qp_solve_count"], 32)
        self.assertEqual(
            ceiling["per_arm"]["qp_backend_invocation_count"], 40
        )
        self.assertEqual(
            ceiling["per_arm"]["stage2_numerical_polish_count"], 8
        )
        self.assertEqual(
            ceiling["per_arm"]["terminal_audit_functional_endpoint_count"],
            8,
        )
        self.assertTrue(ceiling["matched_compute_schedule"])

    def test_runtime_problem_and_factor_units_are_fixed_h(self) -> None:
        field, _, _ = _cold_semantic_field(wall_seconds=0.25)
        signed = SimpleNamespace(
            field_sha256=field.identity_sha256,
            signed_progress=(0.8, 0.6, 0.4, 0.2, 0.1),
        )
        empty = {layer.layer: [] for layer in field.layers}
        history_action = {
            layer.layer: torch.empty(
                (layer.residual.shape[0], 0), dtype=torch.float64
            )
            for layer in field.layers
        }
        built = fixed_e8_runtime._build_fixed_e8_problem(
            field,
            signed,
            accepted_by_layer=empty,
            committed_load_by_layer={layer.layer: 0.0 for layer in field.layers},
            lock=P1ControllerLock(),
            current_history_action_by_layer=history_action,
        )
        np.testing.assert_allclose(
            built.problem.signed_progress,
            float(FIXED_E8_H) * np.asarray(signed.signed_progress),
            rtol=0.0,
            atol=0.0,
        )
        velocity = (1.0, 0.75, 0.5, 0.25, 0.125)
        factors = fixed_e8_runtime.fixed_e8_waypoint_factors(
            field, velocity, step_index=0
        )
        self.assertEqual(
            [factors[layer.weight_name].theta for layer in field.layers],
            [float(FIXED_E8_H) * item for item in velocity],
        )

    def test_runtime_has_no_adaptive_or_rejection_control_reachability(self) -> None:
        tree = ast.parse(inspect.getsource(fixed_e8_runtime._run_fixed_variant))
        called = set()
        attributes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
            elif isinstance(node, ast.Attribute):
                attributes.add(node.attr)
        self.assertFalse(
            {
                "AdaptiveTauClock",
                "begin_trial",
                "reject",
                "backtrack",
                "_run_trial",
            }
            & called
        )
        self.assertNotIn("retry_index", attributes)
        self.assertIn("_fixed_e8_advance_grid_transition", called)
        transition_source = inspect.getsource(
            fixed_e8_runtime._fixed_e8_advance_grid_transition
        )
        self.assertIn("clock.advance", transition_source)
        self.assertIn('ledger.increment("trial")', transition_source)
        source = inspect.getsource(fixed_e8_runtime)
        self.assertNotIn("_functional_p_probe_and_transform", source)
        self.assertNotIn("FUNCTIONAL_P_BUDGET", source)

    def test_cold_action_graph_has_no_native_or_direct_z_call(self) -> None:
        functions = (
            fixed_e8_runtime._run_fixed_variant,
            fixed_e8_runtime._build_fixed_field_with_metric,
            fixed_e8_runtime._fixed_e8_signed_progress_gradient,
            fixed_e8_runtime._fixed_e8_functional_basis_probe,
            fixed_e8_runtime._build_fixed_e8_problem,
            fixed_e8_runtime._fixed_capacity_payload,
        )
        called: set[str] = set()
        for function in functions:
            tree = ast.parse(inspect.getsource(function))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
        self.assertFalse(
            {
                "capture_p1_native_entry",
                "compute_z",
                "load_direct_z",
                "capture_direct_z",
            }
            & called
        )
        diagnostic = inspect.getsource(fixed_e8_runtime.run_fixed_e8_diagnostic)
        self.assertLess(
            diagnostic.index("action_freeze ="),
            diagnostic.index("capture_p1_native_entry("),
        )
        self.assertNotIn("model.generate", inspect.getsource(fixed_e8_runtime))

    def test_resource_forecast_and_live_capacity_are_fail_closed(self) -> None:
        base_lock = ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            forecast = forecast_fixed_e8_panel(
                LOCKS / "p0_artifact_lock.json", base_lock, alias
            )
            self.assertTrue(forecast.fits_envelope)
            self.assertEqual(forecast.functional_basis_endpoints_per_arm, 48)
            self.assertEqual(forecast.qp_solves_per_arm, 32)
            self.assertEqual(forecast.qp_backend_invocations_per_arm, 40)
            self.assertEqual(
                expected_fixed_e8_result_name(alias),
                f"s05-cold-fixed-e8-soft-p1r7-r5-{alias}-v1",
            )
            receipt = validate_fixed_e8_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=(
                    forecast.allocatable_calibration_bytes
                ),
                allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                free_bytes=forecast.allocatable_calibration_bytes,
            )
            self.assertTrue(receipt["passed"])
            self.assertEqual(
                receipt["stable_device_total_bytes"],
                forecast.allocatable_calibration_bytes,
            )
            self.assertEqual(
                receipt["physical_inventory_total_bytes"],
                forecast.physical_total_bytes,
            )
            required = forecast.conservative_gpu_peak_mib * 1024 * 1024
            self.assertEqual(receipt["required_free_bytes"], required)
            reserved_context = validate_fixed_e8_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=(
                    forecast.allocatable_calibration_bytes
                ),
                allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                free_bytes=required,
            )
            self.assertTrue(reserved_context["passed"])
            with self.assertRaisesRegex(ODEBFContractError, "device identity"):
                validate_fixed_e8_runtime_gpu_capacity(
                    forecast,
                    device_property_total_bytes=(
                        forecast.allocatable_calibration_bytes - 1
                    ),
                    allocatable_total_bytes=(
                        forecast.allocatable_calibration_bytes - 1
                    ),
                    free_bytes=required,
                )
            with self.assertRaisesRegex(ODEBFContractError, "insufficient"):
                validate_fixed_e8_runtime_gpu_capacity(
                    forecast,
                    device_property_total_bytes=(
                        forecast.allocatable_calibration_bytes
                    ),
                    allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                    free_bytes=required - 1,
                )

    def test_rooted_panel_lock_case_provenance_and_dry_plan_are_exact(self) -> None:
        source_case = verify_cold_case_seal(
            json.loads(
                (LOCKS / "p1r6_cold_cf_b10_seal.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        provenance = json.loads(
            (LOCKS / "p1r7_fixed_e8_case_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        verified = verify_fixed_e8_case_provenance(
            provenance, source_seal=source_case
        )
        self.assertEqual(verified["request_count"], 10)
        self.assertTrue(verified["ordered_seal_reused_exactly"])
        self.assertFalse(verified["fresh_selection_performed"])

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "provenance.json"

            def write_rooted(value: dict[str, object]) -> None:
                payload = dict(value)
                payload.pop("root_digest", None)
                payload["root_digest"] = canonical_hash(payload)
                candidate.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )

            write_rooted(provenance)
            loaded, _ = load_fixed_e8_case_provenance(
                candidate, source_seal=source_case
            )
            self.assertEqual(loaded, verified)

            schema_version_only = dict(provenance)
            schema_version_only["schema_version"] = schema_version_only.pop(
                "schema"
            )
            write_rooted(schema_version_only)
            with self.assertRaisesRegex(ODEBFContractError, "provenance differs"):
                load_fixed_e8_case_provenance(
                    candidate, source_seal=source_case
                )

            mismatched = dict(provenance)
            mismatched["schema"] = "ode-edit-wrong-schema/v1"
            write_rooted(mismatched)
            with self.assertRaisesRegex(ODEBFContractError, "provenance differs"):
                load_fixed_e8_case_provenance(
                    candidate, source_seal=source_case
                )

        base_sampling = load_p1_sampling_seal(
            LOCKS / "p1r2_p_population_seal.json",
            stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
        )
        schedule = fixed_e8_schedule(base_sampling)
        population = json.loads(
            (LOCKS / "p1r2_p_population_seal.json").read_text(
                encoding="utf-8"
            )
        )
        numerical = json.loads(
            (
                LOCKS / "numerical_lock_s05_fixed_e8_structfunc_soft.json"
            ).read_text(encoding="utf-8")
        )
        validated = validate_fixed_e8_lock(
            numerical,
            controller_identity_sha256=P1ControllerLock().identity(),
            case_root_digest=source_case["root_digest"],
            population_root_digest=population["root_digest"],
            schedule=schedule,
        )
        self.assertEqual(validated["parent_checkpoint"], FIXED_E8_PARENT_HEAD)
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "numerical.json"

            def write_numerical(value: dict[str, object]) -> None:
                payload = dict(value)
                payload.pop("root_digest", None)
                payload["root_digest"] = canonical_hash(payload)
                candidate.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )

            write_numerical(numerical)
            loaded, _ = load_and_validate_fixed_e8_lock(
                candidate,
                controller_identity_sha256=P1ControllerLock().identity(),
                case_root_digest=source_case["root_digest"],
                population_root_digest=population["root_digest"],
                schedule=schedule,
            )
            self.assertEqual(loaded, validated)

            schema_version_only = dict(numerical)
            schema_version_only["schema_version"] = schema_version_only.pop(
                "schema"
            )
            write_numerical(schema_version_only)
            with self.assertRaisesRegex(ODEBFContractError, "lock differs"):
                load_and_validate_fixed_e8_lock(
                    candidate,
                    controller_identity_sha256=P1ControllerLock().identity(),
                    case_root_digest=source_case["root_digest"],
                    population_root_digest=population["root_digest"],
                    schedule=schedule,
                )

            mismatched = dict(numerical)
            mismatched["schema"] = "ode-edit-wrong-schema/v1"
            write_numerical(mismatched)
            with self.assertRaisesRegex(ODEBFContractError, "lock differs"):
                load_and_validate_fixed_e8_lock(
                    candidate,
                    controller_identity_sha256=P1ControllerLock().identity(),
                    case_root_digest=source_case["root_digest"],
                    population_root_digest=population["root_digest"],
                    schedule=schedule,
                )
        changed = json.loads(json.dumps(numerical))
        changed["hard_structural_p_budget_influence_count"] = 1
        body = dict(changed)
        body.pop("root_digest")
        changed["root_digest"] = canonical_hash(body)
        with self.assertRaisesRegex(ODEBFContractError, "numerical lock"):
            validate_fixed_e8_lock(
                changed,
                controller_identity_sha256=P1ControllerLock().identity(),
                case_root_digest=source_case["root_digest"],
                population_root_digest=population["root_digest"],
                schedule=schedule,
            )

        plan = fixed_e8_dry.build_plan("f" * 40, repository_root=ROOT)
        self.assertTrue(plan["execution_hold"])
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["gpu_use"])
        self.assertFalse(plan["slurm_submit"])
        self.assertFalse(plan["result_root_creation"])
        self.assertEqual(plan["new_pair_gpu"], 2)
        self.assertEqual(plan["server1_project_gpu_cap"], 3)
        self.assertEqual(
            [item["result_name"] for item in plan["jobs"]],
            [
                expected_fixed_e8_result_name("llama3-8b-inst"),
                expected_fixed_e8_result_name("qwen2.5-7b-inst"),
            ],
        )
        _validate_raw_free(plan)
        with self.assertRaises(ODEBFContractError):
            fixed_e8_dry.build_plan("not-a-commit", repository_root=ROOT)

    def test_runtime_dispatch_is_additive_and_modes_remain_exclusive(self) -> None:
        from project.run_scripts.ode_bf import p1_runtime

        source = inspect.getsource(p1_runtime.run_p1)
        self.assertIn("fixed_e8_soft_mode: bool = False", source)
        self.assertIn("run_fixed_e8_diagnostic", source)
        self.assertIn("cold_structp_softp_noveto_mode", source)
        self.assertIn("newnll_p_soft_hard_mode", source)
        with mock.patch.object(p1_runtime, "_source_freeze", return_value=None):
            with self.assertRaisesRegex(ODEBFContractError, "mutually exclusive"):
                p1_runtime.run_p1(
                    repo_root=ROOT,
                    alias="llama3-8b-inst",
                    output_root=(
                        ROOT / "local/odebf/results/never-created-fixed-e8-test"
                    ),
                    source_head=FIXED_E8_PARENT_HEAD,
                    cold_structp_softp_noveto_mode=True,
                    fixed_e8_soft_mode=True,
                )
        self.assertFalse(
            (ROOT / "local/odebf/results/never-created-fixed-e8-test").exists()
        )

    def test_submitter_is_checkpoint_approval_gated_before_any_write(self) -> None:
        # The future submitter is shipped dormant.  Missing checkpoint-bound
        # approval must fail before any state/root/scheduler write is reachable.
        calls: list[tuple[str, ...]] = []

        child = "d" * 40

        def fake_run(args, *, check=True):
            del check
            calls.append(tuple(args))
            output = {
                ("git", "rev-parse", "HEAD"): child + "\n",
                ("git", "rev-parse", "HEAD^"): (
                    fixed_e8_submit.FIXED_E8_ZERO_CAPACITY_PARENT_HEAD + "\n"
                ),
                ("git", "rev-parse", "HEAD^^"): (
                    fixed_e8_submit.FIXED_E8_MEMORY_PARENT_HEAD + "\n"
                ),
                ("git", "rev-parse", "HEAD^^^"): (
                    fixed_e8_submit.FIXED_E8_NUMERICAL_SCHEMA_PARENT_HEAD
                    + "\n"
                ),
                ("git", "rev-parse", "HEAD^^^^"): (
                    fixed_e8_submit.FIXED_E8_LAUNCHER_PARENT_HEAD + "\n"
                ),
                ("git", "rev-parse", "HEAD^^^^^"): (
                    fixed_e8_submit.FIXED_E8_REPAIR_PARENT_HEAD + "\n"
                ),
                ("git", "rev-parse", "HEAD^^^^^^"): (
                    fixed_e8_submit.FIXED_E8_REVIEW_PARENT_HEAD + "\n"
                ),
                ("git", "rev-parse", "HEAD^^^^^^^"): FIXED_E8_PARENT_HEAD + "\n",
                ("git", "branch", "--show-current"): (
                    "codex/odeeditsh1-s05-fixed-e8-soft-routing-p1r7-v1\n"
                ),
                (
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ): "",
                (
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    FIXED_E8_PARENT_HEAD,
                    child,
                ): "",
            }[tuple(args)]
            return SimpleNamespace(stdout=output, stderr="", returncode=0)

        with mock.patch.object(fixed_e8_submit, "_run", side_effect=fake_run):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ODEBFContractError, "approval"):
                    fixed_e8_submit._execution_provenance_gate(
                        child
                    )
        self.assertFalse(any(item and item[0] in {"sbatch", "scontrol"} for item in calls))

        with mock.patch.object(fixed_e8_submit, "_run", side_effect=fake_run):
            with mock.patch.dict(
                os.environ,
                {
                    fixed_e8_submit.APPROVAL_ENV: (
                        f"{FIXED_E8_INSTRUCTION_ID}:{child}"
                    )
                },
                clear=True,
            ):
                receipt = fixed_e8_submit._execution_provenance_gate(child)
        self.assertEqual(
            receipt["exact_zero_capacity_parent"],
            fixed_e8_submit.FIXED_E8_ZERO_CAPACITY_PARENT_HEAD,
        )
        self.assertEqual(
            receipt["exact_memory_parent"],
            fixed_e8_submit.FIXED_E8_MEMORY_PARENT_HEAD,
        )
        self.assertEqual(
            receipt["exact_numerical_schema_parent"],
            fixed_e8_submit.FIXED_E8_NUMERICAL_SCHEMA_PARENT_HEAD,
        )
        self.assertEqual(
            receipt["exact_launcher_parent"],
            fixed_e8_submit.FIXED_E8_LAUNCHER_PARENT_HEAD,
        )
        self.assertEqual(
            receipt["exact_repair_parent"],
            fixed_e8_submit.FIXED_E8_REPAIR_PARENT_HEAD,
        )
        self.assertEqual(
            receipt["exact_review_parent"],
            fixed_e8_submit.FIXED_E8_REVIEW_PARENT_HEAD,
        )
        self.assertEqual(receipt["exact_scientific_parent"], FIXED_E8_PARENT_HEAD)

        def wrong_parent(args, *, check=True):
            result = fake_run(args, check=check)
            if tuple(args) == ("git", "rev-parse", "HEAD^"):
                return SimpleNamespace(stdout="e" * 40 + "\n", stderr="", returncode=0)
            return result

        with mock.patch.object(fixed_e8_submit, "_run", side_effect=wrong_parent):
            with mock.patch.dict(
                os.environ,
                {
                    fixed_e8_submit.APPROVAL_ENV: (
                        f"{FIXED_E8_INSTRUCTION_ID}:{child}"
                    )
                },
                clear=True,
            ):
                with self.assertRaisesRegex(ODEBFContractError, "provenance"):
                    fixed_e8_submit._execution_provenance_gate(child)

    def test_sbatch_invocation_token_matches_panel_contract(self) -> None:
        source = (
            ROOT
            / "project/run_scripts/session05_ode_bf_fixed_e8_structfunc_soft.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn(
            f'"${{RUN_TOKEN}}" == "{FIXED_E8_RESULT_TOKEN}"', source
        )
        self.assertNotIn(
            '"${RUN_TOKEN}" == "cold-fixed-e8-structfunc-soft-p1r7-v1"',
            source,
        )

    def test_layer_local_fz_contract_is_retained_by_reference(self) -> None:
        source = inspect.getsource(cold_start_target._ColdLayerLocalTargetOverlay)
        self.assertIn("residual + (", source)
        self.assertIn("target_state_variable", source)
        self.assertEqual(
            cold_start_target._ColdLayerLocalTargetOverlay.DEFINITION,
            "R_l(z_s)+(z-z_s)",
        )
        receipt = fixed_e8_semantic_receipt()
        self.assertEqual(receipt["target_overlay_definition"], "R_l(z_k)+(z-z_k)")
        self.assertFalse(receipt["candidate_coupled"])
        self.assertFalse(receipt["retry_recomputes_target_velocity"])
        self.assertEqual(receipt["native_or_direct_z_cold_access_count"], 0)

    def test_malformed_inventory_and_problem_fail_closed(self) -> None:
        with self.assertRaises(ODEBFContractError):
            FunctionalBasisMetric("functional_p", 0.0, (0.0,) * 4, True)
        with self.assertRaises(ODEBFContractError):
            solve_fixed_e8_routing(
                _problem(),
                FixedE8SoftInventory(
                    FunctionalBasisMetric("functional_p", 0.0, (0.0,) * 5, True),
                    FunctionalBasisMetric(
                        "functional_h_mean", 0.0, (0.0,) * 5, False, "INACTIVE_EMPTY_HISTORY"
                    ),
                    FunctionalBasisMetric(
                        "functional_h_smoothmax", 0.0, (0.0,) * 5, False, "INACTIVE_EMPTY_HISTORY"
                    ),
                    0,
                    "a" * 64,
                    "0" * 64,
                ),
                arm=FixedE8Arm.NEUTRAL,
            )


if __name__ == "__main__":
    unittest.main()
