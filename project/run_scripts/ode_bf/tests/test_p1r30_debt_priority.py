from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from project.run_scripts import session05_ode_bf_p1r30_debt_priority_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    p1r24_disable_historical,
    solve_p1r24_matched_routing,
)
from project.run_scripts.ode_bf.p1r30_debt_priority import (
    P1R30_INSTRUCTION_ID,
    P1R30DebtState,
    P1R30RoutingStatus,
    P1R30_H,
    _RequestwiseCoefficientOverlay,
    debt_priority,
    nominal_demand_from_target_displacement,
    solve_p1r30_a0_relative_routing,
)
from project.run_scripts.ode_bf.p1r30_debt_priority_experiment import (
    p1r30_role_dispatch,
)
from project.run_scripts.ode_bf.p1r30_debt_priority_panel import (
    P1R30_ROLES,
    expected_p1r30_result_name,
)
from project.run_scripts.ode_bf.progress_simplex_routing import (
    progress_simplex_waypoint_factors,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _barrier(
    label: str,
    *,
    linear: np.ndarray,
    gram: np.ndarray,
) -> QuadraticBarrier:
    return QuadraticBarrier(
        label,
        0.0,
        linear,
        gram,
        1.0e6,
        "layer-local-diagonal",
    )


def _problem(
    slopes: tuple[float, ...] = (0.5, 0.3, 0.2, 0.0, -0.1),
) -> RoutingProblem:
    size = len(slopes)
    return p1r24_disable_historical(
        RoutingProblem(
            np.asarray(slopes, dtype=np.float64),
            np.diag(np.asarray((0.7, 0.9, 1.1, 1.3, 1.5))),
            np.diag(np.asarray((0.5, 0.6, 0.7, 0.8, 0.9))),
            1.0,
            np.ones(size),
            1.0e-12,
            1.0e-12,
            _barrier(
                "historical",
                linear=np.zeros(size),
                gram=np.zeros((size, size)),
            ),
            _barrier(
                "pretrained",
                linear=np.asarray((0.8, -0.1, -0.05, 0.0, 0.0)),
                gram=np.diag(np.asarray((0.8, 0.05, 0.05, 0.1, 0.1))),
            ),
        )
    )


def _request_slopes(problem: RoutingProblem) -> tuple[tuple[float, ...], ...]:
    raw_mean = problem.signed_progress / P1R30_H
    delta = np.asarray((0.4, -0.2, 0.1, 0.0, -0.1))
    return (
        tuple(float(item) for item in raw_mean + delta),
        tuple(float(item) for item in raw_mean - delta),
    )


class _FiveLayerToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(3, 3, bias=False) for _ in range(5)
        )
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class P1R30DebtPriorityTests(unittest.TestCase):
    def test_zero_u_only_is_uniform_and_nonuniform_nominal_is_preserved(self) -> None:
        zero = debt_priority((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), step_index=0)
        self.assertTrue(zero.all_u_zero)
        np.testing.assert_allclose(zero.omega, (1.0 / 3.0,) * 3)
        nonuniform = debt_priority((1.0, 2.0, 0.0), (0.0, 0.0, 0.0), step_index=0)
        self.assertFalse(nonuniform.all_u_zero)
        self.assertGreater(nonuniform.omega[1], nonuniform.omega[0])
        self.assertEqual(nonuniform.omega[2], 0.0)

    def test_nominal_demand_is_requestwise_not_batch_mean_dot(self) -> None:
        gradient = torch.tensor([[-2.0, 1.0]], dtype=torch.float64)
        displacement = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        observed = nominal_demand_from_target_displacement(gradient, displacement)
        self.assertEqual(observed.per_request, (0.2, 0.0))
        self.assertAlmostEqual(observed.aggregate, 0.1)
        substituted = max(
            -float(gradient.mean(dim=1) @ displacement.mean(dim=1)), 0.0
        )
        self.assertNotAlmostEqual(observed.aggregate, substituted)
        self.assertEqual(observed.raw_free_payload()["lag_influence_count"], 0)

    def test_debt_is_truthful_requestwise_and_negative_actual_increases_it(self) -> None:
        nominal = nominal_demand_from_target_displacement(
            torch.tensor([[-2.0, -1.0]], dtype=torch.float64),
            torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        )
        state = P1R30DebtState.initial((2.0, 3.0))
        state1, receipt = state.complete(nominal, (1.9, 3.1))
        np.testing.assert_allclose(receipt.actual_progress, (0.1, -0.1))
        np.testing.assert_allclose(state1.debt, (0.1, 0.3))
        self.assertEqual(receipt.raw_free_payload()["negative_actual_count"], 1)
        priority = state1.priority(nominal)
        np.testing.assert_allclose(
            priority.u,
            (0.2 + 0.1 / 7.0, 0.2 + 0.3 / 7.0),
        )

    def test_terminal_eighth_completion_is_not_left_delayed(self) -> None:
        nominal = nominal_demand_from_target_displacement(
            torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
            torch.tensor([[0.25, 0.25]], dtype=torch.float64),
        )
        state = P1R30DebtState.initial((8.0, 8.0))
        for index in range(8):
            state, _ = state.complete(nominal, (7.0 - index, 7.0 - index))
        self.assertEqual(state.step_index, 8)

    def test_neutral_is_exact_p1r24_a0_and_debt_has_zero_influence(self) -> None:
        problem = _problem()
        rho = 0.75
        inherited = solve_p1r24_matched_routing(
            problem, arm=FixedE8Arm.NEUTRAL, rho_write=rho
        )
        priority_a = debt_priority((1.0, 2.0), (0.0, 0.0), step_index=0)
        priority_b = debt_priority((1.0, 2.0), (50.0, 1.0), step_index=0)
        first = solve_p1r30_a0_relative_routing(
            problem,
            arm=FixedE8Arm.NEUTRAL,
            rho_reference=rho,
            requestwise_signed_progress=_request_slopes(problem),
            priority=priority_a,
        )
        second = solve_p1r30_a0_relative_routing(
            problem,
            arm=FixedE8Arm.NEUTRAL,
            rho_reference=rho,
            requestwise_signed_progress=_request_slopes(problem),
            priority=priority_b,
        )
        np.testing.assert_allclose(first.selected_c, inherited.applied_coefficient)
        np.testing.assert_array_equal(first.selected_c, second.selected_c)
        self.assertEqual(first.status, P1R30RoutingStatus.A0_REFERENCE)
        self.assertEqual(
            first.raw_free_payload()["debt_total_update_magnitude_influence_count"],
            0,
        )

    def test_soft_direct_c_constraints_and_signed_progress_are_preserved(self) -> None:
        problem = _problem()
        priority = debt_priority((0.1, 0.8), (0.5, 0.0), step_index=3)
        observed = solve_p1r30_a0_relative_routing(
            problem,
            arm=FixedE8Arm.SOFT,
            rho_reference=0.75,
            requestwise_signed_progress=_request_slopes(problem),
            priority=priority,
        )
        self.assertGreaterEqual(observed.selected_mean_progress + 1.0e-8,
                                observed.reference_mean_progress)
        self.assertGreaterEqual(observed.selected_debt_progress + 1.0e-8,
                                observed.reference_debt_progress)
        self.assertGreaterEqual(
            observed.selected_unnormalized_debt_progress + 1.0e-8,
            observed.reference_unnormalized_debt_progress,
        )
        self.assertLessEqual(observed.energy_ratio, 1.0 + 1.0e-7)
        self.assertTrue(all(item >= -1.0e-10 for item in observed.selected_c))
        self.assertEqual(observed.raw_free_payload()["inverse_slope_operation_count"], 0)

    def test_soft_solver_failure_falls_back_to_exact_c0(self) -> None:
        problem = _problem()
        failed = SimpleNamespace(
            success=False,
            status=9,
            nit=1,
            message="forced numerical failure",
            x=np.ones(5),
        )
        with mock.patch(
            "project.run_scripts.ode_bf.p1r30_debt_priority.minimize",
            return_value=failed,
        ):
            observed = solve_p1r30_a0_relative_routing(
                problem,
                arm=FixedE8Arm.SOFT,
                rho_reference=0.75,
                requestwise_signed_progress=_request_slopes(problem),
                priority=debt_priority((1.0, 2.0), (0.0, 0.0), step_index=0),
            )
        self.assertEqual(observed.status, P1R30RoutingStatus.SOFT_FALLBACK_NEUTRAL)
        np.testing.assert_array_equal(observed.selected_c, observed.c0)

    def test_applied_c_is_waypoint_theta_with_h_exactly_once(self) -> None:
        problem = _problem()
        observed = solve_p1r30_a0_relative_routing(
            problem,
            arm=FixedE8Arm.NEUTRAL,
            rho_reference=0.75,
            requestwise_signed_progress=_request_slopes(problem),
            priority=debt_priority((1.0, 2.0), (0.0, 0.0), step_index=0),
        )
        field = SimpleNamespace(
            layers=tuple(
                SimpleNamespace(
                    weight_name=f"layers.{index + 4}.weight",
                    layer=index + 4,
                    residual=torch.ones((2, 2)),
                    q=torch.ones((2, 2)),
                    factor=SimpleNamespace(global_batch_size=2),
                )
                for index in range(5)
            )
        )
        factors = progress_simplex_waypoint_factors(
            field, observed.velocity, step_index=0
        )
        np.testing.assert_allclose(
            [factors[layer.weight_name].theta for layer in field.layers],
            observed.selected_c,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_same_backward_requestwise_overlay_reduces_to_shared_slope(self) -> None:
        model = _FiveLayerToy()
        fields = tuple(
            SimpleNamespace(
                weight_name=f"layers.{index}.weight",
                q=torch.tensor([[1.0], [0.5], [0.25]]),
                residual=torch.tensor([[0.5], [0.25], [0.125]]),
            )
            for index in range(5)
        )
        inputs = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]], dtype=torch.float32
        )
        request_coefficients = torch.zeros((2, 5), requires_grad=True)
        with _RequestwiseCoefficientOverlay(
            model, fields, request_coefficients, (0, 1)
        ):
            per_loss = model(inputs).sum()
        per_gradient = torch.autograd.grad(per_loss, request_coefficients)[0]

        shared = torch.zeros(5, requires_grad=True)
        expanded = shared.unsqueeze(0).expand(2, -1)
        with _RequestwiseCoefficientOverlay(model, fields, expanded, (0, 1)):
            shared_loss = model(inputs).sum() / 2.0
        shared_gradient = torch.autograd.grad(shared_loss, shared)[0]
        torch.testing.assert_close(
            shared_gradient,
            per_gradient.mean(dim=0),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_reduction_certificate_uses_model_facing_fp32_coordinate(self) -> None:
        raw_mean = np.asarray(
            (1.0e9 + 33.0, 7.0e8 + 17.0, 4.0e8 + 9.0, 2.0e8 + 5.0, 1.0e8 + 3.0),
            dtype=np.float64,
        )
        public_applied = (
            raw_mean.astype(np.float32).astype(np.float64) * P1R30_H
        )
        problem = _problem(tuple(float(item) for item in public_applied))
        delta = np.asarray((3.0, -2.0, 1.0, -0.5, 0.25), dtype=np.float64)
        observed = solve_p1r30_a0_relative_routing(
            problem,
            arm=FixedE8Arm.NEUTRAL,
            rho_reference=0.75,
            requestwise_signed_progress=(
                tuple(float(item) for item in raw_mean + delta),
                tuple(float(item) for item in raw_mean - delta),
            ),
            priority=debt_priority((1.0, 2.0), (0.0, 0.0), step_index=0),
        )
        self.assertEqual(observed.reduction_max_abs_residual, 0.0)
        self.assertGreater(
            observed.reduction_representation_rounding_max_abs,
            P1R30_H * 1.0e-7,
        )
        self.assertEqual(observed.status, P1R30RoutingStatus.A0_REFERENCE)

    def test_router_ast_has_direct_c_constraints_and_no_legacy_inverse_path(self) -> None:
        path = Path(__file__).parents[1] / "p1r30_debt_priority.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        called: set[str] = set()
        forbidden_divisors = {
            "slopes",
            "slope",
            "a_mean",
            "a_debt",
            "request_slopes",
            "applied_mean",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Name):
                    self.assertNotIn(node.right.id, forbidden_divisors)
        self.assertNotIn("solve_exact_strength_soft_hp", imported)
        self.assertNotIn("solve_exact_strength_soft_hp", called)
        source = path.read_text(encoding="utf-8")
        self.assertIn('"c>=0"', source)
        self.assertIn('"a_mean_dot_c>=a_mean_dot_c0"', source)
        self.assertIn('"a_debt_dot_c>=a_debt_dot_c0"', source)
        self.assertIn('"inverse_slope_operation_count": 0', source)

    def test_a1_full_matrix_roles_and_names_are_distinct(self) -> None:
        self.assertEqual(
            P1R30_ROLES,
            (
                "P1R30_B1_RS_REFERENCE_SOFT",
                "P1R30_B10_RS_DEBT_NEUTRAL",
                "P1R30_B10_RS_DEBT_SOFT",
                "P1R30_B10_BG_PAIR",
                "P1R30_B10_BG_DEBT_SOFT",
            ),
        )
        names = {
            expected_p1r30_result_name(alias, role)
            for alias in dry.ALIASES
            for role in P1R30_ROLES
        }
        self.assertEqual(len(names), len(dry.ALIASES) * len(P1R30_ROLES))

    def test_a2_bg_soft_isolation_is_dispatch_only(self) -> None:
        allocation, selected, paired = p1r30_role_dispatch(
            "P1R30_B10_BG_DEBT_SOFT"
        )
        self.assertEqual(allocation, "BG")
        self.assertIs(selected, FixedE8Arm.SOFT)
        self.assertFalse(paired)
        self.assertEqual(
            expected_p1r30_result_name(
                "qwen2.5-7b-inst",
                "P1R30_B10_BG_DEBT_SOFT",
                technical_attempt="a2-isolation-r1",
            ),
            (
                "s05-p1r30-qwen2.5-7b-inst-"
                "b10-bg-debt-priority-soft-isolated-v1-a2-isolation-r1"
            ),
        )
        self.assertEqual(
            p1r30_role_dispatch("P1R30_B10_BG_PAIR"),
            ("BG", None, True),
        )
        self.assertEqual(
            p1r30_role_dispatch("P1R30_B10_RS_DEBT_SOFT"),
            ("RS", FixedE8Arm.SOFT, False),
        )

    def test_a1_dry_plan_completes_eight_distinct_b10_cells(self) -> None:
        neutral = dry.build_plan("a" * 40, "b10-neutral")
        completion = dry.build_plan("a" * 40, "full-matrix-completion")
        self.assertEqual(neutral["trajectory_count"], 2)
        self.assertEqual(completion["trajectory_count"], 6)
        self.assertEqual(completion["job_count"], 4)
        self.assertEqual(completion["stage_max_concurrent_gpu"], 4)
        self.assertEqual(
            [job["role"] for job in completion["jobs"]],
            [
                "P1R30_B10_RS_DEBT_SOFT",
                "P1R30_B10_RS_DEBT_SOFT",
                "P1R30_B10_BG_PAIR",
                "P1R30_B10_BG_PAIR",
            ],
        )
        self.assertEqual(
            len({job["result_name"] for job in neutral["jobs"] + completion["jobs"]}),
            6,
        )

    def test_technical_attempt_changes_only_result_namespace(self) -> None:
        base = dry.build_plan("a" * 40, "b10-neutral")
        repaired = dry.build_plan(
            "a" * 40, "b10-neutral", attempt_tag="tech-r1"
        )
        self.assertIsNone(base["technical_attempt_tag"])
        self.assertEqual(repaired["technical_attempt_tag"], "tech-r1")
        for original, child in zip(base["jobs"], repaired["jobs"], strict=True):
            expected = dict(original)
            expected["result_name"] = f'{original["result_name"]}-tech-r1'
            self.assertEqual(child, expected)
        with self.assertRaisesRegex(ODEBFContractError, "result identity"):
            dry.build_plan(
                "a" * 40, "b10-neutral", attempt_tag="../../alias"
            )

    def test_runtime_namespace_adapter_is_exact_and_fail_closed(self) -> None:
        base = expected_p1r30_result_name(
            "llama3-8b-inst", "P1R30_B10_RS_DEBT_NEUTRAL"
        )
        repaired = expected_p1r30_result_name(
            "llama3-8b-inst",
            "P1R30_B10_RS_DEBT_NEUTRAL",
            technical_attempt="tech-r2",
        )
        self.assertEqual(repaired, f"{base}-tech-r2")
        with self.assertRaisesRegex(ODEBFContractError, "result identity"):
            expected_p1r30_result_name(
                "llama3-8b-inst",
                "P1R30_B10_RS_DEBT_NEUTRAL",
                technical_attempt="../escape",
            )

    def test_numerical_lock_binds_direct_c_and_full_matrix(self) -> None:
        lock_path = (
            Path(__file__).parents[1]
            / "locks/numerical_lock_s05_p1r30_debt_priority.json"
        )
        lock, _ = load_rooted_json(
            lock_path,
            expected_schema="ode-edit-s05-p1r30-debt-priority-lock/v1",
        )
        self.assertEqual(lock["instruction_id"], P1R30_INSTRUCTION_ID)
        self.assertEqual(lock["soft"]["parameterization"], "DIRECT_APPLIED_C_SPACE")
        self.assertEqual(lock["soft"]["inverse_slope_operation_count"], 0)
        self.assertEqual(lock["execution"]["distinct_b10_cell_count"], 8)
        self.assertEqual(lock["execution"]["server2_user_gpu_cap"], 4)

    def test_source_manifest_is_rooted_and_matches_sealed_source_bytes(self) -> None:
        root = Path(__file__).parents[4]
        manifest, _ = load_rooted_json(
            Path(__file__).parents[1]
            / "locks/source_manifest_s05_p1r30_debt_priority.json",
            expected_schema=(
                "ode-edit-s05-p1r30-debt-priority-source-manifest/v1"
            ),
        )
        self.assertEqual(
            manifest["base_checkpoint"],
            "ce8c6c36348752f1407f7d713d30e6b5c727379b",
        )
        paths = [entry["path"] for entry in manifest["entries"]]
        self.assertEqual(paths, sorted(set(paths)))
        sealed_source_head = "8fcf9dea22a684edf6c02556238a5bdcd80e9f51"
        for entry in manifest["entries"]:
            payload = subprocess.run(
                [
                    "git",
                    "show",
                    f"{sealed_source_head}:{entry['path']}",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(len(payload), entry["size"])
            self.assertEqual(
                __import__("hashlib").sha256(payload).hexdigest(),
                entry["sha256"],
            )

    def test_launch_source_has_no_callback_and_exact_stage_mapping(self) -> None:
        scripts = Path(__file__).parents[2]
        sbatch = (
            scripts / "session05_ode_bf_p1r30_debt_priority.sbatch"
        ).read_text(encoding="utf-8")
        submitter = (
            scripts / "session05_ode_bf_submit_p1r30_debt_priority.py"
        ).read_text(encoding="utf-8")
        self.assertIn("P1R30_B10_BG_PAIR P1R30_B10_BG_PAIR", sbatch)
        self.assertIn("a2-qwen-bg-soft-isolated)", sbatch)
        self.assertIn("ROLES=(P1R30_B10_BG_DEBT_SOFT)", sbatch)
        self.assertIn("MODELS=(qwen2.5-7b-inst)", sbatch)
        self.assertIn('technical_attempt="${TECHNICAL_ATTEMPT}"', sbatch)
        self.assertIn('"callback_job_count": 0', submitter)
        self.assertNotIn("--dependency", submitter)
        self.assertNotIn("afterany", submitter.casefold())

    def test_paired_debt_receipts_are_arm_namespaced(self) -> None:
        runtime = (
            Path(__file__).parents[1] / "p1_scalable_batched_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            runtime.count('/ "debt"\n                        / arm_label.lower()'),
            2,
        )


if __name__ == "__main__":
    unittest.main()
