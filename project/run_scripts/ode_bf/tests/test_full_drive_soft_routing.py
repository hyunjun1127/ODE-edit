from __future__ import annotations

import ast
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf import full_drive_physical_field
from project.run_scripts.ode_bf import full_drive_soft_routing
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FIXED_E8_H,
    FIXED_E8_LAYER_ORDER,
    FixedE8SoftInventory,
    FunctionalBasisMetric,
)
from project.run_scripts.ode_bf.full_drive_soft_routing import (
    FULL_DRIVE_LAMBDA_GRID,
    FullDriveArm,
    FullDriveStepMode,
    LambdaReplaySource,
    full_drive_numerical_contract,
    replay_and_select_lambda,
    solve_full_drive_routing,
)
from project.run_scripts.ode_bf.full_drive_dynamic_runtime import (
    FULL_DRIVE_CELL_ORDER,
    FullDriveAllocation,
    FullDriveCell,
    full_drive_runtime_contract,
    propose_full_drive_transition,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _problem(
    *,
    slopes: tuple[float, ...] = (0.50, 0.42, 0.30, 0.20, 0.10),
    pretrained_linear: tuple[float, ...] = (0.09, 0.02, 0.01, 0.01, 0.01),
) -> RoutingProblem:
    size = len(FIXED_E8_LAYER_ORDER)
    return RoutingProblem(
        np.asarray(slopes, dtype=np.float64),
        np.diag(np.asarray((1.0, 1.1, 1.2, 1.3, 1.4), dtype=np.float64)),
        np.diag(np.full(size, 0.01, dtype=np.float64)),
        0.25,
        np.ones(size, dtype=np.float64),
        0.01,
        1.0e-8,
        QuadraticBarrier(
            "historical",
            0.0,
            np.zeros(size, dtype=np.float64),
            np.diag(np.full(size, 0.01, dtype=np.float64)),
            0.5,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            0.0,
            np.asarray(pretrained_linear, dtype=np.float64),
            np.diag(np.full(size, 0.025, dtype=np.float64)),
            0.5,
            "layer-local-diagonal",
        ),
    )


def _inventory() -> FixedE8SoftInventory:
    return FixedE8SoftInventory(
        FunctionalBasisMetric(
            "functional_p",
            0.01,
            (0.12, 0.035, 0.025, 0.020, 0.018),
            True,
        ),
        FunctionalBasisMetric(
            "functional_h_mean",
            0.0,
            (0.0,) * 5,
            False,
            "INACTIVE_EMPTY_HISTORY",
        ),
        FunctionalBasisMetric(
            "functional_h_smoothmax",
            0.0,
            (0.0,) * 5,
            False,
            "INACTIVE_EMPTY_HISTORY",
        ),
        0,
        "a" * 64,
        "b" * 64,
    )


class FullDriveSoftRoutingTests(unittest.TestCase):
    def test_no_soft_is_exact_full_nominal_and_not_a_progress_floor(self) -> None:
        slopes = (0.50, 0.42, 0.30, 0.20, 0.10)
        result = solve_full_drive_routing(
            _problem(slopes=slopes),
            _inventory(),
            edit_slopes=slopes,
            arm=FullDriveArm.NO_SOFT,
            lambda_p=0.0,
        )
        self.assertEqual(result.mode, FullDriveStepMode.JOINT_WRITE)
        self.assertEqual(result.velocity, result.nominal_velocity)
        self.assertAlmostEqual(result.r_edit, 1.0, places=10)
        self.assertEqual(result.raw_free_payload()["progress_floor"], 0.0)
        self.assertIsNone(result.raw_free_payload()["requested_progress"])
        self.assertEqual(
            np.asarray(result.applied_coefficient).tolist(),
            (float(FIXED_E8_H) * np.asarray(result.velocity)).tolist(),
        )

    def test_soft_changes_geometry_without_a_floor_or_capacity_primary(self) -> None:
        slopes = (0.50, 0.42, 0.30, 0.20, 0.10)
        no_soft = solve_full_drive_routing(
            _problem(slopes=slopes),
            _inventory(),
            edit_slopes=slopes,
            arm=FullDriveArm.NO_SOFT,
            lambda_p=0.0,
        )
        soft = solve_full_drive_routing(
            _problem(slopes=slopes),
            _inventory(),
            edit_slopes=slopes,
            arm=FullDriveArm.SOFT,
            lambda_p=1.0,
        )
        self.assertEqual(soft.mode, FullDriveStepMode.JOINT_WRITE)
        self.assertNotEqual(soft.velocity, no_soft.velocity)
        self.assertLess(soft.r_ph, no_soft.r_ph)
        self.assertGreater(soft.r_edit, 0.0)
        receipt = soft.raw_free_payload()
        self.assertEqual(receipt["capacity_primary_objective_count"], 0)
        self.assertEqual(receipt["hard_h_p_budget_influence_count"], 0)
        self.assertEqual(receipt["functional_veto_count"], 0)

    def test_nonpositive_layers_are_exactly_zero_but_positive_domain_survives(self) -> None:
        slopes = (0.5, -0.2, 0.3, 0.0, 0.1)
        result = solve_full_drive_routing(
            _problem(slopes=slopes),
            _inventory(),
            edit_slopes=slopes,
            arm=FullDriveArm.SOFT,
            lambda_p=0.1,
        )
        self.assertEqual(result.active_direction_mask, (True, False, True, False, True))
        self.assertEqual(result.velocity[1], 0.0)
        self.assertEqual(result.velocity[3], 0.0)

    def test_pmax_zero_is_target_only_clock_totality(self) -> None:
        slopes = (0.0, -0.1, -0.2, -0.3, -0.4)
        for arm in FullDriveArm:
            with self.subTest(arm=arm.value):
                result = solve_full_drive_routing(
                    _problem(slopes=slopes),
                    _inventory(),
                    edit_slopes=slopes,
                    arm=arm,
                    lambda_p=1.0,
                )
                self.assertEqual(
                    result.mode, FullDriveStepMode.TARGET_ONLY_PMAX_ZERO
                )
                self.assertEqual(result.p_max, 0.0)
                self.assertEqual(result.velocity, (0.0,) * 5)
                self.assertEqual(result.r_edit, 0.0)
                transition = propose_full_drive_transition(
                    SimpleNamespace(layers=()),
                    result,
                    torch.zeros(3, 10),
                    torch.ones(3, 10),
                    step_index=0,
                )
                self.assertFalse(transition.weight_advanced)
                self.assertTrue(transition.target_advanced)
                self.assertEqual(transition.increment, {})
                self.assertEqual(
                    transition.raw_free_payload()["tau_advance"],
                    float(FIXED_E8_H),
                )

    def test_pmax_zero_without_target_descent_fails_closed(self) -> None:
        result = solve_full_drive_routing(
            _problem(slopes=(0.0, -0.1, -0.2, -0.3, -0.4)),
            _inventory(),
            edit_slopes=(0.0, -0.1, -0.2, -0.3, -0.4),
            arm=FullDriveArm.NO_SOFT,
            lambda_p=0.0,
        )
        with self.assertRaisesRegex(Exception, "TARGET_NO_DESCENT"):
            propose_full_drive_transition(
                SimpleNamespace(layers=()),
                result,
                torch.zeros(3, 10),
                torch.zeros(3, 10),
                step_index=0,
            )

    def test_lambda_grid_and_numerical_ties_are_predeclared(self) -> None:
        self.assertEqual(FULL_DRIVE_LAMBDA_GRID[0], 0.0)
        self.assertEqual(FULL_DRIVE_LAMBDA_GRID[-1], 100.0)
        self.assertEqual(len(FULL_DRIVE_LAMBDA_GRID), 14)
        contract = full_drive_numerical_contract()
        self.assertEqual(contract["progress_floor"], 0.0)
        self.assertEqual(contract["kappa"], 0.0)
        self.assertIsNone(contract["requested_progress"])
        self.assertNotEqual(
            contract["epsilon_j"], contract["solver_kkt_tolerance"]
        )

    def test_runtime_cell_registry_is_dynamic_two_by_two_only(self) -> None:
        self.assertEqual(
            FULL_DRIVE_CELL_ORDER,
            (
                FullDriveCell.RS_FULL_NOSOFT,
                FullDriveCell.RS_FULL_SOFT,
                FullDriveCell.BG_FULL_NOSOFT,
                FullDriveCell.BG_FULL_SOFT,
            ),
        )
        self.assertIs(
            FullDriveCell.RS_FULL_SOFT.allocation,
            FullDriveAllocation.ROBUST_SHARED,
        )
        self.assertIs(
            FullDriveCell.BG_FULL_NOSOFT.routing_arm,
            FullDriveArm.NO_SOFT,
        )
        receipt = full_drive_runtime_contract()
        self.assertEqual(receipt["factorial_shape"], [2, 2, 1])
        self.assertEqual(receipt["target_dynamics"], "DYNAMIC_TARGET")
        self.assertEqual(receipt["target_hold_access_count"], 0)

    def test_physical_slope_adapter_cancels_builder_h_exactly_once(self) -> None:
        slopes = (0.25, -0.125, 0.0, 0.0625, 0.5)
        payload = {
            "schema": "ode-edit-s05-p1r17-physical-w-only-edit-slope/v1",
            "field_sha256": "f" * 64,
            "request_order_sha256": "r" * 64,
            "context_sha256": "c" * 64,
            "layer_order": list(FIXED_E8_LAYER_ORDER),
            "baseline_loss": 2.0,
            "layer_endpoint_loss": [2.0 - item for item in slopes],
            "edit_slopes": list(slopes),
            "edit_slope_definition": "Phi_edit_W(W_k)-Phi_edit_W(W_k+h*B_k_l)",
            "endpoint_factor_sha256": [str(i) * 64 for i in range(1, 6)],
            "model_forward_count": 360,
            "processed_token_count": 1_000,
            "backward_count": 0,
            "routing_objective": "target_new_nll",
            "residual_overlay_access_count": 0,
            "heldout_access_count": 0,
            "target_old_access_count": 0,
            "h_application_count": 1,
            "same_bf16_authoritative_basis_for_measurement_and_write": True,
        }
        receipt = full_drive_physical_field.PhysicalEditSlopeReceipt(
            "f" * 64,
            "r" * 64,
            "c" * 64,
            2.0,
            tuple(2.0 - item for item in slopes),
            slopes,
            tuple(str(i) * 64 for i in range(1, 6)),
            360,
            1_000,
            0,
            full_drive_physical_field.canonical_hash(payload),
        )
        adapted = full_drive_physical_field.physical_slope_problem_adapter(
            receipt
        )
        np.testing.assert_array_equal(
            FIXED_E8_H * np.asarray(adapted.signed_progress),
            np.asarray(slopes),
        )
        self.assertEqual(
            adapted.excluded_nonpositive_layers, (5, 6)
        )

    def test_lambda_replay_uses_smallest_stable_interior_eligible_point(self) -> None:
        source = LambdaReplaySource(
            "llama3-8b-inst",
            "RS",
            "NEUTRAL",
            "DYNAMIC_TARGET",
            0,
            "a" * 40,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
            "f" * 64,
        )
        results = []
        for index, value in enumerate(FULL_DRIVE_LAMBDA_GRID):
            velocity = (
                0.500 - index * 0.001,
                0.300 + index * 0.001,
                0.150,
                0.050,
                0.0,
            )
            results.append(
                SimpleNamespace(
                    mode=FullDriveStepMode.JOINT_WRITE,
                    velocity=velocity,
                    r_edit=0.97,
                    r_ph=(1.0 if index == 0 else 0.97 if index == 1 else 0.90),
                    distance_to_nominal=(0.0 if index == 0 else index * 1.0e-4),
                    lambda_p=value,
                    nominal_r_ph=1.0,
                    identity_sha256=f"{index + 1:064x}",
                )
            )
        with mock.patch.object(
            full_drive_soft_routing,
            "solve_full_drive_routing",
            side_effect=results,
        ):
            lock = replay_and_select_lambda(
                _problem(),
                _inventory(),
                edit_slopes=(0.5, 0.4, 0.3, 0.2, 0.1),
                source=source,
            )
        self.assertEqual(lock.selected_lambda, FULL_DRIVE_LAMBDA_GRID[2])
        self.assertFalse(lock.points[1].eligible)
        self.assertIn(
            "RISK_IMPROVEMENT_INSUFFICIENT",
            lock.points[1].rejection_reasons,
        )
        self.assertTrue(lock.points[2].eligible)
        self.assertEqual(
            lock.raw_free_payload()["selection_rule"]["selection"],
            "SMALLEST_POSITIVE_ELIGIBLE_LAMBDA",
        )

    def test_ast_forbids_kappa_floor_hard_barrier_and_retry_control(self) -> None:
        source = inspect.getsource(solve_full_drive_routing)
        tree = ast.parse(source)
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        forbidden = {
            "FIXED_E8_KAPPA",
            "requested_progress_floor",
            "retry",
            "backtracking",
            "historical_budget",
            "pretrained_budget",
        }
        self.assertFalse((identifiers | attributes) & forbidden)

    def test_physical_endpoint_measurement_uses_six_nohook_endpoints_and_one_h(self) -> None:
        layers = tuple(
            SimpleNamespace(layer=layer, weight_name=f"w{layer}")
            for layer in FIXED_E8_LAYER_ORDER
        )
        field = SimpleNamespace(layers=layers, identity_sha256="f" * 64)
        values = iter((2.0, 1.8, 1.7, 1.6, 1.5, 1.4))

        def observed() -> SimpleNamespace:
            value = next(values)
            return SimpleNamespace(
                value=torch.tensor(value, dtype=torch.float64),
                request_order_sha256="r" * 64,
                context_sha256="c" * 64,
                context_group_sizes=(1, 5),
                context_count=6,
                backward_count=0,
                target_true_suffix_token_counts=(None,) * 10,
                model_forward_count=60,
                processed_token_count=600,
            )

        def factors(_field, velocity, *, step_index):
            self.assertEqual(step_index, 0)
            self.assertEqual(sum(velocity), 1.0)
            ordinal = velocity.index(1.0)
            layer = FIXED_E8_LAYER_ORDER[ordinal]
            return {
                f"w{layer}": SimpleNamespace(
                    layer=layer,
                    theta=float(FIXED_E8_H),
                    residual=torch.ones(1),
                    q=torch.ones(1),
                )
            }

        history = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(digest="history")
        )
        schedule = SimpleNamespace(state_digest="sampler")
        ledger = ComputeLedger()
        with (
            mock.patch.object(
                full_drive_physical_field,
                "fixed_e8_waypoint_factors",
                side_effect=factors,
            ),
            mock.patch.object(
                full_drive_physical_field,
                "_merge_factors",
                side_effect=lambda current, increment: {**current, **increment},
            ),
            mock.patch.object(
                full_drive_physical_field,
                "_virtual_context",
                side_effect=lambda model, candidate: nullcontext(),
            ),
            mock.patch.object(
                full_drive_physical_field,
                "evaluate_routing_objective",
                side_effect=lambda *args, **kwargs: observed(),
            ),
        ):
            receipt = full_drive_physical_field.measure_physical_edit_slopes(
                torch.nn.Linear(1, 1),
                SimpleNamespace(),
                tuple({"request_sha256": str(i)} for i in range(10)),
                field,
                step_index=0,
                cumulative_factors_by_weight={},
                contexts=(("x",),) * 10,
                ledger=ledger,
                touched={},
                history=history,
                schedule=schedule,
                rng_identity=lambda: "rng",
            )
        np.testing.assert_allclose(
            receipt.edit_slopes,
            (0.2, 0.3, 0.4, 0.5, 0.6),
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertEqual(receipt.model_forward_count, 360)
        self.assertEqual(receipt.backward_count, 0)
        self.assertEqual(receipt.raw_free_payload()["h_application_count"], 1)
        self.assertEqual(
            receipt.raw_free_payload()["residual_overlay_access_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
