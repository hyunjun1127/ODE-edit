from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.coupled_demand_soft_transport import (
    CoupledStepStatus,
    EditFloorComponent,
    P1R16_EPS_ALPHA,
    P1R16_EPS_DUAL,
    P1R16_EPS_LOSS,
    P1R16_EPS_P,
    P1R16_EPS_SPEED,
    P1R16_EPS_TRANSPORT_SCALE,
    P1R16_EPS_TRANSPORT_TIE,
    TargetStepStatus,
    build_per_request_gdual_target_step,
    p1r16_source_contract,
    solve_coupled_demand_soft_transport,
)
from project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment import (
    P1R16_STAGE_A_R13_REQUEST_ORDER,
    run_p1r16_stage_a_virtual_rollout,
)
from project.run_scripts.ode_bf.coupled_demand_soft_transport_runtime import (
    aggregate_context_batched_keys,
    build_coupled_physical_demand,
    padded_subject_positions,
    perrequest_dynamic_target_factor_group,
    retain_requestwise_batched_gradients,
)
from project.run_scripts.ode_bf.integrated_physical_writer_runtime import (
    _TerminalLookupCapture,
    build_integrated_physical_field_from_keys,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.tests.test_integrated_physical_writer import (
    _CovarianceFixture,
    _RuntimeModel,
    _runtime_field,
)
from project.run_scripts.ode_bf.tests.test_target_new_nll import _requests


def _barrier(label: str) -> QuadraticBarrier:
    return QuadraticBarrier(
        label,
        0.0,
        np.zeros(5, dtype=np.float64),
        np.zeros((5, 5), dtype=np.float64),
        1.0,
        "layer-local-diagonal",
    )


def _problem(*, trust_radius: float = 10.0) -> RoutingProblem:
    return RoutingProblem(
        np.ones(5, dtype=np.float64),
        np.diag(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])),
        np.eye(5, dtype=np.float64),
        trust_radius,
        np.ones(5, dtype=np.float64),
        1.0e-8,
        1.0e-8,
        _barrier("historical"),
        _barrier("pretrained"),
    )


def _solve(**overrides: object):
    values: dict[str, object] = {
        "a_edit": (1.0, 1.0, 0.0, 0.0, 0.0),
        "a_transport": (1.0, -1.0, 0.0, 0.0, 0.0),
        "current_nohook_loss": 1.0,
        "controller_requested_reduction": 0.0625,
        "transport_loss_current": 0.1,
        "step_index": 0,
        "functional_p_derivative": (0.0,) * 5,
        "structural_p_matrix_raw": np.zeros((5, 5)),
        "cold_entry": False,
        "target_equals_z_base": False,
        "old_residual_exact_zero": False,
    }
    values.update(overrides)
    return solve_coupled_demand_soft_transport(_problem(), **values)


class PerRequestTargetTests(unittest.TestCase):
    def test_gdual_step_has_one_h_and_requested_mean(self) -> None:
        target = torch.zeros((3, 10), dtype=torch.float32)
        z_base = torch.zeros_like(target)
        z_base[0] = torch.arange(1, 11, dtype=torch.float32)
        gradient = torch.ones_like(target, dtype=torch.float64)
        receipt = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=torch.full((10,), 0.25, dtype=torch.float64),
            gradient_by_request=gradient,
            step_index=0,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(receipt.status, TargetStepStatus.TARGET_STEP)
        self.assertTrue(
            all(abs(value - 1.0) < 1.0e-10 for value in receipt.unit_g_norm_by_request)
        )
        self.assertEqual(receipt.h_application_count, 1)
        self.assertAlmostEqual(
            receipt.controller_requested_reduction,
            sum(receipt.predicted_reduction_by_request) / 10.0,
            places=14,
        )

    def test_target_totality_and_nonfinite(self) -> None:
        target = torch.zeros((2, 10), dtype=torch.float32)
        z_base = torch.ones_like(target)
        gradient = torch.zeros_like(target, dtype=torch.float64)
        goal = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=(P1R16_EPS_LOSS,) * 10,
            gradient_by_request=gradient,
            step_index=2,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(goal.status, TargetStepStatus.TARGET_GOAL_MET)
        blocked = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=(P1R16_EPS_LOSS * 2.0,) * 10,
            gradient_by_request=gradient,
            step_index=2,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(blocked.status, TargetStepStatus.TARGET_NO_DESCENT_DIRECTION)
        malformed = gradient.clone()
        malformed[0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            build_per_request_gdual_target_step(
                target_state=target,
                z_base=z_base,
                loss_by_request=(1.0,) * 10,
                gradient_by_request=malformed,
                step_index=0,
                cumulative_g_path_before=(0.0,) * 10,
            )

    def test_source_contract_is_r16_a1(self) -> None:
        contract = p1r16_source_contract()
        self.assertEqual(contract["eps_dual"], P1R16_EPS_DUAL)
        self.assertEqual(contract["eps_alpha"], P1R16_EPS_ALPHA)
        self.assertEqual(contract["eps_speed"], P1R16_EPS_SPEED)
        self.assertEqual(contract["eps_transport_scale"], P1R16_EPS_TRANSPORT_SCALE)
        self.assertEqual(contract["eps_transport_tie"], P1R16_EPS_TRANSPORT_TIE)
        self.assertEqual(contract["eps_P"], P1R16_EPS_P)
        self.assertEqual(contract["model_forwards_per_full_field"], 18)
        self.assertEqual(contract["backwards_per_full_field"], 22)
        self.assertEqual(contract["remaining_step_division_count_in_writer_floor"], 0)


class CoordinateAndDemandTests(unittest.TestCase):
    def test_qwen_nine_of_ten_padding_regression(self) -> None:
        offsets = (3, 1, 2, 4, 1, 4, 2, 0, 5, 1)
        raw = (1,) * 10
        ids = torch.arange(100, dtype=torch.int64).view(10, 10)
        padded, hashes = padded_subject_positions(raw, offsets, ids)
        self.assertEqual(padded, tuple(1 + item for item in offsets))
        self.assertEqual(sum(item > 0 for item in offsets), 9)
        self.assertEqual(len(hashes), 10)
        with self.assertRaises(ODEBFContractError):
            padded_subject_positions((-1,) + raw[1:], offsets, ids)
        singleton = tuple(tuple(int(item) for item in row) for row in ids)
        corrupted = list(singleton)
        corrupted[0] = tuple([corrupted[0][0], 999, *corrupted[0][2:]])
        with self.assertRaises(ODEBFContractError):
            padded_subject_positions(raw, (0,) * 10, ids, tuple(corrupted))

    def test_coupled_demand_includes_old_z_and_one_division(self) -> None:
        current = torch.tensor([[1.0] * 10, [2.0] * 10])
        target = current + 0.25
        desired = target + 0.125
        receipt = build_coupled_physical_demand(
            current_z=current,
            target_state=target,
            desired_target=desired,
        )
        expected = (desired - current) / 0.125
        self.assertTrue(torch.equal(receipt.factor_demand, expected))
        self.assertFalse(torch.equal(receipt.factor_demand, (target - current)))
        self.assertEqual(receipt.division_by_h_count, 1)
        self.assertEqual(receipt.applied_h_count, 0)
        self.assertEqual(receipt.old_z_omission_count, 0)

    def test_generic_field_has_explicit_precomputed_residual_path(self) -> None:
        source = inspect.getsource(build_integrated_physical_field_from_keys)
        self.assertIn("precomputed_residual", source)
        self.assertIn("residual = precomputed_residual.detach()", source)

    def test_coupled_demand_is_the_actual_all_layer_factor_left(self) -> None:
        model = _RuntimeModel()
        keys = {
            layer: torch.randn((8, 10), generator=torch.Generator().manual_seed(layer))
            for layer in range(4, 9)
        }
        current = torch.zeros((8, 10), dtype=torch.float32)
        desired = torch.ones_like(current)
        demand = torch.full_like(current, 8.0)
        field, receipt = build_integrated_physical_field_from_keys(
            model,
            hparams=SimpleNamespace(
                layers=(4, 5, 6, 7, 8),
                rewrite_module_tmp="writers.{}",
                L2=1.0,
            ),
            projector=torch.eye(8, dtype=torch.float32).repeat(5, 1, 1),
            precomputed_keys=keys,
            target_state=desired,
            terminal_current_z=current,
            request_order_sha256=ordered_request_digest_v1(
                [str(item["request_sha256"]) for item in _requests()]
            ),
            accepted_waypoint=0,
            covariance_registry=_CovarianceFixture(),
            projector_sha256="0" * 64,
            residual_tolerance=P1ControllerLock().residual_tolerance,
            precomputed_residual=demand,
        )
        self.assertEqual(
            len({layer.residual.numpy().tobytes() for layer in field.layers}), 1
        )
        self.assertTrue(all(torch.equal(layer.residual, demand) for layer in field.layers))
        from project.run_scripts.ode_bf.functional import tensor_sha256

        self.assertEqual(receipt.residual_sha256, tensor_sha256(demand))

    def test_writer_transport_lookup_applies_left_padding(self) -> None:
        model = _RuntimeModel()
        activation = torch.arange(10 * 8 * 8, dtype=torch.float32).view(10, 8, 8)
        padding = (3, 1, 2, 4, 1, 4, 2, 0, 5, 1)
        with _TerminalLookupCapture(
            model, "target", (1,) * 10, left_padding=padding
        ) as capture:
            model.target(activation)
        expected = torch.stack(
            [activation[row, 1 + pad, :] for row, pad in enumerate(padding)]
        )
        self.assertTrue(torch.equal(capture.value, expected))

    def test_no_model_batched_gradient_and_factor_reference_parity(self) -> None:
        flat = torch.arange(30, dtype=torch.float64).requires_grad_(True)
        target = flat.view(3, 10)
        losses = tuple(
            torch.sum((target[:, row] * float(row + 1)) ** 2)
            for row in range(10)
        )
        observed, _, off = retain_requestwise_batched_gradients(
            losses, flat, target.shape
        )
        reference = torch.stack(
            [2.0 * target.detach()[:, row] * float(row + 1) ** 2 for row in range(10)],
            dim=1,
        )
        self.assertTrue(torch.equal(observed, reference))
        self.assertEqual(off, (0,) * 10)
        values = tuple(
            torch.arange(40, dtype=torch.float32).view(10, 4) + context
            for context in range(6)
        )
        observed_key = aggregate_context_batched_keys(values)
        reference_key = torch.stack(
            (values[0], torch.stack(values[1:]).mean(dim=0))
        ).mean(dim=0).T.contiguous()
        self.assertTrue(torch.equal(observed_key, reference_key))


class SoftTransportSolverTests(unittest.TestCase):
    def test_writer_floor_consumes_request_once_without_remaining_division(self) -> None:
        observed: list[dict[str, object]] = []
        result = solve_coupled_demand_soft_transport(
            _problem(),
            a_edit=(2.0, 0.0, 0.0, 0.0, 0.0),
            a_transport=(0.0,) * 5,
            current_nohook_loss=1.0,
            controller_requested_reduction=0.125,
            transport_loss_current=0.0,
            step_index=7,
            functional_p_derivative=(0.0,) * 5,
            structural_p_matrix_raw=np.zeros((5, 5)),
            cold_entry=False,
            target_equals_z_base=False,
            old_residual_exact_zero=False,
            pre_guard_observer=lambda value: observed.append(dict(value)),
        )
        self.assertAlmostEqual(result.delta_edit, 0.125)
        self.assertEqual(observed[0]["remaining_step_division_count"], 0)
        self.assertGreaterEqual(0.125 * np.dot(result.a_edit, result.velocity), 0.125 - 1e-8)

    def test_zero_scale_transport_is_inactive(self) -> None:
        result = _solve(transport_loss_current=P1R16_EPS_TRANSPORT_SCALE)
        self.assertFalse(result.transport_active)
        self.assertEqual(result.transport_bar_selected, 0.0)
        self.assertFalse(result.transport_allocation_influence)

    def test_positive_negative_and_overshoot_transport_are_symmetric(self) -> None:
        positive = _solve(a_transport=(1.0, -1.0, 0.0, 0.0, 0.0))
        adverse = _solve(a_transport=(-1.0, -1.0, 0.0, 0.0, 0.0))
        overshoot = _solve(a_transport=(10.0, 0.0, 0.0, 0.0, 0.0))
        for result in (positive, adverse, overshoot):
            predicted = result.transport_loss_current - 0.125 * np.dot(
                result.a_transport, result.velocity
            )
            expected = (predicted / result.transport_loss_current) ** 2
            self.assertAlmostEqual(result.transport_predicted_remaining, predicted)
            self.assertAlmostEqual(result.transport_bar_selected, expected)
            self.assertIs(result.status, CoupledStepStatus.JOINT_WRITE)
            self.assertEqual(
                result.transport_guard_veto_termination_influence_count, 0
            )
        self.assertLess(abs(overshoot.transport_predicted_remaining or 0.0), 0.1)

    def test_transport_changes_alpha_only_within_speed_tie(self) -> None:
        result = _solve(a_transport=(1.0, -1.0, 0.0, 0.0, 0.0))
        self.assertTrue(result.transport_allocation_influence)
        self.assertNotEqual(result.stage1_velocity, result.stage2_velocity)
        self.assertLessEqual(
            abs((result.stage2_speed or 0.0) - (result.stage1_speed or 0.0)),
            P1R16_EPS_SPEED + 1.0e-8,
        )
        self.assertGreaterEqual(result.transport_tie_slack or 0.0, -1.0e-8)

    def test_a1_transport_receipt_is_complete_and_has_no_stale_formula(self) -> None:
        result = _solve(a_transport=(1.0, -1.0, 0.0, 0.0, 0.0))
        payload = result.raw_free_payload()
        self.assertEqual(payload["transport_loss_current"], 0.1)
        self.assertIsInstance(payload["transport_predicted_remaining"], float)
        self.assertIsInstance(payload["transport_bar_selected"], float)
        self.assertIsInstance(payload["transport_bar_star"], float)
        self.assertIsInstance(payload["transport_tie_slack"], float)
        self.assertTrue(payload["transport_active"])
        source = inspect.getsource(solve_coupled_demand_soft_transport)
        for stale in (
            "T_" + "scale=",
            "T_bar=" + "-",
            "sum_abs_" + "transport_slope_caps",
            "linear_" + "transport_normalization",
        ):
            self.assertNotIn(stale, source)

    def test_adverse_transport_cannot_block_semantic_write(self) -> None:
        result = _solve(a_transport=(-10.0,) * 5)
        self.assertIs(result.status, CoupledStepStatus.JOINT_WRITE)
        self.assertGreater(result.speed, 0.0)
        self.assertEqual(result.failure_component, EditFloorComponent.NONE)

    def test_cold_recovery_and_invalid_use_boundaries(self) -> None:
        eligible = _solve(
            a_edit=(-1.0,) * 5,
            cold_entry=True,
            target_equals_z_base=True,
            old_residual_exact_zero=True,
        )
        self.assertIs(eligible.status, CoupledStepStatus.COLD_TARGET_ONLY_RECOVERY)
        self.assertEqual(eligible.target_clock_advance_count, 1)
        self.assertEqual(eligible.weight_clock_advance_count, 0)
        self.assertEqual(eligible.candidate_count, 0)
        for override in (
            {"cold_entry": False},
            {"target_equals_z_base": False},
            {"old_residual_exact_zero": False},
            {"step_index": 1},
        ):
            values = {
                "a_edit": (-1.0,) * 5,
                "cold_entry": True,
                "target_equals_z_base": True,
                "old_residual_exact_zero": True,
            }
            values.update(override)
            blocked = _solve(**values)
            self.assertIs(blocked.status, CoupledStepStatus.EDIT_NO_DIRECTION)
            self.assertEqual(blocked.target_clock_advance_count, 0)

    def test_source_has_no_gamma_or_hard_transport_floor(self) -> None:
        source = inspect.getsource(solve_coupled_demand_soft_transport)
        self.assertNotIn("solve_joint_gamma", source)
        self.assertIn('"gamma_access_count": 0', source)
        self.assertNotIn("physical_transport_floor", source)
        self.assertNotIn("/ remaining", source)
        self.assertIn("p1r16-stage2-soft-transport", source)


class TargetBatchingSourceTests(unittest.TestCase):
    def test_target_group_is_six_batched_graphs_ten_backwards(self) -> None:
        source = inspect.getsource(perrequest_dynamic_target_factor_group)
        self.assertIn("for context_ordinal", source)
        self.assertIn('"model_forward_calls": 6', source)
        self.assertIn('"backward_calls": 10', source)
        self.assertIn("overlay.prepare", source)
        self.assertNotIn("evaluate_common_cold_objective", source)
        self.assertNotIn("for request_ordinal", source)


class StageARolloutSystemTests(unittest.TestCase):
    def test_mocked_k8_has_exact_clock_and_compute_contract(self) -> None:
        requests = tuple({"request_sha256": f"{index + 1:064x}"} for index in range(10))
        z_base = torch.ones((8, 10), dtype=torch.float32)
        field = _runtime_field()
        target_call = 0

        def target_group(*_args: object, target_state: torch.Tensor, **_kwargs: object):
            nonlocal target_call
            desired = (target_state + 0.01).contiguous()
            current = (target_state - 0.02).contiguous()
            residual = (target_state - current).contiguous()
            demand = (desired - current) / 0.125
            step = SimpleNamespace(
                status=TargetStepStatus.TARGET_STEP,
                cumulative_g_path_after=(0.125 * (target_call + 1),) * 10,
                raw_free_payload=lambda: {"status": "TARGET_STEP"},
            )
            receipt = SimpleNamespace(
                model_forward_calls=6,
                physical_target_graphs=6,
                autograd_backend_invocations=10,
                backward_calls=10,
                processed_token_count=600,
                target_step=step,
                controller_requested_reduction=0.1,
                raw_free_payload=lambda: {"model_forward_calls": 6},
            )
            demand_receipt = SimpleNamespace(
                raw_free_payload=lambda: {"factor_demand_sha256": "7" * 64}
            )
            target_call += 1
            return (
                {layer: torch.ones((8, 10)) for layer in range(4, 9)},
                current,
                residual,
                desired,
                demand,
                demand_receipt,
                receipt,
            )

        def field_build(*_args: object, precomputed_residual: torch.Tensor, **_kwargs: object):
            from project.run_scripts.ode_bf.functional import tensor_sha256

            receipt = SimpleNamespace(
                residual_sha256=tensor_sha256(precomputed_residual),
                raw_free_payload=lambda: {
                    "residual_sha256": tensor_sha256(precomputed_residual)
                },
            )
            return field, receipt

        writer_micro = tuple(
            SimpleNamespace(
                context_ordinal=index,
                model_forward_calls=1,
                autograd_backend_invocations=1,
                backward_calls=1,
                processed_token_count=10,
            )
            for index in range(6)
        )
        p_micro = tuple(
            SimpleNamespace(
                microbatch_ordinal=index,
                model_forward_calls=1,
                autograd_backend_invocations=1,
                backward_calls=1,
                processed_token_count=10,
            )
            for index in range(6)
        )
        writer = SimpleNamespace(
            a_edit=(1.0,) * 5,
            a_transport=(1.0,) * 5,
            loss_new=1.0,
            loss_transport=0.5,
            raw_free_payload=lambda: {"a_edit": [1.0] * 5},
        )
        technical = SimpleNamespace(
            problem=_problem(),
            structural_p_matrix_raw=np.zeros((5, 5)),
            raw_free_payload=lambda: {"problem": "technical"},
        )
        routing = SimpleNamespace(
            status=CoupledStepStatus.JOINT_WRITE,
            failure_component=EditFloorComponent.NONE,
            velocity=(0.1,) * 5,
            applied_theta=(0.0125,) * 5,
            identity_sha256="9" * 64,
            raw_free_payload=lambda: {"status": "JOINT_WRITE"},
        )

        def solve(*_args: object, pre_guard_observer, **_kwargs: object):
            pre_guard_observer({"all_five_layer_domain": True})
            return routing

        with mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.ordered_request_digest_v1",
            return_value=P1R16_STAGE_A_R13_REQUEST_ORDER,
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.perrequest_dynamic_target_factor_group",
            side_effect=target_group,
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.build_integrated_physical_field_from_keys",
            side_effect=field_build,
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.writer_edit_transport_vjp_microbatch",
            side_effect=lambda *_args, context_ordinal, **_kwargs: writer_micro[context_ordinal],
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.accumulate_writer_vjp_microbatches",
            return_value=writer,
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.functional_p_vjp_microbatch",
            side_effect=lambda *_args, microbatch_ordinal, **_kwargs: p_micro[microbatch_ordinal],
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.accumulate_functional_p_vjp",
            return_value=((0.0,) * 5, {"identity_sha256": "8" * 64}),
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.build_integrated_technical_problem",
            return_value=technical,
        ), mock.patch(
            "project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment.solve_coupled_demand_soft_transport",
            side_effect=solve,
        ):
            result = run_p1r16_stage_a_virtual_rollout(
                torch.nn.Linear(1, 1),
                object(),
                requests=requests,
                contexts=(("{}",), ("{}",) * 5),
                p_anchor_microbatches=tuple(requests for _ in range(6)),
                p_teacher_log_probs=tuple(torch.zeros((10, 2)) for _ in range(6)),
                p_entry_kl=tuple(torch.zeros(10) for _ in range(6)),
                hparams=SimpleNamespace(rewrite_module_tmp="writers.{}", fact_token="subject_last"),
                projector=torch.eye(8).repeat(5, 1, 1),
                covariance_registry=object(),
                projector_sha256="0" * 64,
                controller_lock=object(),
                z_base=z_base,
                lookup_positions=(0,) * 60,
                target_layer_name="target",
                residual_tolerance=1.0e-8,
            )
        self.assertEqual(result.accepted_transition_count, 8)
        self.assertEqual(result.target_clock_count, 8)
        self.assertEqual(result.weight_clock_count, 8)
        self.assertEqual(result.final_tau, 1.0)
        self.assertEqual(result.compute_receipt["production_forward_group_count"], 104)
        self.assertEqual(result.compute_receipt["model_forward_call_count"], 8 * 18)
        self.assertEqual(result.compute_receipt["backward_call_count"], 8 * 22)
        self.assertLess(18, 24)


if __name__ == "__main__":
    unittest.main()
