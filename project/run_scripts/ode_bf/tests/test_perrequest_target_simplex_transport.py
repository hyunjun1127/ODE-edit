from __future__ import annotations

import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.perrequest_target_simplex_transport import (
    JointFloorComponent,
    P1R15_EPS_ALPHA,
    P1R15_EPS_DUAL,
    P1R15_EPS_LOSS,
    P1R15_EPS_P,
    P1R15_EPS_SPEED,
    SimplexStepStatus,
    TargetStepStatus,
    build_per_request_gdual_target_step,
    p1r15_source_contract,
    solve_joint_gamma_feasibility,
    solve_perrequest_simplex_transport,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.p1_perrequest_target_simplex_transport_panel import (
    P1R15_NUMERICAL_LOCK_FILE,
    P1R15_NUMERICAL_LOCK_ROOT,
    expected_p1r15_stage_a_context_sha256,
    load_and_validate_p1r15_numerical_lock,
    load_and_validate_p1r15_stage_a_seal,
    validate_p1r15_source_closure,
)
from project.run_scripts.ode_bf.perrequest_target_simplex_transport_runtime import (
    perrequest_dynamic_target_factor_group,
)
from project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment import (
    P1R15_STAGE_A_R13_REQUEST_ORDER,
    run_p1r15_stage_a_virtual_rollout,
)
from project.run_scripts.ode_bf.tests.test_integrated_physical_writer import (
    _runtime_field,
)


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


class PerRequestTargetTests(unittest.TestCase):
    def test_gdual_step_has_one_h_and_unit_metric(self) -> None:
        target = torch.zeros((3, 10), dtype=torch.float32)
        z_base = torch.zeros_like(target)
        z_base[0] = torch.arange(1, 11, dtype=torch.float32)
        gradient = torch.ones_like(target, dtype=torch.float64)
        loss = torch.full((10,), 0.25, dtype=torch.float64)
        receipt = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=loss,
            gradient_by_request=gradient,
            step_index=0,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(receipt.status, TargetStepStatus.TARGET_STEP)
        self.assertTrue(all(abs(value - 1.0) < 1.0e-10 for value in receipt.unit_g_norm_by_request))
        self.assertTrue(all(0.0 < value <= 0.125 for value in receipt.step_g_path_by_request))
        self.assertEqual(receipt.h_application_count, 1)
        self.assertEqual(receipt.target_clock_advance_count, 0)
        self.assertEqual(receipt.target_clock_advance_eligibility_count, 1)
        # Applied G path is exactly h*rho, never h^2*rho.
        self.assertAlmostEqual(
            receipt.step_g_path_by_request[0],
            0.125 * receipt.rho_by_request[0],
            places=14,
        )

    def test_target_totality_goal_no_descent_and_nonfinite(self) -> None:
        target = torch.zeros((2, 10), dtype=torch.float32)
        z_base = torch.ones_like(target)
        gradient = torch.zeros_like(target, dtype=torch.float64)
        goal = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=(P1R15_EPS_LOSS,) * 10,
            gradient_by_request=gradient,
            step_index=2,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(goal.status, TargetStepStatus.TARGET_GOAL_MET)
        self.assertEqual(goal.target_clock_advance_count, 0)
        self.assertEqual(goal.target_clock_advance_eligibility_count, 1)
        blocked = build_per_request_gdual_target_step(
            target_state=target,
            z_base=z_base,
            loss_by_request=(P1R15_EPS_LOSS * 2.0,) * 10,
            gradient_by_request=gradient,
            step_index=2,
            cumulative_g_path_before=(0.0,) * 10,
        )
        self.assertIs(
            blocked.status, TargetStepStatus.TARGET_NO_DESCENT_DIRECTION
        )
        self.assertEqual(blocked.target_clock_advance_count, 0)
        self.assertEqual(blocked.target_clock_advance_eligibility_count, 0)
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

    def test_target_base_metric_and_cumulative_path_fail_closed(self) -> None:
        target = torch.zeros((2, 10), dtype=torch.float32)
        gradient = torch.ones_like(target, dtype=torch.float64)
        with self.assertRaises(ODEBFContractError):
            build_per_request_gdual_target_step(
                target_state=target,
                z_base=torch.zeros_like(target),
                loss_by_request=(1.0,) * 10,
                gradient_by_request=gradient,
                step_index=0,
                cumulative_g_path_before=(0.0,) * 10,
            )
        with self.assertRaises(ODEBFContractError):
            build_per_request_gdual_target_step(
                target_state=target,
                z_base=torch.ones_like(target),
                loss_by_request=(1.0,) * 10,
                gradient_by_request=gradient,
                step_index=7,
                cumulative_g_path_before=(1.0,) * 10,
            )

    def test_target_constants_are_exact(self) -> None:
        contract = p1r15_source_contract()
        self.assertEqual(contract["eps_loss"], P1R15_EPS_LOSS)
        self.assertEqual(contract["eps_dual"], P1R15_EPS_DUAL)
        self.assertEqual(contract["eps_alpha"], P1R15_EPS_ALPHA)
        self.assertEqual(contract["eps_speed"], P1R15_EPS_SPEED)
        self.assertEqual(contract["eps_P"], P1R15_EPS_P)
        self.assertEqual(contract["online_logical_groups_k8"], 104)
        self.assertEqual(contract["model_forwards_per_full_field"], 132)
        self.assertEqual(contract["backwards_per_full_field"], 22)
        self.assertEqual(contract["stage_b_material_count"], 0)


class SimplexTransportTests(unittest.TestCase):
    def test_joint_gamma_feasible_and_infeasible(self) -> None:
        loose = _problem(trust_radius=10.0)
        feasible = solve_joint_gamma_feasibility(
            loose,
            a_edit=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
            a_transport=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
            delta_edit=0.0625,
            delta_transport=0.0625,
        )
        self.assertTrue(feasible.passed)
        self.assertTrue(feasible.feasible_full_floors)
        self.assertGreaterEqual(feasible.gamma, 1.0 - 1.0e-8)
        tight = _problem(trust_radius=1.0)
        infeasible = solve_joint_gamma_feasibility(
            tight,
            a_edit=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
            a_transport=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
            delta_edit=0.125,
            delta_transport=0.125,
        )
        self.assertTrue(infeasible.passed)
        self.assertFalse(infeasible.feasible_full_floors)
        self.assertLess(infeasible.gamma, 1.0 - 1.0e-8)

    def test_positive_deficit_axis_no_direction_is_not_vacuous(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(),
            a_edit=(1.0, 1.0, 1.0, 1.0, 1.0),
            a_transport=(-1.0, -1.0, -1.0, -1.0, -1.0),
            current_nohook_loss=1.0,
            goal_loss=0.5,
            transport_loss_current=1.0,
            step_index=0,
            functional_p_derivative=(0.0,) * 5,
            structural_p_matrix_raw=np.eye(5),
        )
        self.assertIs(
            result.status,
            SimplexStepStatus.NO_JOINT_PHYSICAL_TRANSPORT_DIRECTION,
        )
        self.assertIs(
            result.failure_component,
            JointFloorComponent.TRANSPORT_NO_DIRECTION,
        )
        self.assertEqual(result.clock_advance_count, 0)

    def test_edit_no_direction_is_distinct_and_observer_has_zero_influence(self) -> None:
        observed: list[dict[str, object]] = []
        kwargs = dict(
            a_edit=(-1.0,) * 5,
            a_transport=(1.0,) * 5,
            current_nohook_loss=1.0,
            goal_loss=0.5,
            transport_loss_current=1.0,
            step_index=0,
            functional_p_derivative=(0.0,) * 5,
            structural_p_matrix_raw=np.eye(5),
        )
        with_observer = solve_perrequest_simplex_transport(
            _problem(),
            **kwargs,
            pre_guard_observer=lambda value: observed.append(dict(value)),
        )
        without_observer = solve_perrequest_simplex_transport(_problem(), **kwargs)
        self.assertEqual(with_observer.identity_sha256, without_observer.identity_sha256)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["overlay_access_count"], 0)
        self.assertIs(
            with_observer.failure_component,
            JointFloorComponent.EDIT_NO_DIRECTION,
        )

    def test_joint_floor_infeasible_has_distinct_component(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(trust_radius=1.0),
            a_edit=(1.0, 0.0, 0.0, 0.0, 0.0),
            a_transport=(0.0, 1.0, 0.0, 0.0, 0.0),
            current_nohook_loss=10.0,
            goal_loss=0.0,
            transport_loss_current=10.0,
            step_index=0,
            functional_p_derivative=(0.0,) * 5,
            structural_p_matrix_raw=np.eye(5),
        )
        self.assertIs(
            result.failure_component,
            JointFloorComponent.JOINT_FLOOR_INFEASIBLE,
        )
        self.assertIsNotNone(result.gamma_certificate)
        self.assertEqual(result.clock_advance_count, 0)

    def test_all_five_domain_can_select_negative_edit_transport_layer(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(),
            a_edit=(3.0, -0.1, 0.0, 0.0, 0.0),
            a_transport=(0.0, 4.0, 0.0, 0.0, 0.0),
            current_nohook_loss=0.9,
            goal_loss=0.8,
            transport_loss_current=0.1,
            step_index=0,
            functional_p_derivative=(0.0, 0.0, 0.0, 0.0, 0.0),
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertIs(result.status, SimplexStepStatus.JOINT_WRITE)
        self.assertGreater(result.velocity[1], 0.0)
        self.assertGreater(result.speed, 0.0)
        self.assertIsNotNone(result.alpha)
        assert result.alpha is not None
        self.assertAlmostEqual(sum(result.alpha), 1.0, places=12)
        self.assertEqual(result.clock_advance_count, 1)
        np.testing.assert_allclose(
            np.asarray(result.velocity),
            result.speed * np.asarray(result.alpha),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(np.all(np.asarray(result.velocity) <= 1.0 + 1.0e-12))

    def test_zero_write_advances_only_when_both_deficits_zero(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(),
            a_edit=(0.0,) * 5,
            a_transport=(0.0,) * 5,
            current_nohook_loss=0.0,
            goal_loss=0.0,
            transport_loss_current=0.0,
            step_index=7,
            functional_p_derivative=(1.0,) * 5,
            structural_p_matrix_raw=np.eye(5),
        )
        self.assertIs(result.status, SimplexStepStatus.ZERO_WRITE_GOAL_MET)
        self.assertEqual(result.velocity, (0.0,) * 5)
        self.assertIsNone(result.alpha)
        self.assertEqual(result.alpha_status, "UNDEFINED_ZERO_SPEED")
        self.assertEqual(result.clock_advance_count, 1)

    def test_P_tie_can_change_degenerate_alpha_but_not_speed(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(),
            a_edit=(1.0, 1.0, 0.0, 0.0, 0.0),
            a_transport=(1.0, 1.0, 0.0, 0.0, 0.0),
            current_nohook_loss=0.1,
            goal_loss=0.0,
            transport_loss_current=0.1,
            step_index=0,
            functional_p_derivative=(10.0, 0.0, 0.0, 0.0, 0.0),
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertTrue(result.p_active)
        self.assertLessEqual(
            abs(result.delta_speed_stage1_to_stage2 or 0.0),
            P1R15_EPS_SPEED + 1.0e-8,
        )
        self.assertIsNotNone(result.delta_alpha_l1_stage1_to_stage2)
        assert result.delta_alpha_l1_stage1_to_stage2 is not None
        self.assertGreater(result.delta_alpha_l1_stage1_to_stage2, 0.5)
        self.assertTrue(result.p_allocation_influence)

    def test_P_has_no_allocation_influence_on_unique_stage1_face(self) -> None:
        problem = _problem()
        problem = RoutingProblem(
            problem.signed_progress,
            problem.capacity_metric,
            problem.trust_metric,
            problem.trust_radius,
            np.asarray([1.0, 1.0e-16, 1.0e-16, 1.0e-16, 1.0e-16]),
            problem.requested_progress,
            problem.minimum_progress,
            problem.historical,
            problem.pretrained,
        )
        result = solve_perrequest_simplex_transport(
            problem,
            a_edit=(2.0, 1.0, 0.0, 0.0, 0.0),
            a_transport=(2.0, 1.0, 0.0, 0.0, 0.0),
            current_nohook_loss=0.1,
            goal_loss=0.0,
            transport_loss_current=0.1,
            step_index=0,
            functional_p_derivative=(10.0, 0.0, 0.0, 0.0, 0.0),
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertTrue(result.p_active)
        self.assertFalse(result.p_allocation_influence)
        assert result.delta_alpha_l1_stage1_to_stage2 is not None
        self.assertLessEqual(
            result.delta_alpha_l1_stage1_to_stage2, P1R15_EPS_ALPHA
        )

    def test_P_zero_scale_is_inactive_and_capacity_remains_authoritative(self) -> None:
        result = solve_perrequest_simplex_transport(
            _problem(),
            a_edit=(2.0, 1.0, 0.0, 0.0, 0.0),
            a_transport=(2.0, 1.0, 0.0, 0.0, 0.0),
            current_nohook_loss=0.1,
            goal_loss=0.0,
            transport_loss_current=0.1,
            step_index=0,
            functional_p_derivative=(0.0,) * 5,
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertIs(result.status, SimplexStepStatus.JOINT_WRITE)
        self.assertFalse(result.p_active)
        self.assertEqual(result.p_scale, 0.0)
        self.assertGreater(result.speed, 0.0)

    def test_target_group_source_retains_request_grads_without_extra_family(self) -> None:
        source = inspect.getsource(perrequest_dynamic_target_factor_group)
        self.assertIn("for request_ordinal", source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn('"model_forward_calls": 120', source)
        self.assertIn('"backward_calls": BATCH_SIZE', source)
        self.assertNotIn("evaluate_routing_objective(", source)
        self.assertNotIn("target_hold", source.lower())


class StageALockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[4]
        cls.locks = cls.repo / "project" / "run_scripts" / "ode_bf" / "locks"

    def test_r13_stage_a_seal_and_numerical_lock(self) -> None:
        seal, seal_sha = load_and_validate_p1r15_stage_a_seal(self.locks)
        self.assertEqual(
            seal["root_digest"],
            "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628",
        )
        self.assertEqual(len(seal_sha), 64)
        self.assertEqual(
            expected_p1r15_stage_a_context_sha256(seal, "llama3-8b-inst"),
            "27b0b1ae80191e4306935d61c52083dca5f4b8adfcd40e388845786196d91dc4",
        )
        self.assertEqual(
            expected_p1r15_stage_a_context_sha256(seal, "qwen2.5-7b-inst"),
            "5e93d637462aeb4056881c61c4e4d73940b7cab2875e8b02482532bb297bccdc",
        )
        with self.assertRaises(ODEBFContractError):
            expected_p1r15_stage_a_context_sha256(seal, "wrong-alias")
        lock, lock_sha = load_and_validate_p1r15_numerical_lock(
            self.locks / P1R15_NUMERICAL_LOCK_FILE
        )
        self.assertEqual(lock["root_digest"], P1R15_NUMERICAL_LOCK_ROOT)
        self.assertEqual(lock["stage_b_material_count"], 0)
        self.assertEqual(lock["stage_b_access_count"], 0)
        self.assertEqual(len(lock_sha), 64)

    def test_source_closure_is_one_arm_and_stage_b_absent(self) -> None:
        receipt = validate_p1r15_source_closure(self.repo)
        self.assertEqual(receipt["one_arm_registry_count"], 1)
        self.assertEqual(receipt["stage_b_material_count"], 0)
        self.assertEqual(receipt["alias_science_branch_count"], 0)


class StageARolloutSystemTests(unittest.TestCase):
    def test_mocked_k8_has_exact_clock_and_compute_contract(self) -> None:
        requests = tuple(
            {"request_sha256": f"{index + 1:064x}"} for index in range(10)
        )
        z_base = torch.ones((8, 10), dtype=torch.float32)
        field = _runtime_field()
        target_call = 0

        def target_group(*_args: object, target_state: torch.Tensor, **_kwargs: object):
            nonlocal target_call
            desired = (target_state + 0.01).contiguous()
            current = (target_state - 0.02).contiguous()
            residual = (target_state - current).contiguous()
            step = SimpleNamespace(
                status=TargetStepStatus.TARGET_STEP,
                cumulative_g_path_after=(0.125 * (target_call + 1),) * 10,
                raw_free_payload=lambda: {
                    "status": TargetStepStatus.TARGET_STEP.value,
                    "step_index": target_call,
                    "target_clock_advance_count": 0,
                    "target_clock_advance_eligibility_count": 1,
                },
            )
            receipt = SimpleNamespace(
                model_forward_calls=120,
                autograd_backend_invocations=10,
                backward_calls=10,
                processed_token_count=600,
                target_step=step,
                goal_target_new_nll=0.5,
                raw_free_payload=lambda: {
                    "target_step": step.raw_free_payload(),
                    "model_forward_calls": 120,
                    "backward_calls": 10,
                },
            )
            target_call += 1
            return {layer: torch.ones((8, 10)) for layer in range(4, 9)}, current, residual, desired, receipt

        def field_build(*_args: object, terminal_current_z: torch.Tensor, target_state: torch.Tensor, **_kwargs: object):
            from project.run_scripts.ode_bf.functional import tensor_sha256

            residual = (target_state - terminal_current_z).contiguous()
            receipt = SimpleNamespace(
                residual_sha256=tensor_sha256(residual),
                raw_free_payload=lambda: {"residual_sha256": tensor_sha256(residual)},
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
            raw_free_payload=lambda: {"a_edit": [1.0] * 5, "a_transport": [1.0] * 5},
        )
        technical = SimpleNamespace(
            problem=_problem(),
            structural_p_matrix_raw=np.zeros((5, 5)),
            raw_free_payload=lambda: {"problem": "technical"},
        )
        routing = SimpleNamespace(
            status=SimplexStepStatus.JOINT_WRITE,
            failure_component=JointFloorComponent.NONE,
            velocity=(0.1,) * 5,
            applied_theta=(0.0125,) * 5,
            identity_sha256="9" * 64,
            raw_free_payload=lambda: {"status": SimplexStepStatus.JOINT_WRITE.value},
        )

        def solve(*_args: object, pre_guard_observer, **_kwargs: object):
            pre_guard_observer(
                {
                    "all_five_layer_domain": True,
                    "overlay_access_count": 0,
                    "positive_layer_mask_access_count": 0,
                }
            )
            return routing
        with mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.ordered_request_digest_v1",
            return_value=P1R15_STAGE_A_R13_REQUEST_ORDER,
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.perrequest_dynamic_target_factor_group",
            side_effect=target_group,
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.build_integrated_physical_field_from_keys",
            side_effect=field_build,
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.writer_edit_transport_vjp_microbatch",
            side_effect=lambda *_args, context_ordinal, **_kwargs: writer_micro[context_ordinal],
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.accumulate_writer_vjp_microbatches",
            return_value=writer,
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.functional_p_vjp_microbatch",
            side_effect=lambda *_args, microbatch_ordinal, **_kwargs: p_micro[microbatch_ordinal],
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.accumulate_functional_p_vjp",
            return_value=((0.0,) * 5, {"identity_sha256": "8" * 64}),
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.build_integrated_technical_problem",
            return_value=technical,
        ), mock.patch(
            "project.run_scripts.ode_bf.perrequest_target_simplex_transport_experiment.solve_perrequest_simplex_transport",
            side_effect=solve,
        ):
            result = run_p1r15_stage_a_virtual_rollout(
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
        self.assertEqual(result.final_tau, 1.0)
        self.assertEqual(result.compute_receipt["production_forward_group_count"], 104)
        self.assertEqual(result.compute_receipt["model_forward_call_count"], 8 * 132)
        self.assertEqual(result.compute_receipt["backward_call_count"], 8 * 22)
        self.assertTrue(
            all(
                row["target_clock_advance_count"] == 1
                and row["weight_clock_advance_count"] == 1
                for row in result.transition_receipts
            )
        )


if __name__ == "__main__":
    unittest.main()
