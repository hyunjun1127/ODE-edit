from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.common_cold_coordinate import (
    CommonColdScale,
    CommonColdScaleMetric,
)
from project.run_scripts.ode_bf.integrated_physical_writer import (
    PHYSICAL_WRITER_EPSILON_E,
    PHYSICAL_WRITER_EPSILON_P,
    PHYSICAL_WRITER_H,
    PhysicalWriterComputePhase,
    PhysicalWriterForwardRole,
    PhysicalWriterGroupLedger,
    PhysicalWriterStepStatus,
    batched_two_output_coefficient_vjp,
    integrated_physical_writer_source_contract,
    physical_theta_from_velocity,
    physical_writer_compute_contract,
    solve_integrated_physical_writer,
)
from project.run_scripts.ode_bf.integrated_physical_writer_runtime import (
    IntegratedFactorAccumulator,
    PhysicalThetaOverlay,
    accumulate_functional_p_vjp,
    accumulate_writer_vjp_microbatches,
    build_integrated_physical_field_from_keys,
    build_integrated_technical_problem,
    capture_all_writer_keys_single_forward,
    commit_verify_restore_integrated_endpoint,
    functional_p_vjp_microbatch,
    shared_dynamic_target_factor_group,
    writer_edit_transport_vjp_microbatch,
)
from project.run_scripts.ode_bf.p1_replay import samplewise_teacher_kl
from project.run_scripts.ode_bf import integrated_physical_writer_selection as r14_selection
from project.run_scripts.ode_bf.p1_backend import CovarianceActionReceipt
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.tests.test_target_new_nll import (
    _Tokenizer,
    _contexts,
    _requests,
)


class _RuntimeTokenizer(_Tokenizer):
    def encode(self, text: str) -> list[int]:
        return self.ids(text)


class _RuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.embedding = torch.nn.Embedding(1024, 8)
        self.writers = torch.nn.ModuleDict(
            {
                str(layer): torch.nn.Linear(8, 8, bias=False)
                for layer in range(4, 9)
            }
        )
        for module in self.writers.values():
            with torch.no_grad():
                module.weight.copy_(torch.eye(8))
        self.target = torch.nn.Identity()
        self.head = torch.nn.Linear(8, 1024, bias=False)
        self.config = SimpleNamespace(
            _name_or_path="qwen-runtime-fixture", hidden_size=8
        )
        self.forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for layer in range(4, 9):
            hidden = torch.tanh(self.writers[str(layer)](hidden))
        hidden = self.target(hidden)
        return SimpleNamespace(logits=self.head(hidden))


class _CovarianceFixture:
    def __init__(self, wall_offset: float = 0.0) -> None:
        self.wall_offset = wall_offset

    def action(
        self, layer: int, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, CovarianceActionReceipt]:
        action = q.detach().to(device="cpu", dtype=torch.float32).contiguous()
        gram = action.T.to(dtype=torch.float64) @ action.to(dtype=torch.float64)
        from project.run_scripts.ode_bf.functional import tensor_sha256

        return action, gram, CovarianceActionReceipt(
            layer,
            f"{layer:064x}",
            1024,
            (q.shape[0], q.shape[0]),
            32,
            tuple(q.shape),
            tensor_sha256(q),
            tensor_sha256(action),
            tensor_sha256(gram),
            True,
            float(layer) + self.wall_offset,
        )


def _runtime_field() -> SimpleNamespace:
    torch.manual_seed(23)
    layers = tuple(
        SimpleNamespace(
            layer=layer,
            weight_name=f"writers.{layer}.weight",
            q=torch.randn((8, 10), dtype=torch.float32) * 0.03,
            residual=torch.randn((8, 10), dtype=torch.float32) * 0.02,
        )
        for layer in range(4, 9)
    )
    return SimpleNamespace(
        layers=layers,
        identity_sha256="f" * 64,
    )


def _barrier(label: str, dimension: int = 5) -> QuadraticBarrier:
    return QuadraticBarrier(
        label,
        0.0,
        np.zeros(dimension, dtype=np.float64),
        np.zeros((dimension, dimension), dtype=np.float64),
        1.0,
        "layer-local-diagonal",
    )


def _problem(
    *,
    capacity: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.4),
    caps: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    trust_radius: float = 1.0,
) -> RoutingProblem:
    dimension = 5
    return RoutingProblem(
        np.ones(dimension, dtype=np.float64),
        np.diag(np.asarray(capacity, dtype=np.float64)),
        np.diag(np.full(dimension, 0.01, dtype=np.float64)),
        trust_radius,
        np.asarray(caps, dtype=np.float64),
        0.01,
        1.0e-8,
        _barrier("historical"),
        _barrier("pretrained"),
    )


class IntegratedPhysicalWriterTests(unittest.TestCase):
    def test_compute_contract_exact_a2_arithmetic(self) -> None:
        low = physical_writer_compute_contract(fixed_online_forward_groups=0)
        self.assertEqual(low.online_forward_groups_per_step, 13)
        self.assertEqual(low.integration_online_forward_groups, 104)
        self.assertEqual(low.writer_vjp_logical_groups_per_step, 1)
        self.assertEqual(low.writer_vjp_microbatch_graphs_per_step, 6)
        self.assertEqual(low.writer_vjp_autograd_invocations_per_step, 6)
        self.assertEqual(low.p_vjp_logical_groups_per_step, 1)
        self.assertEqual(low.p_vjp_microbatch_graphs_per_step, 6)
        self.assertEqual(low.p_vjp_autograd_invocations_per_step, 6)
        self.assertEqual(low.total_online_forward_groups, 104)
        self.assertLess(
            low.total_online_forward_groups,
            low.online_forward_group_ceiling,
        )
        with self.assertRaisesRegex(ODEBFContractError, "not zero"):
            physical_writer_compute_contract(fixed_online_forward_groups=1)
        with self.assertRaisesRegex(ODEBFContractError, "not zero"):
            physical_writer_compute_contract(fixed_online_forward_groups=6)
        with self.assertRaises(ODEBFContractError):
            physical_writer_compute_contract(fixed_online_forward_groups=True)

    def test_append_only_phase_ledger_exact_eight_by_thirteen(self) -> None:
        ledger = PhysicalWriterGroupLedger()
        ledger.record(
            group_id="bootstrap-0",
            phase=PhysicalWriterComputePhase.BOOTSTRAP,
            role=PhysicalWriterForwardRole.BOOTSTRAP,
            step_index=None,
            microbatch_ordinal=None,
            model_forward_calls=1,
            physical_microbatch_graphs=1,
            autograd_backend_invocations=1,
            backward_calls=1,
            processed_tokens=10,
        )
        for step in range(8):
            ledger.record(
                group_id=f"target-{step}",
                phase=PhysicalWriterComputePhase.PRODUCTION,
                role=PhysicalWriterForwardRole.TARGET_FACTOR,
                step_index=step,
                microbatch_ordinal=None,
                model_forward_calls=1,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=1,
                backward_calls=1,
                processed_tokens=60,
            )
            for role, token in (
                (PhysicalWriterForwardRole.WRITER_VJP, "writer"),
                (PhysicalWriterForwardRole.P_VJP, "p"),
            ):
                for microbatch in range(6):
                    ledger.record(
                        group_id=f"{token}-{step}-{microbatch}",
                        phase=PhysicalWriterComputePhase.PRODUCTION,
                        role=role,
                        step_index=step,
                        microbatch_ordinal=microbatch,
                        model_forward_calls=1,
                        physical_microbatch_graphs=1,
                        autograd_backend_invocations=1,
                        backward_calls=1,
                        processed_tokens=10,
                    )
        receipt = ledger.validate_production(fixed_online_forward_groups=0)
        self.assertEqual(receipt["production_forward_group_count"], 104)
        self.assertEqual(
            receipt["writer_vjp_autograd_backend_invocation_count"], 48
        )
        self.assertEqual(receipt["p_vjp_autograd_backend_invocation_count"], 48)
        with self.assertRaisesRegex(ODEBFContractError, "identity differs"):
            ledger.record(
                group_id="target-0",
                phase=PhysicalWriterComputePhase.PRODUCTION,
                role=PhysicalWriterForwardRole.FIXED,
                step_index=None,
                microbatch_ordinal=None,
                model_forward_calls=1,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=0,
                backward_calls=0,
                processed_tokens=0,
            )
        with self.assertRaisesRegex(ODEBFContractError, "pre-freeze"):
            ledger.record(
                group_id="terminal-before-freeze",
                phase=PhysicalWriterComputePhase.TERMINAL,
                role=PhysicalWriterForwardRole.TERMINAL,
                step_index=None,
                microbatch_ordinal=None,
                model_forward_calls=1,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=0,
                backward_calls=0,
                processed_tokens=1,
            )
        ledger.freeze_actions("a" * 64)
        terminal = ledger.record(
            group_id="terminal-after-freeze",
            phase=PhysicalWriterComputePhase.TERMINAL,
            role=PhysicalWriterForwardRole.TERMINAL,
            step_index=None,
            microbatch_ordinal=None,
            model_forward_calls=1,
            physical_microbatch_graphs=1,
            autograd_backend_invocations=0,
            backward_calls=0,
            processed_tokens=1,
        )
        self.assertEqual(terminal.phase, PhysicalWriterComputePhase.TERMINAL)

    def test_batched_two_output_vjp_matches_independent_reference(self) -> None:
        theta = torch.tensor(
            [0.2, -0.1, 0.4, 0.3, -0.2],
            dtype=torch.float32,
            requires_grad=True,
        )
        edit_weights = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5])
        transport_weights = torch.tensor([-2.0, 0.5, 1.5, 4.0, -1.0])
        loss_new = torch.sum((theta * edit_weights - 0.7) ** 2)
        loss_transport = 0.5 * torch.sum(
            (theta * transport_weights + 0.2) ** 2
        )
        reference_edit = -torch.autograd.grad(
            loss_new, theta, retain_graph=True
        )[0]
        reference_transport = -torch.autograd.grad(
            loss_transport, theta, retain_graph=True
        )[0]
        observed = batched_two_output_coefficient_vjp(
            loss_new, loss_transport, theta
        )
        np.testing.assert_allclose(observed.a_edit, reference_edit.numpy())
        np.testing.assert_allclose(
            observed.a_transport, reference_transport.numpy()
        )
        self.assertEqual(observed.h_in_vjp_graph_count, 0)
        self.assertEqual(observed.autograd_backend_invocation_count, 1)

    def test_theta_coordinate_finite_difference_and_single_h(self) -> None:
        theta0 = np.zeros(5, dtype=np.float64)
        direction = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])
        target = np.asarray([0.5, -0.1, 0.3, 0.7, 0.9])

        def loss(theta: np.ndarray) -> float:
            return float(0.5 * np.sum((theta - target) ** 2))

        epsilon = 1.0e-3
        gradient = np.asarray(
            [
                (
                    loss(theta0 + epsilon * np.eye(5)[index])
                    - loss(theta0 - epsilon * np.eye(5)[index])
                )
                / (2.0 * epsilon)
                for index in range(5)
            ]
        )
        applied = physical_theta_from_velocity(direction)
        np.testing.assert_array_equal(applied, PHYSICAL_WRITER_H * direction)
        predicted_reduction = PHYSICAL_WRITER_H * (-gradient) @ direction
        unscaled_linear_reduction = (-gradient) @ applied
        self.assertAlmostEqual(predicted_reduction, unscaled_linear_reduction)
        self.assertNotAlmostEqual(
            predicted_reduction,
            PHYSICAL_WRITER_H * ((-gradient) @ applied),
        )

    def test_a2_totality_branch_table(self) -> None:
        common = {
            "problem": _problem(),
            "step_index": 0,
            "functional_p_derivative": np.zeros(5),
            "structural_p_matrix_raw": np.zeros((5, 5)),
        }
        transport_only = solve_integrated_physical_writer(
            **common,
            a_edit=np.zeros(5),
            a_transport=(0.0, 0.0, 0.0, 1.0, 0.0),
            current_nohook_loss=1.0,
            goal_loss=1.0,
        )
        self.assertEqual(
            transport_only.status,
            PhysicalWriterStepStatus.EDIT_DESCENT_INACTIVE_GOAL_MET,
        )
        self.assertGreater(transport_only.velocity[3], 0.0)
        self.assertEqual(transport_only.clock_advance_count, 1)
        self.assertEqual(
            transport_only.certificates[1].authority_role,
            "AUTHORITATIVE_TRANSPORT_MAXIMUM",
        )
        self.assertAlmostEqual(
            transport_only.certificates[1].signed_progress,
            transport_only.p_transport,
        )

        both_inactive = solve_integrated_physical_writer(
            **common,
            a_edit=np.zeros(5),
            a_transport=np.zeros(5),
            current_nohook_loss=1.0,
            goal_loss=1.0,
        )
        self.assertEqual(
            both_inactive.status,
            PhysicalWriterStepStatus.ZERO_WRITE_GOAL_MET,
        )
        self.assertEqual(both_inactive.velocity, (0.0,) * 5)
        self.assertEqual(both_inactive.clock_advance_count, 1)
        self.assertEqual(both_inactive.candidate_count, 0)

        stopped = solve_integrated_physical_writer(
            **common,
            a_edit=np.zeros(5),
            a_transport=np.ones(5),
            current_nohook_loss=2.0,
            goal_loss=1.0,
        )
        self.assertEqual(
            stopped.status,
            PhysicalWriterStepStatus.NO_PHYSICAL_W_ONLY_DIRECTION,
        )
        self.assertEqual(stopped.clock_advance_count, 0)
        self.assertEqual(stopped.candidate_count, 0)

    def test_all_five_domain_keeps_negative_edit_transport_layer(self) -> None:
        observed = solve_integrated_physical_writer(
            _problem(),
            a_edit=(1.0, 0.5, 0.2, -0.1, 0.0),
            a_transport=(0.0, 0.0, 0.0, 2.0, 0.0),
            current_nohook_loss=2.0,
            goal_loss=1.0,
            step_index=0,
            functional_p_derivative=np.zeros(5),
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertEqual(observed.status, PhysicalWriterStepStatus.JOINT_WRITE)
        self.assertTrue(observed.all_five_layer_domain)
        self.assertLess(observed.a_edit[3], 0.0)
        self.assertGreater(observed.a_transport[3], 0.0)
        self.assertGreater(observed.velocity[3], 0.0)
        self.assertGreaterEqual(
            PHYSICAL_WRITER_H
            * np.dot(observed.a_edit, observed.velocity)
            + 1.0e-10,
            observed.requested_applied_reduction,
        )

    def test_e_then_p_then_capacity_and_step_scaled_p(self) -> None:
        result = solve_integrated_physical_writer(
            _problem(capacity=(10.0, 1.0, 2.0, 3.0, 4.0)),
            a_edit=(1.0, 0.0, 0.0, 0.0, 0.0),
            a_transport=np.zeros(5),
            current_nohook_loss=1.0,
            goal_loss=1.0,
            step_index=0,
            functional_p_derivative=(2.0, -3.0, 0.0, 0.0, 0.0),
            structural_p_matrix_raw=np.eye(5),
        )
        self.assertEqual(result.status, PhysicalWriterStepStatus.JOINT_WRITE)
        self.assertAlmostEqual(
            result.p_proxy_derivative_step[0], PHYSICAL_WRITER_H * 2.0
        )
        self.assertEqual(result.p_proxy_derivative_step[1], 0.0)
        np.testing.assert_allclose(
            result.p_structural_matrix_step,
            PHYSICAL_WRITER_H**2 * np.eye(5),
        )
        self.assertGreater(result.p_scale, 0.0)
        self.assertEqual(
            [certificate.phase for certificate in result.certificates[-3:]],
            [
                "stage1-minimum-dimensionless-physical-transport-E",
                "stage2-minimum-dimensionless-P-tie",
                "stage3-minimum-capacity-within-E-P-ties",
            ],
        )
        self.assertEqual(PHYSICAL_WRITER_EPSILON_E, 1.0e-8)
        self.assertEqual(PHYSICAL_WRITER_EPSILON_P, 1.0e-8)

    def test_zero_p_scale_and_overlay_absence(self) -> None:
        result = solve_integrated_physical_writer(
            _problem(),
            a_edit=np.ones(5),
            a_transport=np.zeros(5),
            current_nohook_loss=1.0,
            goal_loss=1.0,
            step_index=7,
            functional_p_derivative=np.zeros(5),
            structural_p_matrix_raw=np.zeros((5, 5)),
        )
        self.assertEqual(result.p_scale, 0.0)
        self.assertEqual(result.p_tie_status, "P_TIE_INACTIVE")
        self.assertEqual(result.overlay_access_count, 0)
        source = inspect.getsource(solve_integrated_physical_writer)
        self.assertNotIn("flatnonzero", source)
        self.assertNotIn("positive_direction_mask", source)
        contract = integrated_physical_writer_source_contract()
        self.assertEqual(
            contract["overlay_guard_objective_termination_access_count"], 0
        )
        self.assertTrue(contract["all_five_layer_domain"])

    def test_malformed_inputs_fail_closed(self) -> None:
        kwargs = {
            "problem": _problem(),
            "a_edit": np.ones(5),
            "a_transport": np.ones(5),
            "current_nohook_loss": 2.0,
            "goal_loss": 1.0,
            "step_index": 0,
            "functional_p_derivative": np.ones(5),
            "structural_p_matrix_raw": np.eye(5),
        }
        for name, value in (
            ("a_edit", np.ones(4)),
            ("a_transport", [1, 1, 1, 1, float("nan")]),
            ("functional_p_derivative", np.ones(6)),
            ("structural_p_matrix_raw", np.ones((5, 4))),
        ):
            broken = dict(kwargs)
            broken[name] = value
            with self.subTest(name=name), self.assertRaises(
                ODEBFContractError
            ):
                solve_integrated_physical_writer(**broken)

    def test_single_forward_multihook_keys_match_five_call_reference(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing from timm.models.hub is deprecated.*",
                category=FutureWarning,
            )
            from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

        model = _RuntimeModel()
        tokenizer = _RuntimeTokenizer()
        requests = _requests()
        contexts = _contexts()
        observed, receipt = capture_all_writer_keys_single_forward(
            model,
            tokenizer,
            requests,
            contexts,
            rewrite_module_template="writers.{}",
            fact_token_strategy="subject_last",
            cumulative_factors_by_weight={},
        )
        self.assertEqual(receipt.logical_forward_groups, 1)
        self.assertEqual(receipt.model_forward_calls, 1)
        self.assertEqual(model.forward_calls, 1)
        hparams = SimpleNamespace(
            rewrite_module_tmp="writers.{}",
            fact_token="subject_last",
        )
        reference = {
            layer: alpha_main.compute_ks(
                model,
                tokenizer,
                requests,
                hparams,
                layer,
                [list(group) for group in contexts],
            ).T.to(device="cpu", dtype=torch.float32)
            for layer in range(4, 9)
        }
        self.assertEqual(model.forward_calls, 6)
        for layer in range(4, 9):
            self.assertTrue(torch.equal(observed[layer], reference[layer]))

    def test_six_m10_writer_vjps_accumulate_and_keep_transport_separate(self) -> None:
        model = _RuntimeModel()
        tokenizer = _RuntimeTokenizer()
        requests = _requests()
        contexts = _contexts()
        field = _runtime_field()
        desired = torch.randn((8, 10), dtype=torch.float32)
        receipts = tuple(
            writer_edit_transport_vjp_microbatch(
                model,
                tokenizer,
                requests,
                contexts,
                context_ordinal=context,
                field=field,
                cumulative_factors_by_weight={},
                target_layer_name="target",
                lookup_positions=(0,) * 60,
                desired_target=desired,
                metric_norm_squared=(1.0,) * 10,
            )
            for context in range(6)
        )
        aggregate = accumulate_writer_vjp_microbatches(receipts)
        self.assertEqual(aggregate.physical_microbatch_graph_count, 6)
        self.assertEqual(aggregate.autograd_backend_invocation_count, 6)
        self.assertEqual(aggregate.backward_call_count, 6)
        self.assertEqual(aggregate.model_forward_call_count, 6)
        self.assertGreater(aggregate.loss_new, 0.0)
        self.assertGreater(aggregate.loss_transport, 0.0)
        self.assertTrue(any(abs(value) > 0.0 for value in aggregate.a_edit))
        self.assertTrue(
            any(abs(value) > 0.0 for value in aggregate.a_transport)
        )
        self.assertEqual(receipts[0].loss_transport_contribution, aggregate.loss_transport)
        self.assertTrue(
            all(item.loss_transport_contribution == 0.0 for item in receipts[1:])
        )
        self.assertEqual(model.forward_calls, 6)

    def test_functional_p_vjp_six_groups_and_positive_derivative_input(self) -> None:
        model = _RuntimeModel()
        tokenizer = _RuntimeTokenizer()
        field = _runtime_field()
        groups = []
        teachers = []
        baselines = []
        for microbatch in range(6):
            requests = _requests()
            for ordinal, request in enumerate(requests):
                request["request_sha256"] = f"{microbatch * 10 + ordinal + 1:064x}"
            prompts = [
                str(item["prompt"]).format(str(item["subject"]))
                for item in requests
            ]
            encoded = tokenizer(
                prompts, padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**encoded).logits
                attention = encoded["attention_mask"]
                positions = torch.tensor(
                    [
                        int(torch.nonzero(row, as_tuple=False).flatten()[-1])
                        for row in attention
                    ],
                    dtype=torch.long,
                )
                rows = torch.arange(10)
                teacher_logits = logits[rows, positions, :].float().clone()
                teacher_logits[:, 1] += 0.4
                teacher = torch.log_softmax(teacher_logits, dim=-1)
            groups.append(requests)
            teachers.append(teacher)
            baselines.append(torch.zeros(10, dtype=torch.float64))
        before = model.forward_calls
        receipts = tuple(
            functional_p_vjp_microbatch(
                model,
                tokenizer,
                groups[index],
                microbatch_ordinal=index,
                field=field,
                cumulative_factors_by_weight={},
                teacher_log_probs=teachers[index],
                entry_kl=baselines[index],
            )
            for index in range(6)
        )
        derivative, aggregate = accumulate_functional_p_vjp(receipts)
        self.assertEqual(model.forward_calls - before, 6)
        self.assertEqual(aggregate["physical_microbatch_graph_count"], 6)
        self.assertEqual(aggregate["autograd_backend_invocation_count"], 6)
        self.assertEqual(len(derivative), 5)
        self.assertTrue(all(np.isfinite(derivative)))

    def test_functional_p_m10_matches_full60_and_theta_finite_difference(self) -> None:
        model = _RuntimeModel()
        tokenizer = _RuntimeTokenizer()
        field = _runtime_field()
        groups: list[list[dict[str, object]]] = []
        teachers: list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        for microbatch in range(6):
            requests = _requests()
            for ordinal, request in enumerate(requests):
                request["request_sha256"] = (
                    f"{microbatch * 10 + ordinal + 101:064x}"
                )
            prompts = [
                str(item["prompt"]).format(str(item["subject"]))
                for item in requests
            ]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**encoded).logits
                attention = encoded["attention_mask"]
                positions = torch.tensor(
                    [
                        int(torch.nonzero(row, as_tuple=False).flatten()[-1])
                        for row in attention
                    ],
                    dtype=torch.long,
                )
                rows = torch.arange(10)
                teacher_logits = logits[rows, positions, :].float().clone()
                teacher_logits[:, 1 + microbatch] += 0.7
                teacher = torch.log_softmax(teacher_logits, dim=-1)
            groups.append(requests)
            teachers.append(teacher)
            baselines.append(torch.zeros(10, dtype=torch.float64))

        receipts = tuple(
            functional_p_vjp_microbatch(
                model,
                tokenizer,
                groups[index],
                microbatch_ordinal=index,
                field=field,
                cumulative_factors_by_weight={},
                teacher_log_probs=teachers[index],
                entry_kl=baselines[index],
            )
            for index in range(6)
        )
        derivative, _ = accumulate_functional_p_vjp(receipts)
        all_requests = [item for group in groups for item in group]
        prompts = [
            str(item["prompt"]).format(str(item["subject"]))
            for item in all_requests
        ]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        attention = encoded["attention_mask"]
        positions = torch.tensor(
            [
                int(torch.nonzero(row, as_tuple=False).flatten()[-1])
                for row in attention
            ],
            dtype=torch.long,
        )
        rows = torch.arange(60)
        teacher_full = torch.cat(teachers, dim=0)
        baseline_full = torch.cat(baselines, dim=0)

        def full_damage(theta: torch.Tensor) -> torch.Tensor:
            with PhysicalThetaOverlay(model, field, theta):
                logits = model(**encoded).logits
            observed = torch.log_softmax(
                logits[rows, positions, :].float(), dim=-1
            )
            return torch.clamp(
                samplewise_teacher_kl(
                    teacher_full.to(dtype=torch.float64), observed
                )
                - baseline_full,
                min=0.0,
            ).mean()

        theta = torch.zeros(5, dtype=torch.float32, requires_grad=True)
        full_value = full_damage(theta)
        full_gradient = torch.autograd.grad(full_value, theta)[0]
        np.testing.assert_allclose(
            np.asarray(derivative),
            full_gradient.detach().to(dtype=torch.float64).numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        self.assertAlmostEqual(
            sum(item.damage_value for item in receipts),
            float(full_value.detach()),
            places=7,
        )
        epsilon = 1.0e-3
        finite: list[float] = []
        for index in range(5):
            observed = []
            for sign in (-1.0, 1.0):
                probe = torch.zeros(5, dtype=torch.float32)
                probe[index] = sign * epsilon
                probe.requires_grad_(True)
                with torch.no_grad():
                    observed.append(float(full_damage(probe)))
            finite.append((observed[1] - observed[0]) / (2.0 * epsilon))
        np.testing.assert_allclose(
            np.asarray(derivative),
            np.asarray(finite),
            rtol=1.0e-2,
            atol=2.0e-4,
        )

    def test_shared_target_factor_group_has_one_identity_and_key_parity(self) -> None:
        model = _RuntimeModel()
        tokenizer = _RuntimeTokenizer()
        requests = _requests()
        contexts = _contexts()
        target = torch.randn((8, 10), generator=torch.Generator().manual_seed(81))
        order = ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in requests]
        )
        metric = CommonColdScaleMetric.from_z_base(
            target, order, CommonColdScale.BATCH_GLOBAL
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing from timm.models.hub is deprecated.*",
                category=FutureWarning,
            )
            (
                keys,
                current,
                residual,
                desired,
                receipt,
            ) = shared_dynamic_target_factor_group(
                model,
                tokenizer,
                requests,
                contexts,
                target_state=target,
                target_layer_name="target",
                lookup_positions=(-2,) * 60,
                rewrite_module_template="writers.{}",
                fact_token_strategy="subject_last",
                cumulative_factors_by_weight={},
                scale_metric=metric,
            )
            reference, _ = capture_all_writer_keys_single_forward(
                model,
                tokenizer,
                requests,
                contexts,
                rewrite_module_template="writers.{}",
                fact_token_strategy="subject_last",
                cumulative_factors_by_weight={},
            )
        self.assertEqual(receipt.logical_forward_groups, 1)
        self.assertEqual(receipt.model_forward_calls, 120)
        self.assertEqual(receipt.backward_calls, 10)
        self.assertTrue(torch.equal(residual, target - current))
        self.assertFalse(torch.equal(desired, target))
        self.assertTrue(all(torch.equal(keys[layer], reference[layer]) for layer in keys))

    def test_precomputed_key_field_and_raw_structural_step_scaling(self) -> None:
        model = _RuntimeModel()
        keys = {
            layer: torch.randn(
                (8, 10), generator=torch.Generator().manual_seed(layer)
            )
            for layer in range(4, 9)
        }
        target = torch.randn((8, 10), generator=torch.Generator().manual_seed(90))
        current = target - 0.1
        hparams = SimpleNamespace(
            layers=(4, 5, 6, 7, 8),
            rewrite_module_tmp="writers.{}",
            L2=1.0,
        )
        order = ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in _requests()]
        )
        projector = torch.eye(8, dtype=torch.float32).repeat(5, 1, 1)
        first, first_receipt = build_integrated_physical_field_from_keys(
            model,
            hparams=hparams,
            projector=projector,
            precomputed_keys=keys,
            target_state=target,
            terminal_current_z=current,
            request_order_sha256=order,
            accepted_waypoint=0,
            covariance_registry=_CovarianceFixture(0.0),
            projector_sha256="0" * 64,
            residual_tolerance=1.0e-8,
        )
        second, second_receipt = build_integrated_physical_field_from_keys(
            model,
            hparams=hparams,
            projector=projector,
            precomputed_keys=keys,
            target_state=target,
            terminal_current_z=current,
            request_order_sha256=order,
            accepted_waypoint=0,
            covariance_registry=_CovarianceFixture(99.0),
            projector_sha256="0" * 64,
            residual_tolerance=1.0e-8,
        )
        self.assertEqual(first.identity_sha256, second.identity_sha256)
        self.assertEqual(
            first_receipt.field_semantic_sha256,
            second_receipt.field_semantic_sha256,
        )
        self.assertNotEqual(
            first_receipt.covariance_cost_receipt_sha256_by_layer,
            second_receipt.covariance_cost_receipt_sha256_by_layer,
        )
        self.assertEqual(
            len({layer.residual.detach().numpy().tobytes() for layer in first.layers}),
            1,
        )
        problem = build_integrated_technical_problem(
            first,
            a_edit=(1.0, -1.0, 0.5, 0.0, 2.0),
            committed_load_by_layer={layer: 0.0 for layer in range(4, 9)},
            controller_lock=P1ControllerLock(),
        )
        raw = np.asarray(problem.structural_p_matrix_raw)
        self.assertEqual(raw.shape, (5, 5))
        self.assertTrue(np.array_equal(raw, np.diag(np.diag(raw))))
        np.testing.assert_allclose(
            problem.factor_energy_step,
            (PHYSICAL_WRITER_H**2) * np.asarray(problem.factor_energy_raw),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(problem.hard_h_p_budget_influence_count, 0)
        self.assertEqual(
            problem.committed_load_semantics,
            "PERSISTENT_OUTER_HISTORY_ONLY",
        )
        self.assertEqual(
            problem.current_virtual_factor_load_in_committed_load_count,
            0,
        )

    def test_k8_virtual_accumulation_one_endpoint_transaction_and_restore(self) -> None:
        model = _RuntimeModel()
        for module in model.writers.values():
            module.weight.data = module.weight.data.to(dtype=torch.bfloat16)
        keys = {
            layer: torch.randn(
                (8, 10), generator=torch.Generator().manual_seed(100 + layer)
            )
            for layer in range(4, 9)
        }
        target = torch.randn((8, 10), generator=torch.Generator().manual_seed(190))
        current = target - 0.05
        field, _ = build_integrated_physical_field_from_keys(
            model,
            hparams=SimpleNamespace(
                layers=(4, 5, 6, 7, 8),
                rewrite_module_tmp="writers.{}",
                L2=1.0,
            ),
            projector=torch.eye(8, dtype=torch.float32).repeat(5, 1, 1),
            precomputed_keys=keys,
            target_state=target,
            terminal_current_z=current,
            request_order_sha256=ordered_request_digest_v1(
                [str(item["request_sha256"]) for item in _requests()]
            ),
            accepted_waypoint=0,
            covariance_registry=_CovarianceFixture(),
            projector_sha256="0" * 64,
            residual_tolerance=1.0e-8,
        )
        accumulator = IntegratedFactorAccumulator()
        for step in range(8):
            receipt = accumulator.append(
                field,
                physical_theta_from_velocity(
                    np.full(5, 0.1 + 0.01 * step, dtype=np.float64)
                ),
                step_index=step,
            )
            self.assertEqual(receipt.h_application_count, 1)
            self.assertEqual(receipt.parameter_mutation_count, 0)
        accumulator.assert_complete_k8()
        factors = accumulator.factors_by_weight()
        self.assertTrue(all(len(items) == 8 for items in factors.values()))
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name in factors
        }
        before = {
            name: parameter.detach().clone() for name, parameter in parameters.items()
        }
        pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
        verify_calls = 0

        def verify() -> bool:
            nonlocal verify_calls
            verify_calls += 1
            return any(
                not torch.equal(parameters[name].detach(), before[name])
                for name in parameters
            )

        receipt = commit_verify_restore_integrated_endpoint(
            parameters,
            factors,
            accumulated_step_count=8,
            row_block=3,
            transaction_id="r14-toy-endpoint",
            mutation_lock=threading.RLock(),
            post_commit_verify=verify,
        )
        self.assertEqual(verify_calls, 1)
        self.assertEqual(receipt.transaction_commit_count, 1)
        self.assertEqual(receipt.postverify_count, 1)
        self.assertEqual(receipt.explicit_restore_count, 1)
        self.assertEqual(receipt.persistent_commit_count, 0)
        self.assertTrue(receipt.final_w0_restore_exact)
        self.assertTrue(receipt.pointer_restore_exact)
        self.assertTrue(
            all(torch.equal(parameters[name], before[name]) for name in parameters)
        )
        self.assertEqual(
            pointers,
            {name: parameter.data_ptr() for name, parameter in parameters.items()},
        )

    def test_endpoint_transaction_faults_restore_every_write_position(self) -> None:
        for fault in range(5):
            with self.subTest(fault=fault):
                model = _RuntimeModel()
                for module in model.writers.values():
                    module.weight.data = module.weight.data.to(dtype=torch.bfloat16)
                field = _runtime_field()
                accumulator = IntegratedFactorAccumulator()
                accumulator.append(
                    field,
                    physical_theta_from_velocity(np.full(5, 0.1)),
                    step_index=0,
                )
                factors = accumulator.factors_by_weight()
                parameters = {
                    name: parameter
                    for name, parameter in model.named_parameters()
                    if name in factors
                }
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in parameters.items()
                }
                pointers = {
                    name: parameter.data_ptr() for name, parameter in parameters.items()
                }
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    commit_verify_restore_integrated_endpoint(
                        parameters,
                        factors,
                        accumulated_step_count=1,
                        row_block=4,
                        transaction_id=f"r14-fault-{fault}",
                        mutation_lock=threading.RLock(),
                        post_commit_verify=lambda: True,
                        fault_after_writes=fault,
                    )
                self.assertTrue(
                    all(
                        torch.equal(parameters[name], before[name])
                        and parameters[name].data_ptr() == pointers[name]
                        for name in parameters
                    )
                )

    def test_model_free_fresh_seal_excludes_history_and_duplicate_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            dataset = root / "counterfact.json"
            rows = []
            for index in range(100):
                duplicate = index in (4, 5)
                rows.append(
                    {
                        "case_id": index,
                        "requested_rewrite": {
                            "prompt": "{} duplicate" if duplicate else f"{{}} prompt {index}",
                            "relation_id": f"relation-{index}",
                            "subject": "shared-subject" if duplicate else f"subject-{index}",
                            "target_new": {"str": f"new-{index}"},
                            "target_true": {"str": f"old-{index}"},
                        },
                        "paraphrase_prompts": [f"heldout-{index}"],
                    }
                )
            dataset.write_text(json.dumps(rows), encoding="utf-8")
            identities = r14_selection.load_r14_identities(dataset)
            (repo / "prior-seal.json").write_text(
                json.dumps(
                    {
                        "case_ids": [0],
                        "request_sha256": identities[1].request_sha256,
                    }
                ),
                encoding="utf-8",
            )
            (repo / "selection.py").write_text(
                "EXPLICIT_CASE_EXCLUSIONS = {2}\n", encoding="utf-8"
            )
            (repo / "audit.md").write_text("case_id: 3\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "fixture"], cwd=repo, check=True
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                text=True, stdout=subprocess.PIPE,
            ).stdout.strip()
            with mock.patch.object(r14_selection, "R14_SELECTION_BASE", head):
                exclusion = r14_selection.build_r14_historical_exclusion(
                    repo, base_commit=head
                )
                self.assertTrue({0, 2, 3}.issubset(exclusion["prior_case_ids"]))
                self.assertIn(
                    identities[1].request_sha256,
                    exclusion["prior_request_sha256"],
                )
                seal = r14_selection.build_r14_fresh_seal(dataset, exclusion)
                selected = {item["case_id"] for item in seal["requests"]}
                self.assertTrue(selected.isdisjoint({0, 1, 2, 3, 4, 5}))
                self.assertEqual(len(seal["functional_p_anchors"]), 60)
                self.assertFalse(
                    {item["request_sha256"] for item in seal["requests"]}
                    & {
                        item["request_sha256"]
                        for item in seal["functional_p_anchors"]
                    }
                )
                controller = r14_selection.load_r14_controller_requests(
                    dataset, seal
                )
                anchors = r14_selection.load_r14_controller_requests(
                    dataset, seal, population="functional_p_anchors"
                )
                self.assertEqual(len(controller), 10)
                self.assertEqual(len(anchors), 60)
                self.assertTrue(
                    all(
                        set(item)
                        == {
                            "case_id", "request_sha256", "prompt", "subject",
                            "target_new",
                        }
                        for item in (*controller, *anchors)
                    )
                )
                self.assertEqual(seal["prior_group_intersection_count"], 0)
                self.assertEqual(seal["duplicate_group_intersection_count"], 0)
                # A later tracked self-reference cannot change the immutable-base scan.
                (repo / "self-reference.json").write_text(
                    json.dumps({"case_ids": sorted(selected)}), encoding="utf-8"
                )
                subprocess.run(["git", "add", "."], cwd=repo, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "-m", "later"], cwd=repo, check=True
                )
                rebuilt = r14_selection.build_r14_historical_exclusion(
                    repo, base_commit=head
                )
                self.assertEqual(rebuilt["root_digest"], exclusion["root_digest"])
            source = inspect.getsource(r14_selection)
            for forbidden in ("import torch", "p1_runtime", "p1_evaluator"):
                self.assertNotIn(forbidden, source)

    def test_identity_bearing_malformed_json_fails_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "broken-seal.json").write_text(
                '{"case_ids":[1,', encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "fixture"], cwd=repo, check=True
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                text=True, stdout=subprocess.PIPE,
            ).stdout.strip()
            with mock.patch.object(r14_selection, "R14_SELECTION_BASE", head):
                with self.assertRaisesRegex(
                    ODEBFContractError, "identity-bearing JSON"
                ):
                    r14_selection.build_r14_historical_exclusion(
                        repo, base_commit=head
                    )

    def test_partial_hook_registration_failure_cleans_every_hook(self) -> None:
        model = _RuntimeModel()
        field = _runtime_field()
        field.layers[2].weight_name = "target.weight"
        theta = torch.zeros(5, requires_grad=True)
        before = {
            name: len(module._forward_hooks)
            for name, module in model.named_modules()
        }
        from project.run_scripts.ode_bf.integrated_physical_writer_runtime import (
            PhysicalThetaOverlay,
        )

        with self.assertRaisesRegex(ODEBFContractError, "not exact Linear"):
            PhysicalThetaOverlay(model, field, theta).__enter__()
        after = {
            name: len(module._forward_hooks)
            for name, module in model.named_modules()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
