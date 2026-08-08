from __future__ import annotations

import ast
import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf import (
    common_cold_coordinate as common,
    common_coldcoord_fixed_e8_runtime as runtime,
    fixed_e8_soft_routing as routing,
    p1_backend,
    p1_runtime,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Arm,
    FixedE8Stage2FailurePolicy,
    solve_fixed_e8_routing,
)
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    COMMON_COLD_NUMERICAL_LOCK_FILE,
    COMMON_COLD_PARENT_HEAD,
    COMMON_COLD_RESULT_TOKEN,
    expected_common_cold_result_name,
    forecast_common_cold_panel,
    load_and_validate_common_cold_lock,
    verify_common_cold_case_seal,
    validate_common_cold_runtime_gpu_capacity,
)
from project.run_scripts.ode_bf.tests.test_cold_start_target import (
    _ColdLayerOverlayModel,
    _cold_semantic_field,
)
from project.run_scripts.ode_bf.tests.test_fixed_e8_soft_routing import (
    _inventory,
    _problem,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts import (
    session05_ode_bf_common_coldcoord_fixed_e8_dry_plan as common_dry,
)
from project.run_scripts import (
    session05_ode_bf_submit_common_coldcoord_fixed_e8 as common_submit,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _shared_field(*, wall_seconds: float = 0.25, residual_delta: float = 0.0):
    field, solve, risk = _cold_semantic_field(wall_seconds=wall_seconds)
    shared = field.layers[0].residual.clone()
    if residual_delta:
        shared[0, 0] += residual_delta
    layers = []
    for layer in field.layers:
        factor = replace(layer.factor, left=shared.clone())
        layers.append(
            replace(
                layer,
                residual=shared.clone(),
                residual_definition=(
                    p1_backend.SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
                ),
                residual_divisor=1,
                factor=factor,
            )
        )
    shared_field = replace(
        field,
        current_z=field.target_state - shared,
        layers=tuple(layers),
        identity_sha256=canonical_hash(
            {
                "shared": tensor_sha256(shared),
                "wall_seconds": wall_seconds,
            }
        ),
    )
    return shared_field, solve, risk


class _TerminalOverlayModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Identity()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.target(value)


class CommonColdCoordinateTests(unittest.TestCase):
    def test_scale_formulas_are_common_and_zero_rows_are_recorded(self) -> None:
        z_base = torch.arange(1, 51, dtype=torch.float32).reshape(5, 10)
        order = "a" * 64
        gradient = torch.arange(50, 0, -1, dtype=torch.float64).reshape(5, 10)
        gradient[:, 3].zero_()
        shared = float(torch.median(torch.linalg.vector_norm(z_base.double(), dim=0)))

        rs = common.CommonColdScaleMetric.from_z_base(
            z_base, order, common.CommonColdScale.ROBUST_SHARED
        )
        rs_velocity, rs_receipt = rs.velocity(gradient)
        self.assertEqual(rs.shared_speed, shared)
        self.assertEqual(rs_receipt["zero_gradient_count"], 1)
        self.assertTrue(torch.count_nonzero(rs_velocity[:, 3]) == 0)
        for index in set(range(10)) - {3}:
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(rs_velocity[:, index].double())),
                shared,
                places=5,
            )

        bg = common.CommonColdScaleMetric.from_z_base(
            z_base, order, common.CommonColdScale.BATCH_GLOBAL
        )
        bg_velocity, _ = bg.velocity(gradient)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(bg_velocity.double())),
            shared * np.sqrt(10.0),
            places=4,
        )
        self.assertEqual(rs.raw_free_payload()["native_or_direct_z_access_count"], 0)
        self.assertEqual(bg.raw_free_payload()["clipping_or_rescue_count"], 0)

    def test_request_residual_overlay_is_additive_request_major_and_pure(self) -> None:
        model = _TerminalOverlayModel()
        residual = torch.stack(
            [torch.tensor([float(i + 1), float(i + 2), float(i + 3)]) for i in range(10)],
            dim=1,
        ).contiguous()
        positions = tuple(0 for _ in range(60))
        outputs = []
        pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
        versions = {name: value._version for name, value in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        overlay = common.RequestResidualActivationOverlay(
            model, "target", residual, positions
        )
        with overlay:
            for call in range(60):
                value = torch.zeros((1, 1, 3), dtype=torch.float32)
                outputs.append(model(value)[0, 0].detach().clone())
        receipt = overlay.raw_free_payload()
        for call, output in enumerate(outputs):
            self.assertTrue(torch.equal(output, residual[:, call // 6]))
        self.assertEqual(receipt["hook_call_count"], 60)
        self.assertEqual(receipt["maximum_exact_delta_error"], 0.0)
        self.assertEqual(receipt["absolute_replacement_count"], 0)
        self.assertEqual(
            pointers, {name: value.data_ptr() for name, value in model.named_parameters()}
        )
        self.assertEqual(
            versions, {name: value._version for name, value in model.named_parameters()}
        )
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_bf16_additive_assignment_is_exact_while_realized_delta_rounds(self) -> None:
        model = _TerminalOverlayModel()
        residual = torch.ones((3, 10), dtype=torch.float32)
        positions = tuple(0 for _ in range(60))
        before = torch.full((1, 1, 3), 256.0, dtype=torch.bfloat16)
        expected = before + torch.ones_like(before)
        overlay = common.RequestResidualActivationOverlay(
            model, "target", residual, positions
        )
        outputs = []
        with overlay:
            for _ in range(60):
                outputs.append(model(before.clone()).detach())
        receipt = overlay.raw_free_payload()
        self.assertTrue(all(torch.equal(value, expected) for value in outputs))
        self.assertEqual(receipt["maximum_authoritative_assignment_error"], 0.0)
        self.assertGreater(receipt["maximum_exact_delta_error"], 0.0)
        self.assertEqual(receipt["realized_delta_error_decision_influence_count"], 0)

        assignment_error, realized_error = (
            common._authoritative_additive_assignment_errors(
                before[0, 0],
                torch.ones_like(before[0, 0]),
                expected[0, 0],
            )
        )
        self.assertEqual(assignment_error, 0.0)
        self.assertGreater(realized_error, 0.0)
        wrong = expected[0, 0].clone()
        wrong[0] = wrong[0] + torch.tensor(4.0, dtype=torch.bfloat16)
        wrong_assignment_error, _ = common._authoritative_additive_assignment_errors(
            before[0, 0], torch.ones_like(before[0, 0]), wrong
        )
        self.assertGreater(wrong_assignment_error, 0.0)

    def test_singleton_joint_capture_mismatch_is_observation_only(self) -> None:
        joint = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        exact = runtime._singleton_joint_capture_comparison(joint.clone(), joint)
        self.assertTrue(exact["exact_equal"])
        self.assertEqual(exact["exact_mismatch_count"], 0)

        singleton = joint.clone()
        singleton[3, 0] += 0.125
        observed = runtime._singleton_joint_capture_comparison(singleton, joint)
        self.assertFalse(observed["exact_equal"])
        self.assertEqual(observed["exact_mismatch_count"], 1)
        self.assertEqual(observed["exact_mismatch_fraction"], 1 / 8)
        self.assertEqual(
            observed["status"],
            "FINITE_BATCH_KERNEL_NUMERIC_DIFFERENCE_OBSERVED",
        )
        self.assertEqual(
            observed["controller_selection_endpoint_influence_count"], 0
        )
        self.assertEqual(observed["tolerance_or_rescue_count"], 0)
        with self.assertRaises(ODEBFContractError):
            runtime._singleton_joint_capture_comparison(singleton, joint.double())
        invalid = singleton.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            runtime._singleton_joint_capture_comparison(invalid, joint)

    def test_shared_target_field_anchor_gradient_and_layer_isolation(self) -> None:
        field, _, _ = _shared_field()
        model = _ColdLayerOverlayModel()
        target = field.target_state.clone().requires_grad_(True)
        coefficients = torch.tensor((0.125, 0.10, 0.075, 0.05, 0.025))
        hidden = {
            layer.layer: torch.eye(11, dtype=torch.float32)[:2]
            for layer in field.layers
        }
        with common.CommonSharedResidualTargetFieldOverlay(
            model, field, coefficients, target
        ):
            outputs = {
                layer.layer: model.layers[str(layer.layer)](hidden[layer.layer])
                for layer in field.layers
            }
        for index, layer in enumerate(field.layers):
            expected = (
                (hidden[layer.layer] @ layer.q) @ layer.residual.T
            ) * coefficients[index]
            self.assertTrue(torch.allclose(outputs[layer.layer], expected))

        scalar = sum(value.square().sum() for value in outputs.values())
        observed = torch.autograd.grad(scalar, target)[0]
        epsilon = 1.0e-3
        plus = target.detach().clone()
        minus = target.detach().clone()
        plus[0, 0] += epsilon
        minus[0, 0] -= epsilon

        def value_at(value: torch.Tensor) -> float:
            value = value.requires_grad_(True)
            with common.CommonSharedResidualTargetFieldOverlay(
                model, field, coefficients, value
            ):
                return float(
                    sum(
                        model.layers[str(layer.layer)](hidden[layer.layer]).square().sum()
                        for layer in field.layers
                    ).detach()
                )

        finite = (value_at(plus) - value_at(minus)) / (2.0 * epsilon)
        self.assertAlmostEqual(float(observed[0, 0]), finite, delta=2.0e-2)

        changed, _, _ = _shared_field(residual_delta=0.5)
        for ordinal, (left, right) in enumerate(zip(field.layers, changed.layers)):
            if ordinal == 0:
                self.assertNotEqual(tensor_sha256(left.residual), tensor_sha256(right.residual))
            else:
                # The helper intentionally applies a common residual; changing it
                # changes every writer and proves the unique shared tensor identity.
                self.assertNotEqual(tensor_sha256(left.residual), tensor_sha256(right.residual))

    def test_semantic_field_identity_excludes_only_covariance_wall(self) -> None:
        first, solve, risk = _shared_field(wall_seconds=0.25)
        repeated, repeated_solve, repeated_risk = _shared_field(wall_seconds=9.5)
        left = runtime._common_field_semantic_receipt(
            first, solve_history=solve, risk_history=risk
        )
        right = runtime._common_field_semantic_receipt(
            repeated, solve_history=repeated_solve, risk_history=repeated_risk
        )
        self.assertEqual(left["semantic_identity_sha256"], right["semantic_identity_sha256"])
        self.assertNotEqual(left["cost_telemetry"], right["cost_telemetry"])
        changed, changed_solve, changed_risk = _shared_field(residual_delta=0.5)
        changed_receipt = runtime._common_field_semantic_receipt(
            changed, solve_history=changed_solve, risk_history=changed_risk
        )
        self.assertNotEqual(
            left["semantic_identity_sha256"],
            changed_receipt["semantic_identity_sha256"],
        )

    def test_initial_rs_contract_is_checked_before_candidate_action(self) -> None:
        field, _, _ = _shared_field()
        field_receipt = SimpleNamespace(field_semantic_sha256="b" * 64)
        routing_value = SimpleNamespace(
            signed_slopes=(1.0, 2.0, 3.0, 4.0, 5.0),
            p_max=7.0,
            requested_progress=1.75,
            pre_soft_velocity=(0.1, 0.2, 0.3, 0.4, 0.5),
        )
        target = torch.arange(80, dtype=torch.float32).reshape(8, 10)
        metric = common.CommonColdScaleMetric.from_z_base(
            torch.ones((8, 10), dtype=torch.float32),
            "a" * 64,
            common.CommonColdScale.ROBUST_SHARED,
        )
        left = runtime._common_initial_contract(
            bootstrap_target=target,
            field=field,
            field_receipt=field_receipt,
            routing=routing_value,
            metric=metric,
        )
        right = runtime._common_initial_contract(
            bootstrap_target=target.clone(),
            field=field,
            field_receipt=field_receipt,
            routing=routing_value,
            metric=metric,
        )
        self.assertEqual(left, right)
        changed = runtime._common_initial_contract(
            bootstrap_target=target + 1.0,
            field=field,
            field_receipt=field_receipt,
            routing=routing_value,
            metric=metric,
        )
        self.assertNotEqual(left["identity_sha256"], changed["identity_sha256"])
        source = inspect.getsource(runtime._run_common_arm)
        self.assertLess(
            source.index("expected_initial_contract"),
            source.index("candidate_factors ="),
        )

    def test_shared_backend_input_is_explicit_and_legacy_default_is_frozen(self) -> None:
        residual = torch.randn((7, 10), generator=torch.Generator().manual_seed(71))
        current = torch.randn((7, 10), generator=torch.Generator().manual_seed(72))
        value = p1_backend.SharedTerminalResidualInput(
            residual.contiguous(), current.contiguous(), "a" * 64
        )
        payload = value.raw_free_payload()
        self.assertEqual(payload["backend_recapture_count"], 0)
        self.assertEqual(payload["z_minus_h_layer_recompute_count"], 0)
        self.assertEqual(payload["remaining_layer_division_count"], 0)
        signature = inspect.signature(p1_backend.build_p1_dynamic_field)
        self.assertIsNone(signature.parameters["shared_terminal_residual"].default)
        source = inspect.getsource(p1_backend.build_p1_dynamic_field)
        self.assertIn("shared_terminal_residual.residual.clone()", source)
        self.assertIn("len(set(residual_hashes)) != 1", source)
        with self.assertRaises(ODEBFContractError):
            p1_backend.SharedTerminalResidualInput(
                residual.double().contiguous(), current, "a" * 64
            )
        with self.assertRaises(ODEBFContractError):
            p1_backend.SharedTerminalResidualInput(
                residual[:, :9].contiguous(), current[:, :9].contiguous(), "a" * 64
            )

    def test_stage2_failure_policy_is_opt_in_and_preserves_stage1(self) -> None:
        real = routing._solve_slsqp

        def fail_stage2(**kwargs):
            value, certificate = real(**kwargs)
            if kwargs["phase"] == "soft-minimum-capacity-within-xi-tie":
                certificate = replace(
                    certificate,
                    stationarity_residual=1.0,
                    first_false_component="stationarity",
                    passed=False,
                )
                if kwargs["fail_closed"]:
                    observer = kwargs.get("certificate_observer")
                    if observer is not None:
                        observer(certificate)
                    raise ODEBFContractError("NUMERIC_QP_UNCERTIFIED")
            return value, certificate

        with mock.patch.object(routing, "_solve_slsqp", side_effect=fail_stage2):
            with self.assertRaisesRegex(ODEBFContractError, "NUMERIC_QP_UNCERTIFIED"):
                solve_fixed_e8_routing(_problem(), _inventory(), arm=FixedE8Arm.SOFT)
        with mock.patch.object(routing, "_solve_slsqp", side_effect=fail_stage2):
            observed = solve_fixed_e8_routing(
                _problem(),
                _inventory(),
                arm=FixedE8Arm.SOFT,
                stage2_failure_policy=(
                    FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
                ),
            )
        self.assertEqual(observed.selected_solution_source, "CERTIFIED_STAGE1_FALLBACK")
        self.assertEqual(observed.stage1_selection_fallback_count, 1)
        self.assertIsNotNone(observed.stage1_selection_receipt)
        receipt = observed.stage1_selection_receipt
        assert receipt is not None
        self.assertEqual(
            receipt["preserved_stage1_vector_sha256"],
            receipt["applied_vector_sha256"],
        )
        self.assertEqual(
            canonical_hash(list(observed.velocity)), receipt["applied_vector_sha256"]
        )
        self.assertEqual(len(observed.failed_authoritative_certificates), 1)
        self.assertFalse(observed.failed_authoritative_certificates[0].passed)

        with mock.patch.object(
            routing,
            "_solve_slsqp",
            side_effect=RuntimeError("programming defect"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programming defect"):
                solve_fixed_e8_routing(
                    _problem(),
                    _inventory(),
                    arm=FixedE8Arm.SOFT,
                    stage2_failure_policy=(
                        FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
                    ),
                )

    def test_common_seal_lock_resource_and_dispatch_contracts(self) -> None:
        seal = verify_common_cold_case_seal(
            json.loads((LOCKS / COMMON_COLD_CASE_SEAL_FILE).read_text())
        )
        self.assertEqual(
            seal["status"], "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION"
        )
        self.assertEqual(seal["heldout_fields_opened_during_selection"], 0)
        self.assertEqual(len(seal["requests"]), 10)
        self.assertEqual(
            seal["warm_source"]["source_head"],
            "c3d45eba301d3e449a03a21f7ce65b5aa70d07c2",
        )
        self.assertEqual(
            seal["batch_ordered_request_digest_v1"],
            ["984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"],
        )
        self.assertEqual(
            [item["case_id"] for item in seal["requests"]],
            [8614, 6141, 10115, 18369, 19104, 18146, 7960, 16747, 4119, 17055],
        )
        self.assertEqual(seal["requests"][6]["case_id"], 7960)
        self.assertEqual(seal["retired_fresh_draft"]["status"], "RETIRED_BEFORE_USE_BY_A4")
        self.assertEqual(seal["retired_fresh_draft"]["model_load_count"], 0)
        self.assertFalse(seal["unseen_or_fresh_sample_claim_authorized"])

        from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
            common_cold_schedule,
        )
        from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal

        sampling = load_p1_sampling_seal(
            LOCKS / "p1r2_p_population_seal.json",
            stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
        )
        schedule = common_cold_schedule(sampling)
        lock, _ = load_and_validate_common_cold_lock(
            LOCKS / COMMON_COLD_NUMERICAL_LOCK_FILE,
            controller_identity_sha256=(
                __import__(
                    "project.run_scripts.ode_bf.p1_controller",
                    fromlist=["P1ControllerLock"],
                ).P1ControllerLock().identity()
            ),
            case_root_digest=seal["root_digest"],
            population_root_digest=json.loads(
                (LOCKS / "p1r2_p_population_seal.json").read_text()
            )["root_digest"],
            schedule=schedule,
        )
        self.assertEqual(lock["arms"], ["RS-NEUTRAL", "RS-SOFT", "BG-NEUTRAL"])
        forecast = forecast_common_cold_panel(
            LOCKS / "p0_artifact_lock.json",
            ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            "llama3-8b-inst",
        )
        capacity = validate_common_cold_runtime_gpu_capacity(
            forecast,
            device_property_total_bytes=forecast.allocatable_calibration_bytes,
            allocatable_total_bytes=forecast.allocatable_calibration_bytes,
            free_bytes=forecast.allocatable_calibration_bytes,
        )
        self.assertTrue(capacity["passed"])
        self.assertEqual(
            capacity["stable_device_total_bytes"],
            forecast.allocatable_calibration_bytes,
        )
        self.assertEqual(
            capacity["physical_inventory_total_bytes"],
            forecast.physical_total_bytes,
        )
        self.assertEqual(
            capacity["physical_inventory_role"],
            "LOCKED_FORECAST_PROVENANCE_ONLY",
        )
        self.assertTrue(capacity["physical_and_allocatable_semantics_separate"])
        with self.assertRaisesRegex(
            ODEBFContractError, "stable GPU device identity"
        ):
            validate_common_cold_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=forecast.physical_total_bytes,
                allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                free_bytes=forecast.allocatable_calibration_bytes,
            )
        with self.assertRaisesRegex(
            ODEBFContractError, "allocatable GPU capacity"
        ):
            validate_common_cold_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=forecast.allocatable_calibration_bytes,
                allocatable_total_bytes=forecast.allocatable_calibration_bytes + 1,
                free_bytes=forecast.allocatable_calibration_bytes,
            )
        with self.assertRaisesRegex(
            ODEBFContractError, "allocatable GPU capacity"
        ):
            validate_common_cold_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=forecast.allocatable_calibration_bytes,
                allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                free_bytes=forecast.allocatable_calibration_bytes + 1,
            )
        required = forecast.conservative_gpu_peak_mib * 1024 * 1024
        with self.assertRaisesRegex(ODEBFContractError, "insufficient"):
            validate_common_cold_runtime_gpu_capacity(
                forecast,
                device_property_total_bytes=forecast.allocatable_calibration_bytes,
                allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                free_bytes=required - 1,
            )
        for malformed in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(ODEBFContractError, "receipt differs"):
                validate_common_cold_runtime_gpu_capacity(
                    forecast,
                    device_property_total_bytes=malformed,  # type: ignore[arg-type]
                    allocatable_total_bytes=forecast.allocatable_calibration_bytes,
                    free_bytes=forecast.allocatable_calibration_bytes,
                )

        signature = inspect.signature(p1_runtime.run_p1)
        self.assertFalse(signature.parameters["common_cold_fixed_e8_mode"].default)
        tree = ast.parse(inspect.getsource(p1_runtime.run_p1))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertIn("common_cold_fixed_e8_mode", names)

    def test_reused_warm_context_and_evaluator_identity_are_exact(self) -> None:
        requests = tuple(
            {
                "request_sha256": f"{index + 1:064x}",
                "prompt": "{} is located in",
                "subject": f"synthetic-{index}",
            }
            for index in range(10)
        )
        order = __import__(
            "project.run_scripts.ode_bf.request_digest",
            fromlist=["ordered_request_digest_v1"],
        ).ordered_request_digest_v1(
            [item["request_sha256"] for item in requests]
        )
        contexts = [["{}"], ["prefix-a {}", "prefix-b {}", "prefix-c {}", "prefix-d {}", "prefix-e {}"]]
        template_hashes = [
            canonical_hash({"template": item})
            for group in contexts
            for item in group
        ]
        warm = {
            "context_file_sha256": "a" * 64,
            "context_sha256": canonical_hash(contexts),
            "context_template_sha256": template_hashes,
            "evaluation_case_identity_sha256": "b" * 64,
            "target_span_sha256": "c" * 64,
        }
        stream = {
            "panel_kind": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
            "batch_ordered_request_digest_v1": [order],
        }
        with mock.patch.dict(runtime.WARM_ALLOFF_ROOTS, {"llama3-8b-inst": warm}):
            observed = runtime._warm_context_identity_receipt(
                alias="llama3-8b-inst",
                requests=requests,
                stream=stream,
                contexts=contexts,
                context_sha256=canonical_hash(contexts),
            )
            self.assertTrue(observed["context_inventory_exact_equal"])
            self.assertTrue(observed["rendered_context_all_exact_equal"])
            self.assertEqual(
                observed["rendered_context_byte_exact_equal"], [[True] * 6] * 10
            )

            primary = SimpleNamespace(
                request_order_sha256=order,
                evaluator_source_sha256=runtime.PINNED_SOURCE_SHA256[
                    runtime.COUNTERFACT_EVALUATOR_RELATIVE
                ],
                aggregator_source_sha256=runtime.PINNED_SOURCE_SHA256[
                    runtime.OFFICIAL_AGGREGATOR_RELATIVE
                ],
                evaluation_case_identity_sha256="b" * 64,
                target_span_sha256="c" * 64,
                generation_call_count=0,
                boundary_touched=False,
            )
            receipt = SimpleNamespace(primary=primary)
            parity = runtime._warm_evaluator_parity_receipt(
                {("W0_NO_EDIT", 0): receipt},
                alias="llama3-8b-inst",
                request_order_sha256=order,
            )
            self.assertTrue(parity["warm_metric_definition_exact"])
            primary.boundary_touched = True
            with self.assertRaises(ODEBFContractError):
                runtime._warm_evaluator_parity_receipt(
                    {("W0_NO_EDIT", 0): receipt},
                    alias="llama3-8b-inst",
                    request_order_sha256=order,
                )

    def test_r10_h_coupling_freeze_order_and_no_cold_native_access(self) -> None:
        target_source = inspect.getsource(common.write_aware_common_target_velocity)
        self.assertIn("COMMON_COLD_H * item", target_source)
        self.assertIn("CommonSharedResidualTargetFieldOverlay", target_source)
        common_tree = ast.parse(inspect.getsource(common))
        called = {
            node.func.id
            for node in ast.walk(common_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("compute_z", called)
        self.assertNotIn("capture_p1_native_entry", called)
        self.assertNotIn("native_target", called)
        run_source = inspect.getsource(runtime.run_common_coldcoord_fixed_e8_diagnostic)
        self.assertLess(
            run_source.index("_prior_qwen_ordinal6_audit"),
            run_source.index("z_base = capture_cold_z_base"),
        )
        self.assertLess(
            run_source.index("action_freeze ="),
            run_source.index("native_capture = capture_p1_native_entry"),
        )
        ordinal_source = inspect.getsource(runtime._prior_qwen_ordinal6_audit)
        self.assertNotIn("p1r6_cold_cf_b10_seal", ordinal_source)
        self.assertIn("prior_requests = tuple(requests)", ordinal_source)
        self.assertNotIn(
            "prior ordinal6 common capture identity failed", ordinal_source
        )
        self.assertIn("_singleton_joint_capture_comparison", ordinal_source)
        dispatch_source = inspect.getsource(p1_runtime.run_p1)
        self.assertIn("requests=cold_requests", dispatch_source)
        self.assertIn("stream=cold_stream", dispatch_source)
        arm_source = inspect.getsource(runtime._run_common_arm)
        self.assertIn("range(FIXED_E8_GRID_COUNT)", arm_source)
        self.assertIn('"scientific_retry_count": 0', arm_source)
        self.assertNotIn("backtrack", arm_source.lower())

    def test_r10_dry_plan_and_submit_lineage_are_exact(self) -> None:
        first = common_dry.build_plan(COMMON_COLD_PARENT_HEAD)
        second = common_dry.build_plan(COMMON_COLD_PARENT_HEAD)
        self.assertEqual(first, second)
        self.assertFalse(first["model_load"])
        self.assertFalse(first["gpu_use"])
        self.assertFalse(first["slurm_submit"])
        self.assertEqual(first["new_pair_gpu"], 2)
        self.assertEqual(first["server1_project_gpu_cap"], 3)
        self.assertEqual(
            first["panel_kind"], "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION"
        )
        self.assertFalse(first["unseen_or_fresh_sample_claim_authorized"])
        self.assertEqual(
            COMMON_COLD_RESULT_TOKEN,
            "common-coldcoord-fixed-e8-p1r10-r2-v1",
        )
        self.assertEqual(
            expected_common_cold_result_name("llama3-8b-inst"),
            "s05-common-coldcoord-fixed-e8-p1r10-r2-llama3-8b-inst-v1",
        )
        self.assertEqual(
            common_dry.JOB_NAMES,
            {
                "llama3-8b-inst": "odeedit_s05_r10r2_llama",
                "qwen2.5-7b-inst": "odeedit_s05_r10r2_qwen",
            },
        )
        self.assertEqual(
            common_submit.EXECUTION_BRANCH,
            "codex/odeeditsh1-s05-common-coldcoord-fixed-e8-p1r10-v1",
        )
        self.assertEqual(
            common_submit.EXECUTION_REPAIR_PARENT_HEAD,
            "abaa366c8d32918ecf96d4f433248b799703001b",
        )
        self.assertEqual(
            common_submit.FIRST_REPAIR_PARENT_HEAD,
            "d60ddaf765f79f4f4f73c2dc455f8aef7521084e",
        )
        provenance_source = inspect.getsource(
            common_submit._execution_provenance_gate
        )
        self.assertIn('parent != EXECUTION_REPAIR_PARENT_HEAD', provenance_source)
        self.assertIn(
            'first_repair_parent != FIRST_REPAIR_PARENT_HEAD', provenance_source
        )
        self.assertIn(
            'scientific_parent != COMMON_COLD_PARENT_HEAD', provenance_source
        )
        self.assertIn('head != source_head', provenance_source)
        self.assertIn('branch != EXECUTION_BRANCH', provenance_source)
        self.assertIn('or dirty', provenance_source)


if __name__ == "__main__":
    unittest.main()
