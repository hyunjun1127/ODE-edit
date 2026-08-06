from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts import (
    session05_ode_bf_cold_structp_softp_noveto_dry_plan as dry,
)
from project.run_scripts import (
    session05_ode_bf_submit_cold_structp_softp_noveto as submit,
)
from project.run_scripts.ode_bf import cold_start_target as cold
from project.run_scripts.ode_bf import p1_adaptive_runtime as adaptive_runtime
from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import WaypointFactor, tensor_sha256
from project.run_scripts.ode_bf.p1_backend import (
    CovarianceActionReceipt,
    FULL_CURRENT_RESIDUAL_DEFINITION,
    FULL_CURRENT_RESIDUAL_DIVISOR,
    P1DynamicField,
    P1LayerField,
)
from project.run_scripts.ode_bf.p1_cold_structp_softp_noveto_panel import (
    COLD_CASE_SALT,
    COLD_PANEL_LABELS,
    COLD_PARENT_HEAD,
    cold_schedule,
    expected_cold_result_name,
    forecast_cold_panel,
    load_cold_requests,
    validate_cold_runtime_gpu_capacity,
    validate_cold_lock,
    verify_cold_case_seal,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_selection import (
    load_p1_request_identities,
    scan_prior_tracked_seals,
)
from project.run_scripts.ode_bf.p1_state import P1HistoryLedger
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal
from project.run_scripts.ode_bf.woodbury import (
    ProjectorCertificate,
    WoodburyCertificate,
    WoodburyMethod,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"
EASYEDIT = Path("/mnt/raid5/janghj/EasyEdit")
DATASET = EASYEDIT / "data/counterfact/counterfact.json"
BASE = "b43c25c57804353ad67c10185d49db6ad0e22aec"


class _OverlayModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


def _cold_semantic_field(
    *,
    wall_seconds: float,
    mutation: str | None = None,
) -> tuple[P1DynamicField, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(6101)
    target = torch.randn((12, 10), generator=generator, dtype=torch.float32)
    layers: list[P1LayerField] = []
    projector = ProjectorCertificate(
        "1" * 64,
        1.0,
        1.0,
        "artifact-unverified",
        1.0e-10,
    )
    certificate = WoodburyCertificate(
        WoodburyMethod.GENERAL_LU,
        projector,
        10,
        2.0,
        1.0e-12,
        1.0e-12,
        True,
        True,
    )
    for ordinal, layer in enumerate(cold.COLD_LAYER_ORDER):
        residual = torch.randn((12, 10), generator=generator, dtype=torch.float32)
        q = torch.randn((11, 10), generator=generator, dtype=torch.float32)
        key = torch.randn((11, 10), generator=generator, dtype=torch.float32)
        projected = torch.randn((11, 10), generator=generator, dtype=torch.float32)
        action = 0.01 * q
        gram = q.double().T @ action.double()
        if layer == cold.COLD_LAYER_ORDER[0]:
            if mutation == "residual":
                residual = residual.clone()
                residual[0, 0] += 1.0
            elif mutation == "q":
                q = q.clone()
                q[0, 0] += 1.0
                action = 0.01 * q
                gram = q.double().T @ action.double()
            elif mutation == "action":
                action = action.clone()
                action[0, 0] += 1.0
            elif mutation == "gram":
                gram = gram.clone()
                gram[0, 0] += 1.0
        factor = WaypointFactor(
            f"layers.{layer}.weight",
            layer,
            0,
            0,
            ordinal,
            1.0,
            residual,
            q,
        )
        receipt = CovarianceActionReceipt(
            layer,
            str(layer) * 64,
            100,
            (11, 11),
            100000,
            (11, 10),
            tensor_sha256(q),
            tensor_sha256(action),
            tensor_sha256(gram),
            True,
            wall_seconds + ordinal,
        )
        layers.append(
            P1LayerField(
                layer,
                factor.weight_name,
                key,
                projected,
                residual,
                FULL_CURRENT_RESIDUAL_DEFINITION,
                FULL_CURRENT_RESIDUAL_DIVISOR,
                q,
                factor,
                float(torch.sum((residual.T @ residual) * (q.T @ q))),
                action,
                gram,
                receipt,
                certificate,
                torch.empty((12, 0), dtype=torch.float64),
            )
        )
    if mutation == "target":
        target = target.clone()
        target[0, 0] += 1.0
    request_order = "a" * 64 if mutation != "request" else "b" * 64
    field = P1DynamicField(
        0,
        request_order,
        target,
        target - layers[-1].residual,
        tuple(layers),
        canonical_hash({"wall_seconds": wall_seconds, "mutation": mutation}),
        10,
        0,
    )
    solve = {
        layer: torch.empty((0, 0), dtype=torch.float32)
        for layer in cold.COLD_LAYER_ORDER
    }
    risk = {
        layer: torch.empty((0, 0), dtype=torch.float32)
        for layer in cold.COLD_LAYER_ORDER
    }
    return field, solve, risk


def _call_names(function: object) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                result.add(target.id)
            elif isinstance(target, ast.Attribute):
                result.add(target.attr)
    return result


class ColdTargetTests(unittest.TestCase):
    def test_fixed_g_metric_produces_unit_descent_and_fails_closed(self) -> None:
        z_base = torch.arange(1, 41, dtype=torch.float32).reshape(4, 10)
        metric = cold.ColdTargetMetric.from_z_base(z_base, "a" * 64)
        gradient = torch.arange(40, 0, -1, dtype=torch.float32).reshape(4, 10)
        velocity, norms = metric.unit_descent(gradient)
        self.assertEqual(velocity.shape, z_base.shape)
        self.assertEqual(len(norms), 10)
        for value in metric.distance(velocity, torch.zeros_like(velocity)):
            self.assertLessEqual(abs(value - 1.0), cold.COLD_METRIC_TOLERANCE)
        self.assertTrue(torch.all(torch.sum(velocity * gradient, dim=0) < 0.0))
        payload = metric.raw_free_payload()
        self.assertEqual(payload["native_or_direct_z_dependency_count"], 0)
        with self.assertRaisesRegex(ODEBFContractError, "degenerate"):
            cold.ColdTargetMetric.from_z_base(
                torch.zeros((4, 10), dtype=torch.float32), "a" * 64
            )

    def test_cold_field_semantic_identity_excludes_only_covariance_wall(self) -> None:
        first, solve, risk = _cold_semantic_field(wall_seconds=0.25)
        repeated, repeated_solve, repeated_risk = _cold_semantic_field(
            wall_seconds=9.5
        )
        first_receipt = cold.cold_field_semantic_receipt(
            first,
            history_solve_keys_by_layer=solve,
            history_risk_keys_by_layer=risk,
        )
        repeated_receipt = cold.cold_field_semantic_receipt(
            repeated,
            history_solve_keys_by_layer=repeated_solve,
            history_risk_keys_by_layer=repeated_risk,
        )
        self.assertNotEqual(first.identity_sha256, repeated.identity_sha256)
        self.assertEqual(
            first_receipt["semantic_identity_sha256"],
            repeated_receipt["semantic_identity_sha256"],
        )
        self.assertNotEqual(
            first_receipt["cost_telemetry"]["covariance_wall_seconds"],
            repeated_receipt["cost_telemetry"]["covariance_wall_seconds"],
        )
        self.assertEqual(
            tuple(
                first_receipt["cost_telemetry"][
                    "semantic_exclusion_allowlist"
                ]
            ),
            cold.COLD_FIELD_SEMANTIC_EXCLUSION_ALLOWLIST,
        )
        source = inspect.getsource(cold.cold_field_semantic_receipt)
        self.assertNotIn("startswith", source)
        self.assertNotIn("endswith", source)

        for mutation in ("action", "gram", "q", "residual", "request", "target"):
            with self.subTest(mutation=mutation):
                changed, changed_solve, changed_risk = _cold_semantic_field(
                    wall_seconds=0.25,
                    mutation=mutation,
                )
                changed_receipt = cold.cold_field_semantic_receipt(
                    changed,
                    history_solve_keys_by_layer=changed_solve,
                    history_risk_keys_by_layer=changed_risk,
                )
                self.assertNotEqual(
                    first_receipt["semantic_identity_sha256"],
                    changed_receipt["semantic_identity_sha256"],
                )

    def test_repeated_entry_uses_semantic_field_and_pre_soft_contract(self) -> None:
        readiness_field, solve, risk = _cold_semantic_field(wall_seconds=0.1)
        no_soft_field, _, _ = _cold_semantic_field(wall_seconds=0.2)
        soft_field, _, _ = _cold_semantic_field(wall_seconds=0.3)
        readiness_receipt = cold.cold_field_semantic_receipt(
            readiness_field,
            history_solve_keys_by_layer=solve,
            history_risk_keys_by_layer=risk,
        )
        readiness = cold.ColdFieldReadiness(
            reached=True,
            field_sha256=readiness_field.identity_sha256,
            field_semantic_sha256=readiness_receipt[
                "semantic_identity_sha256"
            ],
            field_semantic_receipt=readiness_receipt,
            signed_progress=(1.0, 0.8, 0.6, 0.4, 0.2),
            maximum_feasible_progress=1.0,
            layer_joint_ranks=(10,) * 5,
            nonzero_layer_count=5,
            reason="READY",
            identity_sha256="1" * 64,
        )
        history = P1HistoryLedger(
            layer_order=cold.COLD_LAYER_ORDER,
            maximum_records=40,
        )

        def active(field: P1DynamicField, bf: tuple[float, ...]) -> object:
            return SimpleNamespace(
                field=field,
                raw=SimpleNamespace(values=(1.0, 0.8, 0.6, 0.4, 0.2)),
                projection=SimpleNamespace(values=bf),
                problem=SimpleNamespace(requested_progress=0.5),
                signed_progress=SimpleNamespace(
                    signed_progress=(1.0, 0.8, 0.6, 0.4, 0.2)
                ),
            )

        shared_bf = (0.9, 0.7, 0.5, 0.3, 0.1)
        contracts = {
            cold.COLD_PANEL_LABELS[0]: cold.cold_entry_field_contract(
                cold.COLD_PANEL_LABELS[0],
                active(no_soft_field, shared_bf),  # type: ignore[arg-type]
                history,
            ),
            cold.COLD_PANEL_LABELS[1]: cold.cold_entry_field_contract(
                cold.COLD_PANEL_LABELS[1],
                active(soft_field, shared_bf),  # type: ignore[arg-type]
                history,
            ),
        }
        common = cold.validate_cold_common_entry_contracts(
            readiness, contracts
        )
        self.assertTrue(common["readiness_equals_no_soft_equals_soft"])
        self.assertTrue(common["pre_soft_raw_bf_requested_progress_equal"])
        self.assertNotEqual(
            contracts[cold.COLD_PANEL_LABELS[0]].field_semantic_receipt[
                "full_field_receipt_identity_sha256"
            ],
            contracts[cold.COLD_PANEL_LABELS[1]].field_semantic_receipt[
                "full_field_receipt_identity_sha256"
            ],
        )
        changed_contracts = dict(contracts)
        changed_contracts[cold.COLD_PANEL_LABELS[1]] = (
            cold.cold_entry_field_contract(
                cold.COLD_PANEL_LABELS[1],
                active(soft_field, (0.8, 0.7, 0.5, 0.3, 0.1)),  # type: ignore[arg-type]
                history,
            )
        )
        with self.assertRaisesRegex(ODEBFContractError, "pre-soft routing"):
            cold.validate_cold_common_entry_contracts(
                readiness, changed_contracts
            )
        variant_source = inspect.getsource(cold._run_cold_variant)
        self.assertIn("field_semantic_sha256", variant_source)
        self.assertLess(
            variant_source.index("validate_cold_common_entry_contracts("),
            variant_source.index("while active is not None"),
        )
        self.assertNotIn(
            "active.field.identity_sha256 != bootstrap.readiness.field_sha256",
            variant_source,
        )

    def test_bootstrap_reject_is_pure_then_accepts_half_step_without_write(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        parameter_before = model.weight.detach().clone()
        pointer = model.weight.data_ptr()
        version = model.weight._version
        rng = torch.get_rng_state().clone()
        z_base = torch.ones((2, 10), dtype=torch.float32)
        no_gradient_calls = 0

        def objective(
            _model: object,
            _tokenizer: object,
            _requests: object,
            _contexts: object,
            *,
            layer_name: str,
            lookup_positions: object,
            target_state: torch.Tensor,
            require_gradient: bool,
        ) -> cold.ColdObjectiveReceipt:
            del _model, _tokenizer, _requests, _contexts, layer_name, lookup_positions
            nonlocal no_gradient_calls
            if require_gradient:
                value = 1.0 if no_gradient_calls == 0 else 0.8
                gradient = torch.ones_like(target_state, dtype=torch.float64)
                backward_count = 10
            else:
                no_gradient_calls += 1
                value = 1.1 if no_gradient_calls == 1 else 0.8
                gradient = None
                backward_count = 0
            identity = canonical_hash(
                {
                    "value": value,
                    "target": tensor_sha256(target_state),
                    "gradient": require_gradient,
                    "ordinal": no_gradient_calls,
                }
            )
            return cold.ColdObjectiveReceipt(
                value,
                (value,) * 10,
                tensor_sha256(target_state),
                "1" * 64,
                "2" * 64,
                "3" * 64,
                60,
                600,
                backward_count,
                gradient,
                identity,
            )

        ready = cold.ColdFieldReadiness(
            reached=True,
            field_sha256="4" * 64,
            field_semantic_sha256="6" * 64,
            field_semantic_receipt={
                "semantic_identity_sha256": "6" * 64
            },
            signed_progress=(1.0,) * 5,
            maximum_feasible_progress=1.0,
            layer_joint_ranks=(2,) * 5,
            nonzero_layer_count=5,
            reason="READY",
            identity_sha256="5" * 64,
        )
        requests = tuple({"request_sha256": f"{index:064x}"} for index in range(10))
        hparams = type(
            "HParams",
            (),
            {
                "layers": [4, 5, 6, 7, 8],
                "fact_token": "subject_last",
                "layer_module_tmp": "layer",
            },
        )()
        with (
            mock.patch.object(cold, "capture_cold_z_base", return_value=z_base),
            mock.patch.object(cold, "cold_lookup_positions", return_value=(0,) * 60),
            mock.patch.object(cold, "evaluate_cold_target_objective", side_effect=objective),
            mock.patch.object(cold, "evaluate_cold_field_readiness", return_value=ready),
        ):
            bootstrap = cold.run_cold_bootstrap(
                model,
                object(),
                requests,
                hparams,
                torch.zeros(1),
                (("{}",), ("{}",) * 5),
                covariance_registry=mock.Mock(),
                projector_sha256="6" * 64,
                lock=P1ControllerLock(),
                touched={"weight": model.weight},
                ledger=cold.ComputeLedger(),
            )
        self.assertEqual(bootstrap.n_trial, 2)
        self.assertEqual(bootstrap.k_acc, 1)
        self.assertEqual(len(bootstrap.rejected_trials), 1)
        self.assertEqual(
            bootstrap.rejected_trials[0]["delta_tau_boot"],
            {"numerator": 1, "denominator": 8, "value": 0.125},
        )
        self.assertEqual(
            bootstrap.accepted_steps[0]["delta_tau_boot"],
            {"numerator": 1, "denominator": 16, "value": 0.0625},
        )
        self.assertTrue(torch.equal(model.weight.detach(), parameter_before))
        self.assertEqual(model.weight.data_ptr(), pointer)
        self.assertEqual(model.weight._version, version)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_shared_target_weight_clock_certificate_is_exact(self) -> None:
        z_base = torch.arange(1, 31, dtype=torch.float32).reshape(3, 10)
        metric = cold.ColdTargetMetric.from_z_base(z_base, "b" * 64)
        gradient = torch.ones_like(z_base)
        velocity, _ = metric.unit_descent(gradient)
        validator = cold.cold_target_step_validator(metric, z_base)
        candidate = z_base + 0.125 * velocity
        receipt = validator(z_base, candidate, Fraction(0), Fraction(1, 8))
        self.assertTrue(receipt["target_weight_shared_delta_tau"])
        self.assertEqual(receipt["clip_or_projection_count"], 0)
        with self.assertRaisesRegex(ODEBFContractError, "shared delta-tau"):
            validator(z_base, z_base + 0.25 * velocity, Fraction(0), Fraction(1, 8))

    def test_activation_overlay_is_request_major_and_exception_safe(self) -> None:
        model = _OverlayModel()
        targets = torch.arange(30, dtype=torch.float32).reshape(3, 10)
        positions = (0,) * 60
        pointer = model.layer.weight.data_ptr()
        version = model.layer.weight._version
        rng = torch.get_rng_state().clone()
        overlay = cold.TargetStateActivationOverlay(
            model, "layer", targets, positions
        )
        with overlay:
            for ordinal in range(60):
                observed = model(torch.zeros((1, 2, 3), dtype=torch.float32))
                request_index = ordinal // 6
                torch.testing.assert_close(
                    observed[0, 0], targets[:, request_index], rtol=0.0, atol=0.0
                )
        overlay.assert_complete()
        self.assertEqual(model.layer.weight.data_ptr(), pointer)
        self.assertEqual(model.layer.weight._version, version)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        incomplete = cold.TargetStateActivationOverlay(
            model, "layer", targets, positions
        )
        with incomplete:
            model(torch.zeros((1, 2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ODEBFStateError, "call/cleanup"):
            incomplete.assert_complete()
        self.assertEqual(len(model.layer._forward_hooks), 0)
        exceptional = cold.TargetStateActivationOverlay(
            model, "layer", targets, positions
        )
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with exceptional:
                model(torch.zeros((1, 2, 3), dtype=torch.float32))
                raise RuntimeError("injected")
        self.assertEqual(len(model.layer._forward_hooks), 0)
        self.assertEqual(model.layer.weight.data_ptr(), pointer)
        self.assertEqual(model.layer.weight._version, version)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_cold_action_call_graph_has_no_native_or_direct_z_primitive(self) -> None:
        for function in (
            cold.capture_cold_z_base,
            cold.evaluate_cold_target_objective,
            cold.run_cold_bootstrap,
            cold.write_aware_cold_target_velocity,
            cold._run_cold_variant,
        ):
            calls = _call_names(function)
            self.assertNotIn("capture_p1_native_entry", calls)
            self.assertNotIn("compute_z", calls)
        source = inspect.getsource(cold.run_cold_diagnostic)
        freeze = source.index('"action-freeze.json"')
        native_import = source.index("from .p1_backend import capture_p1_native_entry")
        native_call = source.index("capture_p1_native_entry(", native_import)
        self.assertLess(freeze, native_import)
        self.assertLess(native_import, native_call)
        module_tree = ast.parse(
            (ROOT / "project/run_scripts/ode_bf/cold_start_target.py").read_text(
                encoding="utf-8"
            )
        )
        top_imports = [
            node
            for node in module_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "capture_p1_native_entry" for alias in node.names)
                for node in top_imports
            )
        )

    def test_adaptive_hooks_are_additive_and_legacy_defaults_are_locked(self) -> None:
        field_signature = inspect.signature(adaptive_runtime._build_active_field)
        trial_signature = inspect.signature(adaptive_runtime._run_trial)
        self.assertIs(field_signature.parameters["dynamic_entry"].default, False)
        self.assertIsNone(
            field_signature.parameters["target_state_identity_sha256"].default
        )
        self.assertIsNone(field_signature.parameters["target_velocity_builder"].default)
        self.assertEqual(
            field_signature.parameters["target_state_objective"].default,
            "MARGIN_LOCKED",
        )
        self.assertIsNone(
            trial_signature.parameters["target_state_identity_sha256"].default
        )
        self.assertIsNone(trial_signature.parameters["target_step_validator"].default)
        self.assertEqual(
            trial_signature.parameters["target_state_objective"].default,
            "MARGIN_LOCKED",
        )
        trial_source = inspect.getsource(adaptive_runtime._run_trial)
        self.assertIn("if target_step_certificate is not None", trial_source)
        base_routing_payload = trial_source.split(
            'if target_step_certificate is not None', 1
        )[0].rsplit("routing_payload =", 1)[1]
        self.assertNotIn('"target_step_certificate"', base_routing_payload)

    def test_cold_rollout_policy_is_structural_hard_p_noveto_and_full_tau(self) -> None:
        source = inspect.getsource(cold._run_cold_variant)
        self.assertIn("PreservationConstraintPolicy.LOCKED", source)
        self.assertIn("FunctionalPDecisionPolicy.OBSERVATION_ONLY", source)
        self.assertIn("FunctionalPFieldPolicy.PROBE_ONLY", source)
        self.assertIn("FunctionalPFieldPolicy.SOFT_HARD", source)
        self.assertIn("while active is not None and clock.status == \"ACTIVE\"", source)
        while_header = source.split("while active is not None", 1)[1].split(":", 1)[0]
        self.assertNotIn("first_hit", while_header)
        self.assertIn("first_hit_observation_only", source)
        self.assertIn("full_horizon_continues_after_hit", source)
        self.assertNotIn("PreservationConstraintPolicy.ALL_OFF", source)
        self.assertIn("FULL_CURRENT_RESIDUAL_DEFINITION", source)
        self.assertIn("target_state_objective=\"COLD_TARGET_NEW_NLL_G_UNIT\"", source)

        objective_source = inspect.getsource(cold.evaluate_cold_target_objective)
        self.assertIn("RoutingObjective.TARGET_NEW_NLL", objective_source)
        self.assertNotIn("target_true", objective_source)
        objective_tree = ast.parse(objective_source)
        objective_calls = [
            node
            for node in ast.walk(objective_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "evaluate_routing_objective"
        ]
        self.assertEqual(len(objective_calls), 1)
        objective_keyword = next(
            item for item in objective_calls[0].keywords if item.arg == "objective"
        )
        self.assertEqual(
            ast.unparse(objective_keyword.value),
            "RoutingObjective.TARGET_NEW_NLL",
        )

        diagnostic_source = inspect.getsource(cold.run_cold_diagnostic)
        self.assertIn("bootstrap=bootstrap", diagnostic_source)
        self.assertEqual(
            diagnostic_source.count("_run_cold_variant("),
            1,
        )

    def test_bootstrap_exit_requires_an_accepted_target_only_state(self) -> None:
        source = inspect.getsource(cold.run_cold_bootstrap)
        loop = source.index("while not readiness.reached")
        initial = source.index("AWAITING_FIRST_ACCEPTED_BOOTSTRAP_STATE")
        readiness = source.index("evaluate_cold_field_readiness(")
        self.assertLess(initial, loop)
        self.assertGreater(readiness, loop)
        self.assertIn('payload["accepted_state_objective"]', source)
        self.assertIn('"bootstrap_compute_sha256": ledger.identity()', source)

        receipt = cold.ColdObjectiveReceipt(
            1.0,
            tuple(float(index) for index in range(10)),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            60,
            123,
            10,
            torch.zeros((2, 10), dtype=torch.float64),
            "e" * 64,
        )
        payload = receipt.raw_free_payload()
        self.assertTrue(payload["gradient_present"])
        self.assertNotIn("gradient", payload)
        self.assertEqual(payload["target_old_access_count"], 0)
        self.assertEqual(payload["native_or_direct_z_access_count"], 0)

    def test_exact_hit_trajectory_is_observation_only_and_records_loss_rehit(self) -> None:
        tracker = cold.FirstHitTracker()
        for index, count in enumerate((8, 10, 9, 10), start=1):
            tracker.append(
                cold.FirstHitRecord(
                    index,
                    Fraction(index, 8),
                    f"{index:064x}",
                    count,
                    True,
                )
            )
        payload = cold._cold_exact_hit_trajectory(tracker)
        self.assertEqual(
            [item["event"] for item in payload["accepted_state_events"]],
            ["NO_HIT", "FIRST_HIT", "HIT_LOST", "REHIT"],
        )
        self.assertEqual(payload["hit_loss_count"], 1)
        self.assertEqual(payload["rehit_count"], 1)
        self.assertFalse(payload["persistent_from_first_through_terminal"])
        self.assertTrue(payload["observation_only"])
        self.assertEqual(payload["controller_dependency_count"], 0)

    def test_fresh_seal_is_recomputed_without_heldout_access(self) -> None:
        value = json.loads(
            (LOCKS / "p1r6_cold_cf_b10_seal.json").read_text(encoding="utf-8")
        )
        seal = verify_cold_case_seal(value)
        prior = scan_prior_tracked_seals(ROOT, base_commit=BASE)
        identities = load_p1_request_identities(DATASET)
        collision_counts = Counter(item.collision_sha256 for item in identities)
        duplicate = {
            identity for identity, count in collision_counts.items() if count > 1
        }
        eligible = [
            item
            for item in identities
            if item.case_id not in prior.case_ids
            and item.request_sha256 not in prior.request_sha256
            and item.collision_sha256 not in duplicate
        ]
        ranked = sorted(
            (
                hashlib.sha256(
                    f"{COLD_CASE_SALT}|{item.case_id}".encode("utf-8")
                ).hexdigest(),
                item,
            )
            for item in eligible
        )[:10]
        self.assertEqual(
            [item.case_id for _, item in ranked],
            [item["case_id"] for item in seal["requests"]],
        )
        self.assertEqual(
            ordered_request_digest_v1([item.request_sha256 for _, item in ranked]),
            seal["batch_ordered_request_digest_v1"][0],
        )
        self.assertEqual(seal["heldout_fields_opened_during_selection"], 0)
        loaded = load_cold_requests(DATASET, seal)
        self.assertEqual(len(loaded), 10)

    def test_lock_schedule_forecast_and_dry_plan_are_fail_closed(self) -> None:
        case = verify_cold_case_seal(
            json.loads(
                (LOCKS / "p1r6_cold_cf_b10_seal.json").read_text(encoding="utf-8")
            )
        )
        base_schedule = load_p1_sampling_seal(
            LOCKS / "p1r2_p_population_seal.json",
            stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
        )
        schedule = cold_schedule(base_schedule)
        population = json.loads(
            (LOCKS / "p1r2_p_population_seal.json").read_text(encoding="utf-8")
        )
        numerical = json.loads(
            (
                LOCKS / "numerical_lock_s05_cold_structp_softp_noveto.json"
            ).read_text(encoding="utf-8")
        )
        validate_cold_lock(
            numerical,
            controller_identity_sha256=P1ControllerLock().identity(),
            case_root_digest=case["root_digest"],
            population_root_digest=population["root_digest"],
            schedule=schedule,
        )
        changed = json.loads(json.dumps(numerical))
        changed["joint_cold_ode"]["functional_p_candidate_veto"] = True
        body = dict(changed)
        body.pop("root_digest")
        changed["root_digest"] = canonical_hash(body)
        with self.assertRaisesRegex(ODEBFContractError, "joint numerical"):
            validate_cold_lock(
                changed,
                controller_identity_sha256=P1ControllerLock().identity(),
                case_root_digest=case["root_digest"],
                population_root_digest=population["root_digest"],
                schedule=schedule,
            )
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            forecast = forecast_cold_panel(
                LOCKS / "p0_artifact_lock.json",
                ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
                alias,
            )
            self.assertLessEqual(
                forecast.conservative_gpu_peak_mib,
                forecast.gpu_allocatable_calibration_bytes // (1024 * 1024),
            )
            self.assertLessEqual(
                forecast.conservative_host_peak_mib,
                forecast.host_allocation_memory_mib,
            )
            self.assertLessEqual(forecast.conservative_time_seconds, 86_400)
            self.assertFalse(forecast.scientific_outcome_metric_used)
            receipt = validate_cold_runtime_gpu_capacity(
                forecast,
                physical_total_bytes=forecast.gpu_physical_total_bytes,
                allocatable_total_bytes=forecast.gpu_allocatable_calibration_bytes,
                free_bytes=forecast.gpu_allocatable_calibration_bytes,
            )
            self.assertTrue(receipt["runtime_capacity_pass"])
            self.assertFalse(receipt["host_memory_request_used_as_gpu_capacity"])
            with self.assertRaisesRegex(ODEBFContractError, "physical identity"):
                validate_cold_runtime_gpu_capacity(
                    forecast,
                    physical_total_bytes=forecast.gpu_physical_total_bytes - 1,
                    allocatable_total_bytes=forecast.gpu_allocatable_calibration_bytes,
                    free_bytes=forecast.gpu_allocatable_calibration_bytes,
                )
            with self.assertRaisesRegex(ODEBFContractError, "below forecast"):
                validate_cold_runtime_gpu_capacity(
                    forecast,
                    physical_total_bytes=forecast.gpu_physical_total_bytes,
                    allocatable_total_bytes=forecast.gpu_allocatable_calibration_bytes,
                    free_bytes=forecast.conservative_gpu_peak_mib * 1024 * 1024 - 1,
                )
            self.assertEqual(
                expected_cold_result_name(alias),
                f"s05-cold-structp-softp-noveto-p1r6-{alias}-v1",
            )
        runtime_source = inspect.getsource(adaptive_runtime).replace(" ", "")
        self.assertNotIn("host_memory_request_used_as_gpu_capacity=True", runtime_source)
        run_source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.p1_runtime",
                fromlist=["run_p1"],
            ).run_p1
        )
        self.assertLess(
            run_source.index("validate_cold_runtime_gpu_capacity"),
            run_source.index("load_original_bf16"),
        )
        plan = dry.build_plan("c" * 40, repository_root=ROOT)
        self.assertEqual(plan["panel_labels"], list(COLD_PANEL_LABELS))
        self.assertTrue(plan["pre_submit_review_hold"])
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["gpu_use"])
        self.assertFalse(plan["slurm_submit"])
        self.assertFalse(plan["result_root_creation"])
        self.assertEqual(plan["new_pair_gpu"], 2)
        self.assertEqual(plan["server1_project_gpu_cap"], 3)
        with mock.patch.dict(
            os.environ,
            {submit.APPROVAL_ENV: ""},
            clear=False,
        ):
            with self.assertRaisesRegex(ODEBFContractError, "RUN_APPROVAL"):
                submit._pre_submit("c" * 40)

    def test_execution_repair_chain_is_exact_and_checkpoint_bound(self) -> None:
        execution_head = "c" * 40

        def git_run(
            *,
            head: str = execution_head,
            parent: str = submit.EXECUTION_REPAIR_PARENT,
            scientific_parent: str = COLD_PARENT_HEAD,
            ancestor_returncode: int = 0,
            branch: str = submit.EXECUTION_BRANCH,
            dirty: str = "",
        ):
            def run(args: object, *, check: bool = True) -> object:
                command = tuple(args)  # type: ignore[arg-type]
                values = {
                    ("git", "rev-parse", "HEAD"): head,
                    ("git", "rev-parse", "HEAD^"): parent,
                    (
                        "git",
                        "rev-parse",
                        f"{submit.EXECUTION_REPAIR_PARENT}^",
                    ): scientific_parent,
                    ("git", "branch", "--show-current"): branch,
                    (
                        "git",
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ): dirty,
                }
                if command[:3] == ("git", "merge-base", "--is-ancestor"):
                    return SimpleNamespace(
                        stdout="", returncode=ancestor_returncode
                    )
                if command not in values:
                    raise AssertionError(command)
                output = values[command]
                if command[:2] != ("git", "status"):
                    output += "\n"
                return SimpleNamespace(stdout=output, returncode=0)

            return run

        approval = f"{cold.COLD_INSTRUCTION_ID}:{execution_head}"
        with mock.patch.dict(
            os.environ, {submit.APPROVAL_ENV: approval}, clear=False
        ), mock.patch.object(submit, "_run", side_effect=git_run()):
            receipt = submit._execution_provenance_gate(execution_head)
        self.assertEqual(receipt["execution_head"], execution_head)
        self.assertEqual(
            receipt["execution_repair_parent"], submit.EXECUTION_REPAIR_PARENT
        )
        self.assertEqual(receipt["scientific_parent"], COLD_PARENT_HEAD)

        failures = (
            {
                "name": "direct-b17",
                "source_head": submit.EXECUTION_REPAIR_PARENT,
                "run": git_run(
                    head=submit.EXECUTION_REPAIR_PARENT,
                    parent=COLD_PARENT_HEAD,
                ),
                "approval": (
                    f"{cold.COLD_INSTRUCTION_ID}:"
                    f"{submit.EXECUTION_REPAIR_PARENT}"
                ),
            },
            {
                "name": "wrong-parent",
                "source_head": execution_head,
                "run": git_run(parent="d" * 40),
                "approval": approval,
            },
            {
                "name": "skipped-b17",
                "source_head": execution_head,
                "run": git_run(parent=COLD_PARENT_HEAD),
                "approval": approval,
            },
            {
                "name": "amended-review",
                "source_head": execution_head,
                "run": git_run(scientific_parent="e" * 40),
                "approval": approval,
            },
            {
                "name": "rebased-chain",
                "source_head": execution_head,
                "run": git_run(ancestor_returncode=1),
                "approval": approval,
            },
            {
                "name": "wrong-approval-head",
                "source_head": execution_head,
                "run": git_run(),
                "approval": f"{cold.COLD_INSTRUCTION_ID}:{'f' * 40}",
            },
            {
                "name": "dirty-tracked-tree",
                "source_head": execution_head,
                "run": git_run(dirty=" M scientific.py"),
                "approval": approval,
            },
        )
        for failure in failures:
            with self.subTest(name=failure["name"]), mock.patch.dict(
                os.environ,
                {submit.APPROVAL_ENV: failure["approval"]},
                clear=False,
            ), mock.patch.object(submit, "_run", side_effect=failure["run"]):
                with self.assertRaises(ODEBFContractError):
                    submit._execution_provenance_gate(failure["source_head"])

        source_manifest = json.loads(
            (
                LOCKS / "source_manifest_s05_cold_structp_softp_noveto.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(source_manifest["expected_parent"], COLD_PARENT_HEAD)
        self.assertEqual(
            source_manifest["execution_repair_parent"],
            submit.EXECUTION_REPAIR_PARENT,
        )
        numerical = json.loads(
            (
                LOCKS / "numerical_lock_s05_cold_structp_softp_noveto.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(numerical["parent_source_head"], COLD_PARENT_HEAD)


if __name__ == "__main__":
    unittest.main()
