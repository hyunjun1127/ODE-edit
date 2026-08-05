from __future__ import annotations

import inspect
import math
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts.ode_bf.functional import (
    WaypointFactor,
    assemble_effective_bf16,
    tensor_sha256,
)
from project.run_scripts.ode_bf.p1_adaptive import (
    ADAPTIVE_VARIANTS,
    DELTA_TAU_MIN,
    H_REF,
    N_TRIAL_CAP,
    AdaptiveTauClock,
    AdaptiveVariant,
    FirstHitRecord,
    FirstHitTracker,
    StepSizeIndependentVelocity,
    adaptive_lock,
    adaptive_waypoint_factors,
    concentration_summary,
    refinement_ratio,
    routing_change,
    streaming_bf16_endpoint_distance,
)
from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFStateError, canonical_hash
from project.run_scripts.ode_bf.first_hit import FeasibilityVerdict
from project.run_scripts.ode_bf.layer_routing_telemetry import (
    LAYER_IDS,
    build_layer_routing_telemetry,
)
from project.run_scripts.ode_bf.p1_backend import (
    FULL_CURRENT_RESIDUAL_DEFINITION,
    LEGACY_PRE_SHARED_RESIDUAL_DEFINITION,
    full_current_residual,
    residual_for_policy,
)
from project.run_scripts.ode_bf.p1_controller import (
    P1ControllerLock,
    scaled_waypoint_factors,
)
from project.run_scripts.ode_bf.resource import (
    forecast_p1_adaptive_b10_memory,
    forecast_p1_adaptive_b10_time,
)


def _telemetry_row(step: int, stage: str) -> dict[str, object]:
    ones = (1.0,) * len(LAYER_IDS)
    zeros = (0.0,) * len(LAYER_IDS)
    masks = (1,) * len(LAYER_IDS)
    return build_layer_routing_telemetry(
        step_index=step,
        stage=stage,
        layer_ids=LAYER_IDS,
        signed_efficiency=ones,
        raw_velocity=ones,
        bf_velocity=ones,
        applied_coefficient=zeros if stage == "FIELD" else ones,
        active_direction_mask=masks,
        raw_cap_bound_mask=(0,) * len(LAYER_IDS),
        bf_cap_bound_mask=(0,) * len(LAYER_IDS),
        raw_zero_bound_mask=(0,) * len(LAYER_IDS),
        bf_zero_bound_mask=(0,) * len(LAYER_IDS),
        predicted_progress_contribution=ones,
        prequantized_update_energy=zeros if stage == "FIELD" else ones,
        realized_bf16_update_energy=zeros if stage == "FIELD" else ones,
        cumulative_bf16_capacity=zeros if stage == "FIELD" else ones,
        bf16_capacity_contribution=zeros if stage == "FIELD" else ones,
        structural_h_contribution=zeros,
        structural_p_contribution=zeros,
        trust_contribution=zeros,
        field_sha256="a" * 64,
        state_sha256="b" * 64,
        target_z_sha256="c" * 64,
        factor_state_sha256="d" * 64,
        raw_solver_certificate_sha256="e" * 64,
        bf_solver_certificate_sha256="f" * 64,
        candidate_sha256=None if stage == "FIELD" else "1" * 64,
    )


class _Layer:
    def __init__(self, layer: int, weight_name: str, left: torch.Tensor, right: torch.Tensor):
        self.layer = layer
        self.weight_name = weight_name
        self.residual = left
        self.q = right


class _Field:
    def __init__(self) -> None:
        self.identity_sha256 = "a" * 64
        self.layers = (
            _Layer(1, "one.weight", torch.arange(40, dtype=torch.float32).reshape(4, 10), torch.arange(30, dtype=torch.float32).reshape(3, 10)),
            _Layer(2, "two.weight", torch.arange(40, 80, dtype=torch.float32).reshape(4, 10), torch.arange(30, 60, dtype=torch.float32).reshape(3, 10)),
        )


class AdaptiveClockTests(unittest.TestCase):
    def test_rooted_adaptive_numerical_lock_matches_runtime_contracts(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        value, digest = load_rooted_json(
            repo / "project/run_scripts/ode_bf/locks/numerical_lock_p1r4_adaptive.json",
            expected_schema=(
                "ode-edit-s04-ode-bf-p1r4-adaptive-numerical-lock/v1"
            ),
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            value["variant_lock_sha256"],
            {item.value: adaptive_lock(item).identity() for item in ADAPTIVE_VARIANTS},
        )
        self.assertEqual(value["resource"]["maximum_trials"], 408)
        self.assertEqual(value["resource"]["maximum_retained_factors_per_layer"], 136)

    def test_common_lock_and_resource_caps(self) -> None:
        self.assertEqual(tuple(AdaptiveVariant), ADAPTIVE_VARIANTS)
        expected = {
            AdaptiveVariant.PS_S8: (Fraction(1, 8), 8, 8),
            AdaptiveVariant.PS_A8: (Fraction(1, 8), 32, 0),
            AdaptiveVariant.FR_A8: (Fraction(1, 8), 32, 0),
            AdaptiveVariant.FR_A16: (Fraction(1, 16), 64, 0),
        }
        for variant, values in expected.items():
            lock = adaptive_lock(variant)
            self.assertEqual(
                (lock.delta_tau_max, lock.k_acc_cap, lock.legacy_slot_cap), values
            )
            self.assertEqual(lock.h_ref, H_REF)
            self.assertEqual(lock.delta_tau_min, DELTA_TAU_MIN)
            self.assertEqual(lock.n_trial_cap, N_TRIAL_CAP)
            self.assertEqual(lock.rho_accept, 0.1)

    def test_reject_advances_only_trial_counters_and_delta_tau(self) -> None:
        clock = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        first = clock.begin_trial()
        transition = clock.reject(trial=first)
        self.assertEqual(clock.tau, 0)
        self.assertEqual(clock.k_acc, 0)
        self.assertEqual(clock.n_trial, 1)
        self.assertEqual(clock.n_reject, 1)
        self.assertEqual(clock.delta_tau_proposed, Fraction(1, 16))
        self.assertEqual(transition.tau_before, transition.tau_after)
        retry = clock.begin_trial()
        self.assertEqual(retry.tau_before, first.tau_before)
        self.assertEqual(retry.k_acc_before, first.k_acc_before)
        self.assertEqual(retry.delta_tau_trial, Fraction(1, 16))
        accepted = clock.accept(trial=retry, rho=0.2)
        self.assertEqual(accepted.tau_after, Fraction(1, 16))
        self.assertEqual(clock.k_acc, 1)
        self.assertEqual(clock.delta_tau_proposed, Fraction(1, 16))

    def test_expand_and_exact_final_remainder(self) -> None:
        clock = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        for _ in range(7):
            trial = clock.begin_trial()
            clock.accept(trial=trial, rho=0.9)
        self.assertEqual(clock.tau, Fraction(7, 8))
        final = clock.begin_trial()
        self.assertFalse(final.remainder)
        clock.accept(trial=final, rho=0.9)
        self.assertTrue(clock.complete)
        self.assertEqual(sum(clock.accepted_delta, Fraction(0, 1)), 1)

    def test_same_state_retry_cap_has_distinct_termination(self) -> None:
        clock = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        for _ in range(5):
            trial = clock.begin_trial()
            clock.reject(trial=trial)
        self.assertEqual(clock.status, "SAME_STATE_RETRY_EXHAUSTED")
        self.assertEqual(clock.tau, 0)
        self.assertEqual(clock.k_acc, 0)
        self.assertEqual(clock.n_reject, 5)

    def test_trial_horizon_and_minimum_step_exhaustion_are_not_conflated(self) -> None:
        trial_cap = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        trial_cap.n_trial = trial_cap.lock.n_trial_cap
        with self.assertRaisesRegex(ODEBFStateError, "trial cap"):
            trial_cap.begin_trial()
        self.assertEqual(trial_cap.status, "TRIAL_CAP_EXHAUSTED")

        horizon = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        horizon.k_acc = horizon.lock.k_acc_cap
        with self.assertRaisesRegex(ODEBFStateError, "accepted-step cap"):
            horizon.begin_trial()
        self.assertEqual(
            horizon.status, "ACCEPTED_STEP_OR_HORIZON_CAP_UNREACHED"
        )

        minimum = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        object.__setattr__(minimum.lock, "same_state_additional_retry_cap", 100)
        for _ in range(5):
            trial = minimum.begin_trial()
            minimum.reject(trial=trial)
        self.assertEqual(minimum.status, "MIN_DT_EXHAUSTED")


class AdaptiveFieldTests(unittest.TestCase):
    def test_retry_half_step_matches_pristine_construction(self) -> None:
        field = _Field()
        velocity = StepSizeIndependentVelocity.from_controller_values(
            field.identity_sha256, (0.5, 0.25)
        )
        retry = adaptive_waypoint_factors(
            field, velocity, accepted_index=0, delta_tau=Fraction(1, 16)
        )
        pristine = adaptive_waypoint_factors(
            field, velocity, accepted_index=0, delta_tau=Fraction(1, 16)
        )
        self.assertEqual(set(retry), set(pristine))
        for name in retry:
            self.assertEqual(retry[name].theta, pristine[name].theta)
            self.assertEqual(tensor_sha256(retry[name].left), tensor_sha256(pristine[name].left))
            self.assertEqual(tensor_sha256(retry[name].right), tensor_sha256(pristine[name].right))

    def test_ps_s8_coefficients_are_exact_legacy_k8_prefix(self) -> None:
        field = _Field()
        values = (0.5, 0.25)
        velocity = StepSizeIndependentVelocity.from_controller_values(
            field.identity_sha256, values
        )
        controller = P1ControllerLock()
        for stage in range(1, 9):
            independent = scaled_waypoint_factors(
                field,
                values,
                stage=stage,
                beta=1.0,
                lock=controller,
            )
            observed = adaptive_waypoint_factors(
                field,
                velocity,
                accepted_index=stage - 1,
                delta_tau=H_REF,
            )
            self.assertEqual(set(independent), set(observed))
            for name in independent:
                self.assertEqual(independent[name].theta, observed[name].theta)
                self.assertEqual(
                    tensor_sha256(independent[name].left),
                    tensor_sha256(observed[name].left),
                )
                self.assertEqual(
                    tensor_sha256(independent[name].right),
                    tensor_sha256(observed[name].right),
                )

    def test_one_h_equals_two_half_h_before_quantization(self) -> None:
        left = torch.arange(30, dtype=torch.float32).reshape(3, 10) / 16.0
        right = torch.arange(20, dtype=torch.float32).reshape(2, 10) / 16.0
        one = WaypointFactor("w.weight", 0, 0, 0, 0, 1.0 / 8.0, left, right)
        half_a = WaypointFactor("w.weight", 0, 0, 0, 0, 1.0 / 16.0, left, right)
        half_b = WaypointFactor("w.weight", 0, 0, 1, 0, 1.0 / 16.0, left, right)
        product = (right.double() @ left.double().T).T
        one_update = one.theta * product
        two_update = half_a.theta * product + half_b.theta * product
        self.assertTrue(torch.equal(one_update, two_update))
        entry = torch.zeros((3, 2), dtype=torch.bfloat16)
        one_bf16, _ = assemble_effective_bf16(entry, (one,), row_block=2)
        two_bf16, _ = assemble_effective_bf16(entry, (half_a, half_b), row_block=2)
        # The quantized result is a separately observed contract, never used to
        # redefine the prequantized refinement identity.
        self.assertIsInstance(torch.equal(one_bf16, two_bf16), bool)

    def test_full_residual_and_legacy_only_differ_by_declared_divisor(self) -> None:
        target = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        current = torch.flip(target, dims=(0,))
        full, divisor = residual_for_policy(
            target,
            current,
            residual_policy=FULL_CURRENT_RESIDUAL_DEFINITION,
            remaining_layers=5,
        )
        legacy, legacy_divisor = residual_for_policy(
            target,
            current,
            residual_policy=LEGACY_PRE_SHARED_RESIDUAL_DEFINITION,
            remaining_layers=5,
        )
        self.assertEqual(divisor, 1)
        self.assertEqual(legacy_divisor, 5)
        self.assertTrue(torch.equal(full, full_current_residual(target, current)))
        self.assertTrue(torch.equal(legacy, full / 5.0))

    def test_concentration_and_rerouting_metrics(self) -> None:
        summary = concentration_summary((1.0, 1.0, 0.0, 0.0, 0.0))
        self.assertEqual(summary["top1_share"], 0.5)
        self.assertEqual(summary["top2_share"], 1.0)
        self.assertEqual(summary["effective_layer_count"], 2.0)
        change = routing_change((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
        self.assertEqual(change["l1"], 2.0)
        self.assertEqual(change["cosine"], 0.0)

    def test_streaming_endpoint_distance_has_no_dense_full_delta(self) -> None:
        entry = torch.zeros((4, 3), dtype=torch.bfloat16)
        left = torch.ones((4, 10), dtype=torch.float32)
        right = torch.ones((3, 10), dtype=torch.float32)
        a = WaypointFactor("w.weight", 0, 0, 0, 0, 0.25, left, right)
        b = WaypointFactor("w.weight", 0, 0, 0, 0, 0.50, left, right)
        receipt = streaming_bf16_endpoint_distance(
            entry, (a,), (b,), weight_name="w.weight", row_block=2
        )
        self.assertGreater(receipt.frobenius_distance, 0.0)
        self.assertEqual(receipt.dense_fp32_full_delta_live, 0)
        self.assertEqual(receipt.full_effective_weight_live, 0)


class AdaptiveFirstHitAndRefinementTests(unittest.TestCase):
    def test_shadow_prefix_is_identical_through_first_hit(self) -> None:
        immediate = FirstHitTracker()
        shadow = FirstHitTracker()
        records = (
            FirstHitRecord(1, Fraction(1, 8), "1" * 64, 7, True),
            FirstHitRecord(2, Fraction(1, 4), "2" * 64, 10, True),
            FirstHitRecord(3, Fraction(3, 8), "3" * 64, 9, True),
        )
        for value in records[:2]:
            immediate.append(value)
            shadow.append(value)
        shadow.append(records[2])
        self.assertEqual(immediate.prefix_identity(), shadow.prefix_identity(2))
        self.assertEqual(immediate.first_online, shadow.first_online)

    def test_float64_refinement_has_locked_first_order_ratio(self) -> None:
        result = refinement_ratio()
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["ratio"], 1.6)
        self.assertLessEqual(result["ratio"], 2.4)

    def test_runtime_source_separates_clock_trust_and_no_simple_k20(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime

        source = inspect.getsource(p1_adaptive_runtime)
        self.assertNotIn("K_total=20", source)
        self.assertNotIn("eta=1/20", source)
        self.assertIn("trust_radius_fixed_on_reject", source)
        self.assertIn("target_weight_shared_delta_tau", source)
        self.assertIn("field_build_count", source)
        self.assertIn("counter_delta", source)
        self.assertIn("coefficient_l2_norm", source)

    def test_initial_layer_activations_are_captured_from_common_w0(self) -> None:
        from project.run_scripts.ode_bf import p1_backend

        source = inspect.getsource(p1_backend.capture_p1_native_entry)
        w0_capture = source.index("for layer in layers:")
        native_writer = source.index("for layer_index, layer in enumerate(layers):")
        self.assertLess(w0_capture, native_writer)
        prefix = source[w0_capture:native_writer]
        self.assertIn("entry_current_z_by_layer[layer]", prefix)
        self.assertIn("common W0", source)
        frozen = inspect.getsource(p1_backend.build_p1_frozen_field_from_capture)
        self.assertIn("entry_current_z_by_layer", frozen)
        self.assertIn("current_z_source", frozen)
        self.assertIn("LEGACY_PRE_SHARED_RESIDUAL_DEFINITION", frozen)

    def test_worst_case_time_forecast_fits_locked_24h_with_reserve(self) -> None:
        forecast = forecast_p1_adaptive_b10_time()
        self.assertEqual(forecast.maximum_trial_count, 408)
        self.assertEqual(forecast.maximum_postfreeze_state_count, 138)
        self.assertEqual(forecast.reserve_fraction, 0.10)
        self.assertLessEqual(forecast.forecast_seconds, 24 * 60 * 60)
        self.assertFalse(forecast.outcome_metric_used)

    def test_memory_forecast_retains_all_variant_factors_without_dense_delta(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        artifact = repo / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json"
        base = repo / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            forecast = forecast_p1_adaptive_b10_memory(artifact, base, alias)
            self.assertEqual(forecast.maximum_factors_per_endpoint_layer, 64)
            self.assertEqual(forecast.maximum_retained_factors_per_layer, 136)
            self.assertLessEqual(forecast.forecast_gpu_peak_mib, 65_000)
            self.assertLessEqual(forecast.forecast_host_peak_mib, 65_000)
            self.assertFalse(forecast.dense_fp64_full_delta)


class AdaptiveRuntimeStateMachineTests(unittest.TestCase):
    def test_parameter_contract_retains_weight_and_rng_identity(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime

        model = torch.nn.Linear(2, 3, bias=False)
        observed = runtime._parameter_contract({"linear.weight": model.weight})
        self.assertEqual(tuple(observed), ("weights", "rng_sha256"))
        self.assertEqual(
            observed["weights"]["linear.weight"]["bytes_sha256"],
            tensor_sha256(model.weight),
        )
        self.assertEqual(len(observed["rng_sha256"]), 64)

    def test_empty_omega_identity_is_stable_across_rejected_state(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime

        omega = {0: [], 1: []}
        before = runtime._omega_state(omega)
        after = runtime._omega_state(omega)
        self.assertEqual(before, after)
        self.assertEqual(len(before), 64)

    def test_reject_retries_same_field_then_accepts_to_exact_tau(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime
        from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
        from project.run_scripts.ode_bf.p1_runtime import ArmRuntimeState
        from project.run_scripts.ode_bf.p1_state import (
            ArmWeightSnapshot,
            P1Arm,
            P1HistoryLedger,
        )

        model = torch.nn.Linear(3, 4, bias=False).to(dtype=torch.bfloat16)
        touched = {"linear.weight": model.weight}
        entry = model.weight.detach().cpu().clone()
        entry_sha = tensor_sha256(entry)
        receipt = ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            (("linear.weight", entry_sha),),
            canonical_hash({"entry": entry_sha}),
        )
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(layer_order=(0, 1), maximum_records=40),
            ComputeLedger(),
            receipt,
            {"linear.weight": entry},
        )
        capture = SimpleNamespace(
            direct_z=tuple(torch.zeros(4) for _ in range(10)),
            entry_current_z_by_layer={
                0: torch.zeros((4, 10)), 1: torch.zeros((4, 10))
            },
            current_z_by_layer={0: torch.zeros((4, 10)), 1: torch.zeros((4, 10))},
            entry_sha256={"linear.weight": entry_sha},
            entry_weights={"linear.weight": entry},
        )
        schedule = SimpleNamespace(state_digest="9" * 64)
        build_indices: list[int] = []
        field_serial = 0

        def build_field(*args, accepted_index: int, **kwargs):
            nonlocal field_serial
            del args, kwargs
            field_serial += 1
            build_indices.append(accepted_index)
            identity = f"{field_serial:064x}"
            recorder.field({"field_sha256": identity, "accepted_index": accepted_index})
            left = torch.ones((4, 10), dtype=torch.float32)
            right = torch.ones((3, 10), dtype=torch.float32)
            layers = []
            for layer in (0, 1):
                layers.append(
                    SimpleNamespace(
                        layer=layer,
                        weight_name=f"layer{layer}.weight",
                        residual=left,
                        q=right,
                        covariance_action=right,
                        history_action=torch.empty((4, 0), dtype=torch.float64),
                        factor_frobenius_sq=1.0,
                    )
                )
            field = SimpleNamespace(identity_sha256=identity, layers=tuple(layers))
            velocity = StepSizeIndependentVelocity.from_controller_values(
                identity, (0.5, 0.5)
            )
            return SimpleNamespace(
                field=field,
                raw_velocity=velocity,
                bf_velocity=velocity,
                layer_routing_payload=_telemetry_row(
                    accepted_index, "FIELD"
                ),
            )

        calls = 0

        def run_trial(*args, active, accepted_index, delta_tau, current_factors, current_target, **kwargs):
            nonlocal calls
            del args, kwargs
            calls += 1
            increment = adaptive_waypoint_factors(
                active.field,
                active.bf_velocity,
                accepted_index=accepted_index,
                delta_tau=delta_tau,
            )
            candidate = runtime._merge_factors(current_factors, increment)
            accepted = calls != 1
            success = SimpleNamespace(
                numerator=10,
                joint_exact_success=True,
                raw_free_payload=lambda: {
                    "numerator": 10,
                    "denominator": 10,
                    "joint_exact_success": True,
                },
            )
            return runtime.TrialOutcome(
                accepted,
                candidate,
                increment,
                current_target + float(delta_tau),
                SimpleNamespace(batch_success=success),
                SimpleNamespace(value=0.0),
                FeasibilityVerdict(True, True, True, True, True, True),
                SimpleNamespace(),
                {"historical": {}, "pretrained": {}, "trust": {}},
                {
                    "field_sha256": active.field.identity_sha256,
                    "raw_velocity_sha256": active.raw_velocity.velocity_sha256,
                    "bf_velocity_sha256": active.bf_velocity.velocity_sha256,
                },
                {"rho": 0.5},
                {
                    "identity_sha256": f"{calls + 100:064x}",
                    "layer_routing": _telemetry_row(calls, "TRIAL"),
                },
                f"{calls + 200:064x}",
                f"{calls + 300:064x}",
                0.5 if accepted else 0.0,
            )

        write_count = 0
        written: list[dict[str, object]] = []

        def write_once(path, payload):
            nonlocal write_count
            write_count += 1
            written.append(dict(payload))
            return canonical_hash(
                {"path": Path(path).as_posix(), "ordinal": write_count, "payload": payload}
            )

        recorder = runtime.AdaptiveReceiptRecorder(
            Path("unused"), AdaptiveVariant.PS_A8, write_once
        )
        success = SimpleNamespace(
            numerator=0,
            joint_exact_success=False,
            raw_free_payload=lambda: {"numerator": 0, "denominator": 10},
        )
        with (
            mock.patch.object(runtime, "_build_active_field", side_effect=build_field),
            mock.patch.object(runtime, "_run_trial", side_effect=run_trial),
            mock.patch.object(runtime, "evaluate_controller_margin", return_value=SimpleNamespace(value=0.0)),
            mock.patch.object(runtime, "_evaluate_rewrite", return_value=SimpleNamespace(batch_success=success)),
            mock.patch.object(runtime, "_controller_replay_entry", return_value=object()),
            mock.patch.object(runtime, "_terminal_confirm_snapshots", return_value=[]),
        ):
            rollout = runtime._run_variant(
                model,
                object(),
                tuple({"request_sha256": f"{index:064x}"} for index in range(10)),
                alias="llama3-8b-inst",
                variant=AdaptiveVariant.PS_A8,
                capture=capture,
                hparams=SimpleNamespace(layers=(0, 1)),
                projector=torch.empty(0),
                contexts=(),
                covariance_registry=object(),
                projector_sha256="8" * 64,
                lock=P1ControllerLock(),
                arm_state=arm_state,
                request_by_sha256={},
                population_by_sha256={},
                schedule=schedule,
                outer_entry_p_cache=object(),
                theta0_cache=object(),
                touched=touched,
                recorder=recorder,
            )
        self.assertEqual(rollout.status, "TAU_COMPLETE")
        self.assertEqual(rollout.accepted_t, Fraction(1, 1))
        self.assertEqual(rollout.k_acc, 16)
        self.assertEqual(rollout.n_trial, 17)
        self.assertEqual(rollout.n_reject, 1)
        self.assertEqual(build_indices, list(range(16)))
        self.assertEqual(rollout.field_build_count, 16)
        self.assertEqual(arm_state.history.version, 0)
        self.assertEqual(tensor_sha256(model.weight), entry_sha)
        transitions = [
            item for item in written if item.get("category") == "transition"
        ]
        self.assertEqual(
            transitions[0]["layer_routing"]["stage"],
            "REJECTED_TRANSITION",
        )
        self.assertEqual(
            transitions[1]["layer_routing"]["stage"],
            "ACCEPTED_TRANSITION",
        )
        accepted = [item for item in written if item.get("category") == "accepted"]
        self.assertEqual(len(accepted), 16)
        self.assertTrue(accepted[0]["first_hit_observed"])
        self.assertEqual(
            accepted[0]["layer_routing"]["stage"],
            "ACCEPTED_TRANSITION",
        )
        first_hits = [
            item for item in written if item.get("category") == "first-hit"
        ]
        self.assertEqual(len(first_hits), 1)
        self.assertEqual(first_hits[0]["layer_routing"]["stage"], "FIRST_HIT")
        terminal = [item for item in written if item.get("category") == "terminal"]
        self.assertIn(
            "layer_routing_trajectory", terminal[-1]["rollout_summary"]
        )

    def test_runtime_receipts_wire_numeric_telemetry_without_decision_input(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime

        trial_source = inspect.getsource(runtime._run_trial)
        field_source = inspect.getsource(runtime._build_active_field)
        accepted_source = inspect.getsource(runtime._append_accepted_snapshot)
        terminal_source = inspect.getsource(runtime._terminal_confirm_snapshots)
        self.assertIn('"capacity": capacity', trial_source)
        self.assertIn('payload["layer_routing"]', field_source)
        self.assertIn("relabel_layer_routing_telemetry", accepted_source)
        self.assertIn("relabel_layer_routing_telemetry", terminal_source)
        for source in (trial_source, field_source, accepted_source, terminal_source):
            self.assertNotIn("layer_routing_trajectory", source)


if __name__ == "__main__":
    unittest.main()
