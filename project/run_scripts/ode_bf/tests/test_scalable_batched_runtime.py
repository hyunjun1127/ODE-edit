from __future__ import annotations

import math
import json
import ast
import inspect
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    DynamicRefreshLedger,
    P1R23_LAYER_ORDER,
    ScalableComputeLedger,
    StreamingCaptureMicrobatch,
    UniformRequestAccumulator,
    accumulate_streaming_capture,
    aggregate_native_one_plus_five,
    build_streaming_batch_plan,
    expected_streaming_gradient_counts,
    initial_target_from_capture,
    scalable_ordered_request_digest,
    uniform_context_mean,
)
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    expected_p1r23_result_name,
    load_and_validate_p1r23_lock,
)
from project.run_scripts.ode_bf.p1r23_b100_seal import (
    P1R23_B10_ORDER,
    P1R23_B10_SEAL_ROOT,
    _normalize_template,
    _normalize_text,
    verify_p1r23_b100_seal,
)
from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.functional import assemble_effective_bf16, tensor_sha256
from project.run_scripts.ode_bf.atomic_runtime_optimization import (
    AcceptedPhysicalStateMaterializer,
)
from project.run_scripts.ode_bf import p1_scalable_batched_experiment as experiment
from project.run_scripts.ode_bf.scalable_batched_native import (
    run_official_native_apply,
)
from project.run_scripts.ode_bf.scalable_batched_field import (
    P1R23_TARGET_ALLOCATION_REGISTRY,
    ScalableBatchGlobalMetric,
    ScalableRobustSharedMetric,
    scalable_metric_from_allocation,
    scalable_terminal_residual,
    target_update_from_existing_gradient,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    OrdinalTargetActivationOverlay,
)


def _identities(count: int) -> tuple[str, ...]:
    return tuple(canonical_hash({"ordinal": item}) for item in range(count))


class ScalableBatchPlanTests(unittest.TestCase):
    def test_unequal_last_microbatch_and_inverse_mapping(self) -> None:
        plan = build_streaming_batch_plan(
            _identities(7), (8, 1, 7, 3, 5, 2, 4), microbatch_size=3
        )
        self.assertEqual([len(item.request_ordinals) for item in plan.batches], [3, 3, 1])
        self.assertEqual(plan.bucket_order, (1, 5, 3, 6, 4, 2, 0))
        for request_ordinal, bucket_position in enumerate(plan.inverse_order):
            self.assertEqual(plan.bucket_order[bucket_position], request_ordinal)

    def test_plan_rejects_duplicate_identity(self) -> None:
        identity = canonical_hash({"same": True})
        with self.assertRaisesRegex(ODEBFContractError, "identity inventory"):
            build_streaming_batch_plan((identity, identity), (1, 2), microbatch_size=1)

    def test_scalable_order_digest_preserves_b10_and_is_atomic_b100(self) -> None:
        b10 = _identities(10)
        from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1

        self.assertEqual(
            scalable_ordered_request_digest(b10), ordered_request_digest_v1(b10)
        )
        b100 = _identities(100)
        observed = scalable_ordered_request_digest(b100)
        self.assertEqual(len(observed), 64)
        self.assertNotEqual(observed, canonical_hash(list(b100)))
        with self.assertRaisesRegex(ODEBFContractError, "request count"):
            scalable_ordered_request_digest(_identities(20))

    def test_rooted_lock_and_distinct_result_roles(self) -> None:
        lock = Path(__file__).parents[1] / "locks" / "numerical_lock_s05_scalable_batched_runtime.json"
        value, digest = load_and_validate_p1r23_lock(lock)
        self.assertEqual(value["scientific_grid"]["K"], 8)
        self.assertEqual(len(digest), 64)
        pair = expected_p1r23_result_name(
            "llama3-8b-inst", batch_size=10, role="ODE_BF_K8_PAIR"
        )
        rs_pair = expected_p1r23_result_name(
            "llama3-8b-inst", batch_size=10, role="ODE_BF_K8_RS_PAIR"
        )
        optimized = expected_p1r23_result_name(
            "llama3-8b-inst", batch_size=10, role="OPTIMIZED_NATIVE_K1"
        )
        official = expected_p1r23_result_name(
            "llama3-8b-inst", batch_size=10, role="OFFICIAL_NATIVE"
        )
        calibration = expected_p1r23_result_name(
            "llama3-8b-inst", batch_size=10, role="CALIBRATION"
        )
        self.assertEqual(len({pair, rs_pair, optimized, official, calibration}), 5)
        self.assertIn("rs-neutral-soft-pair", rs_pair)
        with self.assertRaisesRegex(ODEBFContractError, "one-arm"):
            expected_p1r23_result_name(
                "llama3-8b-inst",
                batch_size=10,
                role="ODE_BF_K8_PAIR",
                routing_arm="BG-NEUTRAL",
            )
        self.assertEqual(value["atomic_only"]["persistent_history_append_count"], 0)

    def test_create_once_b100_seal_is_rooted_and_prefix_bound(self) -> None:
        path = Path(__file__).parents[1] / "locks" / "p1r23_b100_prefix_canonical90_seal.json"
        value = verify_p1r23_b100_seal(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(value["ordered_count"], 100)
        self.assertEqual(value["prefix"]["seal_root"], P1R23_B10_SEAL_ROOT)
        self.assertEqual(value["prefix"]["request_order_sha256"], P1R23_B10_ORDER)
        self.assertEqual(value["model_artifact_evaluator_outcome_access_count"], 0)

    def test_b100_collision_normalization_is_model_independent(self) -> None:
        self.assertEqual(_normalize_text("  Å  B\tC "), "å b c")
        self.assertEqual(_normalize_template(" The {}  is "), "the {} is")
        self.assertEqual(_normalize_template(" The {0}  is "), "the {} is")
        with self.assertRaisesRegex(ODEBFContractError, "placeholder"):
            _normalize_template("no placeholder")


class GlobalObjectiveTests(unittest.TestCase):
    def test_no_mean_of_means_for_unequal_partition(self) -> None:
        accumulator = UniformRequestAccumulator(7)
        accumulator.add((0, 1, 2), (1.0, 2.0, 3.0))
        accumulator.add((3, 4, 5), (4.0, 5.0, 6.0))
        accumulator.add((6,), (70.0,))
        observed, values = accumulator.finalize()
        self.assertEqual(values, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0))
        self.assertEqual(observed, math.fsum(values) / 7)
        self.assertNotEqual(observed, (2.0 + 5.0 + 70.0) / 3.0)

    def test_context_then_request_weighting(self) -> None:
        values = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 7.0], [3.0] * 6],
            dtype=torch.float32,
        )
        per_request = uniform_context_mean(values)
        self.assertTrue(torch.equal(per_request, torch.tensor([2.0, 3.0], dtype=torch.float64)))
        self.assertEqual(float(per_request.mean()), 2.5)

    def test_missing_and_duplicate_request_fail(self) -> None:
        missing = UniformRequestAccumulator(2)
        missing.add((0,), (1.0,))
        with self.assertRaisesRegex(ODEBFContractError, "omitted"):
            missing.finalize()
        duplicate = UniformRequestAccumulator(2)
        duplicate.add((0,), (1.0,))
        with self.assertRaisesRegex(ODEBFContractError, "duplicated"):
            duplicate.add((0,), (2.0,))


class TargetAllocationDispatchTests(unittest.TestCase):
    def test_rs_is_robust_shared_and_bg_path_is_unchanged(self) -> None:
        z0 = torch.tensor([[3.0, 0.0, 8.0], [4.0, 5.0, 6.0]])
        order = canonical_hash({"order": [0, 1, 2]})
        rs = scalable_metric_from_allocation(z0, order, "RS")
        bg = scalable_metric_from_allocation(z0, order, "BG")
        direct_bg = ScalableBatchGlobalMetric.from_z0(z0, order)
        self.assertIsInstance(rs, ScalableRobustSharedMetric)
        self.assertIsInstance(bg, ScalableBatchGlobalMetric)
        self.assertEqual(bg.raw_free_payload(), direct_bg.raw_free_payload())
        self.assertEqual(
            P1R23_TARGET_ALLOCATION_REGISTRY,
            {
                "RS": "ROBUST_SHARED_REQUEST_SCALE_V1",
                "BG": "MATCHED_BATCH_GLOBAL_SCALE_V1",
            },
        )
        gradient = torch.tensor([[3.0, 0.0, 5.0], [4.0, 2.0, 12.0]])
        velocity, receipt = rs.velocity(gradient)
        norms = torch.linalg.vector_norm(velocity.to(dtype=torch.float64), dim=0)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, rs.shared_speed)))
        self.assertEqual(receipt["allocation"], "PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED")

    def test_pair_allocation_identity_and_unsupported_fail_closed(self) -> None:
        z0 = torch.tensor([[3.0, 0.0], [4.0, 5.0]])
        order = canonical_hash({"order": [0, 1]})
        left = scalable_metric_from_allocation(z0, order, "RS")
        right = scalable_metric_from_allocation(z0, order, "RS")
        self.assertEqual(left.raw_free_payload(), right.raw_free_payload())
        with self.assertRaisesRegex(ODEBFContractError, "allocation"):
            scalable_metric_from_allocation(z0, order, "UNKNOWN")


class KeyAggregationTests(unittest.TestCase):
    def test_native_one_plus_five_is_not_uniform_six_mean(self) -> None:
        rows = torch.tensor(
            [[[10.0, 20.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]]
        )
        observed = aggregate_native_one_plus_five(rows)
        self.assertTrue(torch.equal(observed, torch.tensor([[6.0], [12.0]])))
        self.assertFalse(torch.equal(observed, rows.mean(dim=1).T))


class StreamingCaptureTests(unittest.TestCase):
    def _capture(self, *, mutate_state_on_batch: int | None = None):
        plan = build_streaming_batch_plan(
            _identities(5), (5, 1, 4, 2, 3), microbatch_size=2
        )
        calls: list[tuple[int, ...]] = []

        def forward(batch):
            calls.append(batch.request_ordinals)
            local = len(batch.request_ordinals)
            keys = {}
            for layer in P1R23_LAYER_ORDER:
                rows = torch.empty((local, 6, 2), dtype=torch.float32)
                for index, ordinal in enumerate(batch.request_ordinals):
                    rows[index, 0] = torch.tensor([100.0 + ordinal, float(layer)])
                    rows[index, 1:] = torch.tensor([float(ordinal), float(layer)])
                keys[layer] = rows
            terminal = torch.tensor(
                [[float(ordinal), float(ordinal + 10)] for ordinal in batch.request_ordinals]
            )
            before = canonical_hash({"state": 0})
            after = canonical_hash(
                {"state": 1 if batch.batch_index == mutate_state_on_batch else 0}
            )
            return StreamingCaptureMicrobatch(
                batch.request_ordinals,
                keys,
                terminal,
                before,
                after,
                processed_token_count=local * 6 * 3,
                padded_token_count=local * 6,
            )

        return plan, calls, forward

    def test_behavioral_capture_restores_request_order_and_r0_zero(self) -> None:
        plan, calls, forward = self._capture()
        capture = accumulate_streaming_capture(plan, forward)
        self.assertEqual(len(calls), 3)
        self.assertEqual(capture.physical_forward_count, 3)
        self.assertTrue(
            torch.equal(
                capture.terminal_z,
                torch.tensor(
                    [[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]]
                ),
            )
        )
        # 0.5*((100+ordinal)+ordinal) in the first feature.
        self.assertTrue(
            torch.equal(
                capture.keys_by_layer[4][0],
                torch.tensor([50.0, 51.0, 52.0, 53.0, 54.0]),
            )
        )
        initial = initial_target_from_capture(capture)
        self.assertEqual(int(torch.count_nonzero(initial.residual)), 0)
        self.assertEqual(initial.capture_identity_sha256, capture.identity_sha256)

    def test_mid_microbatch_model_mutation_fails(self) -> None:
        plan, _calls, forward = self._capture(mutate_state_on_batch=1)
        with self.assertRaisesRegex(ODEBFStateError, "within a capture microbatch"):
            accumulate_streaming_capture(plan, forward)


class DynamicAndAccountingTests(unittest.TestCase):
    def test_ordinal_overlay_uses_global_request_mapping(self) -> None:
        class Toy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Identity()

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.layer(value)

        model = Toy()
        value = torch.zeros((3, 4, 2), dtype=torch.float32)
        residual = torch.tensor(
            [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]],
            dtype=torch.float32,
        )
        with OrdinalTargetActivationOverlay(
            model,
            "layer",
            residual=residual,
            row_request_ordinals=(2, 0, 1),
            padded_lookup_positions=(1, 2, 3),
        ) as overlay:
            observed = model(value)
        self.assertEqual(overlay.fire_count, 1)
        self.assertTrue(torch.equal(observed[0, 1], torch.tensor([30.0, 3.0])))
        self.assertTrue(torch.equal(observed[1, 2], torch.tensor([10.0, 1.0])))
        self.assertTrue(torch.equal(observed[2, 3], torch.tensor([20.0, 2.0])))
        self.assertEqual(int(torch.count_nonzero(observed)), 6)

    def test_bg_metric_and_target_update_apply_h_once(self) -> None:
        z0 = torch.tensor(
            [[3.0, 0.0, 0.0], [4.0, 2.0, 8.0]], dtype=torch.float32
        )
        order = canonical_hash({"request_order": [0, 1, 2]})
        metric = ScalableBatchGlobalMetric.from_z0(z0, order)
        self.assertEqual(metric.shared_speed, 5.0)
        gradient = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
        )
        target, velocity, alpha, receipt = target_update_from_existing_gradient(
            z0, gradient, metric
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(velocity.double())),
            5.0 * math.sqrt(3.0),
            places=5,
        )
        self.assertTrue(torch.allclose(target, z0 + 0.125 * velocity))
        self.assertAlmostEqual(
            alpha, float(-torch.sum(gradient.double() * (0.125 * velocity.double())))
        )
        self.assertEqual(receipt["h_application_count"], 1)

    def test_scalable_residual_r0_is_exact_zero(self) -> None:
        z0 = torch.randn((4, 100), dtype=torch.float32)
        residual = scalable_terminal_residual(
            z0.clone(), z0.clone(), canonical_hash({"order": list(range(100))})
        )
        self.assertEqual(residual.global_batch_size, 100)
        self.assertEqual(int(torch.count_nonzero(residual.residual)), 0)

    def test_waypoint_factor_binds_explicit_global_batch_rank(self) -> None:
        factor = WaypointFactor(
            "layers.4.weight",
            4,
            0,
            0,
            0,
            0.125,
            torch.ones((3, 100), dtype=torch.float32),
            torch.ones((2, 100), dtype=torch.float32),
            global_batch_size=100,
        )
        self.assertEqual(factor.left.shape[1], 100)
        with self.assertRaisesRegex(ODEBFContractError, "global batch"):
            WaypointFactor(
                "layers.4.weight",
                4,
                0,
                0,
                0,
                0.125,
                torch.ones((3, 99), dtype=torch.float32),
                torch.ones((2, 99), dtype=torch.float32),
                global_batch_size=100,
            )

    def test_dynamic_refresh_requires_eight_chained_states(self) -> None:
        ledger = DynamicRefreshLedger()
        for step in range(8):
            digest = lambda label: canonical_hash({"label": label, "step": step})
            ledger.record(
                step_index=step,
                accepted_state_sha256=digest("state"),
                target_sha256=digest("target"),
                key_inventory_sha256=digest("key"),
                slope_sha256=digest("slope"),
                field_sha256=digest("field"),
                field_invocation_index=step + 1,
            )
        receipt = ledger.finalize()
        self.assertEqual(receipt["field_build_count"], 8)
        self.assertEqual(receipt["static_split_count"], 0)

    def test_static_state_reuse_fails(self) -> None:
        ledger = DynamicRefreshLedger()
        state = canonical_hash({"state": "frozen"})
        digest = lambda label, step: canonical_hash({"label": label, "step": step})
        ledger.record(
            step_index=0,
            accepted_state_sha256=state,
            target_sha256=digest("target", 0),
            key_inventory_sha256=digest("key", 0),
            slope_sha256=digest("slope", 0),
            field_sha256=digest("field", 0),
            field_invocation_index=1,
        )
        with self.assertRaisesRegex(ODEBFStateError, "did not advance"):
            ledger.record(
                step_index=1,
                accepted_state_sha256=state,
                target_sha256=digest("target", 1),
                key_inventory_sha256=digest("key", 1),
                slope_sha256=digest("slope", 1),
                field_sha256=digest("field", 1),
                field_invocation_index=2,
            )

    def test_logical_and_physical_counter_arithmetic(self) -> None:
        expected = expected_streaming_gradient_counts(
            request_count=100, microbatch_size=4
        )
        self.assertEqual(expected["physical_microbatch_count_per_group"], 25)
        self.assertEqual(expected["physical_target_backward_calls"], 200)
        self.assertEqual(expected["physical_slope_backward_calls"], 200)
        self.assertEqual(expected["accepted_materialization_count"], 8)
        ledger = ScalableComputeLedger()
        ledger.increment(
            "target",
            logical_forward_groups=1,
            model_forward_calls=25,
            physical_microbatch_graphs=25,
            autograd_invocations=25,
            backward_calls=25,
            target_backward_calls=25,
        )
        self.assertEqual(ledger.totals()["logical_forward_groups"], 1)
        self.assertEqual(ledger.totals()["model_forward_calls"], 25)

    def test_entry_relative_materialization_and_restore(self) -> None:
        class Toy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
                self.layer.weight = torch.nn.Parameter(
                    torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
                )

        model = Toy()
        entry = {"layer.weight": model.layer.weight.detach().clone()}
        left = torch.ones((3, 10), dtype=torch.float32)
        right = torch.full((2, 10), 0.1, dtype=torch.float32)
        first = WaypointFactor(
            "layer.weight", 4, 0, 0, 0, 0.125, left, right
        )
        second = WaypointFactor(
            "layer.weight", 4, 0, 1, 0, 0.125, left * 0.5, right
        )
        materializer = AcceptedPhysicalStateMaterializer(model, entry)
        materializer.materialize({"layer.weight": (first,)}, transition_index=1)
        materializer.materialize(
            {"layer.weight": (first, second)}, transition_index=2
        )
        expected, _ = assemble_effective_bf16(
            entry["layer.weight"], (first, second), row_block=64
        )
        self.assertEqual(tensor_sha256(model.layer.weight), tensor_sha256(expected))
        self.assertEqual(materializer.dense_assembly_count, 2)
        self.assertEqual(materializer.hot_hook_dense_assembly_count, 0)
        receipt = materializer.restore()
        self.assertEqual(
            tensor_sha256(model.layer.weight), tensor_sha256(entry["layer.weight"])
        )
        self.assertTrue(receipt["pointer_identity_preserved"])

    def test_atomic_runtime_has_no_history_append_or_sequential_dispatch(self) -> None:
        source = inspect.getsource(experiment)
        tree = ast.parse(source)
        append_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "record")
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in ("history", "accepted_history")
        ]
        self.assertEqual(append_calls, [])
        self.assertNotIn("P1R20", source)
        self.assertNotIn("sequential_batch=1", source)

    def test_official_native_calls_pinned_entry_with_hparams(self) -> None:
        class Toy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones((2, 2)))

        model = Toy()
        entry = model.weight.detach().clone()
        hparams = SimpleNamespace(marker="PINNED")
        requests = tuple(
            {"request_sha256": canonical_hash({"ordinal": ordinal})}
            for ordinal in range(10)
        )

        def fake_apply(*args, **kwargs):
            self.assertIs(args[3], hparams)
            with torch.no_grad():
                model.weight.add_(1.0)
            return model, {"weight": entry.clone()}

        alpha_main = types.ModuleType("easyeditor.models.alphaedit.AlphaEdit_main")
        alpha_main.apply_AlphaEdit_to_model = mock.Mock(side_effect=fake_apply)
        easyeditor = types.ModuleType("easyeditor")
        models = types.ModuleType("easyeditor.models")
        alphaedit = types.ModuleType("easyeditor.models.alphaedit")
        alphaedit.AlphaEdit_main = alpha_main
        modules = {
            "easyeditor": easyeditor,
            "easyeditor.models": models,
            "easyeditor.models.alphaedit": alphaedit,
            "easyeditor.models.alphaedit.AlphaEdit_main": alpha_main,
        }
        with mock.patch.dict(sys.modules, modules):
            payload, originals = run_official_native_apply(
                model,
                object(),
                requests,
                hparams,
                touched={"weight": model.weight},
            )
        self.assertTrue(payload["direct_z_semantics"])
        self.assertEqual(tensor_sha256(originals["weight"]), tensor_sha256(entry))


if __name__ == "__main__":
    unittest.main()
