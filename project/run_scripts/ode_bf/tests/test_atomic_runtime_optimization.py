from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.atomic_runtime_optimization import (
    AcceptedPhysicalStateMaterializer,
    capture_physical_state,
)
from project.run_scripts.ode_bf.functional import (
    CachedBF16FunctionalTrial,
    CumulativeBF16FunctionalTrial,
    WaypointFactor,
    tensor_sha256,
)
from project.run_scripts.ode_bf import p1_atomic_runtime_optimization as runtime
from project.run_scripts.ode_bf.p1_atomic_runtime_optimization_panel import (
    P1R22_LOCK_FILE,
    validate_p1r22_lock,
)
from project.run_scripts.ode_bf.target_new_nll import (
    RoutingObjective,
    build_target_new_objective_batch_plan,
    evaluate_routing_objective,
)
from project.run_scripts.ode_bf.tests.test_target_new_nll import (
    _ScaledToyCausalLM,
    _Tokenizer,
    _configure,
    _contexts,
    _requests,
)


class _TwoLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(7, 9, bias=False, dtype=torch.bfloat16)
        self.second = torch.nn.Linear(9, 8, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.nn.functional.silu(self.first(value)))


def _factor(
    name: str,
    layer: int,
    shape: tuple[int, int],
    *,
    step: int,
) -> WaypointFactor:
    generator = torch.Generator().manual_seed(100 + step + layer)
    out_features, in_features = shape
    return WaypointFactor(
        name,
        layer,
        0,
        step,
        0,
        0.125,
        torch.randn((out_features, 10), generator=generator) * 0.02,
        torch.randn((in_features, 10), generator=generator) * 0.02,
    )


class AtomicRuntimeOptimizationTests(unittest.TestCase):
    def test_request_microbatch_objective_and_gradient_match_singletons(self) -> None:
        tokenizer = _Tokenizer()
        requests = _requests(new_lengths=(1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
        model = _ScaledToyCausalLM()
        _configure(model, tokenizer, requests, new_logit=1.75)
        singleton = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=_contexts(),
            gradient_input=model.scale,
            request_microbatch_size=1,
        )
        batched = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=_contexts(),
            gradient_input=model.scale,
            request_microbatch_size=2,
        )
        torch.testing.assert_close(batched.loss, singleton.loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            batched.per_request_values,
            singleton.per_request_values,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            batched.input_gradient,
            singleton.input_gradient,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(batched.target_span_sha256, singleton.target_span_sha256)
        self.assertEqual(batched.model_forward_count, 5)
        self.assertEqual(batched.backward_count, 5)
        self.assertEqual(batched.processed_token_count, singleton.processed_token_count)

    def test_cached_bucketed_plan_is_shared_and_matches_direct_batch(self) -> None:
        tokenizer = _Tokenizer()
        requests = _requests(new_lengths=(1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
        model = _ScaledToyCausalLM()
        _configure(model, tokenizer, requests, new_logit=1.75)
        plan = build_target_new_objective_batch_plan(
            model,
            tokenizer,
            requests,
            contexts=_contexts(),
            request_microbatch_size=2,
        )
        direct = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=_contexts(),
            gradient_input=model.scale,
            request_microbatch_size=2,
        )
        cached = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=_contexts(),
            gradient_input=model.scale,
            request_microbatch_size=2,
            batch_plan=plan,
        )
        torch.testing.assert_close(cached.loss, direct.loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            cached.per_request_values, direct.per_request_values, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            cached.input_gradient, direct.input_gradient, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(cached.target_span_sha256, direct.target_span_sha256)
        self.assertEqual(plan.raw_free_payload()["old_target_access_count"], 0)
        self.assertEqual(plan.raw_free_payload()["physical_batch_count"], 5)
        self.assertNotEqual(plan.length_bucket_request_order, tuple(range(10)))

    def test_entry_relative_materialization_matches_virtual_at_multiple_states(self) -> None:
        torch.manual_seed(19)
        model = _TwoLayer()
        entry = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        factors = {
            "first.weight": (
                _factor("first.weight", 4, (9, 7), step=0),
                _factor("first.weight", 4, (9, 7), step=1),
            ),
            "second.weight": (
                _factor("second.weight", 5, (8, 9), step=0),
                _factor("second.weight", 5, (8, 9), step=1),
            ),
        }
        value = torch.randn((4, 7), dtype=torch.bfloat16)
        virtual = copy.deepcopy(model)
        with CumulativeBF16FunctionalTrial(virtual, factors, row_block=3):
            expected = virtual(value)
        materializer = AcceptedPhysicalStateMaterializer(model, entry, row_block=3)
        receipt = materializer.materialize(factors, transition_index=2)
        observed = model(value)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(receipt["weight_count"], 2)
        self.assertEqual(
            set(receipt["realized_bf16_step_energy"]), set(entry)
        )
        self.assertEqual(
            set(receipt["cumulative_bf16_capacity"]), set(entry)
        )
        accounting = materializer.raw_free_payload()
        self.assertEqual(accounting["dense_assembly_count"], 2)
        self.assertEqual(accounting["full_weight_hash_count"], 2)
        self.assertEqual(accounting["hot_hook_dense_assembly_count"], 0)
        self.assertEqual(accounting["hot_hook_full_weight_hash_count"], 0)
        restore = materializer.restore()
        for name, parameter in model.named_parameters():
            self.assertEqual(tensor_sha256(parameter), tensor_sha256(entry[name]))
        self.assertEqual(restore["persistent_commit_count"], 0)

    def test_materialization_is_not_incremental_bf16_accumulation(self) -> None:
        source = inspect.getsource(AcceptedPhysicalStateMaterializer.materialize)
        self.assertIn("self._entry[name]", source)
        self.assertNotIn("assemble_effective_bf16(\n                    parameter", source)
        self.assertNotIn("CumulativeBF16FunctionalTrial", source)

    def test_capture_api_is_one_partial_forward_and_has_no_virtual_trial(self) -> None:
        source = inspect.getsource(capture_physical_state)
        self.assertIn("TraceDict", source)
        self.assertIn("stop=True", source)
        self.assertIn("physical_forward_count", source)
        self.assertNotIn("CumulativeBF16FunctionalTrial", source)
        self.assertNotIn("torch.cuda.empty_cache", source)

    def test_cached_functional_trial_does_not_hash_in_forward_or_setup(self) -> None:
        enter_source = inspect.getsource(CachedBF16FunctionalTrial.__enter__)
        hook_source = inspect.getsource(CachedBF16FunctionalTrial._hook)
        self.assertIn("compute_sha256=False", enter_source)
        self.assertNotIn("tensor_sha256", enter_source)
        self.assertNotIn("assemble_effective_bf16", hook_source)
        self.assertNotIn("tensor_sha256", hook_source)

    def test_runtime_has_no_legacy_capacity_or_receipt_postmutation(self) -> None:
        source = inspect.getsource(runtime._run_arm)
        self.assertIn("_optimized_capacity_payload", source)
        self.assertNotIn("legacy._fixed_capacity_payload", source)
        self.assertNotIn('pending["progress"]', source)
        self.assertIn('"completion": "DELAYED_TO_NEXT_FIELD"', source)
        self.assertIn('"NEXT_FIELD_NOHOOK_NLL_REUSE"', source)

    def test_p1r22_lock_is_rooted_and_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "locks" / P1R22_LOCK_FILE
        value = json.loads(path.read_text(encoding="utf-8"))
        root = value.pop("root_digest")
        from project.run_scripts.ode_bf.contracts import canonical_hash

        self.assertEqual(root, canonical_hash(value))
        value["root_digest"] = root
        validate_p1r22_lock(value)


if __name__ == "__main__":
    unittest.main()
