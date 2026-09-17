"""Runtime/controller CPU boundary mocks; no actual model or GPU execution.

Runtime.__init__ is never called. Five 2x2 CPU weights replace the Llama state,
native fitting/scoring are synthetic stubs, and evidence writers stay in RAM.
The native adapter's schema validator is inspected without calling compute_z.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import types
import unittest
from unittest.mock import patch
import weakref

import torch

from . import runtime, metrics
from .controller import Controller, Limits, BudgetExceeded, run_arm
from .native import GeneralizedNativeFitter, FitBoundary


class MemoryPath:
    def __init__(self, text="memory-runtime"):
        self.text = text
    def __truediv__(self, name):
        return MemoryPath(self.text + "/" + name)
    def mkdir(self, **unused):
        return None
    def __str__(self):
        return self.text


class TinyWeights(torch.nn.Module):
    """Parameter container only: no forward method is implemented or called."""
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict({str(l): torch.nn.Linear(2, 2, bias=False)
                                          for l in runtime.LAYERS})
        self.other = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.register_buffer("fixed_buffer", torch.tensor([2.0]))
        with torch.no_grad():
            for l in runtime.LAYERS:
                self.layers[str(l)].weight.fill_(float(l))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


def online_score(current, past):
    def canonical(value):
        return dict(E=.01, strict_ids=[1], pair_ids=[1], pair_status="AVAILABLE",
                    token_margins={1: .2}, pair_margins={1: .1})
    return dict(controller=dict(base_kl=.1, training_e=.01, current_strict=frozenset({1}),
                current_pair=frozenset({1}), past_h=.01 if past else None,
                past_strict=frozenset({1}) if past else frozenset(),
                past_pair=frozenset({1}) if past else None, canonical_e=.01,
                current_token_margins={1: .2}, current_pair_margins={1: .1},
                past_token_margins={1: .2} if past else {},
                past_pair_margins={1: .1} if past else {}), official_P_N_access=0)


class RuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rng = {"cpu": [1, 2, 3], "cuda": "NOT_USED_CPU_FIXTURE"}
        self.stack.enter_context(patch.object(runtime, "capture_rng", side_effect=lambda: copy.deepcopy(self.rng)))
        self.stack.enter_context(patch.object(runtime, "restore_rng", side_effect=self.restore_rng))
        self.saved = {}
        def save(path, value):
            key = str(path)
            self.assertNotIn(key, self.saved)
            self.saved[key] = copy.deepcopy(value)
            return {"path": key, "sha256": "CPU_MOCK_NO_DISK", "bytes": 0}
        self.stack.enter_context(patch.object(runtime, "save", side_effect=save))
        self.stack.enter_context(patch.object(runtime, "tensor_save", side_effect=save))

    def restore_rng(self, value):
        self.rng = copy.deepcopy(value)

    def make_runtime(self):
        rt = runtime.Runtime.__new__(runtime.Runtime)
        rt.model = TinyWeights()
        rt.params = dict(rt.model.named_parameters())
        rt.W = {l: rt.model.layers[str(l)].weight for l in runtime.LAYERS}
        rt.context = [["{}"]]
        rt.module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=copy.deepcopy(rt.context), COV_CACHE={})
        rt.M = {l: torch.zeros(1, 2, 2) for l in runtime.LAYERS}
        rt.P = {l: torch.eye(2).unsqueeze(0) for l in runtime.LAYERS}
        rt.nonselected = {"other": (rt.model.other, rt.model.other.data_ptr(), rt.model.other._version)}
        rt.buffers = {name: (buf, buf.data_ptr(), buf._version) for name, buf in rt.model.named_buffers()}
        rt.hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks), tuple(module._backward_hooks))
                    for name, module in rt.model.named_modules()}
        rt._pool = weakref.WeakValueDictionary()
        rt.cpuW, rt.whash = {}, {}
        for l, weight in rt.W.items():
            rt.cpuW[l], rt.whash[l] = rt.intern(weight.detach().clone())
        rt.mhash = {l: runtime.tensor_sha(value) for l, value in rt.M.items()}
        rt.phash = {l: runtime.tensor_sha(value) for l, value in rt.P.items()}
        rt.timing = {}
        rt.tok = types.SimpleNamespace(padding_side="right")
        rt.hp = {l: types.SimpleNamespace(layers=[l]) for l in runtime.LAYERS}
        rt.metrics = types.SimpleNamespace(score=online_score)
        def finalize(model, tok, requests, bindings):
            result = []
            for l, hp, history, projector in bindings:
                history.add_(1)
                result.append({"layer": l, "history_append": 1})
            return result
        rt.fitter = types.SimpleNamespace(finalize=finalize)
        rt._refresh_versions()
        rt.guard()
        return rt

    def make_backend(self, rt):
        backend = runtime.Backend.__new__(runtime.Backend)
        backend.rt, backend.out = rt, MemoryPath()
        backend.current = [{"case_id": 1, "requested_rewrite": {"prompt": "{}", "subject": "X"}}]
        backend.past = []
        backend.fit_receipts, backend.score_receipts = [], []
        backend.history_receipt = None
        backend.namespace = "CPU_RUNTIME_CLOSURE"
        def fit(this, records, layer, instrument=True):
            before = this.snapshot()
            value = this.cpuW[layer] + 1.0
            this.adopt_native_weights({layer: value}, before["rng"])
            return dict(weight=value, captures={"synthetic_only": True}, receipt=dict(
                history_append=0, layer=layer, adam_updates=0, compute_z=len(records),
                compute_ks=1, solve=1, entry_weight_sha256=before["whash"][layer],
                endpoint_weight_sha256=this.whash[layer]))
        rt.fit = types.MethodType(fit, rt)
        return backend

    def test_fp32_zero_and_one_are_exact_endpoints(self):
        entry = torch.tensor([1e12, -1.0], dtype=torch.float32)
        native = torch.tensor([-1e12, 7.0], dtype=torch.float32)
        self.assertIs(runtime.fp32_gate(entry, native, 0.0), entry)
        self.assertIs(runtime.fp32_gate(entry, native, 1.0), native)
        for gate in (.5, .75, .613123456789):
            expected = entry + gate * (native - entry)
            self.assertTrue(torch.equal(runtime.fp32_gate(entry, native, gate), expected))

    def test_gate_rejects_nonfinite_bounds_dtype_shape(self):
        x = torch.zeros(2)
        for gate in (float("nan"), float("inf"), -1e-18, 1.001):
            with self.assertRaises(ValueError):
                runtime.fp32_gate(x, x, gate)
        with self.assertRaises(ValueError):
            runtime.fp32_gate(x.double(), x, .5)
        with self.assertRaises(ValueError):
            runtime.fp32_gate(x, torch.zeros(3), .5)

    def test_snapshot_structural_sharing_and_exact_restore(self):
        rt = self.make_runtime()
        snapshot = rt.snapshot()
        self.assertIs(snapshot["W"][4], rt.cpuW[4])
        rt.adopt_native_weights({4: rt.cpuW[4] + 1}, snapshot["rng"])
        changed = rt.snapshot()
        self.assertIs(changed["W"][5], snapshot["W"][5])
        self.assertIsNot(changed["W"][4], snapshot["W"][4])
        self.assertTrue(torch.equal(snapshot["W"][4], torch.full((2, 2), 4.0)))
        rt.restore(snapshot)
        self.assertEqual(rt.state(), snapshot["state"])
        self.assertTrue(torch.equal(rt.W[4], snapshot["W"][4]))

    def test_intern_same_bytes_returns_same_immutable_tensor(self):
        rt = self.make_runtime()
        again, digest = rt.intern(rt.cpuW[4].clone())
        self.assertIs(again, rt.cpuW[4])
        self.assertEqual(digest, rt.whash[4])

    def test_snapshot_inplace_mutation_is_detected(self):
        rt = self.make_runtime()
        snapshot = rt.snapshot()
        snapshot["W"][4].add_(1)
        with self.assertRaisesRegex(AssertionError, "SNAPSHOT"):
            rt.validate_snapshot(snapshot)
        with self.assertRaisesRegex(AssertionError, "CPU_SNAPSHOT_MUTATION"):
            rt.guard()

    def test_selected_nonselected_projector_history_buffer_guards(self):
        mutators = [
            (lambda rt: rt.W[4].add_(1), "SELECTED"),
            (lambda rt: rt.model.other.add_(1), "NONSELECTED"),
            (lambda rt: rt.P[4].add_(1), "P_MUTATION"),
            (lambda rt: rt.M[4].add_(1), "M_MUTATION"),
            (lambda rt: rt.model.fixed_buffer.add_(1), "BUFFER"),
        ]
        for mutate, reason in mutators:
            rt = self.make_runtime()
            with torch.no_grad():
                mutate(rt)
            with self.assertRaisesRegex(AssertionError, reason):
                rt.guard()

    def test_backend_snapshot_validation_matches_controller_arguments(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        snapshot = backend.snapshot()
        backend.validate_snapshot(snapshot, backend.state_token(), backend.history_token())
        with self.assertRaises(AssertionError):
            backend.validate_snapshot(snapshot, "wrong-state", backend.history_token())
        with self.assertRaises(AssertionError):
            backend.validate_snapshot(snapshot, backend.state_token(), "wrong-history")

    def test_native_receipt_existing_history_key_does_not_duplicate(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        steps = backend.fit_native(4)
        self.assertEqual(steps, 0)
        receipt = self.saved["memory-runtime/fits/000-L4/receipt.json"]
        self.assertEqual(receipt["history_append"], 0)
        self.assertFalse(receipt["W_saved"])
        self.assertNotIn("weight", self.saved["memory-runtime/fits/000-L4/native-evidence.pt"])

    def test_history_only_changes_history_token_and_preserves_old_snapshots(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        snapshot = backend.snapshot()
        state_token, history_token = backend.state_token(), backend.history_token()
        backend.finalize_history(runtime.LAYERS)
        self.assertEqual(backend.state_token(), state_token)
        self.assertNotEqual(backend.history_token(), history_token)
        self.assertTrue(all(torch.count_nonzero(snapshot["M"][l]) == 0 for l in runtime.LAYERS))
        self.assertTrue(all(torch.equal(rt.M[l], torch.ones(1, 2, 2)) for l in runtime.LAYERS))
        with self.assertRaisesRegex(AssertionError, "HISTORY_ONCE"):
            backend.finalize_history(runtime.LAYERS)
        rt.restore(snapshot)
        self.assertEqual(backend.history_token(), history_token)

    def test_real_backend_controller_grid_cache_and_single_commit(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        controller = Controller(backend, (4, 8), namespace=backend.namespace, history_layers=runtime.LAYERS)
        selected, _ = run_arm(controller, "G48")
        self.assertEqual(controller.counts["l4_fits"], 1)
        self.assertEqual(controller.counts["suffix_fits"], 2)
        self.assertTrue(all(torch.count_nonzero(rt.M[l]) == 0 for l in runtime.LAYERS))
        controller.seal_selection(selected)
        controller.finalize(selected)
        self.assertEqual(controller.counts["commits"], 1)
        self.assertEqual(self.saved["memory-runtime/history.json"]["appends"], 5)

    def test_partial_budget_with_real_backend_retains_prefix_and_rolls_back(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        limits = Limits(max_suffix_fits=1, reserve_suffix_fits_for_pruning=0)
        controller = Controller(backend, runtime.LAYERS, namespace=backend.namespace,
                                history_layers=runtime.LAYERS, limits=limits)
        with self.assertRaises(BudgetExceeded):
            controller.evaluate((.6, 1., 1., 1., 1.))
        self.assertEqual(controller.counts["suffix_fits"], 1)
        self.assertEqual(controller.counts["endpoints"], 0)
        self.assertEqual(backend.state_token(), controller.entry_token)
        self.assertEqual(backend.history_token(), controller.entry_history)
        self.assertEqual(len(controller.fit_cache), 2)

    def test_gating_changes_one_layer_preserves_history(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        before = backend.snapshot()
        backend.fit_native(8)
        native = backend.snapshot()
        backend.apply_gate(8, before, native, .5)
        self.assertEqual(backend.changed_layers(before, backend.snapshot()), (8,))
        self.assertEqual(backend.history_token(), runtime.digest(before["state"]["M"]))
        self.assertTrue(torch.equal(rt.W[8], torch.full((2, 2), 8.5)))

    def test_generalized_native_check_all_five_physical_layers(self):
        rt = self.make_runtime()
        fitter = GeneralizedNativeFitter.__new__(GeneralizedNativeFitter)
        fitter.torch, fitter.module, fitter.contexts = torch, rt.module, rt.context
        for layer in runtime.LAYERS:
            hp = types.SimpleNamespace(layers=[layer], blue=True, L2=1, v_num_grad_steps=25,
                v_lr=.1, v_weight_decay=.5, clamp_norm_factor=.75, kl_factor=.0625,
                rewrite_module_tmp="layers.{}")
            name, weight = fitter._check(rt.model, rt.tok, hp, rt.M[layer], rt.P[layer], layer)
            self.assertEqual(name, f"layers.{layer}.weight")
            self.assertIs(weight, rt.W[layer])
            hp.layers = [4, 8]
            with self.assertRaises(FitBoundary):
                fitter._check(rt.model, rt.tok, hp, rt.M[layer], rt.P[layer], layer)

    def test_online_score_routes_native_E_not_canonical_and_no_dev(self):
        obj = metrics.OnlineMetrics.__new__(metrics.OnlineMetrics)
        calls = []
        obj.training_E = lambda records: (calls.append(("training", records)) or {"E": .02})
        def canonical(records):
            calls.append(("canonical", records))
            return dict(E=.8 if records == "current" else .03, strict_ids=[1], pair_ids=[1],
                        pair_status="AVAILABLE", token_margins={1: .2}, pair_margins={1: .1})
        obj.canonical = canonical
        obj.base = lambda role: (calls.append(("base", role)) or {"D": .1})
        result = obj.score("current", "past")
        self.assertEqual(result["controller"]["training_e"], .02)
        self.assertEqual(result["controller"]["canonical_e"], .8)
        self.assertEqual(result["controller"]["past_h"], .03)
        self.assertEqual(calls, [("training", "current"), ("canonical", "current"),
                                 ("canonical", "past"), ("base", "S64")])
        self.assertEqual(result["official_P_N_access"], 0)
        self.assertEqual(result["Dev_access"], 0)
        self.assertFalse(result["canonical_Current_mean_is_guard"])

    def test_normalized_native_requests_do_not_mutate_inputs(self):
        record = {"case_id": 1, "requested_rewrite": {"prompt": "{}", "subject": "X",
                  "target_new": {"str": "answer"}, "target_true": {"str": "old"}},
                  "paraphrase_prompts": ["must remain observer-only"]}
        before = copy.deepcopy(record)
        normalized = metrics.normalized_requests([record])
        self.assertEqual(record, before)
        self.assertEqual(normalized[0]["target_new"]["str"], " answer")
        self.assertNotIn("paraphrase_prompts", normalized[0])

    def test_snapshot_replacing_tensor_same_version_is_rejected(self):
        rt = self.make_runtime()
        backend = self.make_backend(rt)
        snap = backend.snapshot()
        expected_state, expected_history = backend.state_token(), backend.history_token()
        replacement = snap["W"][4].clone() + 1
        self.assertEqual(replacement._version, snap["W"][4]._version)
        snap["W"][4] = replacement
        with self.assertRaises(AssertionError):
            backend.validate_snapshot(snap, expected_state, expected_history)

    def test_selected_storage_replacement_is_rejected(self):
        rt = self.make_runtime()
        rt.W[4].data = rt.W[4].detach().clone() + 1
        with self.assertRaises(AssertionError):
            rt.guard()


if __name__ == "__main__":
    unittest.main()
