"""CPU fixtures of the native adapter only; no pretrained/GPU qualification."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.baselines import registry
from official.experiments.prepare import digest, file_sha
from official.runners.server1.native import NativeBindingError, NativeEngine


class FixtureModel(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        layers = [21] if method == "FT" else [4, 5]
        self.mapping = {f"model.layers.{layer}.mlp.down_proj.weight": torch.nn.Parameter(torch.ones(2, 3))
                        for layer in layers}
        self.mapping["model.other.weight"] = torch.nn.Parameter(torch.zeros(2, 3))
        self.config = SimpleNamespace(_name_or_path="fixture/llama")
        self.training = False

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter(self.mapping.items())

    def parameters(self, recurse=True):
        return iter(self.mapping.values())


def records():
    return [dict(case_id=index, occurrence_index=index + 1,
                 requested_rewrite=dict(prompt="{} writes", subject="subject",
                                        target_new={"str": "new"}, target_true={"str": "true"}))
            for index in range(100)]


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        self.temp.cleanup()

    def create(self, method="MEMIT", *, action=None, bad_sha=False, dtype=torch.float32, options=None,
               source_verified=False):
        model = FixtureModel(method)
        if dtype != torch.float32:
            model.mapping[next(iter(model.mapping))] = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
        module = SimpleNamespace(__name__="official.baselines.fixture." + method,
                                 __file__=__file__, COV_CACHE={"stale": torch.ones(1)},
                                 CONTEXT_TEMPLATES_CACHE=[["stale"]], GLOBAL_EDIT_COUNT=99,
                                 layer_stats=lambda *a, **kw: "must not collect")
        hp = SimpleNamespace(layers=[21] if method == "FT" else [4, 5],
                             rewrite_module_tmp="model.layers.{}.mlp.down_proj.weight" if method == "FT"
                             else "model.layers.{}.mlp.down_proj", mom2_n_samples=100000)
        C0 = {}
        for layer in (4, 5):
            path = self.root / f"layer{layer}.npz"
            np.savez(path, **{"mom2.count": np.array(7), "sample_size": np.array(100000),
                             "mom2.constructor": np.array("fixture.SecondMoment()"),
                             "mom2.mom2": np.eye(3, dtype=np.float32) * 7})
            C0[str(layer)] = dict(path=str(path), bytes=path.stat().st_size,
                                 sha256="0" * 64 if bad_sha else file_sha(path), layer=layer,
                                 module=f"model.layers.{layer}.mlp.down_proj",
                                 tensor_validation=dict(shape=[3, 3], dtype="float32", finite=True,
                                     stored_value="UNCENTERED_SECOND_MOMENT_SUM", sample_documents=100000,
                                     masked_token_vector_count=7))
        manifest = dict(assignment=dict(model="llama3", methods=["FT", "MEMIT", "MEMIT_FE"]),
                        model=dict(payload_verification="FRESH_FULL_SHA256", identity=dict(hidden=2, intermediate=3)),
                        C0=C0)

        def apply(model, tokenizer, requests, hparams, **kwargs):
            if method != "FT":
                module.CONTEXT_TEMPLATES_CACHE = [["{}"], ["native. {}"]]
            if method == "MEMIT":
                module.GLOBAL_EDIT_COUNT += 1
            if action:
                return action(model, requests, module)
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if "other" not in name:
                        value.add_(1)
                model.mapping["model.other.weight"].requires_grad_(True)
            return model, {name: value.clone() for name, value in model.named_parameters() if "other" not in name}

        opts = registry.call_options(method, "llama3") if options is None else options
        with patch("official.runners.server1.native.verify_manifest", return_value={}) as verification, \
             patch("official.runners.server1.native.registry.implementation", return_value=(module, object, apply)), \
             patch("official.runners.server1.native.registry.hparams", return_value=hp), \
             patch("official.runners.server1.native.registry.call_options", return_value=opts):
            engine = NativeEngine(model, object(), method, manifest, source_verified=source_verified)
            self.last_manifest_verification_call = verification.call_args
        self.engines.append(engine)
        return engine, model, module

    def test_exact_native_options_and_requests(self):
        options = registry.call_options("MEMIT", "llama3")
        self.assertEqual(options["beta_hse"], 0)
        self.assertFalse(options["save_weights"])
        self.assertFalse(options["copy"])
        self.assertIsNone(options["cache_template"])
        source = records()
        before = digest(source)
        self.assertIsInstance(registry.requests(source, "MEMIT", "llama3")[0]["target_new"], dict)
        self.assertIsInstance(registry.requests(source, "MEMIT_FE", "llama3")[0]["target_new"], str)
        self.assertEqual(registry.requests(source, "FT", "llama3")[0]["prompt"], "subject writes")
        self.assertEqual(digest(source), before)

    def test_MEMIT_cold_cache_and_original_FP32_sum_division(self):
        engine, model, module = self.create()
        self.assertEqual(module.GLOBAL_EDIT_COUNT, 0)
        self.assertIsNone(module.CONTEXT_TEMPLATES_CACHE)
        self.assertEqual(set(module.COV_CACHE), {("fixture_llama", "model.layers.4.mlp.down_proj"),
                                               ("fixture_llama", "model.layers.5.mlp.down_proj")})
        for value in module.COV_CACHE.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.equal(value, torch.eye(3)))
        self.assertEqual(engine.cache_c, {})
        self.assertIsNone(engine.contexts()["CONTEXT_TEMPLATES_CACHE"])

    def test_FE_contexts_have_no_history_or_global_counter(self):
        engine, model, module = self.create("MEMIT_FE")
        receipt = engine.apply(records())
        self.assertFalse(receipt["native_history"])
        self.assertEqual(engine.cache_c, {})
        self.assertNotIn("GLOBAL_EDIT_COUNT", engine.contexts())

    def test_FT_never_loads_C0_or_native_context(self):
        with patch("official.runners.server1.native.np.load", side_effect=AssertionError("FT loadedC0")):
            engine, model, module = self.create("FT")
        self.assertEqual(list(engine.selected_weights), ["model.layers.21.mlp.down_proj.weight"])
        self.assertEqual(engine.cache_c, {})
        # FT has no context even if a fixture carries irrelevant globals.
        module.CONTEXT_TEMPLATES_CACHE = None
        engine.apply(records())
        self.assertEqual(engine.contexts()["CONTEXT_TEMPLATES_CACHE"], None)

    def test_failclosed_C0_cache_miss(self):
        engine, model, module = self.create()
        with self.assertRaisesRegex(NativeBindingError, "RECOMPUTE_FORBIDDEN"):
            module.layer_stats(None)

    def test_bad_C0_SHA_rejected(self):
        with self.assertRaisesRegex(NativeBindingError, "C0_SOURCE_SHA_CHANGED"):
            self.create(bad_sha=True)

    def test_native_selected_FP32_required(self):
        with self.assertRaisesRegex(NativeBindingError, "NATIVE_SELECTED_FP32_SHAPE"):
            self.create(dtype=torch.float64)

    def test_MEMIT_weights_copy_not_H_and_freeze_eval(self):
        engine, model, module = self.create()
        receipt = engine.apply(records())
        self.assertEqual(receipt["batch"], 1)
        self.assertEqual(receipt["requests"], 100)
        self.assertEqual(receipt["cache_c"], {})
        self.assertEqual(module.GLOBAL_EDIT_COUNT, 1)
        self.assertFalse(model.training)
        self.assertTrue(all(not value.requires_grad and value.grad is None for value in model.parameters()))
        self.assertTrue(all(torch.equal(value, torch.ones(2, 3) * 2) for value in engine.selected_weights.values()))

    def test_nonselected_version_guard(self):
        def mutate(model, requests, module):
            with torch.no_grad():
                model.mapping["model.other.weight"].add_(1)
            return model, {}
        engine, model, module = self.create(action=mutate)
        with self.assertRaisesRegex(NativeBindingError, "NONSELECTED_VERSION_CHANGED"):
            engine.apply(records())

    def test_nonselected_pointer_guard(self):
        def mutate(model, requests, module):
            model.mapping["model.other.weight"] = torch.nn.Parameter(torch.zeros(2, 3))
            return model, {}
        engine, model, module = self.create(action=mutate)
        with self.assertRaisesRegex(NativeBindingError, "PARAMETER_IDENTITY_CHANGED"):
            engine.apply(records())

    def test_nonfinite_selected_guard(self):
        def mutate(model, requests, module):
            with torch.no_grad():
                next(iter(model.mapping.values())).fill_(float("nan"))
            return model, {}
        engine, model, module = self.create(action=mutate)
        with self.assertRaisesRegex(NativeBindingError, "SELECTED_NONFINITE"):
            engine.apply(records())

    def test_same_cumulative_model_required(self):
        engine, model, module = self.create(action=lambda *args: (object(), {}))
        with self.assertRaisesRegex(NativeBindingError, "SAME_CUMULATIVE_MODEL"):
            engine.apply(records())

    def test_original_exception_preserved_with_guard_note(self):
        original = ValueError("actual native cause")
        def fail(model, requests, module):
            with torch.no_grad():
                model.mapping["model.other.weight"].add_(1)
            raise original
        engine, model, module = self.create(action=fail)
        with self.assertRaises(ValueError) as caught:
            engine.apply(records())
        self.assertIs(caught.exception, original)
        self.assertIn("NONSELECTED_VERSION_CHANGED", caught.exception.__notes__[0])
        self.assertTrue(all(not value.requires_grad and value.grad is None for value in model.parameters()))

    def test_entry_change_is_not_silently_accepted(self):
        engine, model, module = self.create()
        with torch.no_grad():
            next(iter(model.mapping.values())).add_(3)
        with self.assertRaisesRegex(NativeBindingError, "ENTRY_STATE_CHANGED"):
            engine.apply(records())

    def test_context_restore_for_B2_to_B3(self):
        engine, model, module = self.create()
        engine.apply(records())
        engine.apply(records())
        contexts = engine.contexts()
        saved = {name: value.detach().clone() for name, value in engine.selected_weights.items()}
        expected = engine.apply(records())["post_selected_sha256"]
        resumed, resumed_model, resumed_module = self.create()
        with torch.no_grad():
            for name, value in saved.items():
                resumed.selected_weights[name].copy_(value)
        resumed.restore_contexts(contexts)
        actual = resumed.apply(records())["post_selected_sha256"]
        self.assertEqual(actual, expected)
        self.assertEqual(resumed_module.GLOBAL_EDIT_COUNT, 3)

    def test_context_digest_and_scope_rejection(self):
        engine, model, module = self.create()
        changed = engine.contexts()
        changed["successful_calls"] = 3
        with self.assertRaisesRegex(NativeBindingError, "RESTORE_CONTEXT_DIGEST"):
            engine.restore_contexts(changed)
        changed = engine.contexts()
        changed["method"] = "MEMIT_FE"
        unsigned = dict(changed)
        unsigned.pop("identity_sha256")
        changed["identity_sha256"] = digest(unsigned)
        with self.assertRaisesRegex(NativeBindingError, "RESTORE_CONTEXT_SCOPE"):
            engine.restore_contexts(changed)

    def test_batch100_and_no_batch21(self):
        engine, model, module = self.create()
        with self.assertRaisesRegex(NativeBindingError, "BATCH100_REQUIRED"):
            engine.apply(records()[:2])
        engine.successful_calls = 20
        with self.assertRaisesRegex(NativeBindingError, "NO_BATCH21"):
            engine.apply(records())

    def test_close_removes_only_our_sentinel(self):
        engine, model, module = self.create()
        engine.close()
        self.assertEqual(module.layer_stats(None), "must not collect")
        with self.assertRaisesRegex(NativeBindingError, "ENGINE_CLOSED"):
            engine.apply(records())

    def test_unknown_source_default_verifies_preparation(self):
        engine, model, module = self.create()
        self.assertEqual(self.last_manifest_verification_call.kwargs, {"verify_preparation_source": True})
        self.assertFalse(engine.source_verified)

    def test_separately_verified_frozen_source_receipt(self):
        engine, model, module = self.create(source_verified=True)
        self.assertEqual(self.last_manifest_verification_call.kwargs, {"verify_preparation_source": False})
        receipt = engine.apply(records())
        self.assertTrue(receipt["frozen_source_verified_by_caller"])
        self.assertTrue(receipt["native_source_sha256"])

    def test_native_source_option_boolean_only(self):
        with self.assertRaisesRegex(NativeBindingError, "NATIVE_SOURCE_VERIFICATION_FLAG_TYPE"):
            self.create(source_verified="verified")


if __name__ == "__main__":
    unittest.main()
