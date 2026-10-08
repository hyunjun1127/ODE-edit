"""CPU connector fixtures, not GPT-J model/native GPU certification."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from official.baselines import registry
from official.experiments import checkpoint
from official.runners.server2 import native


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList()
        for _ in range(28):
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            # Deliberately tiny input width; _check_model is mocked only for
            # connector fixtures. No full GPT-J forward is performed.
            block.mlp.fc_out = torch.nn.Linear(2, 4096, bias=True)
            self.transformer.h.append(block)
        self.lm_head = torch.nn.Linear(2, 3)
        self.config = SimpleNamespace(_name_or_path="fixture/original", model_type="gptj",
                                     n_layer=28, n_embd=4096, n_inner=16384, vocab_size=50400)


def records():
    return [dict(case_id=i, requested_rewrite=dict(prompt="{} is", subject="A",
                 target_new={"str": " B"}, target_true={"str": " C"})) for i in range(100)]


class NativeConnectorTests(unittest.TestCase):
    def engine(self, method, *, nonselected_fault=False):
        model = FixtureModel()
        parser = registry.hparams(method, "gptj")
        module = SimpleNamespace(__name__="official.baselines.fixture", __file__=native.__file__,
                                 CONTEXT_TEMPLATES_CACHE=[["outside"]], COV_CACHE={"outside": 1},
                                 STATS_DIR="outside", P="outsideP", P_loaded=False,
                                 cache_c="outsideH", cache_c_new=False)
        calls = []
        context_calls = []

        def contexts(model_arg, tok):
            context_calls.append(model_arg.config._name_or_path)
            if module.CONTEXT_TEMPLATES_CACHE is None:
                module.CONTEXT_TEMPLATES_CACHE = [["{}"], ["native. {}"]]
            return module.CONTEXT_TEMPLATES_CACHE

        def apply(model_arg, tok, requests, hparams, **options):
            calls.append((deepcopy(requests), dict(options), model_arg.config._name_or_path))
            contexts(model_arg, tok) if method != "FT" else None
            with torch.no_grad():
                for value in native.selected_weights(model_arg, hparams, method).values():
                    value.add_(.01)
                if nonselected_fault:
                    model_arg.lm_head.bias.add_(1)
            if method == "ALPHAEDIT_BLUE":
                return model_arg, options["cache_c"] + 1
            if method in ("ALPHAEDIT", "SPHERE"):
                module.cache_c.add_(1)
            return model_arg, {}

        module.get_context_templates = contexts

        def projector(engine):
            full = torch.stack([torch.full((2, 2), float(i)) for i in range(6)])
            engine._projector = full[0:6:5] if method == "ALPHAEDIT_BLUE" else full
            engine.projector_slots = [0, 5] if method == "ALPHAEDIT_BLUE" else list(range(6))

        manifest = dict(model_revision=native.MODEL_REVISION, stats_root="/fixture/data/stats/gpt-j-6b/wikipedia_stats",
                        assets={"projector": dict(path="/fixture/projector.pt")})
        with patch.object(native, "_check_model", return_value=2), \
             patch.object(native, "_member", return_value=({}, Path("/fixture/projector.pt"))), \
             patch.object(registry, "implementation", return_value=(module, type(parser), apply)), \
             patch.object(registry, "hparams", return_value=deepcopy(parser)), \
             patch.object(native.NativeEngine, "_load_projector", projector), \
             patch.object(native.NativeEngine, "_load_native_covariance", return_value=None):
            engine = native.NativeEngine(model, object(), method, manifest)
        return engine, module, calls, context_calls

    def test_registry_parses_six_gptj_profiles(self):
        for method in native.METHODS:
            with self.subTest(method=method):
                hp = registry.hparams(method, "gptj")
                self.assertEqual(hp.layers, [21] if method == "FT" else
                                 [3, 8] if method == "ALPHAEDIT_BLUE" else list(range(3, 9)))
                self.assertEqual(hp.lr if method == "FT" else hp.v_lr, .0005 if method == "FT" else .5)
                self.assertTrue(registry.implementation(method, "gptj")[0].__name__.startswith("official.baselines."))

    def test_all_methods_apply_registry_request_and_native_options(self):
        for method in native.METHODS:
            with self.subTest(method=method):
                engine, module, calls, _ = self.engine(method)
                before_name = engine.model.config._name_or_path
                grad_before = [p.requires_grad for p in engine.model.parameters()]
                receipt = engine.apply(records(), 1)
                self.assertEqual(receipt["requests"], 100)
                self.assertEqual(calls[0][2], "gpt-j-6b")
                self.assertEqual(engine.model.config._name_or_path, before_name)
                self.assertEqual([p.requires_grad for p in engine.model.parameters()], grad_before)
                self.assertEqual(module.CONTEXT_TEMPLATES_CACHE, [["outside"]])
                self.assertEqual(module.COV_CACHE, {"outside": 1})
                self.assertEqual(module.cache_c, "outsideH")
                request = calls[0][0][0]
                self.assertEqual(request["target_new"], {"str": " B"} if method == "ALPHAEDIT_BLUE" else " B")
                self.assertEqual(request["prompt"], "A is" if method == "FT" else "{} is")
                self.assertIsNone(calls[0][1].get("cache_template"))
                self.assertEqual(bool(engine.history()), method in native.HISTORY_METHODS)
                if method != "ALPHAEDIT_BLUE":
                    self.assertFalse(calls[0][1]["copy"])
                    self.assertFalse(calls[0][1]["return_orig_weights"])

    def test_ft_tracks_native_fc_out_bias(self):
        engine, _, _, _ = self.engine("FT")
        self.assertEqual(set(engine.weights()), {"transformer.h.21.mlp.fc_out.weight",
                                                 "transformer.h.21.mlp.fc_out.bias"})
        self.assertEqual(engine.history(), {})

    def test_blue_physical_slots_and_returned_history_carry(self):
        engine, _, calls, _ = self.engine("ALPHAEDIT_BLUE")
        self.assertEqual(engine._projector[:, 0, 0].tolist(), [0, 5])
        engine.apply(records(), 1)
        self.assertEqual([value[0, 0].item() for value in engine.history().values()], [1, 1])
        engine.apply(records(), 2)
        self.assertEqual([value[0, 0].item() for value in engine.history().values()], [2, 2])
        self.assertEqual(calls[1][1]["cache_c"][0, 0, 0].item(), 1)

    def test_restore_exact_selected_weights_history_contexts_no_generation(self):
        for method in native.METHODS:
            with self.subTest(method=method):
                engine, _, _, _ = self.engine(method)
                engine.contexts()
                engine.apply(records(), 1)
                snapshot = dict(method=method, batch=1,
                                weights={k: v.detach().clone() for k, v in engine.weights().items()},
                                cache_c={k: v.clone() for k, v in engine.history().items()},
                                contexts=engine.contexts())
                restored, _, _, generated = self.engine(method)
                receipt = restored.restore(snapshot)
                self.assertFalse(receipt["context_regenerated"])
                self.assertEqual(generated, [])
                self.assertEqual(restored.contexts(), snapshot["contexts"])
                self.assertEqual(generated, [])
                for key, value in snapshot["weights"].items():
                    self.assertTrue(torch.equal(restored.weights()[key], value))
                for key, value in snapshot["cache_c"].items():
                    self.assertTrue(torch.equal(restored.history()[key], value))

    def test_context_global_isolation_and_only_one_cold_generation(self):
        engine, module, _, generated = self.engine("MEMIT")
        self.assertIsNone(engine.contexts(generate=False)["native_templates"])
        first = engine.contexts()
        second = engine.contexts()
        self.assertEqual(first, second)
        self.assertEqual(generated, ["gpt-j-6b"])
        self.assertEqual(module.CONTEXT_TEMPLATES_CACHE, [["outside"]])

    def test_invalid_batch_or_count_cannot_enter_native_writer(self):
        engine, _, calls, _ = self.engine("MEMIT")
        with self.assertRaisesRegex(ValueError, "NONCONTIGUOUS"):
            engine.apply(records(), 2)
        with self.assertRaisesRegex(ValueError, "BATCH_100"):
            engine.apply(records()[:2], 1)
        self.assertEqual(calls, [])

    def test_nonselected_mutation_rejected(self):
        engine, _, _, _ = self.engine("MEMIT", nonselected_fault=True)
        with self.assertRaisesRegex(ValueError, "NONSELECTED_PARAMETER"):
            engine.apply(records(), 1)
        self.assertEqual(engine.batch, 0)

    def test_restore_missing_native_history_or_context_rejected(self):
        engine, _, _, _ = self.engine("ALPHAEDIT")
        snapshot = dict(method="ALPHAEDIT", batch=0, weights=engine.weights(), cache_c={}, contexts={})
        with self.assertRaisesRegex(ValueError, "HISTORY_MAPPING"):
            engine.restore(snapshot)
        snapshot.update(cache_c=engine.history(), contexts=dict(method="ALPHAEDIT", native_templates=None))
        with self.assertRaisesRegex(ValueError, "CONTEXT_NOT_INITIALIZED"):
            engine.restore(snapshot)

    def test_common_checkpoint_B2_to_B3_fixture_weights_history_context_rng(self):
        identity = {key: "CPU_CONNECTOR_FIXTURE_ONLY" for key in checkpoint.IDENTITY_FIELDS}
        for method in native.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                engine, _, _, _ = self.engine(method)
                engine.contexts()
                for batch in range(3):
                    if batch:
                        engine.apply(records(), batch)
                    checkpoint.save(directory, batch=batch, weights=engine.weights(),
                        cache_c=engine.history(), contexts=engine.contexts(),
                        evaluation_cursor={"batch": batch}, identity=identity, method=method,
                        evaluation_complete=True)
                snapshot = checkpoint.load(directory, identity)
                next_rng = torch.rand(4)
                engine.apply(records(), 3)
                restored, _, _, generated = self.engine(method)
                restored.restore(snapshot)
                checkpoint.rng_restore(snapshot["rng"])
                self.assertTrue(torch.equal(next_rng, torch.rand(4)))
                restored.apply(records(), 3)
                # The fake native apply accesses an already restored context;
                # this is not a new context-generator invocation.
                self.assertEqual(generated, ["gpt-j-6b"] if method != "FT" else [])
                self.assertEqual(restored.contexts(), engine.contexts())
                for name, value in engine.weights().items():
                    self.assertTrue(torch.equal(restored.weights()[name], value))
                for name, value in engine.history().items():
                    self.assertTrue(torch.equal(restored.history()[name], value))
                self.assertEqual(len(list(Path(directory).glob("batch-*.pt"))), 1)

    def test_model_family_revision_structure_refused(self):
        model = FixtureModel()
        self.assertEqual(native._check_model(model, {"model_revision": native.MODEL_REVISION}), 16384)
        with self.assertRaisesRegex(ValueError, "REVISION"):
            native._check_model(model, {"model_revision": "different"})
        model.config.model_type = "gpt2"
        with self.assertRaisesRegex(ValueError, "STRUCTURE"):
            native._check_model(model, {"model_revision": native.MODEL_REVISION})


if __name__ == "__main__":
    unittest.main()
