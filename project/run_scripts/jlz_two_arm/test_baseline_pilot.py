"""Native adapter/wiring CPU tests only; no pretrained model or CUDA call."""
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from . import baseline_pilot as bp


def hparams():
    return SimpleNamespace(layers=[4, 5, 6, 7, 8], blue=False,
        v_num_grad_steps=25, v_lr=.1, v_weight_decay=.5, clamp_norm_factor=.75,
        kl_factor=.0625, v_loss_layer=31, fact_token="subject_last",
        mom2_update_weight=15000, mom2_n_samples=100000, mom2_dtype="float32",
        L2=10, nullspace_threshold=.02)


class BaselinePilotTests(unittest.TestCase):
    def test_fixed_named_configs_and_native_hparams(self):
        config = bp._config({"baseline_pilot": {}})
        self.assertEqual(config["families"], list(bp.FAMILIES))
        for family in bp.FAMILIES:
            bp._verify_hparams(hparams(), family)
        changed = hparams()
        changed.v_num_grad_steps = 32
        with self.assertRaisesRegex(bp.BaselineBlocked, "HPARAM_CHANGED"):
            bp._verify_hparams(changed, "MEMIT-H")
        changed = hparams()
        changed.L2 = 1
        with self.assertRaisesRegex(bp.BaselineBlocked, "HPARAM_CHANGED"):
            bp._verify_hparams(changed, "BASE_ALPHAEDIT")

    def test_does_not_add_unknown_or_duplicate_baseline(self):
        with self.assertRaisesRegex(RuntimeError, "FAMILY_IDENTITY"):
            bp._config({"families": ["BLUE"]})
        with self.assertRaisesRegex(RuntimeError, "FAMILY_IDENTITY"):
            bp._config({"families": ["MEMIT-H", "MEMIT-H"]})
        with self.assertRaisesRegex(RuntimeError, "OMISSION_WITHOUT"):
            bp._config({"families": ["MEMIT-H"]})

    def test_container_adapter_matches_tuple_output_and_gradient(self):
        def native_callback(output, layer):
            self.assertEqual(layer, "model.layers.8")
            output[0][:, 1, :] += delta
            return output

        delta = torch.tensor([.1, .2, .3], requires_grad=True)
        base = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        tensor_out = bp._native_container_callback(native_callback)(base.clone(), "model.layers.8")
        tensor_loss = tensor_out.square().sum()
        gradient = torch.autograd.grad(tensor_loss, delta)[0]
        direct = native_callback((base.clone(),), "model.layers.8")[0]
        direct_gradient = torch.autograd.grad(direct.square().sum(), delta)[0]
        self.assertTrue(torch.equal(tensor_out, direct))
        self.assertTrue(torch.equal(gradient, direct_gradient))

    def test_tuple_remains_tuple_and_auxiliary_is_preserved(self):
        value = torch.zeros(1, 2, 3)
        aux = object()
        wrapped = bp._native_container_callback(lambda output, layer: output)
        output = (value, aux)
        self.assertIs(wrapped(output, "layer"), output)
        self.assertIs(wrapped(value, "layer"), value)
        with self.assertRaisesRegex(bp.BaselineBlocked, "CONTAINER_CHANGED"):
            bp._native_container_callback(lambda output, layer: output + (aux,))(value, "layer")

    def test_native_trace_container_view_keeps_actual_model_tensor(self):
        # Load the immutable utility directly: no package/editor/model import.
        path = Path(bp.NATIVE_DEFAULT) / "util/nethook.py"
        spec = importlib.util.spec_from_file_location("baseline_test_native_nethook", path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)

        class Block(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = torch.nn.Sequential(Block(), Block())
        delta = torch.tensor(.2, requires_grad=True)

        def native_callback(output, layer):
            if layer == "0":
                output[0][:, 1, :] += delta
            return output

        trace_cls = bp._trace_dict_compat(native.TraceDict)
        source = torch.ones(2, 3, 4)
        with trace_cls(model, ["0", "1"], edit_output=native_callback) as traces:
            actual = model(source)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertIsInstance(traces["1"].output, tuple)
        self.assertTrue(torch.equal(traces["1"].output[0], actual))
        self.assertEqual(float(torch.autograd.grad(actual.sum(), delta)[0]), 16.)
        self.assertEqual(len(model[0]._forward_hooks), 0)
        self.assertEqual(len(model[1]._forward_hooks), 0)

    def test_w0_restore_copies_bytes_without_checkpoint(self):
        weights = {4: torch.nn.Parameter(torch.ones(3, 4)), 5: torch.nn.Parameter(torch.ones(3, 4) * 2)}
        original = {k: v.detach().clone() for k, v in weights.items()}
        with torch.no_grad():
            for value in weights.values():
                value.add_(123.)
        bp._restore_w0(weights, original)
        self.assertTrue(all(torch.equal(weights[k], original[k]) for k in weights))
        self.assertEqual(bp._weight_hashes(weights), bp._weight_hashes(original))

    def test_rng_roundtrip(self):
        before = bp._rng_capture()
        identity = bp._rng_hash(before)
        random.random()
        np.random.random()
        torch.rand(4)
        self.assertNotEqual(bp._rng_hash(bp._rng_capture()), identity)
        bp._rng_restore(before)
        self.assertEqual(bp._rng_hash(bp._rng_capture()), identity)

    def test_stats_missing_or_regen_is_blocked_without_original_call(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            stats = bp._stats_guard(lambda *a, **kw: calls.append((a, kw)), {})
            args = (None, None, "model.layers.4.mlp.down_proj", tmp, "wikipedia")
            with self.assertRaisesRegex(bp.BaselineBlocked, "C0_MISSING"):
                stats(*args, precision="float32", sample_size=100000)
            with self.assertRaisesRegex(bp.BaselineBlocked, "REGENERATION_FORBIDDEN"):
                stats(*args, force_recompute=True)
        self.assertEqual(calls, [])

    def test_proxy_falls_through_without_eager_default_lookup(self):
        source = SimpleNamespace(available=1)
        proxy = bp._Proxy(source, new_key=4)
        self.assertEqual(proxy.new_key, 4)
        self.assertEqual(proxy.available, 1)

    def test_traceview_only_wraps_output(self):
        value = torch.zeros(1)
        view = bp._TraceView(SimpleNamespace(input=value, output=value))
        self.assertIs(view.input, value)
        self.assertIs(view.output[0], value)


if __name__ == "__main__":
    unittest.main()
