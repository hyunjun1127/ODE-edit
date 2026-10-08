"""CPU compatibility fixtures, not pretrained-model or GPU qualification."""

import contextlib
import importlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from official.baselines.easyedit.util.nethook import get_hidden_state, replace_hidden_state
from official.baselines.blue.util.generate import generate_fast as blue_generate_fast
from official.baselines.sphere.util.generate import generate_fast


compute_z_module = importlib.import_module("official.baselines.sphere.memit.compute_z")
blue_compute_z_module = importlib.import_module("official.baselines.blue.AlphaEdit.compute_z")


class _Batch(dict):
    def to(self, device):
        # compute_z retains its production CUDA placement; this fixture alone
        # maps its tiny fake input tensors to CPU.
        return self


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    unk_token_id = 2

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            return _Batch(input_ids=torch.tensor([[4]]))
        return _Batch(input_ids=torch.tensor([[3, 4, 5]] * len(prompts)),
                      attention_mask=torch.ones(len(prompts), 3, dtype=torch.long))

    def decode(self, ids, **kwargs):
        return " ".join(str(x) for x in ids)


class _Block(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.kind = kind
        self.extra = object()

    def forward(self, hidden):
        value = self.linear(hidden)
        if self.kind == "tuple":
            return (value, self.extra)
        if self.kind == "list":
            return [value, self.extra]
        return value


class _Model(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, vocab_size=9)
        self.embedding = torch.nn.Embedding(9, 4)
        self.layers = torch.nn.ModuleList([_Block(kind), _Block(kind)])
        self.ln_f = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 9)
        self.kind = kind
        self.seen = []

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            output = layer(hidden)
            self.seen.append(type(output))
            if self.kind != "tensor":
                assert output[1] is layer.extra
            hidden = get_hidden_state(output)
        return SimpleNamespace(logits=self.lm_head(self.ln_f(hidden)))


class _GenerationTokenizer(_Tokenizer):
    def __call__(self, prompts, **kwargs):
        return _Batch(input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
                      attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]))


class SphereCompatibilityTests(unittest.TestCase):
    def test_hidden_container_contract_preserves_auxiliary_values(self):
        hidden, replacement, extra = torch.ones(2, 3, 4), torch.zeros(2, 3, 4), object()
        for output in (hidden, (hidden, extra), [hidden, extra]):
            with self.subTest(kind=type(output).__name__):
                self.assertIs(get_hidden_state(output), hidden)
                updated = replace_hidden_state(output, replacement)
                self.assertIs(type(updated), type(output))
                self.assertIs(get_hidden_state(updated), replacement)
                if not isinstance(updated, torch.Tensor):
                    self.assertIs(updated[1], extra)
                    self.assertIs(output[0], hidden)

    def test_hidden_invalid_containers_fail_closed(self):
        for empty in ([], ()):
            with self.assertRaises(ValueError):
                get_hidden_state(empty)
        for value in (None, {"hidden": torch.ones(1)}):
            with self.assertRaises(TypeError):
                get_hidden_state(value)

    def _fit(self, kind, implementation=compute_z_module):
        torch.manual_seed(7)
        model = _Model(kind)
        hp = SimpleNamespace(lm_head_module="lm_head", ln_f_module="ln_f",
                             layer_module_tmp="layers.{}", v_loss_layer=1,
                             fact_token="subject_last", v_lr=.1,
                             v_num_grad_steps=3, kl_factor=.0625,
                             v_weight_decay=.5, clamp_norm_factor=.75)
        request = {"target_new": {"str": "4"}, "prompt": "{} is", "subject": "S"}
        original_zeros, original_tensor = torch.zeros, torch.tensor

        def cpu_factory(factory):
            def call(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return factory(*args, **kwargs)
            return call

        output = io.StringIO()
        with patch.object(implementation.torch, "zeros", cpu_factory(original_zeros)), \
                patch.object(implementation.torch, "tensor", cpu_factory(original_tensor)), \
                patch.object(implementation, "find_fact_lookup_idx", return_value=1), \
                contextlib.redirect_stdout(output):
            target = implementation.compute_z(model, _Tokenizer(), request, hp, 0, [["{}"]])
        self.assertTrue(torch.isfinite(target).all())
        self.assertEqual(len(model.seen), 6)  # Three evaluations, two layers each.
        expected_type = {"tuple": tuple, "list": list, "tensor": torch.Tensor}[kind]
        self.assertTrue(all(t is expected_type for t in model.seen))
        return target.detach(), output.getvalue()

    def test_actual_compute_z_cpu_tensor_matches_legacy_tuple_loss_and_updates(self):
        legacy_target, legacy_log = self._fit("tuple")
        tensor_target, tensor_log = self._fit("tensor")
        self.assertTrue(torch.equal(legacy_target, tensor_target))
        self.assertEqual(legacy_log, tensor_log)

    def test_actual_compute_z_cpu_list_matches_legacy_tuple(self):
        legacy_target, legacy_log = self._fit("tuple")
        list_target, list_log = self._fit("list")
        self.assertTrue(torch.equal(legacy_target, list_target))
        self.assertEqual(legacy_log, list_log)

    def test_actual_blue_compute_z_cpu_tensor_and_list_match_legacy_tuple(self):
        legacy_target, legacy_log = self._fit("tuple", blue_compute_z_module)
        for kind in ("tensor", "list"):
            with self.subTest(kind=kind):
                target, log = self._fit(kind, blue_compute_z_module)
                self.assertTrue(torch.equal(legacy_target, target))
                self.assertEqual(legacy_log, log)

    def _check_cached_generation(self, implementation):
        torch.manual_seed(11)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=13, hidden_size=16,
                                          intermediate_size=32, num_hidden_layers=2,
                                          num_attention_heads=2, num_key_value_heads=2,
                                          max_position_embeddings=32,
                                          attention_dropout=0)).eval()
        model.config._attn_implementation = "eager"
        forward = model.forward
        accumulated = None
        calls = []

        def checked_forward(*args, **kwargs):
            nonlocal accumulated
            query = kwargs["input_ids"]
            accumulated = query.clone() if accumulated is None else torch.cat([accumulated, query], dim=1)
            mask = kwargs["attention_mask"]
            self.assertEqual(mask.shape, accumulated.shape)
            actual = forward(*args, **kwargs)
            reference = forward(input_ids=accumulated, attention_mask=mask, use_cache=False)
            self.assertTrue(torch.allclose(actual.logits[:, -1], reference.logits[:, -1],
                                           atol=2e-6, rtol=2e-5))
            calls.append((query.shape[1], mask.shape[1]))
            return actual

        with patch.object(model, "forward", side_effect=checked_forward):
            text = implementation(model, _GenerationTokenizer(), ["short", "long"], max_out_len=6)
        self.assertEqual(calls, [(2, 2), (1, 3), (1, 4), (1, 5)])
        self.assertEqual(len(text), 2)
        self.assertTrue(text[0].startswith("1 2 "))
        self.assertTrue(text[1].startswith("3 4 5 "))

    def test_cached_sphere_generation_uses_full_prefix_mask_and_native_sampling(self):
        self._check_cached_generation(generate_fast)

    def test_cached_blue_generation_uses_full_prefix_mask_and_native_sampling(self):
        self._check_cached_generation(blue_generate_fast)


if __name__ == "__main__":
    unittest.main()
