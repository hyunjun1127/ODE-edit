import os
import types
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    GpuRuntimeError,
    assert_single_visible_gpu,
    fixed_pretrained_kwargs,
    load_fixed_model,
    offline_environment,
)
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec


class _FakeCuda:
    def __init__(self, count=1, available=True):
        self.count = count
        self.available = available
        self.selected = None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def set_device(self, index):
        self.selected = index

    def get_device_properties(self, index):
        self.selected = index
        return types.SimpleNamespace(name="mock-gpu", total_memory=48_000_000_000)

    def get_device_capability(self, index):
        self.selected = index
        return (8, 6)


class _FakeTokenizer:
    def __init__(self):
        self.init_kwargs = {}
        self._pad_token = "<unset>"
        self._eos_token = "<|im_end|>"
        self.padding_side = "left"

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @property
    def eos_token(self):
        return self._eos_token

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @property
    def pad_token_id(self):
        return self.convert_tokens_to_ids(self._pad_token)

    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(self._eos_token)

    @staticmethod
    def convert_tokens_to_ids(value):
        return {
            "<|endoftext|>": 151643,
            "<|im_end|>": 151645,
            "<unset>": -1,
        }.get(value, 0)


class _FakeTokenizerClass:
    calls = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return _FakeTokenizer()


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=True, _commit_hash=None)
        self.training = True
        self.requires_grad_value = None

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self


class _FakeModelClass:
    calls = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return _FakeModel()


class MV0GpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        _FakeModelClass.calls.clear()
        _FakeTokenizerClass.calls.clear()

    def test_offline_environment_is_scoped(self):
        previous = os.environ.get("HF_HUB_OFFLINE")
        with offline_environment():
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), previous)

    def test_exactly_one_visible_gpu_is_required(self):
        gpu = assert_single_visible_gpu(_FakeCuda())
        self.assertEqual(gpu.visible_device_count, 1)
        self.assertEqual(gpu.index, 0)
        with self.assertRaises(GpuRuntimeError):
            assert_single_visible_gpu(_FakeCuda(count=2))
        with self.assertRaises(GpuRuntimeError):
            assert_single_visible_gpu(_FakeCuda(count=0, available=False))

    def test_load_kwargs_pin_revision_and_forbid_network(self):
        spec = fixed_model_spec("qwen2.5-7b-inst")
        kwargs = fixed_pretrained_kwargs(spec)
        self.assertEqual(kwargs["revision"], spec.revision)
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])

    def test_mock_loader_uses_float32_baseeditor_parity(self):
        spec = fixed_model_spec("qwen2.5-7b-inst")

        def cached_file(repository_id, filename, **kwargs):
            self.assertEqual(repository_id, spec.repository_id)
            self.assertTrue(kwargs["local_files_only"])
            return f"/cache/models/snapshots/{spec.revision}/{filename}"

        # CPU tests cannot construct CUDA parameters; the separate validator is
        # mocked while loader arguments, revision resolution, cache policy and
        # tokenizer policy remain under test.
        with mock.patch(
            "project.run_scripts.ode_edit_motivation.gpu_runtime._validate_loaded_model",
            return_value=spec.revision,
        ):
            runtime = load_fixed_model(
                spec.alias,
                auto_model_class=_FakeModelClass,
                auto_tokenizer_class=_FakeTokenizerClass,
                cached_file_fn=cached_file,
                cuda_api=_FakeCuda(),
            )
        model_call = _FakeModelClass.calls[0]
        tokenizer_call = _FakeTokenizerClass.calls[0]
        self.assertIs(model_call["torch_dtype"], torch.float32)
        self.assertEqual(model_call["device_map"], {"": "cuda:0"})
        self.assertTrue(model_call["local_files_only"])
        self.assertEqual(tokenizer_call["revision"], spec.revision)
        self.assertFalse(runtime.model.config.use_cache)
        self.assertEqual(runtime.tokenizer.padding_side, "right")
        self.assertEqual(runtime.tokenizer.pad_token_id, 151643)
        self.assertEqual(runtime.tokenizer.eos_token_id, 151643)
        self.assertEqual(runtime.observed_tokenizer_commit, spec.revision)


if __name__ == "__main__":
    unittest.main()
