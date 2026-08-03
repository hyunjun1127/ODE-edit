import os
import inspect
import types
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    CHECKPOINT_ORIGINAL_DTYPE_POLICY,
    LEGACY_MOTIVATION_DTYPE_POLICY,
    FixedModelRuntime,
    GpuIdentity,
    GpuRuntimeError,
    _validate_model_dtype_contract,
    assert_single_visible_gpu,
    fixed_pretrained_kwargs,
    load_fixed_model,
    load_fixed_model_checkpoint_original,
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
        self.config = types.SimpleNamespace(
            use_cache=True,
            torch_dtype=torch.bfloat16,
            _commit_hash=None,
        )
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


class _DtypeModel(torch.nn.Module):
    def __init__(
        self,
        parameter_dtype: torch.dtype,
        *,
        config_dtype: torch.dtype,
        mixed: bool = False,
    ) -> None:
        super().__init__()
        self.primary = torch.nn.Parameter(
            torch.ones(2, dtype=parameter_dtype),
            requires_grad=False,
        )
        if mixed:
            self.secondary = torch.nn.Parameter(
                torch.ones(2, dtype=torch.float32),
                requires_grad=False,
            )
        self.config = types.SimpleNamespace(
            use_cache=False,
            torch_dtype=config_dtype,
            _commit_hash=None,
        )


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
        self.assertNotIn("dtype", model_call)
        self.assertEqual(model_call["device_map"], {"": "cuda:0"})
        self.assertTrue(model_call["local_files_only"])
        self.assertEqual(tokenizer_call["revision"], spec.revision)
        self.assertFalse(runtime.model.config.use_cache)
        self.assertEqual(runtime.tokenizer.padding_side, "right")
        self.assertEqual(runtime.tokenizer.pad_token_id, 151643)
        self.assertEqual(runtime.tokenizer.eos_token_id, 151643)
        self.assertEqual(runtime.observed_tokenizer_commit, spec.revision)
        self.assertEqual(runtime.dtype_policy, LEGACY_MOTIVATION_DTYPE_POLICY)

    def test_mock_original_loader_uses_auto_without_legacy_float32(self):
        spec = fixed_model_spec("llama3-8b-inst")

        def cached_file(repository_id, filename, **kwargs):
            self.assertEqual(repository_id, spec.repository_id)
            self.assertTrue(kwargs["local_files_only"])
            return f"/cache/models/snapshots/{spec.revision}/{filename}"

        with mock.patch(
            "project.run_scripts.ode_edit_motivation.gpu_runtime._validate_loaded_model",
            return_value=spec.revision,
        ):
            runtime = load_fixed_model_checkpoint_original(
                spec.alias,
                auto_model_class=_FakeModelClass,
                auto_tokenizer_class=_FakeTokenizerClass,
                cached_file_fn=cached_file,
                cuda_api=_FakeCuda(),
            )
        model_call = _FakeModelClass.calls[0]
        self.assertEqual(model_call["dtype"], "auto")
        self.assertNotIn("torch_dtype", model_call)
        self.assertEqual(model_call["device_map"], {"": "cuda:0"})
        self.assertEqual(runtime.dtype_policy, CHECKPOINT_ORIGINAL_DTYPE_POLICY)
        source = inspect.getsource(load_fixed_model_checkpoint_original)
        self.assertNotIn("llama3-8b-inst", source)
        self.assertNotIn("qwen2.5-7b-inst", source)

    def test_checkpoint_original_dtype_and_metadata_fail_closed(self):
        model = _DtypeModel(
            torch.bfloat16,
            config_dtype=torch.bfloat16,
        )
        observed = _validate_model_dtype_contract(
            model,
            expected_parameter_dtype=torch.bfloat16,
            expected_config_dtype=torch.bfloat16,
        )
        self.assertIs(observed, torch.bfloat16)
        spec = fixed_model_spec("llama3-8b-inst")
        runtime = FixedModelRuntime(
            spec=spec,
            model=model,
            tokenizer=object(),
            gpu=GpuIdentity(1, 0, "mock-gpu", 48_000_000_000, (8, 6)),
            observed_model_commit=spec.revision,
            observed_tokenizer_commit=spec.revision,
            tokenizer_policy="fixture",
            dtype_policy=CHECKPOINT_ORIGINAL_DTYPE_POLICY,
            checkpoint_original_dtype=torch.bfloat16,
        )
        metadata = runtime.metadata()
        self.assertEqual(metadata["dtype"], "torch.bfloat16")
        self.assertEqual(metadata["observed_parameter_dtype"], "torch.bfloat16")
        self.assertEqual(metadata["checkpoint_original_dtype"], "torch.bfloat16")
        self.assertEqual(metadata["dtype_policy"], CHECKPOINT_ORIGINAL_DTYPE_POLICY)

        legacy_model = _DtypeModel(
            torch.float32,
            config_dtype=torch.bfloat16,
        )
        legacy_runtime = FixedModelRuntime(
            spec=spec,
            model=legacy_model,
            tokenizer=object(),
            gpu=GpuIdentity(1, 0, "mock-gpu", 48_000_000_000, (8, 6)),
            observed_model_commit=spec.revision,
            observed_tokenizer_commit=spec.revision,
            tokenizer_policy="fixture",
            dtype_policy=LEGACY_MOTIVATION_DTYPE_POLICY,
            checkpoint_original_dtype=torch.bfloat16,
        )
        legacy_metadata = legacy_runtime.metadata()
        self.assertEqual(legacy_metadata["dtype"], "torch.float32")
        self.assertEqual(
            legacy_metadata["dtype_policy"],
            LEGACY_MOTIVATION_DTYPE_POLICY,
        )
        self.assertEqual(
            legacy_metadata["checkpoint_original_dtype"],
            "torch.bfloat16",
        )

        invalid = (
            _DtypeModel(torch.float32, config_dtype=torch.bfloat16),
            _DtypeModel(
                torch.bfloat16,
                config_dtype=torch.bfloat16,
                mixed=True,
            ),
            _DtypeModel(torch.bfloat16, config_dtype=torch.float32),
        )
        for candidate in invalid:
            with self.subTest(
                candidate=tuple(str(p.dtype) for p in candidate.parameters())
            ):
                with self.assertRaises(GpuRuntimeError):
                    _validate_model_dtype_contract(
                        candidate,
                        expected_parameter_dtype=torch.bfloat16,
                        expected_config_dtype=torch.bfloat16,
                    )


if __name__ == "__main__":
    unittest.main()
