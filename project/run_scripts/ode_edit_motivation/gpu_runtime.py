"""Offline, one-GPU loader for the two fixed Motivation snapshots.

MV-0 is a fidelity gate against EasyEdit's canonical BaseEditor behavior, so
its legacy runtime remains forced float32.  Session 02 method experiments use
a separate explicit entry point that preserves the pinned checkpoint dtype;
the two provenance policies must never be silently conflated.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
import platform
import random
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

from .contracts import ContractError
from .manifests import FixedModelSpec, fixed_model_spec


OFFLINE_ENVIRONMENT: Mapping[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
}

EXPECTED_RUNTIME: Mapping[str, str] = {
    "python": "3.12.0",
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "transformers": "4.57.1",
    "numpy": "2.2.6",
    "accelerate": "1.13.0",
}

LEGACY_MOTIVATION_DTYPE_POLICY = "legacy-motivation-float32"
CHECKPOINT_ORIGINAL_DTYPE_POLICY = "checkpoint-original"
PINNED_CHECKPOINT_DTYPE = torch.bfloat16


class GpuRuntimeError(RuntimeError):
    """The fixed offline GPU runtime contract is unavailable."""


@dataclass(frozen=True, slots=True)
class GpuIdentity:
    visible_device_count: int
    index: int
    name: str
    total_memory: int
    capability: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible_device_count": self.visible_device_count,
            "index": self.index,
            "name": self.name,
            "total_memory": self.total_memory,
            "capability": (
                None if self.capability is None else list(self.capability)
            ),
        }


@dataclass(slots=True)
class FixedModelRuntime:
    spec: FixedModelSpec
    model: torch.nn.Module
    tokenizer: Any
    gpu: GpuIdentity
    observed_model_commit: str
    observed_tokenizer_commit: str
    tokenizer_policy: str
    dtype_policy: str = LEGACY_MOTIVATION_DTYPE_POLICY
    checkpoint_original_dtype: torch.dtype = PINNED_CHECKPOINT_DTYPE

    def metadata(self) -> dict[str, Any]:
        observed_dtype = _single_floating_parameter_dtype(self.model)
        return {
            "model_alias": self.spec.alias,
            "repository_id": self.spec.repository_id,
            "revision": self.spec.revision,
            "dtype": str(observed_dtype),
            "observed_parameter_dtype": str(observed_dtype),
            "checkpoint_original_dtype": str(self.checkpoint_original_dtype),
            "dtype_policy": self.dtype_policy,
            "device": "cuda:0",
            "gpu": self.gpu.to_dict(),
            "observed_model_commit": self.observed_model_commit,
            "observed_tokenizer_commit": self.observed_tokenizer_commit,
            "model_config_use_cache": False,
            "tokenizer_policy": self.tokenizer_policy,
            "offline": True,
            "runtime_versions": runtime_versions(),
        }


@contextlib.contextmanager
def offline_environment() -> Iterator[None]:
    """Temporarily force all Hugging Face loaders into offline mode."""

    previous = {key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}
    os.environ.update(OFFLINE_ENVIRONMENT)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_single_visible_gpu(cuda_api: Any = torch.cuda) -> GpuIdentity:
    """Require Slurm/CUDA visibility to expose exactly one usable GPU."""

    if not bool(cuda_api.is_available()):
        raise GpuRuntimeError("CUDA is unavailable; MV-0 requires one GPU")
    count = int(cuda_api.device_count())
    if count != 1:
        raise GpuRuntimeError(
            f"MV-0 requires exactly one visible GPU, found {count}; "
            "set the scheduler/CUDA visibility envelope to one device"
        )
    cuda_api.set_device(0)
    properties = cuda_api.get_device_properties(0)
    capability: tuple[int, int] | None = None
    if hasattr(cuda_api, "get_device_capability"):
        raw_capability = cuda_api.get_device_capability(0)
        capability = (int(raw_capability[0]), int(raw_capability[1]))
    return GpuIdentity(
        visible_device_count=count,
        index=0,
        name=str(properties.name),
        total_memory=int(properties.total_memory),
        capability=capability,
    )


def seed_runtime(seed: int) -> dict[str, Any]:
    """Seed generation/direct-z RNGs and disable TF32 drift."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("runtime seed must be a non-negative integer")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": importlib.metadata.version("transformers"),
        "numpy": importlib.metadata.version("numpy"),
        "accelerate": importlib.metadata.version("accelerate"),
    }


def assert_fixed_runtime() -> dict[str, str]:
    observed = runtime_versions()
    if observed != dict(EXPECTED_RUNTIME):
        raise GpuRuntimeError(
            f"runtime version mismatch: expected {dict(EXPECTED_RUNTIME)}, "
            f"observed {observed}"
        )
    return observed


def fixed_pretrained_kwargs(spec: FixedModelSpec) -> dict[str, Any]:
    """Common no-network arguments for both model and tokenizer loaders."""

    return {
        "pretrained_model_name_or_path": spec.repository_id,
        "revision": spec.revision,
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _normalize_commit(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(char not in "0123456789abcdef" for char in normalized):
        return None
    return normalized


def _observed_model_commit(model: Any) -> str | None:
    config = getattr(model, "config", None)
    return _normalize_commit(getattr(config, "_commit_hash", None))


def _observed_tokenizer_commit(tokenizer: Any) -> str | None:
    values = [
        getattr(tokenizer, "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if isinstance(getattr(tokenizer, "init_kwargs", None), dict)
        else None,
    ]
    for value in values:
        normalized = _normalize_commit(value)
        if normalized is not None:
            return normalized
    return None


def _cached_snapshot_commit(
    spec: FixedModelSpec,
    cached_file_fn: Any,
    filename: str,
) -> str:
    """Resolve a file locally and require the requested snapshot segment."""

    path = cached_file_fn(
        spec.repository_id,
        filename,
        revision=spec.revision,
        local_files_only=True,
        _raise_exceptions_for_gated_repo=True,
        _raise_exceptions_for_missing_entries=True,
        _raise_exceptions_for_connection_errors=True,
    )
    if path is None:
        raise GpuRuntimeError(
            f"{filename} is absent from the local fixed snapshot: {spec.snapshot_name}"
        )
    parts = str(path).replace("\\", "/").split("/")
    observed: str | None = None
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            observed = _normalize_commit(parts[index + 1])
    if observed != spec.revision:
        raise GpuRuntimeError(
            f"local {filename} resolved to commit {observed!r}, "
            f"expected {spec.revision}"
        )
    return observed


def _validate_loaded_model(
    model: torch.nn.Module,
    spec: FixedModelSpec,
    cached_commit: str,
    *,
    expected_parameter_dtype: torch.dtype,
    expected_config_dtype: torch.dtype | None,
) -> str:
    observed = _observed_model_commit(model)
    if observed is not None and observed != spec.revision:
        raise GpuRuntimeError(
            f"loaded model commit {observed} does not match {spec.revision}"
        )
    wrong_devices: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda" or parameter.device.index not in (None, 0):
            wrong_devices.append(f"{name}:{parameter.device}")
        if len(wrong_devices) >= 3:
            break
    if wrong_devices:
        raise GpuRuntimeError(
            "model parameters are not confined to cuda:0: " + ", ".join(wrong_devices[:3])
        )
    _validate_model_dtype_contract(
        model,
        expected_parameter_dtype=expected_parameter_dtype,
        expected_config_dtype=expected_config_dtype,
    )
    if getattr(getattr(model, "config", None), "use_cache", None) is not False:
        raise GpuRuntimeError("model.config.use_cache must be False for BaseEditor parity")
    return observed or cached_commit


def _floating_parameter_dtypes(model: torch.nn.Module) -> frozenset[torch.dtype]:
    dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if not dtypes:
        raise GpuRuntimeError("loaded model has no floating parameters")
    return frozenset(dtypes)


def _single_floating_parameter_dtype(model: torch.nn.Module) -> torch.dtype:
    dtypes = _floating_parameter_dtypes(model)
    if len(dtypes) != 1:
        observed = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise GpuRuntimeError(f"model floating parameter dtypes are mixed: {observed}")
    return next(iter(dtypes))


def _validate_model_dtype_contract(
    model: torch.nn.Module,
    *,
    expected_parameter_dtype: torch.dtype,
    expected_config_dtype: torch.dtype | None,
) -> torch.dtype:
    """Validate a common, alias-independent loaded dtype contract."""

    observed = _single_floating_parameter_dtype(model)
    if observed is not expected_parameter_dtype:
        raise GpuRuntimeError(
            "model floating parameter dtype differs: "
            f"expected {expected_parameter_dtype}, observed {observed}"
        )
    if expected_config_dtype is not None:
        config_dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
        if config_dtype is not expected_config_dtype:
            raise GpuRuntimeError(
                "model.config.torch_dtype differs: "
                f"expected {expected_config_dtype}, observed {config_dtype}"
            )
    return observed


def _validate_loaded_tokenizer(
    tokenizer: Any,
    spec: FixedModelSpec,
    cached_commit: str,
) -> tuple[str, str]:
    observed = _observed_tokenizer_commit(tokenizer)
    if observed is not None and observed != spec.revision:
        raise GpuRuntimeError(
            f"loaded tokenizer commit {observed} does not match {spec.revision}"
        )
    if spec.alias == "qwen2.5-7b-inst":
        endoftext = "<|endoftext|>"
        token_id = int(tokenizer.convert_tokens_to_ids(endoftext))
        if token_id != 151_643:
            raise GpuRuntimeError(
                "Qwen BaseEditor parity requires <|endoftext|> token id 151643"
            )
        tokenizer.pad_token = endoftext
        tokenizer.eos_token = endoftext
        if tokenizer.pad_token_id != 151_643 or tokenizer.eos_token_id != 151_643:
            raise GpuRuntimeError("Qwen special-token override did not take effect")
        policy = "qwen-baseeditor-endoftext-pad-and-eos"
    elif getattr(tokenizer, "eos_token", None) is None:
        raise GpuRuntimeError("Llama tokenizer has no eos token for padding parity")
    else:
        # EasyEdit uses EOS as the padding token for this Llama snapshot.
        tokenizer.pad_token = tokenizer.eos_token
        policy = "llama-eos-as-pad"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise GpuRuntimeError("fixed tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = eos_token
    tokenizer.padding_side = "right"
    return observed or cached_commit, policy


def load_fixed_model(
    alias: str,
    *,
    auto_model_class: Any | None = None,
    auto_tokenizer_class: Any | None = None,
    cached_file_fn: Any | None = None,
    cuda_api: Any = torch.cuda,
) -> FixedModelRuntime:
    """Load the legacy Motivation runtime in forced float32.

    This entry point is intentionally preserved for MV-0/BaseEditor fidelity.
    Session 02 method experiments must use
    :func:`load_fixed_model_checkpoint_original` instead.
    """

    return _load_fixed_model_with_dtype_policy(
        alias,
        model_dtype_kwargs={"torch_dtype": torch.float32},
        expected_parameter_dtype=torch.float32,
        expected_config_dtype=None,
        dtype_policy=LEGACY_MOTIVATION_DTYPE_POLICY,
        auto_model_class=auto_model_class,
        auto_tokenizer_class=auto_tokenizer_class,
        cached_file_fn=cached_file_fn,
        cuda_api=cuda_api,
    )


def load_fixed_model_checkpoint_original(
    alias: str,
    *,
    auto_model_class: Any | None = None,
    auto_tokenizer_class: Any | None = None,
    cached_file_fn: Any | None = None,
    cuda_api: Any = torch.cuda,
) -> FixedModelRuntime:
    """Load one pinned model in its checkpoint-original BF16 dtype."""

    return _load_fixed_model_with_dtype_policy(
        alias,
        model_dtype_kwargs={"dtype": "auto"},
        expected_parameter_dtype=PINNED_CHECKPOINT_DTYPE,
        expected_config_dtype=PINNED_CHECKPOINT_DTYPE,
        dtype_policy=CHECKPOINT_ORIGINAL_DTYPE_POLICY,
        auto_model_class=auto_model_class,
        auto_tokenizer_class=auto_tokenizer_class,
        cached_file_fn=cached_file_fn,
        cuda_api=cuda_api,
    )


def _load_fixed_model_with_dtype_policy(
    alias: str,
    *,
    model_dtype_kwargs: Mapping[str, Any],
    expected_parameter_dtype: torch.dtype,
    expected_config_dtype: torch.dtype | None,
    dtype_policy: str,
    auto_model_class: Any | None,
    auto_tokenizer_class: Any | None,
    cached_file_fn: Any | None,
    cuda_api: Any,
) -> FixedModelRuntime:
    """Shared offline loader; callers must select one explicit dtype policy."""

    spec = fixed_model_spec(alias)
    assert_fixed_runtime()
    gpu = assert_single_visible_gpu(cuda_api)
    if auto_model_class is None or auto_tokenizer_class is None or cached_file_fn is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils.hub import cached_file
        except ImportError as exc:
            raise GpuRuntimeError("transformers is unavailable in the runtime") from exc
        auto_model_class = auto_model_class or AutoModelForCausalLM
        auto_tokenizer_class = auto_tokenizer_class or AutoTokenizer
        cached_file_fn = cached_file_fn or cached_file

    common = fixed_pretrained_kwargs(spec)
    with offline_environment():
        model_cached_commit = _cached_snapshot_commit(
            spec, cached_file_fn, "config.json"
        )
        tokenizer_cached_commit = _cached_snapshot_commit(
            spec, cached_file_fn, "tokenizer_config.json"
        )
        _cached_snapshot_commit(spec, cached_file_fn, "tokenizer.json")
        tokenizer = auto_tokenizer_class.from_pretrained(
            **common,
            use_fast=True,
        )
        model = auto_model_class.from_pretrained(
            **common,
            **dict(model_dtype_kwargs),
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    model_commit = _validate_loaded_model(
        model,
        spec,
        model_cached_commit,
        expected_parameter_dtype=expected_parameter_dtype,
        expected_config_dtype=expected_config_dtype,
    )
    tokenizer_commit, tokenizer_policy = _validate_loaded_tokenizer(
        tokenizer, spec, tokenizer_cached_commit
    )
    return FixedModelRuntime(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        gpu=gpu,
        observed_model_commit=model_commit,
        observed_tokenizer_commit=tokenizer_commit,
        tokenizer_policy=tokenizer_policy,
        dtype_policy=dtype_policy,
        checkpoint_original_dtype=PINNED_CHECKPOINT_DTYPE,
    )


def rng_state_hash() -> str:
    """Compact identity for the current CPU and visible-CUDA RNG states."""

    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
    return digest.hexdigest()
