"""Read-only EasyEdit AlphaEdit capture and Native/Woodbury P0 identity path."""

from __future__ import annotations

import contextlib
import copy
import io
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import (
    ComputeLedger,
    JointInitializationReceipt,
    LayerFactorReceipt,
)
from .artifacts import ODEBFArtifactGuard
from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from .woodbury import (
    ProjectorCertificate,
    WoodburyCertificate,
    solve_alpha_woodbury,
)


ALPHA_SOLVE_DTYPE = torch.float32
ALPHA_SOLVE_REFERENCE = "Native AlphaEdit original-BF16 canonical-FP32-solve"
ALPHA_SOLVE_CONDITION_MAX_DIMENSION = 256


@dataclass(frozen=True, slots=True)
class AlphaSolveGeometry:
    projector_shape: tuple[int, int]
    key_shape: tuple[int, int]
    covariance_shape: tuple[int, int]
    residual_shape: tuple[int, int]
    output_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class AlphaDenseSolveReceipt:
    layer: int
    reference: str
    solve_dtype: str
    solve_device_class: str
    geometry: AlphaSolveGeometry
    input_dtypes: tuple[str, str, str, str]
    input_device_classes: tuple[str, str, str, str]
    normalized_dtypes: tuple[str, str, str, str, str, str]
    normalized_device_classes: tuple[str, str, str, str, str, str]
    normalized_storage_reused: tuple[bool, bool, bool, bool]
    joint_key_rank: int
    relative_residual: float
    condition_estimate: float | None
    condition_kind: str
    finite: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class AlphaDenseSolveResult:
    update: torch.Tensor
    receipt: AlphaDenseSolveReceipt


def validate_joint_alpha_solve_geometry(
    projector_shape: Sequence[int],
    key_shape: Sequence[int],
    covariance_shape: Sequence[int],
    residual_shape: Sequence[int],
) -> AlphaSolveGeometry:
    p_shape = tuple(int(value) for value in projector_shape)
    k_shape = tuple(int(value) for value in key_shape)
    c_shape = tuple(int(value) for value in covariance_shape)
    r_shape = tuple(int(value) for value in residual_shape)
    if len(p_shape) != 2 or p_shape[0] <= 0 or p_shape[0] != p_shape[1]:
        raise ODEBFContractError("Alpha solve projector geometry is not square")
    dimension = p_shape[0]
    if k_shape != (dimension, BATCH_SIZE):
        raise ODEBFContractError("Alpha solve keys are not one genuine joint B10")
    if c_shape != p_shape:
        raise ODEBFContractError("Alpha solve covariance geometry differs")
    if len(r_shape) != 2 or r_shape[0] <= 0 or r_shape[1] != BATCH_SIZE:
        raise ODEBFContractError("Alpha solve residual is not one genuine joint B10")
    return AlphaSolveGeometry(
        p_shape,
        k_shape,
        c_shape,
        r_shape,
        (dimension, r_shape[0]),
    )


def _normalize_solve_tensor(
    name: str,
    value: torch.Tensor,
    *,
    solve_device: torch.device,
) -> tuple[torch.Tensor, bool]:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.device.type not in ("cpu", "cuda")
    ):
        raise ODEBFContractError(f"Alpha solve {name} tensor contract differs")
    if not torch.isfinite(value).all():
        raise ODEBFContractError(f"Alpha solve {name} contains non-finite values")
    expected_reuse = (
        value.device == solve_device
        and value.dtype is ALPHA_SOLVE_DTYPE
        and value.is_contiguous()
    )
    normalized = (
        value.detach()
        .to(device=solve_device, dtype=ALPHA_SOLVE_DTYPE)
        .contiguous()
    )
    observed_reuse = normalized.data_ptr() == value.data_ptr()
    if observed_reuse != expected_reuse:
        raise ODEBFContractError(f"Alpha solve {name} copy contract differs")
    if (
        normalized.dtype is not ALPHA_SOLVE_DTYPE
        or normalized.device != solve_device
        or not normalized.is_contiguous()
    ):
        raise ODEBFContractError(f"Alpha solve {name} normalization failed")
    return normalized, observed_reuse


def canonical_alpha_fp32_solve(
    projector: torch.Tensor,
    joint_keys: torch.Tensor,
    covariance: torch.Tensor,
    residual: torch.Tensor,
    *,
    layer: int,
    regularization: float | torch.Tensor,
    solve_device: torch.device | str,
    residual_tolerance: float,
    condition_max_dimension: int = ALPHA_SOLVE_CONDITION_MAX_DIMENSION,
) -> AlphaDenseSolveResult:
    """Apply the pinned AlphaEdit source-order equation at an explicit FP32 boundary."""

    device = torch.device(solve_device)
    if device.type not in ("cpu", "cuda"):
        raise ODEBFContractError("Alpha solve device class is unsupported")
    if (
        isinstance(condition_max_dimension, bool)
        or not isinstance(condition_max_dimension, int)
        or condition_max_dimension < 0
    ):
        raise ODEBFContractError("Alpha solve condition dimension lock is invalid")
    tolerance = float(residual_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ODEBFContractError("Alpha solve residual tolerance is invalid")

    tensors = (projector, joint_keys, covariance, residual)
    input_versions = tuple(value._version for value in tensors)
    input_pointers = tuple(value.data_ptr() for value in tensors)
    input_requires_grad = tuple(value.requires_grad for value in tensors)
    geometry = validate_joint_alpha_solve_geometry(
        projector.shape,
        joint_keys.shape,
        covariance.shape,
        residual.shape,
    )
    input_dtypes = tuple(str(value.dtype) for value in tensors)
    input_devices = tuple(value.device.type for value in tensors)

    p32, p_reused = _normalize_solve_tensor(
        "projector", projector, solve_device=device
    )
    k32, k_reused = _normalize_solve_tensor(
        "joint keys", joint_keys, solve_device=device
    )
    c32, c_reused = _normalize_solve_tensor(
        "covariance", covariance, solve_device=device
    )
    r32, r_reused = _normalize_solve_tensor(
        "residual", residual, solve_device=device
    )
    if isinstance(regularization, torch.Tensor):
        if regularization.numel() != 1 or not regularization.is_floating_point():
            raise ODEBFContractError("Alpha solve regularization tensor differs")
        regularization_value = float(regularization.detach().to(device="cpu"))
    else:
        regularization_value = float(regularization)
    if not math.isfinite(regularization_value) or regularization_value <= 0.0:
        raise ODEBFContractError("Alpha solve regularization is not positive finite")
    lambda32 = torch.tensor(
        regularization_value,
        dtype=ALPHA_SOLVE_DTYPE,
        device=device,
    )
    identity32 = torch.eye(
        geometry.projector_shape[0],
        dtype=ALPHA_SOLVE_DTYPE,
        device=device,
    )
    normalized = (p32, k32, c32, r32, lambda32, identity32)
    if (
        {value.dtype for value in normalized} != {ALPHA_SOLVE_DTYPE}
        or {value.device for value in normalized} != {device}
    ):
        raise ODEBFContractError("Alpha solve operands remain mixed after normalization")

    # Keep the exact pinned source multiplication and addition order.
    a32 = p32 @ (k32 @ k32.T + c32) + lambda32 * identity32
    b32 = p32 @ k32 @ r32.T
    update32 = torch.linalg.solve(a32, b32)
    if (
        update32.shape != geometry.output_shape
        or update32.dtype is not ALPHA_SOLVE_DTYPE
        or update32.device != device
    ):
        raise ODEBFContractError("Alpha solve output contract differs")
    relative_residual = float(
        torch.linalg.norm(a32 @ update32 - b32)
        / torch.clamp(
            torch.linalg.norm(a32) * torch.linalg.norm(update32)
            + torch.linalg.norm(b32),
            min=torch.finfo(ALPHA_SOLVE_DTYPE).tiny,
        )
    )
    condition_estimate: float | None = None
    condition_kind = "omitted-large-production-system"
    if geometry.projector_shape[0] <= condition_max_dimension:
        condition_estimate = float(
            torch.linalg.cond(
                a32.detach().to(device="cpu", dtype=torch.float64)
            )
        )
        condition_kind = "float64-exact-small-fixture"
    key_rank = int(torch.linalg.matrix_rank(k32).item())
    finite = bool(
        torch.isfinite(update32).all()
        and math.isfinite(relative_residual)
        and (
            condition_estimate is None
            or math.isfinite(condition_estimate)
        )
    )
    passed = (
        finite
        and key_rank > 1
        and relative_residual <= tolerance
    )

    if (
        tuple(value._version for value in tensors) != input_versions
        or tuple(value.data_ptr() for value in tensors) != input_pointers
        or tuple(value.requires_grad for value in tensors) != input_requires_grad
    ):
        raise ODEBFContractError("Alpha solve mutated or rebound an input operand")
    receipt = AlphaDenseSolveReceipt(
        int(layer),
        ALPHA_SOLVE_REFERENCE,
        str(ALPHA_SOLVE_DTYPE),
        device.type,
        geometry,
        input_dtypes,  # type: ignore[arg-type]
        input_devices,  # type: ignore[arg-type]
        tuple(str(value.dtype) for value in normalized),  # type: ignore[arg-type]
        tuple(value.device.type for value in normalized),  # type: ignore[arg-type]
        (p_reused, k_reused, c_reused, r_reused),
        key_rank,
        relative_residual,
        condition_estimate,
        condition_kind,
        finite,
        passed,
    )
    if not passed:
        raise ODEBFContractError("Alpha dense solve certificate failed")
    return AlphaDenseSolveResult(update32, receipt)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_original_bf16(
    guard: ODEBFArtifactGuard,
) -> tuple[Any, Any, Any]:
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ODEBFContractError("technical P0 requires exactly one visible CUDA device")
    snapshot = guard.base_guard.snapshot
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hparams = AlphaEditHyperParams.from_hparams(str(guard.hparams))
    hparams.device = 0
    hparams.P_loc = str(guard.projector)
    hparams.stats_dir = str(guard.easyedit_root / "examples" / "data" / "stats")
    model.config._name_or_path = hparams.model_name
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise ODEBFContractError("original model parameters are not uniformly BF16")
    if next(model.parameters()).device != torch.device("cuda:0"):
        raise ODEBFContractError("technical P0 model is not on the single visible GPU")
    if tuple(hparams.layers) != tuple(guard.spec["layers"]) or float(hparams.L2) <= 0.0:
        raise ODEBFContractError("pinned AlphaEdit layer/regularization contract differs")
    return model, tokenizer, hparams


def fresh_contexts_twice(
    model: Any,
    tokenizer: Any,
    *,
    seed: int,
) -> tuple[list[list[str]], str]:
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    prior_cache = alpha_main.CONTEXT_TEMPLATES_CACHE
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            alpha_main.CONTEXT_TEMPLATES_CACHE = None
            seed_all(seed)
            first = copy.deepcopy(alpha_main.get_context_templates(model, tokenizer))
            alpha_main.CONTEXT_TEMPLATES_CACHE = None
            seed_all(seed)
            second = copy.deepcopy(alpha_main.get_context_templates(model, tokenizer))
        if first != second or canonical_hash(first) != canonical_hash(second):
            raise ODEBFContractError("fresh AlphaEdit contexts differ across generation")
        alpha_main.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(second)
        return second, canonical_hash(second)
    except BaseException:
        alpha_main.CONTEXT_TEMPLATES_CACHE = prior_cache
        raise
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, 0)


def _normalize_requests(requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("AlphaEdit backend requires one joint B10 batch")
    normalized = copy.deepcopy(list(requests))
    identities: list[str] = []
    for request in normalized:
        identities.append(str(request["request_sha256"]))
        target = str(request["target_new"])
        if not target.startswith(" "):
            target = " " + target
        request["target_new"] = target
        prompt = str(request["prompt"])
        subject = str(request["subject"])
        if "{}" not in prompt:
            if subject not in prompt:
                raise ODEBFContractError("AlphaEdit subject is absent from its prompt")
            prompt = prompt.replace(subject, "{}")
        request["prompt"] = prompt
    if len(set(identities)) != BATCH_SIZE:
        raise ODEBFContractError("AlphaEdit backend received duplicate requests")
    return normalized


@dataclass(slots=True)
class CapturedNativeWBEndpoint:
    native_candidates: dict[str, torch.Tensor]
    wb_candidates: dict[str, torch.Tensor]
    wb_factors: dict[str, tuple[WaypointFactor, ...]]
    entry_weights: dict[str, torch.Tensor]
    entry_sha256: dict[str, str]
    direct_z_sha256: tuple[str, ...]
    key_sha256_by_layer: tuple[tuple[int, str], ...]
    dense_solve_receipts: tuple[AlphaDenseSolveReceipt, ...]
    woodbury_certificates: tuple[tuple[int, WoodburyCertificate], ...]
    initialization: JointInitializationReceipt
    target_backward_count: int


def _execute_alphaedit_canonical_fp32_solve(
    alpha_main: Any,
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    contexts: Sequence[Sequence[str]],
    weights: Mapping[str, torch.nn.Parameter],
    *,
    residual_tolerance: float,
) -> tuple[dict[str, torch.Tensor], tuple[AlphaDenseSolveReceipt, ...]]:
    """Adapter-local reproduction of the pinned source loop with one FP32 solve boundary."""

    if any(parameter.dtype is not torch.bfloat16 for parameter in weights.values()):
        raise ODEBFContractError("canonical Alpha solve requires original BF16 weights")
    resolved_contexts = alpha_main.get_context_templates(model, tokenizer)
    if canonical_hash(resolved_contexts) != canonical_hash(list(contexts)):
        raise ODEBFContractError("canonical Alpha solve context identity differs")
    normalized = copy.deepcopy(list(requests))
    entry = {name: parameter.detach().clone() for name, parameter in weights.items()}
    deltas: dict[str, torch.Tensor] = {}
    receipts: list[AlphaDenseSolveReceipt] = []
    z_layer = int(hparams.layers[-1])
    try:
        direct_z = [
            alpha_main.compute_z(
                model,
                tokenizer,
                request,
                hparams,
                z_layer,
                resolved_contexts,
            )
            for request in normalized
        ]
        if len(direct_z) != BATCH_SIZE:
            raise ODEBFContractError("canonical Alpha solve direct-z count differs")
        zs = torch.stack([value.detach() for value in direct_z], dim=1)
        for layer_index, layer in enumerate(hparams.layers):
            layer_keys = alpha_main.compute_ks(
                model,
                tokenizer,
                normalized,
                hparams,
                layer,
                resolved_contexts,
            ).T
            current_z = alpha_main.get_module_input_output_at_words(
                model,
                tokenizer,
                z_layer,
                context_templates=[request["prompt"] for request in normalized],
                words=[request["subject"] for request in normalized],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs - current_z
            if targets.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("canonical Alpha residual is not joint B10")
            repeat_factor = layer_keys.shape[1] // targets.shape[1]
            if repeat_factor != 1:
                raise ODEBFContractError("canonical Alpha joint B10 was repeated or decomposed")
            residual = targets.repeat_interleave(repeat_factor, dim=1)
            residual = residual / (len(hparams.layers) - layer_index)
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            parameter = weights[weight_name]
            solved = canonical_alpha_fp32_solve(
                alpha_main.P[layer_index],
                layer_keys,
                alpha_main.cache_c[layer_index],
                residual,
                layer=int(layer),
                regularization=hparams.L2,
                solve_device=parameter.device,
                residual_tolerance=residual_tolerance,
            )
            receipts.append(solved.receipt)
            update = alpha_main.upd_matrix_match_shape(
                solved.update,
                parameter.shape,
            )
            with torch.no_grad():
                parameter[...] = parameter + update.float()
            if parameter.dtype is not torch.bfloat16:
                raise ODEBFContractError("canonical Alpha solve changed model dtype")
            deltas[weight_name] = update.detach().to(device="cpu")
            del layer_keys, current_z, targets, residual, update, solved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Preserve the pinned post-solve cache-update order, but explicitly
        # normalize the genuine joint keys to the FP32 cache contract.
        for layer_index, layer in enumerate(hparams.layers):
            layer_keys = alpha_main.compute_ks(
                model,
                tokenizer,
                normalized,
                hparams,
                layer,
                resolved_contexts,
            ).T
            keys32 = layer_keys.detach().to(
                device="cpu",
                dtype=ALPHA_SOLVE_DTYPE,
            )
            if keys32.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("canonical Alpha cache update is not joint B10")
            alpha_main.cache_c[layer_index] += keys32 @ keys32.T
            del layer_keys, keys32
    finally:
        with torch.no_grad():
            for name, parameter in weights.items():
                parameter.copy_(entry[name])
        if any(parameter.dtype is not torch.bfloat16 for parameter in weights.values()):
            raise ODEBFContractError("canonical Alpha solve did not preserve BF16 weights")
    if len(receipts) != len(hparams.layers):
        raise ODEBFContractError("canonical Alpha solve receipt count differs")
    return deltas, tuple(receipts)


def capture_native_and_wb_joint_endpoint(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector_path: Path,
    contexts: Sequence[Sequence[str]],
    *,
    projector_sha256: str,
    mutation_lock: threading.RLock,
    ledger: ComputeLedger,
    model_residual_tolerance: float,
) -> CapturedNativeWBEndpoint:
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main
    from easyeditor.util import nethook

    normalized = _normalize_requests(requests)
    request_order_sha256 = canonical_hash(
        [request["request_sha256"] for request in normalized]
    )
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model,
            f"{hparams.rewrite_module_tmp.format(layer)}.weight",
        )
        for layer in hparams.layers
    }
    if any(parameter.dtype is not torch.bfloat16 for parameter in weights.values()):
        raise ODEBFContractError("AlphaEdit touched parameter is not BF16")
    entry_weights = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in weights.items()
    }
    entry_sha256 = {name: tensor_sha256(value) for name, value in entry_weights.items()}
    pointers = {name: parameter.data_ptr() for name, parameter in weights.items()}
    normalized_layers = tuple(int(layer) for layer in hparams.layers)
    projector = torch.load(projector_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(projector, torch.Tensor)
        or projector.ndim != 3
        or projector.shape[0] != len(normalized_layers)
        or projector.dtype is not torch.float32
    ):
        raise ODEBFContractError("pinned AlphaEdit projector tensor contract differs")

    # Reproduce the pinned source loop in the ODE-BF adapter as the independent
    # Native endpoint. Wrappers observe its genuine joint call and memoize the
    # second cache-update key request so there is one underlying shared key
    # compute per layer. Every patched global is restored before returning.
    direct_z: list[torch.Tensor] = []
    key_by_layer: dict[int, torch.Tensor] = {}
    current_z_by_layer: dict[int, torch.Tensor] = {}
    target_backward_count = 0
    original_backward = torch.autograd.backward
    original_compute_z = alpha_main.compute_z
    original_compute_ks = alpha_main.compute_ks
    original_get_io = alpha_main.get_module_input_output_at_words
    missing = object()
    global_names = (
        "P",
        "P_loaded",
        "P_loaded_from",
        "cache_c",
        "cache_c_new",
        "CONTEXT_TEMPLATES_CACHE",
    )
    saved_globals = {
        name: getattr(alpha_main, name, missing)
        for name in global_names
    }

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_backward_count
        target_backward_count += 1
        ledger.increment("backward")
        ledger.increment("target_backward")
        return original_backward(*args, **kwargs)

    def captured_z(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_compute_z(*args, **kwargs)
        direct_z.append(value.detach().to(device="cpu"))
        return value

    def captured_keys(
        observed_model: Any,
        observed_tokenizer: Any,
        observed_requests: Any,
        observed_hparams: Any,
        layer: int,
        observed_contexts: Any,
    ) -> torch.Tensor:
        normalized_layer = int(layer)
        if normalized_layer in key_by_layer:
            return key_by_layer[normalized_layer].to(
                device=next(observed_model.parameters()).device
            )
        value = original_compute_ks(
            observed_model,
            observed_tokenizer,
            observed_requests,
            observed_hparams,
            layer,
            observed_contexts,
        )
        if value.ndim != 2 or value.shape[0] != BATCH_SIZE:
            raise ODEBFContractError("Native AlphaEdit key compute is not joint B10")
        key_by_layer[normalized_layer] = value.detach().to(device="cpu")
        return value

    def captured_get_io(*args: Any, **kwargs: Any) -> Any:
        value = original_get_io(*args, **kwargs)
        layer = normalized_layers[len(current_z_by_layer)]
        current_z_by_layer[layer] = value[1].detach().T.to(device="cpu")
        return value

    native_deltas: dict[str, torch.Tensor]
    dense_solve_receipts: tuple[AlphaDenseSolveReceipt, ...]
    history_dimension = projector.shape[1]
    try:
        alpha_main.P = projector
        alpha_main.P_loaded = True
        alpha_main.P_loaded_from = str(projector_path)
        alpha_main.cache_c = torch.zeros(
            (len(normalized_layers), history_dimension, history_dimension),
            dtype=projector.dtype,
            device="cpu",
        )
        alpha_main.cache_c_new = True
        alpha_main.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(list(contexts))
        alpha_main.compute_z = captured_z
        alpha_main.compute_ks = captured_keys
        alpha_main.get_module_input_output_at_words = captured_get_io
        torch.autograd.backward = counted_backward
        with mutation_lock, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            native_deltas, dense_solve_receipts = _execute_alphaedit_canonical_fp32_solve(
                alpha_main,
                model,
                tokenizer,
                normalized,
                hparams,
                contexts,
                weights,
                residual_tolerance=model_residual_tolerance,
            )
    finally:
        torch.autograd.backward = original_backward
        alpha_main.compute_z = original_compute_z
        alpha_main.compute_ks = original_compute_ks
        alpha_main.get_module_input_output_at_words = original_get_io
        for name, value in saved_globals.items():
            if value is missing:
                try:
                    delattr(alpha_main, name)
                except AttributeError:
                    pass
            else:
                setattr(alpha_main, name, value)
        with mutation_lock, torch.no_grad():
            for name, parameter in weights.items():
                parameter.copy_(entry_weights[name].to(device=parameter.device))
        torch.cuda.empty_cache()

    if len(direct_z) != BATCH_SIZE or set(key_by_layer) != set(normalized_layers):
        raise ODEBFContractError("Native AlphaEdit initialization receipt differs")
    if set(current_z_by_layer) != set(normalized_layers):
        raise ODEBFContractError("Native AlphaEdit residual capture differs")
    if set(native_deltas) != set(weights):
        raise ODEBFContractError("Native AlphaEdit touched parameter set differs")
    if (
        len(dense_solve_receipts) != len(normalized_layers)
        or any(
            receipt.reference != ALPHA_SOLVE_REFERENCE
            or receipt.solve_dtype != str(ALPHA_SOLVE_DTYPE)
            or receipt.joint_key_rank <= 1
            or not receipt.passed
            for receipt in dense_solve_receipts
        )
    ):
        raise ODEBFContractError("canonical Alpha dense solve receipt differs")
    ledger.increment("native_baseline_dense_delta_peak_live", len(native_deltas))
    ledger.increment(
        "native_baseline_dense_delta_bytes_peak",
        sum(value.numel() * value.element_size() for value in native_deltas.values()),
    )

    zs = torch.stack(direct_z, dim=1)
    native_candidates: dict[str, torch.Tensor] = {}
    wb_candidates: dict[str, torch.Tensor] = {}
    wb_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    key_hashes: list[tuple[int, str]] = []
    certificates: list[tuple[int, WoodburyCertificate]] = []
    factor_receipts: list[LayerFactorReceipt] = []
    try:
        for layer_index, layer in enumerate(normalized_layers):
            layer_keys = key_by_layer[layer].T
            current_z = current_z_by_layer[layer]
            targets = zs - current_z
            if layer_keys.ndim != 2 or layer_keys.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("AlphaEdit shared key is not a joint B10 factor")
            if targets.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("AlphaEdit residual is not joint B10")
            residual = targets / (len(normalized_layers) - layer_index)
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            parameter = weights[weight_name]
            p_layer = projector[layer_index]
            if p_layer.shape != (layer_keys.shape[0], layer_keys.shape[0]):
                raise ODEBFContractError("AlphaEdit projector/key shape differs")
            if not torch.isfinite(p_layer).all():
                raise ODEBFContractError("AlphaEdit projector contains non-finite values")
            p_device = p_layer.to(device=parameter.device)
            keys_device = layer_keys.to(device=parameter.device, dtype=p_device.dtype)
            residual_device = residual.to(device=parameter.device, dtype=p_device.dtype)
            wb = solve_alpha_woodbury(
                p_device,
                keys_device,
                history_keys=None,
                regularization=float(hparams.L2),
                projector_certificate=ProjectorCertificate(
                    projector_sha256,
                    1.0,
                    1.0,
                    "artifact-unverified",
                    1.0e-10,
                ),
                residual_tolerance=model_residual_tolerance,
            )
            factor = WaypointFactor(
                weight_name,
                layer,
                0,
                0,
                0,
                1.0,
                residual_device.detach(),
                wb.q.detach(),
            )
            wb_candidate, _ = assemble_effective_bf16(
                parameter,
                (factor,),
                row_block=64,
            )
            delta = alpha_main.upd_matrix_match_shape(
                native_deltas[weight_name].to(device=parameter.device),
                parameter.shape,
            )
            native_candidate = (
                parameter.detach().to(dtype=torch.float32)
                + delta.to(dtype=torch.float32)
            ).to(dtype=torch.bfloat16)
            del native_deltas[weight_name]
            if not torch.equal(native_candidate, wb_candidate):
                raise ODEBFContractError("Native AlphaEdit/WB BF16 parameter bytes differ")
            numerical_rank = int(torch.linalg.matrix_rank(keys_device.float()))
            factor_receipts.append(
                LayerFactorReceipt(
                    layer,
                    tuple(int(value) for value in keys_device.shape),
                    tuple(int(value) for value in residual_device.shape),
                    numerical_rank,
                    str(keys_device.dtype),
                    keys_device.device.type,
                )
            )
            key_hashes.append((layer, tensor_sha256(layer_keys)))
            certificates.append((layer, wb.certificate))
            native_candidates[weight_name] = native_candidate.detach().to(device="cpu")
            wb_candidates[weight_name] = wb_candidate.detach().to(device="cpu")
            wb_factors[weight_name] = (
                WaypointFactor(
                    weight_name,
                    layer,
                    0,
                    0,
                    0,
                    1.0,
                    residual.detach().to(device="cpu", dtype=torch.float64),
                    wb.q.detach().to(device="cpu", dtype=torch.float64),
                ),
            )
            del p_device, keys_device, residual_device, wb_candidate, native_candidate, delta
            torch.cuda.empty_cache()
    finally:
        native_deltas.clear()
        del projector
        torch.cuda.empty_cache()
    for name, parameter in weights.items():
        if parameter.data_ptr() != pointers[name] or tensor_sha256(parameter) != entry_sha256[name]:
            raise ODEBFContractError("AlphaEdit capture did not restore original W0 exactly")
    if target_backward_count <= 0:
        raise ODEBFContractError("AlphaEdit target path recorded no backward calls")
    initialization = JointInitializationReceipt(
        1,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE,
        0,
        len(normalized_layers),
        0,
        len(normalized_layers),
        len(normalized_layers),
        False,
        tuple(factor_receipts),
        request_order_sha256,
    )
    return CapturedNativeWBEndpoint(
        native_candidates,
        wb_candidates,
        wb_factors,
        entry_weights,
        entry_sha256,
        tuple(tensor_sha256(value) for value in direct_z),
        tuple(key_hashes),
        dense_solve_receipts,
        tuple(certificates),
        initialization,
        target_backward_count,
    )
