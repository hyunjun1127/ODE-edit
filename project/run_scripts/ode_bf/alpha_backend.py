"""Read-only EasyEdit AlphaEdit capture and Native/Woodbury P0 identity path."""

from __future__ import annotations

import contextlib
import copy
import io
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
    woodbury_certificates: tuple[tuple[int, WoodburyCertificate], ...]
    initialization: JointInitializationReceipt
    target_backward_count: int


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

    # Run the pinned EasyEdit implementation once as the independent Native
    # endpoint.  Wrappers observe its genuine joint call and memoize the second
    # cache-update key request so there is one underlying shared key compute per
    # layer.  Every patched global is restored before returning.
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
            native_deltas = alpha_main.execute_AlphaEdit(
                model,
                tokenizer,
                normalized,
                hparams,
                cache_template=None,
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
        tuple(certificates),
        initialization,
        target_backward_count,
    )
