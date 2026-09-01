"""P1R23 Official and optimized direct-z Native comparison paths.

The optimized path changes execution only: direct-z remains the pinned
AlphaEdit ``compute_z`` solve and the five writer layers retain the original
source-order residual divisor and FP32 matrix equation.  Request/key/current-z
rows are merely accumulated in frozen-state microbatches before each solve.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import importlib
import io
import math
import threading
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_runtime import scalable_ordered_request_digest


def _alphaedit_dynamic_cache_snapshot(alpha_module: Any) -> dict[str, Any]:
    """Describe AlphaEdit's dynamic key outer-product cache without raw rows."""

    initialized = bool(getattr(alpha_module, "cache_c_new", False))
    cache = getattr(alpha_module, "cache_c", None)
    if not initialized or not isinstance(cache, torch.Tensor):
        return {
            "status": "UNINITIALIZED",
            "cache_c_new": initialized,
            "sha256": None,
            "shape": None,
            "dtype": None,
            "frobenius_norm": 0.0,
            "layer_frobenius_norm": [],
        }
    observed = cache.detach().to(device="cpu")
    if not observed.is_floating_point() or observed.ndim != 3:
        raise ODEBFStateError("Official Native AlphaEdit cache_c shape/dtype differs")
    layer_norm = [
        float(torch.linalg.vector_norm(observed[index].to(dtype=torch.float64)).item())
        for index in range(observed.shape[0])
    ]
    if not all(math.isfinite(item) for item in layer_norm):
        raise ODEBFStateError("Official Native AlphaEdit cache_c norm is nonfinite")
    return {
        "status": "INITIALIZED",
        "cache_c_new": initialized,
        "sha256": tensor_sha256(observed),
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "frobenius_norm": float(math.sqrt(sum(item * item for item in layer_norm))),
        "layer_frobenius_norm": layer_norm,
    }


def _alphaedit_static_projection_snapshot(alpha_module: Any) -> dict[str, Any]:
    """Separate the static null-space projector P from dynamic cache_c."""

    loaded = bool(getattr(alpha_module, "P_loaded", False))
    projection = getattr(alpha_module, "P", None)
    if not loaded or not isinstance(projection, torch.Tensor):
        raise ODEBFStateError("Official Native AlphaEdit static projector is absent")
    return {
        "cache_kind": "STATIC_NULLSPACE_PROJECTOR_P",
        "loaded": loaded,
        "loaded_from": str(getattr(alpha_module, "P_loaded_from", None)),
        "shape": list(projection.shape),
        "dtype": str(projection.dtype),
        "sha256": tensor_sha256(projection),
        "reload_requested": False,
    }


@contextlib.contextmanager
def _alphaedit_solver_key_dtype_adapter(alpha_module: Any | None = None) -> Any:
    """Promote captured BF16 keys for AlphaEdit's pinned FP32 linear solve."""

    if alpha_module is None:
        alpha_module = importlib.import_module(
            "easyeditor.models.alphaedit.AlphaEdit_main"
        )

    original = alpha_module.compute_ks
    receipt: dict[str, Any] = {
        "active": True,
        "call_count": 0,
        "input_dtypes": [],
        "output_dtype": "torch.float32",
        "value_transform": "dtype_promotion_only",
    }

    def promoted_compute_ks(*args: Any, **kwargs: Any) -> torch.Tensor:
        keys = original(*args, **kwargs)
        if not isinstance(keys, torch.Tensor) or not keys.is_floating_point():
            raise ODEBFContractError("AlphaEdit solver keys are not a floating tensor")
        if keys.dtype not in (torch.bfloat16, torch.float32):
            raise ODEBFContractError("AlphaEdit solver key dtype differs")
        receipt["call_count"] += 1
        receipt["input_dtypes"].append(str(keys.dtype))
        return keys.to(dtype=torch.float32)

    alpha_module.compute_ks = promoted_compute_ks
    try:
        yield receipt
    finally:
        alpha_module.compute_ks = original
        if alpha_module.compute_ks is not original:
            raise ODEBFContractError("AlphaEdit solver key adapter restore differs")


@dataclass(frozen=True, slots=True)
class ScalableNativeCapture:
    candidates: Mapping[str, torch.Tensor]
    entry_sha256: Mapping[str, str]
    request_order_sha256: str
    direct_z_sha256: tuple[str, ...]
    layer_receipts: tuple[Mapping[str, Any], ...]
    target_backward_count: int
    target_solver_max_steps: int
    edit_core_wall_seconds: float
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-optimized-native-k1-capture/v1",
            "request_order_sha256": self.request_order_sha256,
            "request_count": len(self.direct_z_sha256),
            "direct_z_sha256": list(self.direct_z_sha256),
            "entry_sha256": dict(self.entry_sha256),
            "candidate_sha256": {
                name: tensor_sha256(value)
                for name, value in sorted(self.candidates.items())
            },
            "layer_receipts": [dict(item) for item in self.layer_receipts],
            "target_backward_count": self.target_backward_count,
            "target_solver_max_steps": self.target_solver_max_steps,
            "target_algorithm": "PINNED_ALPHAEDIT_DIRECT_Z_COMPUTE_Z",
            "writer_algorithm": "PINNED_ALPHAEDIT_SOURCE_ORDER_FP32",
            "warm_one_gradient_substitution_count": 0,
            "accepted_materialization_count": 1,
            "edit_core_wall_seconds": self.edit_core_wall_seconds,
            "identity_sha256": self.identity_sha256,
        }


def _microbatches(
    requests: Sequence[Mapping[str, Any]], size: int
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    batch = tuple(requests)
    if (
        not batch
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or size > len(batch)
    ):
        raise ODEBFContractError("P1R23 Native microbatch geometry differs")
    return tuple(batch[start : start + size] for start in range(0, len(batch), size))


def _scalable_alpha_fp32_update(
    projector: torch.Tensor,
    joint_keys: torch.Tensor,
    covariance: torch.Tensor,
    residual: torch.Tensor,
    *,
    layer: int,
    regularization: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the pinned AlphaEdit equation with arbitrary joint-B columns."""

    p = projector.detach().to(device=device, dtype=torch.float32).contiguous()
    keys = joint_keys.detach().to(device=device, dtype=torch.float32).contiguous()
    cov = covariance.detach().to(device=device, dtype=torch.float32).contiguous()
    resid = residual.detach().to(device=device, dtype=torch.float32).contiguous()
    batch_size = keys.shape[1]
    if (
        p.ndim != 2
        or p.shape[0] != p.shape[1]
        or keys.shape != (p.shape[0], batch_size)
        or resid.ndim != 2
        or resid.shape[1] != batch_size
        or cov.shape != p.shape
        or batch_size <= 0
        or not all(torch.isfinite(item).all() for item in (p, keys, cov, resid))
        or not math.isfinite(float(regularization))
        or float(regularization) <= 0.0
    ):
        raise ODEBFContractError("P1R23 Native solve geometry differs")
    lam = torch.tensor(float(regularization), dtype=torch.float32, device=device)
    identity = torch.eye(p.shape[0], dtype=torch.float32, device=device)
    started = time.perf_counter()
    # Exact pinned source multiplication and addition order.
    a = p @ (keys @ keys.T + cov) + lam * identity
    b = p @ keys @ resid.T
    update = torch.linalg.solve(a, b)
    wall = time.perf_counter() - started
    relative = float(
        torch.linalg.norm(a @ update - b)
        / torch.clamp(
            torch.linalg.norm(a) * torch.linalg.norm(update) + torch.linalg.norm(b),
            min=torch.finfo(torch.float32).tiny,
        )
    )
    rank = int(torch.linalg.matrix_rank(keys).item())
    if (
        update.shape != (p.shape[0], resid.shape[0])
        or not torch.isfinite(update).all()
        or not math.isfinite(relative)
        or rank <= 1
    ):
        raise ODEBFContractError("P1R23 Native solve certificate failed")
    receipt = {
        "layer": int(layer),
        "batch_size": batch_size,
        "projector_sha256": tensor_sha256(projector),
        "key_sha256": tensor_sha256(joint_keys),
        "covariance_sha256": tensor_sha256(covariance),
        "residual_sha256": tensor_sha256(residual),
        "update_sha256": tensor_sha256(update),
        "key_rank": rank,
        "relative_residual": relative,
        "solve_dtype": str(torch.float32),
        "solve_device_class": device.type,
        "source_expression_order": "P@(K@K.T+C)+lambda*I;P@K@R.T",
        "wall_seconds": wall,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return update, receipt


def capture_optimized_native_k1(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    touched: Mapping[str, torch.nn.Parameter],
    request_microbatch_size: int,
    mutation_lock: threading.RLock,
    ledger: ComputeLedger,
) -> ScalableNativeCapture:
    """Compute one direct-z Native endpoint and restore exact W0."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    batch = tuple(requests)
    layers = tuple(int(item) for item in hparams.layers)
    if (
        not batch
        or tuple(touched) != tuple(
            f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in layers
        )
        or projector.shape[0] != len(layers)
    ):
        raise ODEBFContractError("P1R23 optimized Native entry differs")
    order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in batch]
    )
    entry = {
        name: parameter.detach().clone() for name, parameter in touched.items()
    }
    entry_sha = {name: tensor_sha256(value) for name, value in entry.items()}
    pointers = {name: int(parameter.data_ptr()) for name, parameter in touched.items()}
    direct_z: list[torch.Tensor] = []
    candidates: dict[str, torch.Tensor] = {}
    layer_receipts: list[dict[str, Any]] = []
    target_backward_count = 0
    original_backward = torch.autograd.backward
    target_layer = layers[-1]
    groups = _microbatches(batch, request_microbatch_size)
    resolved_contexts = alpha_main.get_context_templates(model, tokenizer)
    if canonical_hash(resolved_contexts) != canonical_hash(list(contexts)):
        raise ODEBFContractError("P1R23 optimized Native context differs")

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_backward_count
        target_backward_count += 1
        ledger.increment("backward")
        ledger.increment("target_backward")
        return original_backward(*args, **kwargs)

    started = time.perf_counter()
    try:
        torch.autograd.backward = counted_backward
        with mutation_lock, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            for request in batch:
                direct_z.append(
                    alpha_main.compute_z(
                        model,
                        tokenizer,
                        request,
                        hparams,
                        target_layer,
                        resolved_contexts,
                    )
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
            target = torch.stack(direct_z, dim=1)
            for layer_index, layer in enumerate(layers):
                key_parts: list[torch.Tensor] = []
                current_parts: list[torch.Tensor] = []
                for group in groups:
                    key_parts.append(
                        alpha_main.compute_ks(
                            model,
                            tokenizer,
                            list(group),
                            hparams,
                            layer,
                            resolved_contexts,
                        )
                        .T.detach()
                        .to(device="cpu", dtype=torch.float32)
                    )
                    current_parts.append(
                        alpha_main.get_module_input_output_at_words(
                            model,
                            tokenizer,
                            target_layer,
                            context_templates=[str(item["prompt"]) for item in group],
                            words=[str(item["subject"]) for item in group],
                            module_template=hparams.layer_module_tmp,
                            fact_token_strategy=hparams.fact_token,
                        )[1]
                        .T.detach()
                        .to(device="cpu", dtype=torch.float32)
                    )
                keys = torch.cat(key_parts, dim=1).contiguous()
                current = torch.cat(current_parts, dim=1).contiguous()
                if keys.shape[1] != len(batch) or current.shape[1] != len(batch):
                    raise ODEBFContractError("P1R23 optimized Native row coverage differs")
                residual = (target - current) / (len(layers) - layer_index)
                covariance = torch.zeros(
                    (keys.shape[0], keys.shape[0]), dtype=torch.float32
                )
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                update, solve = _scalable_alpha_fp32_update(
                    projector[layer_index],
                    keys,
                    covariance,
                    residual,
                    layer=layer,
                    regularization=float(hparams.L2),
                    device=touched[weight_name].device,
                )
                update = alpha_main.upd_matrix_match_shape(
                    update, touched[weight_name].shape
                )
                with torch.no_grad():
                    touched[weight_name].copy_(touched[weight_name] + update.float())
                candidates[weight_name] = touched[weight_name].detach().cpu().clone()
                solve.update(
                    {
                        "request_microbatch_size": request_microbatch_size,
                        "physical_microbatch_count": len(groups),
                        "source_order_residual_divisor": len(layers) - layer_index,
                        "streaming_key_current_z_accumulation": True,
                    }
                )
                solve["identity_sha256"] = canonical_hash(
                    {key: value for key, value in solve.items() if key != "identity_sha256"}
                )
                layer_receipts.append(solve)
    finally:
        torch.autograd.backward = original_backward
        with mutation_lock, torch.no_grad():
            for name, parameter in touched.items():
                parameter.copy_(entry[name])
    wall = time.perf_counter() - started
    if (
        len(direct_z) != len(batch)
        or len(candidates) != len(layers)
        or target_backward_count <= 0
        or any(
            int(touched[name].data_ptr()) != pointers[name]
            or tensor_sha256(touched[name]) != entry_sha[name]
            or touched[name].grad is not None
            for name in touched
        )
    ):
        raise ODEBFStateError("P1R23 optimized Native did not restore W0")
    payload = {
        "request_order_sha256": order,
        "direct_z_sha256": [tensor_sha256(value) for value in direct_z],
        "candidate_sha256": {
            name: tensor_sha256(value) for name, value in sorted(candidates.items())
        },
        "layer_receipt_sha256": [item["identity_sha256"] for item in layer_receipts],
        "target_backward_count": target_backward_count,
        "target_solver_max_steps": int(hparams.v_num_grad_steps),
        "request_microbatch_size": request_microbatch_size,
        "edit_core_wall_seconds": wall,
        "warm_one_gradient_substitution_count": 0,
    }
    return ScalableNativeCapture(
        candidates,
        entry_sha,
        order,
        tuple(payload["direct_z_sha256"]),
        tuple(layer_receipts),
        target_backward_count,
        int(hparams.v_num_grad_steps),
        wall,
        canonical_hash(payload),
    )


def materialize_native_candidates_once(
    touched: Mapping[str, torch.nn.Parameter],
    candidates: Mapping[str, torch.Tensor],
    entry_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if set(touched) != set(candidates) or set(touched) != set(entry_sha256):
        raise ODEBFContractError("P1R23 Native candidate inventory differs")
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    with torch.no_grad():
        for name, parameter in touched.items():
            candidate = candidates[name].to(device=parameter.device, dtype=parameter.dtype)
            if candidate.shape != parameter.shape or not torch.isfinite(candidate).all():
                raise ODEBFContractError("P1R23 Native candidate tensor differs")
            parameter.copy_(candidate)
    hashes = {name: tensor_sha256(value) for name, value in touched.items()}
    payload = {
        "schema": "ode-edit-s05-p1r23-native-one-materialization/v1",
        "accepted_materialization_count": 1,
        "parameter_sha256": hashes,
        "pointer_identity_preserved": all(
            int(touched[name].data_ptr()) == pointers[name] for name in touched
        ),
        "entry_sha256": dict(entry_sha256),
    }
    if not payload["pointer_identity_preserved"]:
        raise ODEBFStateError("P1R23 Native materialization pointer differs")
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def restore_native_entry(
    touched: Mapping[str, torch.nn.Parameter],
    entry: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(touched) != set(entry):
        raise ODEBFContractError("P1R23 Native restore inventory differs")
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    with torch.no_grad():
        for name, parameter in touched.items():
            parameter.copy_(entry[name].to(device=parameter.device, dtype=parameter.dtype))
    payload = {
        "schema": "ode-edit-s05-p1r23-native-w0-restore/v1",
        "parameter_sha256": {
            name: tensor_sha256(value) for name, value in touched.items()
        },
        "pointer_identity_preserved": all(
            int(touched[name].data_ptr()) == pointers[name] for name in touched
        ),
        "persistent_commit_count": 0,
    }
    if any(
        payload["parameter_sha256"][name] != tensor_sha256(entry[name])
        for name in touched
    ) or not payload["pointer_identity_preserved"]:
        raise ODEBFStateError("P1R23 Native W0 restore differs")
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_official_native_apply_impl(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    touched: Mapping[str, torch.nn.Parameter],
    reset_cache: bool = True,
    cache_history_width: int | None = None,
    cache_template: str | None = None,
    expected_native_compute_z_call_count: int | None = None,
    accepted_z_source: str | None = None,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Invoke the pinned Official EasyEdit AlphaEdit entry exactly once.

    ``reset_cache`` remains true by default for the pre-existing atomic call
    sites.  Sequential callers can bind the official AlphaEdit dynamic
    ``cache_c`` lifecycle explicitly and request a raw-free cache receipt by
    supplying ``cache_history_width``.
    """

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    entry_sha = {name: tensor_sha256(value) for name, value in touched.items()}
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    request_copy = [copy.deepcopy(dict(item)) for item in requests]
    original_compute_ks = alpha_main.compute_ks
    original_compute_z = getattr(alpha_main, "compute_z", None)
    if expected_native_compute_z_call_count is not None and original_compute_z is None:
        raise ODEBFContractError("Official AlphaEdit native compute_z interface is absent")
    original_backward = torch.autograd.backward
    target_backward_count = 0
    native_compute_z_call_count = 0

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_backward_count
        target_backward_count += 1
        return original_backward(*args, **kwargs)

    def counted_compute_z(*args: Any, **kwargs: Any) -> Any:
        nonlocal native_compute_z_call_count
        native_compute_z_call_count += 1
        if original_compute_z is None:
            raise ODEBFContractError("Official AlphaEdit native compute_z interface is absent")
        return original_compute_z(*args, **kwargs)

    cache_entry: dict[str, Any] | None = None
    if cache_history_width is not None:
        logical_width = int(cache_history_width)
        if logical_width < 0 or logical_width % len(requests) != 0:
            raise ODEBFContractError("Official Native cache history width differs")
        if bool(reset_cache) != (logical_width == 0):
            raise ODEBFContractError("Official Native reset/cache history contract differs")
        cache_entry = _alphaedit_dynamic_cache_snapshot(alpha_main)
        if logical_width > 0 and cache_entry["status"] != "INITIALIZED":
            raise ODEBFStateError("Official Native sequential cache entry is absent")
    started = time.perf_counter()
    torch.autograd.backward = counted_backward
    if original_compute_z is not None:
        alpha_main.compute_z = counted_compute_z
    try:
        with _alphaedit_solver_key_dtype_adapter(alpha_main) as adapter_receipt:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                returned_model, originals = alpha_main.apply_AlphaEdit_to_model(
                    model,
                    tokenizer,
                    request_copy,
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    cache_template=cache_template,
                    keep_original_weight=False,
                    reset_cache=bool(reset_cache),
                )
    finally:
        torch.autograd.backward = original_backward
        if original_compute_z is not None:
            alpha_main.compute_z = original_compute_z
    wall = time.perf_counter() - started
    adapter_receipt = {
        **adapter_receipt,
        "restored": alpha_main.compute_ks is original_compute_ks,
        "scope": "PINNED_OFFICIAL_ALPHAEDIT_APPLY_CALL_ONLY",
    }
    if adapter_receipt["call_count"] <= 0 or not adapter_receipt["restored"]:
        raise ODEBFContractError("P1R23 Official Native solver key adapter differs")
    if (
        expected_native_compute_z_call_count is not None
        and native_compute_z_call_count != int(expected_native_compute_z_call_count)
    ):
        raise ODEBFContractError("Official AlphaEdit native compute_z call count differs")
    if returned_model is not model or set(originals) != set(touched):
        raise ODEBFContractError("P1R23 Official Native return contract differs")
    if any(int(touched[name].data_ptr()) != pointers[name] for name in touched):
        raise ODEBFStateError("P1R23 Official Native pointer differs")
    cache_contract: dict[str, Any] | None = None
    if cache_history_width is not None:
        cache_exit = _alphaedit_dynamic_cache_snapshot(alpha_main)
        if cache_exit["status"] != "INITIALIZED":
            raise ODEBFStateError("Official Native sequential cache exit is absent")
        static_projection = _alphaedit_static_projection_snapshot(alpha_main)
        logical_width = int(cache_history_width)
        cache_contract = {
            "schema": "ode-edit-s05-official-alphaedit-dynamic-cache-contract/v1",
            "cache_kind": "DYNAMIC_ALPHAEDIT_KEY_OUTER_PRODUCT_CACHE_C",
            "reset_cache_requested": bool(reset_cache),
            "logical_history_width_at_entry": logical_width,
            "logical_history_width_after_append": logical_width + len(requests),
            "append_request_count": len(requests),
            "entry": cache_entry,
            "exit": cache_exit,
            "solver_consumed_entry_cache": bool(
                logical_width > 0
                and not reset_cache
                and cache_entry is not None
                and float(cache_entry["frobenius_norm"]) > 0.0
            ),
            "static_projection": static_projection,
            "static_projection_distinct_from_dynamic_cache": True,
        }
        cache_contract["identity_sha256"] = canonical_hash(cache_contract)
    payload = {
        "schema": "ode-edit-s05-p1r23-official-native-apply/v1",
        "entry_sha256": entry_sha,
        "edited_sha256": {
            name: tensor_sha256(value) for name, value in touched.items()
        },
        "original_copy_sha256": {
            name: tensor_sha256(value) for name, value in originals.items()
        },
        "request_order_sha256": scalable_ordered_request_digest(
            [str(item["request_sha256"]) for item in requests]
        ),
        "official_entrypoint": "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model",
        "direct_z_semantics": cache_template is None,
        "accepted_z_source": accepted_z_source,
        "accepted_z_cache_template_used": cache_template is not None,
        "native_alphaedit_compute_z_call_count": native_compute_z_call_count,
        "solver_key_dtype_adapter": adapter_receipt,
        "target_backward_count": target_backward_count,
        "alphaedit_dynamic_cache_contract": cache_contract,
        "edit_core_wall_seconds": wall,
        "raw_stdout_stderr_serialized_count": 0,
    }
    if any(
        payload["original_copy_sha256"][name] != entry_sha[name] for name in touched
    ):
        raise ODEBFStateError("P1R23 Official Native original copy differs")
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, originals


def run_official_native_apply(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    touched: Mapping[str, torch.nn.Parameter],
    reset_cache: bool = True,
    cache_history_width: int | None = None,
    cache_template: str | None = None,
    expected_native_compute_z_call_count: int | None = None,
    accepted_z_source: str | None = None,
    observer: Any | None = None,
    call_audit: Any | None = None,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Apply stock AlphaEdit with optional call-scoped observation only.

    Both optional contexts are absent for every existing caller.  They wrap
    the already pinned Official entrypoint and never replace update tensors,
    keys, projector, covariance, or dynamic-cache values.
    """

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    with contextlib.ExitStack() as stack:
        if observer is not None:
            stack.enter_context(observer.observe(alpha_main))
        if call_audit is not None:
            stack.enter_context(call_audit.observe(alpha_main))
        payload, originals = _run_official_native_apply_impl(
            model,
            tokenizer,
            requests,
            hparams,
            touched=touched,
            reset_cache=reset_cache,
            cache_history_width=cache_history_width,
            cache_template=cache_template,
            expected_native_compute_z_call_count=expected_native_compute_z_call_count,
            accepted_z_source=accepted_z_source,
        )
        if observer is not None:
            observer.capture_terminal(model, tokenizer)
    if observer is not None:
        observer.capture_weight_action(touched, originals)
        payload["layer_realization_observer"] = observer.payload(alpha_main)
    if call_audit is not None:
        payload["official_call_audit"] = call_audit.payload(alpha_main)
    if observer is not None or call_audit is not None:
        payload["identity_sha256"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "identity_sha256"}
        )
    return payload, originals


__all__ = [
    "ScalableNativeCapture",
    "capture_optimized_native_k1",
    "materialize_native_candidates_once",
    "restore_native_entry",
    "run_official_native_apply",
]
