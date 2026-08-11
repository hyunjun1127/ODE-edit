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


def run_official_native_apply(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    touched: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Invoke the pinned Official EasyEdit AlphaEdit entry exactly once."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    entry_sha = {name: tensor_sha256(value) for name, value in touched.items()}
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    request_copy = [copy.deepcopy(dict(item)) for item in requests]
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        returned_model, originals = alpha_main.apply_AlphaEdit_to_model(
            model,
            tokenizer,
            request_copy,
            hparams,
            copy=False,
            return_orig_weights=True,
            cache_template=None,
            keep_original_weight=False,
            reset_cache=True,
        )
    wall = time.perf_counter() - started
    if returned_model is not model or set(originals) != set(touched):
        raise ODEBFContractError("P1R23 Official Native return contract differs")
    if any(int(touched[name].data_ptr()) != pointers[name] for name in touched):
        raise ODEBFStateError("P1R23 Official Native pointer differs")
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
        "direct_z_semantics": True,
        "edit_core_wall_seconds": wall,
        "raw_stdout_stderr_serialized_count": 0,
    }
    if any(
        payload["original_copy_sha256"][name] != entry_sha[name] for name in touched
    ):
        raise ODEBFStateError("P1R23 Official Native original copy differs")
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, originals


__all__ = [
    "ScalableNativeCapture",
    "capture_optimized_native_k1",
    "materialize_native_candidates_once",
    "restore_native_entry",
    "run_official_native_apply",
]
