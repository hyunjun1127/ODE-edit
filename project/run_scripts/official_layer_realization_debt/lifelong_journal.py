"""Append-only journals and exact editable-state checkpoint transactions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256

from .contracts import Method, ObservationBoundary
from .runtime import _method_module, _w0_identity


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _torch_save_once(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _cache_snapshot(method: Method) -> dict[str, Any]:
    if method is not Method.ALPHAEDIT:
        module = _method_module(method)
        cache = getattr(module, "COV_CACHE", None)
        if not isinstance(cache, dict):
            raise ObservationBoundary("MEMIT covariance cache is absent")
        rows = [
            [
                repr(key),
                list(value.shape),
                str(value.dtype),
                int(value.data_ptr()),
                int(value._version),
            ]
            for key, value in sorted(cache.items(), key=lambda item: repr(item[0]))
        ]
        return {
            "kind": "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE",
            "identity": canonical_hash(rows),
            "entries": rows,
            "request_history_width": 0,
        }
    module = _method_module(method)
    present = hasattr(module, "cache_c")
    value = getattr(module, "cache_c", None)
    tensor = (
        value.detach().to(device="cpu").contiguous().clone()
        if isinstance(value, torch.Tensor)
        else None
    )
    return {
        "kind": "DYNAMIC_ALPHAEDIT_CACHE_C",
        "cache_c_new": bool(getattr(module, "cache_c_new", False)),
        "present": present,
        "tensor": tensor,
        "sha256": None if tensor is None else tensor_sha256(tensor),
        "shape": None if tensor is None else list(tensor.shape),
        "dtype": None if tensor is None else str(tensor.dtype),
    }


def _restore_cache(method: Method, state: Mapping[str, Any]) -> dict[str, Any]:
    if method is not Method.ALPHAEDIT:
        observed = _cache_snapshot(method)
        if observed["identity"] != state["identity"]:
            raise ObservationBoundary("MEMIT static covariance cache changed")
        return {"exact": True, **observed}
    module = _method_module(method)
    module.cache_c_new = bool(state["cache_c_new"])
    if bool(state["present"]):
        tensor = state["tensor"]
        module.cache_c = (
            tensor.detach().to(device="cpu").contiguous().clone()
            if isinstance(tensor, torch.Tensor)
            else tensor
        )
    elif hasattr(module, "cache_c"):
        delattr(module, "cache_c")
    observed = _cache_snapshot(method)
    if (
        observed["cache_c_new"] != bool(state["cache_c_new"])
        or observed["present"] != bool(state["present"])
        or observed["sha256"] != state["sha256"]
    ):
        raise ObservationBoundary("AlphaEdit dynamic cache restore differs")
    return {
        "exact": True,
        "kind": observed["kind"],
        "cache_c_new": observed["cache_c_new"],
        "present": observed["present"],
        "sha256": observed["sha256"],
        "shape": observed["shape"],
        "dtype": observed["dtype"],
    }


@dataclass(slots=True)
class EditableStateSnapshot:
    method: Method
    weights: dict[str, torch.Tensor]
    weight_identity: dict[str, Any]
    cache: dict[str, Any]

    @classmethod
    def capture(
        cls,
        method: Method,
        touched: Mapping[str, torch.nn.Parameter],
    ) -> "EditableStateSnapshot":
        return cls(
            method=method,
            weights={
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in touched.items()
            },
            weight_identity=_w0_identity(touched),
            cache=_cache_snapshot(method),
        )

    def restore(
        self, touched: Mapping[str, torch.nn.Parameter]
    ) -> dict[str, Any]:
        from easyeditor.util.device import copy_to_param

        if set(touched) != set(self.weights):
            raise ObservationBoundary("editable-state weight inventory differs")
        with torch.no_grad():
            for name, parameter in touched.items():
                copy_to_param(parameter, self.weights[name])
        observed = _w0_identity(touched)
        if (
            observed["sha256"] != self.weight_identity["sha256"]
            or observed["pointers"] != self.weight_identity["pointers"]
        ):
            raise ObservationBoundary("editable-state W pointer/bytes restore differs")
        cache = _restore_cache(self.method, self.cache)
        return {"weights": {"exact": True, **observed}, "cache": cache}

    def release(self) -> None:
        self.weights.clear()
        tensor = self.cache.get("tensor")
        if isinstance(tensor, torch.Tensor):
            del tensor
        self.cache.clear()


def save_checkpoint(
    *,
    path: Path,
    method: Method,
    touched: Mapping[str, torch.nn.Parameter],
    batch_index: int,
    accepted_edit_count: int,
    stream_root: str,
    order_root: str,
    journal_chain_root: str,
    source: Mapping[str, Any],
    evaluator_identity: str,
) -> dict[str, Any]:
    state = EditableStateSnapshot.capture(method, touched)
    payload = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-checkpoint-state.v1",
        "batch_index": int(batch_index),
        "accepted_edit_count": int(accepted_edit_count),
        "weights": state.weights,
        "weight_sha256": state.weight_identity["sha256"],
        "weight_pointers_runtime_only": state.weight_identity["pointers"],
        "cache": state.cache,
        "stream_root": stream_root,
        "order_root": order_root,
        "journal_chain_root": journal_chain_root,
        "source": dict(source),
        "evaluator_identity": evaluator_identity,
        "rng": {
            "python": random.getstate(),
            # NumPy's native state contains an ndarray, which is not accepted
            # by torch.load(weights_only=True).  Store the same bytes in a
            # plain tensor so the recovery path stays fail-closed and does not
            # require unsafe pickle globals.
            "numpy": {
                "algorithm": str(np.random.get_state()[0]),
                "keys": torch.from_numpy(np.random.get_state()[1].copy()),
                "position": int(np.random.get_state()[2]),
                "has_gauss": int(np.random.get_state()[3]),
                "cached_gaussian": float(np.random.get_state()[4]),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }
    digest = _torch_save_once(path, payload)
    bytes_value = path.stat().st_size
    state.release()
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": bytes_value,
        "batch_index": int(batch_index),
        "accepted_edit_count": int(accepted_edit_count),
        "journal_chain_root": journal_chain_root,
    }


def restore_checkpoint(
    path: Path,
    *,
    method: Method,
    touched: Mapping[str, torch.nn.Parameter],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "odeedit.s06.layer-realization-debt.lifelong-checkpoint-state.v1"
    ):
        raise ObservationBoundary("checkpoint schema differs")
    snapshot = EditableStateSnapshot(
        method=method,
        weights=dict(payload["weights"]),
        weight_identity={
            "sha256": dict(payload["weight_sha256"]),
            "pointers": _w0_identity(touched)["pointers"],
            "dtypes": _w0_identity(touched)["dtypes"],
        },
        cache=dict(payload["cache"]),
    )
    restore = snapshot.restore(touched)
    rng = payload["rng"]
    random.setstate(tuple(rng["python"]))
    numpy_state = rng["numpy"]
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_state["keys"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = list(rng["torch_cuda"])
        if len(cuda_states) != torch.cuda.device_count():
            raise ObservationBoundary("checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(cuda_states)
    return {
        "status": "CHECKPOINT_RESTORE_PASS",
        "checkpoint_sha256": sha256_file(path),
        "batch_index": int(payload["batch_index"]),
        "accepted_edit_count": int(payload["accepted_edit_count"]),
        "journal_chain_root": str(payload["journal_chain_root"]),
        "restore": restore,
        "rng_restore_exact": True,
    }


def extend_hash_chain(previous: str, member_sha256: str, ordinal: int) -> str:
    return canonical_hash(
        {
            "previous": previous,
            "member_sha256": member_sha256,
            "ordinal": int(ordinal),
        }
    )


__all__ = [
    "EditableStateSnapshot",
    "extend_hash_chain",
    "restore_checkpoint",
    "save_checkpoint",
    "sha256_file",
    "write_json_once",
]
