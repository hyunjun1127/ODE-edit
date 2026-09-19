"""Batch-local FP32 endpoint ownership; no model or controller policy changes.

An endpoint has a private, authoritative CPU clone and a separate device clone.
PyTorch has no read-only tensor: callers MUST use ``readonly`` and must not let
borrowed device tensors (or aliases/graphs) escape that scope. Results produced
in a scope are provisional until its successful exit. Full header+bytes checks
at both boundaries detect NumPy/.data writes that version counters cannot see.
``evaluate_handle`` inside a scope does metadata/version/epoch checks only, so
weight upload, finite scans and full hashes do not scale with sequence count.

This is a single-threaded ownership protocol, not protection against concurrent
or malicious writes to private attributes. Model/input/source epoch providers
are caller-owned, and must include all nonselected parameter, buffer, hook,
mode, pack, mask, position and source revisions. A pointer/version model guard
is not a whole-model byte hash; existing independent hash checks still apply.
CPU-injected transfers test lifecycle, not real GPU parity or performance.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Callable
import uuid

import torch


TENSOR_SHA_CONVENTION = "sha256(ascii(tuple(shape)|torch.dtype|)+C-order-bytes)-v1"


def identity_sha256(value: Any) -> str:
    """Strict JSON identity; no repr/pointer fallback or mutable stored alias."""
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("IDENTITY_JSON_STRING_KEYS_REQUIRED")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Same header-and-bytes convention as the legacy alltoken oracle."""
    value = value.detach().cpu().contiguous()
    header = f"{tuple(value.shape)}|{value.dtype}|".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


class EndpointSessionError(RuntimeError):
    """Technical failure, never a native-success/fallback signal."""


class StaleEndpointError(EndpointSessionError):
    pass


class EndpointCorruptionError(EndpointSessionError):
    pass


@dataclass(frozen=True)
class RuntimePolicy:
    shape: tuple[int, ...]
    device: str | torch.device
    model_epoch: Any
    input_identity: Any
    source_identity: Any
    epoch_getter: Callable[[], Any]
    input_identity_getter: Callable[[], Any]
    source_identity_getter: Callable[[], Any]
    # Test seam: return the transferred tensor, not an already shared owner.
    transfer: Callable[[torch.Tensor, torch.device], torch.Tensor] | None = None


@dataclass(frozen=True)
class EndpointIdentity:
    session_id: str
    model_identity_sha256: str
    batch_identity_sha256: str
    model_epoch_sha256: str
    input_identity_sha256: str
    source_identity_sha256: str
    endpoint_id: str
    kind: str
    trial_identity_sha256: str | None
    weight_sha256: str
    sha_convention: str
    shape: tuple[int, ...]
    dtype: str
    source_device: str
    device: str


@dataclass(frozen=True)
class EndpointHandle:
    identity: EndpointIdentity
    source_version: int
    cpu_owner_version: int
    device_owner_version: int


@dataclass
class _Slot:
    handle: EndpointHandle
    source: torch.Tensor
    cpu: torch.Tensor
    device: torch.Tensor
    source_metadata: tuple
    cpu_metadata: tuple
    device_metadata: tuple
    callbacks: list[Callable[[], None]] = field(default_factory=list)
    borrows: int = 0


def _metadata(value: torch.Tensor) -> tuple:
    return (value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.storage_offset(), str(value.dtype),
            str(value.device), bool(value.requires_grad), value.grad_fn is None,
            value.grad is None)


class EndpointSession:
    """Exactly one native and at most one candidate inference-residency slot.

    Normal use::

        with EndpointSession(model_id, batch_id, policy) as session:
            native = session.bind_native(cpu_weight)
            with session.readonly(native):
                for cache in caches:
                    weight = session.evaluate_handle(native)
                    # Use the existing suffix/head shapes, no selected write.

    ``readonly`` also yields the device tensor for a single-call integration.
    Candidate replacement invalidates subscribed observations immediately.
    Gradient leaves are separate device-local clones, scoped by
    ``gradient_leaf``; all backward must finish before leaving that scope.
    No model Parameter is changed and no CUDA allocator cache is flushed.
    """

    def __init__(self, model_identity, batch_identity, runtime_policy: RuntimePolicy):
        if not isinstance(runtime_policy, RuntimePolicy):
            raise TypeError("ENDPOINT_RUNTIME_POLICY_REQUIRED")
        self.policy = runtime_policy
        self.shape = tuple(runtime_policy.shape)
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("ENDPOINT_POSITIVE_SHAPE_REQUIRED")
        self.device = torch.device(runtime_policy.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ENDPOINT_CPU_TEST_OR_CUDA_DEVICE_REQUIRED")
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("ENDPOINT_EXPLICIT_CUDA_DEVICE_INDEX_REQUIRED")
        for callback in (runtime_policy.epoch_getter,
                         runtime_policy.input_identity_getter,
                         runtime_policy.source_identity_getter):
            if not callable(callback):
                raise TypeError("ENDPOINT_IDENTITY_GETTERS_REQUIRED")
        self._session_id = uuid.uuid4().hex
        self._identity = dict(
            model_identity_sha256=identity_sha256(model_identity),
            batch_identity_sha256=identity_sha256(batch_identity),
            model_epoch_sha256=identity_sha256(runtime_policy.model_epoch),
            input_identity_sha256=identity_sha256(runtime_policy.input_identity),
            source_identity_sha256=identity_sha256(runtime_policy.source_identity))
        self._slots: dict[str, _Slot] = {}
        self._generation = 0
        self._closed = False
        self._gradient_leaves: dict[int, torch.Tensor] = {}
        self.work = {key: 0 for key in (
            "endpoint_binds", "inference_transfer_calls", "inference_transfer_bytes",
            "inference_h2d_calls", "inference_h2d_bytes", "device_owner_clones",
            "finite_validations", "full_validations", "fast_validations",
            "hash_calls", "hash_bytes", "hash_d2h_calls", "hash_d2h_bytes",
            "readonly_scopes", "evaluate_calls", "gradient_leaf_clones",
            "gradient_leaf_bytes", "candidate_releases", "invalidations")}
        self.work.update(hash_seconds=0., bind_seconds=0.)
        self._check_epoch()

    @property
    def closed(self):
        return self._closed

    @property
    def resident_slots(self):
        return len(self._slots)

    @property
    def resident_inference_bytes(self):
        return sum(s.device.numel() * s.device.element_size() for s in self._slots.values())

    @property
    def gradient_leaf_count(self):
        return len(self._gradient_leaves)

    def _check_epoch(self):
        if self.closed:
            raise StaleEndpointError("ENDPOINT_SESSION_CLOSED")
        for field_name, getter in (
                ("model_epoch_sha256", self.policy.epoch_getter),
                ("input_identity_sha256", self.policy.input_identity_getter),
                ("source_identity_sha256", self.policy.source_identity_getter)):
            if identity_sha256(getter()) != self._identity[field_name]:
                raise StaleEndpointError(f"ENDPOINT_{field_name.upper()}_CHANGED")

    def _slot(self, handle):
        if not isinstance(handle, EndpointHandle):
            raise TypeError("ENDPOINT_HANDLE_REQUIRED")
        slot = self._slots.get(handle.identity.kind)
        # Object identity rejects fabricated/reconstructed handles, even if
        # their public identity happens to equal a currently live handle.
        if slot is None or slot.handle is not handle:
            raise StaleEndpointError("ENDPOINT_HANDLE_RELEASED_REPLACED_OR_FOREIGN")
        return slot

    def _hash(self, value):
        size = value.numel() * value.element_size()
        self.work["hash_calls"] += 1
        self.work["hash_bytes"] += size
        if value.device.type == "cuda":
            self.work["hash_d2h_calls"] += 1
            self.work["hash_d2h_bytes"] += size
        started = time.perf_counter()
        try:
            return tensor_sha256(value)
        finally:
            self.work["hash_seconds"] += time.perf_counter() - started

    def validate_handle(self, handle, *, full=True):
        """Explicit phase-boundary validation; failure closes the whole session."""
        try:
            self._check_epoch()
            slot = self._slot(handle)
            self.work["full_validations" if full else "fast_validations"] += 1
            for name, value, expected in (
                    ("SOURCE", slot.source, slot.source_metadata),
                    ("CPU_OWNER", slot.cpu, slot.cpu_metadata),
                    ("DEVICE_OWNER", slot.device, slot.device_metadata)):
                if _metadata(value) != expected:
                    raise EndpointCorruptionError(f"ENDPOINT_{name}_METADATA_OR_VERSION_MUTATED")
                if full and self._hash(value) != handle.identity.weight_sha256:
                    raise EndpointCorruptionError(f"ENDPOINT_{name}_BYTES_MUTATED")
            return handle.identity
        except BaseException:
            self.close()
            raise

    def has_active_borrow(self, handle):
        return self._slot(handle).borrows > 0

    def _bind(self, kind, weight, trial_identity=None):
        started = time.perf_counter()
        try:
            self._check_epoch()
            if kind == "native" and self._slots:
                raise EndpointSessionError("ENDPOINT_NATIVE_ALREADY_BOUND")
            if kind == "candidate":
                if "native" not in self._slots:
                    raise EndpointSessionError("ENDPOINT_NATIVE_MUST_PRECEDE_CANDIDATE")
                if trial_identity is None:
                    raise ValueError("ENDPOINT_EXPLICIT_TRIAL_IDENTITY_REQUIRED")
                if self._gradient_leaves:
                    raise EndpointSessionError("ENDPOINT_CANDIDATE_BIND_DURING_GRADIENT")
                self.release_candidate()
            if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
                    or weight.dtype != torch.float32 or tuple(weight.shape) != self.shape
                    or weight.layout != torch.strided or weight.requires_grad
                    or weight.grad_fn is not None):
                raise ValueError("ENDPOINT_CPU_FP32_SHAPE_REQUIRED")
            self.work["finite_validations"] += 1
            if not bool(torch.isfinite(weight).all()):
                raise FloatingPointError("ENDPOINT_NONFINITE_WEIGHT")
            source_metadata = _metadata(weight)
            source_sha = self._hash(weight)
            # Never share storage with the controller, its .data, or NumPy.
            cpu = weight.detach().contiguous().clone()
            if self._hash(cpu) != source_sha:
                raise EndpointCorruptionError("ENDPOINT_COPY_SOURCE_RACE")
            transfer = self.policy.transfer or (lambda value, device: value.to(device))
            size = cpu.numel() * cpu.element_size()
            self.work["inference_transfer_calls"] += 1
            self.work["inference_transfer_bytes"] += size
            if self.device.type == "cuda":
                self.work["inference_h2d_calls"] += 1
                self.work["inference_h2d_bytes"] += size
            # This device-local clone isolates even an injected transfer's
            # externally retained result. It is not a second H2D upload.
            transfer_input = cpu.clone() if self.policy.transfer is not None else cpu
            transferred = transfer(transfer_input, self.device)
            if (not isinstance(transferred, torch.Tensor)
                    or transferred.device != self.device
                    or transferred.dtype != torch.float32
                    or tuple(transferred.shape) != self.shape):
                raise EndpointCorruptionError("ENDPOINT_TRANSFER_DEVICE_DTYPE_SHAPE")
            device = transferred.detach().contiguous().clone()
            self.work["device_owner_clones"] += 1
            if self._hash(device) != source_sha:
                raise EndpointCorruptionError("ENDPOINT_TRANSFER_BYTES_CHANGED")
            self._generation += 1
            identity = EndpointIdentity(
                session_id=self._session_id, **self._identity,
                endpoint_id=f"{kind}:{self._generation}", kind=kind,
                trial_identity_sha256=(identity_sha256(trial_identity)
                                       if kind == "candidate" else None),
                weight_sha256=source_sha, sha_convention=TENSOR_SHA_CONVENTION,
                shape=self.shape, dtype=str(torch.float32), source_device="cpu",
                device=str(self.device))
            handle = EndpointHandle(identity, weight._version, cpu._version, device._version)
            self._slots[kind] = _Slot(handle, weight, cpu, device, source_metadata,
                                      _metadata(cpu), _metadata(device))
            self.work["endpoint_binds"] += 1
            self.validate_handle(handle, full=True)
            return handle
        except BaseException:
            self.close()
            raise
        finally:
            self.work["bind_seconds"] += time.perf_counter() - started

    def bind_native(self, cpu_fp32_weight):
        return self._bind("native", cpu_fp32_weight)

    def bind_candidate(self, cpu_fp32_weight, trial_identity):
        return self._bind("candidate", cpu_fp32_weight, trial_identity)

    def cpu_snapshot(self, handle):
        """Owned copy; the authoritative CPU owner is never exposed."""
        self.validate_handle(handle, full=True)
        return self._slot(handle).cpu.clone()

    def evaluate_handle(self, handle):
        """Borrowed FP32 tensor; only legal within a checked ``readonly`` scope."""
        self.validate_handle(handle, full=False)
        slot = self._slot(handle)
        if not slot.borrows:
            raise EndpointSessionError("ENDPOINT_READONLY_SCOPE_REQUIRED")
        self.work["evaluate_calls"] += 1
        return slot.device

    @contextmanager
    def readonly(self, handle):
        """Full SHA at both boundaries; never publish provisional outputs.

        No inference/autograd mode is set here: caller-owned inference scopes
        must preserve their original numerical route. Device owner has no grad.
        Borrowed tensors must not escape, including through returned graphs.
        """
        self.validate_handle(handle, full=True)
        slot = self._slot(handle)
        slot.borrows += 1
        self.work["readonly_scopes"] += 1
        try:
            yield self.evaluate_handle(handle)
            self.validate_handle(handle, full=True)
        except BaseException:
            self.close()
            raise
        finally:
            slot.borrows -= 1

    @contextmanager
    def gradient_leaf(self, handle):
        """Separate leaf; not one of the two inference slots or observations.

        Complete backward inside the scope and return only detached owned
        copies. Session drops its reference and clears .grad on exit; Python
        cannot revoke a graph/leaf deliberately retained by a caller.
        """
        leaf = None
        try:
            with self.readonly(handle) as weight:
                leaf = weight.detach().clone().requires_grad_(True)
                self._gradient_leaves[id(leaf)] = leaf
                self.work["gradient_leaf_clones"] += 1
                self.work["gradient_leaf_bytes"] += leaf.numel() * leaf.element_size()
                yield leaf
        finally:
            if leaf is not None:
                leaf.grad = None
                leaf.requires_grad_(False)
                self._gradient_leaves.pop(id(leaf), None)

    def subscribe_invalidation(self, handle, callback):
        """Register an observation's no-throw close callback; returns unsubscribe."""
        self.validate_handle(handle, full=False)
        slot = self._slot(handle)
        slot.callbacks.append(callback)

        def unsubscribe():
            if callback in slot.callbacks:
                slot.callbacks.remove(callback)
        return unsubscribe

    def _release(self, kind):
        slot = self._slots.pop(kind, None)
        if slot is not None:
            self.work["invalidations"] += 1
            callbacks, slot.callbacks = slot.callbacks, []
            for callback in callbacks:
                callback()

    def release_candidate(self):
        slot = self._slots.get("candidate")
        if slot is not None:
            if slot.borrows:
                raise EndpointSessionError("ENDPOINT_CANNOT_RELEASE_BORROWED_CANDIDATE")
            self._release("candidate")
            self.work["candidate_releases"] += 1

    def close(self):
        """Idempotent release; telemetry remains available after technical failure."""
        self._closed = True
        try:
            self._release("candidate")
        finally:
            self._release("native")
            for leaf in self._gradient_leaves.values():
                leaf.grad = None
            self._gradient_leaves.clear()

    def __enter__(self):
        self._check_epoch()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
