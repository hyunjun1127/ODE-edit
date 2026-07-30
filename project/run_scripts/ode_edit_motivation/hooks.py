"""Repo-local hooks for observing and temporarily applying proposals."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from .contracts import (
    ContractError,
    EditRequest,
    MemitFactorProposal,
    ParameterRecord,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)


class WeightHashMismatchError(RuntimeError):
    """A parameter differs from the proposal's guarded entry snapshot."""


class ConcurrentWeightMutationError(RuntimeError):
    """A temporary application detected an unexpected in-context mutation."""


class TensorHashRuntimeError(RuntimeError):
    """Sanitized tensor-hash runtime failure with no upstream message text."""

    def __init__(self, *, phase: str, category: str, tensor: torch.Tensor):
        self.phase = phase
        self.category = category
        self.dtype = str(tensor.dtype)
        self.shape = tuple(int(dimension) for dimension in tensor.shape)
        self.device = str(tensor.device)
        self.numel = int(tensor.numel())
        super().__init__(f"tensor hash {phase} failed ({category})")


def _classify_tensor_hash_runtime_error(exc: RuntimeError) -> str:
    """Map an ephemeral runtime message to a fixed non-sensitive category."""

    message = str(exc).lower()
    if "out of memory" in message:
        return "cuda_out_of_memory" if "cuda" in message else "host_out_of_memory"
    if "illegal memory access" in message:
        return "cuda_illegal_memory_access"
    if "misaligned address" in message:
        return "cuda_misaligned_address"
    if "device-side assert" in message:
        return "cuda_device_assert"
    if "cublas" in message:
        return "cuda_cublas"
    if "cusolver" in message:
        return "cuda_cusolver"
    if "cuda" in message:
        return "cuda_runtime"
    if any(
        marker in message
        for marker in ("stride", "view size", "divisible", "unsupported")
    ):
        return "tensor_layout"
    if "allocat" in message and "memory" in message:
        return "host_out_of_memory"
    return "unknown_runtime"


def _raise_tensor_hash_runtime(
    *,
    phase: str,
    tensor: torch.Tensor,
    exc: RuntimeError,
) -> None:
    raise TensorHashRuntimeError(
        phase=phase,
        category=_classify_tensor_hash_runtime_error(exc),
        tensor=tensor,
    ) from None


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact tensor storage bytes in logical contiguous order."""

    # Reinterpret bytes only after crossing to CPU.  Some CUDA dtype/layout
    # combinations reject ``view(torch.uint8)`` even after ``contiguous()``,
    # and a device-side contiguous byte copy also creates avoidable GPU peak
    # memory.  Moving first preserves dtype and logical element order; the
    # following contiguous conversion defines the canonical byte stream.
    detached = tensor.detach()
    if detached.is_cuda:
        try:
            torch.cuda.synchronize(detached.device)
        except RuntimeError as exc:
            _raise_tensor_hash_runtime(
                phase="pre_copy_cuda_sync",
                tensor=detached,
                exc=exc,
            )
    try:
        host = detached.cpu()
    except RuntimeError as exc:
        _raise_tensor_hash_runtime(
            phase="device_to_cpu",
            tensor=detached,
            exc=exc,
        )
    try:
        canonical = host.contiguous()
    except RuntimeError as exc:
        _raise_tensor_hash_runtime(
            phase="cpu_contiguous",
            tensor=host,
            exc=exc,
        )
    try:
        value = canonical.view(torch.uint8).reshape(-1)
    except RuntimeError as exc:
        _raise_tensor_hash_runtime(
            phase="cpu_byte_view",
            tensor=canonical,
            exc=exc,
        )
    digest = hashlib.sha256()
    try:
        digest.update(value.numpy().tobytes(order="C"))
    except RuntimeError as exc:
        # PyTorch can be installed without NumPy (notably in minimal CPU test
        # environments).  Preserve the exact same byte stream in bounded
        # chunks; production EasyEdit environments take the fast NumPy path.
        if "Numpy is not available" not in str(exc):
            raise
        chunk_bytes = 1024 * 1024
        for start in range(0, value.numel(), chunk_bytes):
            digest.update(bytes(value[start : start + chunk_bytes].tolist()))
    return digest.hexdigest()


def assert_tensor_sha256_device_parity(device: torch.device | str) -> None:
    """Fail closed unless CUDA hashing matches canonical CPU bytes.

    The probe is deliberately tiny and contains no request/model data.  It
    covers every floating dtype used by the fixed Motivation runtimes and both
    contiguous and transposed layouts.
    """

    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("tensor hash device parity requires available CUDA")
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        base = torch.arange(12, dtype=dtype).reshape(3, 4)
        for cpu_value in (base, base.T):
            expected = tensor_sha256(cpu_value)
            observed = tensor_sha256(cpu_value.to(device=resolved))
            if observed != expected:
                raise ContractError(
                    f"tensor hash CPU/CUDA parity failed for {dtype}"
                )


def resolve_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    for candidate, parameter in model.named_parameters():
        if candidate == name:
            return parameter
    raise LookupError(name)


def resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    for candidate, module in model.named_modules():
        if candidate == name:
            return module
    raise LookupError(name)


def parameter_record(model: torch.nn.Module, name: str) -> ParameterRecord:
    parameter = resolve_parameter(model, name)
    return ParameterRecord(
        name=name,
        sha256=tensor_sha256(parameter),
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
    )


def _metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("hparams metadata contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, torch.dtype):
        return str(value)
    raise ContractError(f"unsupported hparams metadata type: {type(value).__name__}")


def freeze_hparams(hparams: Any) -> tuple[dict[str, Any], str]:
    if isinstance(hparams, dict):
        raw = hparams
    elif hasattr(hparams, "__dict__"):
        raw = vars(hparams)
    else:
        raise ContractError("hparams must be a mapping or expose __dict__")
    sanitized = {
        str(key): _metadata_value(value)
        for key, value in sorted(raw.items(), key=lambda item: str(item[0]))
        if not str(key).startswith("_")
    }
    encoded = canonical_json(sanitized).encode("utf-8")
    return sanitized, sha256_bytes(encoded)


def capture_snapshot(
    model: torch.nn.Module,
    *,
    model_id: str,
    requests: Iterable[EditRequest],
    context_id: str,
    hparams: Any,
    weight_names: Iterable[str],
    provenance_ids: Iterable[str] = (),
) -> SnapshotManifest:
    """Freeze all inputs that must agree for a same-snapshot proposal."""

    request_ids = tuple(request.request_id for request in requests)
    if not request_ids:
        raise ContractError("snapshot requires at least one request")
    _, hparams_hash = freeze_hparams(hparams)
    records = tuple(
        sorted(
            (parameter_record(model, name) for name in weight_names),
            key=lambda record: record.name,
        )
    )
    return SnapshotManifest(
        model_id=model_id,
        context_id=context_id,
        request_ids=request_ids,
        hparams_sha256=hparams_hash,
        parameters=records,
        provenance_ids=tuple(sorted(set(provenance_ids))),
    )


def assert_snapshot_current(model: torch.nn.Module, snapshot: SnapshotManifest) -> None:
    mismatches: list[str] = []
    for record in snapshot.parameters:
        parameter = resolve_parameter(model, record.name)
        actual_hash = tensor_sha256(parameter)
        if (
            actual_hash != record.sha256
            or tuple(parameter.shape) != record.shape
            or str(parameter.dtype) != record.dtype
        ):
            mismatches.append(
                f"{record.name}: expected {record.sha256}/{record.shape}/{record.dtype}, "
                f"got {actual_hash}/{tuple(parameter.shape)}/{parameter.dtype}"
            )
    if mismatches:
        raise WeightHashMismatchError("model no longer matches proposal snapshot:\n" + "\n".join(mismatches))


class TemporaryLowRankApplication:
    """No-dense temporary application using an in-place low-rank ``addmm``.

    The entry guard rejects stale proposals.  The exit guard notices any
    unexpected weight mutation while the context is active.  Full backups are
    deliberate: subtracting a floating-point update is not bit-exact rollback.

    Factors are cast to the parameter dtype before ``addmm``.  This minimizes
    temporary memory, but it is not numerically identical to EasyEdit's native
    materialize-then-``.float()`` insertion.  Use
    :class:`TemporaryExactMemitApplication` for MV-0 parity.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        proposal: MemitFactorProposal,
        *,
        scale: float = 1.0,
        parameter_resolver: Callable[[torch.nn.Module, str], torch.nn.Parameter] = resolve_parameter,
    ) -> None:
        if not math.isfinite(scale):
            raise ContractError("temporary factor scale must be finite")
        self.model = model
        self.proposal = proposal
        self.scale = scale
        self.parameter_resolver = parameter_resolver
        self._backups: dict[str, torch.Tensor] = {}
        self._applied_hashes: dict[str, str] = {}
        self._active = False

    def _apply_factor(
        self,
        parameter: torch.nn.Parameter,
        factor: Any,
    ) -> None:
        left = factor.left.to(device=parameter.device, dtype=parameter.dtype)
        right = factor.right.to(device=parameter.device, dtype=parameter.dtype)
        parameter.addmm_(
            left,
            right.transpose(0, 1),
            beta=1.0,
            alpha=self.scale,
        )

    @property
    def applied_hashes(self) -> dict[str, str]:
        return dict(self._applied_hashes)

    def __enter__(self) -> "TemporaryLowRankApplication":
        if self._active:
            raise RuntimeError("temporary low-rank application is already active")
        assert_snapshot_current(self.model, self.proposal.snapshot)

        changed: list[str] = []
        try:
            with torch.no_grad():
                for factor in self.proposal.factors:
                    parameter = self.parameter_resolver(self.model, factor.weight_name)
                    current_hash = tensor_sha256(parameter)
                    if current_hash != factor.expected_weight_sha256:
                        raise WeightHashMismatchError(
                            f"{factor.weight_name}: expected {factor.expected_weight_sha256}, "
                            f"got {current_hash}"
                        )
                    self._backups[factor.weight_name] = parameter.detach().clone()
                    changed.append(factor.weight_name)
                    self._apply_factor(parameter, factor)
                    self._applied_hashes[factor.weight_name] = tensor_sha256(parameter)
        except BaseException:
            with torch.no_grad():
                for name in changed:
                    self.parameter_resolver(self.model, name).copy_(self._backups[name])
            self._backups.clear()
            self._applied_hashes.clear()
            raise
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_value: BaseException | None, traceback: Any) -> bool:
        if not self._active:
            return False
        mutations: list[str] = []
        for name, applied_hash in self._applied_hashes.items():
            actual_hash = tensor_sha256(self.parameter_resolver(self.model, name))
            if actual_hash != applied_hash:
                mutations.append(f"{name}: expected active hash {applied_hash}, got {actual_hash}")

        rollback_errors: list[str] = []
        with torch.no_grad():
            for name, backup in self._backups.items():
                parameter = self.parameter_resolver(self.model, name)
                parameter.copy_(backup)
                expected = self.proposal.snapshot.parameter(name).sha256
                actual = tensor_sha256(parameter)
                if actual != expected:
                    rollback_errors.append(f"{name}: expected rollback hash {expected}, got {actual}")

        self._active = False
        self._backups.clear()
        self._applied_hashes.clear()
        messages = mutations + rollback_errors
        if messages:
            error = ConcurrentWeightMutationError(
                "temporary proposal guard failed; original snapshot was restored:\n"
                + "\n".join(messages)
            )
            if exc_value is not None:
                if hasattr(exc_value, "add_note"):
                    exc_value.add_note(str(error))
                return False
            raise error
        return False


def _match_update_shape(
    update: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    expected = tuple(shape)
    if tuple(update.shape) == expected:
        return update
    if tuple(update.transpose(0, 1).shape) == expected:
        return update.transpose(0, 1)
    raise ContractError(
        f"MEMIT update shape {tuple(update.shape)} cannot match weight shape {expected}"
    )


class TemporaryExactMemitApplication(TemporaryLowRankApplication):
    """MV-0 parity mode matching EasyEdit's native insertion numerics.

    This mode materializes the dense update on the parameter device.  For the
    Llama/Qwen weight orientation, EasyEdit first evaluates
    ``adj_k @ resid.T`` and only then transposes the result to the writable
    weight shape.  ``native_update_transposed`` preserves that raw factor
    orientation so this adapter invokes the same GEMM geometry instead of
    merely evaluating the mathematically equivalent ``resid @ adj_k.T``.
    It then performs ``weight[...] += update.float()`` just like
    ``apply_memit_to_model``.
    Unlike :class:`TemporaryLowRankApplication`, it is intentionally not a
    no-dense path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        proposal: MemitFactorProposal,
        *,
        parameter_resolver: Callable[
            [torch.nn.Module, str], torch.nn.Parameter
        ] = resolve_parameter,
    ) -> None:
        super().__init__(
            model,
            proposal,
            scale=1.0,
            parameter_resolver=parameter_resolver,
        )

    def _apply_factor(
        self,
        parameter: torch.nn.Parameter,
        factor: Any,
    ) -> None:
        # Preserve factor dtype for the matmul.  EasyEdit moves factors to the
        # weight device without casting, materializes the update, then casts
        # that update to float32 before the in-place addition.
        left = factor.left.to(device=parameter.device)
        right = factor.right.to(device=parameter.device)
        if factor.native_update_transposed:
            # Canonical factor orientation is (resid, adj_k), but EasyEdit's
            # raw call is adj_k @ resid.T followed by shape matching.
            dense_update = right @ left.transpose(0, 1)
        else:
            dense_update = left @ right.transpose(0, 1)
        dense_update = _match_update_shape(dense_update, parameter.shape)
        parameter[...] += dense_update.float()


def _copy_tensors(value: Any, *, detach: bool, clone: bool) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.detach() if detach else value
        return result.clone() if clone else result
    if isinstance(value, dict):
        return type(value)(
            (key, _copy_tensors(item, detach=detach, clone=clone))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_copy_tensors(item, detach=detach, clone=clone) for item in value)
    if isinstance(value, list):
        return [_copy_tensors(item, detach=detach, clone=clone) for item in value]
    return value


@dataclass(slots=True)
class ForwardCapture:
    """Small repo-local forward hook with modern kwargs support."""

    model: torch.nn.Module
    layer_name: str
    retain_input: bool = True
    retain_output: bool = True
    detach: bool = True
    clone: bool = True
    edit_output: Callable[[Any, str], Any] | None = None
    input: Any = field(init=False, default=None)
    output: Any = field(init=False, default=None)
    _handle: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.input = None
        self.output = None
        self._handle = None

    def _hook(self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        del module
        if self.retain_input:
            captured_input: Any
            if args:
                captured_input = args[0] if len(args) == 1 else args
            else:
                captured_input = kwargs.get("hidden_states", kwargs)
            self.input = _copy_tensors(captured_input, detach=self.detach, clone=self.clone)
        if self.edit_output is not None:
            output = self.edit_output(output, self.layer_name)
        if self.retain_output:
            self.output = _copy_tensors(output, detach=self.detach, clone=self.clone)
        return output

    def __enter__(self) -> "ForwardCapture":
        if self._handle is not None:
            raise RuntimeError("forward capture is already active")
        module = resolve_module(self.model, self.layer_name)
        try:
            self._handle = module.register_forward_hook(self._hook, with_kwargs=True)
        except TypeError:
            def legacy_hook(
                hooked_module: torch.nn.Module,
                args: tuple[Any, ...],
                output: Any,
            ) -> Any:
                return self._hook(hooked_module, args, {}, output)

            self._handle = module.register_forward_hook(legacy_hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
