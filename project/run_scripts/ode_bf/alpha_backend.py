"""Read-only EasyEdit AlphaEdit capture and Native/Woodbury P0 identity path."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import math
import random
import threading
import time
from itertools import combinations_with_replacement
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
FOUR_PATH_ORDER = ("N32", "D32", "W32", "W64")
FOUR_PATH_REFERENCE = "ODE-BF dense/Woodbury association diagnostic R2"


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
    wall_seconds: float
    gpu_seconds: float


@dataclass(frozen=True, slots=True)
class BF16ComparisonReceipt:
    reference_path: str
    candidate_path: str
    byte_exact: bool
    mismatch_count: int
    element_count: int
    mismatch_fraction: float
    ordered_ulp_max: int
    ordered_ulp_p50: float
    ordered_ulp_p95: float
    ordered_ulp_p99: float
    percentile_population: str
    ordered_ulp_histogram: tuple[tuple[str, int], ...]
    max_abs_difference: float


@dataclass(frozen=True, slots=True)
class FourPathSolveReceipt:
    layer: int
    path: str
    reference: str
    source_identity_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    source_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    source_dtypes: tuple[tuple[str, str], ...]
    solve_shape: tuple[int, int]
    update_shape: tuple[int, int]
    solve_dtype: str
    assembler_dtype: str
    solve_device_class: str
    joint_rank: int
    residual_rhs_kind: str
    normalized_backward_residual: float
    system_norm_estimate: float
    system_norm_kind: str
    condition_estimate: float
    condition_kind: str
    factor_relative_error_to_d32: float | None
    update_relative_error_to_d32: float
    endpoint_relative_to_n32_edit_error: float
    finite: bool
    certificate_passed: bool
    wall_seconds: float
    gpu_seconds: float
    allocated_bytes_after: int


@dataclass(frozen=True, slots=True)
class FourPathLayerReceipt:
    layer: int
    weight_name_sha256: str
    request_order_sha256: str
    source_identity_sha256: str
    path_receipts: tuple[FourPathSolveReceipt, ...]
    comparisons_to_n32: tuple[BF16ComparisonReceipt, ...]
    pairwise_comparisons: tuple[BF16ComparisonReceipt, ...]
    adjacent_comparisons: tuple[BF16ComparisonReceipt, ...]
    first_adjacent_byte_boundary: str | None


@dataclass(slots=True)
class FourPathLayerResult:
    candidates: dict[str, torch.Tensor]
    factors: dict[str, tuple[WaypointFactor, ...]]
    receipt: FourPathLayerReceipt
    w64_certificate: WoodburyCertificate


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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
    wall_started = time.perf_counter()

    # Keep the exact pinned source multiplication and addition order.
    a32 = p32 @ (k32 @ k32.T + c32) + lambda32 * identity32
    b32 = p32 @ k32 @ r32.T
    update32 = torch.linalg.solve(a32, b32)
    gpu_seconds = 0.0
    if start_event is not None and end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    wall_seconds = time.perf_counter() - wall_started
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
        wall_seconds,
        gpu_seconds,
    )
    if not passed:
        raise ODEBFContractError("Alpha dense solve certificate failed")
    return AlphaDenseSolveResult(update32, receipt)


def _zero_tensor_sha256(shape: Sequence[int], dtype: torch.dtype) -> str:
    if dtype is not torch.float32:
        raise ODEBFContractError("zero covariance hash dtype differs")
    element_count = math.prod(int(value) for value in shape)
    if element_count < 0:
        raise ODEBFContractError("zero covariance hash shape differs")
    remaining = element_count * torch.empty((), dtype=dtype).element_size()
    block = b"\x00" * (1024 * 1024)
    digest = hashlib.sha256()
    while remaining:
        length = min(remaining, len(block))
        digest.update(block[:length])
        remaining -= length
    return digest.hexdigest()


def _ordered_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is not torch.bfloat16:
        raise ODEBFContractError("ordered ULP input is not BF16")
    bits = tensor.detach().to(device="cpu").contiguous().view(torch.int16)
    unsigned = torch.bitwise_and(bits.to(dtype=torch.int32), 0xFFFF)
    negative = torch.bitwise_and(unsigned, 0x8000) != 0
    magnitude = torch.bitwise_and(unsigned, 0x7FFF)
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + unsigned)


def bf16_comparison_receipt(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_path: str,
    candidate_path: str,
) -> BF16ComparisonReceipt:
    if (
        reference.dtype is not torch.bfloat16
        or candidate.dtype is not torch.bfloat16
        or tuple(reference.shape) != tuple(candidate.shape)
    ):
        raise ODEBFContractError("BF16 endpoint comparison geometry differs")
    reference_cpu = reference.detach().to(device="cpu").contiguous()
    candidate_cpu = candidate.detach().to(device="cpu").contiguous()
    ulp = torch.abs(_ordered_bf16(reference_cpu) - _ordered_bf16(candidate_cpu)).view(-1)
    mismatch = ulp != 0
    mismatch_count = int(mismatch.sum().item())
    element_count = int(ulp.numel())
    differing = ulp[mismatch]
    if differing.numel():
        percentiles = torch.quantile(
            differing.to(dtype=torch.float64),
            torch.tensor((0.50, 0.95, 0.99), dtype=torch.float64),
            interpolation="nearest",
        )
        p50, p95, p99 = (float(value) for value in percentiles)
        ulp_max = int(differing.max().item())
    else:
        p50 = p95 = p99 = 0.0
        ulp_max = 0
    histogram = (
        ("0", int((ulp == 0).sum().item())),
        ("1", int((ulp == 1).sum().item())),
        ("2", int((ulp == 2).sum().item())),
        ("3-4", int(((ulp >= 3) & (ulp <= 4)).sum().item())),
        ("5-8", int(((ulp >= 5) & (ulp <= 8)).sum().item())),
        ("9-16", int(((ulp >= 9) & (ulp <= 16)).sum().item())),
        ("17+", int((ulp >= 17).sum().item())),
    )
    max_abs = float(
        torch.max(
            torch.abs(reference_cpu.to(dtype=torch.float32) - candidate_cpu.to(dtype=torch.float32))
        ).item()
    ) if element_count else 0.0
    return BF16ComparisonReceipt(
        reference_path,
        candidate_path,
        mismatch_count == 0,
        mismatch_count,
        element_count,
        0.0 if element_count == 0 else mismatch_count / element_count,
        ulp_max,
        p50,
        p95,
        p99,
        "mismatching-elements",
        histogram,
        max_abs,
    )


def _spectral_norm_power_estimate(matrix: torch.Tensor, *, iterations: int = 6) -> float:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or iterations <= 0:
        raise ODEBFContractError("spectral norm estimate contract differs")
    dimension = matrix.shape[0]
    vector = torch.linspace(
        1.0,
        2.0,
        dimension,
        dtype=matrix.dtype,
        device=matrix.device,
    ).reshape(-1, 1)
    vector = vector / torch.linalg.vector_norm(vector)
    for _ in range(iterations):
        forward = matrix @ vector
        adjoint = matrix.T @ forward
        norm = torch.linalg.vector_norm(adjoint)
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ODEBFContractError("spectral norm power iteration failed")
        vector = adjoint / norm
    estimate = float(torch.linalg.vector_norm(matrix @ vector).item())
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ODEBFContractError("spectral norm estimate is invalid")
    return estimate


def _normalized_backward_residual(
    matrix: torch.Tensor,
    solution: torch.Tensor,
    rhs: torch.Tensor,
    *,
    system_norm_estimate: float,
) -> float:
    numerator = torch.linalg.norm(matrix @ solution - rhs)
    denominator = (
        system_norm_estimate * torch.linalg.norm(solution)
        + torch.linalg.norm(rhs)
    )
    value = float(
        numerator
        / torch.clamp(
            denominator,
            min=torch.finfo(solution.dtype).tiny,
        )
    )
    return value


def _timed_call(device: torch.device, function: Any) -> tuple[Any, float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        started_event = torch.cuda.Event(enable_timing=True)
        finished_event = torch.cuda.Event(enable_timing=True)
        started_event.record()
    else:
        started_event = finished_event = None
    started = time.perf_counter()
    result = function()
    if started_event is not None and finished_event is not None:
        finished_event.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(started_event.elapsed_time(finished_event)) / 1000.0
    else:
        gpu_seconds = 0.0
    return result, time.perf_counter() - started, gpu_seconds


def build_four_path_layer_diagnostic(
    *,
    layer: int,
    weight_name: str,
    entry_weight: torch.Tensor,
    native_matched_update: torch.Tensor,
    projector: torch.Tensor,
    joint_keys: torch.Tensor,
    residual: torch.Tensor,
    regularization: float,
    projector_sha256: str,
    request_order_sha256: str,
    residual_tolerance: float,
    native_receipt: AlphaDenseSolveReceipt,
    row_block: int = 64,
) -> FourPathLayerResult:
    """Construct N32/D32/W32/W64 from one immutable P0 layer identity."""

    if entry_weight.dtype is not torch.bfloat16 or entry_weight.ndim != 2:
        raise ODEBFContractError("four-path entry weight is not a BF16 matrix")
    if native_matched_update.shape != entry_weight.shape:
        raise ODEBFContractError("four-path Native update orientation differs")
    geometry = validate_joint_alpha_solve_geometry(
        projector.shape,
        joint_keys.shape,
        projector.shape,
        residual.shape,
    )
    if tuple(entry_weight.shape) != (
        geometry.residual_shape[0],
        geometry.projector_shape[0],
    ):
        raise ODEBFContractError("four-path weight/source geometry differs")
    if native_receipt.geometry != geometry or not native_receipt.passed:
        raise ODEBFContractError("four-path Native dense receipt differs")
    tolerance = float(residual_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ODEBFContractError("four-path residual tolerance differs")
    lam = float(regularization)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ODEBFContractError("four-path regularization differs")

    guarded = (entry_weight, native_matched_update, projector, joint_keys, residual)
    pointers = tuple(value.data_ptr() for value in guarded)
    versions = tuple(value._version for value in guarded)
    gradients = tuple(
        None
        if value.grad is None
        else (value.grad.data_ptr(), value.grad._version, tensor_sha256(value.grad))
        for value in guarded
    )
    requires_grad = tuple(value.requires_grad for value in guarded)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        torch.cuda.get_rng_state(entry_weight.device).clone()
        if entry_weight.device.type == "cuda"
        else None
    )
    device = entry_weight.device
    p32 = projector.detach().to(device=device, dtype=torch.float32).contiguous()
    k32 = joint_keys.detach().to(device=device, dtype=torch.float32).contiguous()
    r32 = residual.detach().to(device=device, dtype=torch.float32).contiguous()
    lambda32 = torch.tensor(lam, dtype=torch.float32, device=device)
    source_hashes = (
        ("W0", tensor_sha256(entry_weight)),
        ("P", tensor_sha256(projector)),
        ("K", tensor_sha256(joint_keys)),
        ("C", _zero_tensor_sha256(projector.shape, torch.float32)),
        ("R", tensor_sha256(residual)),
        ("lambda", tensor_sha256(lambda32)),
    )
    source_shapes = (
        ("W0", tuple(int(value) for value in entry_weight.shape)),
        ("P", tuple(int(value) for value in projector.shape)),
        ("K", tuple(int(value) for value in joint_keys.shape)),
        ("C", tuple(int(value) for value in projector.shape)),
        ("R", tuple(int(value) for value in residual.shape)),
        ("lambda", ()),
    )
    source_dtypes = (
        ("W0", str(entry_weight.dtype)),
        ("P", str(projector.dtype)),
        ("K", str(joint_keys.dtype)),
        ("C", str(torch.float32)),
        ("R", str(residual.dtype)),
        ("lambda", str(torch.float32)),
    )
    source_identity = canonical_hash(
        {
            "layer": int(layer),
            "request_order_sha256": request_order_sha256,
            "source_hashes": source_hashes,
            "source_shapes": source_shapes,
            "source_dtypes": source_dtypes,
        }
    )

    gram32 = k32 @ k32.T
    # P0 C is exact zero.  Scalar zero addition retains the pinned source
    # expression order without retaining another dense zero matrix.
    gram_plus_c32 = gram32 + torch.zeros((), dtype=torch.float32, device=device)
    identity32 = torch.eye(geometry.projector_shape[0], dtype=torch.float32, device=device)
    a32 = p32 @ gram_plus_c32 + lambda32 * identity32
    g32 = p32 @ k32
    system_norm = _spectral_norm_power_estimate(a32)
    small32 = lambda32 * torch.eye(BATCH_SIZE, dtype=torch.float32, device=device) + k32.T @ g32
    condition_estimate = float(torch.linalg.cond(small32.detach().to(device="cpu", dtype=torch.float64)))
    if not math.isfinite(condition_estimate):
        raise ODEBFContractError("four-path condition estimate is non-finite")
    joint_rank = int(torch.linalg.matrix_rank(k32).item())

    q_d32, d_wall, d_gpu = _timed_call(
        device,
        lambda: torch.linalg.solve(a32, g32),
    )

    def solve_w32() -> torch.Tensor:
        rhs32 = k32.T @ g32
        solution32, info32 = torch.linalg.solve_ex(small32, rhs32)
        if int(info32.max().item()) != 0:
            raise ODEBFContractError("W32 small-system LU failed")
        return (g32 - g32 @ solution32) / lambda32

    q_w32, w32_wall, w32_gpu = _timed_call(device, solve_w32)
    w64, w64_wall, w64_gpu = _timed_call(
        device,
        lambda: solve_alpha_woodbury(
            p32,
            k32,
            history_keys=None,
            regularization=lam,
            projector_certificate=ProjectorCertificate(
                projector_sha256,
                1.0,
                1.0,
                "artifact-unverified",
                1.0e-10,
            ),
            residual_tolerance=tolerance,
        ),
    )
    q_w64 = w64.q
    if (
        q_d32.shape != (geometry.projector_shape[0], BATCH_SIZE)
        or q_w32.shape != q_d32.shape
        or q_w64.shape != q_d32.shape
    ):
        raise ODEBFContractError("four-path thin solve geometry differs")

    n_update = native_matched_update.detach().to(device=device, dtype=torch.float32).T.contiguous()
    d_update = q_d32 @ r32.T
    w32_update = q_w32 @ r32.T
    w64_update = q_w64 @ r32.to(dtype=torch.float64).T
    if tuple(n_update.shape) != geometry.output_shape:
        raise ODEBFContractError("four-path Native solve orientation differs")

    factors = {
        "D32": (
            WaypointFactor(weight_name, layer, 0, 0, 0, 1.0, r32.detach(), q_d32.detach()),
        ),
        "W32": (
            WaypointFactor(weight_name, layer, 0, 0, 0, 1.0, r32.detach(), q_w32.detach()),
        ),
        "W64": (
            WaypointFactor(
                weight_name,
                layer,
                0,
                0,
                0,
                1.0,
                r32.detach(),
                q_w64.detach().to(dtype=torch.float32),
            ),
        ),
    }
    native_candidate = (
        entry_weight.detach().to(dtype=torch.float32)
        + native_matched_update.detach().to(device=device, dtype=torch.float32)
    ).to(dtype=torch.bfloat16)
    candidates: dict[str, torch.Tensor] = {"N32": native_candidate.detach().to(device="cpu")}
    for path in ("D32", "W32", "W64"):
        candidate, _ = assemble_effective_bf16(entry_weight, factors[path], row_block=row_block)
        candidates[path] = candidate.detach().to(device="cpu")
        del candidate

    b_wide32 = g32 @ r32.T
    residuals = {
        "N32": _normalized_backward_residual(
            a32,
            n_update,
            b_wide32,
            system_norm_estimate=system_norm,
        ),
        "D32": _normalized_backward_residual(
            a32,
            q_d32,
            g32,
            system_norm_estimate=system_norm,
        ),
        "W32": _normalized_backward_residual(
            a32,
            q_w32,
            g32,
            system_norm_estimate=system_norm,
        ),
    }
    w64_action = lam * q_w64 + g32.to(dtype=torch.float64) @ (
        k32.to(dtype=torch.float64).T @ q_w64
    )
    w64_denominator = system_norm * torch.linalg.norm(q_w64) + torch.linalg.norm(
        g32.to(dtype=torch.float64)
    )
    residuals["W64"] = float(
        torch.linalg.norm(w64_action - g32.to(dtype=torch.float64))
        / torch.clamp(w64_denominator, min=torch.finfo(torch.float64).tiny)
    )

    d_factor_norm = max(float(torch.linalg.norm(q_d32)), torch.finfo(torch.float32).eps)
    d_update_norm = max(float(torch.linalg.norm(d_update)), torch.finfo(torch.float32).eps)
    factor_errors: dict[str, float | None] = {
        "N32": None,
        "D32": 0.0,
        "W32": float(torch.linalg.norm(q_w32 - q_d32)) / d_factor_norm,
        "W64": float(torch.linalg.norm(q_w64 - q_d32.to(dtype=torch.float64))) / d_factor_norm,
    }
    update_errors = {
        "N32": float(torch.linalg.norm(n_update - d_update)) / d_update_norm,
        "D32": 0.0,
        "W32": float(torch.linalg.norm(w32_update - d_update)) / d_update_norm,
        "W64": float(
            torch.linalg.norm(w64_update - d_update.to(dtype=torch.float64))
        ) / d_update_norm,
    }
    timings = {
        "N32": (native_receipt.wall_seconds, native_receipt.gpu_seconds),
        "D32": (d_wall, d_gpu),
        "W32": (w32_wall, w32_gpu),
        "W64": (w64_wall, w64_gpu),
    }
    solve_shapes = {
        "N32": tuple(int(value) for value in n_update.shape),
        "D32": tuple(int(value) for value in q_d32.shape),
        "W32": tuple(int(value) for value in q_w32.shape),
        "W64": tuple(int(value) for value in q_w64.shape),
    }
    solve_dtypes = {
        "N32": str(torch.float32),
        "D32": str(torch.float32),
        "W32": str(torch.float32),
        "W64": str(torch.float64),
    }
    rhs_kinds = {
        "N32": "wide_rhs_G_Rt",
        "D32": "thin_rhs_G",
        "W32": "thin_rhs_G",
        "W64": "thin_rhs_G",
    }
    path_receipts: list[FourPathSolveReceipt] = []
    n32_edit_norm = max(
        float(
            torch.linalg.norm(
                candidates["N32"].to(dtype=torch.float32)
                - entry_weight.detach().to(device="cpu", dtype=torch.float32)
            )
        ),
        torch.finfo(torch.float32).eps,
    )
    for path in FOUR_PATH_ORDER:
        finite = bool(
            math.isfinite(residuals[path])
            and math.isfinite(update_errors[path])
            and (
                factor_errors[path] is None
                or math.isfinite(float(factor_errors[path]))
            )
            and torch.isfinite(candidates[path].to(dtype=torch.float32)).all()
        )
        certificate_passed = finite and residuals[path] <= tolerance
        if path == "N32":
            certificate_passed = certificate_passed and native_receipt.passed
        if path == "W64":
            certificate_passed = certificate_passed and w64.certificate.passed
        path_receipts.append(
            FourPathSolveReceipt(
                int(layer),
                path,
                FOUR_PATH_REFERENCE,
                source_identity,
                source_hashes,
                source_shapes,
                source_dtypes,
                solve_shapes[path],
                tuple(int(value) for value in entry_weight.shape),
                solve_dtypes[path],
                str(torch.float32),
                device.type,
                joint_rank,
                rhs_kinds[path],
                residuals[path],
                system_norm,
                "deterministic-A2-power-iteration-6",
                w64.certificate.small_condition if path == "W64" else condition_estimate,
                "low-rank-small-system-proxy",
                factor_errors[path],
                update_errors[path],
                float(
                    torch.linalg.norm(
                        candidates[path].to(dtype=torch.float32)
                        - candidates["N32"].to(dtype=torch.float32)
                    )
                ) / n32_edit_norm,
                finite,
                certificate_passed,
                timings[path][0],
                timings[path][1],
                int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
            )
        )

    comparison_map: dict[tuple[str, str], BF16ComparisonReceipt] = {}
    comparisons_to_n32_list: list[BF16ComparisonReceipt] = []
    for path in FOUR_PATH_ORDER:
        comparison = bf16_comparison_receipt(
            candidates["N32"],
            candidates[path],
            reference_path="N32",
            candidate_path=path,
        )
        comparison_map[("N32", path)] = comparison
        comparisons_to_n32_list.append(comparison)
    comparisons_to_n32 = tuple(comparisons_to_n32_list)
    pairwise_list: list[BF16ComparisonReceipt] = []
    for left, right in combinations_with_replacement(FOUR_PATH_ORDER, 2):
        comparison = comparison_map.get((left, right))
        if comparison is None:
            if left == right:
                element_count = int(candidates[left].numel())
                comparison = BF16ComparisonReceipt(
                    left,
                    right,
                    True,
                    0,
                    element_count,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    "mismatching-elements",
                    (
                        ("0", element_count),
                        ("1", 0),
                        ("2", 0),
                        ("3-4", 0),
                        ("5-8", 0),
                        ("9-16", 0),
                        ("17+", 0),
                    ),
                    0.0,
                )
            else:
                comparison = bf16_comparison_receipt(
                    candidates[left],
                    candidates[right],
                    reference_path=left,
                    candidate_path=right,
                )
            comparison_map[(left, right)] = comparison
        pairwise_list.append(comparison)
    pairwise = tuple(pairwise_list)
    adjacent_pairs = (("N32", "D32"), ("D32", "W32"), ("W32", "W64"))
    adjacent = tuple(comparison_map[(left, right)] for left, right in adjacent_pairs)
    first_boundary = next(
        (
            f"{receipt.reference_path}_vs_{receipt.candidate_path}"
            for receipt in adjacent
            if not receipt.byte_exact
        ),
        None,
    )
    if (
        tuple(value.data_ptr() for value in guarded) != pointers
        or tuple(value._version for value in guarded) != versions
        or tuple(
            None
            if value.grad is None
            else (value.grad.data_ptr(), value.grad._version, tensor_sha256(value.grad))
            for value in guarded
        ) != gradients
        or tuple(value.requires_grad for value in guarded) != requires_grad
        or not torch.equal(torch.get_rng_state(), cpu_rng)
        or (
            cuda_rng is not None
            and not torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        )
    ):
        raise ODEBFContractError("four-path construction mutated input or RNG state")
    layer_receipt = FourPathLayerReceipt(
        int(layer),
        hashlib.sha256(weight_name.encode("utf-8")).hexdigest(),
        request_order_sha256,
        source_identity,
        tuple(path_receipts),
        comparisons_to_n32,
        pairwise,
        adjacent,
        first_boundary,
    )
    cpu_factors = {
        path: tuple(
            WaypointFactor(
                factor.weight_name,
                factor.layer,
                factor.correction_cycle,
                factor.step_in_cycle,
                factor.factor_ordinal,
                factor.theta,
                factor.left.detach().to(device="cpu", dtype=torch.float32),
                factor.right.detach().to(device="cpu", dtype=torch.float32),
                factor.joint_batch,
            )
            for factor in values
        )
        for path, values in factors.items()
    }
    return FourPathLayerResult(candidates, cpu_factors, layer_receipt, w64.certificate)


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
    path_candidates: dict[str, dict[str, torch.Tensor]]
    path_factors: dict[str, dict[str, tuple[WaypointFactor, ...]]]
    four_path_layer_receipts: tuple[FourPathLayerReceipt, ...]
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
    path_candidates: dict[str, dict[str, torch.Tensor]] = {
        path: {} for path in FOUR_PATH_ORDER
    }
    path_factors: dict[str, dict[str, tuple[WaypointFactor, ...]]] = {
        path: {} for path in FOUR_PATH_ORDER if path != "N32"
    }
    key_hashes: list[tuple[int, str]] = []
    certificates: list[tuple[int, WoodburyCertificate]] = []
    factor_receipts: list[LayerFactorReceipt] = []
    four_path_receipts: list[FourPathLayerReceipt] = []
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
            delta = alpha_main.upd_matrix_match_shape(
                native_deltas[weight_name],
                parameter.shape,
            )
            layer_result = build_four_path_layer_diagnostic(
                layer=layer,
                weight_name=weight_name,
                entry_weight=parameter,
                native_matched_update=delta,
                projector=p_layer,
                joint_keys=layer_keys,
                residual=residual,
                regularization=float(hparams.L2),
                projector_sha256=projector_sha256,
                request_order_sha256=request_order_sha256,
                residual_tolerance=model_residual_tolerance,
                native_receipt=dense_solve_receipts[layer_index],
                row_block=64,
            )
            del native_deltas[weight_name]
            numerical_rank = int(torch.linalg.matrix_rank(layer_keys.float()))
            factor_receipts.append(
                LayerFactorReceipt(
                    layer,
                    tuple(int(value) for value in layer_keys.shape),
                    tuple(int(value) for value in residual.shape),
                    numerical_rank,
                    str(torch.float32),
                    parameter.device.type,
                )
            )
            key_hashes.append((layer, tensor_sha256(layer_keys)))
            certificates.append((layer, layer_result.w64_certificate))
            four_path_receipts.append(layer_result.receipt)
            for path in FOUR_PATH_ORDER:
                path_candidates[path][weight_name] = layer_result.candidates[path]
                if path != "N32":
                    path_factors[path][weight_name] = layer_result.factors[path]
            del layer_result, delta
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
        path_candidates["N32"],
        path_candidates["W32"],
        path_factors["W32"],
        path_candidates,
        path_factors,
        tuple(four_path_receipts),
        entry_weights,
        entry_sha256,
        tuple(tensor_sha256(value) for value in direct_z),
        tuple(key_hashes),
        dense_solve_receipts,
        tuple(certificates),
        initialization,
        target_backward_count,
    )
