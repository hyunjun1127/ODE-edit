"""Typed science and runtime contracts for Ordered Response-Barrier ODE-Edit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch


class ORBODEContractError(ValueError):
    """A preregistered input or configuration differs from the contract."""


class TechnicalBoundary(RuntimeError):
    """The implementation/runtime failed before a scientific endpoint."""


class StaleStateBoundary(TechnicalBoundary):
    """A residual/key/writer/JVP object was consumed at the wrong state."""


class NumericalMethodBoundary(RuntimeError):
    """The locked numerical method produced a non-finite/contradictory state."""


class ScientificBoundary(RuntimeError):
    """A typed, finite scientific boundary rather than a technical failure."""


class ArmId(str, Enum):
    OFFICIAL = "O"
    QUOTA_CLOSED_LOOP = "QCL"
    NO_QUOTA_FIXED = "NQFIX"
    ORDERED_RESPONSE_FIXED_HORIZON = "ORBFH"
    SWEEP_FROZEN_JACOBI = "JAC"
    ORDERED_RESPONSE_FIRST_HIT = "ORBHit"


class ResidualPolicy(str, Enum):
    OFFICIAL_REMAINING = "OFFICIAL_REMAINING_R_OVER_N"
    CURRENT_REMAINING = "CURRENT_R_OVER_N"
    CURRENT_FULL = "CURRENT_FULL_R"


class ResponsePolicy(str, Enum):
    OFFICIAL = "OFFICIAL"
    UNIT = "UNIT"
    CURRENT_RESPONSE = "CURRENT_RESPONSE"


class RefreshPolicy(str, Enum):
    OFFICIAL_ONE_PASS = "OFFICIAL_ONE_PASS"
    PER_VISIT = "PER_VISIT"
    PER_SWEEP = "PER_SWEEP"
    DERIVED_PREFIX = "DERIVED_PREFIX"


@dataclass(frozen=True, slots=True)
class ArmConfig:
    arm: ArmId
    residual_policy: ResidualPolicy
    response_policy: ResponsePolicy
    refresh_policy: RefreshPolicy
    sweeps: int
    step_size: float
    horizon: float
    first_hit_controls_dynamics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ArmId):
            raise ORBODEContractError("arm must be typed")
        if not all(
            isinstance(value, enum_type)
            for value, enum_type in (
                (self.residual_policy, ResidualPolicy),
                (self.response_policy, ResponsePolicy),
                (self.refresh_policy, RefreshPolicy),
            )
        ):
            raise ORBODEContractError("arm policy must be typed")
        if (
            isinstance(self.sweeps, bool)
            or not isinstance(self.sweeps, int)
            or self.sweeps < 0
            or not math.isfinite(self.step_size)
            or not math.isfinite(self.horizon)
            or self.step_size < 0.0
            or self.horizon < 0.0
        ):
            raise ORBODEContractError("arm clock is invalid")
        if self.arm is ArmId.OFFICIAL:
            if (
                self.sweeps != 1
                or self.step_size != 1.0
                or self.horizon != 1.0
                or self.refresh_policy is not RefreshPolicy.OFFICIAL_ONE_PASS
                or self.response_policy is not ResponsePolicy.OFFICIAL
            ):
                raise ORBODEContractError("Official must remain the stock one-pass bypass")
        elif self.arm is ArmId.ORDERED_RESPONSE_FIRST_HIT:
            if self.refresh_policy is not RefreshPolicy.DERIVED_PREFIX:
                raise ORBODEContractError("ORBHit is only an ORBFH-derived prefix")
        elif (
            self.sweeps <= 0
            or self.sweeps * self.step_size != self.horizon
            or self.horizon != 1.0
            or self.first_hit_controls_dynamics
        ):
            raise ORBODEContractError("scientific arms require a fixed-T first-hit-OFF clock")

    @property
    def uses_response(self) -> bool:
        return self.response_policy is ResponsePolicy.CURRENT_RESPONSE


def canonical_arm_configs(*, sweeps: int = 4) -> tuple[ArmConfig, ...]:
    """Return the immutable O→QCL→NQFIX→ORBFH→JAC→ORBHit order."""

    if sweeps not in (2, 4, 8):
        raise ORBODEContractError("resolution is locked to N in {2,4,8}")
    h = 1.0 / sweeps
    return (
        ArmConfig(
            ArmId.OFFICIAL,
            ResidualPolicy.OFFICIAL_REMAINING,
            ResponsePolicy.OFFICIAL,
            RefreshPolicy.OFFICIAL_ONE_PASS,
            1,
            1.0,
            1.0,
        ),
        ArmConfig(
            ArmId.QUOTA_CLOSED_LOOP,
            ResidualPolicy.CURRENT_REMAINING,
            ResponsePolicy.UNIT,
            RefreshPolicy.PER_VISIT,
            sweeps,
            h,
            1.0,
        ),
        ArmConfig(
            ArmId.NO_QUOTA_FIXED,
            ResidualPolicy.CURRENT_FULL,
            ResponsePolicy.UNIT,
            RefreshPolicy.PER_VISIT,
            sweeps,
            h,
            1.0,
        ),
        ArmConfig(
            ArmId.ORDERED_RESPONSE_FIXED_HORIZON,
            ResidualPolicy.CURRENT_FULL,
            ResponsePolicy.CURRENT_RESPONSE,
            RefreshPolicy.PER_VISIT,
            sweeps,
            h,
            1.0,
        ),
        ArmConfig(
            ArmId.SWEEP_FROZEN_JACOBI,
            ResidualPolicy.CURRENT_FULL,
            ResponsePolicy.CURRENT_RESPONSE,
            RefreshPolicy.PER_SWEEP,
            sweeps,
            h,
            1.0,
        ),
        ArmConfig(
            ArmId.ORDERED_RESPONSE_FIRST_HIT,
            ResidualPolicy.CURRENT_FULL,
            ResponsePolicy.CURRENT_RESPONSE,
            RefreshPolicy.DERIVED_PREFIX,
            sweeps,
            h,
            1.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class ORBODERuntimeLock:
    layers: tuple[int, ...] = (4, 5, 6, 7, 8)
    sweeps: int = 4
    step_size: float = 0.25
    horizon: float = 1.0
    requested_dtype: str = "torch.float32"
    autocast_enabled: bool = False
    tf32_enabled: bool = False
    retry_count: int = 0
    dynamic_z_recompute_count: int = 0
    controller_heldout_influence_count: int = 0

    def __post_init__(self) -> None:
        if (
            self.layers != (4, 5, 6, 7, 8)
            or self.sweeps != 4
            or self.step_size != 0.25
            or self.horizon != 1.0
            or self.requested_dtype != "torch.float32"
            or self.autocast_enabled
            or self.tf32_enabled
            or self.retry_count != 0
            or self.dynamic_z_recompute_count != 0
            or self.controller_heldout_influence_count != 0
        ):
            raise ORBODEContractError("runtime numerical/science lock differs")


@dataclass(frozen=True, slots=True)
class FrozenAnchor:
    target: torch.Tensor
    entry_terminal: torch.Tensor
    scales: torch.Tensor
    active: torch.Tensor
    target_sha256: str
    request_order_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target, torch.Tensor)
            or not isinstance(self.entry_terminal, torch.Tensor)
            or self.target.shape != self.entry_terminal.shape
            or self.target.ndim != 2
            or self.target.dtype is not torch.float32
            or self.entry_terminal.dtype is not torch.float32
            or self.target.requires_grad
            or self.entry_terminal.requires_grad
            or not bool(torch.isfinite(self.target).all())
            or not bool(torch.isfinite(self.entry_terminal).all())
        ):
            raise ORBODEContractError("frozen target/entry terminal must be finite FP32 [D,B]")
        request_count = self.target.shape[1]
        if (
            self.scales.shape != (request_count,)
            or self.scales.dtype is not torch.float32
            or self.active.shape != (request_count,)
            or self.active.dtype is not torch.bool
            or not bool(torch.isfinite(self.scales).all())
            or bool((self.scales < 0).any())
            or len(self.target_sha256) != 64
            or len(self.request_order_sha256) != 64
        ):
            raise ORBODEContractError("frozen anchor request identity differs")

    @property
    def request_count(self) -> int:
        return int(self.target.shape[1])


@dataclass(frozen=True, slots=True)
class ResponseStatistics:
    g: float
    r: float
    u: float
    per_request_g: tuple[float | None, ...]
    per_request_r: tuple[float | None, ...]
    active_count: int
    nonpositive_request_fraction_when_batch_positive: float | None


def build_frozen_anchor(
    *,
    target: torch.Tensor,
    entry_terminal: torch.Tensor,
    target_sha256: str,
    request_order_sha256: str,
) -> FrozenAnchor:
    target = target.detach().to(device="cpu", dtype=torch.float32).contiguous()
    current = entry_terminal.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if target.shape != current.shape or target.ndim != 2:
        raise ORBODEContractError("target/entry terminal shape differs")
    scales = torch.linalg.vector_norm(target - current, dim=0).float().contiguous()
    active = scales > 0.0
    return FrozenAnchor(
        target=target,
        entry_terminal=current,
        scales=scales,
        active=active,
        target_sha256=target_sha256,
        request_order_sha256=request_order_sha256,
    )


def normalized_potential(anchor: FrozenAnchor, terminal: torch.Tensor) -> tuple[float, tuple[float | None, ...]]:
    """Evaluate Vz and request residual excursions without an epsilon floor."""

    current = terminal.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if current.shape != anchor.target.shape or not bool(torch.isfinite(current).all()):
        raise NumericalMethodBoundary("terminal observation is non-finite or shape-mismatched")
    residual = anchor.target - current
    if not bool(anchor.active.any()):
        return 0.0, tuple(None for _ in range(anchor.request_count))
    active = anchor.active
    normalized = residual[:, active].double() / anchor.scales[active].double().unsqueeze(0)
    per_request = torch.sum(normalized.square(), dim=0)
    value = 0.5 * float(per_request.mean().item())
    q = torch.linalg.vector_norm(residual.double(), dim=0)
    excursions: list[float | None] = []
    for index in range(anchor.request_count):
        excursions.append(
            None if not bool(active[index]) else float(q[index].item() / float(anchor.scales[index]))
        )
    if not math.isfinite(value) or any(item is not None and not math.isfinite(item) for item in excursions):
        raise NumericalMethodBoundary("normalized realization potential is non-finite")
    return value, tuple(excursions)


def response_statistics(
    *,
    anchor: FrozenAnchor,
    terminal: torch.Tensor,
    response: torch.Tensor,
) -> ResponseStatistics:
    """Compute the locked FP64 reductions and h-independent response velocity."""

    current = terminal.detach().to(device="cpu", dtype=torch.float32).contiguous()
    psi = response.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        current.shape != anchor.target.shape
        or psi.shape != anchor.target.shape
        or not bool(torch.isfinite(current).all())
        or not bool(torch.isfinite(psi).all())
    ):
        raise NumericalMethodBoundary("response observation is non-finite or shape-mismatched")
    if not bool(anchor.active.any()):
        # A zero anchor is a finite scientific observation.  It carries no
        # response command and must reach the ordinary no-op endpoint rather
        # than being promoted to an implementation failure.
        return ResponseStatistics(
            g=0.0,
            r=0.0,
            u=0.0,
            per_request_g=tuple(None for _ in range(anchor.request_count)),
            per_request_r=tuple(None for _ in range(anchor.request_count)),
            active_count=0,
            nonpositive_request_fraction_when_batch_positive=None,
        )
    active = anchor.active
    scales = anchor.scales[active].double().unsqueeze(0)
    residual = (anchor.target - current)[:, active].double() / scales
    normalized_response = psi[:, active].double() / scales
    per_g = torch.sum(residual * normalized_response, dim=0)
    per_r = torch.sum(normalized_response.square(), dim=0)
    g = float(per_g.mean().item())
    r = float(per_r.mean().item())
    if not math.isfinite(g) or not math.isfinite(r) or r < 0.0:
        raise NumericalMethodBoundary("response reduction is invalid")
    if r == 0.0 or g <= 0.0:
        u = 0.0
    else:
        u = min(1.0, g / r)
    fraction = None
    if g > 0.0:
        fraction = float((per_g <= 0.0).double().mean().item())
    per_request_g: list[float | None] = [None] * anchor.request_count
    per_request_r: list[float | None] = [None] * anchor.request_count
    active_ordinals = torch.nonzero(active, as_tuple=False).flatten().tolist()
    for active_ordinal, request_ordinal in enumerate(active_ordinals):
        per_request_g[int(request_ordinal)] = float(per_g[active_ordinal].item())
        per_request_r[int(request_ordinal)] = float(per_r[active_ordinal].item())
    return ResponseStatistics(
        g=g,
        r=r,
        u=u,
        per_request_g=tuple(per_request_g),
        per_request_r=tuple(per_request_r),
        active_count=int(active.sum().item()),
        nonpositive_request_fraction_when_batch_positive=fraction,
    )


def assert_full_fp32(model: torch.nn.Module) -> Mapping[str, Any]:
    counts: dict[str, int] = {}
    quantized = bool(getattr(model, "is_quantized", False))
    for parameter in model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + int(parameter.numel())
    autocast_cuda = bool(torch.is_autocast_enabled())
    try:
        autocast_cpu = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # PyTorch < 2.4
        autocast_cpu = bool(getattr(torch, "is_autocast_cpu_enabled", lambda: False)())
    tf32_matmul = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
    tf32_cudnn = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    if (
        quantized
        or set(counts) != {"torch.float32"}
        or autocast_cuda
        or autocast_cpu
        or tf32_matmul
        or tf32_cudnn
    ):
        raise TechnicalBoundary(
            "FULL_FP32 model gate failed: "
            f"quantized={quantized}, counts={counts}, autocast_cuda={autocast_cuda}, "
            f"autocast_cpu={autocast_cpu}, tf32_matmul={tf32_matmul}, tf32_cudnn={tf32_cudnn}"
        )
    return {
        "requested_dtype": "torch.float32",
        "loaded_model_dtype": "torch.float32",
        "parameter_dtype_counts": counts,
        "quantized": False,
        "autocast_enabled": autocast_cuda or autocast_cpu,
        "tf32_enabled": tf32_matmul or tf32_cudnn,
        "bf16_fp16_cast_count": 0,
    }
