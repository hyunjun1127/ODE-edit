"""Exact low-rank Alpha solves for symmetric and general projectors.

For ``A = lambda I + P X X^T`` and ``G = P K`` the general contraction is

``Q = lambda^-1 [G - Z (lambda I + X^T Z)^-1 X^T G]``, ``Z=P X``.

Only a fully verified symmetric-idempotent projector may use the specialized
``Z^T Z`` / ``Z^T G`` Cholesky path.  A sampled or unknown certificate always
uses the nonsymmetric LU/QR path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .contracts import BATCH_SIZE, ODEBFContractError, finite, positive


class WoodburyMethod(str, Enum):
    SYMMETRIC_CHOLESKY = "symmetric-idempotent-cholesky"
    GENERAL_LU = "general-projector-lu"
    GENERAL_QR = "general-projector-qr"


@dataclass(frozen=True, slots=True)
class ProjectorCertificate:
    source_sha256: str
    symmetry_residual: float
    idempotence_residual: float
    verification_kind: str
    tolerance: float

    def __post_init__(self) -> None:
        if len(self.source_sha256) != 64:
            raise ODEBFContractError("projector source identity is not SHA-256")
        for name in ("symmetry_residual", "idempotence_residual"):
            value = finite(name, getattr(self, name))
            if value < 0.0:
                raise ODEBFContractError(f"{name} is negative")
        if self.verification_kind not in ("full-matrix", "action-probe", "artifact-unverified"):
            raise ODEBFContractError("projector verification kind is invalid")
        positive("projector tolerance", self.tolerance)

    @property
    def exact_symmetric_idempotent(self) -> bool:
        return (
            self.verification_kind == "full-matrix"
            and self.symmetry_residual <= self.tolerance
            and self.idempotence_residual <= self.tolerance
        )


@dataclass(frozen=True, slots=True)
class WoodburyCertificate:
    method: WoodburyMethod
    projector: ProjectorCertificate
    small_dimension: int
    small_condition: float
    small_residual: float
    alpha_linear_residual: float
    finite: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class WoodburyResult:
    q: torch.Tensor
    certificate: WoodburyCertificate


def full_projector_certificate(
    projector: torch.Tensor,
    *,
    source_sha256: str,
    tolerance: float = 1.0e-10,
    maximum_elements: int = 1_000_000,
) -> ProjectorCertificate:
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ODEBFContractError("projector must be square")
    if projector.numel() > maximum_elements:
        raise ODEBFContractError("full projector verification exceeds its memory lock")
    matrix = projector.detach().to(device="cpu", dtype=torch.float64)
    denominator = max(float(torch.linalg.norm(matrix)), torch.finfo(torch.float64).eps)
    symmetry = float(torch.linalg.norm(matrix - matrix.T)) / denominator
    idempotence = float(torch.linalg.norm(matrix @ matrix - matrix)) / denominator
    return ProjectorCertificate(
        source_sha256,
        symmetry,
        idempotence,
        "full-matrix",
        tolerance,
    )


def _validate_inputs(
    projector: torch.Tensor,
    current_keys: torch.Tensor,
    history_keys: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ODEBFContractError("Alpha projector must be square")
    dimension = projector.shape[0]
    if current_keys.ndim != 2 or current_keys.shape != (dimension, BATCH_SIZE):
        raise ODEBFContractError("current Alpha keys must have shape [d,10]")
    if history_keys is None:
        history_keys = torch.empty((dimension, 0), dtype=current_keys.dtype, device=current_keys.device)
    if history_keys.ndim != 2 or history_keys.shape[0] != dimension:
        raise ODEBFContractError("historical Alpha keys have an incompatible shape")
    if projector.device != current_keys.device or history_keys.device != current_keys.device:
        raise ODEBFContractError("Alpha projector/current/history devices differ")
    for name, tensor in (
        ("projector", projector),
        ("current_keys", current_keys),
        ("history_keys", history_keys),
    ):
        if tensor.device.type not in ("cpu", "cuda"):
            raise ODEBFContractError(f"Woodbury backend received unsupported {name} device")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ODEBFContractError(f"{name} must use FP32 or FP64")
        if not torch.isfinite(tensor).all():
            raise ODEBFContractError(f"{name} contains non-finite values")
    return (
        projector.detach(),
        current_keys.detach().to(dtype=torch.float64),
        history_keys.detach().to(dtype=torch.float64),
    )


def _apply_projector(projector: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Keep the potentially huge read-only projector in its pinned dtype and cast
    # only the thin result to FP64.  No dense FP64 projector copy is retained.
    result = projector @ values.to(dtype=projector.dtype)
    return result.to(dtype=torch.float64)


def solve_alpha_woodbury(
    projector: torch.Tensor,
    current_keys: torch.Tensor,
    *,
    history_keys: torch.Tensor | None,
    regularization: float,
    projector_certificate: ProjectorCertificate,
    maximum_condition: float = 1.0e12,
    residual_tolerance: float = 1.0e-8,
) -> WoodburyResult:
    matrix, current, history = _validate_inputs(projector, current_keys, history_keys)
    lam = positive("Alpha regularization", regularization)
    max_condition = positive("maximum condition", maximum_condition)
    residual_tolerance = positive("Woodbury residual tolerance", residual_tolerance)
    combined = torch.cat((current, history), dim=1)
    z = _apply_projector(matrix, combined)
    g = _apply_projector(matrix, current)
    identity = torch.eye(
        combined.shape[1],
        dtype=torch.float64,
        device=combined.device,
    )

    if projector_certificate.exact_symmetric_idempotent:
        small = lam * identity + z.T @ z
        rhs = z.T @ g
        chol, info = torch.linalg.cholesky_ex(small)
        if int(info.max()) != 0:
            raise ODEBFContractError("symmetric Woodbury Cholesky failed")
        small_solution = torch.cholesky_solve(rhs, chol)
        method = WoodburyMethod.SYMMETRIC_CHOLESKY
    else:
        # Both contractions must use raw X on the left for a general P.
        small = lam * identity + combined.T @ z
        rhs = combined.T @ g
        condition = float(torch.linalg.cond(small))
        use_qr = not torch.isfinite(torch.tensor(condition)) or condition > max_condition
        if not use_qr:
            small_solution, info = torch.linalg.solve_ex(small, rhs)
            use_qr = int(info.max()) != 0
        if use_qr:
            if small.device.type == "cuda":
                least_squares = torch.linalg.lstsq(
                    small.to(device="cpu"),
                    rhs.to(device="cpu"),
                    driver="gelsy",
                )
                small_solution = least_squares.solution.to(device=small.device)
            else:
                least_squares = torch.linalg.lstsq(small, rhs, driver="gelsy")
                small_solution = least_squares.solution
            method = WoodburyMethod.GENERAL_QR
        else:
            method = WoodburyMethod.GENERAL_LU

    q = (g - z @ small_solution) / lam
    small_denominator = max(float(torch.linalg.norm(rhs)), torch.finfo(torch.float64).eps)
    small_residual = float(torch.linalg.norm(small @ small_solution - rhs)) / small_denominator
    alpha_action = lam * q + z @ (combined.T @ q)
    alpha_denominator = max(float(torch.linalg.norm(g)), torch.finfo(torch.float64).eps)
    alpha_residual = float(torch.linalg.norm(alpha_action - g)) / alpha_denominator
    condition = float(torch.linalg.cond(small))
    all_finite = bool(
        torch.isfinite(q).all()
        and torch.isfinite(small_solution).all()
        and torch.isfinite(torch.tensor((condition, small_residual, alpha_residual))).all()
    )
    passed = all_finite and small_residual <= residual_tolerance and alpha_residual <= residual_tolerance
    certificate = WoodburyCertificate(
        method,
        projector_certificate,
        int(small.shape[0]),
        condition,
        small_residual,
        alpha_residual,
        all_finite,
        passed,
    )
    if not passed:
        raise ODEBFContractError("Woodbury feasibility certificate failed")
    return WoodburyResult(q, certificate)
