"""Low-rank isolated and historical AlphaEdit algebra for ODE-Edit hooks.

The isolated path implements ``cache_c = 0`` for atomic diagnostics.  The
historical path represents canonical ``cache_c = H H.T`` through key columns
``H`` and never materializes the dense cache.  Both are deliberately separate from
``projector_adapter.project_proposal``: post-hoc ``B @ P`` changes the right
factor of an already computed proposal, whereas genuine AlphaEdit puts ``P``
*inside* its normal equation.  The two paths must remain distinguishable in
run metadata and analysis.

For keys ``K`` with shape ``[d_in, n]``, residuals ``R`` with shape
``[d_out, n]``, and a pinned projector ``P``, upstream AlphaEdit's first-edit
equation is

``A X = (P K) R.T,  A = lambda I + (P K) K.T``.

Writing ``U = P K`` and ``M = lambda I + K.T U`` gives the exact Woodbury
identity ``X = U M^-1 R.T``.  We retain ``U M^-1`` and ``R`` as factors rather
than materializing the ``d_in x d_out`` update.  For Llama/Qwen down-projector
weights the stored model update is ``X.T = R M^-T U.T``.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from .contracts import (
    ContractError,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    SnapshotManifest,
    orient_easyedit_factor,
)


class AlphaEditFactorError(ContractError):
    """Raised when isolated AlphaEdit factor algebra cannot be verified."""


GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX = (
    "alphaedit-genuine-isolated-first-edit-woodbury-v1"
)
"""Solver tag reserved for the genuine ``P``-inside-solve construction."""

GENUINE_ISOLATED_ALPHAEDIT_DENSE_SOLVER_PREFIX = (
    "alphaedit-genuine-isolated-first-edit-upstream-dense-v1"
)
"""Solver tag for the numerically upstream-identical dense ``P``-inside solve."""

UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX = (
    "alphaedit-unprojected-isolated-first-edit-woodbury-v1"
)
"""Solver tag for the same isolated Alpha algebra with ``P = I``."""

GENUINE_HISTORICAL_ALPHAEDIT_SOLVER_PREFIX = (
    "alphaedit-genuine-historical-lowrank-woodbury-v1"
)
"""Solver tag for canonical ``P @ (K K.T + cache_c)`` with low-rank history."""

POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN = "/alphaedit-projector/"
"""Existing :mod:`projector_adapter` tag for a post-hoc ``B @ P`` proposal."""


class AlphaEditProposalKind(str, Enum):
    """Provenance category; genuine and post-hoc paths are not interchangeable."""

    GENUINE_ISOLATED_FIRST_EDIT = "genuine-isolated-first-edit"
    GENUINE_HISTORICAL = "genuine-historical"
    UNPROJECTED_ISOLATED_FIRST_EDIT = "unprojected-isolated-first-edit"
    POSTHOC_RIGHT_PROJECTED = "posthoc-right-projected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AlphaEditRightLeak:
    """Dense-free ``||Delta W (I - P)||_F / ||Delta W||_F`` diagnostics."""

    update_frobenius: float
    right_leak_frobenius: float
    right_leak_ratio: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise AlphaEditFactorError(f"{name} must be finite")
            if value < 0.0:
                raise AlphaEditFactorError(f"{name} must be non-negative")
        if self.update_frobenius <= 0.0:
            raise AlphaEditFactorError("update_frobenius must be positive")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class LowRankProposalComparison:
    """Dense-free aggregate similarity between aligned factor proposals."""

    weight_count: int
    reference_frobenius: float
    candidate_frobenius: float
    difference_frobenius: float
    cosine: float
    relative_frobenius_to_reference: float

    def __post_init__(self) -> None:
        if isinstance(self.weight_count, bool) or self.weight_count <= 0:
            raise AlphaEditFactorError("weight_count must be positive")
        for name in self.__dataclass_fields__:
            if name == "weight_count":
                continue
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise AlphaEditFactorError(f"{name} must be finite")
        if self.reference_frobenius <= 0.0 or self.candidate_frobenius <= 0.0:
            raise AlphaEditFactorError("proposal Frobenius norms must be positive")
        if self.difference_frobenius < 0.0:
            raise AlphaEditFactorError("difference_frobenius must be non-negative")
        if self.relative_frobenius_to_reference < 0.0:
            raise AlphaEditFactorError(
                "relative_frobenius_to_reference must be non-negative"
            )
        if not -1.0 <= self.cosine <= 1.0:
            raise AlphaEditFactorError("cosine must lie in [-1, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _validate_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise AlphaEditFactorError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in normalized):
        raise AlphaEditFactorError(f"{name} must be a full SHA-256 digest")
    return normalized


def _validate_weight_shape(weight_shape: Sequence[int]) -> tuple[int, int]:
    try:
        shape = tuple(weight_shape)
    except TypeError as exc:
        raise AlphaEditFactorError("weight_shape must be a two-dimensional sequence") from exc
    if len(shape) != 2:
        raise AlphaEditFactorError("AlphaEdit target weight must be a matrix")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
        raise AlphaEditFactorError("weight_shape dimensions must be positive integers")
    return (int(shape[0]), int(shape[1]))


def _validate_l2(l2: float) -> float:
    if isinstance(l2, bool) or not isinstance(l2, (int, float)):
        raise AlphaEditFactorError("l2 must be a finite positive scalar")
    value = float(l2)
    if not math.isfinite(value) or value <= 0.0:
        raise AlphaEditFactorError("l2 must be a finite positive scalar")
    return value


def _require_finite_matrix(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise AlphaEditFactorError(f"{name} must be a tensor")
    if value.ndim != 2:
        raise AlphaEditFactorError(f"{name} must be a matrix")
    if not value.is_floating_point():
        raise AlphaEditFactorError(f"{name} must be floating point")
    if value.requires_grad:
        raise AlphaEditFactorError(f"{name} must not require gradients")
    if not bool(torch.isfinite(value).all()):
        raise AlphaEditFactorError(f"{name} must be finite")


def _working_dtype(*values: torch.Tensor) -> torch.dtype:
    # Float64 is retained for exact CPU reference tests.  Production P is
    # pinned float32, so this never promotes a multi-GiB projector merely
    # because an activation happened to be fp16/bf16.
    if any(value.dtype == torch.float64 for value in values):
        return torch.float64
    return torch.float32


def _factor_norm_sq(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``||left @ right.T||_F^2`` using rank-sized Gram matrices."""

    return torch.sum((left.transpose(0, 1) @ left) * (right.transpose(0, 1) @ right))


def _factor_inner(
    left_a: torch.Tensor,
    right_a: torch.Tensor,
    left_b: torch.Tensor,
    right_b: torch.Tensor,
) -> torch.Tensor:
    """Return a Frobenius inner product without materializing either update."""

    return torch.sum(
        (left_a.transpose(0, 1) @ left_b)
        * (right_a.transpose(0, 1) @ right_b)
    )


def _scalar(value: torch.Tensor, *, name: str) -> float:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise AlphaEditFactorError(f"{name} is non-finite")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise AlphaEditFactorError(f"{name} is non-finite")
    return result


def _sqrt_nonnegative(value: float, *, name: str, scale: float = 1.0) -> float:
    # Gram reductions can be a few ulps below zero after cancellation.  A
    # material negative value is a contract failure, not an opportunity to
    # silently hide an invalid update.
    tolerance = 1e-10 * max(1.0, scale)
    if value < -tolerance:
        raise AlphaEditFactorError(f"{name} is materially negative")
    return math.sqrt(max(value, 0.0))


def alphaedit_proposal_kind(proposal: MemitFactorProposal) -> AlphaEditProposalKind:
    """Classify only the explicit solver tags used by this diagnostic."""

    if not isinstance(proposal, MemitFactorProposal):
        raise AlphaEditFactorError("proposal must be a MemitFactorProposal")
    # A post-hoc name intentionally retains its Alpha-base prefix for lineage,
    # so this discriminator must take the explicit transformation tag first.
    if POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN in proposal.solver_name:
        return AlphaEditProposalKind.POSTHOC_RIGHT_PROJECTED
    if proposal.solver_name.startswith(GENUINE_HISTORICAL_ALPHAEDIT_SOLVER_PREFIX):
        return AlphaEditProposalKind.GENUINE_HISTORICAL
    if proposal.solver_name.startswith(
        (GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX, GENUINE_ISOLATED_ALPHAEDIT_DENSE_SOLVER_PREFIX)
    ):
        return AlphaEditProposalKind.GENUINE_ISOLATED_FIRST_EDIT
    if proposal.solver_name.startswith(UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX):
        return AlphaEditProposalKind.UNPROJECTED_ISOLATED_FIRST_EDIT
    return AlphaEditProposalKind.UNKNOWN


def _solve_isolated_alphaedit_factor(
    *,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor | None,
    l2: float,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Factor the genuine isolated AlphaEdit first-edit update.

    ``keys`` is ``K`` with shape ``[d_in, n]`` and ``residuals`` is the
    already distributed ``R`` with shape ``[d_out, n]``.  ``projector`` is
    the pinned AlphaEdit matrix for the exact layer.  No covariance cache is
    accepted here: this function is intentionally valid only for ``cache_c=0``.

    The returned factor is in *model weight orientation* and therefore safely
    handles both native and transposed PyTorch weight layouts through
    :func:`orient_easyedit_factor`.
    """

    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    expected_weight_sha256 = _validate_sha256(
        expected_weight_sha256,
        name="expected_weight_sha256",
    )
    shape = _validate_weight_shape(weight_shape)
    lambda_value = _validate_l2(l2)
    for name, value in (("keys", keys), ("residuals", residuals)):
        _require_finite_matrix(value, name=name)
    if keys.device != residuals.device:
        raise AlphaEditFactorError(
            "keys and residuals must share one device"
        )
    if keys.shape[1] == 0:
        raise AlphaEditFactorError("keys must contain at least one edit column")
    if residuals.shape[1] != keys.shape[1]:
        raise AlphaEditFactorError("keys and residuals must share their edit rank")
    if projector is not None:
        _require_finite_matrix(projector, name="projector")
        if projector.device != keys.device:
            raise AlphaEditFactorError(
                "keys, residuals, and projector must share one device"
            )
        if projector.shape[0] != projector.shape[1]:
            raise AlphaEditFactorError("projector must be square")
        if projector.shape[0] != keys.shape[0]:
            raise AlphaEditFactorError("projector dimension must equal key width")

    work_dtype = _working_dtype(
        keys,
        residuals,
        *((projector,) if projector is not None else ()),
    )
    with torch.no_grad():
        k = keys.detach().to(dtype=work_dtype)
        resid = residuals.detach().to(dtype=work_dtype)
        projected_keys = (
            k
            if projector is None
            else projector.detach().to(dtype=work_dtype) @ k
        )
        if not bool(torch.isfinite(projected_keys).all()):
            raise AlphaEditFactorError("P @ K produced non-finite values")
        small_system = k.transpose(0, 1) @ projected_keys
        small_system.diagonal().add_(lambda_value)
        if not bool(torch.isfinite(small_system).all()):
            raise AlphaEditFactorError("AlphaEdit Woodbury system is non-finite")
        try:
            # (M.T)^-1 U.T, transposed, is U M^-1.  This is exactly the
            # adjusted-key factor in the upstream dense linear solve.
            adjusted_keys = torch.linalg.solve(
                small_system.transpose(0, 1),
                projected_keys.transpose(0, 1),
            ).transpose(0, 1)
        except RuntimeError as exc:
            raise AlphaEditFactorError("AlphaEdit Woodbury system is singular") from exc
        if not bool(torch.isfinite(adjusted_keys).all()):
            raise AlphaEditFactorError("AlphaEdit adjusted keys are non-finite")

    # ``adjusted_keys @ resid.T`` is the native AlphaEdit solve orientation.
    # This helper intentionally delegates shape/transposition choice to the
    # same canonical contract used for MEMIT factors.
    return orient_easyedit_factor(
        adjusted_keys,
        resid,
        weight_name=weight_name,
        weight_shape=shape,
        expected_weight_sha256=expected_weight_sha256,
    )


def solve_isolated_alphaedit_factor(
    *,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    l2: float,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Factor a genuine isolated AlphaEdit solve with ``P`` inside ``A``.

    This is not a ``B @ P`` transform.  Use
    :func:`solve_unprojected_isolated_alphaedit_factor` plus
    :func:`posthoc_alphaedit_right_project_factor` for that separate
    same-Alpha-base ablation.
    """

    return _solve_isolated_alphaedit_factor(
        keys=keys,
        residuals=residuals,
        projector=projector,
        l2=l2,
        weight_name=weight_name,
        weight_shape=weight_shape,
        expected_weight_sha256=expected_weight_sha256,
    )


def solve_isolated_alphaedit_factor_upstream_dense(
    *,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    l2: float,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Factor the exact dense RHS equation evaluated by upstream AlphaEdit.

    This is algebraically identical to :func:`solve_isolated_alphaedit_factor`,
    but intentionally evaluates
    ``solve(P @ (K @ K.T) + lambda I, (P @ K) @ R.T)`` in the same order as
    the pinned Official implementation.  The resulting dense update is then
    projected onto the deterministic numerical column space of ``R`` so the
    actuator remains low rank.  This preserves the scientific equation while
    avoiding an FP32 operation-order discrepancy on ill-conditioned large
    projectors.
    """

    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    expected_weight_sha256 = _validate_sha256(
        expected_weight_sha256,
        name="expected_weight_sha256",
    )
    shape = _validate_weight_shape(weight_shape)
    lambda_value = _validate_l2(l2)
    for name, value in (
        ("keys", keys),
        ("residuals", residuals),
        ("projector", projector),
    ):
        _require_finite_matrix(value, name=name)
    if len({keys.device, residuals.device, projector.device}) != 1:
        raise AlphaEditFactorError(
            "keys, residuals, and projector must share one device"
        )
    if keys.shape[1] == 0 or residuals.shape[1] != keys.shape[1]:
        raise AlphaEditFactorError("keys and residuals must share non-zero rank")
    if projector.shape[0] != projector.shape[1] or projector.shape[0] != keys.shape[0]:
        raise AlphaEditFactorError("projector dimension must equal key width")

    work_dtype = _working_dtype(keys, residuals, projector)
    with torch.no_grad():
        k = keys.detach().to(dtype=work_dtype)
        resid = residuals.detach().to(dtype=work_dtype)
        p = projector.detach().to(dtype=work_dtype)
        projected_keys = p @ k
        system = p @ (k @ k.transpose(0, 1))
        system.diagonal().add_(lambda_value)
        native_rhs = projected_keys @ resid.transpose(0, 1)
        if not bool(torch.isfinite(system).all()) or not bool(
            torch.isfinite(native_rhs).all()
        ):
            raise AlphaEditFactorError("dense AlphaEdit system is non-finite")
        try:
            native_update = torch.linalg.solve(system, native_rhs)
        except RuntimeError as exc:
            raise AlphaEditFactorError("dense AlphaEdit system is singular") from exc
        if not bool(torch.isfinite(native_update).all()):
            raise AlphaEditFactorError("dense AlphaEdit update is non-finite")

        # ``R`` may contain repeated context columns (rank one for a single
        # request).  SVD supplies a deterministic orthonormal basis for its
        # numerical column space without dividing by a small residual entry.
        try:
            residual_basis, singular_values, _ = torch.linalg.svd(
                resid,
                full_matrices=False,
            )
        except RuntimeError as exc:
            raise AlphaEditFactorError("dense AlphaEdit residual basis failed") from exc
        if singular_values.numel() == 0 or not bool(torch.isfinite(singular_values).all()):
            raise AlphaEditFactorError("dense AlphaEdit residual basis is non-finite")
        cutoff = (
            torch.finfo(work_dtype).eps
            * max(int(resid.shape[0]), int(resid.shape[1]))
            * singular_values[0]
        )
        rank = int(torch.count_nonzero(singular_values > cutoff).item())
        if rank <= 0:
            raise AlphaEditFactorError("dense AlphaEdit residual basis has zero rank")
        right_basis = residual_basis[:, :rank].contiguous()
        adjusted_keys = (native_update @ right_basis).contiguous()
        if not bool(torch.isfinite(adjusted_keys).all()):
            raise AlphaEditFactorError("dense AlphaEdit low-rank projection is non-finite")

    return orient_easyedit_factor(
        adjusted_keys,
        right_basis,
        weight_name=weight_name,
        weight_shape=shape,
        expected_weight_sha256=expected_weight_sha256,
    )


def solve_historical_alphaedit_factor(
    *,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    history_keys: torch.Tensor,
    l2: float,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Factor canonical AlphaEdit with ``cache_c = H @ H.T``.

    For ``G=[K,H]`` and ``U=P@G``, Woodbury gives
    ``(lambda I + U G.T)^-1 P K = U (lambda I + G.T U)^-1 E_K``.
    Only the current-key columns selected by ``E_K`` multiply the residual.
    An empty ``H`` is valid and exactly reduces to the isolated first edit.
    """

    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    expected_weight_sha256 = _validate_sha256(
        expected_weight_sha256,
        name="expected_weight_sha256",
    )
    shape = _validate_weight_shape(weight_shape)
    lambda_value = _validate_l2(l2)
    for name, value in (
        ("keys", keys),
        ("residuals", residuals),
        ("projector", projector),
        ("history_keys", history_keys),
    ):
        _require_finite_matrix(value, name=name)
    if len({keys.device, residuals.device, projector.device, history_keys.device}) != 1:
        raise AlphaEditFactorError("historical AlphaEdit tensors must share one device")
    if keys.shape[1] == 0 or residuals.shape[1] != keys.shape[1]:
        raise AlphaEditFactorError("current keys and residuals must share non-zero rank")
    if projector.shape[0] != projector.shape[1] or projector.shape[0] != keys.shape[0]:
        raise AlphaEditFactorError("historical projector/key dimensions differ")
    if history_keys.shape[0] != keys.shape[0]:
        raise AlphaEditFactorError("historical key width differs from current keys")

    work_dtype = _working_dtype(keys, residuals, projector, history_keys)
    with torch.no_grad():
        k = keys.detach().to(dtype=work_dtype)
        resid = residuals.detach().to(dtype=work_dtype)
        p = projector.detach().to(dtype=work_dtype)
        history = history_keys.detach().to(dtype=work_dtype)
        combined = torch.cat((k, history), dim=1)
        projected = p @ combined
        small_system = combined.transpose(0, 1) @ projected
        small_system.diagonal().add_(lambda_value)
        if not bool(torch.isfinite(small_system).all()):
            raise AlphaEditFactorError("historical AlphaEdit system is non-finite")
        try:
            adjusted_all = torch.linalg.solve(
                small_system.transpose(0, 1),
                projected.transpose(0, 1),
            ).transpose(0, 1)
        except RuntimeError as exc:
            raise AlphaEditFactorError("historical AlphaEdit system is singular") from exc
        adjusted_current = adjusted_all[:, : k.shape[1]]
        if not bool(torch.isfinite(adjusted_current).all()):
            raise AlphaEditFactorError("historical AlphaEdit adjusted keys are non-finite")

    return orient_easyedit_factor(
        adjusted_current,
        resid,
        weight_name=weight_name,
        weight_shape=shape,
        expected_weight_sha256=expected_weight_sha256,
    )


def solve_unprojected_isolated_alphaedit_factor(
    *,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    l2: float,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Factor the isolated Alpha equation with the exact identity ``P = I``.

    No identity matrix is materialized: with ``P = I``, ``P @ K`` is simply
    ``K``.  This supplies the unprojected Alpha base needed for a fair
    post-hoc ``Delta W_base @ P`` ablation without importing or using MEMIT
    factors.
    """

    return _solve_isolated_alphaedit_factor(
        keys=keys,
        residuals=residuals,
        projector=None,
        l2=l2,
        weight_name=weight_name,
        weight_shape=weight_shape,
        expected_weight_sha256=expected_weight_sha256,
    )


def make_isolated_alphaedit_proposal(
    *,
    snapshot: SnapshotManifest,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    l2: float,
    weight_name: str,
    solver_suffix: str,
    semantics: ProposalSemantics = ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
    residual_denominator: int | None = 1,
) -> MemitFactorProposal:
    """Return one identity-bound genuine AlphaEdit factor proposal.

    The caller must pass the residual after its intended layer-distribution
    rule.  For an upstream-style ordered layer pass use
    ``semantics=ORDERED_GAUSS_SEIDEL`` and ``residual_denominator=None``;
    for a same-snapshot proposal use the positive common denominator that was
    used to construct ``residuals``.
    """

    if not isinstance(snapshot, SnapshotManifest):
        raise AlphaEditFactorError("snapshot must be a SnapshotManifest")
    if not isinstance(solver_suffix, str) or not solver_suffix.strip():
        raise AlphaEditFactorError("solver_suffix must not be empty")
    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    record = snapshot.parameter(weight_name)
    factor = solve_isolated_alphaedit_factor(
        keys=keys,
        residuals=residuals,
        projector=projector,
        l2=l2,
        weight_name=weight_name,
        weight_shape=record.shape,
        expected_weight_sha256=record.sha256,
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=(factor,),
        semantics=semantics,
        solver_name=(
            f"{GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX}/"
            f"{solver_suffix.strip()}"
        ),
        residual_denominator=residual_denominator,
    )


def make_upstream_dense_isolated_alphaedit_proposal(
    *,
    snapshot: SnapshotManifest,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    l2: float,
    weight_name: str,
    solver_suffix: str,
    semantics: ProposalSemantics = ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
    residual_denominator: int | None = 1,
) -> MemitFactorProposal:
    """Return one identity-bound dense-reference genuine AlphaEdit proposal."""

    if not isinstance(snapshot, SnapshotManifest):
        raise AlphaEditFactorError("snapshot must be a SnapshotManifest")
    if not isinstance(solver_suffix, str) or not solver_suffix.strip():
        raise AlphaEditFactorError("solver_suffix must not be empty")
    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    record = snapshot.parameter(weight_name)
    factor = solve_isolated_alphaedit_factor_upstream_dense(
        keys=keys,
        residuals=residuals,
        projector=projector,
        l2=l2,
        weight_name=weight_name,
        weight_shape=record.shape,
        expected_weight_sha256=record.sha256,
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=(factor,),
        semantics=semantics,
        solver_name=(
            f"{GENUINE_ISOLATED_ALPHAEDIT_DENSE_SOLVER_PREFIX}/"
            f"{solver_suffix.strip()}"
        ),
        residual_denominator=residual_denominator,
    )


def make_historical_alphaedit_proposal(
    *,
    snapshot: SnapshotManifest,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    projector: torch.Tensor,
    history_keys: torch.Tensor,
    l2: float,
    weight_name: str,
    solver_suffix: str,
    semantics: ProposalSemantics = ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
    residual_denominator: int | None = 1,
) -> MemitFactorProposal:
    """Return one identity-bound canonical history-aware AlphaEdit proposal."""

    if not isinstance(snapshot, SnapshotManifest):
        raise AlphaEditFactorError("snapshot must be a SnapshotManifest")
    if not isinstance(solver_suffix, str) or not solver_suffix.strip():
        raise AlphaEditFactorError("solver_suffix must not be empty")
    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    record = snapshot.parameter(weight_name)
    factor = solve_historical_alphaedit_factor(
        keys=keys,
        residuals=residuals,
        projector=projector,
        history_keys=history_keys,
        l2=l2,
        weight_name=weight_name,
        weight_shape=record.shape,
        expected_weight_sha256=record.sha256,
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=(factor,),
        semantics=semantics,
        solver_name=(
            f"{GENUINE_HISTORICAL_ALPHAEDIT_SOLVER_PREFIX}/"
            f"{solver_suffix.strip()}"
        ),
        residual_denominator=residual_denominator,
    )


def make_unprojected_isolated_alphaedit_proposal(
    *,
    snapshot: SnapshotManifest,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    l2: float,
    weight_name: str,
    solver_suffix: str,
    semantics: ProposalSemantics = ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
    residual_denominator: int | None = 1,
) -> MemitFactorProposal:
    """Return the identity-``P`` Alpha base for a post-hoc BP ablation.

    This is intentionally an AlphaEdit first-edit solve with ``P = I`` rather
    than a MEMIT factor.  Pair it with
    :func:`make_posthoc_alphaedit_proposal` only when measuring the algebraic
    difference between ``Delta W_base @ P`` and the genuine P-inside-solve
    AlphaEdit update.
    """

    if not isinstance(snapshot, SnapshotManifest):
        raise AlphaEditFactorError("snapshot must be a SnapshotManifest")
    if not isinstance(solver_suffix, str) or not solver_suffix.strip():
        raise AlphaEditFactorError("solver_suffix must not be empty")
    if not isinstance(weight_name, str) or not weight_name.strip():
        raise AlphaEditFactorError("weight_name must not be empty")
    record = snapshot.parameter(weight_name)
    factor = solve_unprojected_isolated_alphaedit_factor(
        keys=keys,
        residuals=residuals,
        l2=l2,
        weight_name=weight_name,
        weight_shape=record.shape,
        expected_weight_sha256=record.sha256,
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=(factor,),
        semantics=semantics,
        solver_name=(
            f"{UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX}/"
            f"{solver_suffix.strip()}"
        ),
        residual_denominator=residual_denominator,
    )


def posthoc_alphaedit_right_project_factor(
    base_factor: LowRankFactor,
    projector: torch.Tensor,
) -> LowRankFactor:
    """Return the explicitly post-hoc ``Delta W_base @ P`` factor.

    This function never re-solves AlphaEdit's normal equation.  If
    ``Delta W_base = L R.T``, then ``Delta W_base @ P = L (P.T @ R).T``.
    It accepts only the right-side orientation required by that equation and
    rejects a collapsed projected direction.
    """

    if not isinstance(base_factor, LowRankFactor):
        raise AlphaEditFactorError("base_factor must be a LowRankFactor")
    _require_finite_matrix(projector, name="projector")
    if projector.shape[0] != projector.shape[1]:
        raise AlphaEditFactorError("projector must be square")
    if (
        base_factor.left.device != base_factor.right.device
        or base_factor.right.device != projector.device
    ):
        raise AlphaEditFactorError("base factor and projector must share one device")
    if base_factor.right.shape[0] != projector.shape[0]:
        raise AlphaEditFactorError(
            "post-hoc right projection requires matching projector input width"
        )
    work_dtype = _working_dtype(base_factor.right, projector)
    with torch.no_grad():
        right = projector.detach().to(dtype=work_dtype).transpose(0, 1) @ (
            base_factor.right.detach().to(dtype=work_dtype)
        )
        projected_norm = _scalar(
            torch.linalg.vector_norm(right),
            name="post-hoc projected right norm",
        )
    if projected_norm <= 0.0:
        raise AlphaEditFactorError("post-hoc projection collapsed the update")
    return LowRankFactor(
        weight_name=base_factor.weight_name,
        left=base_factor.left,
        right=right,
        expected_weight_sha256=base_factor.expected_weight_sha256,
        native_update_transposed=base_factor.native_update_transposed,
    )


def make_posthoc_alphaedit_proposal(
    *,
    unprojected_alpha_base: MemitFactorProposal,
    projector: torch.Tensor,
    solver_suffix: str,
) -> MemitFactorProposal:
    """Build one post-hoc BP proposal from an unprojected Alpha base only.

    Restricting the input provenance to
    :class:`AlphaEditProposalKind.UNPROJECTED_ISOLATED_FIRST_EDIT` prevents a
    caller from accidentally treating a MEMIT proposal as this AlphaEdit
    ablation.  A single factor is required because each AlphaEdit layer has
    its own pinned ``P`` matrix.
    """

    if not isinstance(unprojected_alpha_base, MemitFactorProposal):
        raise AlphaEditFactorError("unprojected_alpha_base must be a factor proposal")
    if (
        alphaedit_proposal_kind(unprojected_alpha_base)
        is not AlphaEditProposalKind.UNPROJECTED_ISOLATED_FIRST_EDIT
    ):
        raise AlphaEditFactorError(
            "post-hoc AlphaEdit BP requires an unprojected isolated Alpha base"
        )
    if len(unprojected_alpha_base.factors) != 1:
        raise AlphaEditFactorError("post-hoc AlphaEdit BP requires exactly one layer factor")
    if not isinstance(solver_suffix, str) or not solver_suffix.strip():
        raise AlphaEditFactorError("solver_suffix must not be empty")
    factor = posthoc_alphaedit_right_project_factor(
        unprojected_alpha_base.factors[0],
        projector,
    )
    return MemitFactorProposal(
        snapshot=unprojected_alpha_base.snapshot,
        factors=(factor,),
        semantics=unprojected_alpha_base.semantics,
        solver_name=(
            f"{unprojected_alpha_base.solver_name}"
            f"{POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN}"
            f"unprojected-alpha-base/{solver_suffix.strip()}"
        ),
        residual_denominator=unprojected_alpha_base.residual_denominator,
    )


def alphaedit_factor_right_leak(
    factor: LowRankFactor,
    projector: torch.Tensor,
) -> AlphaEditRightLeak:
    """Measure the canonical right-side projector leak without a dense update.

    This is exactly ``||Delta W (I-P)||_F / ||Delta W||_F``.  It is defined
    only when ``P`` acts on the *right* (input) dimension of the stored model
    weight.  Fixed Llama/Qwen down-projection weights use this orientation;
    other layouts fail closed rather than silently reporting a left-side
    quantity under a right-leak name.
    """

    if not isinstance(factor, LowRankFactor):
        raise AlphaEditFactorError("factor must be a LowRankFactor")
    _require_finite_matrix(projector, name="projector")
    if projector.shape[0] != projector.shape[1]:
        raise AlphaEditFactorError("projector must be square")
    if factor.left.device != factor.right.device or factor.right.device != projector.device:
        raise AlphaEditFactorError("factor and projector must share one device")
    if factor.right.shape[0] != projector.shape[0]:
        raise AlphaEditFactorError(
            "right-leak requires projector dimension equal to factor right width"
        )

    work_dtype = _working_dtype(factor.left, factor.right, projector)
    with torch.no_grad():
        left = factor.left.detach().to(dtype=work_dtype)
        right = factor.right.detach().to(dtype=work_dtype)
        p = projector.detach().to(dtype=work_dtype)
        # (I-P).T @ R = R - P.T @ R; no d_in x d_in identity is materialized.
        residual_right = right - p.transpose(0, 1) @ right
        update_sq = _scalar(_factor_norm_sq(left, right), name="update norm squared")
        leak_sq = _scalar(
            _factor_norm_sq(left, residual_right),
            name="right leak norm squared",
        )
    update_norm = _sqrt_nonnegative(update_sq, name="update norm squared")
    leak_norm = _sqrt_nonnegative(
        leak_sq,
        name="right leak norm squared",
        scale=update_sq,
    )
    if update_norm <= 0.0:
        raise AlphaEditFactorError("right-leak requires a non-zero update")
    return AlphaEditRightLeak(
        update_frobenius=update_norm,
        right_leak_frobenius=leak_norm,
        right_leak_ratio=leak_norm / update_norm,
    )


def compare_low_rank_proposals(
    reference: MemitFactorProposal,
    candidate: MemitFactorProposal,
) -> LowRankProposalComparison:
    """Compare aligned proposals with rank-sized Gram products only.

    The reference is the denominator for the relative Frobenius difference.
    Both proposals must have the same exact guarded entry state and exactly
    the same edited weight set; factors may have different ranks and order.
    """

    if not isinstance(reference, MemitFactorProposal) or not isinstance(
        candidate, MemitFactorProposal
    ):
        raise AlphaEditFactorError("both values must be MemitFactorProposal instances")
    reference.assert_same_entry_snapshot(candidate)
    reference_factors = {factor.weight_name: factor for factor in reference.factors}
    candidate_factors = {factor.weight_name: factor for factor in candidate.factors}
    if set(reference_factors) != set(candidate_factors):
        raise AlphaEditFactorError("proposal weight sets differ")

    reference_sq = 0.0
    candidate_sq = 0.0
    inner = 0.0
    for weight_name in sorted(reference_factors):
        ref = reference_factors[weight_name]
        other = candidate_factors[weight_name]
        if ref.weight_shape != other.weight_shape:
            raise AlphaEditFactorError(f"proposal weight shape differs for {weight_name}")
        if ref.left.device != ref.right.device or other.left.device != other.right.device:
            raise AlphaEditFactorError("each factor must use one device")
        if ref.left.device != other.left.device:
            raise AlphaEditFactorError("proposal factors must share one device")
        work_dtype = _working_dtype(ref.left, ref.right, other.left, other.right)
        with torch.no_grad():
            ref_left = ref.left.detach().to(dtype=work_dtype)
            ref_right = ref.right.detach().to(dtype=work_dtype)
            other_left = other.left.detach().to(dtype=work_dtype)
            other_right = other.right.detach().to(dtype=work_dtype)
            reference_sq += _scalar(
                _factor_norm_sq(ref_left, ref_right),
                name=f"reference norm squared for {weight_name}",
            )
            candidate_sq += _scalar(
                _factor_norm_sq(other_left, other_right),
                name=f"candidate norm squared for {weight_name}",
            )
            inner += _scalar(
                _factor_inner(ref_left, ref_right, other_left, other_right),
                name=f"inner product for {weight_name}",
            )
    if not all(math.isfinite(value) for value in (reference_sq, candidate_sq, inner)):
        raise AlphaEditFactorError("proposal Gram reductions are non-finite")
    reference_norm = _sqrt_nonnegative(
        reference_sq,
        name="reference norm squared",
    )
    candidate_norm = _sqrt_nonnegative(
        candidate_sq,
        name="candidate norm squared",
    )
    if reference_norm <= 0.0 or candidate_norm <= 0.0:
        raise AlphaEditFactorError("proposal comparison requires non-zero updates")
    difference_sq = reference_sq + candidate_sq - 2.0 * inner
    difference_norm = _sqrt_nonnegative(
        difference_sq,
        name="difference norm squared",
        scale=reference_sq + candidate_sq + abs(2.0 * inner),
    )
    cosine = inner / (reference_norm * candidate_norm)
    # Numerical roundoff can leave a perfect self-comparison infinitesimally
    # outside the legal interval, but a material excursion is a contract bug.
    if cosine < -1.0 - 1e-10 or cosine > 1.0 + 1e-10:
        raise AlphaEditFactorError("proposal cosine is outside [-1, 1]")
    return LowRankProposalComparison(
        weight_count=len(reference_factors),
        reference_frobenius=reference_norm,
        candidate_frobenius=candidate_norm,
        difference_frobenius=difference_norm,
        cosine=max(-1.0, min(1.0, cosine)),
        relative_frobenius_to_reference=difference_norm / reference_norm,
    )
