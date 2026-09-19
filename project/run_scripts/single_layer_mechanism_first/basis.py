"""Bounded CPU FP64 decision basis and factor contractions.

All matrices use writer coordinates: W/D [out,in], K [in,tokens],
A [out,tokens]. No dense reference covariance or full-sized SVD is formed.
These routines do not run a model or silently transfer CUDA tensors to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

SEED = 20260919
TOP_COMPONENTS = 3
OVERSAMPLING = 5
POWER_ITERATIONS = 2
EPS64 = np.finfo(np.float64).eps


class BasisTechnicalError(ValueError):
    """Invalid/nonfinite input, not a scientific no-direction outcome."""


def _matrix(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise BasisTechnicalError(f"{name}: explicit CPU ownership required")
        value = value.detach().numpy()
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype.kind not in "fiu":
        raise BasisTechnicalError(f"{name}: expected real rank-two matrix")
    if not np.isfinite(value).all():
        raise BasisTechnicalError(f"{name}: nonfinite")
    return np.asarray(value, dtype=np.float64)


def _inner(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.einsum("ij,ij->", left, right, dtype=np.float64))


def _norm(value: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, _inner(value, value))))


@dataclass
class BasisResult:
    directions: tuple[np.ndarray, ...]
    receipt: dict

    @property
    def rank(self) -> int:
        return len(self.directions)

    def reconstruct(self, coefficients: Sequence[float]) -> np.ndarray:
        a = np.asarray(coefficients, dtype=np.float64)
        if a.shape != (self.rank,) or not np.isfinite(a).all():
            raise BasisTechnicalError("invalid coefficients")
        out = np.zeros(tuple(self.receipt["shape"]), dtype=np.float64)
        for coefficient, direction in zip(a, self.directions):
            out += coefficient * direction
        if not np.isfinite(out).all():
            raise BasisTechnicalError("nonfinite coefficient reconstruction")
        return out


def _append(candidate: np.ndarray, directions: list[np.ndarray],
            *, absolute_cutoff: float) -> dict:
    original_norm = _norm(candidate)
    residual = candidate.copy()
    for _ in range(2):
        for prior in directions:
            residual -= _inner(prior, residual) * prior
    residual_norm = _norm(residual)
    ratio = min(1.0, residual_norm / original_norm) if original_norm else 0.0
    added = residual_norm > absolute_cutoff
    if added:
        directions.append(residual / residual_norm)
    return {"input_norm": original_norm, "orthogonal_residual_norm": residual_norm,
            "angle_to_existing_span_radians": float(np.arcsin(ratio)),
            "dependency_absolute_cutoff": absolute_cutoff, "added": added,
            "reason": "ADDED" if added else "ZERO_OR_NUMERICALLY_DEPENDENT"}


def _basis_audit(target: np.ndarray, directions: Sequence[np.ndarray]) -> dict:
    gram = np.array([[_inner(a, b) for b in directions] for a in directions])
    residual = target.copy()
    coefficients = [_inner(target, d) for d in directions]
    for coefficient, direction in zip(coefficients, directions):
        residual -= coefficient * direction
    target_norm = _norm(target)
    error = _norm(residual)
    return {"gradient_span_coefficients": coefficients,
            "gradient_span_absolute_error": error,
            "gradient_span_relative_error": error / target_norm if target_norm else 0.0,
            "orthonormal_gram": gram.tolist(),
            "orthonormal_max_abs_error": float(np.max(np.abs(gram - np.eye(len(directions)))))
            if directions else 0.0}


def build_functional_basis(projected_descent: Any, *,
                           project: Callable[[np.ndarray], Any] | None = None,
                           threads: int = 8) -> BasisResult:
    """Randomized top-three plus exact residual of the supplied H=-GQ.

The top-three are approximate randomized singular components, not certified
exact singular vectors. The untruncated residual makes H lie in their joint
span, subject to the reported FP64 projection/MGS/dependency tolerance.
"""
    h = _matrix(projected_descent, "projected_descent")
    h_norm = _norm(h)
    relative_cutoff = max(h.shape, default=0) * EPS64
    cutoff = relative_cutoff * h_norm
    receipt = {"schema_version": 1, "shape": list(h.shape), "dtype": "float64",
               "seed": SEED, "randomized_top_components": TOP_COMPONENTS,
               "oversampling": OVERSAMPLING, "power_iterations": POWER_ITERATIONS,
               "dense_full_svd": False, "target_norm": h_norm,
               "dependency_relative_cutoff": relative_cutoff,
               "orthogonalization_passes": 2, "components": []}
    if h_norm == 0 or min(h.shape, default=0) == 0:
        return BasisResult((), {**receipt, "status": "NO_PROJECTED_DIRECTION",
                                "rank": 0, **_basis_audit(h, ())})
    with threadpool_limits(limits=threads):
        width = min(min(h.shape), TOP_COMPONENTS + OVERSAMPLING)
        omega = np.random.default_rng(SEED).standard_normal((h.shape[1], width))
        left, _ = linalg.qr(h @ omega, mode="economic", check_finite=False)
        for _ in range(POWER_ITERATIONS):
            right, _ = linalg.qr(h.T @ left, mode="economic", check_finite=False)
            left, _ = linalg.qr(h @ right, mode="economic", check_finite=False)
        small = left.T @ h
        us, singular, vh = linalg.svd(small, full_matrices=False, check_finite=False)
        ul = left @ us
        residual = h.copy()
        directions: list[np.ndarray] = []
        for index in range(min(TOP_COMPONENTS, len(singular))):
            component = np.outer(ul[:, index] * singular[index], vh[index])
            residual -= component
            # A right-space projection is linear in each writer row. Preserve
            # the known rank-one factorization rather than multiplying a full
            # 4096x14336 component through Q three separate times.
            if project:
                projected_right = _matrix(project(vh[index:index+1]), "projected component right factor")
                if projected_right.shape != (1,h.shape[1]):
                    raise BasisTechnicalError("right-space projector changed row-factor shape")
                projected = np.outer(ul[:,index]*singular[index], projected_right[0])
            else:
                projected = component
            if projected.shape != h.shape:
                raise BasisTechnicalError("projector changed shape")
            row = _append(projected, directions, absolute_cutoff=cutoff)
            receipt["components"].append({"kind": "RANDOMIZED_SINGULAR_COMPONENT",
                                           "index": index, "singular_value": float(singular[index]),
                                           "projection_route": "RANK_ONE_RIGHT_FACTOR_FP64" if project else "ALREADY_PROJECTED_INPUT", **row})
        receipt["unprojected_exact_residual_norm"] = _norm(residual)
        projected = _matrix(project(residual), "projected residual") if project else residual
        if projected.shape != h.shape:
            raise BasisTechnicalError("projector changed residual shape")
        row = _append(projected, directions, absolute_cutoff=cutoff)
        receipt["components"].append({"kind": "EXACT_REMAINDER_BEFORE_REPROJECTION", **row})
        receipt.update(_basis_audit(h, directions))
    receipt["span_correctness_relative_ceiling"] = 32 * relative_cutoff
    if receipt["gradient_span_relative_error"] > receipt["span_correctness_relative_ceiling"]:
        raise BasisTechnicalError("functional basis does not reconstruct the supplied projected gradient")
    receipt.update({"status": "BASIS_BUILT", "rank": len(directions),
                    "span_statement": "FP64_RECONSTRUCTION_WITH_REPORTED_RESIDUAL_NOT_BITEXACT"})
    return BasisResult(tuple(directions), receipt)


def covariance_action(displacement: Any, document_keys: Iterable[Any], *,
                      expected_documents: int = 512,
                      project: Callable[[np.ndarray], Any] | None = None,
                      threads: int = 8) -> tuple[np.ndarray, dict]:
    """Return -E C_R Q with document means; keys are streamed once in order."""
    e = _matrix(displacement, "displacement")
    if expected_documents < 1:
        raise BasisTechnicalError("positive expected_documents required")
    accumulated = np.zeros_like(e)
    tokens = []
    with threadpool_limits(limits=threads):
        for index, raw in enumerate(document_keys):
            if index >= expected_documents:
                raise BasisTechnicalError("excess reference documents")
            k = _matrix(raw, "document keys")
            if k.shape[0] != e.shape[1] or k.shape[1] < 1:
                raise BasisTechnicalError("key shape/empty-document mismatch")
            accumulated -= ((e @ k) @ k.T) / (expected_documents * k.shape[1])
            tokens.append(k.shape[1])
        if len(tokens) != expected_documents:
            raise BasisTechnicalError("missing reference documents")
        action = _matrix(project(accumulated), "projected covariance action") if project else accumulated
    if action.shape != e.shape:
        raise BasisTechnicalError("projector changed covariance action shape")
    return action, {"documents": len(tokens), "valid_input_tokens": sum(tokens),
                    "tokens_per_document": tokens, "dense_covariance_created": False,
                    "reduction": "FP64_DOCUMENT_ORDER_DOCUMENT_TOKEN_MEAN_THEN_DOCUMENT_MEAN",
                    "unprojected_action_norm": _norm(accumulated), "projected_action_norm": _norm(action)}


def append_covariance_direction(basis: BasisResult, displacement: Any,
                                document_keys: Iterable[Any], *,
                                expected_documents: int = 512,
                                project: Callable[[np.ndarray], Any] | None = None,
                                threads: int = 8) -> BasisResult:
    if basis.rank > 4 or "covariance" in basis.receipt:
        raise BasisTechnicalError("only one covariance direction after <=4 functional directions")
    action, evidence = covariance_action(displacement, document_keys,
                                         expected_documents=expected_documents,
                                         project=project, threads=threads)
    if list(action.shape) != basis.receipt["shape"]:
        raise BasisTechnicalError("covariance and basis shape mismatch")
    directions = list(basis.directions)
    cutoff = max(action.shape) * EPS64 * _norm(action)
    row = _append(action, directions, absolute_cutoff=cutoff)
    gram = np.array([[_inner(a, b) for b in directions] for a in directions])
    receipt = {**basis.receipt, "functional_rank": basis.rank,
               "functional_orthonormal_gram": basis.receipt["orthonormal_gram"],
               "covariance": {**evidence, **row}, "rank": len(directions),
               "orthonormal_gram": gram.tolist(),
               "orthonormal_max_abs_error": float(np.max(np.abs(gram - np.eye(len(directions)))))
               if directions else 0.0}
    return BasisResult(tuple(directions), receipt)


def contract_factors(activation_gradient: Any, keys: Any,
                     basis: BasisResult | Sequence[np.ndarray], *, threads: int = 8) -> np.ndarray:
    """J_k=<A,D_k K>; no model or dense per-document writer gradient."""
    a, k = _matrix(activation_gradient, "A"), _matrix(keys, "K")
    # A BasisResult is validated at construction. Do not rescan five dense
    # immutable writer directions for every one of the 512 factor documents.
    directions = basis.directions if isinstance(basis, BasisResult) else tuple(
        _matrix(direction, "direction") for direction in basis)
    if a.shape[1] != k.shape[1]:
        raise BasisTechnicalError("factor token dimensions differ")
    output = []
    with threadpool_limits(limits=threads):
        for direction in directions:
            d = direction
            if d.shape != (a.shape[0], k.shape[0]):
                raise BasisTechnicalError("factor/basis writer coordinate mismatch")
            output.append(_inner(a, d @ k))
    result = np.asarray(output, dtype=np.float64)
    if not np.isfinite(result).all():
        raise BasisTechnicalError("nonfinite factor contraction")
    return result
