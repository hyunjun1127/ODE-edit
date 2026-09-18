"""Small NumPy/FP64 fixtures for edit-preserving correction mathematics.

This is a CPU mathematical reference, NOT a production editor, rank estimator
for model activations, optimizer, or certificate for FP32 neural execution.
Only small matrices are supported intentionally: SVD uses full_matrices=True.
The ambiguity band is an explicit numerical DESIGN CHOICE, not model evidence.
"""

from __future__ import annotations

import numpy as np


AMBIGUITY_BAND_FACTOR = 10.0


def matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("expected a finite rank-two matrix")
    return result


def rank_diagnostic(value):
    """Rank of captured entries, with no performance-dependent truncation.

    tau=max(shape)*eps64*s_max. A singular value in [tau/10, 10*tau]
    makes the fixture unresolved; it must NOT unlock a repair direction.
    This says nothing about the error in activations captured in FP32.
    """
    value = matrix(value)
    singular = np.linalg.svd(value, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(value.shape, default=0) * np.finfo(np.float64).eps * largest
    if largest == 0.0:
        rank, ambiguous = 0, False
    else:
        rank = int(np.count_nonzero(singular > threshold))
        ambiguous = bool(np.any((singular >= threshold / AMBIGUITY_BAND_FACTOR)
                                & (singular <= threshold * AMBIGUITY_BAND_FACTOR)))
    return {
        "rank": rank,
        "status": "RANK_UNRESOLVED" if ambiguous else "RESOLVED",
        "threshold": float(threshold),
        "singular_values": singular.tolist(),
        "ambiguity_band_factor": AMBIGUITY_BAND_FACTOR,
        "policy": "FP64 captured-matrix diagnostic; unresolved means no repair",
    }


def orthogonal_projector_basis(projector, tolerance=1e-12):
    """Validate an ideal toy P; do not silently reinterpret native P_raw."""
    projector = matrix(projector)
    if projector.shape[0] != projector.shape[1]:
        raise ValueError("projector must be square")
    scale = max(1.0, float(np.linalg.norm(projector)))
    if (np.linalg.norm(projector - projector.T) > tolerance * scale
            or np.linalg.norm(projector @ projector - projector) > tolerance * scale):
        raise ValueError("input is not an orthogonal projector within toy tolerance")
    eigenvalues, vectors = np.linalg.eigh(projector)
    return vectors[:, eigenvalues > 0.5]


def intersection(projector, edit_keys):
    """Return P intersect ker(K_E.T); no-repair on ambiguous rank."""
    projector, edit_keys = matrix(projector), matrix(edit_keys)
    if projector.shape[0] != edit_keys.shape[0]:
        raise ValueError("input dimensions differ")
    allowed = orthogonal_projector_basis(projector)
    reduced = allowed.T @ edit_keys
    diagnostic = rank_diagnostic(reduced)
    result = {"projector_rank": allowed.shape[1], "rank": diagnostic,
              "status": diagnostic["status"]}
    if diagnostic["status"] != "RESOLVED":
        return {**result, "null_dimension": None, "Q": None, "N": None}
    left, _, _ = np.linalg.svd(reduced, full_matrices=True)
    null_basis = allowed @ left[:, diagnostic["rank"]:]
    q = null_basis @ null_basis.T
    null_dimension = null_basis.shape[1]
    return {**result, "status": "REPAIR_SPACE_EMPTY" if null_dimension == 0 else "RESOLVED",
            "null_dimension": null_dimension, "Q": q, "N": null_basis}


def native_family_dimension(right_factor, edit_keys, output_dimension=1):
    """Physical dimension of {CA : CA K_E=0}, excluding coefficient gauges.

    Per output row this is rank(A)-rank(A K_E). It is invariant to any
    invertible change of coefficient basis. Gauge freedom alone changes no W.
    """
    right_factor, edit_keys = matrix(right_factor), matrix(edit_keys)
    if right_factor.shape[1] != edit_keys.shape[0]:
        raise ValueError("input dimensions differ")
    if not isinstance(output_dimension, int) or output_dimension < 1:
        raise ValueError("positive integer output dimension required")
    a_rank = rank_diagnostic(right_factor)
    if a_rank["status"] != "RESOLVED":
        return {"status": "RANK_UNRESOLVED"}
    _, _, vh = np.linalg.svd(right_factor, full_matrices=False)
    row_basis = vh[:a_rank["rank"]]
    response_rank = rank_diagnostic(row_basis @ edit_keys)
    if response_rank["status"] != "RESOLVED":
        return {"status": "RANK_UNRESOLVED"}
    row_dimension = a_rank["rank"] - response_rank["rank"]
    return {
        "status": "RESOLVED", "rank_A": a_rank["rank"],
        "rank_response": response_rank["rank"],
        "coefficient_gauge_per_output": right_factor.shape[0] - a_rank["rank"],
        "physical_dimension_per_output": row_dimension,
        "physical_dimension": output_dimension * row_dimension,
    }


def descent_diagnostic(gradient, allowed_projector, edit_null_projector):
    gradient = matrix(gradient)
    allowed_projector, edit_null_projector = matrix(allowed_projector), matrix(edit_null_projector)
    orthogonal_projector_basis(allowed_projector)
    orthogonal_projector_basis(edit_null_projector)
    if np.linalg.norm(edit_null_projector @ allowed_projector - edit_null_projector) > 1e-12:
        raise ValueError("edit nullspace is not contained in allowed space")
    projected = gradient @ edit_null_projector
    chi = float(np.sum(projected * projected))
    denominator = float(np.sum((gradient @ allowed_projector) ** 2))
    return {"direction": -projected, "chi": chi,
            "directional_derivative": -chi,
            "remaining_gradient_fraction": None if denominator == 0.0 else chi / denominator}


def covariance_energy(delta, covariance):
    delta, covariance = matrix(delta), matrix(covariance)
    return float(np.sum((delta @ covariance) * delta))


def native_ridge(keys, residual, covariance):
    """Toy P=I ridge minimizer; preserves the declared objective exactly."""
    keys, residual, covariance = matrix(keys), matrix(residual), matrix(covariance)
    if np.linalg.norm(covariance - covariance.T) > 1e-12 or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("strictly positive definite symmetric covariance required")
    return residual @ np.linalg.solve(keys @ keys.T + covariance, keys).T


def nonlinear_suffix(hidden):
    """Tiny deterministic contextual map, not a transformer surrogate claim."""
    hidden = matrix(hidden)
    causal_sum = np.triu(np.ones((hidden.shape[1], hidden.shape[1])))
    return np.tanh(np.tanh(hidden) @ causal_sum)
