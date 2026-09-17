"""CPU-only FP64 geometry for the sealed L4-preserving repair contract.

No model execution, target fitting, GPU operation, or scientific acceptance is
performed here.  Full selected-weight arrays are read in bounded FP64 chunks;
only the at-most-three output directions are materialized in FP32.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GeometryPolicy:
    """Technical choices fixed before any actual-model quality observation."""

    gradient_zero_absolute: float = 1e-12
    dependency_relative: float = 1e-8
    basis_fp32_gram_absolute: float = 2e-7
    psd_negative_absolute: float = 1e-14
    psd_negative_relative: float = 1e-10
    hessian_zero_absolute: float = 1e-18
    condition_cap: float = 1e6
    whitening_residual_absolute: float = 1e-8
    symmetry_absolute: float = 1e-14
    symmetry_relative: float = 1e-10
    chunk_elements: int = 262144


DEFAULT_GEOMETRY_POLICY = GeometryPolicy()


class GeometryError(RuntimeError):
    def __init__(self, code: str, receipt: dict | None = None):
        super().__init__(code)
        self.code = code
        self.receipt = receipt or {}


def _array(value):
    # Tensor inputs must already be CPU; do not silently transfer large GPU data.
    if hasattr(value, "device") and str(value.device) != "cpu":
        raise GeometryError("BASIS_INPUT_MUST_BE_CPU")
    if hasattr(value, "detach"):
        value = value.detach().numpy()
    result = np.asarray(value)
    if result.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise GeometryError("BASIS_INPUT_DTYPE")
    if not result.flags.c_contiguous:
        raise GeometryError("BASIS_INPUT_MUST_BE_CONTIGUOUS")
    return result


def _gram(arrays, chunk_elements):
    flat = [value.reshape(-1) for value in arrays]
    matrix = np.zeros((len(flat), len(flat)), dtype=np.float64)
    if not flat:
        return matrix
    for start in range(0, flat[0].size, chunk_elements):
        block = np.stack([x[start:start + chunk_elements] for x in flat]).astype(
            np.float64, copy=False
        )
        if not np.isfinite(block).all():
            raise GeometryError("NONFINITE_BASIS_INPUT")
        matrix += block @ block.T
    return matrix


def orthonormal_basis(
    gradients: Mapping[str, object], policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY
):
    """Return ``(list[FP32 ndarray], JSON-safe receipt)`` in declared input order.

    A two-pass modified Gram--Schmidt QR is carried out in FP64. Directions are
    represented by coefficients of the original gradients. Inner products and
    residual norms are evaluated on the actual chunked vectors, avoiding the
    catastrophic cancellation of ``c.T @ Gram @ c`` for dependent gradients.
    The output's *actual FP32* Gram is independently recorded and checked.
    """
    names = list(gradients)
    if len(names) > 3:
        raise GeometryError("BASIS_MORE_THAN_THREE_GRADIENTS")
    arrays = [_array(gradients[name]) for name in names]
    shape = arrays[0].shape if arrays else ()
    if any(value.shape != shape for value in arrays):
        raise GeometryError("BASIS_SHAPE_MISMATCH")
    if arrays and arrays[0].size == 0:
        raise GeometryError("BASIS_EMPTY_WEIGHT")
    flat = [value.reshape(-1) for value in arrays]
    original_gram = _gram(arrays, policy.chunk_elements)
    coefficients: list[np.ndarray] = []
    rows = []

    def inner(left, right):
        total = 0.0
        for start in range(0, flat[0].size, policy.chunk_elements):
            block = np.stack([x[start:start + policy.chunk_elements] for x in flat])
            block = block.astype(np.float64, copy=False)
            lv, rv = left @ block, right @ block
            total += float(lv @ rv)
        return total

    for index, name in enumerate(names):
        original_norm = math.sqrt(max(0.0, float(original_gram[index, index])))
        coefficient = np.eye(len(names), dtype=np.float64)[index]
        if original_norm <= policy.gradient_zero_absolute:
            rows.append({"name": name, "status": "NUMERICAL_ZERO", "norm": original_norm})
            continue
        for _ in range(2):
            for basis_coefficient in coefficients:
                coefficient -= inner(basis_coefficient, coefficient) * basis_coefficient
        residual = math.sqrt(max(0.0, inner(coefficient, coefficient)))
        threshold = max(policy.gradient_zero_absolute, policy.dependency_relative * original_norm)
        row = {"name": name, "norm": original_norm, "residual_norm": residual,
               "drop_threshold": threshold, "relative_residual": residual / original_norm}
        if residual <= threshold:
            row["status"] = "NUMERICAL_DEPENDENCY"
        else:
            coefficient /= residual
            coefficients.append(coefficient)
            row["status"] = "RETAINED"
        rows.append(row)

    basis = [np.empty(shape, dtype=np.float32) for _ in coefficients]
    if arrays:
        for start in range(0, flat[0].size, policy.chunk_elements):
            block = np.stack([x[start:start + policy.chunk_elements] for x in flat])
            block = block.astype(np.float64, copy=False)
            for output, coefficient in zip(basis, coefficients, strict=True):
                output.reshape(-1)[start:start + policy.chunk_elements] = coefficient @ block
    actual_gram = _gram(basis, policy.chunk_elements)
    residual = float(np.max(np.abs(actual_gram - np.eye(len(basis))))) if basis else 0.0
    receipt = {"method": "FP64_TWO_PASS_STREAMED_MODIFIED_GRAM_SCHMIDT_QR",
               "policy": asdict(policy), "gradient_names": names, "shape": list(shape),
               "input_gram_fp64": original_gram.tolist(), "input_status": rows,
               "basis_coefficients_fp64": [x.tolist() for x in coefficients],
               "actual_fp32_basis_gram": actual_gram.tolist(),
               "actual_fp32_basis_gram_max_abs_residual": residual,
               "actual_dimension": len(basis), "output_dtype": "float32"}
    if residual > policy.basis_fp32_gram_absolute:
        raise GeometryError("BASIS_FP32_ORTHOGONALITY_UNRESOLVED", receipt)
    return basis, receipt


def whiten_gn(H, b, policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY):
    """Return JSON-safe minimal-shift whitening receipt; ``R`` is a nested list.

    Only sub-roundoff asymmetry is averaged before ``eigh`` and its magnitude is
    reported. Significant negative curvature and unresolved near-zero curvature
    are technical errors, never normal no-repair fallback.
    """
    H, b = np.asarray(H, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if b.ndim != 1 or b.size > 3 or H.shape != (b.size, b.size):
        raise GeometryError("GN_SHAPE")
    if not np.isfinite(H).all() or not np.isfinite(b).all():
        raise GeometryError("GN_NONFINITE")
    if b.size == 0:
        return {"status": "ZERO_DIMENSION", "R": [], "lambda_num": 0.0,
                "eigenvalues": [], "policy": asdict(policy)}
    scale = float(np.max(np.abs(H)))
    asymmetry = float(np.max(np.abs(H - H.T)))
    symmetry_limit = policy.symmetry_absolute + policy.symmetry_relative * scale
    if asymmetry > symmetry_limit:
        raise GeometryError("GN_ASYMMETRIC", {"asymmetry": asymmetry, "limit": symmetry_limit})
    symmetric = (H + H.T) * 0.5
    values, vectors = np.linalg.eigh(symmetric)
    hmin, hmax = float(values[0]), float(values[-1])
    negative_limit = policy.psd_negative_absolute + policy.psd_negative_relative * max(abs(hmax), abs(hmin))
    base = {"policy": asdict(policy), "H_input": H.tolist(), "b": b.tolist(),
            "eigenvalues": values.tolist(), "asymmetry_max_abs": asymmetry,
            "negative_eigenvalue_tolerance": negative_limit}
    if hmin < -negative_limit:
        raise GeometryError("GN_SIGNIFICANT_NEGATIVE_EIGENVALUE", base)
    if hmax <= policy.hessian_zero_absolute:
        if float(np.linalg.norm(b)) > policy.gradient_zero_absolute:
            raise GeometryError("GN_ZERO_WITH_SIGNIFICANT_GRADIENT", base)
        return {**base, "status": "NUMERICAL_ZERO_GN_AND_GRADIENT", "R": None,
                "lambda_num": 0.0, "whitening_residual_max_abs": None}
    shift = max(0.0, (hmax - policy.condition_cap * hmin) / (policy.condition_cap - 1.0))
    shifted = values + shift
    if np.min(shifted) <= 0:
        raise GeometryError("GN_SHIFT_NOT_POSITIVE", {**base, "lambda_num": shift})
    R = (vectors * (1.0 / np.sqrt(shifted))) @ vectors.T
    Htilde = symmetric + shift * np.eye(b.size)
    whitening_error = float(np.max(np.abs(R.T @ Htilde @ R - np.eye(b.size))))
    condition = float(shifted[-1] / shifted[0])
    receipt = {**base, "status": "WHITENED", "lambda_num": shift,
               "Htilde": Htilde.tolist(), "R": R.tolist(),
               "condition_number": condition,
               "whitening_residual_max_abs": whitening_error}
    if whitening_error > policy.whitening_residual_absolute or condition > policy.condition_cap * (1 + 1e-9):
        raise GeometryError("GN_WHITENING_UNRESOLVED", receipt)
    return receipt
