"""Validate FE's ridge prior and repair only storage-roundoff indefiniteness.

An uncentered second moment is positive semidefinite. Casting an already
rounded FP32 cache to FP64 cannot restore that property. An unrestricted LU
solve can succeed on the resulting indefinite ridge system and produce an
arbitrarily large update. Keep healthy priors byte-identical; a correction of
an unhealthy prior requires an explicit policy and produces an audit record.
"""
from contextlib import contextmanager

import numpy as np
import torch

POLICIES = ("strict", "project_roundoff")


class FECovarianceError(ValueError):
    pass


def prepare_covariance(covariance, *, policy="strict"):
    """Return (prior, receipt), without modifying the supplied cache tensor.

    The optional repair is the nearest symmetric matrix whose eigenvalues
    exceed the source cache's resolution eps_storage * ||C||_F (or the FP64
    factorization scale, if larger). Clipping negatives to zero leaves an
    unidentifiable, nearly singular prior and can make the update even larger.
    Only eigenvalues below the resolution change. Negative eigenvalues larger than
    one FP32/FP64 storage-resolution unit of the whole matrix are rejected;
    such a cache needs investigation/recomputation, not this repair.
    """
    if policy not in POLICIES:
        raise FECovarianceError("FE_COVARIANCE_POLICY")
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] == 0
            or covariance.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(covariance).all()):
        raise FECovarianceError("FE_COVARIANCE_SHAPE_DTYPE_FINITE")
    if not torch.equal(covariance, covariance.T):
        raise FECovarianceError("FE_COVARIANCE_ASYMMETRIC")
    work = covariance.detach().to(device="cpu", dtype=torch.float64, copy=True)
    factor, info = torch.linalg.cholesky_ex(work)
    del factor
    receipt = dict(policy=policy, input_dtype=str(covariance.dtype),
                   shape=list(covariance.shape), cholesky_info=int(info),
                   repaired=False, corrected_eigenvalues=0)
    if int(info) == 0:
        return covariance, receipt
    from scipy.linalg import eigh
    norm = float(torch.linalg.vector_norm(work))
    if not np.isfinite(norm) or norm == 0:
        raise FECovarianceError("FE_COVARIANCE_ZERO_OR_INVALID_SCALE")
    storage_bound = torch.finfo(covariance.dtype).eps * norm
    factorization_floor = torch.finfo(torch.float64).eps * work.shape[0] * norm
    floor = max(storage_bound, factorization_floor)
    if policy == "strict":
        minimum = float(eigh(work.numpy(), subset_by_index=(0, 0),
                             eigvals_only=True, check_finite=False, driver="evr")[0])
        if minimum < -factorization_floor:
            raise FECovarianceError(
                "FE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE: inspect the source cache; "
                "project_roundoff must be explicitly selected for a new run")
        receipt.update(minimum_eigenvalue=minimum, psd_singular=True)
        return covariance, receipt
    values, vectors = eigh(work.numpy(), subset_by_value=(-np.inf, floor),
                           check_finite=False, driver="evr")
    if len(values) == 0 or not np.isfinite(values).all():
        raise FECovarianceError("FE_COVARIANCE_EIGEN_AUDIT_FAILED")
    if float(values[0]) < -storage_bound:
        raise FECovarianceError("FE_COVARIANCE_NEGATIVITY_EXCEEDS_STORAGE_ROUNDOFF")
    values = torch.from_numpy(values.copy())
    vectors = torch.from_numpy(vectors.copy())
    residual = torch.linalg.vector_norm(work @ vectors - vectors * values, dim=0)
    if not torch.isfinite(residual).all() or float(residual.max()) > 16 * factorization_floor:
        raise FECovarianceError("FE_COVARIANCE_EIGEN_RESIDUAL")
    shifts = floor - values
    corrected = work + (vectors * shifts) @ vectors.T
    # The two triangular halves can differ by a final FP64 multiplication bit.
    corrected = (corrected + corrected.T) * 0.5
    factor, final_info = torch.linalg.cholesky_ex(corrected)
    del factor
    if int(final_info) != 0:
        raise FECovarianceError("FE_COVARIANCE_REPAIR_NOT_POSITIVE_DEFINITE")
    receipt.update(repaired=True, corrected_eigenvalues=len(values),
                   original_eigenvalues=values.tolist(), eigenvalue_floor=floor,
                   eigen_residual_norms=residual.tolist(), matrix_frobenius=norm,
                   storage_roundoff_bound=storage_bound,
                   correction_frobenius=float(shifts.norm()),
                   relative_correction=float(shifts.norm()) / norm,
                   output_dtype="torch.float64", repaired_cholesky_info=0)
    return corrected.to(covariance.device), receipt


class _LinalgProxy:
    def __init__(self, original, observations):
        self.original = original
        self.observations = observations

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, system, rhs, *args, **kwargs):
        if (system.ndim != 2 or system.shape[0] != system.shape[1]
                or system.dtype != torch.float64
                or not torch.isfinite(system).all()
                or not torch.isfinite(rhs).all()):
            raise FECovarianceError("FE_RIDGE_SYSTEM_SHAPE_DTYPE_FINITE")
        if not torch.equal(system, system.T):
            raise FECovarianceError("FE_RIDGE_SYSTEM_ASYMMETRIC")
        factor, info = self.original.cholesky_ex(system)
        del factor
        if int(info) != 0:
            raise FECovarianceError(
                "FE_RIDGE_SYSTEM_NOT_POSITIVE_DEFINITE: no weight update committed "
                "for this layer; inspect covariance and accumulated history")
        # Keep the author's solve and arithmetic order on the validated system.
        result = self.original.solve(system, rhs, *args, **kwargs)
        if not torch.isfinite(result).all():
            raise FECovarianceError("FE_RIDGE_UPDATE_NONFINITE")
        self.observations.append(dict(cholesky_info=0,
                                      update_frobenius=float(result.norm())))
        return result


class _TorchProxy:
    def __init__(self, original, observations):
        self.original = original
        self.linalg = _LinalgProxy(original.linalg, observations)

    def __getattr__(self, name):
        return getattr(self.original, name)


def checked_ridge_solve(system, rhs):
    """The same positive-definiteness gate for the EasyEdit FE history path."""
    return _LinalgProxy(torch.linalg, []).solve(system, rhs)


@contextmanager
def guarded_author_solve(native):
    """Guard only this author module; never monkeypatch the global torch module."""
    original = native.torch
    observations = []
    native.torch = _TorchProxy(original, observations)
    try:
        yield observations
    finally:
        native.torch = original
