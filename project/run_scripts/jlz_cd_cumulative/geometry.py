"""Frozen full-context CD entry geometry; no candidate or causal solves.

The dense metric, Cholesky, key factors and owner writer map are scratch.
Only S/C/J and the retained energy factor N survive the one entry build.
No tensor in this cache is a durable artifact or a replacement commit solve.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from project.run_scripts.jlz_realized_writer.geometry import prepare_metric, owner_weights
from project.run_scripts.jlz_realization.common import digest, tensor_sha as _source_tensor_sha


def tensor_sha(value):
    # Historical helper's memoryview cast rejects zero-length retained factors.
    # Preserve its exact shape/dtype header and empty byte stream without
    # touching the source namespace or changing any nonempty tensor identity.
    if value.numel() == 0:
        return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode()).hexdigest()
    return _source_tensor_sha(value)


def _require(ok, label):
    if not ok:
        raise RuntimeError(label)


def _finite(value, label):
    _require(bool(torch.isfinite(value).all()), label + '_NONFINITE')


@torch.no_grad()
def load_native_c0(stat_path, shape, *, device='cpu'):
    """Same native FP32 moment/count division as build_prior_from_npz.

    Return the *raw* FP64 covariance.  The entry builder uses its symmetric
    representation for alignment, exactly as prepare_metric does for A.
    """
    with np.load(Path(stat_path), allow_pickle=False) as archive:
        raw_array = archive['mom2.mom2']
        count = archive['mom2.count']
        _require(count.size == 1, 'C0_COUNT_SHAPE')
        count_value = float(count.item())
        _require(math.isfinite(count_value) and count_value > 0
                 and count_value.is_integer(), 'C0_COUNT')
        _require(raw_array.dtype == np.float32 and raw_array.shape == tuple(shape), 'C0_NATIVE_SHAPE_DTYPE')
        raw = torch.from_numpy(raw_array)
    _finite(raw, 'C0_NATIVE')
    raw.div_(int(count_value))
    return raw.to(device=device, dtype=torch.float64)


@dataclass(frozen=True)
class FrozenCD:
    S: torch.Tensor                 # [B,q] full context response
    C: torch.Tensor                 # [B,B] full cross-owner cost metric
    J: torch.Tensor                 # [d_out,B] actual cumulative alignment
    N: torch.Tensor                 # [B,rank] nonnegative energy reference
    owners: torch.Tensor            # [q], runtime occurrence ownership
    weights: torch.Tensor           # [q], native weighted LS weights
    lambda_c: float
    receipt: dict

    @property
    def request_count(self):
        return self.S.shape[0]

    @property
    def retained_rank(self):
        return self.N.shape[1]

    @property
    def operator_hash(self):
        return self.receipt['operator_hash']

    @property
    def full_operator_compatible(self):
        return self.receipt['full_operator_compatible']

    @property
    def operator_compatible(self):
        return self.full_operator_compatible

    @property
    def lambda_cov(self):
        return self.lambda_c

    def to(self, device):
        """Move only compact detached caches, preserving entry identities."""
        return replace(self, **{name: getattr(self, name).to(device).detach()
                                for name in ('S', 'C', 'J', 'N', 'owners', 'weights')})


@torch.no_grad()
def build_geometry(keys, owners, raw_metric, C0, Wentry, Winitial,
                   lambda_cov=15000., *, request_count=None, identity=None,
                   cache_device=None):
    """Build once from all native rewrite+KL rows at one arm/batch entry.

    raw_metric must be the unchanged native build_prior_from_npz FP64 output;
    its symmetrization and QR-SVD cutoff are exactly the V13 writer's.  C0
    must already include native FP32 count division.  Physical Wentry/Winitial
    are selected FP32 weights in writer orientation, not sums of ideal U.
    The caller may place all scratch on CPU and drop/recompute factors at
    actual commit.  This path does not silently use a lower precision solve.
    """
    started = time.monotonic()
    _require(keys.ndim == 2 and min(keys.shape) > 0, 'ENTRY_KEYS_SHAPE')
    n, q = keys.shape
    device = raw_metric.device
    _require(raw_metric.dtype == torch.float64 and raw_metric.shape == (n, n), 'ENTRY_METRIC_SHAPE_DTYPE')
    _require(C0.ndim == 2 and C0.shape == (n, n) and C0.is_floating_point(), 'ENTRY_C0_SHAPE')
    _require(Wentry.dtype == Winitial.dtype == torch.float32
             and Wentry.ndim == 2 and Wentry.shape == Winitial.shape
             and Wentry.shape[1] == n, 'ENTRY_PHYSICAL_WEIGHT_SHAPE_DTYPE')
    _require(math.isfinite(lambda_cov) and lambda_cov > 0, 'ENTRY_LAMBDA_C')
    _require(owners.ndim == 1 and owners.shape[0] == q
             and owners.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8), 'ENTRY_OWNER_DTYPE_SHAPE')
    owners = owners.detach().to(device=device, dtype=torch.long)
    B = int(owners.max()) + 1 if request_count is None else request_count
    _require(isinstance(B, int) and not isinstance(B, bool) and B > 0, 'ENTRY_REQUEST_COUNT')
    _require(bool(((owners >= 0) & (owners < B)).all()), 'ENTRY_OWNER_RANGE')
    weights = owner_weights(owners, B)
    K = keys.detach().to(device=device, dtype=torch.float64)
    cov = C0.detach().to(device=device, dtype=torch.float64)
    _finite(K, 'ENTRY_K'); _finite(cov, 'ENTRY_C0')
    C0_symmetric = (cov + cov.T) * .5
    del cov
    metric_started = time.monotonic()
    A, L, metric = prepare_metric(raw_metric)
    metric['factor_seconds'] = time.monotonic() - metric_started
    E = torch.zeros((B, q), dtype=torch.float64, device=device)
    E[owners, torch.arange(q, device=device)] = 1.
    root = weights.sqrt()
    # Match V13 operation order: weighted K before triangular solve.
    X = torch.linalg.solve_triangular(L, K * root, upper=False)
    factor_started = time.monotonic()
    Q, R = torch.linalg.qr(X, mode='reduced')
    left, singular, right = torch.linalg.svd(R, full_matrices=False)
    qr_svd_seconds = time.monotonic() - factor_started
    cutoff = float(torch.finfo(torch.float64).eps * max(X.shape) * singular[0])
    keep = singular > cutoff
    rank = int(keep.sum())
    basis = Q @ left[:, keep]
    vr = right[keep]
    compressed = (E * root) @ vr.T
    N = compressed / singular[keep]
    white = N @ basis.T
    Bmap = torch.linalg.solve_triangular(L.T, white.T, upper=True).T
    S = Bmap @ K
    C = N @ N.T
    # Reconstruct physical displacement only here, then release it.
    alignment_started = time.monotonic()
    delta = Wentry.detach().to(device=device, dtype=torch.float64)
    delta.sub_(Winitial.detach().to(device=device, dtype=torch.float64))
    delta_zero = not bool(torch.count_nonzero(delta))
    J = delta @ (C0_symmetric @ Bmap.T)
    alignment_seconds = time.monotonic() - alignment_started
    _finite(S, 'ENTRY_S'); _finite(C, 'ENTRY_C'); _finite(J, 'ENTRY_J'); _finite(N, 'ENTRY_N')
    weighted_response = S * root
    expected_response = compressed @ vr
    response_error = float((weighted_response - expected_response).norm())
    response_tolerance = 1e-10 + 1e-8 * float(expected_response.norm())
    _require(response_error <= response_tolerance, 'ENTRY_RESPONSE_NUMERICAL_FAILURE')
    owner_error = float(((S - E) * root).norm())
    operator_tolerance = 1e-10 + 1e-8 * float((E * root).norm())
    hashes = dict(K=tensor_sha(K), owners=tensor_sha(owners), weights=tensor_sha(weights),
                  raw_metric=tensor_sha(raw_metric), A=tensor_sha(A), C0_symmetric=tensor_sha(C0_symmetric),
                  Wentry=tensor_sha(Wentry), Winitial=tensor_sha(Winitial),
                  singular=tensor_sha(singular[keep]), S=tensor_sha(S), C=tensor_sha(C),
                  J=tensor_sha(J), N=tensor_sha(N))
    receipt = dict(schema='JLZ_CD_CUMULATIVE_FROZEN_ENTRY_V1', identity=identity or {},
                   shapes=dict(input=n, output=Wentry.shape[0], requests=B, native_rows=q),
                   owner_counts=torch.bincount(owners, minlength=B).tolist(),
                   hashes=hashes, lambda_c=float(lambda_cov), rank=rank,
                   singular_values=singular.tolist(), cutoff=cutoff,
                   cutoff_rule='eps64*max(X.shape)*sigma_max; strict>',
                   positive_discarded_count=int(((singular > 0) & ~keep).sum()),
                   retained_condition=float(singular[keep][0] / singular[keep][-1]) if rank else None,
                   metric=metric, response_numerical_verified=True,
                   response_error=response_error, response_tolerance=response_tolerance,
                   full_operator_compatible=owner_error <= operator_tolerance,
                   full_operator_owner_error=owner_error, full_operator_tolerance=operator_tolerance,
                   response_fastpath_used=False, delta_zero=delta_zero,
                   full_cross_owner_metric=True, zero_columns_retained=True,
                   scratch_device=str(device), dense_factor_retained=False,
                   commit_cholesky_recompute_required=True, cholesky_count=1, qr_svd_count=1,
                   qr_svd_seconds=qr_svd_seconds, alignment_seconds=alignment_seconds,
                   substep_timings_included_in_total=True,
                   source_solver='project.run_scripts.jlz_realized_writer.geometry',
                   seconds=time.monotonic() - started, persisted_tensors=False)
    # The cache lifetime binds its own arm/batch identity, while the numerical
    # operator hash can agree across the two independently built cold arms.
    receipt['operator_hash'] = digest(dict(hashes=hashes, cutoff=receipt['cutoff_rule'],
                                          lambda_c=float(lambda_cov)))
    geometry = FrozenCD(*(v.detach() for v in (S, C, J, N, owners, weights)), float(lambda_cov), receipt)
    return geometry if cache_device is None else geometry.to(cache_device)


def _target(D, geometry):
    _require(D.ndim == 2 and D.shape == geometry.J.shape, 'COST_TARGET_SHAPE')
    _require(D.dtype in (torch.float32, torch.float64), 'COST_TARGET_DTYPE')
    _finite(D, 'COST_TARGET')
    return D.double()


def projected_action(D, geometry, row_indices=None):
    """Full cross-owner differentiable action, cast back to model FP32.

    The default never substitutes incidence E, even for compatible D values.
    Every physical group therefore has a gradient path to every owner block.
    """
    target = _target(D, geometry)
    S = geometry.S if row_indices is None else geometry.S[:, row_indices]
    return (target @ S.to(D.device)).to(D.dtype)


def writer_cost(D, geometry, *, alpha, allocation_price=1., include_components=False):
    """FP64 full quadratic/alignment and selected sign(0)=0 D gradient."""
    _require(alpha in (0, 1), 'COST_ALPHA')
    _require(math.isfinite(allocation_price) and allocation_price >= 0, 'COST_PRICE')
    D64 = _target(D, geometry)
    C = geometry.C.to(D.device); J = geometry.J.to(D.device); N = geometry.N.to(D.device)
    DC = D64 @ C
    Q = (DC * D64).sum()
    factor_Q = (D64 @ N).square().sum()
    _finite(Q, 'COST_Q'); _finite(factor_Q, 'COST_FACTOR_Q')
    _require(abs(float(Q.detach() - factor_Q.detach())) <= 1e-10 + 1e-8 * float(factor_Q.detach()), 'COST_FACTOR_ENERGY_MISMATCH')
    cross = (D64 * J).sum()
    cumulative = 2. * alpha * geometry.lambda_c * cross.abs()
    Pi = Q + cumulative
    Q_gradient = 2. * DC
    cumulative_gradient = 2. * alpha * geometry.lambda_c * cross.sign() * J
    gradient = Q_gradient + cumulative_gradient
    _finite(gradient, 'COST_GRADIENT')
    result = dict(Q=float(Q.detach()), factor_Q=float(factor_Q.detach()),
                  c=float(cross.detach()), cross=float(cross.detach()),
                  cumulative=float(cumulative.detach()), Pi=float(Pi.detach()),
                  weighted_cost=float((allocation_price * Pi).detach()),
                  gradient=allocation_price * gradient)
    if include_components:
        # Transient private tensors, consumed and discarded by the u wrapper.
        # These reuse DC/J and never add a solve, matmul, backward, or decision.
        result['_Q_gradient_D'] = allocation_price * Q_gradient
        result['_cumulative_gradient_D'] = allocation_price * cumulative_gradient
    return result


def requested_action(u, anchors):
    """Declared FP32 multiply and storage boundary, not a FP64 D rewrite."""
    _require(u.dtype == anchors.dtype == torch.float32
             and u.ndim == 2 and anchors.shape == (u.shape[1],), 'REQUESTED_ACTION_FP32_SHAPE')
    return u * anchors.to(u.device)[None, :]


def cost_gradient_u(u, anchors, geometry, *, alpha, allocation_price=1.):
    """Analytic exact autograd cast-chain: g_D64 -> g_D32 -> a32*g_D32."""
    D = requested_action(u, anchors)
    result = writer_cost(D, geometry, alpha=alpha, allocation_price=allocation_price,
                         include_components=True)
    gradient = result.pop('gradient').to(u.dtype) * anchors.to(u.device)[None, :]
    Q_gradient = result.pop('_Q_gradient_D').to(u.dtype) * anchors.to(u.device)[None, :]
    cumulative_gradient = result.pop('_cumulative_gradient_D').to(u.dtype) * anchors.to(u.device)[None, :]
    result['Q_gradient_u_l2'] = Q_gradient.double().norm(dim=0).tolist()
    result['cumulative_gradient_u_l2'] = cumulative_gradient.double().norm(dim=0).tolist()
    result['component_gradient_norm_definition'] = 'separately weighted FP64 D components -> FP32 -> anchor32; FP64 per-owner norms'
    result['component_sum_may_differ_from_combined_cast'] = True
    _finite(gradient, 'COST_U_GRADIENT')
    return gradient, result


def realization_diagnostics(D, geometry):
    target = _target(D, geometry)
    owners = geometry.owners.to(D.device)
    root = geometry.weights.to(D.device).sqrt()
    owner_target = target[:, owners]
    action = target @ geometry.S.to(D.device)
    denominator = float((owner_target * root).norm())
    error = float(((owner_target - action) * root).norm())
    return dict(weighted_target_norm=denominator, discarded_target_norm=error,
                discarded_target_relative=error / denominator if denominator else 0.,
                requested_norm=float(target.norm()), projected_norm=float(action.norm()),
                projected_response_used=True, rank=geometry.retained_rank)
