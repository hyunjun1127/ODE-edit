"""CPU-only geometry and actual-writer diagnostics for AlphaEdit key causality.

Samples are rows except for the explicitly named writer operands K[d,n],
R[o,n], delta[o,d].  All diagnostic reductions use FP64; no native operand is
modified and no function here is an alternative native writer.  Spectra use
the exact sample Gram (at most 1,000 rows), never a feature-space full SVD.
Roundoff-negative Gram eigenvalues are reported and clipped only inside the
diagnostic summary.  Significant negative curvature is a technical error.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


MAX_EXACT_SAMPLES = 1000
CONTEXT_WEIGHTS = (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)


class GeometryError(ValueError):
    """Invalid/nonfinite operands or substantial negative curvature."""


def _matrix(value: Any, name: str, *, samples: bool = False) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise GeometryError(f'{name}: expected a FP32/FP64 matrix')
    if min(value.shape) < 1:
        raise GeometryError(f'{name}: empty matrix')
    if samples and value.shape[0] > MAX_EXACT_SAMPLES:
        raise GeometryError(f'{name}: exact Gram limited to {MAX_EXACT_SAMPLES} samples')
    # Do not allocate a d*d bool tensor for large, memory-mapped P/M matrices.
    for first in range(0, value.shape[0], 128):
        if not np.isfinite(value[first:first + 128]).all():
            raise GeometryError(f'{name}: nonfinite input')
    return value


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(value).all():
        raise GeometryError(f'{name}: nonfinite diagnostic arithmetic')
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _eigh_psd(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Only accepts a small symmetric diagnostic Gram; clipping is recorded."""
    gram = _matrix(gram, 'gram', samples=True).astype(np.float64, copy=False)
    if gram.shape[0] != gram.shape[1]:
        raise GeometryError('gram must be square')
    scale = float(np.linalg.norm(gram, ord=np.inf))
    tolerance = 64.0 * np.finfo(np.float64).eps * len(gram) * scale
    asymmetry = float(np.max(np.abs(gram - gram.T)))
    if asymmetry > tolerance:
        raise GeometryError(f'gram asymmetry {asymmetry} > numerical bound {tolerance}')
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    _finite(values, 'eigenvalues')
    minimum = float(values[0])
    if minimum < -tolerance:
        raise GeometryError(f'negative Gram eigenvalue {minimum} < {-tolerance}')
    negative = values < 0
    receipt = {
        'minimum_eigenvalue_before_roundoff_clip': minimum,
        'negative_roundoff_count': int(negative.sum()),
        'negative_roundoff_mass': float(-values[negative].sum()),
        'negative_tolerance': tolerance,
        'asymmetry_max': asymmetry,
        'diagnostic_symmetrization_only': True,
        'native_operands_modified': False,
    }
    values = np.maximum(values, 0.0)[::-1]
    return values, vectors[:, ::-1], receipt


def _spectrum(gram: np.ndarray) -> dict[str, Any]:
    eigenvalues, _, receipt = _eigh_psd(gram)
    trace = float(eigenvalues.sum())
    squared = float(eigenvalues @ eigenvalues)
    return {
        'status': 'ZERO_ENERGY' if trace == 0 else 'DEFINED',
        'trace': trace,
        'participation_ratio': _ratio(trace * trace, squared),
        'eigenvalues_descending': eigenvalues.tolist(),
        'top1_energy_share': _ratio(float(eigenvalues[:1].sum()), trace),
        'top5_energy_share': _ratio(float(eigenvalues[:5].sum()), trace),
        'top20_energy_share': _ratio(float(eigenvalues[:20].sum()), trace),
        'numerical': receipt,
    }


def _center_gram(gram: np.ndarray) -> np.ndarray:
    row = gram.mean(axis=1, keepdims=True)
    return gram - row - row.T + float(gram.mean())


def geometry_summary(samples: np.ndarray, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Raw/centered/unit geometry for one <=1000-request panel.

    Zero keys remain in raw and centered denominators.  Unit geometry excludes
    exactly zero rows, with explicit count/indices; no epsilon normalization.
    The caller, not this function, supplies projected samples q=P k when needed.
    """
    original = _matrix(samples, 'samples', samples=True)
    keys = original.astype(np.float64, copy=False)
    gram = _finite(keys @ keys.T, 'sample Gram')
    energy = _finite(np.einsum('ij,ij->i', keys, keys), 'row energies')
    total = float(energy.sum())
    mean = keys.mean(axis=0)
    mean_energy = float(len(keys) * (mean @ mean))
    centered = keys - mean
    # Direct centered Gram avoids catastrophic cancellation of nearly constant
    # keys.  Unit-centered uses explicit unit rows for the same reason.
    centered_gram = _finite(centered @ centered.T, 'centered Gram')
    nonzero = energy > 0.0
    if nonzero.any():
        unit = keys[nonzero] / np.sqrt(energy[nonzero, None])
        unit_centered = unit - unit.mean(axis=0)
        unit_raw = _spectrum(_finite(unit @ unit.T, 'unit Gram'))
        unit_cov = _spectrum(_finite(unit_centered @ unit_centered.T, 'unit centered Gram'))
    else:
        unit_raw = unit_cov = {'status': 'NO_NONZERO_SAMPLES', 'participation_ratio': None}
    descending = np.sort(energy)[::-1]
    return {
        'schema': 'alpha-key-geometry-v1',
        'orientation': 'samples_as_rows',
        'input_dtype': str(original.dtype), 'reduction_dtype': 'float64',
        'n': len(keys), 'd': keys.shape[1], 'metadata': dict(metadata or {}),
        'raw': _spectrum(gram), 'centered': _spectrum(centered_gram),
        'unit_raw': unit_raw, 'unit_centered': unit_cov,
        'mean_energy_fraction': _ratio(mean_energy, total),
        'norm_energy_ess': _ratio(total * total, float(energy @ energy)),
        'top1pct_sample_count': int(math.ceil(0.01 * len(keys))),
        'top5pct_sample_count': int(math.ceil(0.05 * len(keys))),
        'top1pct_sample_energy_share': _ratio(float(descending[:math.ceil(.01 * len(keys))].sum()), total),
        'top5pct_sample_energy_share': _ratio(float(descending[:math.ceil(.05 * len(keys))].sum()), total),
        'norm_energy_per_sample': energy.tolist(),
        'zero_norm_count': int((~nonzero).sum()),
        'zero_norm_indices': np.flatnonzero(~nonzero).tolist(),
        'unit_n': int(nonzero.sum()),
        'zero_policy': 'raw_and_centered_include; unit_excludes_exact_zero; no_epsilon',
    }


def compare_keys(a: np.ndarray, b: np.ndarray, *, request_ids_a: Sequence[Any] | None = None,
                 request_ids_b: Sequence[Any] | None = None) -> dict[str, Any]:
    """Paired same-request diagnostics; norm ratio is ||b_i||/||a_i||.

    IDs, when provided, must be equal and unique: positional pairing is never
    silently changed.  Raw/centered Gram alignment is Frobenius cosine, not a
    claim of functional equivalence. Pairwise distances exclude diagonal pairs.
    """
    a = _matrix(a, 'a', samples=True).astype(np.float64, copy=False)
    b = _matrix(b, 'b', samples=True).astype(np.float64, copy=False)
    if a.shape != b.shape:
        raise GeometryError('paired key shapes differ')
    if (request_ids_a is None) != (request_ids_b is None):
        raise GeometryError('both ID sequences are required')
    identity = 'POSITION_ONLY_CALLER_RESPONSIBLE'
    if request_ids_a is not None:
        ids_a, ids_b = list(request_ids_a), list(request_ids_b)
        if ids_a != ids_b or len(ids_a) != len(a) or len(set(ids_a)) != len(a):
            raise GeometryError('paired ID mismatch, duplicate, or cardinality error')
        identity = 'EXACT_ORDERED_IDS'
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 0) & (nb > 0)
    cosine: list[float | None] = [None] * len(a)
    ratio: list[float | None] = [None] * len(a)
    for i in range(len(a)):
        if valid[i]:
            cosine[i] = float(np.dot(a[i] / na[i], b[i] / nb[i]))
        if na[i] > 0:
            ratio[i] = float(nb[i] / na[i])
    ga, gb = _finite(a @ a.T, 'paired Gram a'), _finite(b @ b.T, 'paired Gram b')
    ca, cb = a - a.mean(axis=0), b - b.mean(axis=0)
    cga, cgb = ca @ ca.T, cb @ cb.T
    def alignment(x: np.ndarray, y: np.ndarray) -> float | None:
        return _ratio(float(np.sum(x * y)), float(np.linalg.norm(x) * np.linalg.norm(y)))
    upper = np.triu_indices(len(a), k=1)
    d2a = np.maximum(np.diag(ga)[:, None] + np.diag(ga)[None, :] - 2 * ga, 0.0)[upper]
    d2b = np.maximum(np.diag(gb)[:, None] + np.diag(gb)[None, :] - 2 * gb, 0.0)[upper]
    distance_change = np.sqrt(d2b) - np.sqrt(d2a)
    return {
        'schema': 'alpha-key-paired-v1', 'identity': identity, 'n': len(a),
        'cosine_per_request': cosine, 'norm_ratio_b_over_a': ratio,
        'cosine_defined_n': int(valid.sum()),
        'zero_norm_a_indices': np.flatnonzero(na == 0).tolist(),
        'zero_norm_b_indices': np.flatnonzero(nb == 0).tolist(),
        'key_change_norm': np.linalg.norm(b - a, axis=1).tolist(),
        'raw_gram_alignment': alignment(ga, gb),
        'centered_gram_alignment': alignment(cga, cgb),
        'pairwise_distances': {
            'unique_unordered_pairs': len(d2a),
            'a_mean': None if not len(d2a) else float(np.sqrt(d2a).mean()),
            'b_mean': None if not len(d2b) else float(np.sqrt(d2b).mean()),
            'change_mean': None if not len(d2a) else float(distance_change.mean()),
            'change_rms': None if not len(d2a) else float(np.sqrt(np.mean(distance_change ** 2))),
            'change_quantiles_0_50_95_99_100': None if not len(d2a) else np.quantile(distance_change, [0, .5, .95, .99, 1]).tolist(),
        },
    }


def context_geometry(context_keys: np.ndarray, *, layer: int, space: str = 'raw',
                     writer_mean: np.ndarray | None = None,
                     metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Six per-context summaries plus native bare/generated group weighting.

    Pass the actual native writer_mean when it was captured.  Otherwise the CPU
    grouped-mean reconstruction is explicitly NOT a native bitwise assertion.
    No pooled 6000x6000 Gram is allocated.
    """
    keys = np.asarray(context_keys)
    if keys.ndim != 3 or keys.shape[1] != 6:
        raise GeometryError('context_keys must have shape [requests,6,features]')
    for index in range(6):
        _matrix(keys[:, index, :], f'context{index}', samples=True)
    common = dict(metadata or {}, layer=int(layer), space=space)
    calculated = np.stack((keys[:, 0], keys[:, 1:].mean(axis=1)), axis=1).mean(axis=1)
    actual = calculated if writer_mean is None else _matrix(writer_mean, 'writer_mean', samples=True)
    if actual.shape != calculated.shape:
        raise GeometryError('writer_mean shape mismatch')
    return {
        'schema': 'alpha-context-geometry-v1', 'layer': int(layer), 'space': space,
        'context_names': ['bare', 'generated0', 'generated1', 'generated2', 'generated3', 'generated4'],
        'context_weights': list(CONTEXT_WEIGHTS),
        'group_reduction': 'mean([bare,mean(generated5)])',
        'writer_mean_source': 'CAPTURED_NATIVE' if writer_mean is not None else 'CPU_GROUP_RECONSTRUCTION_NOT_NATIVE_BITWISE',
        'reconstruction_max_abs_difference': float(np.max(np.abs(actual.astype(np.float64) - calculated.astype(np.float64)))),
        'contexts': [geometry_summary(keys[:, j], metadata=dict(common, context=j)) for j in range(6)],
        'writer_mean': geometry_summary(actual, metadata=dict(common, context='native_group_mean')),
    }


def _action_energy(delta: np.ndarray, keys: np.ndarray, block: int = 128) -> np.ndarray:
    """Column-wise ||delta k||² without storing a full output-token action."""
    answer = np.zeros(keys.shape[1], dtype=np.float64)
    for start in range(0, delta.shape[0], block):
        action = _finite(delta[start:start + block].astype(np.float64) @ keys, 'action')
        answer += np.einsum('ij,ij->j', action, action)
    return _finite(answer, 'action energy')


def _quadratic_trace(left: np.ndarray, middle: np.ndarray, block: int) -> float:
    """trace(left middle left.T), with only block x d extra scratch."""
    result = 0.0
    for start in range(0, left.shape[0], block):
        rows = left[start:start + block].astype(np.float64)
        product = np.zeros_like(rows)
        # Block conversion avoids an implicit entire FP64 copy of M.
        for first in range(0, middle.shape[0], block):
            product += rows[:, first:first + block] @ middle[first:first + block].astype(np.float64)
        result += float(np.sum(rows * product))
    if not math.isfinite(result):
        raise GeometryError('nonfinite quadratic trace')
    return result


def history_coverage(timestamp_keys: np.ndarray, current_keys: np.ndarray, actual_delta: np.ndarray,
                     *, M: np.ndarray | None = None, P: np.ndarray | None = None,
                     totals: Mapping[str, float] | None = None, block_size: int = 128) -> dict[str, Any]:
    """All bank records weight1, including superseded records.

    `totals` may contain independently bound projected_history_trace and
    native_write_history_penalty to avoid repeating expensive d² reductions.
    The caller owns that identity binding. P is used as stored, not orthogonalized.
    Exact trace diagnostics cost O(d³) for dense P/M, but use bounded scratch;
    compute/reuse the totals once per state rather than for every branch.
    """
    ks, kc = _matrix(timestamp_keys, 'timestamp_keys'), _matrix(current_keys, 'current_keys')
    delta = _matrix(actual_delta, 'actual_delta')
    if ks.shape != kc.shape or delta.shape[1] != ks.shape[0]:
        raise GeometryError('history key/delta shape mismatch')
    if block_size < 1:
        raise GeometryError('block_size must be positive')
    ks, kc = ks.astype(np.float64, copy=False), kc.astype(np.float64, copy=False)
    stamp_penalty = _action_energy(delta, ks, block_size)
    current_penalty = _action_energy(delta, kc, block_size)
    if P is None:
        projected_stamp_trace = projected_cur_trace = None
    else:
        P = _matrix(P, 'P')
        if P.shape != (ks.shape[0], ks.shape[0]):
            raise GeometryError('P shape mismatch')
        projected_stamp_trace = float(_action_energy(P, ks, block_size).sum())
        projected_cur_trace = float(_action_energy(P, kc, block_size).sum())
    totals = dict(totals or {})
    for name, value in totals.items():
        if not math.isfinite(value):
            raise GeometryError(f'nonfinite history total {name}')
    total_trace = totals.get('projected_history_trace')
    total_penalty = totals.get('native_write_history_penalty')
    if M is not None:
        M = _matrix(M, 'M')
        if M.shape != (ks.shape[0], ks.shape[0]):
            raise GeometryError('M shape mismatch')
        if total_penalty is None:
            total_penalty = _quadratic_trace(delta, M, block_size)
        if total_trace is None and P is not None:
            total_trace = _quadratic_trace(P, M, block_size)
    return {
        'schema': 'alpha-history-coverage-v1', 'bank_n': ks.shape[1], 'bank_weight': 1,
        'superseded_excluded_from_statistics': False,
        'stamp_penalty': float(stamp_penalty.sum()), 'current_penalty': float(current_penalty.sum()),
        'penalty_signed_current_minus_stamp_per_request': (current_penalty - stamp_penalty).tolist(),
        'stamp_penalty_per_request': stamp_penalty.tolist(), 'current_penalty_per_request': current_penalty.tolist(),
        'projected_stamp_trace': projected_stamp_trace, 'projected_current_trace': projected_cur_trace,
        'projected_history_trace': total_trace, 'native_write_history_penalty': total_penalty,
        'projected_trace_coverage': None if projected_stamp_trace is None or total_trace is None else _ratio(projected_stamp_trace, total_trace),
        'native_write_penalty_coverage': None if total_penalty is None else _ratio(float(stamp_penalty.sum()), total_penalty),
        'total_identity': 'CALLER_BOUND' if totals else 'DIRECT_GIVEN_OPERANDS' if M is not None else 'NOT_MEASURED',
        'projector_policy': 'stored P as supplied; no orthogonalization',
        'native_operands_modified': False,
    }


def writer_modes(K: np.ndarray, R: np.ndarray, actual_delta: np.ndarray, *,
                 history_whitened_gram: np.ndarray | None = None,
                 timestamp_keys: np.ndarray | None = None, current_history_keys: np.ndarray | None = None,
                 M: np.ndarray | None = None, P: np.ndarray | None = None,
                 history_totals: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Actual Delta K response plus optional caller-supplied ideal H diagnostics.

    H=Z.T A^-1 Z must come from the same state, P basis, lambda and key binding.
    This module does not solve the native writer or assume stored P is exactly
    orthogonal. The ideal relation is explicitly compared with *actual* Delta K.
    Degenerate eigenvector demand may depend on eigensolver basis; aggregate
    response/error identities are meaningful without a unique individual mode.
    """
    k, residual, delta = _matrix(K, 'K'), _matrix(R, 'R'), _matrix(actual_delta, 'actual_delta')
    if k.shape[1] > MAX_EXACT_SAMPLES:
        raise GeometryError('writer sample Gram exceeds exact limit')
    if residual.shape[1] != k.shape[1] or delta.shape != (residual.shape[0], k.shape[0]):
        raise GeometryError('K[d,n]/R[o,n]/delta[o,d] orientation mismatch')
    k, residual = k.astype(np.float64, copy=False), residual.astype(np.float64, copy=False)
    response = np.empty_like(residual)
    for start in range(0, delta.shape[0], 128):
        response[start:start + 128] = delta[start:start + 128].astype(np.float64) @ k
    _finite(response, 'actual Delta K')
    response_energy = np.einsum('ij,ij->j', response, response)
    target_energy = np.einsum('ij,ij->j', residual, residual)
    remaining = residual - response
    result: dict[str, Any] = {
        'schema': 'alpha-writer-modes-v1',
        'orientation': {'K': 'd,n', 'R': 'o,n', 'actual_delta': 'o,d'},
        'n': k.shape[1], 'input_dim': k.shape[0], 'output_dim': residual.shape[0],
        'native_solver_replaced': False, 'diagnostic_dtype': 'float64',
        'actual_response_energy': float(response_energy.sum()),
        'target_demand_energy': float(target_energy.sum()),
        'actual_remaining_error_energy': float(np.sum(remaining * remaining)),
        'actual_response_energy_per_request': response_energy.tolist(),
        'target_demand_energy_per_request': target_energy.tolist(),
        'actual_gain_along_target_per_request': [_ratio(float(np.dot(residual[:, i], response[:, i])), float(target_energy[i])) for i in range(k.shape[1])],
        'history_whitened_modes_status': 'NOT_MEASURED',
    }
    if history_whitened_gram is not None:
        h = _matrix(history_whitened_gram, 'H', samples=True)
        if h.shape != (k.shape[1], k.shape[1]):
            raise GeometryError('H shape mismatch')
        values, vectors, numerical = _eigh_psd(h)
        demand = residual @ vectors
        actual_modes = response @ vectors
        modal_demand_energy = np.einsum('ij,ij->j', demand, demand)
        gain = values / (1.0 + values)
        ideal_response = (demand * gain) @ vectors.T
        ideal_error = float(np.sum(modal_demand_energy / (1.0 + values) ** 2))
        result.update({
            'history_whitened_modes_status': 'CALLER_SUPPLIED_H_DIAGNOSTIC_ONLY',
            'mode_eigenvalues': values.tolist(), 'ideal_mode_gain': gain.tolist(),
            'mode_target_demand_energy': modal_demand_energy.tolist(),
            'mode_actual_response_energy': np.sum(actual_modes * actual_modes, axis=0).tolist(),
            'mode_actual_gain_along_target': [_ratio(float(np.dot(demand[:, i], actual_modes[:, i])), float(modal_demand_energy[i])) for i in range(len(values))],
            'ideal_remaining_error_formula': ideal_error,
            'ideal_remaining_error_direct': float(np.sum((residual - ideal_response) ** 2)),
            'actual_vs_ideal_response_frobenius': float(np.linalg.norm(response - ideal_response)),
            'actual_vs_ideal_response_relative': _ratio(float(np.linalg.norm(response - ideal_response)), float(np.linalg.norm(ideal_response))),
            'H_numerical': numerical,
            'degenerate_mode_basis_warning': 'individual demand in repeated eigenvalue blocks is basis-dependent',
        })
    if (timestamp_keys is None) != (current_history_keys is None):
        raise GeometryError('both timestamp and current history keys are required')
    if timestamp_keys is not None:
        result['history_coverage'] = history_coverage(timestamp_keys, current_history_keys, delta, M=M, P=P, totals=history_totals)
    else:
        result['history_coverage'] = {'status': 'NOT_MEASURED'}
    return result
