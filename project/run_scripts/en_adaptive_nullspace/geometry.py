"""Weighted current geometry, one unchanged FP64 TSQR/SVD, no model execution.

All original captured columns remain present. Logical identity is supplied by
an oracle manifest, never inferred from approximately equal key vectors.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math
import numpy as np
from threadpoolctl import threadpool_limits
from project.run_scripts.single_layer_edit_preserving_correction import geometry as legacy


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def nested_weights(requests, columns):
    """requests=[{sequences:[{identity: hash, columns:[int,...]}]}].

    Duplicate sequence occurrences share their unique sequence mass equally.
    Their actual byte columns remain separate. A reused cache column receives
    summed alias mass. Every nonpadding input position must be listed once per
    occurrence; the caller seals tokens/mask/positions/BOS in identity.
    """
    if not requests:
        raise ValueError('empty requests')
    weights = np.zeros(columns, dtype=np.float64)
    receipt = []
    for request_index, request in enumerate(requests):
        sequences = request['sequences']
        counts = Counter(s['identity'] for s in sequences)
        if not counts:
            raise ValueError('request has no protection sequences')
        lengths = {}
        for seq in sequences:
            length = len(seq['columns'])
            if seq['identity'] in lengths and lengths[seq['identity']] != length:
                raise ValueError('same full sequence identity has differing valid lengths')
            lengths[seq['identity']] = length
        per_request = 0.
        for seq in sequences:
            ids = np.asarray(seq['columns'], dtype=np.int64)
            if not len(ids) or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= columns):
                raise ValueError('invalid input-position column map')
            mass = 1. / (len(requests) * len(counts) * counts[seq['identity']])
            weight = mass / len(ids)
            np.add.at(weights, ids, weight)
            per_request += mass
            receipt.append(dict(request_index=request_index, sequence_identity=seq['identity'],
                                columns=ids.tolist(), position_weight=weight,
                                duplicate_occurrences=counts[seq['identity']]))
        if not math.isclose(per_request, 1. / len(requests), abs_tol=1e-14):
            raise ValueError('request mass mismatch')
    if np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1., abs_tol=1e-13):
        raise ValueError('missing columns or nonunit total mass')
    return weights, receipt


def request_alias_weights(manifest):
    """Per-request alias mass, independent of other cases sharing byte columns.

    Pass this list as controller ``request_columns``. Weights remain global
    (sum 1/B per case); diagnostics normalizes each request conditionally.
    """
    cases = manifest['case_ids']
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('unique request order required')
    masses = {case: {} for case in cases}
    for alias in manifest['aliases']:
        case, column, weight = alias['case_id'], alias['actual_key_column'], alias['weight']
        if case not in masses or not isinstance(column, int) or column < 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError('invalid current alias')
        masses[case][column] = masses[case].get(column, 0.) + weight
    result = []
    for case in cases:
        ids = sorted(masses[case])
        weights = [masses[case][i] for i in ids]
        total = math.fsum(weights)
        if not math.isclose(total, 1./len(cases), abs_tol=1e-12):
            raise ValueError('case alias mass must equal 1/B')
        result.append(dict(case_id=case, columns=ids, weights=weights, total_weight=total))
    return result


def first_representatives(prefix_identities):
    first, result = {}, []
    for i, identity in enumerate(prefix_identities):
        result.append(first.setdefault(identity, i))
    return np.asarray(result, dtype=np.int64)


@dataclass
class WeightedGeometry:
    keys: np.ndarray
    weights: np.ndarray
    representatives: np.ndarray
    basis: np.ndarray
    vectors: np.ndarray  # descending singular order, only exact-blocked modes
    singular: np.ndarray  # complete min(original_shape) singular spectrum
    diagnostic: dict
    block_columns: int = 2048

    @property
    def rank(self):
        return self.vectors.shape[1]

    @property
    def numerical_released(self):
        return int(np.count_nonzero(self.singular[:self.rank] <= self.diagnostic['numerical_cutoff']))

    @property
    def group_ends(self):
        s = self.singular[:self.rank][::-1]
        return [i + 1 for i in range(len(s)) if i + 1 == len(s) or s[i + 1] != s[i]]

    def space(self, released_modes=0):
        if released_modes not in [0, *self.group_ends]:
            raise ValueError('invalid boundary or split equal singular group')
        blocked = self.vectors[:, :self.rank - released_modes]
        return legacy.RightSpace(self.basis, blocked, 'RESOLVED',
                                 dict(kind='EN_ADAPTIVE', released_modes=released_modes))

    def compare_exact_space(self, previous_blocked):
        """Operator Frobenius difference in the SAME sealed V coordinates.

        Accept a previously captured unweighted exact blocked basis, avoiding
        a second SVD. Direct residuals avoid near-equal trace subtraction.
        Input-key and V provenance equality remains the caller's obligation.
        """
        old = np.asarray(array(previous_blocked), dtype=np.float64)
        if old.ndim != 2 or old.shape[0] != self.basis.shape[1]:
            raise ValueError('previous blocked coordinates mismatch')
        with threadpool_limits(limits=8):
            a = old - self.vectors @ (self.vectors.T @ old)
            b = self.vectors - old @ (old.T @ self.vectors)
            difference = math.sqrt(float(np.sum(a*a) + np.sum(b*b)))
        return dict(previous_blocked_rank=old.shape[1], weighted_blocked_rank=self.rank,
                    projector_difference_frobenius=difference,
                    denominator='max(1,sqrt(previous_exact_dimension))',
                    projector_difference_relative=difference / max(1., math.sqrt(self.basis.shape[1]-old.shape[1])),
                    extra_svd_calls=0)

    def direction(self, gradient, row):
        return self.space(int(row['released_modes'])).project(array(gradient))

    def blocks(self):
        for start in range(0, self.keys.shape[1], self.block_columns):
            stop = min(start + self.block_columns, self.keys.shape[1])
            yield start, np.asarray(self.keys[:, start:stop], dtype=np.float64) * np.sqrt(self.weights[start:stop])

    def action_norm(self, delta):
        d = np.asarray(array(delta), dtype=np.float64)
        total = 0.
        with threadpool_limits(limits=8):
            for _, block in self.blocks():
                response = d @ block
                total += float(np.sum(response * response))
        return math.sqrt(total)

    def gradient_spectrum(self, gradient, native_delta, loss, row_block=128):
        g = np.asarray(array(gradient), dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != self.keys.shape[0] or not np.isfinite(g).all():
            raise ValueError('invalid gradient')
        energies = np.zeros(self.rank, dtype=np.float64)
        exact_energy = allowed_energy = inner_exact = 0.
        # Direct residual matrix, not subtraction of two large squared norms.
        with threadpool_limits(limits=8):
            for start in range(0, g.shape[0], row_block):
                part = g[start:start + row_block]
                gv = part @ self.basis
                modes = gv @ self.vectors
                residual = gv - modes @ self.vectors.T
                exact = residual @ self.basis.T
                exact_energy += float(np.sum(exact * exact))
                inner_exact += float(np.sum(part * exact))
                allowed_energy += float(np.sum(gv * gv))
                energies += np.sum(modes * modes, axis=0)
        d = np.asarray(array(native_delta), dtype=np.float64)
        if d.shape != g.shape or not np.isfinite(d).all():
            raise ValueError('invalid actual native delta')
        return dict(eigenvalues=(self.singular[:self.rank][::-1] ** 2).tolist(),
                    mode_energies=energies[::-1].tolist(), exact_energy=exact_energy,
                    allowed_energy=allowed_energy, gradient_inner_exact=inner_exact,
                    energy_reconstruction_gap=exact_energy + float(energies.sum()) - allowed_energy,
                    loss=float(loss), native_norm=float(np.linalg.norm(d)),
                    native_action=self.action_norm(d), group_ends=self.group_ends,
                    numerical_released=self.numerical_released,
                    exact_rank=self.rank, allowed_dimension=self.basis.shape[1],
                    rank_cutoff=self.diagnostic['rank_cutoff'],
                    numerical_cutoff=self.diagnostic['numerical_cutoff'])

    def diagnostics(self, ideal, actual, request_columns=None):
        ideal = np.asarray(array(ideal), dtype=np.float64)
        actual = np.asarray(array(actual), dtype=np.float64)
        rounding = actual - ideal
        values = dict(ideal=ideal, actual=actual, rounding=rounding)
        result = {}
        with threadpool_limits(limits=8):
            for name, value in values.items():
                result[name + '_norm'] = float(np.linalg.norm(value))
                if name == 'actual' and request_columns is not None:
                    # Reuse the same actual-response matrix multiplication for
                    # global and every request statistic. No extra model pass.
                    actual_column_energy = np.empty(self.keys.shape[1], dtype=np.float64)
                    response2 = 0.
                    for start, block in self.blocks():
                        action = value @ block
                        energy = np.sum(action*action, axis=0)
                        response2 += float(energy.sum())
                        actual_column_energy[start:start+len(energy)] = energy / self.weights[start:start+len(energy)]
                    result[name + '_response'] = math.sqrt(response2)
                else:
                    result[name + '_response'] = self.action_norm(value)
                leakage2 = 0.
                for start in range(0, value.shape[0], 128):
                    v = value[start:start + 128]
                    leakage = v - (v @ self.basis) @ self.basis.T
                    leakage2 += float(np.sum(leakage * leakage))
                result[name + '_P_leakage'] = math.sqrt(leakage2)
            if request_columns is not None:
                response, rows = [], []
                for request in request_columns:
                    if not isinstance(request, dict):
                        raise ValueError('per-case alias weights required; global column weights are insufficient')
                    ids = np.asarray(request['columns'], dtype=np.int64)
                    weights = np.asarray(request['weights'], dtype=np.float64)
                    if ids.ndim != 1 or ids.shape != weights.shape or not len(ids) or np.any(ids < 0) or np.any(ids >= len(self.weights)) or not np.isfinite(weights).all() or np.any(weights <= 0):
                        raise ValueError('invalid per-request response map')
                    mass = float(weights.sum())
                    rms = math.sqrt(float(np.dot(actual_column_energy[ids], weights)) / mass)
                    response.append(rms)
                    rows.append(dict(case_id=request['case_id'], conditional_RMS=rms,
                                     alias_weight_mass=mass, captured_columns=len(ids)))
                result['request_response'] = rows
                result['request_response_quantiles'] = dict(zip(('p50', 'p95', 'max'),
                    np.quantile(response, [.5, .95, 1.]).tolist()))
        result['finite'] = all(math.isfinite(x) for x in result.values() if isinstance(x, (int, float)))
        return result


def build_geometry(keys, weights, representative_indices, basis, *, block_columns=2048, threads=8):
    k, v = array(keys), np.asarray(array(basis), dtype=np.float64)
    w = np.asarray(array(weights), dtype=np.float64)
    reps = np.asarray(array(representative_indices), dtype=np.int64)
    if k.ndim != 2 or v.ndim != 2 or k.shape[0] != v.shape[0]:
        raise ValueError('key/basis shape mismatch')
    if w.shape != (k.shape[1],) or reps.shape != w.shape or not len(w):
        raise ValueError('weight/representative cardinality mismatch')
    if not np.isfinite(w).all() or np.any(w <= 0) or not math.isclose(float(w.sum()), 1., abs_tol=1e-13):
        raise ValueError('strict positive unit-sum weights required')
    if np.any(reps < 0) or np.any(reps > np.arange(len(reps))) or np.any(reps[reps] != reps):
        raise ValueError('representatives must be deterministic first columns')
    with threadpool_limits(limits=threads):
        v = legacy._basis(v)
        def reduced_blocks():
            for start in range(0, len(w), block_columns):
                stop = min(start + block_columns, len(w))
                kb = np.asarray(k[:, start:stop], dtype=np.float64)
                if not np.isfinite(kb).all():
                    raise ValueError('nonfinite keys')
                yield (v.T @ kb) * np.sqrt(w[start:stop])
        source = legacy.ColumnBlocks(v.shape[1], len(w), reduced_blocks, 'REQUEST_SEQUENCE_TOKEN_WEIGHTED')
        receipt, u = legacy._svd(source, left_vectors=True, threads=threads)
        singular = np.asarray(receipt['singular_values'], dtype=np.float64)
        duplicate2 = 0.
        for start in range(0, len(w), block_columns):
            stop = min(start + block_columns, len(w))
            delta = np.asarray(k[:, start:stop], dtype=np.float64) - k[:, reps[start:stop]].astype(np.float64)
            e = (v.T @ delta) * np.sqrt(w[start:stop])
            duplicate2 += float(np.sum(e * e))
    cutoff = receipt['threshold']
    rank = receipt['rank']
    diagnostic = dict(receipt, rank_cutoff=cutoff, numerical_cutoff=max(cutoff, math.sqrt(duplicate2)),
                      duplicate_difference_frobenius=math.sqrt(duplicate2),
                      representative_groups=int(len(np.unique(reps))), all_original_columns_retained=True,
                      weights_sum=float(w.sum()), columns=len(w), svd_calls=1,
                      equal_singular_group_policy='BITWISE_EQUAL_SINGULAR_VALUES_INDIVISIBLE',
                      rank_ambiguity_status=receipt['status'])
    return WeightedGeometry(k, w, reps, v, u[:, :rank], singular, diagnostic, block_columns)
