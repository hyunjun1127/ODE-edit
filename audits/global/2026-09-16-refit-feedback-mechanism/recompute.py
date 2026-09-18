"""Recompute stored REFIT evidence and small algebraic checks; no model/GPU."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIX = ROOT / 'local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1'
REFRESH = ROOT / 'local/reviews/bg-tw-method-review-2026-09-15/source/experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1'
LIFE = ROOT / 'local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b, scale=1.0):
    return [[x + scale * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, s):
    return [[s * x for x in row] for row in a]


def maxdiff(a, b):
    return max(abs(x - y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def main():
    source_paths = [SIX / 'z-iterations.csv', REFRESH / 'raw-writer-history-links.csv',
                    REFRESH / 'final-populations.csv', LIFE / 'final-summary.csv',
                    LIFE / 'AlphaEdit-layer-summary.csv']
    rows = read(source_paths[0])
    second = [r for r in rows if r['arm'] == 'REFIT4' and r['fit_stage'] == 'second_fit']
    hist = Counter(int(r['adam_updates']) for r in second)
    assert hist == {0: 853, 24: 146, 21: 1}
    first = [r for r in rows if r['arm'] == 'REFIT4' and r['fit_stage'] == 'first_fit']
    assert len(first) == 1000 and all(int(r['adam_updates']) == 24 for r in first)
    history = read(source_paths[1])
    by_batch = {}
    for batch in sorted({int(r['batch']) for r in history}):
        selected = [r for r in history if int(r['batch']) == batch]
        assert len(selected) == 6 and len({r['policy'] for r in selected}) == 6
        by_batch[batch] = {key: len({r[key] for r in selected})
                           for key in ('raw_entry_M', 'raw_endpoint_M')}
        assert all(v == 1 for v in by_batch[batch].values())
    assert len(by_batch) == 10

    expected = {'N4': (1938, 1423, 7108), 'REFIT4': (1950, 1405, 7175),
                'FROZEN2': (1958, 1467, 7070), 'I2': (1961, 1456, 7086),
                'FROZEN4': (1969, 1487, 7056), 'I4': (1966, 1489, 7056)}
    suffix = [r for r in read(source_paths[2])
              if r['population'] == 'suffix' and r['group'] == 'ALL']
    suffix_table = {}
    for policy, wanted in expected.items():
        p = [r for r in suffix if r['policy'] == policy and r['metric'] == 'PS']
        n = [r for r in suffix if r['policy'] == policy and r['metric'] == 'NS']
        assert len(p) == len(n) == 1
        p, n = p[0], n[0]
        actual = (int(p['numerator']), int(p['new_strict_numerator']), int(n['numerator']))
        assert actual == wanted
        assert int(p['prompt_denominator']) == 2000 and int(n['prompt_denominator']) == 10000
        suffix_table[policy] = {'P_success': actual[0], 'P_strict': actual[1],
                                'N_success': actual[2], 'P_den': 2000, 'N_den': 10000}

    # Fixed input, single affine weight: physical sequential and accumulated
    # writes have the same real-arithmetic endpoint and all-token response.
    a = [[0.3, -0.2, 0.5], [0.1, 0.4, -0.1]]
    kc = [[0.8, 0.1], [-0.3, 0.6], [0.2, -0.1]]
    r0 = [[1.2, -0.4], [0.7, 0.9]]
    c = mm(a, kc)
    first_delta = scale(mm(r0, a), 0.75)
    second_resid = add(r0, mm(first_delta, kc), -1)
    repeated = add(first_delta, mm(second_resid, a))
    x_end = add(scale(r0, 1.75), scale(mm(r0, c), 0.75), -1)
    compressed = mm(x_end, a)
    error = maxdiff(repeated, compressed)
    assert error < 1e-14
    native = mm(r0, a)
    formula_difference = scale(mm(add(r0, mm(r0, c), -1), a), 0.75)
    difference_error = maxdiff(add(repeated, native, -1), formula_difference)
    assert difference_error < 1e-14
    # Same-geometry split of successful/deficient request RHS is linear.
    r_pass = [[r[0], 0.] for r in second_resid]
    r_fail = [[0., r[1]] for r in second_resid]
    split_error = maxdiff(mm(second_resid, a), add(mm(r_pass, a), mm(r_fail, a)))
    assert split_error < 1e-14
    # Scalar ridge writer minimizes (r-d)^2 + d^2 at d=r/2. Repeated
    # residual fitting changes the endpoint, and need not improve that objective.
    r, ridge_native = 1., .5
    frozen2 = .75 * ridge_native + .5 * (r - .75 * ridge_native)
    assert frozen2 == .6875
    native_objective = (r-ridge_native)**2 + ridge_native**2
    frozen_objective = (r-frozen2)**2 + frozen2**2
    assert frozen_objective > native_objective
    # CBF linearization alone does not certify a finite Euler step.
    # h(x)=1-||x||^2, x=(1,0), v=(0,1) has grad(h).v=0,
    # yet h(x+eta*v)<0 for every eta>0.
    eta = .1
    euler_h = 1 - (1 + eta**2)
    assert euler_h < 0

    result = {
        'scope': 'Stored CSV arithmetic and CPU algebra only; no tensor replay or model evaluation',
        'second_Adam_histogram': dict(sorted(hist.items())),
        'second_Adam_total': sum(int(r['adam_updates']) for r in second),
        'second_positive_request_count': sum(int(r['adam_updates']) > 0 for r in second),
        'second_positive_request_mean_Adam': 3525 / 147,
        'history_hash_unique_counts_across_six_policies': by_batch,
        'history_hash_limitation': 'Published hashes agree. This is not an independent raw tensor or K equality check.',
        'suffix_final_metrics': suffix_table,
        'algebra': {
            'repeat_vs_accumulated_endpoint_maxabs': error,
            'frozen2_minus_native_formula_maxabs': difference_error,
            'same_geometry_RHS_split_maxabs': split_error,
            'scalar_native': ridge_native,
            'scalar_frozen2': frozen2,
            'scalar_native_objective': native_objective,
            'scalar_frozen2_objective': frozen_objective,
            'tangent_Euler_barrier_value': euler_h,
        },
        'source_files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in source_paths],
    }
    out = Path(__file__).with_name('checks.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
