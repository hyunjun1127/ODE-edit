"""Small deterministic algebra checks; no model, dataset, or GPU execution."""
import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONTRACT = ROOT / 'plans/global/2026-09-18-fixed-key-reference-constrained-write-contract-v1.json'


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def flat(a):
    return [x for row in a for x in row]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def close(a, b, tol=1e-10):
    assert len(a) == len(b)
    assert max((abs(x - y) for x, y in zip(a, b)), default=0) <= tol


def solve(a, b):
    n = len(b)
    if n == 0:
        return []
    rows = [list(row) + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda k: abs(rows[k][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError('singular')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for row in range(n):
            if row != col:
                scale = rows[row][col]
                rows[row] = [x - scale * y for x, y in zip(rows[row], rows[col])]
    return [rows[i][-1] for i in range(n)]


def small_qp(h, d):
    """Enumerate dual active sets for tiny verification problems only."""
    n, dim = len(h), len(h[0])
    gram = [[dot(x, y) for y in h] for x in h]
    solutions = []
    for size in range(n + 1):
        for active in itertools.combinations(range(n), size):
            try:
                values = solve([[gram[i][j] for j in active] for i in active], [d[i] for i in active])
            except ValueError:
                continue
            if any(v < -1e-9 for v in values):
                continue
            alpha = [0.0] * n
            for i, value in zip(active, values):
                alpha[i] = value
            delta = [-sum(alpha[i] * h[i][j] for i in range(n)) for j in range(dim)]
            slack = [-d[i] - dot(h[i], delta) for i in range(n)]
            if min(slack) < -1e-9:
                continue
            assert max(abs(alpha[i] * slack[i]) for i in range(n)) < 1e-8
            primal = 0.5 * dot(delta, delta)
            dual = dot(alpha, d) - 0.5 * dot(alpha, [dot(row, alpha) for row in gram])
            assert abs(primal - dual) < 1e-8
            solutions.append((primal, delta, alpha))
    return min(solutions, key=lambda x: x[0]) if solutions else None


checks = []

# Single-row closed form fixes the sign of both the dual and the write.
one = small_qp([[2.0, 0.0]], [3.0])
close(one[1], [-1.5, 0.0])
close(one[2], [0.75])
checks.append({'name': 'single_constraint_closed_form_and_dual_sign', 'pass': True})

# Current response and a nonorthogonal reference can coexist in fixed-key space.
p = [[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
q = [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
k_edit = [[1.], [0.], [0.]]
k_ref = [[1.], [1.], [0.]]
native = [[1., 2., 0.], [3., 3., 0.]]
raw = [[[1., 1., 4.], [0., 0., 0.]], [[0., 0., 0.], [1., 1., -2.]]]
hs = [flat(matmul(g, q)) for g in raw]
damage = [dot(flat(g), flat(native)) for g in raw]
answer = small_qp(hs, damage)
d = [answer[1][:3], answer[1][3:]]
close(flat(matmul(d, q)), flat(d))
close(flat(matmul(d, p)), flat(d))
close(flat(matmul(d, k_edit)), [0., 0.])
close(flat(matmul(add(native, d), k_edit)), flat(matmul(native, k_edit)))
close(flat(matmul(add(native, d), k_ref)), [0., 0.])
checks.append({'name': 'fixed_key_current_exact_and_reference_total_update_constraint', 'pass': True, 'native_ref_action': flat(matmul(native, k_ref)), 'correction_ref_action': flat(matmul(d, k_ref))})

# Redundant rows and positive normalization preserve the solution.
redundant = small_qp(hs + [[2 * x for x in hs[0]]], damage + [2 * damage[0]])
close(redundant[1], answer[1])
scaled = small_qp([[v / 7 for v in hs[0]], [v * 3 for v in hs[1]]], [damage[0] / 7, damage[1] * 3])
close(scaled[1], answer[1])
checks.append({'name': 'redundancy_and_positive_row_scaling', 'pass': True})

# An exact current lock can make a harmed reference unreachable.
assert small_qp([[0., 0.]], [1.]) is None
# Farkas witness: nonnegative weights (1,1) cancel rows but weighted d is positive.
conflict_h = [[1., 0.], [-1., 0.]]
assert small_qp(conflict_h, [1., 1.]) is None
close([conflict_h[0][j] + conflict_h[1][j] for j in range(2)], [0., 0.])
checks.append({'name': 'zero_projected_gradient_and_conflicting_constraints_not_feasible', 'pass': True})

# Factor gradient and Gram are exact without storing full per-example weight gradients.
a1, k1 = [[1., 2.], [-1., 0.5]], [[1., 3.], [2., -1.], [0., 4.]]
a2, k2 = [[2.], [3.]], [[-1.], [2.], [5.]]
u1, u2 = matmul(q, k1), matmul(q, k2)
g1, g2 = matmul(a1, transpose(k1)), matmul(a2, transpose(k2))
h1, h2 = matmul(a1, transpose(u1)), matmul(a2, transpose(u2))
close(flat(h1), flat(matmul(g1, q)))
close(flat(h2), flat(matmul(g2, q)))
trace_product = matmul(matmul(transpose(a1), a2), matmul(transpose(u2), u1))
close([sum(trace_product[i][i] for i in range(len(trace_product)))], [dot(flat(h1), flat(h2))])
checks.append({'name': 'factor_gradient_and_Gram_identity', 'pass': True})

# At a rounded center outside Q, the RHS needs raw g, not projected h.
native_point = [0.5, 0.3]
actual_center_delta = [1e-5, -0.2]
center = [x + y for x, y in zip(native_point, actual_center_delta)]
value = (1 + sum(center)) ** 2
grad = [2 * (1 + sum(center))] * 2
projected = [0., grad[1]]
proposed = [0., -0.4]
bound = 1.0
rhs_d = value - bound - dot(grad, actual_center_delta)
linear_value = value + dot(grad, [x - y for x, y in zip(proposed, actual_center_delta)])
close([linear_value - bound], [rhs_d + dot(projected, proposed)])
wrong_rhs = value - bound - dot(projected, actual_center_delta)
assert abs(wrong_rhs - rhs_d) > 1e-6
checks.append({'name': 'physical_center_RHS_includes_rounding_component', 'pass': True})

# Two SQP rounds are a budget, not nonlinear convergence.
delta, values = 0., []
for _ in range(2):
    current = 2. + delta
    grad = 2. * current
    d_rhs = current * current - 1. - grad * delta
    delta = small_qp([[grad]], [d_rhs])[1][0]
    values.append((2. + delta) ** 2)
assert values[-1] > 1. + 1e-4
checks.append({'name': 'finite_nonlinear_counterexample_requires_fallback_after_two_rounds', 'pass': True, 'candidate_q_values': values, 'bound': 1.0})

# Changing the challenger without recomputing entry changes the question itself.
entry = {'answer': 1., 'old_challenger': 5., 'new_challenger': 2.}
native_scores = dict(entry)  # No actual model change at all.
old_bound = entry['answer'] - entry['old_challenger']
same_new_bound = entry['answer'] - entry['new_challenger']
observed = native_scores['answer'] - native_scores['new_challenger']
assert observed - old_bound == 3. and observed - same_new_bound == 0.
checks.append({'name': 'challenger_identity_must_match_entry_bound', 'pass': True})

contract = json.loads(CONTRACT.read_text())
assert contract['reference']['train_facts'] == 512
assert contract['solver']['gradient_row_evaluations_cap'] == contract['selection']['gradient_candidate_cap'] + contract['selection']['active_cap']
assert contract['execution']['sequential_new_batches_after_P1'] + 1 == contract['execution']['sequential_total_batches']
assert not contract['execution']['B2_plus_native_or_z_shared']
assert not contract['reference']['official_P_N_visible_to_controller']
assert not contract['reference']['factual_bank_ready']
assert not contract['execution']['dispatch_performed']
checks.append({'name': 'contract_budget_and_own_chain_consistency', 'pass': True})

receipt = {
    'status': 'PASS_CPU_ALGEBRA_ONLY',
    'checks': checks,
    'n_checks': len(checks),
    'model_loaded': False,
    'model_numerical_validation': 'NOT_RUN',
    'factual_bank_built': False,
    'GPU_experiment_submitted': False,
    'scope': 'Hand-sized deterministic linear algebra and counterexamples; does not establish large-model conditioning, runtime, or efficacy.'
}
(OUT / 'math-checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'checks': len(checks), 'nonlinear_counterexample_last_q': values[-1]}))
