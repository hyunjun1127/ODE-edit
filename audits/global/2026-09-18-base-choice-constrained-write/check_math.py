"""Deterministic CPU algebra/counterexample audit, not a model experiment."""
import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def close(a, b, tol=1e-9):
    assert len(a) == len(b)
    assert max((abs(x - y) for x, y in zip(a, b)), default=0) <= tol


def solve(a, b):
    n = len(b)
    rows = [list(row) + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(rows[i][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError('singular')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [x / scale for x in rows[col]]
        for i in range(n):
            if i != col:
                scale = rows[i][col]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[col])]
    return [rows[i][-1] for i in range(n)]


def qp(h, b):
    """Tiny exact-active-set enumeration for <h_i,D> >= b_i."""
    n, dim = len(h), len(h[0])
    gram = [[dot(x, y) for y in h] for x in h]
    choices = []
    for size in range(n + 1):
        for active in itertools.combinations(range(n), size):
            try:
                values = solve([[gram[i][j] for j in active] for i in active], [b[i] for i in active])
            except ValueError:
                continue
            if any(v < -1e-9 for v in values):
                continue
            alpha = [0.] * n
            for i, value in zip(active, values):
                alpha[i] = value
            delta = [sum(alpha[i] * h[i][j] for i in range(n)) for j in range(dim)]
            slack = [dot(h[i], delta) - b[i] for i in range(n)]
            if min(slack) < -1e-9:
                continue
            primal = 0.5 * dot(delta, delta)
            dual = dot(alpha, b) - 0.5 * dot(alpha, [dot(row, alpha) for row in gram])
            assert abs(primal - dual) < 1e-8
            assert max(abs(alpha[i] * slack[i]) for i in range(n)) < 1e-8
            choices.append({'D': delta, 'alpha': alpha, 'objective': primal})
    return min(choices, key=lambda x: x['objective']) if choices else None


def greedy_id(logits):
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


checks = []

# Choice, probability, and original margin preservation are different conditions.
base = [.60, .20, .20]
states = {'native': [.35, .40, .25], 'A': [.55, .30, .15], 'B': [.60, .25, .15]}
table = {}
for name, probabilities in states.items():
    margin = math.log(probabilities[0] / max(probabilities[1:]))
    attenuation = math.log(base[0] / probabilities[0])
    table[name] = {'margin': margin, 'attenuation': attenuation, 'choice_safe': greedy_id(probabilities) == 0}
assert not table['native']['choice_safe'] and table['A']['choice_safe'] and table['B']['choice_safe']
assert table['A']['attenuation'] > 0 and table['B']['attenuation'] == 0
assert table['A']['margin'] < math.log(3) and table['B']['margin'] < math.log(3)
checks.append({'name': 'choice_probability_original_margin_distinct', 'pass': True, 'values': table})

# Softmax cancels from a same-prefix pairwise margin.
logits = [2.7, -1.3, .9, 2.1]
log_z = math.log(sum(math.exp(x) for x in logits))
logp = [x - log_z for x in logits]
close([logp[0] - max(logp[1:])], [logits[0] - max(logits[1:])])
checks.append({'name': 'logit_gap_equals_log_probability_gap', 'pass': True})

# A previously unimportant competitor can become the actual winner.
base_logits, candidate = [4., 3., -5.], [2., 1., 3.]
assert greedy_id(base_logits) == 0
assert candidate[0] - candidate[1] > 0
assert candidate[0] - max(candidate[1:]) < 0
checks.append({'name': 'frozen_base_competitor_can_false_pass', 'pass': True})

# Margin >= 0 alone does not certify the specified token under ties.
assert greedy_id([1., 1.]) == 0
target_id = 1
assert [1., 1.][target_id] - [1., 1.][0] == 0
assert greedy_id([1., 1.]) != target_id
checks.append({'name': 'tie_requires_exact_token_ID_guard', 'pass': True})

# Correct dual sign and safe-reference zero-pressure solution.
one = qp([[2., 0.]], [.134])
close(one['D'], [.067, 0.])
close(one['alpha'], [.0335])
close(qp([[2., 0.]], [-.3])['D'], [0., 0.])
checks.append({'name': 'positive_margin_dual_reconstruction_and_zero_safe_update', 'pass': True})

# Omitting an initially safe row can damage it. Global local-row scan restores it.
h, b = [[1., 0.], [-1., 1.]], [1., -.25]
first = qp([h[0]], [b[0]])
assert dot(h[1], [0., 0.]) >= b[1]
assert dot(h[1], first['D']) < b[1]
full = qp(h, b)
close(full['D'], [1., .75])
checks.append({'name': 'initially_safe_reference_must_remain_in_full_problem', 'pass': True})

# GSS ordering may change the subset order, not the final full local optimum.
def working_qp(rows, bounds):
    normalized_b = [v / norm(row) for row, v in zip(rows, bounds)]
    working = [max(range(len(rows)), key=lambda i: normalized_b[i])]
    order = list(working)
    while True:
        result = qp([rows[i] for i in working], [bounds[i] for i in working])
        assert result is not None
        missing = [i for i in range(len(rows)) if i not in working and dot(rows[i], result['D']) < bounds[i] - 1e-9]
        if not missing:
            return result, order
        # Worst violation is mandatory; diversity is the tie rule in this tiny audit.
        def priority(i):
            violation = (bounds[i] - dot(rows[i], result['D'])) / norm(rows[i])
            distance = min(1 - dot(rows[i], rows[j]) / (norm(rows[i]) * norm(rows[j])) for j in working)
            return violation, distance, -i
        new = max(missing, key=priority)
        working.append(new)
        order.append(new)

result, order = working_qp(h, b)
close(result['D'], full['D'])
close([result['objective']], [full['objective']])
checks.append({'name': 'working_set_with_full_scan_matches_full_QP', 'pass': True, 'row_order': order})

# Direction alone cannot justify dropping a stricter offset.
same_direction = qp([[1., 0.], [1., 0.]], [1., 2.])
close(same_direction['D'], [2., 0.])
assert qp([[1., 0.], [-1., 0.]], [1., 1.]) is None
assert qp([[0., 0.]], [1.]) is None
checks.append({'name': 'offset_strength_conflict_and_zero_gradient', 'pass': True})

# Updating only the latest worst branch can oscillate despite a feasible solution.
# A single reference has two gaps: -.1+x and 1-20x+y.
old_h, old_b = [1., 0.], .1
new_h, new_b = [-20., 1.], -1.
first = qp([old_h], [old_b])
assert 1 + dot(new_h, first['D']) < 0
replacement = qp([new_h], [new_b])
close(replacement['D'], [0., 0.])
assert dot(old_h, replacement['D']) - old_b < 0
retained = qp([old_h, new_h], [old_b, new_b])
close(retained['D'], [.1, 1.])
checks.append({'name': 'retain_exposed_pairs_to_avoid_competitor_switch_oscillation', 'pass': True, 'retained_D': retained['D']})

# Raw g is required in the actual-center RHS if rounding leaves a tiny off-space component.
raw_g, projected_g = [2., 3.], [0., 3.]
actual_center, proposed = [1e-5, -.2], [0., .3]
mu_center = -.4
rhs = -mu_center + dot(raw_g, actual_center)
linear_margin = mu_center + dot(raw_g, [x - y for x, y in zip(proposed, actual_center)])
close([dot(projected_g, proposed) - rhs], [linear_margin])
assert abs(rhs - (-mu_center + dot(projected_g, actual_center))) > 1e-6
checks.append({'name': 'actual_center_rounding_term_in_positive_margin_RHS', 'pass': True})

# EOS and censored-prefix claims differ even in a deterministic toy generator.
base_map = {(): [3., 1., 0.], (0,): [0., 3., 1.], (0, 1): [0., 1., 3.]}
same_map = {(): [1., .5, 0.], (0,): [0., .5, .4], (0, 1): [0., .3, .4]}
def generate(mapping, cap):
    tokens = []
    for _ in range(cap):
        token = greedy_id(mapping[tuple(tokens)])
        tokens.append(token)
        if token == 2:
            break
    return tokens
assert generate(base_map, 3) == generate(same_map, 3) == [0, 1, 2]
censored_map = dict(same_map)
censored_map[(0, 1)] = [3., 2., 1.]
assert generate(base_map, 2) == generate(censored_map, 2)
assert generate(base_map, 3) != generate(censored_map, 3)
checks.append({'name': 'greedy_prefix_induction_EOS_and_censor_scope', 'pass': True})

contract_path = ROOT / 'plans/global/2026-09-18-base-choice-constrained-write-contract-v2.json'
contract = json.loads(contract_path.read_text())
assert contract['primary']['reference_constraints_use_all_512']
assert not contract['GSS']['reference_bank_subsampling'] and not contract['GSS']['gradient_subsampling']
assert contract['base_capsule']['max_protected_positions'] == 512 * 16
solver = contract['local_solver']
assert solver['pair_scalar_gradient_evaluations_cap'] == solver['pair_rows_round1_cap'] + solver['pair_rows_round2_cap'] == 1536
assert contract['constraints']['exposed_pairs_retained_within_batch']
assert not contract['primary']['probability_floor_constraint']
assert not contract['experiment']['B2_plus_native_shared']
assert not contract['experiment']['submitted']
old = json.loads((ROOT / 'plans/global/2026-09-18-fixed-key-reference-constrained-write-contract-v1.json').read_text())
assert old['status'] == 'SUPERSEDED_NOT_RUNNABLE'
checks.append({'name': 'all_reference_contract_caps_and_supersession', 'pass': True})

receipt = {
    'status': 'PASS_CPU_ALGEBRA_ONLY',
    'checks': checks,
    'n_checks': len(checks),
    'model_loaded': False,
    'base_capsules_built': False,
    'GPU_runs': 0,
    'model_numerical_validation': 'NOT_RUN',
    'scope': 'Small deterministic algebra and counterexamples, not numerical validation or efficacy of the LLM method.'
}
(OUT / 'math-checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'checks': len(checks)}))
