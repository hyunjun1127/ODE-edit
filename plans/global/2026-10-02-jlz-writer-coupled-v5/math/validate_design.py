"""Small algebra/gradient checks for the fixed-basis v5 design, stdlib only.

These constructed fixtures do not load a model or establish GPU/FP32 parity,
locality improvements, or speed. Python floats approximate the real arithmetic
identities being checked; production materialized FP32 forward and optimized
backward require their own qualification.
"""
from pathlib import Path
import hashlib
import json
import math


def transpose(a):
    return [list(col) for col in zip(*a)]


def matmul(a, b):
    assert a and b and len(a[0]) == len(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b, scale=1.0):
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    return [[x + scale * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def outer(x, y):
    return [[a * b for b in y] for a in x]


def norm2(a):
    return sum(x * x for row in a for x in row)


def dot(a, b):
    return sum(x * y for ar, br in zip(a, b) for x, y in zip(ar, br))


def maxabs(a):
    return max(abs(x) for row in a for x in row)


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def matrix_close(a, b, tolerance=1e-10):
    error = maxabs(add(a, b, -1.0))
    assert error <= tolerance * max(1.0, maxabs(a), maxabs(b)), error
    return error


def solve(a, b):
    """Partial-pivot elimination for tiny fixtures, not a production solver."""
    n, nrhs = len(a), len(b[0])
    assert all(len(row) == n for row in a) and len(b) == n
    work = [list(ar) + list(br) for ar, br in zip(a, b)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(work[i][j]))
        assert abs(work[pivot][j]) > 1e-14
        work[j], work[pivot] = work[pivot], work[j]
        divisor = work[j][j]
        work[j] = [x / divisor for x in work[j]]
        for i in range(n):
            if i != j:
                factor = work[i][j]
                work[i] = [x - factor * y for x, y in zip(work[i], work[j])]
    return [row[n:n + nrhs] for row in work]


def cholesky(a):
    """Check the SPD assumption, including when the input keys are deficient."""
    n = len(a)
    result = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            value = a[i][j] - sum(result[i][k] * result[j][k] for k in range(j))
            if i == j:
                assert value > 0.0
                result[i][j] = math.sqrt(value)
            else:
                result[i][j] = value / result[j][j]
    matrix_close(matmul(result, transpose(result)), a)
    return result


def geometry(a, groups, weights):
    """Groups are request-major lists of context key vectors."""
    n_requests, width = len(groups), len(a)
    assert len(weights) == n_requests
    means, columns, rows, scatter = [], [], [], zeros(width, width)
    for r, (keys, alpha) in enumerate(zip(groups, weights)):
        assert len(keys) == len(alpha) and all(w >= 0 for w in alpha)
        close(sum(alpha), 1.0)
        mean = [sum(w * key[j] for key, w in zip(keys, alpha)) for j in range(width)]
        means.append(mean)
        for key, w in zip(keys, alpha):
            columns.append([math.sqrt(w) * x for x in key])
            row = [0.0] * n_requests
            row[r] = math.sqrt(w)
            rows.append(row)
            centered = [x - y for x, y in zip(key, mean)]
            scatter = add(scatter, outer(centered, centered), w)
    c, rmap, kbar = transpose(columns), rows, transpose(means)
    second = matmul(c, transpose(c))
    m = add(a, second)
    cholesky(a)
    cholesky(m)
    p = solve(m, kbar)
    t = solve(a, c)
    s = add(identity(len(rows)), matmul(transpose(c), t))
    cholesky(s)
    p_woodbury = matmul(t, solve(s, rmap))
    return dict(P=p, P_woodbury=p_woodbury, M=m, Kbar=kbar, C=c,
                R=rmap, scatter=scatter, second=second)


def ridge_check(a, groups, weights, d):
    g = geometry(a, groups, weights)
    p, kbar, m = g['P'], g['Kbar'], g['M']
    u = matmul(d, transpose(p))
    normal_error = matrix_close(matmul(u, m), matmul(d, transpose(kbar)))
    woodbury_error = matrix_close(p, g['P_woodbury'])
    mean_error = matrix_close(matmul(g['C'], g['R']), kbar)
    variance_error = matrix_close(g['second'], add(matmul(kbar, transpose(kbar)), g['scatter']))
    fit = 0.0
    for r, (keys, alpha) in enumerate(zip(groups, weights)):
        target = [[row[r]] for row in d]
        for key, w in zip(keys, alpha):
            fit += 0.5 * w * norm2(add(matmul(u, transpose([key])), target, -1.0))
    preservation = 0.5 * dot(matmul(u, a), u)
    e = add(identity(len(groups)), matmul(transpose(kbar), p), -1.0)
    reduced = 0.5 * dot(matmul(d, e), d)
    close(fit + preservation, reduced)
    return dict(passed=True,normal_equation_maxabs=normal_error,
                Woodbury_vs_primal_maxabs=woodbury_error,
                C_times_R_vs_Kbar_maxabs=mean_error,
                context_variance_identity_maxabs=variance_error,
                explicit_fit=fit, explicit_preservation=preservation,
                explicit_optimum=fit + preservation, reduced_optimum=reduced,
                context_columns=len(g['C'][0]), requests=len(groups), input_dimension=len(a))


def linear(x, w):
    return matmul(x, transpose(w))


def tanh(a):
    return [[math.tanh(x) for x in row] for row in a]


def materialize(w, d, p):
    return add(w, matmul(d, transpose(p)))


def target_loss(y, target):
    return 0.5 * norm2(add(y, target, -1.0))


def main():
    checks = {}
    a1 = [[1.5, .2], [.2, 1.1]]
    a2 = [[.9, .1], [.1, 1.4]]
    groups = [[[1.0, .2], [.7, -.1]], [[.4, 1.1], [.1, .9]]]
    weights = [[.5, .5], [.5, .5]]
    d1 = [[.25, -.1], [.15, .2]]
    d2 = [[.12, .3], [-.18, .08]]
    checks['full_context_ridge'] = ridge_check(a1, groups, weights, d1)

    # Collinear context keys AND duplicate requests: no inverse of K or P.
    deficient = [[[scale, 2 * scale] for scale in [1., 2., .5, 1.5, .8, 1.2]]] * 2
    native_weights = [[.5, .1, .1, .1, .1, .1]] * 2
    checks['rank_deficient_keys_spd_A'] = ridge_check(a1, deficient, native_weights, d1)

    # A two-layer nonlinear all-token network. Each D column has one request
    # owner, but the physical weight acts on EVERY row, including the extra row.
    x = [list(key) for group in groups for key in group] + [[-.3, .8]]
    w1, w2 = [[.8, -.2], [.3, .7]], [[1., .4], [-.1, .6]]
    p1 = geometry(a1, groups, weights)['P']
    entry_h2 = tanh(linear(x, w1))
    entry_groups2 = [entry_h2[:2], entry_h2[2:4]]
    p2 = geometry(a2, entry_groups2, weights)['P']
    target = [[.1, -.2], [.3, .4], [-.2, .5], [.4, -.1], [.2, .3]]

    def forward(dl1, dl2, second_basis=p2, factored=False):
        if factored:
            h1 = add(linear(x, w1), matmul(matmul(x, p1), transpose(dl1)))
            h2 = tanh(h1)
            y = add(linear(h2, w2), matmul(matmul(h2, second_basis), transpose(dl2)))
        else:
            h1 = linear(x, materialize(w1, dl1, p1))
            h2 = tanh(h1)
            y = linear(h2, materialize(w2, dl2, second_basis))
        return y, h2

    y, h2 = forward(d1, d2)
    factored, _ = forward(d1, d2, factored=True)
    forward_error = matrix_close(y, factored)
    g2 = add(y, target, -1.0)
    analytic2 = matmul(transpose(g2), matmul(h2, p2))
    gx2 = matmul(g2, materialize(w2, d2, p2))
    gh1 = [[g * (1 - h * h) for g, h in zip(gr, hr)] for gr, hr in zip(gx2, h2)]
    analytic1 = matmul(transpose(gh1), matmul(x, p1))
    finite_difference_errors = []
    finite_difference_gradients = []
    step = 1e-6
    for layer, (d, analytic) in enumerate([(d1, analytic1), (d2, analytic2)]):
        measured = zeros(2, 2)
        for i in range(2):
            for j in range(2):
                plus, minus = [row[:] for row in d], [row[:] for row in d]
                plus[i][j] += step
                minus[i][j] -= step
                yp = forward(plus, d2)[0] if layer == 0 else forward(d1, plus)[0]
                ym = forward(minus, d2)[0] if layer == 0 else forward(d1, minus)[0]
                measured[i][j] = (target_loss(yp, target) - target_loss(ym, target)) / (2 * step)
        finite_difference_errors.append(matrix_close(analytic, measured, 2e-8))
        finite_difference_gradients.append(measured)
    # Omitting the upper write from the input VJP loses a real upstream path.
    gx2_wrong = matmul(g2, w2)
    gh1_wrong = [[g * (1 - h * h) for g, h in zip(gr, hr)] for gr, hr in zip(gx2_wrong, h2)]
    wrong1 = matmul(transpose(gh1_wrong), matmul(x, p1))
    omitted_upstream_error = maxabs(add(wrong1, analytic1, -1.0))
    assert omitted_upstream_error > 1e-3
    checks['fixed_basis_all_token_gradient'] = dict(
        passed=True, materialized_vs_factored_real_arithmetic_maxabs=forward_error,
        all_coordinates_finite_difference_maxabs=finite_difference_errors,
        analytic_gradients=[analytic1, analytic2], finite_difference_gradients=finite_difference_gradients,
        omitted_upper_write_input_VJP_error=omitted_upstream_error,
        caveat='Python-float algebra only; not production FP32 GEMM/materialization/backward parity')

    # The loss on request 0 depends on request 1's D column after writing.
    # This coupling is absent from request-indexed subject-only injections.
    g_first = zeros(len(x), 2)
    g_first[0] = list(g2[0])
    cross_gradient = matmul(transpose(g_first), matmul(h2, p2))
    other_request_gradient = math.sqrt(sum(row[1] ** 2 for row in cross_gradient))
    assert other_request_gradient > 1e-4
    plus, minus = [row[:] for row in d2], [row[:] for row in d2]
    plus[0][1] += step
    minus[0][1] -= step
    fd_cross = (target_loss([forward(d1, plus)[0][0]], [target[0]]) -
                target_loss([forward(d1, minus)[0][0]], [target[0]])) / (2 * step)
    close(cross_gradient[0][1], fd_cross, 2e-8)
    # An extra non-owner row is changed too: there is no subject mask.
    unmodified_y = linear(tanh(linear(x, w1)), w2)
    extra_row_change = maxabs(add([y[-1]], [unmodified_y[-1]], -1.0))
    assert extra_row_change > 1e-4
    checks['cross_request_and_all_token_action'] = dict(
        passed=True, request_0_loss_gradient_to_request_1_D_norm=other_request_gradient,
        selected_cross_gradient=cross_gradient[0][1], selected_finite_difference=fd_cross,
        extra_row_output_change=extra_row_change)

    zero = zeros(2, 2)
    y0, _ = forward(zero, zero)
    zero_error = matrix_close(y0, unmodified_y)
    zero_gradient2 = matmul(transpose(add(y0, target, -1.0)), matmul(entry_h2, p2))
    assert norm2(zero_gradient2) > 0.0
    checks['zero_payload_entry_identity'] = dict(
        passed=True, output_maxabs=zero_error, task_gradient_at_zero_norm=math.sqrt(norm2(zero_gradient2)),
        note='Zero payload retains its differentiable physical parameterization')

    # Commit the same fixed-basis weights. Re-solving the upper basis with
    # current keys defines a different model, even though D was not changed.
    committed_w1, committed_w2 = materialize(w1, d1, p1), materialize(w2, d2, p2)
    committed = linear(tanh(linear(x, committed_w1)), committed_w2)
    commit_error = matrix_close(y, committed)
    current_p2 = geometry(a2, [h2[:2], h2[2:4]], weights)['P']
    changed_p2_error = maxabs(add(current_p2, p2, -1.0))
    resolved_y, _ = forward(d1, d2, second_basis=current_p2)
    resolve_output_change = maxabs(add(resolved_y, y, -1.0))
    assert changed_p2_error > 1e-4 and resolve_output_change > 1e-4
    checks['fixed_basis_fit_commit_identity'] = dict(
        passed=True, same_basis_fit_commit_output_maxabs=commit_error,
        current_key_resolve_basis_change=changed_p2_error,
        current_key_resolve_output_change=resolve_output_change,
        rule='Commit accepted materialized weights with the fitted P; do not re-solve current keys')

    # Causal toy: tokens before the subject are changed by an all-token write.
    # A cache BEFORE the first write is still constant; an upper prefix is not.
    causal = [[1., 0., 0.], [.4, .6, 0.], [.2, .3, .5]]
    prefix_input = [row[:] for row in x[:3]]
    old_prefix = matmul(causal, tanh(linear(prefix_input, w1)))
    new_prefix = matmul(causal, tanh(linear(prefix_input, committed_w1)))
    before_subject_change = maxabs(add([new_prefix[0]], [old_prefix[0]], -1.0))
    assert before_subject_change > 1e-4
    future_changed = [row[:] for row in prefix_input]
    future_changed[2] = [5., -7.]
    future_output = matmul(causal, tanh(linear(future_changed, committed_w1)))
    causal_prefix_error = matrix_close(future_output[:2], new_prefix[:2])
    checks['causal_cache_boundary'] = dict(
        passed=True, prefix_input_equal_for_all_D=True,
        upper_pre_subject_prefix_change=before_subject_change,
        future_token_change_effect_on_earlier_outputs=causal_prefix_error,
        scope='Constructed causal token mixer; not a Transformer KV-cache implementation test')

    result = dict(
        status='PASS_STDLIB_V5_ALGEBRA_NOT_MODEL_OR_GPU_VALIDATION',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        check_count=len(checks), GPU_calls=0, model_forwards=0,
        production_changes=0, claims_not_validated=[
            'FP32 materialized versus low-rank forward numerical parity',
            'Optimized custom backward versus production dense autograd parity',
            'Transformer architecture adapters and cache implementation',
            'Locality, edit success, runtime, or memory improvement'],
        checks=checks)
    Path(__file__).with_name('validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
