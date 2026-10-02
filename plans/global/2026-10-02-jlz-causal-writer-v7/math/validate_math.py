#!/usr/bin/env python3
"""CPU float64 algebra/gradient tests for the proposed causal JLZ writer.

Only Python's standard library is needed. The toy task is a smooth surrogate,
not a language model, native NLL, GPU qualification, or a performance result.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import struct
import sys


def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def tr(a):
    return [list(x) for x in zip(*a)]


def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in tr(b)] for row in a]


def add(a, b, factor=1.0):
    return [[x + factor*y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, s):
    return [[s*x for x in row] for row in a]


def diagonal(values):
    return [[x if i == j else 0.0 for j in range(len(values))]
            for i, x in enumerate(values)]


def columns(a, indices):
    return [[row[i] for i in indices] for row in a]


def maxabs(a):
    return max((abs(x) for row in a for x in row), default=0.0)


def inner(a, b):
    return sum(x*y for ar, br in zip(a, b) for x, y in zip(ar, br))


def solve(a, b):
    """SPD Cholesky solve; no inverse and no undisclosed jitter."""
    n, width = len(a), len(b[0])
    lower, y, x = zeros(n, n), zeros(n, width), zeros(n, width)
    for i in range(n):
        for j in range(i+1):
            value = a[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if value <= 0:
                    raise ValueError('The test requires an SPD matrix')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value/lower[j][j]
    for col in range(width):
        for i in range(n):
            y[i][col] = (b[i][col]-sum(lower[i][k]*y[k][col] for k in range(i)))/lower[i][i]
        for i in reversed(range(n)):
            x[i][col] = (y[i][col]-sum(lower[k][i]*x[k][col] for k in range(i+1, n)))/lower[i][i]
    return x


class Node:
    """Small reverse-mode matrix AD; the solve pullback is implicit."""
    def __init__(self, value, parents=()):
        self.value = value
        self.parents = parents
        self.grad = zeros(len(value), len(value[0]))

    def __add__(self, other):
        other = node(other)
        return Node(add(self.value, other.value), ((self, lambda g: g), (other, lambda g: g)))

    def __sub__(self, other):
        return self + node(other).times(-1)

    def __matmul__(self, other):
        other = node(other)
        return Node(mm(self.value, other.value),
                    ((self, lambda g: mm(g, tr(other.value))),
                     (other, lambda g: mm(tr(self.value), g))))

    def times(self, value):
        return Node(scale(self.value, value), ((self, lambda g: scale(g, value)),))

    @property
    def T(self):
        return Node(tr(self.value), ((self, tr),))

    def cols(self, indices):
        indices = list(indices)
        def back(g):
            out = zeros(len(self.value), len(self.value[0]))
            for i, row in enumerate(g):
                for local, original in enumerate(indices):
                    out[i][original] += row[local]
            return out
        return Node(columns(self.value, indices), ((self, back),))

    def tanh(self):
        result = [[math.tanh(x) for x in row] for row in self.value]
        return Node(result, ((self, lambda g: [[a*(1-b*b) for a, b in zip(ar, br)]
                                               for ar, br in zip(g, result)]),))

    def trace(self):
        n = len(self.value)
        return Node([[sum(self.value[i][i] for i in range(n))]],
                    ((self, lambda g: scale(eye(n), g[0][0])),))

    def root(self):
        q = self.value[0][0]
        # This suite uses well-conditioned PSD examples. Unexpected negatives
        # fail instead of silently introducing a production repair rule.
        if q < -1e-13:
            raise ValueError('Negative squared norm outside toy roundoff tolerance')
        out = math.sqrt(max(q, 0.0))
        return Node([[out]], ((self, lambda g: [[g[0][0]/(2*out) if out else 0.0]]),))

    def backward(self):
        order, seen = [], set()
        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p, _ in n.parents:
                visit(p)
            order.append(n)
        visit(self)
        for n in order:
            n.grad = zeros(len(n.value), len(n.value[0]))
        self.grad = [[1.0]]
        for n in reversed(order):
            for parent, pullback in n.parents:
                parent.grad = add(parent.grad, pullback(n.grad))


def node(value):
    return value if isinstance(value, Node) else Node(value)


def cat(nodes):
    widths = [len(n.value[0]) for n in nodes]
    out = [[] for _ in nodes[0].value]
    parents, start = [], 0
    for n, width in zip(nodes, widths):
        for i, row in enumerate(n.value):
            out[i].extend(row)
        indices = list(range(start, start+width))
        parents.append((n, lambda g, indices=indices: columns(g, indices)))
        start += width
    return Node(out, tuple(parents))


def spd_solve(a, b):
    a, b = node(a), node(b)
    x = solve(a.value, b.value)
    # The matrix presented to this primitive is symmetric positive definite.
    # Its unconstrained matrix pullback is contracted through its construction.
    return Node(x, ((a, lambda g: scale(mm(solve(a.value, g), tr(x)), -1)),
                    (b, lambda g: solve(a.value, g))))


def writer(k, a, omega, z, form='dense'):
    if form == 'dense':
        return spd_solve(node(a) + k @ omega @ k.T, k @ omega @ tr(z))
    root_weights = diagonal([math.sqrt(omega[i][i]) for i in range(len(omega))])
    f = k @ root_weights
    aif = spd_solve(a, f)
    return aif @ spd_solve(node(eye(len(omega))) + f.T @ aif,
                           node(root_weights) @ tr(z))


def rand(rng, n, m, magnitude=1.0):
    return [[rng.uniform(-magnitude, magnitude) for _ in range(m)] for _ in range(n)]


def fixture(batch):
    rng = random.Random(9187+batch)
    counts = [1+(2*r)%3 for r in range(batch)]
    owners = [r for r, count in enumerate(counts) for _ in range(count)]
    contexts, dim, layers = len(owners), 3, 3
    z = [[float(owner == r) for owner in owners] for r in range(batch)]
    weights = []
    for count in counts:
        weights.extend([1.0] if count == 1 else ([.5, .5] if count == 2 else [.5, .5, 0.0]))
    injection = zeros(batch, 2*contexts)
    for c, owner in enumerate(owners):
        injection[owner][2*c+1] = 1.0
    task_weights = [1/(batch*counts[r]) for r in owners]
    selected = [r for r in range(batch) if r % 2 == 0]
    actual_weights = [1/(len(selected)*counts[r]) if r in selected else 0.0 for r in owners]
    a = []
    for _ in range(layers):
        raw = rand(rng, dim, dim, .3)
        a.append(add(mm(raw, tr(raw)), scale(eye(dim), .55)))
    return {'batch': batch, 'dim': dim, 'contexts': contexts, 'owners': owners,
            'counts': counts, 'z': z, 'omega': diagonal(weights), 'injection': injection,
            'task_weights': task_weights, 'actual_weights': actual_weights,
            'x': rand(rng, dim, 2*contexts, .7), 'target': rand(rng, dim, contexts, .8),
            'a': a, 'b': [add(eye(dim), rand(rng, dim, dim, .2)) for _ in range(layers)],
            'w': [rand(rng, dim, dim, .25) for _ in range(layers)],
            'd': [rand(rng, dim, batch, .27) for _ in range(layers)],
            'sigma': [1.1, .9, 1.3]}


def keys_and_all_tokens(x, b, microbatch):
    """Two causal token positions per context, then one logical key barrier."""
    contexts = len(x.value[0])//2
    parts = []
    for start in range(0, contexts, microbatch):
        count = min(microbatch, contexts-start)
        chunk = x.cols(range(2*start, 2*(start+count)))
        causal = zeros(2*count, 2*count)
        for c in range(count):
            causal[2*c][2*c+1] = .17
        parts.append((node(b) @ chunk + chunk @ causal).tanh())
    all_tokens = cat(parts)
    return all_tokens.cols(range(1, 2*contexts, 2)), all_tokens


def weighted_error(prediction, target, weights):
    error = prediction - target
    return (error @ diagonal(weights) @ error.T).trace().times(.5)


def run_candidate(f, arm='A', form='dense', microbatch=None, detach_p=False,
                  cached_first_p=None, fixed_weights=None):
    ds = [Node([row[:] for row in d]) for d in f['d']]
    mb = microbatch or f['contexts']
    x_native = node(f['x'])
    for l, d in enumerate(ds):
        _, h = keys_and_all_tokens(x_native, f['b'][l], mb)
        x_native = x_native + node(f['w'][l]) @ h + d @ f['injection']
    native_subject = x_native.cols(range(1, 2*f['contexts'], 2))
    primary = weighted_error(native_subject, f['target'], f['task_weights'])
    x_actual = node(f['x'])
    captures, gq, eq = [], [], []
    for l, d in enumerate(ds):
        k, h = keys_and_all_tokens(x_actual, f['b'][l], mb)
        p = writer(k, f['a'][l], f['omega'], f['z'], form)
        if l == 0 and cached_first_p is not None:
            p = node(cached_first_p)
        if detach_p:
            p = node(p.value)
        u = d @ p.T
        if fixed_weights is not None:
            # Direct committed-weight replay uses no writer output in forward.
            effective = node(fixed_weights[l])
        else:
            effective = node(f['w'][l]) + u
        g = p.T @ f['a'][l] @ p
        r = p.T @ k - f['z']
        e = r @ f['omega'] @ r.T
        factor = 1/(f['batch']*f['sigma'][l]**2)
        gq.append((d @ g @ d.T).trace().times(factor))
        eq.append((d @ e @ d.T).trace().times(factor))
        x_actual = x_actual + effective @ h
        captures.append({'K': k, 'P': p, 'U': u, 'G': g, 'E': e,
                         'effective_weight': effective, 'x': x_actual})
    if arm == 'A':
        policy_g = sum_nodes([q.root() for q in gq])
        policy_e = sum_nodes([q.root() for q in eq])
    else:
        policy_g = sum_nodes(gq).root()
        policy_e = sum_nodes(eq).root()
    actual_subject = x_actual.cols(range(1, 2*f['contexts'], 2))
    auxiliary = weighted_error(actual_subject, f['target'], f['actual_weights'])
    loss = primary + policy_g.times(.07) + policy_e.times(.09) + auxiliary.times(.13)
    return {'loss': loss, 'ds': ds, 'captures': captures,
            'native': x_native, 'actual': x_actual,
            'primary': primary, 'auxiliary': auxiliary,
            'policy_g': policy_g, 'policy_e': policy_e}


def sum_nodes(nodes):
    out = node([[0.0]])
    for item in nodes:
        out = out + item
    return out


def assert_close(name, observed, expected, checks, atol=2e-10, rtol=2e-10):
    error = abs(observed-expected)
    limit = atol + rtol*max(abs(observed), abs(expected))
    if error > limit:
        raise AssertionError((name, observed, expected, error, limit))
    checks.append({'name': name, 'absolute_error': error, 'limit': limit})


def matrix_close(name, observed, expected, checks, atol=2e-10):
    assert_close(name, maxabs(add(observed, expected, -1)), 0.0, checks, atol, 0)


def solve_vjp_test(checks):
    f = fixture(3)
    rng = random.Random(22)
    k = rand(rng, f['dim'], f['contexts'], .8)
    q = rand(rng, f['dim'], f['batch'], .6)
    direction = rand(rng, f['dim'], f['contexts'], .7)
    kn = Node(k)
    p = writer(kn, f['a'][1], f['omega'], f['z'])
    scalar = (node(tr(q)) @ p).trace()
    scalar.backward()
    a, omega, z = f['a'][1], f['omega'], f['z']
    m = add(a, mm(mm(k, omega), tr(k)))
    y = solve(m, q)
    explicit = add(mm(mm(y, add(z, mm(tr(p.value), k), -1)), omega),
                   mm(mm(mm(p.value, tr(y)), k), omega), -1)
    matrix_close('implicit_solve_AD_equals_closed_form_K_VJP', kn.grad, explicit, checks)
    h = 1e-6
    plus = writer(node(add(k, direction, h)), a, omega, z).value
    minus = writer(node(add(k, direction, -h)), a, omega, z).value
    fd = inner(q, scale(add(plus, minus, -1), 1/(2*h)))
    analytic = inner(explicit, direction)
    assert_close('dynamic_solve_K_VJP_directional_finite_difference', analytic, fd, checks, 2e-8, 2e-8)
    return {'finite_difference': fd, 'analytic': analytic, 'step': h,
            'absolute_error': abs(fd-analytic), 'zero_context_weight_present': True}


def stored_weight_identity_test(checks):
    """Actual stored context weights need not sum to exactly one in float64."""
    rng = random.Random(109)
    stored_tenth = struct.unpack('f', struct.pack('f', .1))[0]
    weights = [.5] + [stored_tenth]*5
    z = [[1.0]*6 + [0.0]*6, [0.0]*6 + [1.0]*6]
    omega = diagonal(weights*2)
    a = scale(eye(3), .7)
    k = rand(rng, 3, 12, .8)
    p = writer(node(k), a, omega, z).value
    g = mm(mm(tr(p), a), p)
    residual = add(mm(tr(p), k), z, -1)
    e = mm(mm(residual, omega), tr(residual))
    kbar = mm(mm(k, omega), tr(z))
    mass = mm(mm(z, omega), tr(z))
    exact = add(mass, mm(tr(kbar), p), -1)
    matrix_close('stored_FP32_weights_general_GE_identity', add(g, e), exact, checks)
    unit_assumption_gap = maxabs(add(add(g, e), add(eye(2), mm(tr(kbar), p), -1), -1))
    if unit_assumption_gap <= 1e-9:
        raise AssertionError('Stored-weight test must expose the false exact-unit assumption')
    matrix_close('stored_FP32_weights_dense_dual', p, writer(node(k), a, omega, z, 'dual').value, checks)
    return {'stored_FP32_tenth_as_FP64': stored_tenth,
            'stored_per_request_context_mass': sum(weights),
            'general_identity': 'G+E = Z Omega Z^T - Kbar^T P',
            'incorrect_exact_I_identity_max_gap': unit_assumption_gap,
            'weights_were_not_renormalized': True}


def direct_linear_vjp_test(checks):
    rng = random.Random(1701)
    x = rand(rng, 7, 5, .8)
    d = rand(rng, 4, 2, .4)
    p = rand(rng, 5, 2, .5)
    w = rand(rng, 4, 5, .6)
    incoming = rand(rng, 7, 4, .7)
    xn, dn, pn = Node(x), Node(d), Node(p)
    output = xn @ (node(w) + dn @ pn.T).T
    loss = (node(tr(incoming)) @ output).trace()
    loss.backward()
    gd = mm(tr(incoming), mm(x, p))
    gp = mm(tr(x), mm(incoming, d))
    gx = mm(incoming, add(w, mm(d, tr(p))))
    matrix_close('direct_linear_D_VJP', dn.grad, gd, checks)
    matrix_close('direct_linear_P_VJP', pn.grad, gp, checks)
    matrix_close('direct_linear_input_VJP_keeps_entry_and_update_weight', xn.grad, gx, checks)
    hd, hp, hx = rand(rng, 4, 2), rand(rng, 5, 2), rand(rng, 7, 5)
    def value(x_, d_, p_):
        return inner(incoming, mm(x_, tr(add(w, mm(d_, tr(p_))))))
    step = 1e-6
    plus = value(add(x, hx, step), add(d, hd, step), add(p, hp, step))
    minus = value(add(x, hx, -step), add(d, hd, -step), add(p, hp, -step))
    fd = (plus-minus)/(2*step)
    predicted = inner(gx, hx) + inner(gd, hd) + inner(gp, hp)
    assert_close('direct_linear_joint_X_D_P_directional_FD', predicted, fd, checks, 2e-8, 2e-8)
    return {'joint_directional_fd': fd, 'analytic': predicted,
            'absolute_error': abs(fd-predicted), 'finite_difference_step': step,
            'D_vjp': 'G^T (X P)', 'P_vjp': 'X^T (G D)',
            'input_vjp': 'G (W_entry + D P^T)',
            'arithmetic_scope': 'binary64 real-graph test; reassociation across FP32 casts not qualified'}


def gradient_test(f, arm, checks):
    candidate = run_candidate(f, arm)
    candidate['loss'].backward()
    exact = [[row[:] for row in d.grad] for d in candidate['ds']]
    omitted = run_candidate(f, arm, detach_p=True)
    omitted['loss'].backward()
    frozen = [d.grad for d in omitted['ds']]
    h, errors, omitted_errors = 1e-6, [], []
    fd_gradients = []
    for l, d in enumerate(f['d']):
        layer_fd = zeros(len(d), len(d[0]))
        for i, row in enumerate(d):
            for r, original in enumerate(row):
                row[r] = original+h
                plus = run_candidate(f, arm)['loss'].value[0][0]
                row[r] = original-h
                minus = run_candidate(f, arm)['loss'].value[0][0]
                row[r] = original
                fd = (plus-minus)/(2*h)
                layer_fd[i][r] = fd
                errors.append(abs(fd-exact[l][i][r]))
                omitted_errors.append(abs(fd-frozen[l][i][r]))
        fd_gradients.append(layer_fd)
    assert_close('total_causal_gradient_B%d_arm%s' % (f['batch'], arm), max(errors), 0.0, checks, 3e-8, 0)
    missing = max(omitted_errors)
    if missing <= 1e-5:
        raise AssertionError('The chosen toy did not expose the detached-P error')
    return {'batch': f['batch'], 'arm': arm, 'coordinate_count': len(errors),
            'finite_difference_step': h, 'absolute_tolerance': 3e-8,
            'max_absolute_error': max(errors), 'detached_P_max_error': missing,
            'detached_P_forward_loss_equals_full': omitted['loss'].value[0][0] == candidate['loss'].value[0][0],
            'tested_path': 'native surrogate + dynamic G/E policy + selected actual surrogate'}


def permutation_fixture(f, permutation):
    out = dict(f)
    context_order = [c for old in permutation for c, owner in enumerate(f['owners']) if owner == old]
    token_order = [t for c in context_order for t in (2*c, 2*c+1)]
    inverse = {old: new for new, old in enumerate(permutation)}
    out['owners'] = [inverse[f['owners'][c]] for c in context_order]
    out['counts'] = [f['counts'][r] for r in permutation]
    out['z'] = [[float(owner == r) for owner in out['owners']] for r in range(f['batch'])]
    out['omega'] = diagonal([f['omega'][c][c] for c in context_order])
    out['injection'] = columns([f['injection'][r] for r in permutation], token_order)
    out['x'] = columns(f['x'], token_order)
    out['target'] = columns(f['target'], context_order)
    out['task_weights'] = [f['task_weights'][c] for c in context_order]
    out['actual_weights'] = [f['actual_weights'][c] for c in context_order]
    out['d'] = [columns(d, permutation) for d in f['d']]
    return out, context_order, token_order


def structural_tests(f, checks):
    original = run_candidate(f)
    original['loss'].backward()
    for l, capture in enumerate(original['captures']):
        k, p, g, e = [capture[name].value for name in ('K', 'P', 'G', 'E')]
        kbar = mm(mm(k, f['omega']), tr(f['z']))
        mass = mm(mm(f['z'], f['omega']), tr(f['z']))
        identity = add(mass, mm(tr(kbar), p), -1)
        matrix_close('dynamic_G_plus_E_identity_B%d_L%d' % (f['batch'], l), add(g, e), identity, checks)
        pd = writer(node(k), f['a'][l], f['omega'], f['z'], 'dual').value
        matrix_close('dense_dual_dynamic_writer_B%d_L%d' % (f['batch'], l), p, pd, checks)
        u, d = capture['U'].value, f['d'][l]
        physical_q = sum(mm(mm(u, f['a'][l]), tr(u))[i][i] for i in range(f['dim']))
        compact_q = sum(mm(mm(d, g), tr(d))[i][i] for i in range(f['dim']))
        assert_close('dynamic_write_trace_identity_B%d_L%d' % (f['batch'], l), physical_q, compact_q, checks)
        residual = add(mm(u, k), mm(d, f['z']), -1)
        residual_q = sum(mm(mm(residual, f['omega']), tr(residual))[i][i] for i in range(f['dim']))
        compact_e = sum(mm(mm(d, e), tr(d))[i][i] for i in range(f['dim']))
        assert_close('dynamic_realization_trace_identity_B%d_L%d' % (f['batch'], l), residual_q, compact_e, checks)
    dual = run_candidate(f, form='dual')
    dual['loss'].backward()
    assert_close('dense_dual_total_loss_B%d' % f['batch'], dual['loss'].value[0][0], original['loss'].value[0][0], checks)
    for l in range(3):
        matrix_close('dense_dual_total_gradient_B%d_L%d' % (f['batch'], l), dual['ds'][l].grad, original['ds'][l].grad, checks)
    for mb in sorted(set([1, 2, f['contexts']])):
        other = run_candidate(f, microbatch=mb)
        other['loss'].backward()
        assert_close('global_key_barrier_microbatch_loss_B%d_mb%d' % (f['batch'], mb), other['loss'].value[0][0], original['loss'].value[0][0], checks)
        for l in range(3):
            matrix_close('global_key_barrier_gradient_B%d_mb%d_L%d' % (f['batch'], mb, l), other['ds'][l].grad, original['ds'][l].grad, checks)
    cached = run_candidate(f, cached_first_p=original['captures'][0]['P'].value)
    cached['loss'].backward()
    assert_close('first_writer_cache_loss_B%d' % f['batch'], cached['loss'].value[0][0], original['loss'].value[0][0], checks)
    for l in range(3):
        matrix_close('first_writer_cache_gradient_B%d_L%d' % (f['batch'], l), cached['ds'][l].grad, original['ds'][l].grad, checks)
    changed = dict(f)
    changed['d'] = [[row[:] for row in d] for d in f['d']]
    changed['d'][2][0][0] += .2
    upper_change = run_candidate(changed)
    for l in range(3):
        matrix_close('upper_D_does_not_change_own_or_lower_K_B%d_L%d' % (f['batch'], l), upper_change['captures'][l]['K'].value, original['captures'][l]['K'].value, checks, 0)
    changed['d'] = [[row[:] for row in d] for d in f['d']]
    changed['d'][0][0][0] += .2
    lower_change = run_candidate(changed)
    p_change = maxabs(add(lower_change['captures'][1]['P'].value, original['captures'][1]['P'].value, -1))
    if p_change <= 1e-5:
        raise AssertionError('Lower D should alter an upper writer')
    weights = [c['effective_weight'].value for c in original['captures']]
    committed = run_candidate(f, fixed_weights=weights)
    matrix_close('committed_all_token_forward_equals_candidate_B%d' % f['batch'], committed['actual'].value, original['actual'].value, checks, 0)
    for l in range(3):
        matrix_close('committed_K_equals_built_K_B%d_L%d' % (f['batch'], l), committed['captures'][l]['K'].value, original['captures'][l]['K'].value, checks, 0)
    repeat = run_candidate(f)
    matrix_close('same_candidate_rebuild_output_B%d' % f['batch'], repeat['actual'].value, original['actual'].value, checks, 0)
    for l in range(3):
        matrix_close('same_candidate_rebuild_weight_B%d_L%d' % (f['batch'], l), repeat['captures'][l]['effective_weight'].value, weights[l], checks, 0)
    if f['batch'] > 1:
        perm = list(reversed(range(f['batch'])))
        pf, pc, pt = permutation_fixture(f, perm)
        reordered = run_candidate(pf)
        reordered['loss'].backward()
        assert_close('request_permutation_loss', reordered['loss'].value[0][0], original['loss'].value[0][0], checks)
        matrix_close('request_permutation_actual_output', reordered['actual'].value, columns(original['actual'].value, pt), checks)
        for l in range(3):
            matrix_close('request_permutation_shared_U_L%d' % l, reordered['captures'][l]['U'].value, original['captures'][l]['U'].value, checks)
            matrix_close('request_permutation_D_gradient_L%d' % l, reordered['ds'][l].grad, columns(original['ds'][l].grad, perm), checks)
            matrix_close('request_permutation_key_columns_L%d' % l, reordered['captures'][l]['K'].value, columns(original['captures'][l]['K'].value, pc), checks)
    return {'batch': f['batch'], 'context_counts': f['counts'],
            'all_contexts_kept_in_actual_graph': True,
            'zero_writer_weights': sum(f['omega'][c][c] == 0 for c in range(f['contexts'])),
            'lower_D_perturbation_upper_P_max_change': p_change,
            'first_layer_cache_exact': True, 'same_candidate_commit_exact': True,
            'microbatch_sizes': sorted(set([1, 2, f['contexts']]))}


def zero_convention_test(checks):
    d = Node(zeros(2, 3))
    norm = (d @ d.T).trace().root()
    norm.backward()
    assert_close('zero_root_selected_subgradient_value', norm.value[0][0], 0.0, checks, 0, 0)
    matrix_close('zero_root_selected_subgradient', d.grad, zeros(2, 3), checks, 0)


def run():
    checks = []
    solve_result = solve_vjp_test(checks)
    stored_weights = stored_weight_identity_test(checks)
    linear_vjp = direct_linear_vjp_test(checks)
    fixtures = [fixture(1), fixture(3)]
    structural = [structural_tests(f, checks) for f in fixtures]
    gradients = [gradient_test(f, arm, checks) for f in fixtures for arm in ('A', 'B')]
    zero_convention_test(checks)
    return {'passed': True, 'method_version': 'JLZ causal writer v7',
            'arithmetic': 'Python float (binary64 on this interpreter)',
            'toy_layers': 3, 'tokens_per_context': 2,
            'graph': 'causal all-token writer branch plus separate subject-only native surrogate',
            'solve_vjp': solve_result, 'structural_cases': structural,
            'stored_context_weight_identity': stored_weights,
            'direct_linear_vjp': linear_vjp,
            'total_gradient_checks': gradients, 'checks': checks,
            'limitations': [
                'This is a tiny smooth CPU surrogate, not native LM NLL or a model pilot.',
                'The tests do not establish FP32 materialization, mixed-dtype VJP equivalence, memory use, runtime, or task/locality gains.',
                'The SPD matrices are well conditioned; production SPD qualification and roundoff handling remain necessary.',
                'All current rewrite keys participate; selected actual loss rows do not change the writer fit set.',
                'Dynamic key refresh does not eliminate ridge shrinkage, inherited virtual/actual mismatch, or all-token side effects.',
                'Native clamp and Adam optimizer behavior are outside these algebra tests; no warmup is introduced.',
                'Teacher detach and pulse scheduling remain implementation contracts, not a fixed-objective convergence claim.'
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('validation-results.json'))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    script = Path(__file__).resolve()
    receipt = {'command': shlex.join([sys.executable, '-B', str(script), '--output', str(args.output.resolve())]),
               'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'python_version': sys.version,
               'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
               'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'passed': True, 'result_path': str(args.output.resolve())}
    args.output.with_name('run-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'passed': True, 'identity_and_structural_checks': len(result['checks']),
                      'total_gradient_cases': len(result['total_gradient_checks']),
                      'max_total_gradient_absolute_error': max(c['max_absolute_error'] for c in result['total_gradient_checks']),
                      'min_detached_P_error': min(c['detached_P_max_error'] for c in result['total_gradient_checks']),
                      'result': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
