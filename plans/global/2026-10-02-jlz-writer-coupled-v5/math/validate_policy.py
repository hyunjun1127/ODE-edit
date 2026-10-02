"""Stdlib checks of v5 proximal policy and native replay loss definitions.

Constructed Python-float fixtures only: these do not implement the production
runner, test a model, qualify FP32 kernels, or establish locality improvement.
Run without arguments to write policy-validation.json beside this file.
"""
from pathlib import Path
import hashlib
import json
import math
import random


def dot(x, y):
    assert len(x) == len(y)
    return sum(a * b for a, b in zip(x, y))


def norm(x):
    return math.sqrt(dot(x, x))


def close(x, y, tolerance=1e-10):
    assert math.isclose(x, y, rel_tol=tolerance, abs_tol=tolerance), (x, y)


def prox(y, tau, weight, radius):
    assert tau > 0 and weight >= 0 and radius > 0
    size = norm(y)
    if size == 0:
        return [0.0] * len(y)
    magnitude = min(radius, max(0.0, size - tau * weight))
    return [value * magnitude / size for value in y]


def prox_value(x, y, tau, weight):
    return sum((a - b) ** 2 for a, b in zip(x, y)) / (2 * tau) + weight * norm(x)


def check_prox():
    rng = random.Random(2102026)
    cases = [([0.0, 0.0], .2, .4, .75), ([.03, .04], .2, .25, .75),
             ([.3, .4], .2, .25, .75), ([3.0, 4.0], .2, .25, .75)]
    for dimension in (1, 2, 5):
        for _ in range(30):
            cases.append(([rng.uniform(-2, 2) for _ in range(dimension)],
                          rng.uniform(.01, 1), rng.uniform(0, 2), rng.uniform(.1, 1)))
    max_kkt, counts = 0.0, dict(zero=0, interior=0, boundary=0)
    for y, tau, weight, radius in cases:
        x = prox(y, tau, weight, radius)
        r = norm(x)
        assert r <= radius + 1e-12
        if r == 0:
            residual = max(0.0, norm(y) / tau - weight)
            counts['zero'] += 1
        else:
            normal_multiplier = max(0.0, (norm(y) - radius) / tau - weight) if abs(r - radius) < 1e-10 else 0.0
            residual = norm([(a - b) / tau + (weight + normal_multiplier) * a / r
                             for a, b in zip(x, y)])
            counts['boundary' if abs(r - radius) < 1e-10 else 'interior'] += 1
        max_kkt = max(max_kkt, residual)
        assert residual < 1e-10
    # Independent polar grid samples the entire 2-D ball, not just y's ray.
    grid_gaps = []
    for y, tau, weight, radius in cases[:4] + [([-.6, .15], .13, .6, .4)]:
        exact = prox_value(prox(y, tau, weight, radius), y, tau, weight)
        sampled = min(prox_value([radius * i / 100 * math.cos(2 * math.pi * j / 96),
                                 radius * i / 100 * math.sin(2 * math.pi * j / 96)], y, tau, weight)
                      for i in range(101) for j in range(96))
        assert exact <= sampled + 1e-11
        grid_gaps.append(sampled - exact)
    # Relative-coordinate norm and ball are algebraically the native payload ones.
    a, v, coefficient, radius = 2.3, [.12, -.17], .5, .75
    d = [a * x for x in v]
    close(coefficient * norm(d) / a ** 2, coefficient * norm(v) / a)
    assert (norm(d) <= radius * a) == (norm(v) <= radius)
    return dict(passed=True, cases=len(cases), branches=counts, max_KKT_residual=max_kkt,
                polar_grid_points_per_case=101 * 96, grid_minimum_minus_exact=grid_gaps,
                relative_native_norm_and_ball_identity=True)


def quad(x, matrix, linear, gradient=True):
    hx = [dot(row, x) for row in matrix]
    return .5 * dot(x, hx) - dot(linear, x), [h - b for h, b in zip(hx, linear)] if gradient else None


def grouped_step(x, g, tau, weights, sizes, radius):
    result, offset = [], 0
    for weight, size in zip(weights, sizes):
        result.extend(prox([x[j] - tau * g[j] for j in range(offset, offset + size)],
                           tau, weight, radius))
        offset += size
    assert offset == len(x)
    return result


def regularizer(x, weights, sizes):
    value, offset = 0.0, 0
    for weight, size in zip(weights, sizes):
        value += weight * norm(x[offset:offset + size])
        offset += size
    return value


def group_max(g, sizes):
    result, offset = 0.0, 0
    for size in sizes:
        result = max(result, norm(g[offset:offset + size]))
        offset += size
    return result


def proximal_run(matrix, linear, weights, sizes, candidate_cap=25, radius=.75):
    """Small reference state machine; rejected trials preserve accepted state."""
    assert candidate_cap >= 2
    x = [0.0] * len(linear)
    f, g = quad(x, matrix, linear)
    calls, backwards, accepted, rejected = 1, 1, 0, 0
    nu, max_gradient = radius / 4, group_max(g, sizes)
    if max_gradient == 0:
        return dict(x=x, candidates=calls, backwards=backwards, accepted=0, rejected=0,
                    exit='zero_initial_gradient', ledger=[])
    tau = nu / max_gradient
    ledger, last_trial = [], None
    while calls < candidate_cap:
        trial = grouped_step(x, g, tau, weights, sizes, radius)
        if trial == x:
            break  # Prox-variable identity, not just identical materialized W.
        displacement = [u - v for u, v in zip(trial, x)]
        trial_backwards = calls + 1 < candidate_cap
        tf, tg = quad(trial, matrix, linear, gradient=trial_backwards)
        calls += 1
        backwards += int(trial_backwards)
        majorant = f + dot(g, displacement) + dot(displacement, displacement) / (2 * tau)
        old_total = f + regularizer(x, weights, sizes)
        trial_total = tf + regularizer(trial, weights, sizes)
        accept = tf <= majorant + 1e-14
        ledger.append(dict(candidate=calls, tau=tau, accepted=accept,
                           smooth_majorization_gap=tf - majorant,
                           composite_change=trial_total - old_total,
                           max_trial_displacement=group_max(displacement, sizes)))
        last_trial = trial
        if accept:
            assert trial_total <= old_total + 1e-12
            accepted += 1
            x, f = trial, tf
            if tg is not None:
                g = tg
                current_max = group_max(g, sizes)
                tau = min(tau, nu / current_max) if current_max else tau
        else:
            rejected += 1
            tau *= .5  # Do not reset this to the last accepted tau next trial.
    return dict(x=x, candidates=calls, backwards=backwards, accepted=accepted,
                rejected=rejected, exit='candidate_cap' if calls == candidate_cap else 'prox_identity',
                last_trial=last_trial, ledger=ledger)


def check_solver():
    # PSD coupled Hessian with enough curvature to reject the first trial.
    matrix = [[120., 0., 0., 0.], [0., 2., .2, 0.],
              [0., .2, 5., 0.], [0., 0., 0., .7]]
    linear, weights, sizes = [.2, -.35, .03, .01], [.03, .1], [2, 2]
    full = proximal_run(matrix, linear, weights, sizes)
    assert full['candidates'] == 25 and full['backwards'] == 24
    assert full['accepted'] > 0 and full['rejected'] > 0
    assert full['accepted'] + full['rejected'] == 24
    assert full['ledger'][0]['max_trial_displacement'] < .75 / 4 + 1e-12
    first_rejections = [row for row in full['ledger'] if not row['accepted']]
    for first, second in zip(first_rejections, first_rejections[1:]):
        if second['candidate'] == first['candidate'] + 1:
            close(second['tau'], first['tau'] / 2)
    stopped = proximal_run(matrix, linear, weights, sizes, candidate_cap=2)
    assert stopped['x'] == [0.0] * 4 and stopped['last_trial'] != stopped['x']
    assert stopped['rejected'] == 1 and stopped['backwards'] == 1
    # Group 0 stays exactly zero at first, then reactivates via coupling.
    h, b, x = [[2., 1.5], [1.5, 2.]], [.1, 1.], [0., 0.]
    path = []
    for _ in range(10):
        f, g = quad(x, h, b)
        trial = grouped_step(x, g, .1, [.2, .1], [1, 1], .75)
        tf, _ = quad(trial, h, b)
        displacement = [u - v for u, v in zip(trial, x)]
        assert tf <= f + dot(g, displacement) + dot(displacement, displacement) / .2 + 1e-14
        assert tf + regularizer(trial, [.2, .1], [1, 1]) <= f + regularizer(x, [.2, .1], [1, 1]) + 1e-14
        path.append(trial)
        x = trial
    assert path[0][0] == 0 and any(row[0] != 0 for row in path[1:])
    assert prox([.01, -.02], 1.0, .1, .75) == [0.0, 0.0]
    return dict(passed=True, candidates=full['candidates'], backwards=full['backwards'],
                accepted_updates=full['accepted'], rejected_trials=full['rejected'],
                maximum_accepted_composite_change=max(row['composite_change'] for row in full['ledger'] if row['accepted']),
                ledger=full['ledger'], exhausted_rejection_returns_last_accepted=True,
                zero_threshold=True, reactivation_path=path)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logistic(theta, features, bias):
    s = dot(theta, features) + bias
    p = sigmoid(s)
    loss = max(-s, 0.0) + math.log1p(math.exp(-abs(s)))
    return loss, [(p - 1.0) * a for a in features], s, p


def current_kl(theta, features, bias, teacher_logit):
    _, _, s, p = logistic(theta, features, bias)
    q = sigmoid(teacher_logit)
    loss = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    gradient = [p * (1 - p) * (s - teacher_logit) * a for a in features]
    return loss, gradient


def past_replay(theta, facts):
    if not facts:
        return 0.0, [0.0] * len(theta)
    value, gradient = 0.0, [0.0] * len(theta)
    for fact in facts:
        kv, kg = current_kl(theta, fact['kl_features'], fact['kl_bias'], fact['teacher'])
        value += .0625 * kv / len(facts)
        gradient = [g + .0625 * h / len(facts) for g, h in zip(gradient, kg)]
        for features, bias, committed, weight in fact['rewrite']:
            nll, ng, _, _ = logistic(theta, features, bias)
            degradation = max(0.0, nll - committed)
            value += weight * .5 * degradation ** 2 / len(facts)
            gradient = [g + weight * degradation * h / len(facts) for g, h in zip(gradient, ng)]
    return value, gradient


def native(theta, batch):
    value, gradient = 0.0, [0.0] * len(theta)
    for r in range(batch):
        for context, weight in enumerate((.4, .6)):
            features = [math.sin((r + 1) * (context + 1) * (j + 2)) / len(theta) for j in range(len(theta))]
            bias = .15 * r - .2 * context
            nll, ng, _, _ = logistic(theta, features, bias)
            kl, kg = current_kl(theta, features, bias, bias)
            value += weight * (nll + .0625 * kl)
            gradient = [g + weight * (n + .0625 * k) for g, n, k in zip(gradient, ng, kg)]
        anchor, block = 1.0 + .3 * r, theta[2 * r:2 * r + 2]
        value += .5 * norm(block) / anchor
        for j, v in enumerate(block):
            gradient[2 * r + j] += .5 * v / (anchor * norm(block))
    return value, gradient


def check_normalization():
    records, largest_fd = [], 0.0
    for batch in (1, 2, 5):
        theta = [.04 + .015 * j for j in range(2 * batch)]
        facts = []
        for r in range(2):
            features = [math.cos((j + 1) * (r + 1)) / len(theta) for j in range(len(theta))]
            facts.append(dict(kl_features=features, kl_bias=.2 * r, teacher=-.1 + .1 * r,
                              rewrite=[(features, .1, .4, .25),
                                       ([-v for v in features], -.2, 1.2, .75)]))
        nv, ng = native(theta, batch)
        rv, rg = past_replay(theta, facts)
        mean_value, sum_value = nv / batch + rv, nv + batch * rv
        mean_gradient = [g / batch + r for g, r in zip(ng, rg)]
        sum_gradient = [g + batch * r for g, r in zip(ng, rg)]
        close(sum_value, batch * mean_value)
        for x, y in zip(sum_gradient, mean_gradient):
            close(x, batch * y)
        # Equivalent objectives also need reciprocal step scaling; identical
        # numeric tau on mean and SUM gradients would be a different solver.
        weights = [.5 / (1.0 + .3 * r) for r in range(batch)]
        sum_trial = grouped_step(theta, sum_gradient, .13, weights, [2] * batch, .75)
        mean_trial = grouped_step(theta, mean_gradient, batch * .13,
                                  [w / batch for w in weights], [2] * batch, .75)
        for x, y in zip(sum_trial, mean_trial):
            close(x, y)
        duplicated_value, duplicated_gradient = past_replay(theta, facts * 3)
        close(rv, duplicated_value)
        for x, y in zip(rg, duplicated_gradient):
            close(x, y)
        for j in range(len(theta)):
            step, plus, minus = 1e-6, list(theta), list(theta)
            plus[j] += step
            minus[j] -= step
            upper = native(plus, batch)[0] + batch * past_replay(plus, facts)[0]
            lower = native(minus, batch)[0] + batch * past_replay(minus, facts)[0]
            error = abs((upper - lower) / (2 * step) - sum_gradient[j])
            largest_fd = max(largest_fd, error)
            assert error < 2e-8
        empty_value, empty_gradient = past_replay(theta, [])
        assert empty_value == 0 and empty_gradient == [0.0] * len(theta)
        close(nv / batch, nv / batch + empty_value)
        records.append(dict(batch=batch, sum_value=sum_value, mean_value=mean_value,
                            reference_mean=rv, replicated_reference_mean=duplicated_value,
                            empty_reference_A_B_equal=True,
                            mean_vs_sum_prox_equal_with_reciprocal_step_scaling=True))
    return dict(passed=True, fixtures=records, largest_finite_difference_gradient_error=largest_fd,
                scope='Objective scaling/reference-duplication identity; NOT logical-edit-batch invariance')


def context_degradation(current, committed, weights):
    assert len(current) == len(committed) == len(weights)
    close(sum(weights), 1.0)
    return sum(.5 * w * max(0.0, x - b) ** 2 for x, b, w in zip(current, committed, weights))


def check_context_hinge():
    weights = [.5, .5]
    loss = context_degradation([1.0, 3.0], [2.0, 2.0], weights)
    close(loss, .25)
    wrong_after_average = .5 * max(0.0, .5 * (1 - 2) + .5 * (3 - 2)) ** 2
    assert loss > wrong_after_average == 0
    assert context_degradation([1., 1.5], [2., 2.], weights) == 0
    assert context_degradation([2., 2.], [2., 2.], weights) == 0
    # First average target-token NLL inside a context; hinge is per context.
    current_mean, commit_mean = sum([1., 5.]) / 2, sum([2., 2.]) / 2
    correct_token_mean = context_degradation([current_mean], [commit_mean], [1.0])
    wrong_tokenwise = context_degradation([1., 5.], [2., 2.], weights)
    close(correct_token_mean, .5)
    assert wrong_tokenwise != correct_token_mean
    finite_difference_errors = []
    for delta in (-.6, 0., .4):
        step = 1e-7
        numerical = (.5 * max(0., delta + step) ** 2 - .5 * max(0., delta - step) ** 2) / (2 * step)
        analytic = max(0., delta)
        error = abs(numerical - analytic)
        assert error < 3e-8
        finite_difference_errors.append(error)
    return dict(passed=True, contextwise_degradation=loss,
                wrong_hinge_after_context_mean=wrong_after_average,
                target_token_mean_before_hinge=correct_token_mean,
                wrong_tokenwise_hinge=wrong_tokenwise,
                improvement_is_free=True, equality_zero_value_and_gradient=True,
                finite_difference_gradient_errors=finite_difference_errors)


def main():
    source = Path(__file__)
    checks = dict(group_norm_ball_prox=check_prox(), proximal_solver=check_solver(),
                  native_reference_normalization=check_normalization(),
                  native_context_degradation=check_context_hinge())
    result = dict(status='PASS_STDLIB_POLICY_FIXTURES', method_id='jlz-writer-coupled-v5',
                  source_file=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checks=checks,
                  not_tested=['production runner', 'GPU or model inference', 'FP32 materialization/autograd',
                              'runtime qualification epsilon', 'memory reservoir/version implementation',
                              '25-candidate progress on language models', 'locality/retention improvement'])
    target = source.with_name('policy-validation.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], checks=list(checks), output=str(target)), ensure_ascii=False))


if __name__ == '__main__':
    main()
