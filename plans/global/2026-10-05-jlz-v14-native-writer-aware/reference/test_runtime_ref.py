"""CPU numerical and scheduling qualifications, with executable negative controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import traceback

import numpy as np

from runtime_ref import (Row, causal_two_layer, fingerprint, probe_candidate,
                         solve_forward, solve_vjp, solve_vjp_cached_A,
                         stream_candidate, whole_batch_rows)

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(2026100502)
RECORDS = []


def close(actual, expected, atol=2e-10, rtol=2e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def record(name, **measurements):
    RECORDS.append(dict(test=name, status="PASS", measurements=measurements))


def matrix(n, nonsymmetric=False):
    X = RNG.normal(size=(n, n))
    A = X @ X.T + .9 * np.eye(n)
    if nonsymmetric:
        skew = RNG.normal(size=(n, n))
        A += .7 * (skew - skew.T)  # SPD symmetric part => invertible, not symmetrized.
    return A


def fd(function, value, eps=2e-6):
    result = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        step = eps * max(1., abs(float(value[index])))
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def test_cached_vjp_finite_differences():
    errors = []
    shapes = [(5, 3, False), (5, 3, True), (3, 7, True), (4, 5, False)]
    for n, count, nonsymmetric in shapes:
        A, K = matrix(n, nonsymmetric), RNG.normal(size=(n, count))
        if count == 5:
            K[:, 1] = K[:, 0]
            K[:, 2] = 0
        Gp = RNG.normal(size=K.shape)
        ledger = {}
        tape = solve_forward(A, K, ledger)
        result = solve_vjp(tape, Gp, ledger=ledger)
        expected = fd(lambda key: float(np.sum(np.linalg.solve(A + key @ key.T, key) * Gp)), K)
        close(result, expected, atol=2e-8, rtol=4e-6)
        assert ledger == {"primal_solves": 1, "transpose_solves": 1}
        assert not tape.P.flags.writeable and not tape.K.flags.writeable and not tape.B.flags.writeable
        errors.append(float(np.max(np.abs(result - expected))))
    record("cached_solve_VJP_finite_differences", shapes=shapes, max_abs_error=max(errors),
           primal_solve_count_per_forward=1, new_primal_solves_in_reverse=0,
           transpose_solve_count_per_reverse=1, duplicate_and_zero_columns=True)


def test_vjp_negative_controls_and_binding():
    A, K = matrix(5, True), RNG.normal(size=(5, 3))
    Gp = RNG.normal(size=K.shape)
    tape = solve_forward(A, K)
    correct = solve_vjp(tape, Gp)
    expected = fd(lambda key: float(np.sum(np.linalg.solve(A + key @ key.T, key) * Gp)), K)
    Z = np.linalg.solve(tape.B.T, Gp)
    wrong_term = Z - Z @ tape.P.T @ K - K @ tape.P.T @ Z
    wrong_Z = np.linalg.solve(tape.B, Gp)
    wrong_transpose = wrong_Z - (wrong_Z @ tape.P.T + tape.P @ wrong_Z.T) @ K
    correct_error = float(np.linalg.norm(correct - expected))
    wrong_errors = [float(np.linalg.norm(wrong_term - expected)), float(np.linalg.norm(wrong_transpose - expected))]
    assert correct_error < 1e-7 and min(wrong_errors) > 1e-3
    old = tape.P.copy()
    A[0, 0] += 1
    K[0, 0] += .1
    assert fingerprint(A, K) != tape.binding
    assert np.array_equal(tape.P, old)
    record("incorrect_last_term_and_nonsymmetric_transpose_negative_controls",
           correct_FD_error=correct_error, wrong_last_term_error=wrong_errors[0],
           no_transpose_solve_error=wrong_errors[1], mutation_changes_binding=True,
           cached_values_immutable=True)


def test_fixed_A_transpose_identity_and_cached_VJP():
    transpose_errors, gradient_errors, finite_errors = [], [], []
    for n, count, nonsymmetric in [(6, 3, False), (6, 3, True), (3, 8, True), (4, 5, False)]:
        A, K = matrix(n, nonsymmetric), RNG.normal(size=(n, count))
        if count == 5:
            K[:, 1] = K[:, 0]
            K[:, 2] = 0
        Gp = RNG.normal(size=K.shape)
        tape = solve_forward(A, K)
        dense_Z = np.linalg.solve(tape.B.T, Gp)
        fixed_Z = np.linalg.solve(A.T, Gp - K @ (tape.P.T @ Gp))
        close(fixed_Z, dense_Z)
        ledger = {}
        fixed_gradient = solve_vjp_cached_A(A, tape, Gp, ledger=ledger)
        dense_gradient = solve_vjp(tape, Gp)
        expected = fd(lambda key: float(np.sum(np.linalg.solve(A + key @ key.T, key) * Gp)), K)
        close(fixed_gradient, dense_gradient)
        close(fixed_gradient, expected, atol=3e-8, rtol=5e-6)
        assert ledger == {"fixed_A_transpose_solves": 1}
        transpose_errors.append(float(np.max(np.abs(fixed_Z - dense_Z))))
        gradient_errors.append(float(np.max(np.abs(fixed_gradient - dense_gradient))))
        finite_errors.append(float(np.max(np.abs(fixed_gradient - expected))))
    record("cached_P_fixed_A_transpose_identity_and_VJP",
           identity="B^-T G = solve(A^T, G-K(P^T G))",
           max_transpose_solve_difference=max(transpose_errors),
           max_VJP_difference=max(gradient_errors), max_VJP_FD_error=max(finite_errors),
           SPD_and_nonsymmetric_A=True, duplicate_zero_columns_and_batch_wider_than_dimension=True,
           additional_B_or_primal_P_solves_in_reverse=0,
           fixed_A_transpose_solves_per_reverse=1)


def test_fixed_A_singular_or_unqualified_native_B_fallback():
    A = np.diag([0., 2., 3.])
    K = np.array([[1., 0.], [.2, 1.], [.1, -.3]])
    Gp = np.array([[.2, -.4], [.7, .5], [-.1, .9]])
    tape = solve_forward(A, K)
    dense = solve_vjp(tape, Gp)
    singular_ledger = {}
    singular = solve_vjp_cached_A(A, tape, Gp, ledger=singular_ledger)
    fallback_ledger = {}
    qualified_fallback = solve_vjp_cached_A(A, tape, Gp, A_backend_qualified=False, ledger=fallback_ledger)
    expected = fd(lambda key: float(np.sum(np.linalg.solve(A + key @ key.T, key) * Gp)), K)
    close(singular, dense, atol=0, rtol=0)
    close(qualified_fallback, dense, atol=0, rtol=0)
    close(singular, expected, atol=3e-8, rtol=5e-6)
    assert singular_ledger == {"singular_A_B_fallbacks": 1, "transpose_solves": 1}
    assert fallback_ledger == {"native_B_fallbacks": 1, "transpose_solves": 1}
    changed = A.copy()
    changed[1, 1] += .1
    try:
        solve_vjp_cached_A(changed, tape, Gp)
    except ValueError:
        pass
    else:
        raise AssertionError("stale fixed-A cache was accepted")
    record("singular_or_unqualified_A_retains_native_B_transpose_backend",
           native_B_invertible=True, singular_A=True,
           fallback_VJP_FD_max_error=float(np.max(np.abs(singular - expected))),
           singular_and_explicit_unqualified_fallbacks_bitwise_match=True,
           changed_A_candidate_binding_rejected=True)


def causal_inputs():
    return dict(A1=matrix(4, True), A2=matrix(3, True), K1=RNG.normal(size=(4, 5)),
                R1=RNG.normal(size=(2, 5)) * .3, R2=RNG.normal(size=(3, 5)) * .4,
                link=RNG.normal(size=(3, 2)), skip=RNG.normal(size=(3, 2)),
                base_key=RNG.normal(size=(3, 5)) * .3,
                base_out=RNG.normal(size=(3, 5)) * .2, target=RNG.normal(size=(3, 5)))


def test_causal_full_gradient_and_first_fixed_cache():
    values = causal_inputs()
    result = causal_two_layer(**values, backward=True)
    complete = causal_two_layer(**values, backward=True, skip_fixed_first_key=False)
    close(result["R1_gradient"], complete["R1_gradient"], atol=0, rtol=0)
    close(result["R2_gradient"], complete["R2_gradient"], atol=0, rtol=0)
    errors = {}
    for name in ("R1", "R2", "K1"):
        reference = fd(lambda x: causal_two_layer(**(values | {name: x}))["loss"], values[name])
        gradient = complete[name + "_gradient"]
        close(gradient, reference, atol=3e-8, rtol=5e-6)
        errors[name] = float(np.max(np.abs(gradient - reference)))
    assert result["ledger"]["transpose_solves"] == 1
    assert complete["ledger"]["transpose_solves"] == 2
    assert np.linalg.norm(result["R1_gradient"]) > .01
    record("causal_two_layer_full_R_key_P_gradient_and_fixed_first_key_skip",
           max_abs_errors=errors, cached_first_R_gradient_norm=float(np.linalg.norm(result["R1_gradient"])),
           required_transpose_solves=1, full_unused_first_key_adjoint_transpose_solves=2,
           R_gradients_bitwise_identical_when_skipping_unused_first_key=True)


def test_causal_upper_solve_negative_control():
    values = causal_inputs()
    good = causal_two_layer(**values, backward=True)
    frozen = causal_two_layer(**values, backward=True, stop_upper_solve=True)
    error = float(np.linalg.norm(good["R1_gradient"] - frozen["R1_gradient"]))
    assert error > 1e-3
    close(good["R2_gradient"], frozen["R2_gradient"], atol=0, rtol=0)
    record("stopping_upper_solve_VJP_is_a_different_gradient", lower_R_gradient_error=error,
           upper_direct_R_gradient_unchanged=True)


def test_unequal_owner_rows_and_short_microbatch_tail():
    owners = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    aggregation = np.array([.5, .25, .25, 1., .5, .5, .5, 1/6, 1/6, 1/6])
    X, R = RNG.normal(size=(3, 10)), RNG.normal(size=(2, 4)) * .2
    A, target = matrix(3, True), RNG.normal(size=(2, 10))
    all_rows = [list(range(10))]
    base = whole_batch_rows(A, X, R, owners, aggregation, target, all_rows)
    differences = []
    for width in (1, 3, 4, 6):
        chunks = [list(range(i, min(i + width, 10))) for i in range(0, 10, width)]
        result = whole_batch_rows(A, X, R, owners, aggregation, target, chunks)
        close(result["loss"], base["loss"])
        close(result["R_gradient"], base["R_gradient"])
        close(result["X_gradient"], base["X_gradient"])
        assert result["ledger"] == {"primal_solves": 1, "transpose_solves": 1}
        differences.append(float(np.max(np.abs(result["R_gradient"] - base["R_gradient"]))))
    # Independent owner-normalized forward expression, not the chunk reducer.
    P = np.linalg.solve(A + base["tape"].K @ base["tape"].K.T, base["tape"].K)
    row_errors = .5 * np.sum((np.tanh(R @ P.T @ X) - target) ** 2, axis=0)
    independent_loss = sum(float(row_errors[owners == r].mean()) for r in range(4))
    close(base["loss"], independent_loss)
    fd_R = fd(lambda r: whole_batch_rows(A, X, r, owners, aggregation, target, all_rows)["loss"], R)
    fd_X = fd(lambda x: whole_batch_rows(A, x, R, owners, aggregation, target, all_rows)["loss"], X)
    close(base["R_gradient"], fd_R, atol=3e-8, rtol=5e-6)
    close(base["X_gradient"], fd_X, atol=3e-8, rtol=5e-6)
    record("whole_B_owner_weighted_SUM_microbatch_partition_and_short_tail",
           owner_row_counts=[3, 1, 2, 4], logical_B=4, key_dimension=3,
           microbatch_widths=[1, 3, 4, 6], max_R_gradient_partition_error=max(differences),
           R_FD_max_error=float(np.max(np.abs(base["R_gradient"] - fd_R))),
           X_FD_max_error=float(np.max(np.abs(base["X_gradient"] - fd_X))),
           whole_B_solve_not_split=True)


def row(owner, component, value, number):
    # Deliberately cancellation-sensitive gradient sums exercise row order.
    return Row(owner, component, value, np.array([number, number * .17, -number * .4], dtype=np.float64))


def canonical_schedule(groups, norm, norm_gradient):
    nll, kl = np.zeros(len(norm)), np.zeros(len(norm))
    gradient = np.zeros_like(norm_gradient)
    for group in groups:
        for r in group:
            (nll if r.component == "nll" else kl)[r.owner] += r.value
            gradient += r.gradient * (.0625 if r.component == "kl" else 1.)
    gradient += norm_gradient
    return nll + .0625 * kl + norm, nll, kl, gradient


def interleaved_groups():
    return [[row(0, "nll", .07, 1e12)],
            [row(1, "nll", .02, -1e12)],
            [row(0, "nll", .01, .0123)],
            [row(1, "kl", -.001, .08)],
            [row(0, "kl", -.8, -.3)],
            [row(2, "nll", .06, 1.2)],
            [row(2, "kl", 0., -.4)]]


def test_schedule_complete_owner_witness_and_memory_replay():
    groups = interleaved_groups()
    norm, ng = np.zeros(3), np.array([.3, -.9, .07])
    J, nll, kl, grad = canonical_schedule(groups, norm, ng)
    result = stream_candidate(groups, norm, ng, memory_groups=2)
    # Owner0's early partial .07 is NOT a witness; complete J0=.03.
    assert result["witness"]["owner"] == 2 and result["witness"]["group"] == 6
    assert result["replay_order"] == [0, 1, 2, 3]
    assert result["backward_order"] == list(range(7))
    assert result["norm_gradient_additions"] == 1 and result["update_allowed"]
    assert result["peak_retained_groups"] <= 2
    for name, expected in [("J", J), ("nll", nll), ("kl", kl), ("gradient", grad)]:
        assert np.array_equal(result[name], expected), name
    record("completed_owner_witness_not_partial_rows_and_bounded_prefix_replay",
           witness=result["witness"], replay_groups=result["replay_order"],
           backward_order=result["backward_order"], prefix_graph_retention_cap=2,
           observed_peak_retained_groups=result["peak_retained_groups"],
           measured_objective_and_gradient_bitwise_match=True)


def test_schedule_all_pass_and_final_cap():
    groups = [[row(0, "nll", .049, .2), row(1, "nll", .03, .3)],
              [row(0, "kl", -.0003, -.1)], [row(1, "kl", 0, .2)]]
    ng = np.array([.1, .2, .3])
    passed = stream_candidate(groups, np.zeros(2), ng, memory_groups=0)
    assert passed["stop_all"] and not passed["update_allowed"]
    assert passed["backward_calls"] == 0 and passed["replay_order"] == []
    capped = stream_candidate(interleaved_groups(), np.zeros(3), ng, memory_groups=0, final_cap=True)
    assert not capped["stop_all"] and not capped["update_allowed"]
    assert capped["backward_calls"] == 0 and capped["replay_order"] == []
    assert capped["forward_calls"] == 7 and capped["norm_gradient_additions"] == 0
    record("all_pass_stop_and_final_cap_are_value_only",
           all_pass_backward_calls=passed["backward_calls"], cap_backward_calls=capped["backward_calls"],
           cap_forward_calls=capped["forward_calls"], no_optimizer_permission_at_stop=True)


def test_schedule_equality_and_negative_KL_boundary():
    ng = np.array([.1, .2, .3])
    equality = [[row(0, "nll", .05, 1)], [row(1, "nll", .01, 2)]]
    eq = stream_candidate(equality, np.zeros(2), ng)
    assert eq["witness"]["owner"] == 0 and eq["witness"]["J"] == .05
    assert eq["update_allowed"] and eq["backward_order"] == [0, 1]
    # A native FP32-scale negative KL makes J < .05 despite norm > .05.
    groups = [[row(0, "nll", 1e-7, .2)], [row(0, "kl", -3e-6, .4)]]
    norm = np.array([.05000005])
    result = stream_candidate(groups, norm, ng, memory_groups=0)
    assert norm[0] > .05 and result["J"][0] < .05
    assert result["stop_all"] and result["witness"] is None and result["backward_calls"] == 0
    record("strict_threshold_equality_and_negative_KL_no_positivity_shortcut",
           equality_J=eq["witness"]["J"], tiny_negative_KL=-3e-6,
           requested_norm=float(norm[0]), complete_J=float(result["J"][0]),
           false_norm_only_witness_avoided=True)


def test_schedule_late_invalid_and_replay_identity_abort():
    ng = np.array([.1, .2, .3])
    groups = [[row(0, "nll", .08, .4)], [row(1, "nll", .01, .2)],
              [row(1, "kl", float("nan"), .3)]]
    bad = stream_candidate(groups, np.zeros(2), ng)
    assert bad["backward_calls"] == 2 and bad["status"] == "ABORTED"
    assert not bad["update_allowed"] and np.array_equal(bad["gradient"], np.zeros(3))
    assert bad["norm_gradient_additions"] == 0
    def changed(index, rows):
        r = rows[0]
        return [Row(r.owner, r.component, np.nextafter(r.value, np.inf), r.gradient)]
    replay_bad = stream_candidate(interleaved_groups(), np.zeros(3), ng,
                                  memory_groups=1, replay_transform=changed)
    assert replay_bad["status"] == "ABORTED" and replay_bad["error"] == "REPLAY_CERTIFICATE_MISMATCH"
    assert not replay_bad["update_allowed"] and np.array_equal(replay_bad["gradient"], np.zeros(3))
    record("late_nonfinite_and_replay_bit_identity_abort_whole_update",
           backward_calls_before_late_nonfinite=bad["backward_calls"], late_error=bad["error"],
           replay_error=replay_bad["error"], candidate_adjoints_discarded=True,
           optimizer_updates_allowed=0)


def test_schedule_short_logical_batch_and_prefix_only_replay():
    # A trailing logical B=2 has unequal row counts 1/3 and a short row group.
    groups = [[row(1, "nll", .01, .03)], [row(0, "nll", .051, -.04)],
              [row(1, "nll", .005, .07), row(1, "kl", -.0001, .08)]]
    ng = np.array([.11, .13, -.17])
    J, _, _, gradient = canonical_schedule(groups, np.zeros(2), ng)
    result = stream_candidate(groups, np.zeros(2), ng, memory_groups=0)
    assert result["J"].shape == (2,) and result["witness"]["group"] == 1
    assert result["replay_order"] == [0] and result["forward_order"] == [0, 1, 2]
    assert result["backward_order"] == [0, 1, 2]
    assert np.array_equal(result["J"], J) and np.array_equal(result["gradient"], gradient)
    record("short_logical_batch_original_order_and_only_evicted_prefix_replayed",
           logical_B=2, owner_row_counts=[1, 3], witness_group=1,
           replay_groups=result["replay_order"], backward_order=result["backward_order"],
           measured_objective_and_gradient_bitwise_match=True)


def test_detached_probe_exact_witness_and_unequal_tail():
    groups = [[row(0, "nll", .04, 1e12), row(1, "nll", .003, .08)],
              [row(2, "nll", .01, -1e12)],
              [row(0, "kl", -.01, .003)],
              [row(1, "kl", .004, .04), row(2, "kl", 0., .05)]]
    norm, ng = np.array([.02, .01, 0.]), np.array([.2, -.1, .9])
    J, nll, kl, grad = canonical_schedule(groups, norm, ng)
    result = probe_candidate(groups, norm, ng)
    assert result["chosen_owner"] == 0 and result["probe_groups"] == [0, 2]
    assert result["witness"]["source"] == "COMPLETE_OWNER_PROBE"
    assert result["value_forward_order"] == [0, 2]
    assert result["training_forward_order"] == result["backward_order"] == [0, 1, 2, 3]
    assert not result["fallback_full_value"] and result["update_allowed"]
    for name, expected in [("J", J), ("nll", nll), ("kl", kl), ("gradient", grad)]:
        assert np.array_equal(result[name], expected), name
    record("default_detached_complete_owner_probe_then_single_original_order_FB",
           logical_B=3, row_groups=4, probe_groups=result["probe_groups"],
           forward_calls=result["forward_calls"], baseline_full_value_plus_FB_calls=8,
           backward_calls=result["backward_calls"], measured_J_and_gradient_bitwise_match=True,
           this_is_a_toy_call_count_not_a_measured_speedup=True)


def test_detached_probe_fallback_reuse_and_all_pass():
    groups = interleaved_groups()
    ng = np.array([.1, -.2, .3])
    norm = np.array([.001, 0., 0.])
    result = probe_candidate(groups, norm, ng)
    J, _, _, gradient = canonical_schedule(groups, norm, ng)
    assert result["chosen_owner"] == 0 and result["probe_groups"] == [0, 2, 4]
    assert result["fallback_full_value"] and result["witness"]["owner"] == 2
    assert len(result["value_forward_order"]) == len(groups)
    assert len(set(result["value_forward_order"])) == len(groups)
    assert result["backward_order"] == list(range(len(groups)))
    assert np.array_equal(result["J"], J) and np.array_equal(result["gradient"], gradient)
    passing = [[row(1, "nll", .02, .2)], [row(0, "nll", .01, -.3)],
               [row(0, "kl", -.001, .4), row(1, "kl", 0., .1)]]
    passed = probe_candidate(passing, np.zeros(2), ng)
    assert passed["chosen_owner"] == 0  # Original-owner tie, not first row's owner.
    assert passed["stop_all"] and passed["backward_calls"] == 0 and not passed["update_allowed"]
    assert sorted(passed["value_forward_order"]) == [0, 1, 2]
    assert passed["training_forward_order"] == []
    record("probe_miss_reuses_detached_groups_and_all_pass_has_no_backward",
           fallback_value_forward_calls=len(result["value_forward_order"]),
           fallback_groups_once=True, original_order_reduction_bitwise_match=True,
           all_pass_logical_B=2, all_pass_backward_calls=0, deterministic_tie_owner=0)


def test_detached_probe_equality_negative_KL_cap_and_late_failure():
    ng = np.array([.1, .2, .3])
    groups = [[row(0, "nll", .05, .1)], [row(1, "nll", .01, .2)]]
    equal = probe_candidate(groups, np.zeros(2), ng)
    assert equal["update_allowed"] and equal["witness"]["J"] == .05
    negative = [[row(0, "nll", 1e-7, .1)], [row(0, "kl", -3e-6, .2)]]
    passing = probe_candidate(negative, np.array([.05000005]), ng)
    assert passing["stop_all"] and passing["backward_calls"] == 0 and passing["J"][0] < .05
    cap = probe_candidate(groups, np.zeros(2), ng, final_cap=True)
    assert cap["chosen_owner"] is None and cap["probe_groups"] == []
    assert cap["backward_calls"] == 0 and cap["forward_calls"] == 2 and not cap["update_allowed"]
    bad_groups = [[row(0, "nll", .08, .2)], [row(1, "nll", .01, .3)],
                  [row(1, "kl", np.nan, .4)]]
    bad = probe_candidate(bad_groups, np.zeros(2), ng)
    assert bad["status"] == "ABORTED" and bad["backward_calls"] == 2
    assert not bad["update_allowed"] and np.array_equal(bad["gradient"], np.zeros(3))
    assert bad["norm_gradient_additions"] == 0
    def altered(index, rows):
        r = rows[0]
        return [Row(r.owner, r.component, np.nextafter(r.value, np.inf), r.gradient)]
    invalid_replay = probe_candidate(groups, np.zeros(2), ng, replay_transform=altered)
    assert invalid_replay["error"] == "REPLAY_CERTIFICATE_MISMATCH"
    assert not invalid_replay["update_allowed"]
    record("probe_threshold_negative_KL_cap_late_failure_and_replay_identity",
           equality_is_nonterminal=True, norm_only_lower_bound_not_used=True,
           negative_KL_complete_J=float(passing["J"][0]), cap_backward_calls=0,
           late_failure_prior_backward_calls=2, late_failure_update_allowed=False,
           changed_replay_aborts=True)


def main():
    failed = []
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception:
                failed.append(dict(test=name, status="FAIL", traceback=traceback.format_exc()))
    audit = dict(scope="V14 prospective runtime algebra and toy stop/adjoint scheduling only",
                 status="PASS" if not failed else "FAIL", passed=len(RECORDS), failed=len(failed),
                 runtime=dict(python=platform.python_version(), numpy=np.__version__, device="CPU"),
                 source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                for name in ("runtime_ref.py", "test_runtime_ref.py", "requirements.txt")},
                 tests=RECORDS+failed,
                 limitations=["No production runner was changed and no model/GPU forward was run.",
                              "Dense NumPy solve proves cached P reuse, not actual factor reuse or speedup.",
                              "FP32 cast/materialized-weight adjoints and GPU numerical parity remain unqualified here.",
                              "The smooth two-layer model is a derivative reference, not a language-model performance test.",
                              "The schedule uses supplied row gradients and a same-candidate identity token; production must bind full execution state.",
                              "Bitwise scheduling checks concern this CPU canonical reducer, not arbitrary floating-point kernel reorderings.",
                              "No locality, accuracy, convergence, or wall-clock improvement is claimed."])
    (HERE/"runtime-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(dict(status=audit["status"], passed=audit["passed"], failed=audit["failed"])))
    if failed:
        for item in failed:
            print(item["traceback"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
