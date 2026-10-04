"""Deterministic CPU qualification of the writer algebra, not model quality."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

from writer import applied_fp32_diagnostics, repeat_context_targets, solve_writer


HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261005)
RECORDS: list[dict] = []


def spd(n: int) -> np.ndarray:
    C = RNG.normal(size=(n, n))
    return C @ C.T + np.eye(n)


def weights(q: int) -> np.ndarray:
    return np.full(q, 1.0 / q)


def close(a, b, *, rtol=2e-10, atol=2e-11):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def record(name, diagnostics):
    RECORDS.append({"test": name, "status": "PASS", "measurements": diagnostics})


def test_full_rank_equality_and_minimum_energy():
    A, K, T = spd(7), RNG.normal(size=(7, 3)), RNG.normal(size=(2, 3))
    result = solve_writer(A, K, T, weights(3))
    close(result.update @ K, T)
    assert result.diagnostics["rank"] == 3
    AinvK = np.linalg.solve(A, K)
    reference = np.linalg.solve((K.T @ AinvK).T, T.T).T @ AinvK.T
    close(result.update, reference)
    L = np.linalg.cholesky(A)
    X = np.linalg.solve(L, K)
    left, _, _ = np.linalg.svd(X, full_matrices=False)
    white_delta = RNG.normal(size=(2, 7)) @ (np.eye(7) - left @ left.T)
    delta_U = np.linalg.solve(L.T, white_delta.T).T
    alternative = result.update + delta_U
    close(alternative @ K, T)
    energy_alternative = np.sum((alternative @ A) * alternative)
    energy_gap = float(energy_alternative - result.diagnostics["energy_A"])
    close(energy_gap, np.sum(white_delta * white_delta))
    assert energy_gap > 0
    record("full_rank_equality_minimum_energy", {**result.diagnostics, "energy_gap_feasible_perturbation": energy_gap, "gram_reference_update_error": float(np.linalg.norm(result.update - reference))})


def test_compatible_duplicate_keys():
    base = RNG.normal(size=(5, 2))
    K = base[:, [0, 1, 0]]
    D = RNG.normal(size=(2, 2))
    T = D[:, [0, 1, 0]]
    result = solve_writer(spd(5), K, T, np.array([0.2, 0.3, 0.5]))
    close(result.update @ K, T)
    assert result.diagnostics["rank"] == 2
    assert result.diagnostics["compatible_with_retained_subspace"]
    record("compatible_duplicate_keys", result.diagnostics)


def test_conflicting_duplicate_keys_weighted_projection():
    K = np.array([[1.0, 1.0], [0.0, 0.0]])
    T = np.array([[1.0, 3.0]])
    result = solve_writer(np.eye(2), K, T, np.array([0.25, 0.75]))
    close(result.update @ K, [[2.5, 2.5]])
    assert not result.diagnostics["compatible_with_retained_subspace"]
    assert result.diagnostics["status"] == "RANGE_PROJECTED_WEIGHTED_LEAST_SQUARES"
    assert result.diagnostics["retained_LS_orthogonality_norm"] < 1e-12
    record("conflicting_duplicates_weighted_LS", result.diagnostics)


def test_zero_target_is_required_cross_talk_constraint():
    K = np.array([[1.0, 1.0], [0.0, 1.0]])
    T = np.array([[1.0, 0.0]])
    full = solve_writer(np.eye(2), K, T, weights(2))
    dropped = solve_writer(np.eye(2), K[:, :1], T[:, :1], weights(1))
    close(full.update @ K, T)
    close(dropped.update @ K[:, 1:], [[1.0]])
    record("retain_zero_target_prevents_cross_talk", {"full_writer_zero_column_error": float(np.linalg.norm(full.update @ K[:, 1:])), "dropped_writer_zero_column_error": float(np.linalg.norm(dropped.update @ K[:, 1:]))})


def test_context_roles_variable_owners_and_history_separation():
    # Owner 0 has canonical/prefix/KL; owner 1 has canonical/KL. No 6+1 assumption.
    K_rows = np.eye(5)
    owners = np.array([0, 0, 0, 1, 1])
    roles = ["rewrite", "rewrite", "kl", "rewrite", "kl"]
    row_weights = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2])
    D = np.array([[2.0, 3.0], [-1.0, 4.0]])
    K, T, w, selected = repeat_context_targets(K_rows, owners, roles, row_weights, D, included_roles=("rewrite", "kl"))
    close(w[owners == 0].sum(), 0.5)
    close(w[owners == 1].sum(), 0.5)
    close(T[:, [2, 4]], D)
    assert np.array_equal(selected, np.arange(5))
    result = solve_writer(np.eye(5), K, T, w)
    close(result.update @ K, T)
    # Native history support remains rewrite-only. This example has one native
    # rewrite group per owner; real adapters must also preserve nested means.
    K_hist = np.stack([K_rows[:, (owners == owner) & (np.array(roles) == "rewrite")].mean(axis=1) for owner in range(2)], axis=1)
    close(K_hist[:, 0], np.array([0.5, 0.5, 0, 0, 0]))
    close(K_hist[:, 1], np.array([0, 0, 0, 1, 0]))
    H_append = K_hist @ K_hist.T
    assert H_append[2, 2] == 0 and H_append[4, 4] == 0
    record("variable_context_counts_KL_owner_target_history_separate", {"constraint_count": len(selected), "owner_constraint_weight_sums": [float(w[owners == i].sum()) for i in range(2)], "KL_targets": T[:, [2, 4]].tolist(), "history_KL_only_dimensions_energy": float(H_append[2, 2] + H_append[4, 4]), "weighted_residual_norm": result.diagnostics["weighted_residual_norm"]})


def test_mean_exact_does_not_imply_context_exact():
    K = np.array([[2.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0]])
    result = solve_writer(np.eye(2), K.mean(axis=1, keepdims=True), target, weights(1))
    close(result.update @ K.mean(axis=1, keepdims=True), target)
    context_effects = result.update @ K
    assert np.max(np.abs(context_effects - 1)) > 0.5
    record("mean_exact_not_context_exact", {"context_effects": context_effects.tolist(), "mean_effect": float(context_effects.mean()), "context_target_error": float(np.linalg.norm(context_effects - 1))})


def test_context_exact_energy_no_lower_than_mean_exact():
    A, K, D = spd(6), RNG.normal(size=(6, 4)), RNG.normal(size=(2, 2))
    T = D[:, [0, 0, 1, 1]]
    context = solve_writer(A, K, T, weights(4))
    K_mean = np.stack([K[:, :2].mean(axis=1), K[:, 2:].mean(axis=1)], axis=1)
    mean = solve_writer(A, K_mean, D, weights(2))
    close(context.update @ K, T)
    close(context.update @ K_mean, D)
    assert context.diagnostics["energy_A"] >= mean.diagnostics["energy_A"] - 1e-10
    record("context_feasible_energy_ge_mean_same_keys_metric", {"context_energy": context.diagnostics["energy_A"], "mean_energy": mean.diagnostics["energy_A"], "ratio": context.diagnostics["energy_A"] / mean.diagnostics["energy_A"]})


def test_z_conditional_context_exact_preserves_direct_allocation_shares():
    # Two owners have heterogeneous context counts; owner 1 plans no edit at
    # any layer. Layer 1 is zero for both owners, yet all constraints remain.
    owners = np.array([0, 0, 0, 1, 1])
    row_weights = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 4, 1 / 4])
    dimensions = [(7, 3), (6, 2), (8, 4)]  # (key width, output width)
    anchors = np.array([[2.0, 1.7], [3.0, 2.4], [4.0, 3.1]])
    plan_magnitudes = []
    realized_magnitudes = []
    layer_residuals = []
    for layer, (n, d) in enumerate(dimensions):
        K = RNG.normal(size=(n, len(owners)))
        D = np.zeros((d, 2))
        if layer != 1:
            D[:, 0] = RNG.normal(size=d)
        T = D[:, owners]
        result = solve_writer(spd(n), K, T, row_weights)
        assert result.diagnostics["rank"] == len(owners)
        assert result.diagnostics["status"] == "COMPATIBLE_EXACT_WITHIN_TOLERANCE"
        realized = result.update @ K
        close(realized, T)
        plan_magnitudes.append(np.linalg.norm(D, axis=0) / anchors[layer])
        realized_magnitudes.append(np.linalg.norm(realized, axis=0) / anchors[layer, owners])
        layer_residuals.append(result.diagnostics["weighted_residual_norm"])
    plan_magnitudes = np.stack(plan_magnitudes)
    realized_magnitudes = np.stack(realized_magnitudes)
    close(realized_magnitudes, plan_magnitudes[:, owners])
    planned_total = plan_magnitudes.sum(axis=0)
    assert planned_total[0] > 0 and planned_total[1] == 0
    planned_share = plan_magnitudes[:, 0] / planned_total[0]
    owner0_columns = owners == 0
    direct_share = realized_magnitudes[:, owner0_columns] / realized_magnitudes[:, owner0_columns].sum(axis=0, keepdims=True)
    close(direct_share, np.broadcast_to(planned_share[:, None], direct_share.shape))
    close(realized_magnitudes[:, owners == 1], np.zeros((3, 2)))
    assert planned_share[1] == 0 and np.all(direct_share[1] == 0)
    # Never normalize tiny floating-point leakage to invent an allocation for
    # a zero-budget owner. Its share is null, with an explicit semantic status.
    records = [
        {"owner": 0, "status": "PLANNED_EDIT", "plan_share": planned_share.tolist(), "context_direct_shares": direct_share.T.tolist()},
        {"owner": 1, "status": "NO_PLANNED_EDIT", "plan_share": None, "context_direct_shares": None},
    ]
    record("conditional_context_exact_direct_share_matches_plan", {
        "layers": [{"key_width": n, "output_width": d} for n, d in dimensions],
        "context_counts": [3, 2],
        "same_entry_anchors": anchors.tolist(),
        "layer_weighted_residuals": layer_residuals,
        "max_share_error": float(np.max(np.abs(direct_share - planned_share[:, None]))),
        "zero_owner_max_direct_relative_magnitude": float(np.max(realized_magnitudes[:, owners == 1])),
        "owners": records,
        "guarantee_scope": "Per-context direct U_l k_lrc norm shares under exact realization and identical entry anchors only; not cumulative hidden displacement, task contribution, virtual trajectory equality, or locality.",
    })


def test_metric_symmetrization_and_whitening_energy():
    A = spd(5)
    skew = RNG.normal(size=(5, 5))
    skew = 0.01 * (skew - skew.T)
    K, T = RNG.normal(size=(5, 3)), RNG.normal(size=(2, 3))
    symmetric = solve_writer(A, K, T, weights(3))
    input_skew = solve_writer(A + skew, K, T, weights(3))
    close(symmetric.update, input_skew.update)
    L = np.linalg.cholesky(A)
    close(symmetric.diagnostics["energy_A"], np.sum((symmetric.update @ L) ** 2))
    assert input_skew.diagnostics["A_skew_frobenius"] > 0
    record("metric_symmetrization_whitening_energy", {"skew_recorded": input_skew.diagnostics["A_skew_frobenius"], "update_difference": float(np.linalg.norm(input_skew.update - symmetric.update))})


def test_zero_layer_and_rank_zero():
    zero_layer = solve_writer(spd(4), RNG.normal(size=(4, 3)), np.zeros((2, 3)), weights(3))
    close(zero_layer.update, np.zeros((2, 4)))
    rank_zero_exact = solve_writer(np.eye(3), np.zeros((3, 2)), np.zeros((1, 2)), weights(2))
    rank_zero_conflict = solve_writer(np.eye(3), np.zeros((3, 2)), np.ones((1, 2)), weights(2))
    assert rank_zero_exact.diagnostics["compatible_with_retained_subspace"]
    assert not rank_zero_conflict.diagnostics["compatible_with_retained_subspace"]
    assert rank_zero_conflict.diagnostics["rank"] == 0
    close(rank_zero_conflict.update, np.zeros((1, 3)))
    record("zero_D_layer_and_zero_rank", {"zero_layer_energy": zero_layer.diagnostics["energy_A"], "zero_rank_exact_status": rank_zero_exact.diagnostics["status"], "zero_rank_nonzero_status": rank_zero_conflict.diagnostics["status"], "zero_rank_nonzero_residual": rank_zero_conflict.diagnostics["weighted_residual_norm"]})


def test_near_singular_truncation_is_not_exact():
    K = np.diag([1.0, 1e-17])
    T = np.array([[1.0, 1.0]])
    result = solve_writer(np.eye(2), K, T, weights(2))
    assert result.diagnostics["rank"] == 1
    assert result.diagnostics["numerical_truncation"]
    assert not result.diagnostics["compatible_with_retained_subspace"]
    close(result.update @ K, [[1.0, 0.0]])
    assert result.diagnostics["retained_LS_orthogonality_norm"] < 1e-12
    record("near_singular_explicit_numeric_truncation_status", result.diagnostics)


def test_overdetermined_LS_retained_orthogonality():
    A, K, T = spd(3), RNG.normal(size=(3, 7)), RNG.normal(size=(2, 7))
    w = np.arange(1, 8, dtype=np.float64)
    w /= w.sum()
    result = solve_writer(A, K, T, w)
    assert result.diagnostics["rank"] == 3
    assert not result.diagnostics["compatible_with_retained_subspace"]
    assert result.diagnostics["retained_LS_orthogonality_norm"] < 1e-10
    # Full row rank X: unweighted U-space stationarity also vanishes.
    close(((result.update @ K - T) * w[None, :]) @ K.T, np.zeros((2, 3)))
    record("overdetermined_LS_orthogonality", result.diagnostics)


def test_retained_ill_conditioning_cannot_be_labeled_numerically_exact():
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    K = rotation @ np.diag([1.0, 1e-12]) @ rotation.T
    result = solve_writer(np.eye(2), K, np.array([[1.0, 3.0]]), weights(2))
    assert result.diagnostics["rank"] == 2
    assert result.diagnostics["compatible_with_retained_subspace"]
    assert not result.diagnostics["numerical_projection_verified"]
    assert result.diagnostics["status"] == "NUMERICAL_REALIZATION_FAILURE"
    assert result.diagnostics["weighted_residual_norm"] > 1e-8
    record("compatible_geometry_but_retained_ill_conditioning_failure", result.diagnostics)


def test_fp32_applied_update_is_separately_reported():
    K = np.eye(2)
    T = np.array([[1.0, 1.0]])
    exact = solve_writer(np.eye(2), K, T, weights(2))
    large = applied_fp32_diagnostics(np.full((1, 2), 1e8), exact, K, T)
    close(large["ideal_equality_error"], 0)
    assert large["applied_equality_error"] > 1
    assert large["unchanged_weight_fraction"] == 1
    normal = applied_fp32_diagnostics(np.full((1, 2), 0.013), exact, K, T)
    assert normal["applied_equality_error"] < 1e-6
    record("FP32_ideal_cast_applied_distinction", {"large_base_weight_counterexample": large, "small_base_weight_example": normal})


def test_arbitrary_batch_tail_shape_and_constraint_microbatch():
    outputs = []
    for B in (1, 3, 4, 7):
        n = B + 2
        A, K, T = spd(n), RNG.normal(size=(n, B)), RNG.normal(size=(3, B))
        result = solve_writer(A, K, T, weights(B))
        close(result.update @ K, T)
        # Capture microbatches reassemble one logical constraint matrix. They
        # never become separate writes or separate optimizations.
        assembled = np.concatenate([K[:, start:start + 2] for start in range(0, B, 2)], axis=1)
        close(assembled, K)
        repacked = solve_writer(A, assembled, T, weights(B))
        close(result.update, repacked.update)
        outputs.append({"B": B, "key_width": n, "error": result.diagnostics["weighted_residual_norm"]})
    counts = [min(4, 11 - start) for start in range(0, 11, 4)]
    assert counts == [4, 4, 3] and sum(counts) == 11
    record("arbitrary_B_and_tail_with_one_joint_solve", {"shapes": outputs, "N11_B4_tail_schedule": counts})


def test_no_jitter_or_weight_sanitizing_fallback():
    rejects = []
    invalid_inputs = [
        (np.diag([1.0, -1.0]), np.array([0.5, 0.5]), np.linalg.LinAlgError, "indefinite_A"),
        (np.eye(2), np.array([1.0, 0.0]), ValueError, "zero_weight"),
        (np.eye(2), np.array([1.0, 1.0]), ValueError, "unnormalized_weights"),
    ]
    for A, w, exception, name in invalid_inputs:
        try:
            solve_writer(A, np.eye(2), np.ones((1, 2)), w)
        except exception:
            rejects.append(name)
        else:
            raise AssertionError(f"Expected explicit rejection of {name}")
    record("invalid_inputs_raise_without_jitter_or_fallback", {"explicit_rejections": rejects})


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as error:
            failures.append({"test": test.__name__, "status": "FAIL", "error": repr(error)})
    audit = {
        "scope": "CPU FP64 numerical reference only",
        "seed": 20261005,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command": "uv run --no-project --with numpy==2.5.3 python test_ref.py",
        "tests_passed": len(RECORDS),
        "tests_failed": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("writer.py", "test_ref.py")},
        "tests": RECORDS + failures,
        "limitations": [
            "CPU only; no model or pilot speed qualification.",
            "Exact means retained row-space compatibility and reported numerical tolerance; numeric rank truncation is always recorded.",
            "Mean or context exactness does not prove virtual trajectory equality or locality preservation.",
            "The FP32 application diagnostic excludes model activation evaluation rounding.",
            "Context constraints may cost more energy and may be inconsistent; the LS result must not be labeled exact.",
            "KL injection targets are owner D; history support stays native rewrite-only with adapter-defined nested mean.",
        ],
    }
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": audit["status"], "tests_passed": audit["tests_passed"], "tests_failed": audit["tests_failed"], "audit": str(HERE / "audit.json"), "failures": failures}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
