"""Executable mathematical qualifications; no model/locality claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from writer import ridge, shared_requested_budget_bound, vjp

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261005)
RECORDS = []


def record(name, **measurements):
    RECORDS.append({"test": name, "status": "PASS", "measurements": measurements})


def close(a, b, rtol=2e-10, atol=2e-11):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def spd(n):
    V = RNG.normal(size=(n, n))
    return V @ V.T + 0.8 * np.eye(n)


def finite_gradient(function, value, eps=2e-6):
    result = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        step = eps * max(1.0, abs(float(value[index])))
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def test_dense_woodbury():
    errors = []
    for n, d, count in [(7, 3, 2), (3, 4, 7), (4, 2, 1), (5, 2, 5)]:
        A, K, R = spd(n), RNG.normal(size=(n, count)), RNG.normal(size=(d, count))
        if count == 5:
            K[:, 1] = K[:, 0]
            K[:, 2] = 0
        dense, small = ridge(A, K, R), ridge(A, K, R, woodbury=True)
        close(dense.update, small.update)
        close(dense.P, small.P)
        close(dense.update @ K, R @ dense.M)
        errors.append(float(np.linalg.norm(dense.update - small.update)))
    record("dense_woodbury_and_batch_realization", update_errors=errors,
           max_update_error=max(errors), includes_batch_wider_than_key_dimension=True)


def test_R_and_K_adjoint():
    A, K, R = spd(4), RNG.normal(size=(4, 3)), RNG.normal(size=(2, 3))
    X, target = RNG.normal(size=(4, 5)), RNG.normal(size=(2, 5))
    result = ridge(A, K, R)
    error = result.update @ X - target
    grad_U = error @ X.T + 0.17 * result.update
    grad_R, grad_K = vjp(result, grad_U)

    def objective(r, k):
        U = ridge(A, k, r).update
        return 0.5 * float(np.sum((U @ X - target) ** 2) + 0.17 * np.sum(U * U))

    fd_R = finite_gradient(lambda x: objective(x, K), R)
    fd_K = finite_gradient(lambda x: objective(R, x), K)
    close(grad_R, fd_R, rtol=3e-6, atol=3e-8)
    close(grad_K, fd_K, rtol=3e-6, atol=3e-8)
    record("full_native_ridge_adjoint_R_and_K",
           R_max_abs_error=float(np.max(np.abs(grad_R - fd_R))),
           K_max_abs_error=float(np.max(np.abs(grad_K - fd_K))))


def test_energy_identity_bound():
    ratios = []
    for n, d, count in [(4, 3, 2), (3, 5, 6), (8, 2, 4)]:
        A, K, R = spd(n), RNG.normal(size=(n, count)), RNG.normal(size=(d, count))
        solved = ridge(A, K, R)
        identity = float(np.trace(R @ (solved.M - solved.M @ solved.M) @ R.T))
        bound = float(np.sum(R * R) / 4)
        close(solved.energy, identity)
        assert solved.energy <= bound + 1e-11
        ratios.append(solved.energy / bound)
    scalar = ridge(np.ones((1, 1)), np.ones((1, 1)), np.array([[3.0]]))
    close(scalar.energy, 9 / 4)
    record("native_energy_identity_and_sharp_quarter_bound", Q_over_quarter_R_sq=ratios,
           equality_example_Q=scalar.energy, equality_example_quarter_R_sq=2.25)


def test_shared_budget_different_dimensions():
    count, radius = 5, np.array([0.75, 0.6, 0.5, 0.0, 0.2])
    fractions = RNG.uniform(0.0, 1.0, size=(3, count))
    fractions /= np.sum(fractions, axis=0)
    residuals, anchors, total_Q = [], [], 0.0
    for index, (n, d) in enumerate([(6, 3), (4, 7), (9, 2)]):
        a = RNG.uniform(0.5, 3.0, size=count)
        directions = RNG.normal(size=(d, count))
        directions /= np.linalg.norm(directions, axis=0)
        R = directions * (a * radius * fractions[index])[None, :]
        A, K = spd(n), RNG.normal(size=(n, count))
        total_Q += ridge(A, K, R).energy
        residuals.append(R)
        anchors.append(a)
    measured = shared_requested_budget_bound(residuals, anchors, radius)
    close(measured["usage_per_request"], radius)
    assert total_Q <= measured["quarter_residual_frobenius_sq"] + 1e-11
    assert measured["quarter_residual_frobenius_sq"] <= measured["shared_budget_energy_bound"] + 1e-11
    record("shared_requested_budget_total_energy_bound", total_Q=total_Q, **measured,
           layer_input_dimensions=[6, 4, 9], layer_output_dimensions=[3, 7, 2])


def test_scalar_contraction_is_independent_of_R():
    A, K = np.array([[2.0]]), np.array([[3.0]])
    capacity, expected = 4.5, 4.5 / 5.5
    ratios = []
    for r in [-20.0, -0.1, 0.2, 30.0]:
        actual = (ridge(A, K, np.array([[r]])).update @ K).item()
        ratios.append(actual / r)
    close(ratios, expected)
    record("scalar_contraction_independent_of_target_generation", capacity=capacity,
           realized_over_requested=ratios)


def test_inherited_tracking_creates_unplanned_R():
    # First local delta is 2.0; ridge applies only 1.0. Identity transport
    # carries the gap to layer 2, where the planned local delta is zero.
    D1, D2 = 2.0, 0.0
    actual1 = (ridge(np.ones((1, 1)), np.ones((1, 1)), np.array([[D1]])).update).item()
    z2_virtual, h2_actual = D1 + D2, actual1
    tracking_R2 = z2_virtual - h2_actual
    tracking_U2 = ridge(np.ones((1, 1)), np.ones((1, 1)), np.array([[tracking_R2]])).update.item()
    direct_U2 = ridge(np.ones((1, 1)), np.ones((1, 1)), np.array([[D2]])).update.item()
    close([tracking_R2, tracking_U2, direct_U2], [1.0, 0.5, 0.0])
    record("tracking_adds_inherited_R_on_zero_planned_delta", planned_D2=D2,
           tracking_R2=tracking_R2, tracking_U2=tracking_U2, direct_U2=direct_U2,
           direct_final_gap=z2_virtual - actual1)


def test_zero_owner_leakage_and_request_freeze():
    # Holding one coefficient fixed does not hold that subject's output fixed:
    # native batch M is generally dense even when actual K is held constant.
    A, K = np.eye(2), np.array([[1.0, 1.0], [0.0, 1.0]])
    before = ridge(A, K, np.array([[0.0, 1.0]]))
    after = ridge(A, K, np.array([[0.0, 2.0]]))
    y0, y1 = before.update @ K, after.update @ K
    assert y0[0, 0] > 0 and y1[0, 0] > y0[0, 0]
    close(after.R[:, 0], before.R[:, 0])
    record("direct_zero_owner_and_frozen_request_still_receive_crosstalk",
           M=before.M.tolist(), unchanged_request_coefficient=float(before.R[0, 0]),
           before_actual=y0.tolist(), after_actual=y1.tolist(),
           frozen_request_output_change=float(y1[0, 0] - y0[0, 0]))


def test_actual_upper_key_chain():
    A1, K1, R1 = spd(3), RNG.normal(size=(3, 2)), RNG.normal(size=(2, 2))
    A2, K20, R2 = spd(4), RNG.normal(size=(4, 2)), RNG.normal(size=(3, 2))
    C, T, target = RNG.normal(size=(4, 2)), RNG.normal(size=(3, 2)), RNG.normal(size=(3, 2))

    def forward(r1, r2):
        first = ridge(A1, K1, r1)
        v1 = first.update @ K1
        actual_K2 = K20 + C @ v1
        second = ridge(A2, actual_K2, r2)
        error = T @ v1 + second.update @ actual_K2 - target
        objective = 0.5 * float(np.sum(error * error) + 0.13 * np.sum(second.update ** 2))
        return objective, first, second, error

    _, first, second, error = forward(R1, R2)
    grad_U2 = error @ second.K.T + 0.13 * second.update
    grad_R2, grad_K2_writer = vjp(second, grad_U2)
    grad_K2 = grad_K2_writer + second.update.T @ error
    grad_v1 = T.T @ error + C.T @ grad_K2
    grad_R1, _ = vjp(first, grad_v1 @ K1.T)
    detached_R1, _ = vjp(first, (T.T @ error) @ K1.T)
    fd_R1 = finite_gradient(lambda x: forward(x, R2)[0], R1)
    fd_R2 = finite_gradient(lambda x: forward(R1, x)[0], R2)
    close(grad_R1, fd_R1, rtol=5e-6, atol=4e-8)
    close(grad_R2, fd_R2, rtol=5e-6, atol=4e-8)
    detached_gap = float(np.linalg.norm(detached_R1 - fd_R1))
    assert detached_gap > 1e-3
    record("causal_actual_upper_key_adjoint_and_detach_counterexample",
           R1_max_abs_error=float(np.max(np.abs(grad_R1 - fd_R1))),
           R2_max_abs_error=float(np.max(np.abs(grad_R2 - fd_R2))),
           detached_key_R1_gradient_error_norm=detached_gap,
           upper_key_writer_cotangent_norm=float(np.linalg.norm(grad_K2_writer)))


def main():
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    audit = {
        "scope": "v14 native-ridge writer-aware planning algebra only",
        "status": "PASS", "passed": len(RECORDS), "failed": 0,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU"},
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ["writer.py", "test_ref.py", "requirements.txt"]},
        "tests": RECORDS,
        "limitations": [
            "Synthetic FP64 arrays only; no actual language-model forward, GPU or production integration.",
            "No claim about locality, paraphrase, accuracy, end-to-end speed or convergence.",
            "Fixed SPD entry metric A in each candidate; A/history differentiation is not tested.",
            "Toy causal chain checks upper-key dependency, not architecture-specific attention or all-token effects.",
            "Energy bound constrains ideal A-energy for native mean-key ridge, not semantic damage or effective FP32 realization.",
            "Direct requested budget eliminates inherited RHS additions, but does not freeze zero-owner outputs or remove cross-talk.",
            "Writer-aware planning can trade requested budget across directions; it cannot remove native contraction at fixed keys and metric."
        ]
    }
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "passed": len(RECORDS), "failed": 0,
                      "audit": str(HERE / "audit.json")}))


if __name__ == "__main__":
    main()
