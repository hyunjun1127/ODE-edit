from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from project.run_scripts.ode_bf.p2r6_certified_convex_qp import (
    ConvexQuadraticTie,
    P2R6_QP_EXTERNAL_TOLERANCE,
    solve_certified_semantic_region_qp,
)


def _mass(layer_count: int = 5, request_count: int = 10) -> np.ndarray:
    value = np.zeros((request_count, layer_count * request_count), dtype=np.float64)
    for request in range(request_count):
        value[request, request::request_count] = 1.0
    return value


def _dense_fixture() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(50, 50))
    objective = raw.T @ raw / 50.0 + np.eye(50) * 0.25
    cross = rng.normal(scale=0.02, size=50)
    response = rng.normal(scale=0.01, size=(10, 50))
    for request in range(10):
        response[request, request] += 1.0
    lower = np.full(10, 0.2, dtype=np.float64)
    start = np.zeros(50, dtype=np.float64)
    start[:10] = 0.2
    return start, objective, cross, response, lower, _mass()


def _assert_certificate(receipt: dict[str, object]) -> None:
    assert receipt["certificate_pass"] is True
    assert max(
        float(receipt["r_pri"]),
        float(receipt["r_dual"]),
        float(receipt["r_stat"]),
        float(receipt["r_comp"]),
    ) <= P2R6_QP_EXTERNAL_TOLERANCE
    assert len(receipt["ordered_slacks"]) == len(receipt["dual_multipliers"])
    assert receipt["objective_psd_certificate"] is True


def test_dense_50d_full_inequality_certificate() -> None:
    result = solve_certified_semantic_region_qp(
        *_dense_fixture(), stage="DENSE50D"
    )
    _assert_certificate(dict(result.receipt))
    assert result.value.shape == (50,)


def test_negative_cross_and_constraint_release_are_certified() -> None:
    start = np.asarray([1.0, 0.0], dtype=np.float64)
    result = solve_certified_semantic_region_qp(
        start,
        np.eye(2, dtype=np.float64),
        np.asarray([-0.8, -0.2], dtype=np.float64),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        np.asarray([0.1], dtype=np.float64),
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        stage="NEGATIVE_CROSS_CONSTRAINT_RELEASE",
    )
    _assert_certificate(dict(result.receipt))
    assert result.value[1] > 0.1
    assert result.receipt["constraint_release_capability"] == (
        "ALL_INEQUALITIES_REMAIN_PRIMAL_DUAL_VARIABLES"
    )


def test_rank_deficient_semantic_face_is_certified() -> None:
    start, objective, cross, response, lower, mass = _dense_fixture()
    response[1] = response[0]
    lower[1] = lower[0]
    result = solve_certified_semantic_region_qp(
        start,
        objective,
        cross,
        response,
        lower,
        mass,
        stage="RANK_DEFICIENT_SEMANTIC_FACE",
    )
    _assert_certificate(dict(result.receipt))
    active = np.asarray(result.receipt["ordered_slacks"]) <= 1.0e-8
    assert np.linalg.matrix_rank(response) < response.shape[0]
    assert int(np.sum(active)) > 0


def test_structural_p_quadratic_tie_is_certified() -> None:
    start = np.asarray([0.5, 0.5], dtype=np.float64)
    structural = np.diag([1.0, 4.0]).astype(np.float64)
    structural_cross = np.zeros(2, dtype=np.float64)
    p_star = float(start @ structural @ start)
    result = solve_certified_semantic_region_qp(
        start,
        np.diag([4.0, 1.0]).astype(np.float64),
        np.asarray([-0.2, -0.8], dtype=np.float64),
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        np.asarray([0.5], dtype=np.float64),
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        stage="P_TIE",
        tie=ConvexQuadraticTie(
            structural,
            structural_cross,
            p_star + 1.0e-6,
        ),
    )
    _assert_certificate(dict(result.receipt))
    assert result.receipt["tie_enabled"] is True
    assert result.receipt["ordered_constraint_names"][-1] == "structural_p_tie"


def test_exact_qwen_capture_new_solver_strict_pass() -> None:
    raw = os.environ.get("P2R6_QWEN_REPLAY_CAPSULE")
    if raw is None:
        pytest.skip("private Qwen replay capsule path is supplied only to the pre-GPU gate")
    root = Path(raw)
    arrays = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in ("S", "b", "Q_C", "c_C", "alpha_start", "mass_matrix")
    }
    result = solve_certified_semantic_region_qp(
        arrays["alpha_start"],
        arrays["Q_C"],
        arrays["c_C"],
        arrays["S"],
        arrays["b"],
        arrays["mass_matrix"],
        stage="CAPTURED_QWEN_AS_OUTER05_AR_CAP",
    )
    _assert_certificate(dict(result.receipt))
    assert result.receipt["r_pri"] <= 1.0e-8
    assert result.receipt["r_stat"] <= 1.0e-8

