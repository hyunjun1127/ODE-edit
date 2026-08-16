from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from project.run_scripts.ode_bf.p2r6_certified_convex_qp import (
    ConvexQuadraticTie,
    P2R6CertifiedQPError,
    P2R6_QP_EXTERNAL_TOLERANCE,
    solve_certified_semantic_region_qp,
)
from project.run_scripts.ode_bf.p2r6_semantic_region_controller import (
    _semantic_region_optimum,
)
from project.run_scripts.session05_ode_bf_p2r6_red_r2_build_amended_replay import (
    FIXED_MEMBERS,
    build as build_amended_replay,
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
    dual = np.asarray(receipt["dual_multipliers"], dtype=np.float64)
    raw_slack = np.asarray(receipt["ordered_slacks"], dtype=np.float64)
    assert np.max(np.abs(dual * raw_slack)) == pytest.approx(
        receipt["r_comp"], rel=0.0, abs=1.0e-18
    )
    assert receipt["backend_constraint_envelope"] == 0.0


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


def test_legacy_qwen_capture_is_exact_infeasible() -> None:
    raw = os.environ.get("P2R6_QWEN_LEGACY_REPLAY_CAPSULE")
    if raw is None:
        pytest.skip("private legacy Qwen replay path is supplied only to the pre-GPU gate")
    root = Path(raw)
    arrays = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in ("S", "b", "Q_C", "c_C", "alpha_start", "mass_matrix")
    }
    exact = linprog(
        np.zeros(50, dtype=np.float64),
        A_ub=np.concatenate((arrays["mass_matrix"], -arrays["S"]), axis=0),
        b_ub=np.concatenate((np.ones(10), -arrays["b"])),
        bounds=[(0.0, None)] * 50,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
            "ipm_optimality_tolerance": 1.0e-12,
        },
    )
    assert exact.status == 2
    minimax = linprog(
        np.concatenate((np.zeros(50), np.ones(1))),
        A_ub=np.concatenate(
            (
                np.concatenate(
                    (arrays["mass_matrix"], np.zeros((10, 1))), axis=1
                ),
                np.concatenate((-arrays["S"], -np.ones((10, 1))), axis=1),
            ),
            axis=0,
        ),
        b_ub=np.concatenate((np.ones(10), -arrays["b"])),
        bounds=[(0.0, None)] * 51,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
            "ipm_optimality_tolerance": 1.0e-12,
        },
    )
    assert minimax.success
    assert minimax.fun == pytest.approx(9.396528963605145e-09, abs=1.0e-15)
    with pytest.raises(P2R6CertifiedQPError):
        solve_certified_semantic_region_qp(
            arrays["alpha_start"],
            arrays["Q_C"],
            arrays["c_C"],
            arrays["S"],
            arrays["b"],
            arrays["mass_matrix"],
            stage="LEGACY_QWEN_AS_OUTER05_AR_CAP",
        )


def test_b2_amended_qwen_replay_new_solver_strict_pass() -> None:
    raw = os.environ.get("P2R6_QWEN_LEGACY_REPLAY_CAPSULE")
    if raw is None:
        pytest.skip("private legacy Qwen replay path is supplied only to the pre-GPU gate")
    root = Path(raw)
    arrays = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in ("S", "d", "b", "Q_C", "c_C", "mass_matrix")
    }
    alpha_start, _scale, b_new, xi_new, e1_receipt = _semantic_region_optimum(
        arrays["S"], arrays["d"], arrays["d"], scale_policy="CURRENT_DEFICIT"
    )
    assert xi_new == pytest.approx(0.9816568567493456, abs=1.0e-15)
    assert np.max(np.abs(b_new - arrays["b"])) == pytest.approx(
        4.068674066548539e-07, abs=1.0e-15
    )
    assert np.max(b_new - arrays["S"] @ alpha_start) <= 1.0e-8
    assert np.max(arrays["mass_matrix"] @ alpha_start - 1.0) <= 1.0e-8
    assert e1_receipt["e1_semantic_nonnegative_constraint_enabled"] is True
    result = solve_certified_semantic_region_qp(
        alpha_start,
        arrays["Q_C"],
        arrays["c_C"],
        arrays["S"],
        b_new,
        arrays["mass_matrix"],
        stage="B2_AMENDED_QWEN_AS_OUTER05_AR_CAP",
    )
    _assert_certificate(dict(result.receipt))
    assert result.receipt["r_pri"] <= P2R6_QP_EXTERNAL_TOLERANCE
    assert result.receipt["r_stat"] <= P2R6_QP_EXTERNAL_TOLERANCE


def test_b2_amended_replay_builder_preserves_fixed_members(
    tmp_path: Path,
) -> None:
    raw = os.environ.get("P2R6_QWEN_LEGACY_REPLAY_CAPSULE")
    if raw is None:
        pytest.skip("private legacy Qwen replay path is supplied only to the pre-GPU gate")
    legacy = Path(raw)
    destination = tmp_path / "amended"
    receipt = build_amended_replay(legacy, destination, "test-source-head")
    assert receipt["status"] == "B2_AMENDED_REPLAY_EXACT_FEASIBLE"
    capsule = json.loads((destination / "capsule.json").read_text())
    assert capsule["fixed_member_byte_identity_all"] is True
    assert capsule["feasible_set_rhs_envelope"] == 0.0
    for name in FIXED_MEMBERS:
        assert (destination / f"{name}.npy").read_bytes() == (
            legacy / f"{name}.npy"
        ).read_bytes()
