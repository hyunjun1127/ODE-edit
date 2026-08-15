from __future__ import annotations

from pathlib import Path
import inspect
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from project.run_scripts.ode_bf import p2r6_semantic_region_controller as controller

from project.run_scripts import session05_ode_bf_p2r6_pilot_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p2r5_sdrt_writer import (
    SDRTQuadratics,
    pooled_nonnegative_realization_calibration,
    solve_sdrt_routing,
)
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.p2r6_pilot_panel import (
    LOCK_FILE,
    PARENT,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p2r6_pilot_runtime import (
    P2R6_RUNTIME_POLICY,
    p2r6_phase_arms,
)
from project.run_scripts.ode_bf.p2r6_semantic_region_controller import (
    P2R6_ARMS,
    P2R6_CAP_ARMS,
    P2R6_E1_XI_AUTHORITY,
    P2R6_HIGHS_INTERNAL_TOLERANCE,
    _solve_region_quadratic,
    _semantic_region_optimum,
    p2r6_forbidden_influence_receipt,
    solve_p2r6_routing,
    solve_p2r6_shadow_panel,
)


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "project/run_scripts/ode_bf"


def _response() -> torch.Tensor:
    response = torch.zeros((10, 50), dtype=torch.float64)
    for request in range(10):
        response[request, request] = 2.0
        response[request, 10 + request] = 1.0
        response[request, 20 + request] = 0.5
    return response


def _quadratics() -> SDRTQuadratics:
    capacity = torch.eye(50, dtype=torch.float64)
    structural = torch.eye(50, dtype=torch.float64)
    for request in range(10):
        capacity[request, request] = 10.0
        capacity[10 + request, 10 + request] = 1.0
        structural[request, request] = 0.1
        structural[10 + request, 10 + request] = 10.0
    return SDRTQuadratics(
        capacity_gram=capacity,
        capacity_cross=torch.zeros(50, dtype=torch.float64),
        prior_capacity=0.0,
        structural_p_gram=structural,
        structural_p_cross=torch.zeros(50, dtype=torch.float64),
        prior_structural_p=0.0,
        identity_sha256="fixture",
    )


def test_a0_is_exact_p2r5_cap_and_aeta_removes_eta_decision_only() -> None:
    response = _response()
    deficit = torch.ones(10, dtype=torch.float64)
    entry = deficit.clone()
    predicted = torch.arange(1, 11, dtype=torch.float64)
    calibration = pooled_nonnegative_realization_calibration(
        [predicted], [predicted * 0.4]
    )
    a0 = solve_p2r6_routing(
        response, deficit, entry, calibration, _quadratics(), arm="A0-CAP"
    )
    legacy = solve_sdrt_routing(
        response, deficit, calibration, _quadratics(), arm="SDRT-CAP"
    )
    assert torch.equal(a0.allocation, legacy.allocation)
    assert a0.eta_decision == pytest.approx(0.4)
    assert a0.eta_decision_influence_count == 1
    aeta = solve_p2r6_routing(
        response, deficit, entry, calibration, _quadratics(), arm="AETA-CAP"
    )
    assert aeta.eta_observation == pytest.approx(0.4)
    assert aeta.eta_decision == 1.0
    assert aeta.eta_decision_influence_count == 0
    assert aeta.e2_enabled and aeta.exact_response_face_enabled


@pytest.mark.parametrize("arm", ("AR-CAP", "AS-CAP", "AR-STRUCTP", "AS-STRUCTP"))
def test_region_arms_use_raw_s_inequality_and_certify_mass(arm: str) -> None:
    response = _response()
    deficit = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    entry = torch.linspace(1.0, 0.1, 10, dtype=torch.float64)
    route = solve_p2r6_routing(
        response,
        deficit,
        entry,
        pooled_nonnegative_realization_calibration([], []),
        _quadratics(),
        arm=arm,
    )
    assert route.eta_decision == 1.0
    assert route.eta_decision_influence_count == 0
    assert not route.e2_enabled
    assert not route.exact_response_face_enabled
    assert torch.min(route.allocation) >= -1.0e-8
    assert torch.max(torch.sum(route.allocation, dim=0)) <= 1.0 + 1.0e-8
    assert torch.min(route.raw_predicted_response - route.semantic_lower_bound) >= -1.0e-8
    assert route.semantic_region_max_violation <= 1.0e-8
    assert all(item["certificate_pass"] for item in route.solver_receipts)


@pytest.mark.parametrize("controller", ("AR", "AS"))
def test_structp_minimizes_p_inside_same_semantic_region(controller: str) -> None:
    response = _response()
    deficit = torch.ones(10, dtype=torch.float64)
    entry = torch.ones(10, dtype=torch.float64)
    calibration = pooled_nonnegative_realization_calibration([], [])
    cap = solve_p2r6_routing(
        response, deficit, entry, calibration, _quadratics(), arm=f"{controller}-CAP"
    )
    structp = solve_p2r6_routing(
        response,
        deficit,
        entry,
        calibration,
        _quadratics(),
        arm=f"{controller}-STRUCTP",
    )
    assert torch.equal(cap.semantic_lower_bound, structp.semantic_lower_bound)
    assert structp.marginal_structural_p <= cap.marginal_structural_p + 1.0e-8
    assert torch.linalg.vector_norm(cap.allocation - structp.allocation) > 1.0e-6


def test_shadow_panel_is_same_state_model_free_and_has_all_four_arms() -> None:
    panel = solve_p2r6_shadow_panel(
        _response(),
        torch.ones(10, dtype=torch.float64),
        torch.ones(10, dtype=torch.float64),
        pooled_nonnegative_realization_calibration([], []),
        _quadratics(),
    )
    assert tuple(item.arm for item in panel.routes) == P2R6_CAP_ARMS
    assert (panel.model_forward_count, panel.model_backward_count, panel.materialization_count) == (0, 0, 0)
    assert len(panel.pairwise_allocation_l2) == 6
    assert panel.route("AR-CAP").arm == "AR-CAP"


def test_entry_anchored_e1_rows_are_ratio_normalized_for_dynamic_scale() -> None:
    response = _response()
    deficit = torch.logspace(-8, -2, 10, dtype=torch.float64)
    entry = torch.logspace(-4, 4, 10, dtype=torch.float64)
    route = solve_p2r6_routing(
        response,
        deficit,
        entry,
        pooled_nonnegative_realization_calibration([], []),
        _quadratics(),
        arm="AS-CAP",
    )
    e1 = route.solver_receipts[0]
    assert e1["constraint_row_normalization"] == "DIVIDE_BY_SEMANTIC_SCALE"
    assert e1["highs_internal_tolerance"] == P2R6_HIGHS_INTERNAL_TOLERANCE
    assert P2R6_HIGHS_INTERNAL_TOLERANCE < 1.0e-8
    assert e1["semantic_scale_min"] == pytest.approx(1.0e-4)
    assert e1["semantic_scale_max"] == pytest.approx(1.0e4)
    assert e1["e1_xi"] == e1["xi_recomputed"]
    assert e1["e1_xi_authority"] == P2R6_E1_XI_AUTHORITY
    assert e1["xi_solver_decision_influence_count"] == 0
    assert e1["certificate_pass"] is True
    assert route.semantic_region_max_violation <= 1.0e-8
    source = inspect.getsource(
        __import__(
            "project.run_scripts.ode_bf.p2r6_semantic_region_controller",
            fromlist=["_semantic_region_optimum"],
        )._semantic_region_optimum
    )
    assert source.count("response @ allocation") == 1
    assert "deficit - semantic_response" in source
    assert "lower - semantic_response" in source


def test_e1_backend_auxiliary_xi_is_observation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    original = controller.linprog

    def underreporting_linprog(*args: object, **kwargs: object) -> SimpleNamespace:
        solved = original(*args, **kwargs)
        shifted = np.asarray(solved.x, dtype=np.float64).copy()
        shifted[-1] = max(0.0, float(shifted[-1]) - 1.0e-4)
        return SimpleNamespace(
            success=solved.success,
            status=solved.status,
            message=solved.message,
            nit=getattr(solved, "nit", -1),
            x=shifted,
        )

    monkeypatch.setattr(controller, "linprog", underreporting_linprog)
    response = _response().numpy()
    deficit = np.linspace(3.0, 4.0, 10, dtype=np.float64)
    _alpha, _scale, _lower, xi, receipt = _semantic_region_optimum(
        response,
        deficit,
        deficit,
        scale_policy="CURRENT_DEFICIT",
    )
    assert receipt["xi_recertification_delta"] > 1.0e-8
    assert receipt["xi_solver_decision_influence_count"] == 0
    assert receipt["e1_xi_authority"] == P2R6_E1_XI_AUTHORITY
    assert xi == receipt["xi_recomputed"]
    assert receipt["certificate_pass"] is True


def test_e1_start_region_slack_is_observation_not_a_false_gate() -> None:
    response = _response().numpy()
    response[0, :] = -1.0e-6
    response[0, 0] = 0.0
    deficit = torch.ones(10, dtype=torch.float64).numpy()
    deficit[0] = 0.0
    entry = deficit.copy()
    _start, _scale, _lower, _xi, receipt = _semantic_region_optimum(
        response,
        deficit,
        entry,
        scale_policy="ENTRY_ANCHORED_DEFICIT",
    )
    assert receipt["certificate_pass"] is True
    assert receipt["ratio_objective_violation"] <= 1.0e-8
    assert receipt["semantic_region_violation"] > 0.0
    assert receipt["e1_start_in_semantic_region"] is (
        receipt["semantic_region_violation"] <= 1.0e-8
    )
    source = inspect.getsource(_semantic_region_optimum)
    assert "max(mass_violation, ratio_objective_violation, negative_violation)" in source
    assert "max(mass_violation, semantic_violation, negative_violation)" not in source


def test_quadratic_polish_expands_newly_active_mass_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boundary_crossing_minimize(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.asarray([0.5, 0.0], dtype=np.float64),
            success=False,
            status=8,
            message="fixture boundary crossing",
            nit=1,
            nfev=1,
            njev=1,
        )

    monkeypatch.setattr(controller, "minimize", boundary_crossing_minimize)
    selected, receipt = _solve_region_quadratic(
        np.asarray([0.5, 0.0], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.asarray([-2.0, 0.0], dtype=np.float64),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        np.asarray([0.1], dtype=np.float64),
        stage="ACTIVE_SET_EXPANSION_FIXTURE",
    )
    assert selected == pytest.approx(np.asarray([1.0, 0.0]))
    assert receipt["active_set_expansion_count"] == 1
    assert receipt["active_set_polish_round_count"] >= 2
    assert receipt["mass_violation"] <= 1.0e-8
    assert receipt["optimality"] <= 1.0e-8
    assert receipt["certificate_pass"] is True
    source = inspect.getsource(_solve_region_quadratic)
    assert "candidate < -P2R6_NUMERICAL_EPSILON" in source
    assert "c_mass < -P2R6_NUMERICAL_EPSILON" in source
    assert "c_semantic < -P2R6_NUMERICAL_EPSILON" in source


def test_forbidden_influence_and_runtime_policy_are_exact() -> None:
    receipt = p2r6_forbidden_influence_receipt()
    assert receipt["eta_decision_influence_count_for_aeta_ar_as"] == 0
    assert receipt["shadow_model_forward_count"] == 0
    assert receipt["shadow_model_backward_count"] == 0
    assert receipt["shadow_materialization_count"] == 0
    for key in (
        "semantic_debt_input_count",
        "explicit_lag_input_count",
        "remaining_horizon_division_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_count",
        "retry_count",
        "backtracking_count",
        "historical_h_influence_count",
        "sequential_state_count",
    ):
        assert receipt[key] == 0
    assert P2R6_RUNTIME_POLICY.allowed_arms == P2R6_ARMS
    assert P2R6_RUNTIME_POLICY.shadow_solver is solve_p2r6_shadow_panel


def test_phase_inventory_lock_and_dry_plans() -> None:
    lock, _ = load_and_validate_lock(PACKAGE / "locks" / LOCK_FILE)
    assert lock["eta_decision_influence_count_aeta_ar_as"] == 0
    assert p2r6_phase_arms("phase1") == P2R6_CAP_ARMS
    assert p2r6_phase_arms("phase2", "AR") == (
        "A0-CAP",
        "AR-CAP",
        "AR-STRUCTP",
    )
    phase1 = dry.build_plan(PARENT, phase="phase1")
    assert phase1["job_count"] == 2
    assert phase1["endpoint_attempt_count"] == 8
    assert phase1["array"] == "0-1%2"
    phase2 = dry.build_plan(PARENT, phase="phase2", selected_controller="AS")
    assert phase2["job_count"] == 4
    assert phase2["endpoint_attempt_count"] == 12
    assert phase2["array"] == "0-3%4"
    with pytest.raises(ODEBFContractError):
        p2r6_phase_arms("phase2", None)


def test_protected_target_response_materializer_bytes_and_thin_hook() -> None:
    protected = (
        "project/run_scripts/ode_bf/p2r1_rms_tangent_target.py",
        "project/run_scripts/ode_bf/p2r2_residual_transport_writer.py",
        "project/run_scripts/ode_bf/p2r4_phaseb_atomic_runtime.py",
        "project/run_scripts/ode_bf/atomic_runtime_optimization.py",
        "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    )
    for relative in protected:
        parent = subprocess.run(
            ["git", "show", f"{PARENT}:{relative}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert (ROOT / relative).read_bytes() == parent
    runtime = (PACKAGE / "p2r6_pilot_runtime.py").read_text()
    shared = (PACKAGE / "p2r5_stage_a_runtime.py").read_text()
    for symbol in (
        "p2r1_target_update",
        "measure_request_layer_response",
        "p2r2_waypoint_factors",
        "AcceptedPhysicalStateMaterializer",
        "_evaluate_frozen_state",
    ):
        assert f"def {symbol}(" not in runtime
        assert symbol in shared
    assert "run_p2_atomic_arm_case(" in runtime
    assert '"shadow_model_forward_count": 0' in shared
    assert shared.count("materializer.materialize(") == 1


def test_invalid_arm_and_nonfinite_fail_close() -> None:
    with pytest.raises(ODEBFContractError):
        solve_p2r6_routing(
            _response(),
            torch.ones(10, dtype=torch.float64),
            torch.ones(10, dtype=torch.float64),
            pooled_nonnegative_realization_calibration([], []),
            _quadratics(),
            arm="NEUTRAL",
        )
    broken = _response()
    broken[0, 0] = torch.nan
    with pytest.raises(ODEBFContractError):
        solve_p2r6_routing(
            broken,
            torch.ones(10, dtype=torch.float64),
            torch.ones(10, dtype=torch.float64),
            pooled_nonnegative_realization_calibration([], []),
            _quadratics(),
            arm="AR-CAP",
        )


def test_entrypoint_runtime_args_and_resource_contract_are_bound() -> None:
    params = inspect.signature(run_p1).parameters
    for name in (
        "p2r6_phase",
        "p2r6_case_index",
        "p2r6_selected_controller",
        "p2r6_attempt_suffix",
    ):
        assert name in params
    help_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "project.run_scripts.session05_ode_bf_p2r6_pilot",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--phase {phase1,phase2}" in help_result.stdout
    assert "--selected-controller {AR,AS}" in help_result.stdout
    sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p2r6_pilot.sbatch").read_text()
    submitter = (
        ROOT / "project/run_scripts/session05_ode_bf_submit_p2r6_pilot.py"
    ).read_text()
    for token in (
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=65000M",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodelist=devbox",
        'readonly EXPECTED_BRANCH="codex/p2r6-semantic-region-controller-pilot-v1"',
    ):
        assert token in sbatch
    assert "PROJECT_GPU_CAP = 4" in submitter
    assert '"RUNNING,CONFIGURING"' in submitter
    assert '"JobState=PENDING"' in submitter
    assert '"Reason=JobHeldUser"' in submitter
