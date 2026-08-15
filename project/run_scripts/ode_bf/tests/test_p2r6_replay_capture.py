from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p2r5_sdrt_writer import SDRTCalibration, SDRTQuadratics
from project.run_scripts.ode_bf.p2r5_stage_a_runtime import run_p2_atomic_arm_case
from project.run_scripts.ode_bf.p2r6_pilot_runtime import run_p2r6_pilot_case
from project.run_scripts.ode_bf.p2r6_replay_capture import (
    CAPTURE_OUTER_STEP,
    P2R6ReplayCaptureComplete,
    build_capture_hook,
    load_capsule,
)


def _quadratics() -> SDRTQuadratics:
    payload = {
        "capacity": "identity",
        "structural": "identity",
        "prior_capacity": 0.0,
        "prior_structural_p": 0.0,
    }
    return SDRTQuadratics(
        capacity_gram=torch.eye(50, dtype=torch.float64),
        capacity_cross=torch.zeros(50, dtype=torch.float64),
        prior_capacity=0.0,
        structural_p_gram=torch.eye(50, dtype=torch.float64),
        structural_p_cross=torch.zeros(50, dtype=torch.float64),
        prior_structural_p=0.0,
        identity_sha256=canonical_hash(payload),
    )


def _calibration() -> SDRTCalibration:
    payload = {
        "eta": 1.0,
        "prior_step_count": 5,
        "pair_count": 50,
        "numerator": 1.0,
        "denominator": 1.0,
        "status": "POOLED_NNLS_CERTIFIED",
    }
    return SDRTCalibration(1.0, 5, 50, 1.0, 1.0, payload["status"], canonical_hash(payload))


def test_capture_hook_is_outer5_only_and_seals_private_finite_arrays(tmp_path: Path) -> None:
    response = torch.zeros((10, 50), dtype=torch.float64)
    for request in range(10):
        response[request, request] = 1.0
    deficit = torch.ones(10, dtype=torch.float64)
    hook = build_capture_hook(
        instrumentation_source_head="a" * 40,
        instrumentation_source_tree="b" * 40,
    )
    common = {
        "alias": "qwen2.5-7b-inst",
        "selected_arm": "AS-CAP",
        "case_index": 1,
        "request_order_sha256": "c" * 64,
        "expected_w0_sha256": "d" * 64,
        "case_root": tmp_path / "case",
    }
    hook(
        response,
        deficit,
        deficit,
        _calibration(),
        _quadratics(),
        outer_step=CAPTURE_OUTER_STEP - 1,
        **common,
    )
    assert not (tmp_path / "case" / "private-replay").exists()
    with pytest.raises(P2R6ReplayCaptureComplete) as captured:
        hook(
            response,
            deficit,
            deficit,
            _calibration(),
            _quadratics(),
            outer_step=CAPTURE_OUTER_STEP,
            **common,
        )
    arrays, receipt = load_capsule(captured.value.capsule_root)
    assert set(arrays) == {
        "Q_C",
        "S",
        "alpha_start",
        "b",
        "c_C",
        "d",
        "mass_matrix",
        "semantic_scale",
    }
    assert all(bool(torch.isfinite(torch.from_numpy(item)).all()) for item in arrays.values())
    assert receipt["instrumentation_decision_influence_count"] == 0
    assert receipt["instrumentation_model_forward_count"] == 0
    assert receipt["instrumentation_model_backward_count"] == 0
    assert receipt["instrumentation_materialization_count"] == 0
    assert receipt["terminal_evaluator_access_count"] == 0
    assert receipt["intentional_stop_before_shadow_solve"] is True
    assert oct(os.stat(captured.value.capsule_root).st_mode & 0o777) == "0o700"
    for path in captured.value.capsule_root.iterdir():
        assert path.is_file() and not path.is_symlink()
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_capture_runtime_hook_precedes_shadow_and_intentional_stop_restores_w0() -> None:
    arm_source = inspect.getsource(run_p2_atomic_arm_case)
    hook_index = arm_source.index("runtime_policy.pre_shadow_hook(")
    shadow_index = arm_source.index("runtime_policy.shadow_solver(")
    materialize_index = arm_source.index("materializer.materialize(")
    assert hook_index < shadow_index < materialize_index
    pilot_source = inspect.getsource(run_p2r6_pilot_case)
    intentional_index = pilot_source.index("except P2R6ReplayCaptureComplete")
    restore_index = pilot_source.index("_restore_exact_w0(", intentional_index)
    terminal_index = pilot_source.index("P2R6_CAPTURE_A1_COMPLETE", restore_index)
    assert intentional_index < restore_index < terminal_index
    assert '"terminal_evaluator_access_count": 0' in pilot_source
    assert '"scientific_endpoint_count": 0' in pilot_source


def test_capture_source_has_no_red_equation_or_solver_repair() -> None:
    source = inspect.getsource(build_capture_hook)
    assert "_semantic_region_optimum" in source
    assert 'scale_policy="CURRENT_DEFICIT"' in source
    assert "_solve_region_quadratic" not in source
    assert "S alpha" not in source
    assert "tolerance" not in source.lower()
