from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from project.run_scripts import session05_ode_bf_p2r6_llama_qp_capture_dry_plan as dry
from project.run_scripts.ode_bf.p2r5_sdrt_writer import SDRTCalibration, SDRTQuadratics
from project.run_scripts.ode_bf.p2r5_stage_a_runtime import _run_arm_case
from project.run_scripts.ode_bf.p2r6_llama_qp_capture import (
    LlamaOuter1QPCaptureObserver,
    P2R6LlamaCaptureComplete,
    expected_capture_result_name,
)
from project.run_scripts.ode_bf.p2r6_pilot_runtime import expected_p2r6_result_name


def _geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SDRTCalibration, SDRTQuadratics]:
    response = torch.zeros((10, 50), dtype=torch.float64)
    for request in range(10):
        response[request, request::10] = torch.linspace(0.2, 1.0, 5)
    deficit = torch.linspace(0.05, 0.5, 10, dtype=torch.float64)
    calibration = SDRTCalibration(1.0, 0, 0, 0.0, 0.0, "ENTRY", "fixture")
    quadratics = SDRTQuadratics(
        torch.eye(50, dtype=torch.float64),
        torch.zeros(50, dtype=torch.float64),
        0.0,
        torch.eye(50, dtype=torch.float64),
        torch.zeros(50, dtype=torch.float64),
        0.0,
        "fixture",
    )
    return response, deficit, deficit.clone(), calibration, quadratics


def test_observer_is_noop_at_outer0_then_seals_outer1_once(tmp_path: Path) -> None:
    root = tmp_path / "private-capsule"
    observer = LlamaOuter1QPCaptureObserver(
        capsule_root=root,
        source_head="capture-head",
        request_order_sha256="request-order",
        expected_w0="w0",
    )
    args = _geometry()
    observer(*args, arm="AR-CAP", outer_step=0)
    assert observer.invocation_count == 1
    assert observer.capture_count == 0
    assert not root.exists()

    with pytest.raises(P2R6LlamaCaptureComplete) as stopped:
        observer(*args, arm="AR-CAP", outer_step=1)
    assert stopped.value.raw_free_receipt["outer1_selected_qp_solve_count"] == 0
    assert observer.capture_count == 1
    capsule = json.loads((root / "capsule.json").read_text())
    assert capsule["outer_step"] == 1
    assert capsule["added_model_forward_count"] == 0
    assert capsule["added_model_backward_count"] == 0
    assert capsule["added_materialization_count"] == 0
    assert capsule["terminal_evaluator_count"] == 0
    assert capsule["scientific_endpoint_count"] == 0
    assert set(capsule["members"]) >= {
        "S", "d", "b", "Q_C", "c_C", "M", "alpha_start", "s"
    }
    assert all(item["finite"] for item in capsule["members"].values())
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert all(oct(path.stat().st_mode & 0o777) == "0o600" for path in root.iterdir())


def test_observer_rejects_wrong_arm_and_duplicate_or_late_capture(tmp_path: Path) -> None:
    args = _geometry()
    wrong = LlamaOuter1QPCaptureObserver(
        capsule_root=tmp_path / "wrong",
        source_head="head",
        request_order_sha256="order",
        expected_w0="w0",
    )
    with pytest.raises(ValueError):
        wrong(*args, arm="A0-CAP", outer_step=0)

    late = LlamaOuter1QPCaptureObserver(
        capsule_root=tmp_path / "late",
        source_head="head",
        request_order_sha256="order",
        expected_w0="w0",
    )
    with pytest.raises(RuntimeError):
        late(*args, arm="AR-CAP", outer_step=2)


def test_pre_route_observer_precedes_shadow_and_route() -> None:
    source = inspect.getsource(_run_arm_case)
    observer = source.index("runtime_policy.pre_route_observer(")
    shadow = source.index("runtime_policy.shadow_solver(")
    route = source.index("runtime_policy.route_solver(")
    assert observer < shadow < route


def test_capture_name_and_dry_plan_are_single_invocation_no_endpoint() -> None:
    expected = expected_capture_result_name("a1")
    assert expected == expected_p2r6_result_name(
        "llama3-8b-inst",
        phase="capture",
        case_index=5,
        attempt_suffix="a1",
    )
    plan = dry.build_plan("head")
    assert plan["capture_invocation_budget"] == 1
    assert plan["outer0_physical_prefix_count"] == 1
    assert plan["selected_outer1_qp_solve_count"] == 0
    assert plan["scientific_endpoint_count"] == 0
    assert plan["terminal_evaluator_count"] == 0
    assert plan["w0_restore_required"] is True
    assert plan["phase2_status"] == "CLOSED"


def test_capture_wrapper_contains_exact_w0_restore_and_no_evaluator_call() -> None:
    from project.run_scripts.ode_bf import p2r6_llama_qp_capture as module

    source = inspect.getsource(module.run_p2r6_llama_qp_capture)
    assert "_restore_exact_w0(" in source
    assert '"W0_pointer_restored_exact"' in source
    assert '"W0_byte_restored_exact"' in source
    assert '"terminal_evaluator_count": 0' in source
    assert '"scientific_endpoint_count": 0' in source

