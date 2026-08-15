from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess

from project.run_scripts import session05_ode_bf_p2r5_stage_a_dry_plan as dry
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.p2r5_stage_a_panel import (
    LOCK_FILE,
    PARENT,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p2r5_stage_a_runtime import STAGE_A_CASES


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "project/run_scripts/ode_bf"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parent_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{PARENT}:{relative}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def test_protected_target_response_materializer_and_evaluator_bytes() -> None:
    protected = (
        "project/run_scripts/ode_bf/p2r1_rms_tangent_target.py",
        "project/run_scripts/ode_bf/p2r2_residual_transport_writer.py",
        "project/run_scripts/ode_bf/p2r4_phaseb_atomic_runtime.py",
        "project/run_scripts/ode_bf/p2r4_phaseb_receipts.py",
        "project/run_scripts/ode_bf/scalable_batched_field.py",
        "project/run_scripts/ode_bf/atomic_runtime_optimization.py",
        "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    )
    for relative in protected:
        current = (ROOT / relative).read_bytes()
        parent = _parent_bytes(relative)
        assert current == parent
        assert _sha(current) == _sha(parent)


def test_lock_and_stage_a_dry_plan_are_exact() -> None:
    lock, _ = load_and_validate_lock(PACKAGE / "locks" / LOCK_FILE)
    plan = dry.build_plan(PARENT)
    assert plan["job_count"] == 4
    assert plan["endpoint_attempt_count"] == 8
    assert plan["request_attempt_count"] == 80
    assert plan["array"] == "0-3%4"
    assert plan["stage_b_status"] == "CLOSED_PENDING_GH_STAGE_A_REVIEW"
    observed = {
        (item["model"], item["case_index"], tuple(item["arms"]))
        for item in plan["jobs"]
    }
    assert len(observed) == 4
    assert STAGE_A_CASES == {
        "llama3-8b-inst": (3, 5),
        "qwen2.5-7b-inst": (1, 4),
    }
    assert lock["clamp_policy"] == "ON_DECISION_ACTIVE_EVERY_TARGET_MICROSTEP"
    repair = dry.build_plan(PARENT, attempt_suffix="tech-r3")
    assert repair["endpoint_attempt_count"] == 6
    assert repair["request_attempt_count"] == 60
    assert [tuple(item["arms"]) for item in repair["jobs"]] == [
        ("SDRT-STRUCTP",),
        ("SDRT-CAP", "SDRT-STRUCTP"),
        ("SDRT-CAP", "SDRT-STRUCTP"),
        ("SDRT-CAP",),
    ]


def test_runtime_reuses_protected_interfaces_and_one_materialization() -> None:
    source = (PACKAGE / "p2r5_stage_a_runtime.py").read_text()
    for name in (
        "p2r1_target_update",
        "measure_request_layer_response",
        "build_proposal_quadratics",
        "p2r2_waypoint_factors",
        "AcceptedPhysicalStateMaterializer",
        "_evaluate_frozen_state",
        "_evaluate_terminal_z_panel",
    ):
        assert name in source
        assert f"def {name}(" not in source
    assert source.count("materializer.materialize(") == 1
    assert source.count("measure_request_layer_response(") == 1
    assert '"candidate_forward_count": 0' in source
    assert '"candidate_materialization_count": 0' in source
    assert '"one_joint_materialization_count": 1' in source
    assert '"heldout_controller_access_count": 0' in source


def test_current_w_target_response_and_deficit_refresh_are_ordered() -> None:
    source = (PACKAGE / "p2r5_stage_a_runtime.py").read_text()
    outer = source.index("for outer in range(8):")
    terminal = source.index("current_terminal = physical.terminal_z.clone()", outer)
    target = source.index("for inner in range(P2R1_MICROSTEPS_PER_OUTER_STATE):", terminal)
    deficit_forward = source.index("z_objective = evaluate_scalable_target_new_objective(", target)
    field = source.index("field = build_scalable_dynamic_field(", deficit_forward)
    response = source.index("response = measure_request_layer_response(", field)
    deficit = source.index("deficit = clamp_safe_semantic_deficit(", response)
    route = source.index("route = solve_sdrt_routing(", deficit)
    materialize = source.index("materialization = materializer.materialize(", route)
    recapture = source.index("next_physical = capture_scalable_physical_state(", materialize)
    assert outer < terminal < target < deficit_forward < field < response < deficit < route < materialize < recapture
    assert "current_target - current_terminal" in source
    assert "entry_writer_field.target_state - entry_writer_field.current_z" in source
    assert "entry_writer_field.current_terminal" not in source
    assert "remaining_horizon" not in source


def test_w0_action_freeze_and_arm_isolation_are_explicit() -> None:
    source = (PACKAGE / "p2r5_stage_a_runtime.py").read_text()
    for token in (
        "for arm in selected_arms:",
        "seed_all(COMMON_SEED)",
        "P2R5 cross-arm W0 leak detected",
        '"actions_frozen_before_evaluator": True',
        '"W0_restored": True',
        '"retry_count": 0',
        '"backtracking_count": 0',
    ):
        assert token in source


def test_entrypoint_and_launcher_resources_are_bound() -> None:
    params = inspect.signature(run_p1).parameters
    assert "p2r5_stage_a_case_index" in params
    assert "p2r5_attempt_suffix" in params
    assert "p2r5_stage_a_arms" in params
    sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p2r5_stage_a.sbatch").read_text()
    submitter = (ROOT / "project/run_scripts/session05_ode_bf_submit_p2r5_stage_a.py").read_text()
    for token in (
        "#SBATCH --array=0-3%4",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=65000M",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodelist=devbox",
    ):
        assert token in sbatch
    assert "readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)" in sbatch
    assert "readonly CASES=(3 5 1 4)" in sbatch
    assert 'if [[ "${ATTEMPT_SUFFIX}" == "tech-r3" ]]' in sbatch
    assert 'PROJECT_GPU_CAP = 4' in submitter
    assert 'STAGE_GPU_MAX = 4' in submitter
    assert '"stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW"' in submitter


def test_no_silent_neutral_fallback_or_forbidden_strength_controls() -> None:
    writer = (PACKAGE / "p2r5_sdrt_writer.py").read_text()
    assert 'P2R5_ARMS = ("SDRT-CAP", "SDRT-STRUCTP")' in writer
    assert '"neutral_fallback_count": 0' in writer
    assert '"eta_floor_count": 0' in writer
    assert '"hard_p_budget_influence_count": 0' in writer
    assert '"functional_p_veto_count": 0' in writer
    assert "solve_p2r2_routing" not in writer
    assert "minimum_remaining" not in writer


def test_semantic_face_quadratic_backend_uses_exact_source_backed_constraints() -> None:
    writer = (PACKAGE / "p2r5_sdrt_writer.py").read_text()
    assert 'method="trust-constr"' in writer
    assert "LinearConstraint(face_matrix, face_value, face_value)" in writer
    assert "NonlinearConstraint(" in writer
    assert '"gtol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE' in writer
    assert '"xtol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE' in writer
    assert '"barrier_tol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE' in writer
    assert 'method="SLSQP"' not in writer
