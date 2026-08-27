from __future__ import annotations

import ast
from pathlib import Path

from project.run_scripts.barrier_guided_ode.r3.g2_convergence import (
    FINAL_NORMALIZED_Q_KL_GAP_CEILING,
    FINAL_NORMALIZED_TARGET_LOGIT_GAP_CEILING,
    FINAL_RELATIVE_PHYSICAL_DISTANCE_CEILING,
    FIRST_ORDER_RATIO_FLOOR,
    N_GRID,
)
from project.run_scripts.barrier_guided_ode.r3.g2_experiment import ARM_ORDER, expected_run_id
from project.run_scripts.barrier_guided_ode.r3.solver import ControllerArm


REPO = Path(__file__).resolve().parents[5]


def test_g2_matrix_and_cauchy_thresholds_are_predeclared() -> None:
    assert N_GRID == (4, 8, 16, 32)
    assert ARM_ORDER == (ControllerArm.PLAIN, ControllerArm.FISHER, ControllerArm.FULL)
    assert FIRST_ORDER_RATIO_FLOOR == 1.25
    assert FINAL_RELATIVE_PHYSICAL_DISTANCE_CEILING == 0.05
    assert FINAL_NORMALIZED_TARGET_LOGIT_GAP_CEILING == 0.05
    assert FINAL_NORMALIZED_Q_KL_GAP_CEILING == 0.05
    assert expected_run_id("llama3-8b-inst").endswith("convergence-v1")
    assert expected_run_id("qwen2.5-7b-inst").endswith("convergence-v1")


def test_g2_runtime_has_no_forbidden_scientific_call_identifiers() -> None:
    forbidden = {
        "brentq",
        "bisection",
        "line_search",
        "retraction",
        "localize",
        "localizer",
        "ridge",
        "damping",
        "fallback",
        "retry",
        "clip_grad_norm_",
    }
    for relative in (
        "project/run_scripts/barrier_guided_ode/r3/g2_experiment.py",
        "project/run_scripts/barrier_guided_ode/r3/g2_trajectory.py",
    ):
        tree = ast.parse((REPO / relative).read_text())
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not forbidden.intersection(identifiers)


def test_g2_sbatch_is_exact_two_model_array_with_cap_safe_throttle() -> None:
    text = (REPO / "project/run_scripts/session05_bgode_r3_g2.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert '0)\n    readonly MODEL_ALIAS="llama3-8b-inst"' in text
    assert '1)\n    readonly MODEL_ALIAS="qwen2.5-7b-inst"' in text
