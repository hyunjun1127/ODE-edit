from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[5]


def test_g2_trajectory_augments_locked_fd_boundary_without_changing_tolerance() -> None:
    source = (REPO / "project/run_scripts/barrier_guided_ode/r3/g2_trajectory.py").read_text()
    tree = ast.parse(source)
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    for field in (
        "arm",
        "step_count",
        "node_index",
        "completed_node_count",
        "physical_action_count",
        "current_node_write_count",
        "failure_stage",
    ):
        assert field in constants
    assert "FD_ABSOLUTE_TOLERANCE" not in source
    assert "FD_RELATIVE_TOLERANCE" not in source


def test_g2_experiment_publishes_failure_only_after_exact_w0_restore() -> None:
    source = (REPO / "project/run_scripts/barrier_guided_ode/r3/g2_experiment.py").read_text()
    restore = source.index("_assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)", source.index("except NumericalBoundary"))
    publish = source.index('_write_json_once(output / "failure-boundary.json", failure)', restore)
    assert restore < publish
    assert '"tolerance_change_count": 0' in source
    assert '"threshold_change_count": 0' in source


def test_tech_r1_launcher_preserves_original_matrix_and_lock() -> None:
    text = (REPO / "project/run_scripts/session05_bgode_r3_g2_tech_r1.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "bgode-r3-g2-tech-r1-source-manifest.json" in text
    assert "bgode-r3-g2-cauchy-numerical-lock.json" not in text
