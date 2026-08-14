from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r42_objective_alignment_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1r42_independent_b10x10_panel import (
    LOCK_FILE,
    METHODS,
    expected_result_name,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r42_objective_aligned_target import (
    prepare_p1r42_target_proposal,
    select_p1r42_target_proposal,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R42ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_matrix_and_protected_sources(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(
            lock["accepted_p1r39_checkpoint"],
            "763457560f2efb177a56310dfd87526772cf8158",
        )
        self.assertEqual(
            lock["fresh_stream_root"],
            "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        )
        self.assertEqual(
            lock["all_request_order_sha256"],
            "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        )
        self.assertEqual(lock["methods"], METHODS)
        self.assertEqual(
            sha256_file(ROOT / "project/run_scripts/ode_bf/p1r35_full_current_residual.py"),
            "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
        )
        self.assertEqual(
            sha256_file(ROOT / "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py"),
            "b2c59ac75798b9d70fbbc370c6d9f805d33cbeb8f683be81bb9effe7d1dbfe53",
        )

    def test_controller_is_tensor_only_and_forbidden_methods_absent(self) -> None:
        proposal = inspect.getsource(prepare_p1r42_target_proposal)
        selection = inspect.getsource(select_p1r42_target_proposal)
        for forbidden in ("model(", ".backward(", "evaluate_scalable", "heldout"):
            self.assertNotIn(forbidden, proposal)
            self.assertNotIn(forbidden, selection)
        for forbidden in (
            "P1R38AdamState",
            "P1R38_LR_BY_ALIAS",
            "request_cap_radius",
            "lock.trust_radius",
            "semantic_velocity_decay(",
            "gamma *",
        ):
            self.assertNotIn(forbidden, proposal)
        self.assertIn("semantic_held", proposal)
        self.assertIn("safe_preservation", proposal)
        self.assertIn("selection_added_model_forward_count", selection)

    def test_result_names_are_four_nonaliased_cells(self) -> None:
        names = {
            expected_result_name(alias, method)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 4)

    def test_b1_submission_state_namespace_is_attempt_scoped(self) -> None:
        source = (ROOT / "project/run_scripts/session05_ode_bf_submit_p1r42_objective_alignment_b10x10.py").read_text()
        self.assertIn('f"{attempt_suffix}-{source_head[:12]}" if smoke', source)

    def test_scheduler_target_is_server2_not_devbox(self) -> None:
        paths = (
            ROOT / "project/run_scripts/session05_ode_bf_p1r42_objective_alignment_b1.sbatch",
            ROOT / "project/run_scripts/session05_ode_bf_p1r42_objective_alignment_b10x10.sbatch",
            ROOT / "project/run_scripts/session05_ode_bf_submit_p1r42_objective_alignment_b10x10.py",
        )
        for path in paths:
            source = path.read_text()
            self.assertIn("server2", source)
            self.assertNotIn("devbox", source)

    def test_dry_plan_has_four_cells_and_forty_independent_cases(self) -> None:
        plan = dry.build_plan("0" * 40, repository_root=ROOT)
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 40)
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertEqual(
            [job["method"] for job in plan["jobs"]],
            [
                "PR-P1R42-OBJECTIVE-ALIGNED-NEUTRAL",
                "PR-P1R42-OBJECTIVE-ALIGNED-SOFT",
                "PR-P1R42-OBJECTIVE-ALIGNED-NEUTRAL",
                "PR-P1R42-OBJECTIVE-ALIGNED-SOFT",
            ],
        )


if __name__ == "__main__":
    unittest.main()
