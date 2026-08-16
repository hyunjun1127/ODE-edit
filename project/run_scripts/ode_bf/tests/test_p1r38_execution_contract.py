from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import session05_ode_bf_p1r38_perrequest_b10x10_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1r38_independent_b10x10_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p1r38_perrequest_target import prepare_p1r38_target_proposal, select_p1r38_target_proposal


ROOT = Path(__file__).resolve().parents[4]


class P1R38ExecutionContractTests(unittest.TestCase):
    def test_lock_stream_matrix_and_protected_p1r35(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(lock["fresh_stream_root"], "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6")
        self.assertEqual(lock["all_request_order_sha256"], "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c")
        self.assertEqual(sha256_file(ROOT / "project/run_scripts/ode_bf/p1r35_full_current_residual.py"), "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b")
        self.assertEqual(sha256_file(ROOT / "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py"), "b2c59ac75798b9d70fbbc370c6d9f805d33cbeb8f683be81bb9effe7d1dbfe53")

    def test_dry_plan_has_four_nonaliased_cells_and_forty_cases(self) -> None:
        plan = dry.build_plan("0" * 40, repository_root=ROOT)
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 40)
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertEqual([job["method"] for job in plan["jobs"]], [
            "PR-P1R35-NEUTRAL", "PR-P1R35-SOFT",
            "PR-P1R35-NEUTRAL", "PR-P1R35-SOFT",
        ])

    def test_controller_is_tensor_only_and_selection_reuses_endpoint(self) -> None:
        proposal_source = inspect.getsource(prepare_p1r38_target_proposal)
        selection_source = inspect.getsource(select_p1r38_target_proposal)
        for forbidden in ("model(", ".backward(", "evaluate_scalable", "heldout"):
            self.assertNotIn(forbidden, proposal_source)
            self.assertNotIn(forbidden, selection_source)
        self.assertIn("replace(", selection_source)
        self.assertIn("selection_added_model_forward_count", selection_source)
        self.assertIn("CURRENT_STATE_INSTANTANEOUS_NO_CARRY", proposal_source)

    def test_b1_wrapper_bounds_microbatch_without_changing_b10(self) -> None:
        source = (ROOT / "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py").read_text(encoding="utf-8")
        self.assertEqual(
            source.count("request_microbatch_size=min(request_microbatch_size, len(requests))"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
