from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r39_normalized_gradient_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1r39_independent_b10x10_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r41_trust_clipped_target import (
    prepare_p1r41_target_proposal,
    select_p1r41_target_proposal,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R41ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_and_matrix(self) -> None:
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
        plan = dry.build_plan("0" * 40, repository_root=ROOT)
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 40)
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertEqual(
            [job["method"] for job in plan["jobs"]],
            [
                "P1R41-TRUST-CLIPPED-NEUTRAL",
                "P1R41-TRUST-CLIPPED-SOFT",
                "P1R41-TRUST-CLIPPED-NEUTRAL",
                "P1R41-TRUST-CLIPPED-SOFT",
            ],
        )

    def test_parent_controller_and_writer_remain_byte_identical(self) -> None:
        self.assertEqual(
            sha256_file(
                ROOT / "project/run_scripts/ode_bf/p1r39_normalized_gradient_target.py"
            ),
            "98356016cd87d258f9c9e889ccb59f6f14e3f869e0f9284d2c90721e38c6537a",
        )
        self.assertEqual(
            sha256_file(
                ROOT / "project/run_scripts/ode_bf/p1r35_full_current_residual.py"
            ),
            "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
        )

    def test_controller_tensor_only_and_no_same_step_retry(self) -> None:
        proposal = inspect.getsource(prepare_p1r41_target_proposal)
        selection = inspect.getsource(select_p1r41_target_proposal)
        for forbidden in ("model(", ".backward(", "evaluate_scalable", "heldout"):
            self.assertNotIn(forbidden, proposal)
            self.assertNotIn(forbidden, selection)
        for forbidden in (
            "P1R38AdamState",
            "P1R38_LR_BY_ALIAS",
            "m_hat",
            "v_hat",
            "gamma_semantic_velocity",
            "objective_alignment_target",
        ):
            self.assertNotIn(forbidden, proposal)
        self.assertIn("radius_same_step_retry_count", proposal)
        self.assertIn("same_step_retry_count", selection)
        self.assertIn("selection_added_model_forward_count", selection)


if __name__ == "__main__":
    unittest.main()
