from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1r39_independent_b10x10_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p1r39_normalized_gradient_target import prepare_p1r39_target_proposal, select_p1r39_target_proposal


ROOT = Path(__file__).resolve().parents[4]


class P1R39ExecutionContractTests(unittest.TestCase):
    def test_lock_stream_and_protected_sources(self) -> None:
        lock, _ = load_and_validate_lock(ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE)
        self.assertEqual(lock["fresh_stream_root"], "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6")
        self.assertEqual(lock["all_request_order_sha256"], "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c")
        self.assertEqual(sha256_file(ROOT / "project/run_scripts/ode_bf/p1r38_perrequest_target.py"), "2ee532a3b4293cafb99116b1a587a18f4eef2489deb26b6e5cb3afcd8e3795a3")
        self.assertEqual(sha256_file(ROOT / "project/run_scripts/ode_bf/p1r35_full_current_residual.py"), "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b")

    def test_controller_is_tensor_only_and_forbidden_state_absent(self) -> None:
        proposal = inspect.getsource(prepare_p1r39_target_proposal)
        selection = inspect.getsource(select_p1r39_target_proposal)
        for forbidden in ("model(", ".backward(", "evaluate_scalable", "heldout"):
            self.assertNotIn(forbidden, proposal)
            self.assertNotIn(forbidden, selection)
        for forbidden in ("P1R38AdamState", "P1R38_LR_BY_ALIAS", "m_hat", "v_hat", "request_cap_radius"):
            self.assertNotIn(forbidden, proposal)
        self.assertIn("CURRENT_STATE_INSTANTANEOUS_NO_CARRY", proposal)
        self.assertIn("selection_added_model_forward_count", selection)


if __name__ == "__main__":
    unittest.main()
