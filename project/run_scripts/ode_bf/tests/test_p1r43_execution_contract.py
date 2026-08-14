from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import session05_ode_bf_p1r43_rho_free_b10x10_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r36_independent_b10x10_runtime import (
    _p1r43_panel_metric,
)
from project.run_scripts.ode_bf.p1r43_independent_b10x10_panel import (
    LOCK_FILE,
    METHODS,
    expected_result_name,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r43_rho_free_target import (
    prepare_p1r43_rescue_proposal,
    prepare_p1r43_target_proposal,
    select_p1r43_target_proposal,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R43ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_and_matrix(self) -> None:
        lock, _ = load_and_validate_lock(ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE)
        self.assertEqual(lock["exact_p1r42_parent"], "ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c")
        self.assertEqual(lock["stream_root"], "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6")
        self.assertEqual(lock["methods"], list(METHODS))
        self.assertEqual(lock["trust_rho_decision_influence_count"], 0)
        self.assertEqual(lock["heldout_gen_controller_access_count"], 0)

    def test_target_policy_is_tensor_only_and_trust_state_absent(self) -> None:
        sources = "\n".join(
            inspect.getsource(value)
            for value in (
                prepare_p1r43_target_proposal,
                prepare_p1r43_rescue_proposal,
                select_p1r43_target_proposal,
            )
        )
        for forbidden in (
            "model(", ".backward(", "evaluate_scalable", "heldout",
            "radius_before", "radius_after", "0.25", "0.75",
            "P1R24_FREEZE_THRESHOLD", "semantic_velocity_decay",
        ):
            self.assertNotIn(forbidden, sources)
        self.assertIn("entry_semantic_gradient_norm", sources)
        self.assertIn("rescue_endpoint", sources)

    def test_four_nonaliased_cells_and_dry_plan(self) -> None:
        names = {
            expected_result_name(alias, method)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 4)
        plan = dry.build_plan("0" * 40, repository_root=ROOT)
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 40)
        self.assertEqual(plan["terminal_panels"], ["EFF_Z_INJECT", "GEN_Z_INJECT", "EFF_W", "GEN_W"])

    def test_server1_cap4_launch_contract(self) -> None:
        for relative in (
            "project/run_scripts/session05_ode_bf_p1r43_rho_free_b1.sbatch",
            "project/run_scripts/session05_ode_bf_p1r43_rho_free_b10x10.sbatch",
            "project/run_scripts/session05_ode_bf_submit_p1r43_rho_free_b10x10.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("devbox", source)
            self.assertNotIn("server2", source)
        submitter = (ROOT / "project/run_scripts/session05_ode_bf_submit_p1r43_rho_free_b10x10.py").read_text(encoding="utf-8")
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)

    def test_terminal_panel_uses_pinned_primary_metrics_schema(self) -> None:
        payload = {
            "primary": {
                "metrics": {
                    "efficacy": {"correct_count": 9},
                    "generalization": {"correct_count": 17},
                }
            }
        }
        self.assertEqual(_p1r43_panel_metric(payload, "efficacy"), {"correct_count": 9})
        self.assertEqual(
            _p1r43_panel_metric(payload, "generalization"), {"correct_count": 17}
        )
        with self.assertRaises(ODEBFContractError):
            _p1r43_panel_metric({"primary": {"efficacy": {}}}, "efficacy")

    def test_technical_replacement_names_are_distinct(self) -> None:
        names = {
            expected_result_name(alias, method, attempt_suffix="tech-r2")
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 4)
        self.assertTrue(all("-tech-r2-v1" in name for name in names))


if __name__ == "__main__":
    unittest.main()
