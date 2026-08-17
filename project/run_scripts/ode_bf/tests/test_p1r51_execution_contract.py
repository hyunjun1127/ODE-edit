from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import session05_ode_bf_p1r51_rsa_a1_dry_plan as dry
from project.run_scripts.ode_bf.p1r51_independent_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p1r51_independent_runtime import PHASES, expected_p1r51_result_name
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    prepare_p1r51_rescue_proposal,
    prepare_p1r51_target_proposal,
    select_p1r51_target_proposal,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R51ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_and_only_allocation_delta(self) -> None:
        lock, _ = load_and_validate_lock(ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE)
        self.assertEqual(lock["exact_p1r43_parent"], "11508b6da11d606521b703037034e1814b70d8a8")
        self.assertEqual(lock["stream_root"], "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6")
        self.assertEqual(lock["target_direction_policy"], "REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION")
        self.assertEqual(lock["writer_router_change_count"], 0)
        self.assertEqual(lock["p2_operator_access_count"], 0)

    def test_policy_is_tensor_only_and_forbidden_controls_absent(self) -> None:
        source = "\n".join(
            inspect.getsource(value)
            for value in (
                prepare_p1r51_target_proposal,
                prepare_p1r51_rescue_proposal,
                select_p1r51_target_proposal,
            )
        )
        for forbidden in (
            "model(", ".backward(", "evaluate_scalable", "heldout",
            "P2R", "radius", "g_floor", "R_max", "B_max",
            "remaining_steps", "historical", "sequential",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("prepare_p1r43_rescue_proposal", source)
        self.assertIn("select_p1r43_target_proposal", source)

    def test_all_phase_names_are_nonaliased(self) -> None:
        names = {
            expected_p1r51_result_name(alias, phase=phase)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for phase in PHASES
        }
        self.assertEqual(len(names), 8)

    def test_dry_plans_are_two_model_and_cap_safe(self) -> None:
        for phase in PHASES:
            plan = dry.build_plan("0" * 40, phase, repository_root=ROOT)
            self.assertEqual(plan["job_count"], 2)
            self.assertEqual(plan["array"], "0-1%2")
            self.assertEqual(plan["project_gpu_cap"], 4)
            self.assertLessEqual(plan["stage_gpu_max"], 2)
            self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 2)

    def test_main_only_launcher_and_server1_contract(self) -> None:
        sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p1r51_rsa_a1.sbatch").read_text(encoding="utf-8")
        submitter = (ROOT / "project/run_scripts/session05_ode_bf_submit_p1r51_rsa_a1.py").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_BRANCH="main"', sbatch)
        self.assertIn('BRANCH = "main"', submitter)
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)
        self.assertIn("devbox", sbatch)
        self.assertNotIn("server2", sbatch)


if __name__ == "__main__":
    unittest.main()
