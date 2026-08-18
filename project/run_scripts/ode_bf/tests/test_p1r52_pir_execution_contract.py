from __future__ import annotations

import inspect
import subprocess
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r52_pir_atomic_b10x10 as runner,
    session05_ode_bf_p1r52_pir_atomic_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.p1r52_frozen_pi_quota_writer import (
    plan_sequential_writer,
)
from project.run_scripts.ode_bf.p1r52_pir_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R52PIRExecutionContractTests(unittest.TestCase):
    def test_numerical_lock(self) -> None:
        value, file_sha = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(value["arms"], ["pir-j0", "pir-g", "pir-u"])
        self.assertEqual(value["additional_current_slope_backward_count"], 0)
        self.assertEqual(value["current_q_solve_count_per_k8_sequential_arm"], 32)
        self.assertEqual(value["sequential_p_receipt"], "ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE")
        self.assertEqual(len(file_sha), 64)

    def test_three_cell_dry_plan(self) -> None:
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["array"], "0-2%3")
        self.assertEqual(plan["total_endpoint_attempts"], 30)
        self.assertEqual([job["arm"] for job in plan["jobs"]], ["pir-j0", "pir-g", "pir-u"])
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 3)

    def test_launcher_contract(self) -> None:
        sbatch_path = ROOT / "project/run_scripts/session05_ode_bf_p1r52_pir_atomic_b10x10.sbatch"
        sbatch = sbatch_path.read_text()
        self.assertIn("#SBATCH --array=0-2%3", sbatch)
        self.assertIn("readonly ARMS=(pir-j0 pir-g pir-u)", sbatch)
        self.assertNotIn("qwen", sbatch.lower())
        subprocess.run(["bash", "-n", str(sbatch_path)], check=True)

    def test_source_manifest(self) -> None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertEqual(len(runner._source_gate(head)), 64)

    def test_j0_fpiq_planner_source_is_unchanged_by_pir_policy(self) -> None:
        source = inspect.getsource(plan_sequential_writer)
        self.assertNotIn("PIR-G", source)
        self.assertNotIn("PIR-U", source)

    def test_current_main_clamp_closure_remains_bound(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py"
        ).read_text()
        self.assertIn("POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA", source)
        self.assertIn("post_cast", source)


if __name__ == "__main__":
    unittest.main()
