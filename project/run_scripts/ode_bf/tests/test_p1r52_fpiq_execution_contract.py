from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r52_fpiq_atomic_b10x10 as runner,
    session05_ode_bf_p1r52_fpiq_atomic_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.p1r52_fpiq_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R52FPiQExecutionContractTest(unittest.TestCase):
    def test_numerical_lock(self) -> None:
        value, file_sha = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(value["arms"], ["j0", "sv", "fpiq"])
        self.assertEqual(value["additional_slope_group_count_per_k8_sequential_arm"], 32)
        self.assertEqual(len(file_sha), 64)

    def test_three_cell_dry_plan(self) -> None:
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["array"], "0-2%3")
        self.assertEqual(plan["total_endpoint_attempts"], 30)
        self.assertEqual(
            [job["arm"] for job in plan["jobs"]], ["j0", "sv", "fpiq"]
        )
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 3)

    def test_launcher_contract(self) -> None:
        sbatch = (
            ROOT
            / "project/run_scripts/session05_ode_bf_p1r52_fpiq_atomic_b10x10.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --array=0-2%3", sbatch)
        self.assertIn("readonly ARMS=(j0 sv fpiq)", sbatch)
        self.assertNotIn("qwen", sbatch.lower())
        subprocess.run(
            ["bash", "-n", str(ROOT / "project/run_scripts/session05_ode_bf_p1r52_fpiq_atomic_b10x10.sbatch")],
            check=True,
        )

    def test_source_manifest(self) -> None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertEqual(len(runner._source_gate(head)), 64)


if __name__ == "__main__":
    unittest.main()
