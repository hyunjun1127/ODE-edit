from __future__ import annotations

import unittest
from pathlib import Path

from project.run_scripts.session04_ode_alloc_submit_p0_r1 import (
    APPROVED_NUMERICAL_DIFF_PATHS,
    JOB_NAMES,
    _changed_json_paths,
    _gpu_count,
)


class P0SubmissionContractTests(unittest.TestCase):
    def test_project_gpu_tres_parser_is_fail_closed(self) -> None:
        self.assertEqual(_gpu_count("gres/gpu:1"), 1)
        self.assertEqual(_gpu_count("gres/gpu:a6000:2(S:0-1)"), 2)
        self.assertEqual(_gpu_count("gpu=1"), 1)
        self.assertIsNone(_gpu_count("N/A"))

    def test_exact_pair_names_and_resources_are_locked(self) -> None:
        self.assertEqual(
            JOB_NAMES,
            {
                "llama3-8b-inst": "odealloc_s04_p0_llama",
                "qwen2.5-7b-inst": "odealloc_s04_p0_qwen",
            },
        )
        sbatch = (
            Path(__file__).resolve().parents[2]
            / "session04_ode_alloc_p0_r1.sbatch"
        ).read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=04:00:00",
            "#SBATCH --export=NONE",
        ):
            self.assertIn(directive, sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)

    def test_numerical_diff_contract_is_path_exact(self) -> None:
        self.assertEqual(
            _changed_json_paths({"solver": {"fixed_k": 8}}, {"solver": {"fixed_k": 9}}),
            {"solver.fixed_k"},
        )
        self.assertNotIn("solver.fixed_k", APPROVED_NUMERICAL_DIFF_PATHS)


if __name__ == "__main__":
    unittest.main()
