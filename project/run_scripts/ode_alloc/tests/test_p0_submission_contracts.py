from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock
from pathlib import Path

from project.run_scripts import session04_ode_alloc_submit_p0_r1 as submit_module
from project.run_scripts.session04_ode_alloc_submit_p0_r1 import (
    APPROVED_NUMERICAL_DIFF_PATHS,
    CANONICAL_NODE,
    JOB_NAMES,
    _changed_json_paths,
    _gpu_count,
    _node_local_gpu_totals,
    _entrypoint,
)


class P0SubmissionContractTests(unittest.TestCase):
    def test_project_gpu_tres_parser_is_fail_closed(self) -> None:
        self.assertEqual(_gpu_count("gres/gpu:1"), 1)
        self.assertEqual(_gpu_count("gres/gpu:a6000:2(S:0-1)"), 2)
        self.assertEqual(_gpu_count("gpu=1"), 1)
        self.assertIsNone(_gpu_count("N/A"))
        local, cluster = _node_local_gpu_totals(
            [
                {
                    "gpus": 1,
                    "nodes": ("devbox",),
                    "pending_unconstrained": False,
                },
                {
                    "gpus": 2,
                    "nodes": (CANONICAL_NODE,),
                    "pending_unconstrained": False,
                },
            ]
        )
        self.assertEqual((local, cluster), (2, 3))
        conservative, _ = _node_local_gpu_totals(
            [{"gpus": 1, "nodes": (), "pending_unconstrained": True}]
        )
        self.assertEqual(conservative, 1)

    def test_exact_pair_names_and_resources_are_locked(self) -> None:
        self.assertEqual(
            JOB_NAMES,
            {
                "llama3-8b-inst": "odealloc_s04_p0r3_llama",
                "qwen2.5-7b-inst": "odealloc_s04_p0r3_qwen",
            },
        )
        sbatch = (
            Path(__file__).resolve().parents[2]
            / "session04_ode_alloc_p0_r1.sbatch"
        ).read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --nodelist=server2",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=04:00:00",
            "#SBATCH --export=NONE",
        ):
            self.assertIn(directive, sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertIn('--run-token "${RUN_TOKEN}"', sbatch)
        self.assertIn('"${RUN_TOKEN}" == "r3"', sbatch)

    def test_numerical_diff_contract_is_path_exact(self) -> None:
        self.assertEqual(
            _changed_json_paths({"solver": {"fixed_k": 8}}, {"solver": {"fixed_k": 9}}),
            {"solver.fixed_k"},
        )
        self.assertNotIn("solver.fixed_k", APPROVED_NUMERICAL_DIFF_PATHS)

    def test_entrypoint_success_does_not_emit_false_hold(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        def successful_main() -> int:
            print('{"status":"SUBMITTED_PAIR"}')
            return 0

        with mock.patch.object(submit_module, "main", side_effect=successful_main):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    _entrypoint()
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), '{"status":"SUBMITTED_PAIR"}')
        self.assertNotIn("PRE_SUBMIT_HOLD", stderr.getvalue())

    def test_entrypoint_exception_emits_hold_and_reraises(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            submit_module, "main", side_effect=RuntimeError("opaque failure")
        ):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(RuntimeError):
                    _entrypoint()
        value = json.loads(stderr.getvalue())
        self.assertEqual(value["status"], "PRE_SUBMIT_HOLD")
        self.assertEqual(value["error_type"], "RuntimeError")
        self.assertNotIn("opaque failure", stderr.getvalue())

    def test_entrypoint_system_exit_is_not_classified_as_hold(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(submit_module, "main", side_effect=SystemExit(7)):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    _entrypoint()
        self.assertEqual(caught.exception.code, 7)
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
