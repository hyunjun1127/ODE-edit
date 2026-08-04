from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import session04_ode_bf_submit_p1 as submit_module
from project.run_scripts.session04_ode_bf_p1_dry_plan import JOB_NAMES, build_plan


REPO_ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh2/ODE-edit")


class P1SubmissionTests(unittest.TestCase):
    def test_exact_pair_namespace_resources_and_repeatable_plan(self) -> None:
        self.assertEqual(
            JOB_NAMES,
            {
                "llama3-8b-inst": "odebf_s04_p1_seqb10_llama",
                "qwen2.5-7b-inst": "odebf_s04_p1_seqb10_qwen",
            },
        )
        first = json.dumps(
            build_plan("f" * 40, repository_root=REPO_ROOT),
            sort_keys=True,
            separators=(",", ":"),
        )
        second = json.dumps(
            build_plan("f" * 40, repository_root=REPO_ROOT),
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(first, second)
        plan = json.loads(first)
        self.assertEqual(plan["edit_batch_size"], 10)
        self.assertEqual(plan["sequential_batch_count"], 4)
        self.assertEqual(plan["logical_edits_per_arm"], 40)
        self.assertEqual(plan["arms"], ["N32_NATIVE", "F_G", "F_BF", "R_BF"])
        self.assertEqual(len(plan["jobs"]), 2)
        for job in plan["jobs"]:
            self.assertEqual((job["gpu"], job["cpu"]), (1, 8))
            self.assertEqual(job["memory_mib"], 65_000)
            self.assertEqual(job["time"], "24:00:00")
            self.assertLessEqual(job["memory_forecast"]["forecast_gpu_peak_mib"], 65_000)
            self.assertLessEqual(job["memory_forecast"]["forecast_host_peak_mib"], 65_000)

    def test_sbatch_locks_server2_one_gpu_and_canonical_offline_env(self) -> None:
        sbatch = (
            Path(__file__).resolve().parents[2] / "session04_ode_bf_p1.sbatch"
        ).read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --nodelist=server2",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=24:00:00",
            "#SBATCH --export=NONE",
        ):
            self.assertIn(directive, sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertIn('"${RUN_TOKEN}" == "seqb10-native-floor-p1-v1"', sbatch)
        self.assertIn("export HF_HUB_OFFLINE=1", sbatch)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", sbatch)
        self.assertIn('--run-token "${RUN_TOKEN}"', sbatch)

    def test_lock_seal_cross_identity_and_r4_immutability(self) -> None:
        lock_gate = submit_module._lock_and_seal_gate()
        self.assertEqual(
            lock_gate["stream_root_digest"],
            "c57bca3fea781be34394e40d27530a032ccff539e7b637010669676c5300ae25",
        )
        self.assertEqual(
            lock_gate["population_root_digest"],
            "144b432293eea78775a74da5faf9fc837ca301b33a7e5e0c2b8088cde4721120",
        )
        self.assertEqual(len(submit_module._r4_immutability_gate()), 8)

    def test_allowed_checkpoint_scope_is_exact_and_excludes_foreign_namespaces(self) -> None:
        self.assertEqual(len(submit_module.P1_ALLOWED_CHANGED_PATHS), 24)
        self.assertTrue(
            all(
                path.startswith("project/run_scripts/ode_bf/")
                or path.startswith("project/run_scripts/session04_ode_bf_")
                for path in submit_module.P1_ALLOWED_CHANGED_PATHS
            )
        )
        self.assertFalse(
            any(
                "session03" in path.casefold() or "knowledge-revision" in path.casefold()
                for path in submit_module.P1_ALLOWED_CHANGED_PATHS
            )
        )

    def test_entrypoint_success_systemexit_and_sanitized_exception(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(submit_module, "main", return_value=0):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    submit_module._entrypoint()
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stderr.getvalue(), "")

        stderr = io.StringIO()
        with mock.patch.object(submit_module, "main", side_effect=SystemExit(7)):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    submit_module._entrypoint()
        self.assertEqual(caught.exception.code, 7)
        self.assertEqual(stderr.getvalue(), "")

        stderr = io.StringIO()
        with mock.patch.object(
            submit_module,
            "main",
            side_effect=RuntimeError("private endpoint must not be emitted"),
        ):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(RuntimeError):
                    submit_module._entrypoint()
        value = json.loads(stderr.getvalue())
        self.assertEqual(value["status"], "PRE_SUBMIT_OR_PARTIAL_HOLD")
        self.assertNotIn("private endpoint", stderr.getvalue())
        self.assertFalse(value["retry_or_resubmit"])


if __name__ == "__main__":
    unittest.main()
