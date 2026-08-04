from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import session04_ode_bf_submit_p0 as submit_module
from project.run_scripts.ode_bf.artifacts import load_rooted_json, sha256_file
from project.run_scripts.session04_ode_bf_dry_plan import JOB_NAMES, build_plan


REPO_ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh2/ODE-edit")


class P0SubmissionTests(unittest.TestCase):
    def test_exact_pair_resources_namespace_and_dry_repeat(self) -> None:
        self.assertEqual(
            JOB_NAMES,
            {
                "llama3-8b-inst": "odebf_s04_p0r4_llama",
                "qwen2.5-7b-inst": "odebf_s04_p0r4_qwen",
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
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertTrue(all(job["memory_mib"] == 65000 for job in plan["jobs"]))
        self.assertTrue(all(job["gpu"] == 1 and job["cpu"] == 8 for job in plan["jobs"]))
        self.assertTrue(all(job["memory_forecast"]["edit_batch_size"] == 10 for job in plan["jobs"]))
        self.assertEqual(plan["diagnostic_paths"], ["N32", "D32", "W32", "W64"])
        self.assertTrue(
            all(job["r4_forecast_host_peak_mib"] <= 65000 for job in plan["jobs"])
        )
        self.assertEqual(plan["prospective_technical_path"], "W64")
        self.assertFalse(plan["w32_fallback"])
        self.assertIn("W64-virtual-vs-commit", plan["strict_byte_gates"])

    def test_sbatch_locks_one_gpu_server2_and_no_array(self) -> None:
        sbatch = (
            Path(__file__).resolve().parents[2] / "session04_ode_bf_p0.sbatch"
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
        self.assertIn(
            '"${RUN_TOKEN}" == "w64-receipt-field-r4-b10"',
            sbatch,
        )
        self.assertIn('--run-token "${RUN_TOKEN}"', sbatch)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", sbatch)

    def test_six_path_cuda_repair_content_is_exactly_preserved(self) -> None:
        self.assertEqual(len(submit_module.PRESERVED_REPAIR_SHA256), 6)
        for relative, expected in submit_module.PRESERVED_REPAIR_SHA256.items():
            self.assertEqual(sha256_file(REPO_ROOT / relative), expected)

    def test_r3_technical_lock_is_additive_to_the_pinned_artifact_lock(self) -> None:
        lock_root = REPO_ROOT / "project/run_scripts/ode_bf/locks"
        pinned, _ = load_rooted_json(lock_root / "p0_artifact_lock.json")
        prospective, digest = load_rooted_json(
            lock_root / "numerical_lock_p0_r3.json",
            expected_schema="ode-edit-s04-ode-bf-p0-numerical-lock/v1",
        )
        self.assertEqual(
            pinned["locks"]["project/run_scripts/ode_bf/locks/numerical_lock_p0.json"],
            "23fe5621612f715c52ef70f94a10ae7eaba759b4ac209e0e0f3e09c81ea9feef",
        )
        self.assertEqual(
            digest,
            "40421f3f8ef0e47df842268bb68b9c9548398e27e0a9afecb125ab1b6f853fa1",
        )
        self.assertEqual(prospective["prospective_technical_path"]["primary_path"], "W64")
        self.assertFalse(prospective["prospective_technical_path"]["w32_fallback"])

    def test_r0_through_r3_roots_logs_and_receipts_are_immutable(self) -> None:
        self.assertEqual(
            submit_module._r0_immutability_gate(),
            submit_module.R0_IMMUTABILITY_SHA256,
        )
        self.assertEqual(
            submit_module.BASE_HEAD,
            "81ed70588b8ed7ebc3816a3b097531f35a89f094",
        )
        self.assertEqual(
            submit_module._r2_immutability_gate(),
            (
                submit_module.R2_PER_ROOT_SHA256,
                submit_module.R2_LOG_STATE_IMMUTABILITY_SHA256,
            ),
        )
        self.assertEqual(
            submit_module._r1_immutability_gate(),
            (
                submit_module.R1_RESULT_IMMUTABILITY_SHA256,
                submit_module.R1_LOG_STATE_IMMUTABILITY_SHA256,
            ),
        )
        self.assertEqual(
            submit_module._r3_immutability_gate(),
            (
                submit_module.R3_PER_ROOT_SHA256,
                submit_module.R3_LOG_STATE_IMMUTABILITY_SHA256,
            ),
        )

    def test_entrypoint_success_and_systemexit_are_not_false_holds(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(submit_module, "main", return_value=0):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
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

    def test_entrypoint_exception_is_sanitized_and_reraised(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            submit_module,
            "main",
            side_effect=RuntimeError("private endpoint should not be emitted"),
        ):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(RuntimeError):
                    submit_module._entrypoint()
        value = json.loads(stderr.getvalue())
        self.assertEqual(value["status"], "PRE_SUBMIT_OR_PARTIAL_HOLD")
        self.assertEqual(value["exception_class"], "RuntimeError")
        self.assertNotIn("private endpoint", stderr.getvalue())
        self.assertFalse(value["retry_or_resubmit"])


if __name__ == "__main__":
    unittest.main()
