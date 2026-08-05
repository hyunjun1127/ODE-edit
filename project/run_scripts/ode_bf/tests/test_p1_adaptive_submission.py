from __future__ import annotations

import json
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import session04_ode_bf_p1_adaptive as entry
from project.run_scripts import session04_ode_bf_p1_adaptive_dry_plan as dry
from project.run_scripts import session04_ode_bf_submit_p1_adaptive as submit
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_adaptive import ADAPTIVE_VARIANTS
from project.run_scripts.ode_bf.p1_runtime import (
    expected_p1r4_adaptive_result_name,
)


class AdaptiveDryPlanTests(unittest.TestCase):
    def test_plan_is_common_policy_b10_and_fits_resource_caps(self) -> None:
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["edit_batch_size"], 10)
        self.assertEqual(plan["sequential_batch_count"], 1)
        self.assertEqual(plan["common_controller"], "R_BF")
        self.assertEqual(plan["variants"], [item.value for item in ADAPTIVE_VARIANTS])
        self.assertFalse(plan["scientific_promotion_authorized"])
        self.assertFalse(plan["reject_advances_tau"])
        self.assertTrue(plan["accepted_state_field_refresh_only"])
        self.assertEqual(len(plan["jobs"]), 2)
        for job in plan["jobs"]:
            self.assertEqual(job["gpu"], 1)
            self.assertEqual(job["cpu"], 8)
            self.assertEqual(job["memory_mib"], 65_000)
            self.assertEqual(job["time"], "24:00:00")
            self.assertLessEqual(
                job["memory_forecast"]["forecast_gpu_peak_mib"], 65_000
            )
            self.assertLessEqual(
                job["memory_forecast"]["forecast_host_peak_mib"], 65_000
            )
            self.assertLessEqual(job["time_forecast"]["forecast_seconds"], 86_400)

    def test_plan_is_byte_repeatable(self) -> None:
        first = json.dumps(dry.build_plan("b" * 40), sort_keys=True, separators=(",", ":"))
        second = json.dumps(dry.build_plan("b" * 40), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)


class AdaptiveEntryAndSubmissionTests(unittest.TestCase):
    def test_entry_parser_locks_alias_and_run_token(self) -> None:
        for alias in MODEL_ALIASES:
            args = entry._parser().parse_args(
                [
                    "--model",
                    alias,
                    "--output-root",
                    expected_p1r4_adaptive_result_name(alias),
                    "--source-head",
                    "c" * 40,
                    "--run-token",
                    entry.ADAPTIVE_RESULT_TOKEN,
                ]
            )
            self.assertEqual(args.model, alias)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry._parser().parse_args(
                [
                    "--model",
                    MODEL_ALIASES[0],
                    "--output-root",
                    "x",
                    "--source-head",
                    "c" * 40,
                    "--run-token",
                    "wrong",
                ]
            )

    def test_sbatch_has_locked_one_visible_device_resources(self) -> None:
        source = submit.SBATCH.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --nodelist=server2",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=24:00:00",
            "#SBATCH --export=NONE",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "CUDA_VISIBLE_DEVICES",
            entry.ADAPTIVE_RESULT_TOKEN,
        ):
            self.assertIn(fragment, source)

    def test_submit_command_is_single_sbatch_with_exact_namespace(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="12345\n", stderr="")
        with mock.patch.object(submit, "_run", return_value=completed) as invoked:
            job_id = submit._submit_one(
                alias=MODEL_ALIASES[0],
                output_root=Path("/tmp/locked-root"),
                source_head="d" * 40,
                log_parent=Path("/tmp/locked-log"),
            )
        self.assertEqual(job_id, "12345")
        command = invoked.call_args.args[0]
        self.assertEqual(command[0], "sbatch")
        self.assertEqual(command.count("sbatch"), 1)
        self.assertIn(dry.JOB_NAMES[MODEL_ALIASES[0]], command)
        self.assertEqual(command[-1], entry.ADAPTIVE_RESULT_TOKEN)

    def test_create_once_receipt_rejects_second_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            digest = submit._write_once(path, {"status": "SEALED"})
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                submit._write_once(path, {"status": "OVERWRITE"})

    def test_changed_scope_is_exact_and_has_no_simple_k20_namespace(self) -> None:
        self.assertEqual(len(submit.ALLOWED_CHANGED_PATHS), 15)
        self.assertTrue(
            all(path.startswith("project/run_scripts/") for path in submit.ALLOWED_CHANGED_PATHS)
        )
        combined = "\n".join(sorted(submit.ALLOWED_CHANGED_PATHS))
        self.assertNotIn("K20", combined)
        self.assertNotIn("session03", combined.casefold())


if __name__ == "__main__":
    unittest.main()
