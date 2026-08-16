from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest import mock

import project.run_scripts.session04_ode_alloc_submit_p1 as submit
from project.run_scripts.ode_alloc.contracts import MODEL_ALIASES
from project.run_scripts.ode_alloc.p0_artifacts import ArtifactReceipt
from project.run_scripts.ode_alloc.p1_contracts import P1_RUN_TOKEN
from project.run_scripts.ode_alloc.p1_runtime import (
    NUMERICAL_LOCK_SHA256,
    P1_EXECUTION_TOKEN,
    P1_REPAIR_EXPECTED_BASE,
    expected_p1r1_result_name,
)


ROOT = Path(__file__).resolve().parents[2]


class P1SubmissionContractTests(unittest.TestCase):
    def test_exact_pair_names_roots_and_resources(self) -> None:
        self.assertEqual(
            submit.JOB_NAMES,
            {
                "llama3-8b-inst": "odealloc_s04_p1r1_llama",
                "qwen2.5-7b-inst": "odealloc_s04_p1r1_qwen",
            },
        )
        sbatch = (ROOT / "session04_ode_alloc_p1.sbatch").read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --nodelist=server2",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=12:00:00",
        ):
            self.assertIn(directive, sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertIn('"${RUN_TOKEN}" == "p1v1"', sbatch)
        self.assertIn('"${EXECUTION_TOKEN}" == "p1r1v1"', sbatch)
        self.assertIn("exec /usr/bin/srun --unbuffered", sbatch)
        self.assertEqual(P1_RUN_TOKEN, "p1v1")
        self.assertEqual(P1_EXECUTION_TOKEN, "p1r1v1")
        self.assertEqual(P1_REPAIR_EXPECTED_BASE, submit.P1_REPAIR_EXPECTED_BASE)
        self.assertEqual(
            expected_p1r1_result_name(
                "qwen2.5-7b-inst", NUMERICAL_LOCK_SHA256, P1_EXECUTION_TOKEN
            ),
            "s04-p1r1-matched-qwen2.5-7b-inst-905a3bbc",
        )

    def test_changed_paths_are_exact_and_foreign_free(self) -> None:
        self.assertEqual(len(submit.P1_CHANGED_PATHS), 6)
        self.assertTrue(
            all(
                path.startswith("project/run_scripts/ode_alloc/")
                or path.startswith("project/run_scripts/session04_ode_alloc_")
                for path in submit.P1_CHANGED_PATHS
            )
        )
        self.assertFalse(
            any("session03" in path or "knowledge-revision" in path for path in submit.P1_CHANGED_PATHS)
        )
        self.assertEqual(len(submit.P0_TERMINAL_FILES), 6)
        self.assertEqual(len(submit.P1_R0_IMMUTABLE_FILES), 8)
        self.assertEqual(len(submit.P1_FROZEN_SOURCE_FILES), 4)
        self.assertEqual(
            submit.P1_R0_RUNTIME_SHA256,
            "d6e271946c999d3226568c997be7783f90ce017a7f475b120bbc6b520265fa43",
        )

    def test_repair_source_gate_locks_runtime_and_exact_parent(self) -> None:
        source = Path(submit.__file__).read_text(encoding="utf-8")
        self.assertIn("parent != P1_REPAIR_EXPECTED_BASE", source)
        self.assertIn("changed != P1_CHANGED_PATHS", source)
        self.assertIn("sha256_file(runtime) != P1_R1_RUNTIME_SHA256", source)
        self.assertNotEqual(submit.P1_R1_RUNTIME_SHA256, "RUNTIME_SHA256_TO_LOCK")

    def test_pinned_artifact_receipt_uses_canonical_field_contract(self) -> None:
        fields = set(ArtifactReceipt.__dataclass_fields__)
        self.assertIn("counterfact", fields)
        self.assertIn("hparams", fields)
        source = Path(submit.__file__).read_text(encoding="utf-8")
        self.assertIn("receipt.counterfact", source)
        self.assertIn("receipt.hparams", source)
        self.assertNotIn("receipt.dataset_sha256", source)
        self.assertNotIn("receipt.hparams_sha256", source)

    def test_gpu_parser_and_pair_memory_fail_closed(self) -> None:
        self.assertEqual(submit._gpu_count("gres/gpu=2"), 2)
        self.assertEqual(submit._gpu_count("gpu:1"), 1)
        self.assertIsNone(submit._gpu_count("cpu=8"))
        self.assertLessEqual(
            submit.PAIR_HOST_MEMORY_MIB,
            2 * submit.MEMORY_CAP_MIB_PER_GPU,
        )
        self.assertEqual(tuple(submit.JOB_NAMES), MODEL_ALIASES)

    def test_entrypoint_success_exception_and_system_exit_contract(self) -> None:
        with mock.patch.object(submit, "main", return_value=0), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            self.assertEqual(submit._entrypoint(), 0)
            self.assertEqual(stderr.getvalue(), "")
        with mock.patch.object(submit, "main", side_effect=RuntimeError("x")), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            with self.assertRaises(RuntimeError):
                submit._entrypoint()
            self.assertIn("PRE_SUBMIT_HOLD", stderr.getvalue())
        with mock.patch.object(submit, "main", side_effect=SystemExit(0)), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            with self.assertRaises(SystemExit):
                submit._entrypoint()
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
