from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import (
    session05_ode_bf_scalable_batched_runtime as entry,
    session05_ode_bf_scalable_batched_runtime_dry_plan as dry,
)
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.contracts import canonical_hash


ROOT = Path(__file__).resolve().parents[4]


class ScalableBatchedLaunchTests(unittest.TestCase):
    def test_entry_dispatches_exact_atomic_role_and_batch(self) -> None:
        with mock.patch.object(entry, "run_p1", return_value={"status": "PASS"}) as call:
            code = entry.main(
                [
                    "--model",
                    "qwen2.5-7b-inst",
                    "--batch-size",
                    "100",
                    "--role",
                    "ODE_BF_K8_PAIR",
                    "--output-root",
                    "/tmp/absent-p1r23-test",
                    "--source-head",
                    "a" * 40,
                    "--run-token",
                    "scalable-batched-runtime-p1r23-v1",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["scalable_batched_role"], "ODE_BF_K8_PAIR")
        self.assertEqual(call.call_args.kwargs["scalable_batched_batch_size"], 100)

    def test_dry_plans_are_atomic_and_have_exact_role_counts(self) -> None:
        calibration = dry.build_plan("a" * 40, "calibration", repository_root=ROOT)
        b10 = dry.build_plan("a" * 40, "b10", repository_root=ROOT)
        b100 = dry.build_plan("a" * 40, "b100", repository_root=ROOT)
        self.assertEqual(calibration["job_count"], 2)
        self.assertEqual(b10["job_count"], 6)
        self.assertEqual(b100["job_count"], 6)
        for value in (calibration, b10, b100):
            self.assertEqual(value["estimand"], "ATOMIC")
            self.assertEqual(value["sequential_round_count"], 0)
            self.assertEqual(value["persistent_history_append_count"], 0)
            self.assertLessEqual(value["array_max_concurrent_gpu"], 4)
        self.assertEqual(
            {item["role"] for item in b10["jobs"]},
            {"ODE_BF_K8_PAIR", "OPTIMIZED_NATIVE_K1", "OFFICIAL_NATIVE"},
        )

    def test_sbatch_has_server2_atomic_stage_and_no_static_split(self) -> None:
        source = (
            ROOT
            / "project/run_scripts/session05_ode_bf_scalable_batched_runtime.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --nodelist=server2", source)
        self.assertIn('REPO_ROOT="${SLURM_SUBMIT_DIR:', source)
        self.assertIn("ODE_BF_K8_PAIR", source)
        self.assertNotIn("P1R20", source)
        self.assertNotIn("B10×10", source)
        self.assertNotIn("TARGET_HOLD", source)

    def test_runtime_has_no_alias_specific_scientific_branch(self) -> None:
        tree = ast.parse(inspect.getsource(run_p1))
        alias_comparisons = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(item, ast.Constant)
                and item.value in ("llama3-8b-inst", "qwen2.5-7b-inst")
                for item in (node.left, *node.comparators)
            )
        ]
        self.assertEqual(alias_comparisons, [])

    def test_source_manifest_is_rooted_and_matches_declared_files(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "source_manifest_s05_scalable_batched_runtime.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        root_digest = value.pop("root_digest")
        self.assertEqual(root_digest, canonical_hash(value))
        self.assertEqual(value["entry_count"], len(value["entries"]))
        for item in value["entries"]:
            source = ROOT / item["path"]
            payload = source.read_bytes()
            self.assertEqual(len(payload), item["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            observed = subprocess.run(
                ["git", "hash-object", str(source)],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(observed, item["object_id"])


if __name__ == "__main__":
    unittest.main()
