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
    session05_ode_bf_atomic_runtime_optimization as entry,
    session05_ode_bf_atomic_runtime_optimization_dry_plan as dry,
    session05_ode_bf_submit_atomic_runtime_conformance as conformance_submit,
    session05_ode_bf_submit_atomic_runtime_optimization as production_submit,
)
from project.run_scripts.ode_bf import p1_atomic_runtime_optimization as runtime
from project.run_scripts.ode_bf.p1_runtime import run_p1


ROOT = Path(__file__).resolve().parents[4]


class AtomicRuntimeOptimizationLaunchTests(unittest.TestCase):
    def test_entry_dispatches_exact_opt_in_modes(self) -> None:
        with mock.patch.object(entry, "run_p1", return_value={"status": "PASS"}) as call:
            code = entry.main(
                [
                    "--model", "qwen2.5-7b-inst",
                    "--output-root", "/tmp/absent-p1r22-test",
                    "--source-head", "a" * 40,
                    "--run-token", "atomic-runtime-optimization-p1r22-v1",
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue(call.call_args.kwargs["atomic_runtime_optimization_mode"])
        self.assertFalse(call.call_args.kwargs["atomic_runtime_conformance_mode"])
        with mock.patch.object(entry, "run_p1", return_value={"status": "PASS"}) as call:
            code = entry.main(
                [
                    "--model", "llama3-8b-inst",
                    "--output-root", "/tmp/absent-p1r22-c0c1-test",
                    "--source-head", "b" * 40,
                    "--run-token", "atomic-runtime-optimization-p1r22-v1",
                    "--conformance-only",
                ]
            )
        self.assertEqual(code, 0)
        self.assertFalse(call.call_args.kwargs["atomic_runtime_optimization_mode"])
        self.assertTrue(call.call_args.kwargs["atomic_runtime_conformance_mode"])

    def test_dry_plan_has_two_jobs_four_cells_and_cap_two(self) -> None:
        plan = dry.build_plan("a" * 40, repository_root=ROOT)
        self.assertEqual(plan["model_job_count"], 2)
        self.assertEqual(plan["trajectory_count"], 4)
        self.assertEqual(plan["array_max_concurrent_gpu"], 2)
        self.assertEqual([job["cells"] for job in plan["jobs"]], [
            ["BG-NEUTRAL", "BG-SOFT"],
            ["BG-NEUTRAL", "BG-SOFT"],
        ])
        self.assertTrue(all(job["forecast"]["fits_envelope"] for job in plan["jobs"]))

    def test_launchers_are_server2_array_two_and_have_no_retry(self) -> None:
        for name in (
            "session05_ode_bf_atomic_runtime_optimization.sbatch",
            "session05_ode_bf_atomic_runtime_conformance.sbatch",
        ):
            source = (ROOT / "project/run_scripts" / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-1%2", source)
            self.assertIn("#SBATCH --nodelist=server2", source)
            self.assertIn('REPO_ROOT="${SLURM_SUBMIT_DIR:', source)
            self.assertNotIn("retry", source.lower())
            self.assertNotIn("target-hold", source.lower())

    def test_submitters_count_only_running_janghj_gpu_jobs(self) -> None:
        for module in (production_submit, conformance_submit):
            source = inspect.getsource(module._active_user_gpu_jobs)
            self.assertIn('"-t", "R"', source)
            self.assertNotIn(" PENDING", source)
            self.assertNotIn("PD", source)
            self.assertIn('"gpu" in line.lower()', source)

    def test_runtime_has_no_alias_scientific_branch(self) -> None:
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

    def test_terminal_evaluation_is_w_only_and_inner_heldout_is_absent(self) -> None:
        source = inspect.getsource(runtime.run_p1r22_atomic_optimization)
        self.assertIn("_evaluate_stepwise_state", source)
        self.assertNotIn("z_oracle", source.lower())
        arm_source = inspect.getsource(runtime._run_arm)
        self.assertNotIn("evaluate_counterfact", arm_source)
        self.assertIn('"inner_step_heldout_evaluation_count": 0', arm_source)

    def test_source_manifest_is_rooted_and_matches_declared_files(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "source_manifest_s05_atomic_runtime_optimization.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        root_digest = value.pop("root_digest")
        from project.run_scripts.ode_bf.contracts import canonical_hash

        self.assertEqual(root_digest, canonical_hash(value))
        self.assertEqual(value["entry_count"], len(value["entries"]))
        self.assertEqual(
            value["expected_parent"],
            "d604953a046b6af4da7386dd6c64c20fc848f625",
        )
        self.assertEqual(
            value["execution_head_policy"],
            "RUNTIME_GIT_HEAD_BOUND_BY_SUBMISSION_RECEIPT",
        )
        for item in value["entries"]:
            source = ROOT / item["path"]
            payload = source.read_bytes()
            self.assertEqual(source.stat().st_size, item["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            object_id = subprocess.run(
                ["git", "hash-object", str(source)],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(object_id, item["object_id"])


if __name__ == "__main__":
    unittest.main()
