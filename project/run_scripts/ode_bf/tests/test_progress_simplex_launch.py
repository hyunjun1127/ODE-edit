from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import unittest

from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    expected_p1r23_result_name,
)
from project.run_scripts.session05_ode_bf_progress_simplex_router_dry_plan import (
    ROLES,
)


ROOT = Path(__file__).resolve().parents[4]


class ProgressSimplexLaunchTests(unittest.TestCase):
    def test_progress_simplex_roles_have_four_distinct_paired_names(self) -> None:
        observed = {
            expected_p1r23_result_name(alias, batch_size=10, role=role)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for role in ROLES
        }
        self.assertEqual(len(observed), 4)
        self.assertTrue(all("progress-simplex" in item for item in observed))
        self.assertTrue(all("native" not in item for item in observed))
        with self.assertRaises(ODEBFContractError):
            expected_p1r23_result_name(
                "llama3-8b-inst",
                batch_size=100,
                role="PROGRESS_SIMPLEX_BG_PAIR",
                routing_arm="SIMPLEX-SOFT",
            )

    def test_progress_simplex_lock_and_launcher_are_atomic_b10_only(self) -> None:
        lock, _ = load_rooted_json(
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_progress_simplex_router.json",
            expected_schema="ode-edit-s05-p1r23-progress-simplex-router-lock/v1",
        )
        self.assertEqual(lock["execution"]["batch_size"], 10)
        self.assertEqual(lock["execution"]["trajectory_count"], 8)
        self.assertEqual(lock["execution"]["native_rerun_count"], 0)
        self.assertEqual(lock["execution"]["B100_access_count"], 0)
        self.assertEqual(lock["scientific_grid"]["persistent_history_append_count"], 0)
        script = (
            ROOT / "project/run_scripts/session05_ode_bf_progress_simplex_router.sbatch"
        ).read_text(encoding="utf-8")
        self.assertNotIn("0-3%4", script)
        self.assertIn("PROGRESS_SIMPLEX_BG_PAIR", script)
        self.assertIn("PROGRESS_SIMPLEX_RS_PAIR", script)
        self.assertNotIn("OPTIMIZED_NATIVE", script)
        self.assertNotIn("OFFICIAL_NATIVE", script)
        self.assertNotIn("B100", script)

    def test_original_b10_reproduction_launcher_is_server1_provenance_only(self) -> None:
        script = (
            ROOT / "project/run_scripts/session05_ode_bf_progress_simplex_router.sbatch"
        ).read_text(encoding="utf-8")
        submitter = (
            ROOT
            / "project/run_scripts/session05_ode_bf_submit_progress_simplex_router.py"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --nodelist=devbox", script)
        self.assertIn("p1r23-progress-simplex-original-b10-repro-v1", script)
        self.assertIn("codex/p1r23-progress-simplex-original-b10-repro-v1", script)
        self.assertIn("ReqNodeList=devbox", submitter)
        self.assertIn("original-b10-repro", submitter)

    def test_lock_is_raw_json_and_has_no_outcome_tuning(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_progress_simplex_router.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["numerical"]["outcome_tuning_count"], 0)
        self.assertEqual(value["progress_simplex"]["per_layer_upper_cap_count"], 0)
        self.assertEqual(value["progress_simplex"]["postsolve_clip_count"], 0)
        self.assertEqual(value["global_trust"]["model_specific_budget_count"], 0)

    def test_source_manifest_binds_exact_scientific_checkpoint_objects(self) -> None:
        manifest, _ = load_rooted_json(
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "source_manifest_s05_progress_simplex_router.json",
            expected_schema="ode-edit-s05-p1r23-progress-simplex-source-manifest/v1",
        )
        head = manifest["scientific_checkpoint"]
        self.assertEqual(len(manifest["entries"]), manifest["entry_count"])
        for item in manifest["entries"]:
            tree = subprocess.run(
                ["git", "ls-tree", head, "--", item["path"]],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip().split(None, 3)
            payload = subprocess.run(
                ["git", "show", f"{head}:{item['path']}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertEqual(tree[0], item["mode"])
            self.assertEqual(tree[2], item["object_id"])
            self.assertEqual(len(payload), item["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
