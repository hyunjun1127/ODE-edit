from __future__ import annotations

import inspect
import hashlib
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8SoftInventory,
    FunctionalBasisMetric,
)
from project.run_scripts.ode_bf.historical_h0_sequential_runtime import (
    HISTORY_COUNTS,
    METHODS,
    expected_historical_h0_result_name,
)
from project.run_scripts.ode_bf.historical_h0_sequential_selection import (
    verify_historical_h0_fresh_seal,
)
from project.run_scripts.ode_bf.p1_backend import _validate_history_keys
from project.run_scripts.ode_bf.strength_preserving_routing import _risk_functions
from project.run_scripts import session05_ode_bf_historical_h0_sequential_dry_plan as dry


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


class HistoricalH0SequentialTest(unittest.TestCase):
    def test_fresh_seal_and_lock_are_rooted(self) -> None:
        seal = verify_historical_h0_fresh_seal(
            json.loads(
                (LOCKS / "p1r20_historical_h0_fresh_cf_b100_seal.json").read_text()
            )
        )
        lock, _ = load_rooted_json(
            LOCKS / "numerical_lock_s05_historical_h0_sequential.json",
            expected_schema="ode-edit-s05-historical-h0-bg-sequential-p1r20-lock/v1",
        )
        self.assertEqual(len(seal["requests"]), 100)
        self.assertEqual(len(seal["batch_ordered_request_digest_v1"]), 10)
        self.assertEqual(lock["fresh_seal_root"], seal["root_digest"])
        self.assertEqual(lock["all_request_order_sha256"], seal["all_request_order_sha256"])
        self.assertEqual(lock["history"]["round_entry_counts"], list(HISTORY_COUNTS))

    def test_matrix_and_dry_plan_repeat(self) -> None:
        first = dry.build_plan("0" * 40)
        second = dry.build_plan("0" * 40)
        self.assertEqual(first, second)
        self.assertEqual(first["trajectory_count"], 8)
        self.assertEqual(first["array_max_concurrent_gpu"], 4)
        self.assertEqual(
            [job["method"] for job in first["jobs"][:4]], list(METHODS)
        )
        self.assertEqual(
            len({job["result_name"] for job in first["jobs"]}), 8
        )

    def test_result_namespace_is_alias_method_specific(self) -> None:
        names = {
            expected_historical_h0_result_name(alias, method)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 8)

    def test_alpha_history_accepts_ten_round_prefix_only(self) -> None:
        layers = (4, 5, 6, 7, 8)
        valid = {layer: torch.zeros((3, 90), dtype=torch.float32) for layer in layers}
        self.assertEqual(set(_validate_history_keys(valid, layers)), set(layers))
        invalid = {layer: torch.zeros((3, 100), dtype=torch.float32) for layer in layers}
        with self.assertRaisesRegex(Exception, "prior-B10 prefix"):
            _validate_history_keys(invalid, layers)

    def test_structural_only_excludes_all_functional_scores(self) -> None:
        barrier = SimpleNamespace(
            gram=np.eye(5, dtype=np.float64),
            linear=np.zeros(5, dtype=np.float64),
        )
        problem = SimpleNamespace(historical=barrier, pretrained=barrier)
        inventory = FixedE8SoftInventory(
            FunctionalBasisMetric("functional_p", 1.0, (2.0,) * 5, True),
            FunctionalBasisMetric("functional_h_mean", 1.0, (2.0,) * 5, True),
            FunctionalBasisMetric("functional_h_smoothmax", 1.0, (2.0,) * 5, True),
            10,
            "1" * 64,
            "2" * 64,
        )
        labels = [item[0] for item in _risk_functions(problem, inventory, structural_only=True)]
        self.assertEqual(labels, ["structural_historical", "structural_pretrained"])

    def test_sequential_path_removes_inner_k_probes_and_future_access(self) -> None:
        from project.run_scripts.ode_bf import historical_h0_sequential_runtime as runtime
        from project.run_scripts.ode_bf import common_coldcoord_fixed_e8_runtime as common

        runtime_source = inspect.getsource(runtime)
        common_source = inspect.getsource(common._run_common_arm)
        self.assertIn("inner_k_heldout_evaluation_count", runtime_source)
        self.assertIn("future_batch_controller_access_count", runtime_source)
        self.assertIn("if sequential_lightweight", common_source)
        self.assertNotIn("load_counterfact_cases_after_freeze", common_source)

    def test_launcher_is_cap_four_and_devbox_pinned(self) -> None:
        source = (
            ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --array=0-7%4", source)
        self.assertIn("#SBATCH --nodelist=devbox", source)
        self.assertIn("--mem=65000M", source)
        self.assertIn("--time=23:59:00", source)

    def test_source_manifest_rehashes_every_member(self) -> None:
        path = LOCKS / "source_manifest_s05_historical_h0_sequential.json"
        value = json.loads(path.read_text())
        root = value.pop("root_digest")
        from project.run_scripts.ode_bf.contracts import canonical_hash

        self.assertEqual(root, canonical_hash(value))
        self.assertEqual(value["entry_count"], len(value["entries"]))
        for item in value["entries"]:
            member = ROOT / item["path"]
            payload = member.read_bytes()
            self.assertEqual(member.stat().st_mode, item["mode"])
            self.assertEqual(len(payload), item["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            object_id = subprocess.run(
                ["git", "hash-object", str(member)],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(object_id, item["object_id"])


if __name__ == "__main__":
    unittest.main()
