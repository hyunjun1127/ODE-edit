from __future__ import annotations

import json
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.artifacts import (
    load_rooted_json,
    sha256_regular_tree,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.firewall import evaluator_request_hash
from project.run_scripts.ode_bf.resource import (
    GPU_CAP_SERVER2,
    GPU_MEMORY_REQUEST_MIB,
    SchedulerJob,
    assert_pair_capacity,
    forecast_p0_b10_memory,
    gpu_count_from_tres,
    node_local_gpu_totals,
)
from project.run_scripts.ode_bf.selection import (
    build_p0_b10_seal,
    load_sealed_joint_requests,
    verify_p0_b10_seal,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh2/ODE-edit")
LOCK_ROOT = PACKAGE_ROOT / "locks"
BASE_LOCK = REPO_ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"


class SelectionSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact, _ = load_rooted_json(
            LOCK_ROOT / "p0_artifact_lock.json",
            expected_schema="ode-edit-s04-ode-bf-p0-artifact-lock/v1",
        )
        base = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
        cls.dataset = Path(base["easyedit_root"]) / base["counterfact"]["path"]
        cls.artifact = artifact

    def test_p0_b10_seal_rebuild_is_byte_semantically_exact(self) -> None:
        locked = json.loads((LOCK_ROOT / "p0_b10_seal.json").read_text(encoding="utf-8"))
        rebuilt = build_p0_b10_seal(self.dataset, REPO_ROOT)
        self.assertEqual(rebuilt, locked)
        self.assertEqual(verify_p0_b10_seal(rebuilt)["edit_batch_size"], 10)

    def test_sealed_request_projection_reads_only_rewrite_surface(self) -> None:
        seal = json.loads((LOCK_ROOT / "p0_b10_seal.json").read_text(encoding="utf-8"))
        requests = load_sealed_joint_requests(self.dataset, seal)
        self.assertEqual(len(requests), 10)
        self.assertEqual(len({item["case_id"] for item in requests}), 10)
        self.assertEqual(
            [evaluator_request_hash(item) for item in requests],
            [item["request_sha256"] for item in requests],
        )
        forbidden = {"paraphrase_prompts", "neighborhood_prompts", "generation_prompts"}
        self.assertTrue(all(not forbidden.intersection(item) for item in requests))


class ArtifactAndResourceTests(unittest.TestCase):
    def test_rooted_locks_and_held_ode_alloc_tree_are_exact(self) -> None:
        for name, schema in (
            ("numerical_lock_p0.json", "ode-edit-s04-ode-bf-p0-numerical-lock/v1"),
            ("p0_b10_seal.json", "ode-edit-s04-ode-bf-p0-b10-seal/v1"),
            ("p0_cpu_sampling_seal.json", "ode-edit-s04-ode-bf-p0-cpu-sampling-seal/v1"),
            ("p0_artifact_lock.json", "ode-edit-s04-ode-bf-p0-artifact-lock/v1"),
        ):
            value, digest = load_rooted_json(LOCK_ROOT / name, expected_schema=schema)
            self.assertEqual(len(value["root_digest"]), 64)
            self.assertEqual(len(digest), 64)
        artifact, _ = load_rooted_json(LOCK_ROOT / "p0_artifact_lock.json")
        held = artifact["held_ode_alloc_local"]
        self.assertEqual(
            sha256_regular_tree(REPO_ROOT / held["relative_path"]),
            (held["tree_sha256"], held["file_count"]),
        )

    def test_genuine_b10_forecast_fits_without_batch_reduction(self) -> None:
        forecasts = {
            alias: forecast_p0_b10_memory(
                LOCK_ROOT / "p0_artifact_lock.json",
                BASE_LOCK,
                alias,
            )
            for alias in MODEL_ALIASES
        }
        self.assertEqual(set(forecasts), set(MODEL_ALIASES))
        for forecast in forecasts.values():
            self.assertEqual(forecast.edit_batch_size, 10)
            self.assertEqual(forecast.factor_rank_per_waypoint, 10)
            self.assertEqual(forecast.k_total, 8)
            self.assertLess(forecast.forecast_gpu_peak_mib, GPU_MEMORY_REQUEST_MIB)
            self.assertLess(forecast.forecast_host_peak_mib, GPU_MEMORY_REQUEST_MIB)
            self.assertFalse(forecast.dense_full_history_matrix)

    def test_node_local_cap_excludes_devbox_and_counts_pending_unknown(self) -> None:
        records = (
            SchedulerJob("1", "odeedit_s03_x", "RUNNING", 2, ("devbox",), ("devbox",)),
            SchedulerJob("2", "unrelated", "RUNNING", 3, ("server2",), ("server2",)),
        )
        self.assertEqual(node_local_gpu_totals(records), (0, 2))
        self.assertEqual(assert_pair_capacity(records), (0, 2))
        pending = records + (
            SchedulerJob("3", "odebf_pending", "PENDING", 1, (), ()),
        )
        self.assertEqual(node_local_gpu_totals(pending), (1, 3))
        self.assertEqual(assert_pair_capacity(pending), (1, 3))
        over = pending + (
            SchedulerJob("4", "odealloc_local", "RUNNING", 1, ("server2",), ("server2",)),
        )
        with self.assertRaisesRegex(ODEBFContractError, "GPU cap"):
            assert_pair_capacity(over)
        self.assertEqual(GPU_CAP_SERVER2, 3)

    def test_gpu_tres_parser_is_fail_closed(self) -> None:
        self.assertEqual(gpu_count_from_tres("gres/gpu:1"), 1)
        self.assertEqual(gpu_count_from_tres("gres/gpu:a6000:2(S:0-1)"), 2)
        self.assertEqual(gpu_count_from_tres("gpu=1"), 1)
        self.assertIsNone(gpu_count_from_tres("N/A"))


if __name__ == "__main__":
    unittest.main()
