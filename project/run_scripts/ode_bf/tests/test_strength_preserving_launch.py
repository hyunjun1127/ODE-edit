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
    session05_ode_bf_strength_preserving_router as entry,
    session05_ode_bf_strength_preserving_router_dry_plan as dry,
)
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.p1_strength_preserving_router_panel import (
    STRENGTH_PRESERVING_CELLS,
    expected_strength_preserving_result_name,
    load_and_validate_strength_preserving_lock,
)
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    common_cold_schedule,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


ROOT = Path(__file__).resolve().parents[4]


class StrengthPreservingLaunchTests(unittest.TestCase):
    def test_result_matrix_is_exact_two_by_two_by_two(self) -> None:
        names = {
            expected_strength_preserving_result_name(alias, cell)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for cell in STRENGTH_PRESERVING_CELLS
        }
        self.assertEqual(len(names), 8)
        self.assertTrue(all("target-hold" not in item for item in names))
        self.assertTrue(all("tech-r1" in item for item in names))

    def test_entry_dispatches_one_cell_only(self) -> None:
        with mock.patch.object(
            entry,
            "run_p1",
            return_value={"status": "PASS"},
        ) as observed:
            code = entry.main(
                [
                    "--model",
                    "qwen2.5-7b-inst",
                    "--cell",
                    "BG-SOFT",
                    "--output-root",
                    "/tmp/absent-p1r19-test",
                    "--source-head",
                    "a" * 40,
                    "--run-token",
                    "strength-preserving-router-p1r19-a2-tech-r1-v1",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(observed.call_count, 1)
        self.assertEqual(
            observed.call_args.kwargs["strength_preserving_cell"], "BG-SOFT"
        )

    def test_dry_plan_has_eight_tasks_and_cap_four(self) -> None:
        plan = dry.build_plan("a" * 40, repository_root=ROOT)
        self.assertEqual(plan["trajectory_count"], 8)
        self.assertEqual(plan["array_max_concurrent_gpu"], 4)
        self.assertEqual(plan["hold_count"], 0)
        self.assertEqual([item["array_index"] for item in plan["jobs"]], list(range(8)))
        self.assertTrue(all(item["forecast"]["fits_envelope"] for item in plan["jobs"]))

    def test_runtime_dispatch_has_no_alias_scientific_branch(self) -> None:
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

    def test_sbatch_has_dynamic_only_array_and_no_retry(self) -> None:
        source = (
            ROOT
            / "project/run_scripts/session05_ode_bf_strength_preserving_router.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-7%4", source)
        self.assertNotIn("TARGET-HOLD", source)
        self.assertNotIn("retry", source.lower())
        self.assertIn('REPO_ROOT="${SLURM_SUBMIT_DIR:', source)
        self.assertIn("#SBATCH --nodelist=server2", source)
        self.assertNotIn("#SBATCH --nodelist=devbox", source)

    def test_lock_binds_a2_zero_compute_and_structural_p(self) -> None:
        locks = ROOT / "project/run_scripts/ode_bf/locks"
        schedule = common_cold_schedule(
            load_p1_sampling_seal(
                locks / "p1r2_p_population_seal.json",
                stream_path=locks / "p1r2_seqb10_stream_seal.json",
            )
        )
        observed, _ = load_and_validate_strength_preserving_lock(
            locks / "numerical_lock_s05_strength_preserving_router.json",
            case_root_digest=(
                "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
            ),
            request_order_sha256=(
                "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
            ),
            schedule=schedule,
        )
        self.assertEqual(observed["router_added_target_backward_count"], 0)
        self.assertEqual(observed["router_added_processed_token_count"], 0)
        self.assertEqual(
            observed["structural_p_definition"],
            "tr(B_l*C0_l*B_l^T)_PINNED_WIKIPEDIA_SECOND_MOMENT",
        )

    def test_source_manifest_is_rooted_and_matches_every_declared_file(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "source_manifest_s05_strength_preserving_router.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        root_digest = value.pop("root_digest")
        from project.run_scripts.ode_bf.contracts import canonical_hash

        self.assertEqual(root_digest, canonical_hash(value))
        self.assertEqual(value["entry_count"], len(value["entries"]))
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
