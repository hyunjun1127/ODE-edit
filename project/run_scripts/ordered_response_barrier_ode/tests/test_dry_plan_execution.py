from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ordered_response_barrier_ode import dry_plan, preflight


class DryPlanResourceTests(unittest.TestCase):
    def test_create_once_parent_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root / "outside"
            outside.mkdir()
            link = root / "link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(preflight.PreflightBoundary):
                dry_plan._require_create_once_target(link / "result")

    def test_squeue_count_and_pattern_scope(self) -> None:
        text = "1|odeedit_alpha|RUNNING|gpu:a6000:2\n2|other|RUNNING|gpu:a6000:4\n3|odeedit_beta|PENDING|gpu:a6000:1\n"
        total, rows = dry_plan.parse_squeue_gpu_rows(text, "odeedit_*")
        self.assertEqual(total, 2)
        self.assertEqual([item["job_id"] for item in rows], ["1"])

    def test_live_throttle_respects_cap(self) -> None:
        cap = {
            "node": "devbox",
            "job_patterns": "odeedit_*",
            "max_project_gpus": 4,
        }

        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "1|odeedit_running|RUNNING|gpu:a6000:1\n", "")

        value = dry_plan.query_live_resources(cap, runner=runner)
        self.assertEqual(value["active_project_gpus"], 1)
        self.assertEqual(value["throttle"], 3)
        self.assertEqual(value["post_release_max"], 4)

    def test_no_free_slot_fails_closed(self) -> None:
        cap = {"node": "devbox", "job_patterns": "odeedit_*", "max_project_gpus": 2}

        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "1|odeedit_running|RUNNING|gpu:a6000:2\n", "")

        with self.assertRaises(preflight.PreflightBoundary):
            dry_plan.query_live_resources(cap, runner=runner)

    def test_slurm_scripts_have_locked_resource_and_wave_contracts(self) -> None:
        root = Path(__file__).resolve().parents[4]
        round0 = (root / "project/run_scripts/session06_orbode_server1_round0.sbatch").read_text()
        remaining = (root / "project/run_scripts/session06_orbode_server1_remaining.sbatch").read_text()
        for text in (round0, remaining):
            self.assertEqual(text.count("#SBATCH --mem="), 1)
            self.assertIn("#SBATCH --mem=182272M", text)
            self.assertIn("#SBATCH --export=NONE", text)
            self.assertIn("#SBATCH --gres=gpu:1", text)
            self.assertIn("#SBATCH --array=0-3%4", text)
            self.assertIn("easyeditsh1-official-readonly-v1", text)
            self.assertIn("IMPORTED_EASYEDIT", text)
            self.assertNotIn("PYTHONPATH=\"${REPO_ROOT}:${EASYEDIT_ARTIFACT_ROOT}", text)
        self.assertIn("--wave round0", round0)
        self.assertNotIn("--round0-root", round0)
        self.assertIn("--wave remaining", remaining)
        self.assertIn("--round0-root", remaining)

    def test_plan_declares_unique_slurm_log_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "state").mkdir()
            (root / "logs").mkdir()
            cap = {
                "job_patterns": "odeedit_*",
                "max_project_gpus": 4,
                "mem_mib_per_gpu": 183_296,
                "node": "devbox",
                "server": "server1",
            }
            with (
                mock.patch.object(dry_plan, "parse_local_cap", return_value=cap),
                mock.patch.object(
                    dry_plan,
                    "run_preflight",
                    return_value={"receipt_identity_sha256": "a" * 64},
                ),
            ):
                plan = dry_plan.build_dry_plan(
                    repo_root=root,
                    authoritative_root=root,
                    easyedit_source_root=root,
                    easyedit_artifact_root=root,
                    hf_hub_cache=root,
                    source_head="h",
                    source_tree="t",
                    wave="round0",
                    output_root=root / "state/wave-r1",
                    log_root=root / "logs/wave-r1",
                    local_caps=root / "caps.tsv",
                    live_resources=False,
                )
            self.assertEqual(
                plan["sbatch"]["cli_overrides"]["output"],
                str(root / "logs/wave-r1/%A_%a.out"),
            )
            self.assertEqual(
                plan["sbatch"]["cli_overrides"]["error"],
                str(root / "logs/wave-r1/%A_%a.err"),
            )


if __name__ == "__main__":
    unittest.main()
