from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from project.run_scripts import (
    session05_ode_bf_submit_universal_observability as submit,
    session05_ode_bf_universal_observability as entry,
    session05_ode_bf_universal_observability_dry_plan as dry,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.ode_bf_observability import paired_arm_ids
from project.run_scripts.ode_bf.p1_universal_observability_panel import (
    UNIVERSAL_OBS_RESULT_TOKEN,
    expected_universal_observability_result_name,
)


ROOT = Path(__file__).resolve().parents[4]


class UniversalObservabilityLaunchTests(unittest.TestCase):
    SOURCE_HEAD = "d" * 40

    @staticmethod
    def _git_run(
        source_head: str,
        *,
        head: str | None = None,
        parent: str | None = None,
        branch: str | None = None,
        dirty: str = "",
        ancestor_returncode: int = 0,
    ):
        outputs = {
            ("git", "rev-parse", "HEAD"): (head or source_head) + "\n",
            ("git", "rev-parse", "HEAD^"): (
                parent or submit.EXECUTION_PARENT
            )
            + "\n",
            ("git", "branch", "--show-current"): (
                branch or submit.EXECUTION_BRANCH
            )
            + "\n",
            ("git", "status", "--porcelain", "--untracked-files=no"): dirty,
        }

        def run(args, *, check=True):
            key = tuple(args)
            if key == (
                "git",
                "merge-base",
                "--is-ancestor",
                submit.UNIVERSAL_OBS_BASE_RUNTIME,
                head or source_head,
            ):
                return SimpleNamespace(
                    stdout="", stderr="", returncode=ancestor_returncode
                )
            return SimpleNamespace(stdout=outputs[key], stderr="", returncode=0)

        return run

    @staticmethod
    def _write_owned_wave(
        state_root: Path,
        jobs: list[dict[str, object]],
        *,
        source_head: str,
        job_ids: dict[str, str],
        receipt_source_head: str | None = None,
    ) -> None:
        wave_token = submit._wave_token(jobs)
        for job in jobs:
            cell_id = str(job["cell_id"])
            submit._write_once(
                submit._held_job_path(state_root, cell_id),
                {
                    "schema": f"{submit.EXECUTION_SCHEMA_NAMESPACE}-held-job/v1",
                    "instruction_id": submit.UNIVERSAL_OBS_INSTRUCTION_ID,
                    "amendment_id": submit.UNIVERSAL_OBS_AMENDMENT_ID,
                    "execution_amendment": submit.EXECUTION_AMENDMENT,
                    "source_head": receipt_source_head or source_head,
                    "alias": submit.LLAMA_ALIAS,
                    "cell_id": cell_id,
                    "job_name": job["job_name"],
                    "run_token": job["run_token"],
                    "job_id": job_ids[cell_id],
                    "scheduler_hold": True,
                    "submission_namespace": submit.SUBMISSION_NAMESPACE,
                    "wave_token": wave_token,
                },
            )
        submit._write_once(
            submit._wave_state_paths(state_root, jobs)["submission_receipt"],
            {
                "schema": (
                    f"{submit.EXECUTION_SCHEMA_NAMESPACE}-submission-receipt/v1"
                ),
                "instruction_id": submit.UNIVERSAL_OBS_INSTRUCTION_ID,
                "amendment_id": submit.UNIVERSAL_OBS_AMENDMENT_ID,
                "execution_amendment": submit.EXECUTION_AMENDMENT,
                "source_head": receipt_source_head or source_head,
                "intent_sha256": "0" * 64,
                "alias": submit.LLAMA_ALIAS,
                "submission_namespace": submit.SUBMISSION_NAMESPACE,
                "wave_token": wave_token,
                "jobs": job_ids,
                "submitted_cells": [str(job["cell_id"]) for job in jobs],
                "deferred_cells_due_to_capacity": [],
                "all_accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )

    def test_dry_plan_is_llama_only_and_preserves_the_three_cell_pairs(self) -> None:
        plan = dry.build_plan("f" * 40, repository_root=ROOT)

        self.assertTrue(plan["llama_sh1_only"])
        self.assertFalse(plan["qwen_submission_authorized"])
        self.assertTrue(plan["execution_authorized"])
        self.assertTrue(plan["execution_checkpoint_bound"])
        self.assertFalse(plan["execution_hold"])
        self.assertTrue(plan["held_atomic_release_required"])
        self.assertEqual(plan["execution_amendment"], "A2")
        self.assertEqual(plan["execution_head_policy"], "runtime-git-head")
        self.assertEqual(plan["execution_parent"], submit.EXECUTION_PARENT)
        self.assertEqual(plan["approval_token"], "A2:" + "f" * 40)
        self.assertFalse(plan["slurm_submit"])
        self.assertEqual(plan["server1_project_gpu_cap"], 4)
        self.assertEqual(plan["fast_launch_cell_order"], list(dry.CELL_ORDER))
        self.assertEqual(
            [job["cell_id"] for job in plan["jobs"]], list(dry.CELL_ORDER)
        )
        self.assertEqual(
            [job["job_name"] for job in plan["jobs"]],
            [dry.JOB_NAMES[cell_id] for cell_id in dry.CELL_ORDER],
        )
        for job in plan["jobs"]:
            cell_id = job["cell_id"]
            self.assertEqual(job["alias"], dry.LLAMA_ALIAS)
            self.assertEqual(job["live_arms"], list(paired_arm_ids(cell_id)))
            self.assertEqual(
                job["result_name"],
                expected_universal_observability_result_name(
                    dry.LLAMA_ALIAS, cell_id
                ),
            )
            self.assertEqual(
                job["runtime_kwargs"],
                {"universal_observability_cell": cell_id},
            )
            self.assertEqual(
                (job["gpu"], job["cpu"], job["memory_mib"], job["time"]),
                (1, 8, 65_000, "23:59:00"),
            )

    def test_capacity_selection_obeys_the_fast_cell_order(self) -> None:
        plan = dry.build_plan("e" * 40, repository_root=ROOT)
        jobs = submit._validate_plan(plan)
        expected = list(dry.CELL_ORDER)
        for active_gpu, count in ((0, 3), (1, 3), (2, 2), (3, 1)):
            with self.subTest(active_gpu=active_gpu):
                selected = submit._select_fast_launch_jobs(
                    jobs, active_project_gpu=active_gpu
                )
                self.assertEqual(
                    [job["cell_id"] for job in selected], expected[:count]
                )
                self.assertLessEqual(active_gpu + len(selected), 4)
        with self.assertRaisesRegex(ODEBFContractError, "cap is full"):
            submit._select_fast_launch_jobs(jobs, active_project_gpu=4)
        with self.assertRaisesRegex(ODEBFContractError, "active GPU count"):
            submit._select_fast_launch_jobs(jobs, active_project_gpu=-1)
        oversized = [dict(job) for job in jobs]
        oversized[0]["gpu"] = 2
        with self.assertRaisesRegex(ODEBFContractError, "cap would be exceeded"):
            submit._select_fast_launch_jobs(oversized, active_project_gpu=3)

    def test_a2_provenance_positive_binding_uses_the_runtime_head(self) -> None:
        source_head = self.SOURCE_HEAD
        with mock.patch.object(
            submit, "_run", side_effect=self._git_run(source_head)
        ):
            with mock.patch.dict(
                os.environ,
                {submit.APPROVAL_ENV: f"A2:{source_head}"},
                clear=True,
            ):
                receipt = submit._execution_provenance_gate(source_head)
        self.assertEqual(receipt["checkpoint_bound_approval"], f"A2:{source_head}")
        self.assertEqual(receipt["execution_head"], source_head)
        self.assertEqual(receipt["exact_execution_parent"], submit.EXECUTION_PARENT)
        self.assertEqual(receipt["branch"], submit.EXECUTION_BRANCH)
        self.assertTrue(receipt["tracked_and_index_clean"])

    def test_a2_provenance_rejects_every_missing_or_different_binding(self) -> None:
        source_head = self.SOURCE_HEAD
        with mock.patch.object(
            submit, "_run", side_effect=self._git_run(source_head)
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ODEBFContractError, "approval"):
                    submit._execution_provenance_gate(source_head)
        failures = (
            ("head", {"head": "e" * 40}),
            ("parent", {"parent": "e" * 40}),
            ("branch", {"branch": "wrong-branch"}),
            ("dirty", {"dirty": " M tracked.py\n"}),
            ("ancestor", {"ancestor_returncode": 1}),
        )
        for label, kwargs in failures:
            with self.subTest(label=label):
                with mock.patch.object(
                    submit,
                    "_run",
                    side_effect=self._git_run(source_head, **kwargs),
                ):
                    with mock.patch.dict(
                        os.environ,
                        {submit.APPROVAL_ENV: f"A2:{source_head}"},
                        clear=True,
                    ):
                        with self.assertRaisesRegex(ODEBFContractError, "provenance"):
                            submit._execution_provenance_gate(source_head)
        with self.assertRaisesRegex(ODEBFContractError, "source head"):
            submit._execution_provenance_gate("not-a-commit")

    def test_namespace_roots_are_create_once_and_fail_closed(self) -> None:
        jobs = submit._validate_plan(
            dry.build_plan("c" * 40, repository_root=ROOT)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "local/odebf/state"
            state_root.mkdir(parents=True)
            with mock.patch.object(submit, "REPO_ROOT", root):
                baseline = submit._namespace_gates(state_root, jobs)
            self.assertEqual(set(baseline["result_roots"]), set(dry.CELL_ORDER))
        for label in ("result", "log", "state"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state_root = root / "local/odebf/state"
                state_root.mkdir(parents=True)
                if label == "result":
                    (
                        root
                        / "local/odebf/results"
                        / expected_universal_observability_result_name(
                            dry.LLAMA_ALIAS, dry.CELL_ORDER[0]
                        )
                    ).mkdir(parents=True)
                elif label == "log":
                    log = root / "local/odebf/logs"
                    log.mkdir(parents=True)
                    (log / f"{dry.JOB_NAMES[dry.CELL_ORDER[0]]}-123.out").touch()
                else:
                    submit._wave_state_paths(state_root, jobs)["intent"].touch()
                with mock.patch.object(submit, "REPO_ROOT", root):
                    with self.assertRaisesRegex(ODEBFContractError, f"{label}.*collides"):
                        submit._namespace_gates(state_root, jobs)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ODEBFContractError, "state root"):
                submit._state_root_gate(Path(directory))

    def test_two_wave_cap_selection_skips_owned_rs_cells_and_selects_bg(self) -> None:
        source_head = "b" * 40
        jobs = submit._validate_plan(
            dry.build_plan(source_head, repository_root=ROOT)
        )
        first_wave = submit._select_fast_launch_jobs(jobs, active_project_gpu=2)
        self.assertEqual(
            [job["cell_id"] for job in first_wave], ["RS-NEUTRAL", "RS-SOFT"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "local/odebf/state"
            first_ids = {"RS-NEUTRAL": "101", "RS-SOFT": "102"}
            self._write_owned_wave(
                state_root,
                first_wave,
                source_head=source_head,
                job_ids=first_ids,
            )
            scheduler = [
                {
                    "job_id": "100",
                    "job_name": "odeedit_s05_r12_existing_llama",
                    "state": "RUNNING",
                    "gpu": 1,
                    "project_job": True,
                },
                *[
                    {
                        "job_id": first_ids[str(job["cell_id"])],
                        "job_name": job["job_name"],
                        "state": "PENDING",
                        "gpu": 1,
                        "project_job": True,
                    }
                    for job in first_wave
                ],
            ]
            with mock.patch.object(submit, "REPO_ROOT", root):
                ownership = submit._load_prior_cell_ownership(
                    state_root, jobs, source_head=source_head
                )
                submit._validate_prior_wave_receipts(
                    state_root, ownership, source_head=source_head
                )
                submit._reconcile_cell_ownership(
                    state_root, jobs, ownership, scheduler
                )
                remaining = [
                    job for job in jobs if str(job["cell_id"]) not in ownership
                ]
                active = sum(row["gpu"] for row in scheduler if row["project_job"])
                second_wave = submit._select_fast_launch_jobs(
                    remaining, active_project_gpu=active
                )
                namespaces = submit._namespace_gates(state_root, second_wave)
            self.assertLess(active, 4)
            self.assertEqual([job["cell_id"] for job in second_wave], ["BG-NEUTRAL"])
            self.assertEqual(namespaces["wave_token"], "bg_neutral")
            self.assertNotEqual(
                submit._wave_state_paths(state_root, first_wave)["intent"],
                submit._wave_state_paths(state_root, second_wave)["intent"],
            )

    def test_prior_ownership_rejects_duplicate_wrong_state_and_unowned_root(self) -> None:
        source_head = "a" * 40
        jobs = submit._validate_plan(
            dry.build_plan(source_head, repository_root=ROOT)
        )
        first = [jobs[0]]
        scheduler = [
            {
                "job_id": "701",
                "job_name": first[0]["job_name"],
                "state": "PENDING",
                "gpu": 1,
                "project_job": True,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "local/odebf/state"
            self._write_owned_wave(
                state_root,
                first,
                source_head=source_head,
                job_ids={"RS-NEUTRAL": "701"},
            )
            with mock.patch.object(submit, "REPO_ROOT", root):
                ownership = submit._load_prior_cell_ownership(
                    state_root, jobs, source_head=source_head
                )
                submit._validate_prior_wave_receipts(
                    state_root, ownership, source_head=source_head
                )
                duplicate = scheduler + [dict(scheduler[0], job_id="702")]
                with self.assertRaisesRegex(ODEBFContractError, "duplicates"):
                    submit._reconcile_cell_ownership(
                        state_root, jobs, ownership, duplicate
                    )
                wrong = [dict(scheduler[0], job_id="702")]
                with self.assertRaisesRegex(ODEBFContractError, "receipt differs"):
                    submit._reconcile_cell_ownership(
                        state_root, jobs, ownership, wrong
                    )
                (
                    root
                    / "local/odebf/results"
                    / expected_universal_observability_result_name(
                        dry.LLAMA_ALIAS, "BG-NEUTRAL"
                    )
                ).mkdir(parents=True)
                with self.assertRaisesRegex(ODEBFContractError, "lacks held-job ownership"):
                    submit._reconcile_cell_ownership(
                        state_root, jobs, ownership, scheduler
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "local/odebf/state"
            self._write_owned_wave(
                state_root,
                first,
                source_head=source_head,
                receipt_source_head="f" * 40,
                job_ids={"RS-NEUTRAL": "701"},
            )
            with self.assertRaisesRegex(ODEBFContractError, "held-job receipt"):
                submit._load_prior_cell_ownership(
                    state_root, jobs, source_head=source_head
                )

    def test_entrypoint_and_sbatch_forward_one_explicit_cell(self) -> None:
        source = inspect.getsource(entry.main)
        self.assertIn("universal_observability_cell=args.cell", source)
        sbatch = (
            ROOT / "project/run_scripts/session05_ode_bf_universal_observability.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('"${RUN_TOKEN}" == "' + UNIVERSAL_OBS_RESULT_TOKEN + '"', sbatch)
        self.assertIn('--cell "${CELL_ID}"', sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch)
        self.assertIn("#SBATCH --mem=65000M", sbatch)
        self.assertIn("#SBATCH --time=23:59:00", sbatch)
        self.assertIn("HF_HUB_OFFLINE=1", sbatch)
        self.assertIn(submit.SESSION_ID, sbatch)
        self.assertIn(submit.EXECUTION_PARENT, sbatch)
        self.assertIn(submit.EXECUTION_BRANCH, sbatch)
        self.assertNotIn("EXPECTED_EXECUTION_HEAD", sbatch)
        self.assertIn('"$(git rev-parse HEAD)" == "${SOURCE_HEAD}"', sbatch)
        self.assertIn('"$(git rev-parse HEAD^)" == "${EXPECTED_EXECUTION_PARENT}"', sbatch)
        submit_source = inspect.getsource(submit._source_manifest_gate)
        self.assertIn('"runtime-git-head"', submit_source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
