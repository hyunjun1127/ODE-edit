from __future__ import annotations

import ast
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import session05_ode_bf_integrated_physical_writer as entry
from project.run_scripts import session05_ode_bf_integrated_physical_writer_dry_plan as dry
from project.run_scripts import session05_ode_bf_submit_integrated_physical_writer as submit
from project.run_scripts import session05_ode_bf_integrated_physical_writer_package as package
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_ARM_REGISTRY,
    P1R14_SCIENTIFIC_SOURCE_PATHS,
    P1R14_SESSION_SOURCE_PATHS,
    P1R14_TEST_SOURCE_PATHS,
    load_and_validate_p1r14_source_manifest,
    p1r14_forecast,
    validate_p1r14_source_closure,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class IntegratedPhysicalWriterLaunchTests(unittest.TestCase):
    def test_one_arm_registry_and_exact_compute_ceiling(self) -> None:
        self.assertEqual(len(P1R14_ARM_REGISTRY), 1)
        forecast = p1r14_forecast()
        self.assertEqual(forecast.online_forward_group_count, 104)
        self.assertEqual(forecast.online_forward_group_ceiling, 110)
        self.assertTrue(forecast.fits)

    def test_all_scientific_and_session_sources_have_clean_closure(self) -> None:
        receipt = validate_p1r14_source_closure(REPO_ROOT)
        self.assertEqual(
            len(receipt["paths"]),
            len(P1R14_SCIENTIFIC_SOURCE_PATHS)
            + len(P1R14_SESSION_SOURCE_PATHS),
        )
        self.assertEqual(receipt["legacy_runtime_import_count"], 0)
        self.assertEqual(receipt["legacy_arm_dispatch_count"], 0)
        self.assertEqual(receipt["alias_specific_science_config_count"], 0)
        self.assertEqual(len(P1R14_TEST_SOURCE_PATHS), 4)

    def test_entry_is_generic_and_submitter_is_llama_owner_only(self) -> None:
        parser = entry._parser()
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            parsed = parser.parse_args(
                [
                    "--model", alias,
                    "--output-root", "/tmp/result",
                    "--source-head", "a" * 40,
                    "--run-token", "integrated-physical-writer-p1r14-v1",
                ]
            )
            self.assertEqual(parsed.model, alias)
        self.assertEqual(dry.LLAMA_ALIAS, "llama3-8b-inst")
        self.assertEqual(dry.SERVER1_PROJECT_GPU_CAP, 4)
        self.assertEqual(submit.SUBMISSION_NAMESPACE.count("llama"), 1)

    def test_session_python_has_no_legacy_runtime_import(self) -> None:
        for relative in P1R14_SESSION_SOURCE_PATHS:
            path = REPO_ROOT / relative
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(item.name for item in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            joined = "\n".join(imports)
            self.assertNotIn("p1_runtime", joined)
            self.assertNotIn("fixed_e8_runtime", joined)

    def test_sbatch_is_valid_and_exact_resource_shape(self) -> None:
        path = REPO_ROOT / "project/run_scripts/session05_ode_bf_integrated_physical_writer.sbatch"
        subprocess.run(["bash", "-n", str(path)], check=True)
        source = path.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=23:59:00",
            "--export=NONE",
            "--offline",
            "--frozen",
            "--no-sync",
        ):
            self.assertIn(token, source)
        self.assertNotIn("rsync", source)
        self.assertNotIn("curl", source)
        self.assertIn("server1 P1R14 launcher is Llama-only", source)
        self.assertNotIn("llama3-8b-inst|qwen2.5-7b-inst", source)

    def test_scheduler_counts_all_project_prefixes(self) -> None:
        rows = "\n".join(
            (
                "1|odeedit_a|RUNNING|gres/gpu:1",
                "2|odebf_b|PENDING|gres/gpu:2",
                "3|odealloc_c|RUNNING|gres/gpu:1",
                "4|unrelated|RUNNING|gres/gpu:4",
            )
        )
        completed = subprocess.CompletedProcess([], 0, stdout=rows, stderr="")
        with mock.patch.dict("os.environ", {"USER": "tester"}), mock.patch.object(
            submit, "_run", return_value=completed
        ):
            observed = submit._scheduler()
        self.assertEqual(
            sum(int(item["gpu"]) for item in observed if item["project_job"]),
            4,
        )

    def test_sh2_ack_is_bound_to_every_outer_package_hash(self) -> None:
        local = {
            "source_tree": "b" * 40,
            "archive_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "receipt_sha256": "c" * 64,
            "manifest_root": "d" * 64,
            "receipt_root": "e" * 64,
            "normalized_tree_digest": "f" * 64,
        }
        value = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-sh2-package-ack/v1",
            "status": "PACKAGE_VERIFICATION_PASS",
            "package_id": package.PACKAGE_ID,
            "source_head": "1" * 40,
            "source_tree": local["source_tree"],
            "receiver_session": package.SH2_RECIPIENT_SESSION,
            "archive_sha256": local["archive_sha256"],
            "manifest_sha256": local["manifest_sha256"],
            "receipt_sha256": local["receipt_sha256"],
            "manifest_root": local["manifest_root"],
            "receipt_root": local["receipt_root"],
            "normalized_tree_digest": local["normalized_tree_digest"],
            "model_gpu_slurm_result_root_action_count": 0,
        }
        value["root_digest"] = canonical_hash(value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ack.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            observed = submit._load_sh2_package_ack(
                path, source_head="1" * 40, local_package=local
            )
            self.assertEqual(observed["status"], "PACKAGE_VERIFICATION_PASS")
            broken = dict(value)
            broken["archive_sha256"] = "0" * 64
            broken["root_digest"] = canonical_hash(
                {key: item for key, item in broken.items() if key != "root_digest"}
            )
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ODEBFContractError):
                submit._load_sh2_package_ack(
                    path, source_head="1" * 40, local_package=local
                )

    def test_verified_package_is_bound_to_dry_plan_identities(self) -> None:
        plan = {
            "fresh_seal_root": "a" * 64,
            "historical_exclusion_root": "b" * 64,
            "source_manifest_root": "c" * 64,
            "source_manifest_sha256": "d" * 64,
            "source_closure_sha256": "e" * 64,
            "artifacts": {"llama3-8b-inst": {"projector_sha256": "f" * 64}},
            "numerical_locks": {
                "llama3-8b-inst": {"root_digest": "1" * 64}
            },
        }
        local = {
            "fresh_seal_root": plan["fresh_seal_root"],
            "historical_exclusion_root": plan["historical_exclusion_root"],
            "source_manifest_root": plan["source_manifest_root"],
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "source_closure_sha256": plan["source_closure_sha256"],
            "artifact_identities": plan["artifacts"],
            "numerical_lock_identities": plan["numerical_locks"],
        }
        submit._assert_package_plan_binding(local, plan)
        for key in (
            "fresh_seal_root",
            "historical_exclusion_root",
            "source_manifest_root",
            "source_manifest_sha256",
            "source_closure_sha256",
            "artifact_identities",
            "numerical_lock_identities",
        ):
            broken = copy.deepcopy(local)
            broken[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ODEBFContractError):
                submit._assert_package_plan_binding(broken, plan)

    def test_source_manifest_requires_committed_mode_and_blob_identity(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/ode_bf/locks/"
            "source_manifest_s05_integrated_physical_writer.json"
        )
        value = json.loads(source.read_text(encoding="utf-8"))
        observed, _ = load_and_validate_p1r14_source_manifest(
            REPO_ROOT, source, source_head="a" * 40
        )
        self.assertEqual(len(observed["entries"]), 16)
        for missing in ("mode", "git_blob_oid", "object_algorithm"):
            broken = json.loads(json.dumps(value))
            broken["entries"][0].pop(missing)
            broken["entry_root"] = canonical_hash(broken["entries"])
            broken.pop("root_digest")
            broken["root_digest"] = canonical_hash(broken)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "source.json"
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(missing=missing), self.assertRaises(
                    ODEBFContractError
                ):
                    load_and_validate_p1r14_source_manifest(
                        REPO_ROOT, path, source_head="a" * 40
                    )


if __name__ == "__main__":
    unittest.main()
