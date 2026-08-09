from __future__ import annotations

import copy
import io
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import session05_ode_bf_integrated_physical_writer_package as package
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_RUN_ATTEMPT_ID,
)


class IntegratedPhysicalWriterPackageTests(unittest.TestCase):
    def test_manifest_requires_mode_object_and_exact_regular_archive(self) -> None:
        members = {
            "source.bundle": b"bundle",
            "lock-hash-index.json": b"{}\n",
            "source-tree-inventory.json": b"{}\n",
            "source-object-scan.json": b"{}\n",
            "portable-contract.json": b"{}\n",
        }
        archive = package._tar_bytes(members)
        entries = [
            {
                "path": name,
                "mode": 0o600,
                "size": len(data),
                "sha256": package._sha256(data),
                "git_blob_oid": package._git_blob_oid(data),
                "object_algorithm": "GIT_BLOB_SHA1",
                "role": "RAW_FREE_PROVENANCE",
                "semantic_role": "PORTABLE_SOURCE_OR_LOCK_EVIDENCE",
                "archive_kind": "REGULAR_FILE",
                "committed_source_member": False,
            }
            for name, data in sorted(members.items())
        ]
        manifest = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-package-manifest/v1",
            "package_id": package.PACKAGE_ID,
            "source_head": "a" * 40,
            "source_tree": "b" * 40,
            "exact_parent": package.EXECUTION_PARENT,
            "run_attempt_id": P1R14_RUN_ATTEMPT_ID,
            "run_attempt_namespace_root": "9" * 64,
            "bundle_source_head": "a" * 40,
            "bundle_source_tree": "b" * 40,
            "bundle_exact_parent": package.EXECUTION_PARENT,
            "prerequisite_count": 0,
            "complete_history": True,
            "empty_bare_verify": True,
            "strict_fsck": True,
            "checkout_clean": True,
            "bundle_proof_identity": "c" * 64,
            "source_tree_inventory_root": "d" * 64,
            "source_object_scan_root": "e" * 64,
            "lock_index_root": "f" * 64,
            "object_algorithm": "GIT_BLOB_SHA1",
            "entries": entries,
            "entry_count": len(entries),
            "archive_member_count": len(package.ARCHIVE_MEMBER_NAMES),
            "entry_root": canonical_hash(entries),
            "archive_sha256": package._sha256(archive),
            "archive_size": len(archive),
            "normalized_tree_digest": canonical_hash(
                [
                    [
                        item["path"], item["mode"], item["size"],
                        item["sha256"], item["git_blob_oid"],
                    ]
                    for item in entries
                ]
            ),
            "recipient_session": package.SH2_RECIPIENT_SESSION,
            "secure_create_once_recipient": "SH2_SERVER2_CANONICAL_SESSION",
            "scientific_promotion_authorized": False,
            "model_dataset_cache_credential_payload_count": 0,
            "raw_prompt_target_tensor_payload_count": 0,
            "case_ids_already_committed": True,
        }
        manifest["root_digest"] = canonical_hash(manifest)
        observed = package.verify_package_manifest(manifest, archive)
        self.assertEqual(observed["root_digest"], manifest["root_digest"])
        broken_namespace = copy.deepcopy(manifest)
        broken_namespace["run_attempt_id"] = "stale"
        broken_namespace.pop("root_digest")
        broken_namespace["root_digest"] = canonical_hash(broken_namespace)
        with self.assertRaises(ODEBFContractError):
            package.verify_package_manifest(broken_namespace, archive)
        for missing in ("mode", "git_blob_oid"):
            broken = copy.deepcopy(manifest)
            broken["entries"][0].pop(missing)
            broken["entry_root"] = canonical_hash(broken["entries"])
            broken.pop("root_digest")
            broken["root_digest"] = canonical_hash(broken)
            with self.subTest(missing=missing), self.assertRaises(
                ODEBFContractError
            ):
                package.verify_package_manifest(broken, archive)

    def test_archive_rejects_inventory_and_path_escape(self) -> None:
        with self.assertRaises(ODEBFContractError):
            package._tar_bytes({"source.bundle": b"x"})
        with self.assertRaises(ODEBFContractError):
            package._tar_bytes(
                {
                    "source.bundle": b"x",
                    "lock-hash-index.json": b"y",
                    "source-tree-inventory.json": b"i",
                    "source-object-scan.json": b"s",
                    "../portable-contract.json": b"z",
                }
            )

    def test_self_contained_bundle_verifies_in_empty_bare_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            subprocess.run(["git", "init", "-b", "p1r14-test", str(repo)], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "P1R14 Test"], cwd=repo, check=True)
            (repo / "source.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, stdout=subprocess.PIPE)
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
            (repo / "source.txt").write_text("child\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-am", "child"], cwd=repo, check=True, stdout=subprocess.PIPE)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
            bundle = root / "source.bundle"
            subprocess.run(
                ["git", "bundle", "create", str(bundle), "refs/heads/p1r14-test"],
                cwd=repo,
                check=True,
            )
            with mock.patch.object(package, "REPO_ROOT", repo), mock.patch.object(
                package, "PACKAGE_REF", "refs/heads/p1r14-test"
            ), mock.patch.object(package, "EXECUTION_PARENT", base):
                proof = package._verify_bundle(
                    bundle, head, verify_imported_contract=False
                )
            self.assertTrue(proof["complete_history"])
            self.assertEqual(proof["prerequisite_count"], 0)
            self.assertEqual(proof["source_head"], head)
            self.assertEqual(proof["exact_parent"], base)

    def test_case_id_exception_scanner_rejects_raw_content(self) -> None:
        good = {
            "case_ids": [1, "2"],
            "request_sha256": "a" * 64,
            "status": "RAW_FREE",
        }
        receipt = package._case_id_content_scan(good)
        self.assertEqual(receipt["case_id_value_count"], 2)
        for bad in (
            {"case_id": 1, "prompt": "raw"},
            {"case_id": 1, "target_new": "raw"},
            {"case_id": 1, "tensor": [1.0]},
            {"case_id": 1, "credential": "secret"},
        ):
            with self.subTest(bad=tuple(bad)), self.assertRaises(
                ODEBFContractError
            ):
                package._case_id_content_scan(bad)


if __name__ == "__main__":
    unittest.main()
