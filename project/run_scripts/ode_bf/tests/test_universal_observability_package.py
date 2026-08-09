from __future__ import annotations

import hashlib
import inspect
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts import (
    session05_ode_bf_universal_observability_package as package,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash


class UniversalObservabilityPackageTests(unittest.TestCase):
    def test_identity_uses_r13_and_corrected_r12_r1_closure(self) -> None:
        self.assertEqual(
            package.PACKAGE_ID, "UNIVERSAL_OBSERVABILITY_R13_SH2_BUNDLE_A1"
        )
        self.assertEqual(
            package.R12_PACKAGE_ID,
            package.UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID,
        )
        self.assertEqual(set(package.REFERENCE_ARMS), set(package.R13_LIVE_CELL_IDS))
        self.assertEqual(len(package.REFERENCE_ARMS), 3)
        self.assertEqual(len(package.FACTORIAL_ARM_IDS), 8)
        self.assertEqual(set(package.OBSERVABILITY_DIAGNOSTICS), {"D1", "D2", "D3", "D4"})
        for required in (
            "w0_stepwise_sha256",
            "native_stepwise_sha256",
            "n32_native_sha256",
        ):
            self.assertIn(required, package.R12_REFERENCE_LINKS)

    def test_raw_free_gate_rejects_private_and_absolute_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe.json"
            safe.write_text(
                json.dumps({"request_sha256": "a" * 64, "relative": "reference/x"}),
                encoding="utf-8",
            )
            self.assertEqual(package.validate_raw_free_json(safe)["relative"], "reference/x")
            for key, value in (
                ("prompt", "raw"),
                ("credential", "secret"),
                ("safe_path", "/absolute/private/path"),
            ):
                malformed = root / f"{key}.json"
                malformed.write_text(json.dumps({key: value}), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ODEBFContractError):
                    package.validate_raw_free_json(malformed)

    def test_non_thin_bundle_header_requires_one_advertised_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "source.bundle"
            child = "1" * 40
            advertised = "refs/heads/r13-checkpoint"
            bundle.write_bytes(
                (f"# v2 git bundle\n{child} {advertised}\n\n").encode("utf-8")
                + b"PACK"
            )
            parsed = package._parse_bundle_header(bundle)
            self.assertEqual(parsed["prerequisites"], [])
            self.assertEqual(parsed["heads"], [{"commit": child, "name": advertised}])

    def test_source_history_scan_allows_only_the_public_template_exception(self) -> None:
        source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        observed = package._source_bundle_content_scan(source_head)
        self.assertEqual(observed["forbidden_path_match_count"], 0)
        self.assertEqual(
            observed["public_nonprivate_template_paths"], ["local/README.md"]
        )

    def test_entry_manifest_tar_binds_mode_size_sha_and_blob_oid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
            entries, digest = package._entry_manifest(
                (("z/second", second, "TEST"), ("a/first", first, "TEST")),
                source_head=source_head,
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(
                [entry["path"] for entry in entries],
                [package.PACKAGE_ID + "/a/first", package.PACKAGE_ID + "/z/second"],
            )
            for entry in entries:
                self.assertEqual(entry["mode"], 0o600)
                self.assertEqual(entry["archive_kind"], "REGULAR_FILE")
                self.assertEqual(entry["object_algorithm"], "GIT_BLOB_SHA1")
                self.assertEqual(len(entry["object_id"]), 40)
                self.assertFalse(entry["committed_membership"])
            archive = root / "package.tar"
            inventory = (("z/second", second, "TEST"), ("a/first", first, "TEST"))
            package._write_deterministic_tar(archive, inventory)
            package._verify_deterministic_tar(archive, entries)

    def test_manifest_rejects_missing_mode_and_blob_oid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.write_bytes(b"test")
            source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
            entries, _ = package._entry_manifest(
                (("source", source, "TEST"),), source_head=source_head
            )
            for field in ("mode", "object_id"):
                malformed = dict(entries[0])
                malformed.pop(field)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ODEBFContractError, "manifest entry"
                ):
                    package._validate_manifest_entries([malformed])

    def test_tar_rejects_blob_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(b"contents")
            source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
            inventory = (("source", source, "TEST"),)
            entries, _ = package._entry_manifest(inventory, source_head=source_head)
            archive = root / "package.tar"
            package._write_deterministic_tar(archive, inventory)
            malformed = dict(entries[0])
            malformed["object_id"] = "0" * 40
            with self.assertRaisesRegex(ODEBFContractError, "archive hash"):
                package._verify_deterministic_tar(archive, [malformed])

    def test_r12_handoff_identity_rejects_archive_mismatch(self) -> None:
        """The portable R12 origin is bound by bytes, roots, and package ID."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "reference.tar"
            with tarfile.open(archive_path, mode="w") as archive:
                info = tarfile.TarInfo("empty")
                info.size = 0
                archive.addfile(info, io.BytesIO())
            archive_sha = package.sha256_file(archive_path)
            normalized = "n" * 64
            closure = "c" * 64
            manifest = {
                "schema": "test-manifest/v1",
                "package_id": package.R12_PACKAGE_ID,
                "runtime_technical_repair_parent": package.UNIVERSAL_OBS_BASE_RUNTIME,
                "bundle": {
                    "complete_history": True,
                    "prerequisites": [],
                    "self_contained_verification": {
                        "empty_bare_repository_import": True
                    },
                },
                "normalized_tree_digest": normalized,
                "frozen_reference_closure": {"root_digest": closure},
                "entries": [],
                "entry_count": 0,
            }
            manifest["root_digest"] = canonical_hash(manifest)
            manifest_path = root / package.R12_PACKAGE_MANIFEST
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            manifest_sha = package.sha256_file(manifest_path)
            receipt = {
                "schema": "test-receipt/v1",
                "package_id": package.R12_PACKAGE_ID,
                "transport_executed_by_packager": False,
                "archive": {
                    "name": archive_path.name,
                    "sha256": archive_sha,
                    "size": archive_path.stat().st_size,
                },
                "manifest": {
                    "sha256": manifest_sha,
                    "root_digest": manifest["root_digest"],
                },
            }
            receipt["root_digest"] = canonical_hash(receipt)
            receipt_path = root / package.R12_PACKAGE_RECEIPT
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            receipt_sha = package.sha256_file(receipt_path)
            constants = {
                "UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256": archive_sha,
                "UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256": manifest_sha,
                "UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256": receipt_sha,
                "UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT": manifest["root_digest"],
                "UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT": receipt["root_digest"],
                "UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE": normalized,
                "UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT": closure,
            }
            with mock.patch.multiple(package, **constants):
                observed, _receipt, _archive = package._validate_r12_handoff(root)
                self.assertEqual(observed["package_id"], package.R12_PACKAGE_ID)
                archive_path.write_bytes(b"tampered")
                with self.assertRaisesRegex(ODEBFContractError, "archive identity"):
                    package._validate_r12_handoff(root)

    def test_packager_has_no_transport_or_model_action(self) -> None:
        source = inspect.getsource(package)
        for forbidden in (
            "rsync",
            "scp",
            "ssh",
            "srun",
            "transformers",
            "from_pretrained",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        # The source inventory intentionally names the session's .sbatch
        # wrapper.  It must not, however, execute a scheduler submission.
        self.assertNotIn('["sbatch"', source)
        self.assertNotIn("['sbatch'", source)
        self.assertIn('"transport_executed_by_packager": False', source)
        self.assertIn('"empty_bare_repository_import": True', source)
        self.assertIn('"complete_history": True', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
