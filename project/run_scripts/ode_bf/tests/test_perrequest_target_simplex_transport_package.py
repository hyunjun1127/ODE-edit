from __future__ import annotations

import io
import json
import tarfile
import unittest

from project.run_scripts import session05_ode_bf_perrequest_simplex_transport_package as package
from project.run_scripts.ode_bf.contracts import canonical_hash


class P1R15PackageTests(unittest.TestCase):
    def test_member_manifest_has_mode_object_size_sha_role(self) -> None:
        members = {"a.json": b"{}\n", "source.bundle": b"bundle"}
        rows = package._member_rows(members)
        self.assertEqual([row["path"] for row in rows], ["a.json", "source.bundle"])
        for row in rows:
            self.assertEqual(
                set(row),
                {
                    "path",
                    "mode",
                    "size_bytes",
                    "sha256",
                    "git_blob_oid",
                    "object_algorithm",
                    "archive_kind",
                    "role",
                },
            )
            self.assertEqual(row["mode"], 0o600)
            self.assertEqual(row["archive_kind"], "REGULAR")

    def test_missing_mode_or_object_is_detectably_noncanonical(self) -> None:
        row = package._member_rows({"a": b"x"})[0]
        rooted = canonical_hash([row])
        without_mode = dict(row)
        without_mode.pop("mode")
        without_object = dict(row)
        without_object.pop("git_blob_oid")
        self.assertNotEqual(rooted, canonical_hash([without_mode]))
        self.assertNotEqual(rooted, canonical_hash([without_object]))

    def test_tar_is_regular_deterministic_and_non_traversing(self) -> None:
        members = {"b.json": b"b", "a.json": b"a"}
        first = package._tar_bytes(members)
        second = package._tar_bytes(members)
        self.assertEqual(first, second)
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
            rows = archive.getmembers()
            self.assertEqual([row.name for row in rows], ["a.json", "b.json"])
            self.assertTrue(all(row.isfile() and not row.issym() for row in rows))

    def test_portable_contract_forbids_stage_b(self) -> None:
        source = package.create_package.__code__.co_consts
        rendered = json.dumps([str(item) for item in source])
        self.assertIn("stage_b_material_count", rendered)
        self.assertIn("scientific_promotion_authorized", rendered)
        self.assertIn("qwen_model_gpu_slurm_owner_after_receiver_pass_only", rendered)


if __name__ == "__main__":
    unittest.main()
