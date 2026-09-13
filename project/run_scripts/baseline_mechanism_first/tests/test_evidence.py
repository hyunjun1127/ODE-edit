import hashlib
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.baseline_mechanism_first.evidence import (
    EvidenceUse, inspect_evidence, inventory_members, reuse_manifest,
)


class EvidenceTests(unittest.TestCase):
    def test_local_bytes_do_not_claim_full_read_or_tensor_parity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"source"; p.write_bytes(b"source")
            out = inspect_evidence(p, kind=EvidenceUse.ACCESSIBLE_RAW_SOURCE,
                                   expected_sha256=hashlib.sha256(b"source").hexdigest(), expected_bytes=6)
            self.assertEqual(out["status"], "FILE_BYTES_AVAILABLE")
            self.assertFalse(out["full_read_claim"])
            self.assertFalse(out["tensor_deserialized"])
            self.assertEqual(out["model_forward_count"], 0)

    def test_absent_and_mismatch_are_not_available(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"raw"
            self.assertEqual(inspect_evidence(p, kind=EvidenceUse.MISSING)["status"], "ABSENT_LOCAL")
            p.write_bytes(b"wrong")
            self.assertEqual(inspect_evidence(p, kind=EvidenceUse.ACCESSIBLE_RAW_SOURCE,
                                             expected_sha256="0"*64)["status"], "INTEGRITY_MISMATCH")

    def test_symlink_parent_and_dangling_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/"real").mkdir(); (root/"real"/"raw").write_bytes(b"ok")
            (root/"link").symlink_to(root/"real", target_is_directory=True)
            (root/"dangling").symlink_to(root/"absent")
            for p in (root/"link"/"raw", root/"dangling", root/"dangling"/"child"):
                self.assertEqual(inspect_evidence(p, kind=EvidenceUse.ACCESSIBLE_RAW_SOURCE)["status"], "SYMLINK_REJECTED")

    def test_no_implicit_remote_path_mapping(self):
        rows = inventory_members([dict(path="/data/remote/current.json", sha256="a"*64, bytes=1)], path_map={})
        self.assertEqual(rows[0]["status"], "NO_EXPLICIT_LOCAL_BINDING")
        self.assertEqual(reuse_manifest(rows)["aggregate_to_case_reconstruction"], 0)


if __name__ == "__main__": unittest.main()
