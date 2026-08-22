from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.alphaedit_runtime_path_seal import canonical_hash
from project.run_scripts.ode_bf.p4_sealed_stream import (
    P4SealedStreamError,
    STREAM_SCHEMA,
    preflight_sealed_stream,
    stream_readiness,
)


class P4SealedStreamTests(unittest.TestCase):
    def test_absent_package_remains_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = stream_readiness(Path(temporary) / "sealed-stream")
        self.assertEqual(receipt.status, "BLOCKED_SEALED_STREAM_TRANSFER")
        self.assertFalse(receipt.full_read)

    def test_full_read_and_member_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sealed-stream"
            root.mkdir()
            member = root / "samples.jsonl"
            member.write_bytes(b'{"case_id":"case01"}\n')
            identity = {
                "path": member.name,
                "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                "size": member.stat().st_size,
            }
            binding = {
                "model_aliases": ["llama3-8b-inst", "qwen2.5-7b-inst"],
                "independent_slice_count_per_model": 10,
                "entries_per_slice": 10,
                "slice_membership_root": "slice-root",
                "order_root": "order-root",
                "seed_identity": "seed-id",
                "dataset_identity": "dataset-id",
                "evaluator_identity": "evaluator-id",
            }
            manifest = {
                "schema": STREAM_SCHEMA,
                "binding": binding,
                "members": [identity],
                "member_root": canonical_hash([identity]),
            }
            manifest["root_digest"] = canonical_hash(manifest)
            raw = json.dumps(manifest, sort_keys=True).encode()
            (root / "manifest.json").write_bytes(raw)
            kwargs = {
                "expected_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "expected_package_root": manifest["member_root"],
                "expected_binding": binding,
            }
            receipt = preflight_sealed_stream(root, **kwargs)
            self.assertEqual(receipt["status"], "TRANSFER_FULL_READ_PASS")
            member.write_bytes(member.read_bytes() + b"tampered")
            with self.assertRaises(P4SealedStreamError):
                preflight_sealed_stream(root, **kwargs)

    def test_binding_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sealed-stream"
            root.mkdir()
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(P4SealedStreamError):
                preflight_sealed_stream(
                    root,
                    expected_manifest_sha256="0" * 64,
                    expected_package_root="0" * 64,
                    expected_binding={},
                )


if __name__ == "__main__":
    unittest.main()
