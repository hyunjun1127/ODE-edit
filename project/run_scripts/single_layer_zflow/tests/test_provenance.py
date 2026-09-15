"""Small CPU fixtures for provenance; no model assets or scheduler access."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from project.run_scripts.single_layer_zflow import provenance as p


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_full_member_and_size_hash_fail_closed(self):
        path = self.write("source.py", b"x = 1\n")
        wanted = hashlib.sha256(path.read_bytes()).hexdigest()
        member = p.file_member(path, expected_sha=wanted, expected_bytes=6)
        self.assertEqual(member["sha256"], wanted)
        self.assertEqual(member["verification"], "FULL_SHA256")
        with self.assertRaises(ValueError):
            p.file_member(path, expected_sha="0" * 64)
        with self.assertRaises(ValueError):
            p.file_member(path, expected_bytes=7)

    def test_big_member_declares_inherited_hash_not_new_content_verification(self):
        path = self.write("model.safetensors", b"already verified bytes")
        with patch.object(p, "file_sha", side_effect=AssertionError("unnecessary content rehash")):
            member = p.file_member(path, expected_sha="a" * 64,
                                   expected_bytes=path.stat().st_size, full_hash=False)
        self.assertEqual(member["verification"], "INHERITED_FULL_SHA256_CURRENT_SIZE")
        self.assertEqual(member["sha256"], "a" * 64)
        with self.assertRaises(ValueError):
            p.file_member(path, full_hash=False)

    def test_symlink_default_rejection_explicit_model_allowance(self):
        target = self.write("blob", b"data")
        path = self.root / "snapshot" / "weight"
        path.parent.mkdir()
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            p.file_member(path)
        member = p.file_member(path, allow_symlink=True)
        self.assertEqual(member["realpath"], str(target))
        self.assertTrue(member["symlink"])

    def test_create_once_no_overwrite_and_parent_symlink_rejection(self):
        path = self.root / "sealed" / "lock.json"
        p.exclusive_write(path, b"original")
        with self.assertRaises(FileExistsError):
            p.exclusive_write(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        link = self.root / "link"
        link.symlink_to(path.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            p.exclusive_write(link / "new.json", b"forbidden")

    def shard(self, name, key):
        header = p.encoded({key: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}})
        return self.write(name, struct.pack("<Q", len(header)) + header + b"\x00" * 8)

    def test_safetensor_index_header_and_physical_coverage(self):
        path = self.shard("model-1.safetensors", "model.weight")
        got = p.safetensor_header(path, {"model.weight"})
        self.assertEqual(got["tensor_count"], 1)
        self.assertTrue(got["index_membership_exact"])
        with self.assertRaises(ValueError):
            p.safetensor_header(path, {"wrong.weight"})
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            p.safetensor_header(path, {"model.weight"})

    def test_safetensor_truncation_and_pointer_only_rejected(self):
        path = self.write("pointer.safetensors", b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaises(ValueError):
            p.safetensor_header(path, set())
        path.write_bytes(b"short")
        with self.assertRaises(ValueError):
            p.safetensor_header(path, set())

    def test_model_all_members_exact_and_shards_not_rehashed(self):
        first = self.shard("model-1.safetensors", "layer1")
        second = self.shard("model-2.safetensors", "layer2")
        for name in ("config.json", "generation_config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer_config.json"):
            self.write(name, b"{}")
        self.write("model.safetensors.index.json", p.encoded({"weight_map": {"layer1": first.name, "layer2": second.name}}))
        prior = [p.file_member(path) for path in sorted(self.root.iterdir())]
        original_file_sha = p.file_sha

        def reject_shard_hash(path):
            if Path(path).suffix == ".safetensors":
                raise AssertionError("large checkpoint rehash not expected")
            return original_file_sha(path)

        with patch.object(p, "file_sha", side_effect=reject_shard_hash):
            got = p.model_inventory(self.root, prior)
        self.assertEqual(got["shard_count"], 2)
        self.assertEqual(got["weight_map_entries"], 2)
        self.assertEqual(len(got["members"]), 8)
        with self.assertRaises(ValueError):
            p.model_inventory(self.root, prior[:-1])

    def test_model_tokenizer_same_size_corruption_detected(self):
        shard = self.shard("model-1.safetensors", "weight")
        for name in ("config.json", "generation_config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer_config.json"):
            self.write(name, b"{}")
        self.write("model.safetensors.index.json", p.encoded({"weight_map": {"weight": shard.name}}))
        prior = [p.file_member(path) for path in sorted(self.root.iterdir())]
        (self.root / "tokenizer.json").write_bytes(b"[]")
        with self.assertRaises(ValueError):
            p.model_inventory(self.root, prior)

    def test_execution_freeze_rejects_uncommitted_source(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write(p.PACKAGE + "/native_binding.py", b"# untracked\n")
        with self.assertRaisesRegex(ValueError, "not committed/clean"):
            p.execution_source(self.root)

    def test_output_scope_and_existing_paths_fail_before_assets(self):
        with self.assertRaisesRegex(ValueError, "under this task"):
            p.prepare(self.root / "outside", self.root)
        with patch.object(p, "TASK_ROOT", self.root):
            existing = self.root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                p.prepare(existing, self.root)

    def test_canonical_json_hash_disallows_nonfinite(self):
        self.assertEqual(p.digest({"b": 1, "a": "한글"}), p.digest({"a": "한글", "b": 1}))
        with self.assertRaises(ValueError):
            p.digest({"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
