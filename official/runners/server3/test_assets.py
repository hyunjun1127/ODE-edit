"""Fast tests for the server3 preflight's byte-verification boundary."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from official.runners.server3 import assets


def _result():
    return {"assets": {}, "blockers": []}


class AssetPreflightTests(unittest.TestCase):
    def test_same_size_is_not_verified_and_changed_bytes_fail_exact_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "asset.bin"
            target.write_bytes(b"correct")
            expected = hashlib.sha256(b"correct").hexdigest()
            found = assets._file(_result(), "asset", str(target), root,
                                 expected_sha=expected, expected_bytes=7,
                                 unverified_blocks=True)
            self.assertEqual(found["verification"], "HISTORICAL_SHA_AND_SIZE_ONLY")
            result = _result()
            assets._file(result, "asset", str(target), root, expected_sha=expected,
                         expected_bytes=7, hash_content=True, unverified_blocks=True)
            self.assertFalse(result["blockers"])
            target.write_bytes(b"changed")
            result = _result()
            member = assets._file(result, "asset", str(target), root,
                                  expected_sha=expected, expected_bytes=7,
                                  hash_content=True, unverified_blocks=True)
            self.assertEqual(member["verification"], "SHA256_MISMATCH")
            self.assertIn("SHA_MISMATCH", {b["code"] for b in result["blockers"]})

    def test_physical_map_requires_all_layers_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            covariance = {}
            for layer in assets.LAYERS:
                target = root / f"c{layer}.npz"
                target.write_bytes(f"layer {layer}".encode())
                covariance[layer] = {
                    "path": str(target), "sha256": assets._sha(target),
                    "bytes": target.stat().st_size, "shape": [2, 2],
                    "dtype": "float32", "npz_key": "mom2.mom2"}
            projector_path = root / "p.pt"
            projector_path.write_bytes(b"projector")
            projector = {"path": str(projector_path), "sha256": assets._sha(projector_path),
                         "bytes": projector_path.stat().st_size,
                         "shape": [5, 2, 2], "dtype": "float32",
                         "layer_mapping": {layer: int(layer) - 4 for layer in assets.LAYERS}}
            historical = {"models": {"qwen2.5-7b-inst": {"assets": {
                "covariance": covariance, "projector": projector}}}}
            size_members = [
                {"kind": "covariance", "layer": int(layer),
                 "relative_path": f"c{layer}.npz", "size": row["bytes"],
                 "sha256": row["sha256"], "shape": row["shape"], "dtype": row["dtype"]}
                for layer, row in covariance.items()]
            size_members.append({"kind": "projector", "layer": None,
                                 "relative_path": "p.pt", "size": projector["bytes"],
                                 "sha256": projector["sha256"],
                                 "shape": projector["shape"], "dtype": projector["dtype"]})
            manifest = {"covariance": dict(covariance), "projector": dict(projector)}
            with (patch.object(assets, "HISTORICAL", root / "historical.json"),
                  patch.object(assets, "SIZE_SEAL", root / "size-seal.json")):
                (root / "historical.json").write_text(json.dumps(historical))
                (root / "size-seal.json").write_text(json.dumps({"models": {
                    "qwen2.5-7b-inst": {"members": size_members}}}))
                report = _result()
                assets._physical(report, manifest, root, historical, False)
                self.assertEqual(sum(x["code"] == "CONTENT_UNVERIFIED"
                                     for x in report["blockers"]), 6)
                report = _result()
                assets._physical(report, manifest, root, historical, True)
                self.assertFalse(report["blockers"])
                self.assertEqual(report["assets"]["projector"]["verification"],
                                 "SHA256_VERIFIED")
                del manifest["covariance"]["7"]
                report = _result()
                assets._physical(report, manifest, root, historical, True)
                self.assertIn("COVARIANCE_LAYER_MAP_INVALID",
                              {x["code"] for x in report["blockers"]})

    def test_generation_manifest_requires_all_three_locked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            lock_files = {}
            for name in ("attribute_snippets.json", "idf.npy", "tfidf_vocab.json"):
                target = root / name
                target.write_bytes(name.encode())
                files[name] = {"path": str(target), "sha256": assets._sha(target),
                               "bytes": target.stat().st_size}
                lock_files[name] = {"sha256": assets._sha(target),
                                    "bytes": target.stat().st_size,
                                    "source_url": "https://example.invalid/" + name}
            lock = root / "generation.lock.json"
            lock.write_text(json.dumps({"reference_identity_sha256": "identity",
                                        "reference_files": lock_files}))
            reference = root / "manifest.json"
            reference.write_text(json.dumps({"identity_sha256": "identity", "files": files}))
            with patch.object(assets, "GENERATION", lock):
                report = _result()
                assets._generation(report, str(reference), root, False, True)
                self.assertEqual(sum(x["code"] == "CONTENT_UNVERIFIED"
                                     for x in report["blockers"]), 3)
                report = _result()
                assets._generation(report, str(reference), root, True, True)
                self.assertFalse(report["blockers"])
                self.assertEqual(report["assets"]["generation_reference"]["verification"],
                                 "SHA256_VERIFIED")
                files["idf.npy"]["sha256"] = "0" * 64
                reference.write_text(json.dumps({"identity_sha256": "identity", "files": files}))
                report = _result()
                assets._generation(report, str(reference), root, False, True)
                self.assertIn("GENERATION_LOCK_MISMATCH",
                              {x["code"] for x in report["blockers"]})

    def test_missing_manifest_returns_explicit_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "absent.json"
            report = assets.preflight(target)
            self.assertFalse(report["ready_to_submit"])
            self.assertEqual(report["blockers"][0]["code"], "ASSET_MANIFEST_MISSING")
            self.assertEqual(report["blockers"][0]["path"], str(target))

    def test_frozen_source_lock_must_match_actual_official_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            source = attempt / "source"
            official = source / "official"
            official.mkdir(parents=True)
            member = official / "one.py"
            member.write_text("value = 1\n")
            lock_path = attempt / "source-lock.json"
            with patch.object(assets, "WORKTREE", source), patch.object(assets, "OFFICIAL", official):
                digest = assets._official_tree_sha256(_result())
                lock_path.write_text(json.dumps({"code_commit": "a" * 40,
                                                 "official_tree_sha256": digest}))
                env = {"ODEEDIT_SOURCE_LOCK": str(lock_path),
                       "ODEEDIT_CODE_COMMIT": "a" * 40,
                       "ODEEDIT_OFFICIAL_TREE_SHA256": digest}
                with patch.dict(os.environ, env):
                    result = _result()
                    assets._source(result)
                    self.assertFalse(result["blockers"])
                    self.assertEqual(result["source"]["kind"], "frozen_archive")
                    member.write_text("value = 2\n")
                    result = _result()
                    assets._source(result)
                    self.assertIn("SOURCE_TREE_MISMATCH",
                                  {b["code"] for b in result["blockers"]})

    def test_assets_digest_ignores_host_paths_and_verification_state(self):
        example = Path(assets.__file__).with_name("assets.local.example.json")
        first = assets.preflight(example)
        with tempfile.TemporaryDirectory() as directory:
            altered = json.loads(example.read_text())
            altered["model_snapshot"] = str(Path(directory) / "missing-model")
            altered["output_root"] = directory
            altered["runtime_python"] = str(Path(directory) / "missing-python")
            altered["cf_source"] = str(Path(directory) / "missing-cf")
            altered["zsre_source"] = str(Path(directory) / "missing-zsre")
            path = Path(directory) / "changed-paths.json"
            path.write_text(json.dumps(altered))
            second = assets.preflight(path)
        self.assertEqual(first["assets_sha256"], second["assets_sha256"])
        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_wandb_inventory_does_not_read_or_emit_credential_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "wandb.env"
            env_file.write_text("WANDB_API_KEY=never-report-this-value\n")
            python = root / "sdk" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n")
            python.chmod(0o700)
            manifest = {"wandb_env": str(env_file), "wandb_sdk_python": str(python)}
            fake_run = type("Run", (), {"stdout": "0.30.0\n"})()
            with patch.object(assets.subprocess, "run", return_value=fake_run):
                with patch.dict(os.environ, {"ODEEDIT_WANDB_PROJECT_VERIFIED": ""}):
                    report = {"blockers": [], "pending": []}
                    assets._wandb(report, manifest, root)
                    self.assertIn("WANDB_CREDENTIAL_MISSING",
                                  {b["code"] for b in report["blockers"]})
                    self.assertNotIn("never-report-this-value", json.dumps(report))
                with patch.dict(os.environ, {"ODEEDIT_WANDB_PROJECT_VERIFIED": "1"}):
                    report = {"blockers": [], "pending": []}
                    assets._wandb(report, manifest, root)
                    self.assertFalse(report["blockers"])
                    self.assertEqual(report["pending"][0]["code"],
                                     "WANDB_ONLINE_STARTUP_PENDING")


if __name__ == "__main__":
    unittest.main()
