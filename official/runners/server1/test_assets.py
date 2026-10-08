"""Small CPU metadata/schema fixtures; not a GPU/model qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from official.experiments.prepare import digest, file_sha
from official.runners.server1.assets import (
    AssetBindingError, SCHEMA, bind_stream, member, validate_c0, verify_manifest,
)


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def c0(self, **overrides):
        values = {"mom2.constructor": np.array("easyeditor.util.runningstats.SecondMoment()"),
                  "mom2.count": np.array(23), "mom2.mom2": np.eye(3, dtype=np.float32) * 23,
                  "sample_size": np.array(100000)}
        values.update(overrides)
        path = self.root / "fixture.npz"
        np.savez(path, **values)
        return path

    def test_exact_native_sum_count_not_document_denominator(self):
        receipt = validate_c0(self.c0(), 3)
        self.assertEqual(receipt["masked_token_vector_count"], 23)
        self.assertEqual(receipt["sample_documents"], 100000)
        self.assertEqual(receipt["normalization"], "sum / masked_token_vector_count")
        self.assertEqual(receipt["stored_value"], "UNCENTERED_SECOND_MOMENT_SUM")
        self.assertFalse(receipt["PSD_eigenvalidation"])

    def test_wrong_width_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_SHAPE"):
            validate_c0(self.c0(), 4)

    def test_wrong_dtype_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_SHAPE"):
            validate_c0(self.c0(**{"mom2.mom2": np.eye(3)}), 3)

    def test_zero_count_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_COUNT"):
            validate_c0(self.c0(**{"mom2.count": np.array(0)}), 3)

    def test_float_count_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_INTEGER"):
            validate_c0(self.c0(**{"mom2.count": np.array(23.0)}), 3)

    def test_wrong_sample_count_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_COUNT"):
            validate_c0(self.c0(**{"sample_size": np.array(100)}), 3)

    def test_nonfinite_rejected(self):
        values = np.eye(3, dtype=np.float32)
        values[1, 2] = np.nan
        with self.assertRaisesRegex(AssetBindingError, "C0_NONFINITE"):
            validate_c0(self.c0(**{"mom2.mom2": values}), 3)

    def test_negative_diagonal_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_NEGATIVE_DIAGONAL"):
            validate_c0(self.c0(**{"mom2.mom2": -np.eye(3, dtype=np.float32)}), 3)

    def test_object_array_is_not_unpickled(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_SHAPE"):
            validate_c0(self.c0(**{"mom2.mom2": np.array([[object()]], dtype=object)}), 3)

    def test_unexpected_schema_rejected(self):
        with self.assertRaisesRegex(AssetBindingError, "C0_NATIVE_SCHEMA"):
            validate_c0(self.c0(extra=np.array(1)), 3)

    def test_missing_path_typed(self):
        with self.assertRaisesRegex(AssetBindingError, "ASSET_MISSING") as caught:
            member(self.root / "missing")
        self.assertEqual(caught.exception.receipt()["status"], "BLOCKED_ASSET_IDENTITY")

    def test_stat_only_is_not_hash_pass(self):
        path = self.root / "data.bin"
        path.write_bytes(b"safe fixture")
        row = member(path, hash_bytes=False, expected_sha=file_sha(path))
        self.assertIsNone(row["sha256"])
        self.assertEqual(row["verification"], "CURRENT_STAT_ONLY_NOT_SHA_VERIFIED")

    def test_wrong_hash_rejected(self):
        path = self.root / "data.bin"
        path.write_bytes(b"fixture")
        with self.assertRaisesRegex(AssetBindingError, "ASSET_SHA_MISMATCH"):
            member(path, expected_sha="0" * 64)

    def test_unknown_symlink_rejected(self):
        original = self.root / "data.bin"
        original.write_bytes(b"fixture")
        linked = self.root / "linked"
        linked.symlink_to(original)
        with self.assertRaisesRegex(AssetBindingError, "ASSET_UNEXPECTED_SYMLINK"):
            member(linked)

    def test_immutable_manifest_identity_and_tamper(self):
        path = self.root / "data.bin"
        path.write_bytes(b"fixture")
        data = dict(schema=SCHEMA, server="server1", sample=member(path), GPU_qualification=False)
        data["assets_sha256"] = digest(data)
        self.assertFalse(verify_manifest(data)["GPU_qualification"])
        changed = copy.deepcopy(data)
        changed["sample"]["bytes"] = 0
        with self.assertRaisesRegex(AssetBindingError, "ASSET_MANIFEST_DIGEST"):
            verify_manifest(changed)

    def test_actual_file_change_rejected(self):
        path = self.root / "data.bin"
        path.write_bytes(b"fixture")
        data = dict(schema=SCHEMA, server="server1", sample=member(path))
        data["assets_sha256"] = digest(data)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(AssetBindingError, "ASSET_CURRENT_(STAT|SHA)_CHANGED"):
            verify_manifest(data)

    def test_wrong_dataset_not_substituted(self):
        path = self.root / "wrong-cf.json"
        path.write_text("[]\n")
        with self.assertRaisesRegex(AssetBindingError, "ASSET_SHA_MISMATCH"):
            bind_stream(path, "cf")

    def source_and_data_manifest(self):
        source = self.root / "preparation-source.py"
        data = self.root / "actual-data.json"
        source.write_bytes(b"original source")
        data.write_bytes(b"original data")
        value = dict(schema=SCHEMA, server="server1",
                     official_source_members={"SOURCES.json": member(source)},
                     streams={"cf": {"source": member(data)}})
        value["assets_sha256"] = digest(value)
        return value, source, data

    def test_preparation_source_verification_default_failclosed(self):
        manifest, source, data = self.source_and_data_manifest()
        source.write_bytes(b"updated source")
        with self.assertRaisesRegex(AssetBindingError, "ASSET_CURRENT_(STAT|SHA)_CHANGED"):
            verify_manifest(manifest)

    def test_frozen_source_mode_skips_only_historical_source_paths(self):
        manifest, source, data = self.source_and_data_manifest()
        old_provenance = copy.deepcopy(manifest["official_source_members"])
        source.write_bytes(b"updated source")
        receipt = verify_manifest(manifest, verify_preparation_source=False)
        self.assertFalse(receipt["preparation_source_paths_verified"])
        self.assertTrue(receipt["separately_verified_frozen_source_required"])
        self.assertFalse(receipt["actual_execution_source_equivalence_claimed"])
        self.assertEqual(manifest["official_source_members"], old_provenance)
        data.write_bytes(b"updated data")
        with self.assertRaisesRegex(AssetBindingError, "ASSET_CURRENT_(STAT|SHA)_CHANGED"):
            verify_manifest(manifest, verify_preparation_source=False)

    def test_frozen_source_option_does_not_skip_manifest_digest(self):
        manifest, source, data = self.source_and_data_manifest()
        manifest["official_source_members"]["SOURCES.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(AssetBindingError, "ASSET_MANIFEST_DIGEST"):
            verify_manifest(manifest, verify_preparation_source=False)

    def test_source_option_rejects_nonboolean(self):
        manifest, source, data = self.source_and_data_manifest()
        with self.assertRaisesRegex(AssetBindingError, "ASSET_SOURCE_VERIFICATION_FLAG_TYPE"):
            verify_manifest(manifest, verify_preparation_source="false")


if __name__ == "__main__":
    unittest.main()
