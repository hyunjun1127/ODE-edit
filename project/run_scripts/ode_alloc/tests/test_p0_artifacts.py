from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_alloc.contracts import ODEAllocContractError
from project.run_scripts.ode_alloc.p0_artifacts import (
    _safe_relative,
    load_artifact_lock,
    validate_artifact_lock_structure,
)


class P0ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = (
            Path(__file__).resolve().parents[1] / "p0_artifact_lock_r1.json"
        )

    def test_tracked_lock_has_both_pinned_bf16_models_and_five_covariances(self) -> None:
        value, raw_sha, canonical_sha = load_artifact_lock(self.lock)
        self.assertEqual(len(raw_sha), 64)
        self.assertEqual(len(canonical_sha), 64)
        self.assertEqual(set(value["models"]), {"llama3-8b-inst", "qwen2.5-7b-inst"})
        for model in value["models"].values():
            self.assertEqual(model["config_torch_dtype"], "bfloat16")
            self.assertEqual(set(model["covariance"]), {"4", "5", "6", "7", "8"})

    def test_lock_structure_rejects_alias_specific_omission(self) -> None:
        value = json.loads(self.lock.read_text(encoding="utf-8"))
        del value["models"]["qwen2.5-7b-inst"]
        with self.assertRaisesRegex(ODEAllocContractError, "aliases"):
            validate_artifact_lock_structure(value)

    def test_relative_artifact_path_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises((ODEAllocContractError, ValueError, FileNotFoundError)):
                _safe_relative(root, "../foreign")


if __name__ == "__main__":
    unittest.main()
