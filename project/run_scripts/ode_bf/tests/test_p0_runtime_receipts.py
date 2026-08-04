from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p0_runtime import (
    StageRecorder,
    _atomic_write_once,
    expected_result_name,
    write_failure_once,
)


class P0RuntimeReceiptTests(unittest.TestCase):
    def test_stage_receipts_are_create_once_and_rng_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = StageRecorder(root)
            before = torch.get_rng_state().clone()
            first = recorder.record("post_fixture", {"shape": [12, 10]})
            after = torch.get_rng_state().clone()
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(len(first), 64)
            self.assertEqual(recorder.last_stage, "post_fixture")
            with self.assertRaises(FileExistsError):
                _atomic_write_once(root / "stage-01-post_fixture.json", {"x": 1})

    def test_failure_receipt_hashes_message_and_allowlists_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "local/odebf/results" / expected_result_name(
                "llama3-8b-inst"
            )
            secret = "prompt-subject-target-must-not-appear"
            try:
                raise RuntimeError(secret)
            except RuntimeError as exc:
                digest, receipt = write_failure_once(output, exc, repo_root=repo)
            self.assertEqual(len(digest), 64)
            self.assertEqual(receipt["exception_class"], "RuntimeError")
            self.assertEqual(
                receipt["exception_message_sha256"],
                hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            )
            raw = (output / "failure.json").read_text(encoding="utf-8")
            self.assertNotIn(secret, raw)
            self.assertFalse(receipt["retry_permitted"])
            with self.assertRaises(FileExistsError):
                try:
                    raise RuntimeError("second")
                except RuntimeError as exc:
                    write_failure_once(output, exc, repo_root=repo)

    def test_failure_namespace_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaisesRegex(ODEBFContractError, "namespace"):
                try:
                    raise RuntimeError("opaque")
                except RuntimeError as exc:
                    write_failure_once(repo / "outside", exc, repo_root=repo)


if __name__ == "__main__":
    unittest.main()
