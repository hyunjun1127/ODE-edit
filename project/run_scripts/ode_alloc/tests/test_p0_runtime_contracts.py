from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_alloc.contracts import ODEAllocContractError
from project.run_scripts.ode_alloc.p0_runtime import (
    _canonical_write_once,
    expected_result_name,
    tensor_sha256,
)


class P0RuntimeContractTests(unittest.TestCase):
    def test_output_name_is_lock_hash_bound_and_alias_closed(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            expected_result_name("llama3-8b-inst", digest),
            "s04-p0-native-identity-r1-llama3-8b-inst-aaaaaaaa",
        )
        with self.assertRaises(ODEAllocContractError):
            expected_result_name("foreign", digest)

    def test_tensor_hash_is_layout_stable_and_value_sensitive(self) -> None:
        value = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
        self.assertEqual(tensor_sha256(value), tensor_sha256(value.clone()))
        changed = value.clone()
        changed[0, 0] = 7
        self.assertNotEqual(tensor_sha256(value), tensor_sha256(changed))

    def test_terminal_metadata_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            first = _canonical_write_once(path, {"status": "PASS"})
            self.assertEqual(len(first), 64)
            with self.assertRaises(FileExistsError):
                _canonical_write_once(path, {"status": "REWRITE"})


if __name__ == "__main__":
    unittest.main()
