from __future__ import annotations

import json
import ast
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.fzcb_completion_value.contracts import K0_CASE_IDS, K1_CASE_IDS, NumericalLock
from project.run_scripts.fzcb_completion_value.hashing import canonical_json, write_json_once
from project.run_scripts.fzcb_completion_value.snapshots import WeightSnapshot


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(3, 4, bias=False)
        self.right = torch.nn.Linear(4, 2, bias=False)


class RuntimeContractTests(unittest.TestCase):
    def test_candidate_exact_copy_rollback_and_pointer(self) -> None:
        model = TinyModel()
        names = ("left.weight", "right.weight")
        entry = WeightSnapshot.capture(model, names)
        tangents = tuple(torch.ones_like(dict(model.named_parameters())[name]) for name in names)
        entry.apply_tangents(model, tangents, 0.25)
        self.assertNotEqual(entry.current_root(model), entry.root)
        receipt = entry.restore(model)
        self.assertTrue(receipt["bytes_exact"])
        self.assertTrue(receipt["pointer_exact"])
        self.assertEqual(entry.current_root(model), entry.root)

    def test_lock_denominators_and_forbidden_feature_counts(self) -> None:
        lock = NumericalLock()
        self.assertEqual(lock.k_steps, 4)
        self.assertEqual(lock.progress_grid, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertFalse(set(K0_CASE_IDS) & set(K1_CASE_IDS))
        self.assertEqual(
            (lock.dense_inverse_count, lock.explicit_kronecker_count,
             lock.value_gradient_count, lock.cbf_count, lock.full_qcqp_count,
             lock.alphaedit_count),
            (0, 0, 0, 0, 0, 0),
        )

    def test_nonfinite_outcomes_are_typed_valid_json(self) -> None:
        encoded = canonical_json({"unreachable": float("inf"), "noise": float("nan")})
        self.assertEqual(json.loads(encoded), {"noise": "NaN", "unreachable": "+Infinity"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json_once(path, {"unreachable": float("inf")})
            self.assertEqual(json.loads(path.read_text())["unreachable"], "+Infinity")

    def test_direct_z_single_authority_is_fail_closed_in_runtime(self) -> None:
        runtime = Path(__file__).resolve().parents[1] / "runtime.py"
        tree = ast.parse(runtime.read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "run_official_once"
        ]
        self.assertEqual(len(calls), 1)
        source = runtime.read_text()
        self.assertIn("endpoint.direct_z_compute_count != 1", source)
        self.assertIn("endpoint.direct_z_recompute_count != 0", source)


if __name__ == "__main__":
    unittest.main()
