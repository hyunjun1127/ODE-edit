"""Small CPU fixtures for fail-close inventory/order and checkpoint semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from .integrity import (IntegrityError, audit_metrics, canonical_hash,
                        audit_checkpoint, confirm_unchanged, finite_count, snapshot_members)


class IntegrityTests(unittest.TestCase):
    def test_inventory_rehash_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text('{"a":1}')
            inventory = snapshot_members(root)
            confirm_unchanged(root, inventory)
            (root / "a.json").write_text('{"a":2,"b":3}')
            with self.assertRaisesRegex(IntegrityError, "RAW_CHANGED"):
                confirm_unchanged(root, inventory)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "target").write_text("x")
            (root / "alias").symlink_to(root / "target")
            with self.assertRaisesRegex(IntegrityError, "SYMLINK"):
                snapshot_members(root)

    def test_finite_fail_close(self):
        self.assertEqual(finite_count({"a": [1., 2., None]}), 2)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(IntegrityError, "NONFINITE"):
                finite_count({"a": [value]})

    def test_prompt_cardinality_order(self):
        from .integrity import KINDS
        records = [{"request_sha256": "a"}, {"request_sha256": "b"}]
        panels = {kind: [dict(request_sha256=r["request_sha256"], prompt_index=i)
                         for r in records for i in range(n)] for kind, n in KINDS.items()}
        audit_metrics(panels, records, "toy")
        panels["rephrase_target_new"].reverse()
        with self.assertRaisesRegex(IntegrityError, "METRIC_ORDER"):
            audit_metrics(panels, records, "toy")

    def test_canonical_hash_stable(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash([1, 2]), canonical_hash([2, 1]))

    def test_cpu_checkpoint_byte_identity(self):
        import torch
        from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256, tensor_set_sha256
        weights = {f"model.layers.{i}.mlp.down_proj.weight": torch.full((2, 3), float(i)) for i in range(4, 9)}
        cache = torch.ones(5, 3, 3)
        commit = dict(committed_weight_sha256=tensor_set_sha256(weights),
                      committed_M_content_sha256=tensor_sha256(cache))
        receipt = dict(selected_weight_sha256=commit["committed_weight_sha256"],
                       M_sha256=commit["committed_M_content_sha256"])
        metadata = {"batch": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(dict(weights=weights, alpha_cache=cache, cache_c_new=False,
                            commit=commit, metadata=metadata), path)
            result = audit_checkpoint(path, receipt, commit, metadata)
            self.assertEqual(result["status"], "CPU_TENSOR_FULL_REHASH_PASS")
            bad = dict(receipt, M_sha256="0" * 64)
            with self.assertRaisesRegex(IntegrityError, "CHECKPOINT_CACHE_HASH"):
                audit_checkpoint(path, bad, commit, metadata)


if __name__ == "__main__":
    unittest.main()
