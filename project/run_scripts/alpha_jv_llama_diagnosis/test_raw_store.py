"""Synthetic local evidence storage; no model/backend or scientific endpoint."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import torch

from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.ordered_response_barrier_ode.adapters import FixedZArtifact
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import OverlayDelta, tensor_set_sha256, tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash
from .fixture import EntrySnapshot, FixedTargetBundle
from .normalization_views import NormalizationView
from .raw_store import LocalRawStore, RawStoreBoundary


def fixture(root):
    contexts = (("{}", "The {}"),)
    weights = {"block.weight": torch.zeros((2, 3), dtype=torch.float32)}
    module = SimpleNamespace(cache_c=torch.zeros((5, 3, 3), dtype=torch.float32), cache_c_new=True)
    entry = EntrySnapshot.capture(weights, module)
    target = torch.ones((2, 1), dtype=torch.float32)
    z = FixedZArtifact(target, tensor_sha256(target), "a" * 64, "b" * 64)
    bundle = FixedTargetBundle(z, {"identity_sha256": "b" * 64, "token_ids": torch.tensor([[1, 2, 3]])},
                              canonical_hash(contexts), entry.W_sha256, entry.M_sha256, True)
    norm = NormalizationView.from_source(FrozenNormalization.capture(target, torch.zeros_like(target), entry.W_sha256))
    store = LocalRawStore(root, contexts=contexts, source_identity={"runtime": "CPU_FIXTURE"})
    store.fixed_target(bundle, entry, norm, {"qN_ref": 3., "qF_ref": 4.})
    left = torch.tensor([[1.], [2.]], dtype=torch.float32)
    right = torch.tensor([[1.], [2.], [3.]], dtype=torch.float32)
    delta = OverlayDelta("block.weight", 4, left, right, .5, 0, "d" * 64)
    return store, entry, target, delta


def endpoint(store, entry, target, deltas, *, label="PREFIX", path="PATH", prefix=True):
    weights = {key: value.clone() for key, value in entry.weights.items()}
    for delta in deltas:
        weights[delta.weight_name].addmm_(delta.left, delta.right.T, alpha=delta.coefficient)
    ep = dict(selected_weight_endpoint_sha256=tensor_set_sha256(weights),
              terminal_activation_sha256=tensor_sha256(target), history_append_count=0 if prefix else 1,
              effective_T=1, effective_N=2, parent_T=4, parent_N=8)
    state = dict(alpha_cache=entry.alpha_cache if prefix else entry.alpha_cache + 1, cache_c_new=True)
    store.endpoint(path, label, ep, weights, target, entry, state, deltas)
    return weights, ep


class RawStoreTests(unittest.TestCase):
    def test_fixed_target_complete_fields_saved_once_local_mode(self):
        with TemporaryDirectory() as tmp:
            store, entry, _target, _delta = fixture(Path(tmp) / "raw")
            target = torch.load(store.root / "fixed-target.pt", weights_only=True)
            self.assertEqual(target["bundle"]["W_sha256"], entry.W_sha256)
            self.assertEqual(target["bundle"]["semantic_inventory"]["token_ids"].tolist(), [[1, 2, 3]])
            self.assertEqual(target["contexts"], [["{}", "The {}"]])
            self.assertEqual((store.root / "fixed-target.pt").stat().st_mode & 0o777, 0o600)
            receipt = json.loads((store.root / "fixed-target-receipt.json").read_text())
            self.assertFalse(receipt["replay_verified"])
            self.assertFalse(receipt["target"]["publication_allowed"])

    def test_journal_is_incremental_and_final_post_node_survives(self):
        with TemporaryDirectory() as tmp:
            store, _entry, target, delta = fixture(Path(tmp) / "raw")
            second = delta.with_coefficient(.25)
            store.node("P", 0, dict(stage="PRE_NODE", terminal=target, increments_before=()))
            store.node("P", 0, dict(stage="POST_NODE", terminal=target, increments_after=(delta,)))
            store.node("P", 1, dict(stage="PRE_NODE", terminal=target, increments_before=(delta,)))
            store.node("P", 1, dict(stage="POST_NODE", terminal=target, increments_after=(delta, second)))
            self.assertEqual(len(list((store.root / "P/journal").glob("*.pt"))), 2)
            receipt = json.loads((store.root / "P/node-01-post_node-receipt.json").read_text())
            self.assertEqual(receipt["journal"]["increment_count"], 2)
            self.assertFalse(receipt["dense_W_per_node_saved"])
            self.assertFalse(receipt["journal"]["replay_verified"])

    def test_journal_cannot_change_existing_prefix(self):
        with TemporaryDirectory() as tmp:
            store, _entry, _target, delta = fixture(Path(tmp) / "raw")
            store.journal("P", (delta,))
            with self.assertRaisesRegex(RawStoreBoundary, "CHRONOLOGICAL_JOURNAL_PREFIX_CHANGED"):
                store.journal("P", (delta.with_coefficient(.1),))

    def test_prefix_entry_M_terminal_captured_M_and_actual_W_verified(self):
        with TemporaryDirectory() as tmp:
            store, entry, target, delta = fixture(Path(tmp) / "raw")
            endpoint(store, entry, target, (delta,), label="PREFIX", prefix=True)
            endpoint(store, entry, target, (delta, delta), label="TERMINAL", prefix=False)
            prefix = torch.load(store.root / "PATH/endpoint-PREFIX.pt", weights_only=True)
            terminal = torch.load(store.root / "PATH/endpoint-TERMINAL.pt", weights_only=True)
            self.assertTrue(torch.equal(prefix["alpha_cache"], entry.alpha_cache))
            self.assertTrue(torch.equal(terminal["alpha_cache"], entry.alpha_cache + 1))
            self.assertTrue(prefix["cache_c_new"])
            self.assertTrue(terminal["cache_c_new"])
            self.assertFalse(json.loads((store.root / "PATH/endpoint-PREFIX-receipt.json").read_text())["replay_verified"])
            self.assertEqual(store.verify_reconstruction("PATH", "PREFIX")["status"], "EXACT_RECONSTRUCTION_PASS")
            self.assertEqual(store.verify_reconstruction("PATH", "TERMINAL")["status"], "EXACT_RECONSTRUCTION_PASS")

    def test_weight_sha_wrong_and_missing_method_state_fail(self):
        with TemporaryDirectory() as tmp:
            store, entry, target, _delta = fixture(Path(tmp) / "raw")
            ep = dict(selected_weight_endpoint_sha256="e" * 64,
                      terminal_activation_sha256=tensor_sha256(target), history_append_count=0)
            with self.assertRaisesRegex(RawStoreBoundary, "ENDPOINT_ACTUAL_WEIGHT_SHA"):
                store.endpoint("P", "bad", ep, entry.weights, target, entry,
                               dict(alpha_cache=entry.alpha_cache, cache_c_new=True))
            ep["selected_weight_endpoint_sha256"] = entry.W_sha256
            with self.assertRaisesRegex(RawStoreBoundary, "ENDPOINT_METHOD_STATE_BOUNDARY"):
                store.endpoint("P", "badM", ep, entry.weights, target, entry, None)

    def test_create_once_and_symlink_parent_containment(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store, _entry, _target, delta = fixture(base / "raw")
            store.journal("P", (delta,))
            with self.assertRaises(FileExistsError):
                LocalRawStore(base / "raw", contexts=(), source_identity={})
            (base / "outside").mkdir()
            (base / "link").symlink_to(base / "outside", target_is_directory=True)
            with self.assertRaisesRegex(RawStoreBoundary, "SYMLINK_RAW_PATH"):
                LocalRawStore(base / "link/new", contexts=(), source_identity={})
            (store.root / "escape").symlink_to(base / "outside", target_is_directory=True)
            with self.assertRaisesRegex(RawStoreBoundary, "SYMLINK_RAW_PATH"):
                store.journal("escape", (delta,))
            self.assertEqual(list((base / "outside").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
