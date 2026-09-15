"""CPU disk transaction tests; not actual Llama/GPU resume certification."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from project.run_scripts.single_layer_zflow.durable import (
    CheckpointIntegrityError, CheckpointStore, prepare_state, tensor_sha256,
)
from project.run_scripts.single_layer_zflow.transaction import CommitConflict, CommitError


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkpoints"
        self.store = CheckpointStore(self.root)
        self.w = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
        self.m = torch.eye(5, dtype=torch.float32) / 3
        self.x = torch.tensor([[.1, .2], [.3, -.1], [-.2, .4]], dtype=torch.float64)
        self.b = torch.arange(10, dtype=torch.float64).reshape(2, 5) / 20
        self.k = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        self.s = (self.b @ self.m.double() @ self.b.T + self.b @ self.b.T) / 2
        self.wc = (self.w.double() + self.x @ self.b).float()
        self.kwargs = {
            "config": {"lambda_write": 1, "lambda_flow": 1, "barrier": "off"},
            "source": {"commit": "cpu-fixture-source", "base": "fixture-W0"},
            "context": {"templates": [["{}"], ["test {}"]], "request_ids": [1, 2]},
            "rng": {"python": random.Random(71).getstate(),
                    "torch_cpu": torch.Generator().manual_seed(71).get_state()},
            "ledger": {"oracle": 3, "accepted": 1, "rejected": 1},
            "cache_resume_fingerprint": "fixture-positions-v1",
        }

    def prepare(self, *, accepted=1, w=None, m=None):
        w = self.w if w is None else w
        m = self.m if m is None else m
        x = self.x if accepted else torch.zeros_like(self.x)
        wc = (w.double() + x @ self.b).float() if accepted else w.clone()
        return prepare_state(w, m, wc, x, self.b, self.s, self.k,
                             accepted=accepted, request_count=2,
                             parity_evidence={"passed": True, "scope": "CPU fixture only"},
                             cost_evidence={"passed": True, "scope": "CPU fixture only"})

    def publish(self, prepared=None, *, batch="B001", parent=None, index=2, **changes):
        return self.store.publish(batch, parent, index,
                                  self.prepare() if prepared is None else prepared,
                                  **(self.kwargs | changes))

    def test_private_preparation_fp32_history_and_actual_fp64_cost(self):
        w, m = self.w.clone(), self.m.clone()
        prepared = self.prepare()
        delta = self.wc.double() - self.w.double()
        expected = (((delta @ self.m.double()) * delta).sum() + delta.square().sum()) / 4
        self.assertEqual(prepared.details["actual_delta_cost_fp64"], float(expected))
        torch.testing.assert_close(prepared.tensors["M"], self.m + self.k @ self.k.T, atol=0, rtol=0)
        self.assertEqual(prepared.details["history_append"], 1)
        prepared.tensors["W"].zero_()
        prepared.tensors["M"].zero_()
        torch.testing.assert_close(self.w, w, atol=0, rtol=0)
        torch.testing.assert_close(self.m, m, atol=0, rtol=0)

    def test_complete_bundle_load_and_rng_roundtrip(self):
        first = self.publish()
        loaded = self.store.load("B001", expected_config_sha256=first["config_sha256"],
                                 expected_source_sha256=first["source_sha256"],
                                 expected_cache_resume_fingerprint="fixture-positions-v1")
        self.assertEqual(first["receipt_sha256"], loaded["receipt"]["receipt_sha256"])
        self.assertEqual(loaded["metadata"]["next_batch_index"], 2)
        self.assertEqual(loaded["metadata"]["context"], self.kwargs["context"])
        self.assertEqual(loaded["rng"]["python"], self.kwargs["rng"]["python"])
        torch.testing.assert_close(loaded["rng"]["torch_cpu"], self.kwargs["rng"]["torch_cpu"], atol=0, rtol=0)
        self.assertEqual(set(loaded["tensors"]), {"W", "M", "X", "B", "S", "K"})
        torch.testing.assert_close(loaded["tensors"]["W"], self.wc, atol=0, rtol=0)
        self.assertFalse(first["replayed"])

    def test_same_payload_retry_is_noop_and_changed_payload_conflicts(self):
        prepared = self.prepare()
        first = self.publish(prepared)
        before = {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (self.root / "B001").iterdir()}
        second = self.publish(prepared)
        self.assertTrue(second["replayed"])
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        self.assertEqual(before, {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                                  for p in (self.root / "B001").iterdir()})
        with self.assertRaises(CommitConflict):
            self.publish(prepared, ledger={"oracle": 4})
        torch.testing.assert_close(self.store.load("B001")["tensors"]["M"],
                                   self.m + self.k @ self.k.T, atol=0, rtol=0)

    def test_modified_prepared_endpoint_is_refused_before_publication(self):
        prepared = self.prepare()
        prepared.tensors["M"].zero_()
        with self.assertRaisesRegex(CommitConflict, "prepared endpoint"):
            self.publish(prepared)
        self.assertFalse((self.root / "B001").exists())
        prepared = self.prepare()
        prepared.tensors["B"].zero_()
        with self.assertRaisesRegex(CommitConflict, "tensor inventory"):
            self.publish(prepared)

    def test_no_update_keeps_w_m_append_zero_but_advances_index(self):
        first = self.publish(self.prepare(accepted=0))
        self.assertEqual(first["status"], "NO_UPDATE")
        self.assertEqual(first["history_append"], 0)
        loaded = self.store.load("B001")
        torch.testing.assert_close(loaded["tensors"]["W"], self.w, atol=0, rtol=0)
        torch.testing.assert_close(loaded["tensors"]["M"], self.m, atol=0, rtol=0)
        self.assertEqual(loaded["metadata"]["prepared"]["actual_delta_cost_fp64"], 0)
        second = self.publish(batch="B002", parent=first, index=3)
        self.assertEqual(second["next_batch_index"], 3)

    def test_resume_next_batch_matches_uninterrupted_state_and_rng(self):
        first = self.publish()
        loaded = CheckpointStore(self.root).load("B001")
        torch_rng = torch.Generator()
        torch_rng.set_state(loaded["rng"]["torch_cpu"])
        python_rng = random.Random()
        python_rng.setstate(loaded["rng"]["python"])
        baseline_torch = torch.Generator().manual_seed(71)
        baseline_python = random.Random(71)
        torch.testing.assert_close(torch.rand(5, generator=torch_rng),
                                   torch.rand(5, generator=baseline_torch), atol=0, rtol=0)
        self.assertEqual(python_rng.random(), baseline_python.random())
        second_prepared = self.prepare(w=loaded["tensors"]["W"], m=loaded["tensors"]["M"])
        second = self.publish(second_prepared, batch="B002", parent=first, index=3)
        expected_m = (self.m + self.k @ self.k.T) + self.k @ self.k.T
        expected_w = (self.wc.double() + self.x @ self.b).float()
        resumed = self.store.load("B002")
        torch.testing.assert_close(resumed["tensors"]["W"], expected_w, atol=0, rtol=0)
        torch.testing.assert_close(resumed["tensors"]["M"], expected_m, atol=0, rtol=0)
        self.assertEqual(second["history_append"], 1)
        self.assertEqual(resumed["metadata"]["parent"]["receipt_sha256"], first["receipt_sha256"])

    def test_stale_w_m_wrong_parent_source_and_skipped_index_refused(self):
        first = self.publish()
        with self.assertRaisesRegex(CommitConflict, "entry W/M"):
            self.publish(batch="B002", parent=first, index=3)
        loaded = self.store.load("B001")
        prepared = self.prepare(w=loaded["tensors"]["W"], m=loaded["tensors"]["M"])
        with self.assertRaisesRegex(CommitConflict, "next batch"):
            self.publish(prepared, batch="B003", parent=first, index=4)
        with self.assertRaisesRegex(CommitConflict, "source/config"):
            self.publish(prepared, batch="B002", parent=first, index=3, source={"commit": "changed"})
        with self.assertRaisesRegex(CommitConflict, "parent receipt"):
            self.publish(prepared, batch="B002", parent=first | {"payload_sha256": "wrong"}, index=3)
        with self.assertRaisesRegex(CommitConflict, "first batch"):
            self.publish(prepared, batch="B002", index=3)

    def test_crash_before_publish_never_becomes_resume_authority(self):
        for point in ("after_payload", "after_metadata", "after_manifest", "before_publish"):
            with self.subTest(point=point):
                store = CheckpointStore(self.root / point)
                def crash(name):
                    if name == point:
                        raise RuntimeError("injected crash")
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    store.publish("B001", None, 2, self.prepare(), crash_hook=crash, **self.kwargs)
                self.assertFalse((store.root / "B001").exists())
                self.assertEqual(len(list(store.root.glob(".B001.partial-*"))), 1)
                with self.assertRaises(CheckpointIntegrityError):
                    store.load("B001")
                # Retry uses a fresh private staging; failed evidence remains.
                receipt = store.publish("B001", None, 2, self.prepare(), **self.kwargs)
                self.assertFalse(receipt["replayed"])
                self.assertEqual(len(list(store.root.glob(".B001.partial-*"))), 1)

    def test_crash_after_atomic_publication_retries_without_double_append(self):
        def crash(point):
            if point == "after_publish":
                raise RuntimeError("injected post-publish crash")
        with self.assertRaises(RuntimeError):
            self.publish(crash_hook=crash)
        loaded = self.store.load("B001")
        replay = self.publish()
        self.assertTrue(replay["replayed"])
        torch.testing.assert_close(loaded["tensors"]["M"], self.m + self.k @ self.k.T, atol=0, rtol=0)

    def test_corruption_incomplete_and_symlink_members_fail_closed(self):
        self.publish()
        state = self.root / "B001" / "state.pt"
        with state.open("r+b") as handle:
            handle.seek(40)
            original = handle.read(1)
            handle.seek(40)
            handle.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(CheckpointIntegrityError, "SHA/size"):
            self.store.load("B001")
        incomplete = self.root / "B002"
        incomplete.mkdir()
        with self.assertRaises(CheckpointIntegrityError):
            self.store.load("B002")
        symlink = self.root / "B003"
        symlink.symlink_to(self.root / "B001", target_is_directory=True)
        with self.assertRaises(CheckpointIntegrityError):
            self.store.load("B003")

    def test_expected_resume_identity_is_checked(self):
        self.publish()
        for field in ("expected_config_sha256", "expected_source_sha256",
                      "expected_cache_resume_fingerprint"):
            with self.subTest(field=field), self.assertRaises(CheckpointIntegrityError):
                self.store.load("B001", **{field: "wrong"})

    def test_failed_parity_cost_and_invalid_no_update_never_prepare(self):
        args = (self.w, self.m, self.wc, self.x, self.b, self.s, self.k)
        with self.assertRaisesRegex(CommitError, "COMMIT_PARITY_FAIL"):
            prepare_state(*args, accepted=1, request_count=2)
        with self.assertRaisesRegex(CommitError, "COMMIT_COST_FAIL"):
            prepare_state(*args, accepted=1, request_count=2, parity_evidence={"passed": True})
        with self.assertRaisesRegex(ValueError, "no-update"):
            prepare_state(*args, accepted=0, request_count=2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_tensor_digest_handles_noncontiguous_view_and_dtype(self):
        value = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        self.assertEqual(tensor_sha256(value), tensor_sha256(value.clone()))
        self.assertNotEqual(tensor_sha256(value), tensor_sha256(value.double()))
        self.assertEqual(tensor_sha256(value.bfloat16()), tensor_sha256(value.bfloat16().clone()))

    def test_batch_path_escape_is_rejected(self):
        for name in ("../escape", "/absolute", ".partial-001", "a/b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.publish(batch=name)

    def test_source_input_and_rng_mutation_after_publish_cannot_change_bundle(self):
        first = self.publish()
        self.kwargs["rng"]["torch_cpu"].zero_()
        self.kwargs["context"]["request_ids"].append(999)
        loaded = self.store.load("B001")
        self.assertEqual(loaded["metadata"]["context"]["request_ids"], [1, 2])
        self.assertFalse(bool((loaded["rng"]["torch_cpu"] == 0).all()))
        self.assertEqual(first["receipt_sha256"], loaded["receipt"]["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
