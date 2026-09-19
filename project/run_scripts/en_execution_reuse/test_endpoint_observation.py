"""Small CPU ownership/cache fixtures, NOT actual model/GPU runtime evidence.

No model is loaded and no CUDA API is called. The injected transfer is a CPU
clone; H2D counters must stay zero. The scalar scoring fixture keeps the legacy
target-position union/log-softmax/FP64 NLL reduction and invariant head chunks.
"""
from dataclasses import replace
import hashlib
import unittest

import torch
import torch.nn.functional as F

from .endpoint_session import (EndpointCorruptionError, EndpointSession,
                               EndpointSessionError, RuntimePolicy,
                               StaleEndpointError, identity_sha256, tensor_sha256)
from .observation import (CoverageSpec, ObservationBundle, ObservationError,
                          ObservationKey)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.state = dict(epoch={"nonselected": 7, "hooks": [], "training": False},
                          inputs={"ids": "ids-sha", "mask": "mask-sha", "positions": "pos-sha"},
                          source={"source_commit": "fixture", "runtime": "cpu"})
        self.transfers = []

        def transfer(weight, device):
            self.assertEqual(device.type, "cpu")
            result = weight.clone()
            self.transfers.append(result)
            return result

        self.policy = RuntimePolicy(
            shape=(3, 4), device="cpu", model_epoch=self.state["epoch"],
            input_identity=self.state["inputs"], source_identity=self.state["source"],
            epoch_getter=lambda: self.state["epoch"],
            input_identity_getter=lambda: self.state["inputs"],
            source_identity_getter=lambda: self.state["source"], transfer=transfer)
        self.session = EndpointSession({"model": "tiny-linear"}, {"batch": 1, "arm": "reuse"}, self.policy)
        self.addCleanup(self.session.close)
        self.weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 16
        self.native = self.session.bind_native(self.weight)
        self.rows = [dict(sequence_id="7:canonical:new", cache=0, positions=[1, 2],
                          labels=[1, 2], branch="new", kind="canonical", case_id=7),
                     dict(sequence_id="7:canonical:old", cache=0, positions=[1],
                          labels=[2], branch="old", kind="canonical", case_id=7)]
        self.coverage = CoverageSpec.from_rows(self.rows, hidden_shapes={0: (1, 4, 3)},
                                                valid_positions={0: [1, 2, 3]})
        self.hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 5
        self.scored = {r["sequence_id"]: dict(r, nll=0.25+i, strict=(i == 0),
                                            predictions=list(r["labels"]))
                       for i, r in enumerate(self.rows)}

    def bundle(self, handle=None, coverage=None):
        return ObservationBundle(self.session, handle or self.native,
                                 input_manifest=self.state["inputs"],
                                 teacher_identity={"store": "fixture", "sha256": "teacher-sha"},
                                 coverage=coverage or self.coverage)

    def populate(self, bundle, *, hidden=None, current=None, past=None):
        with self.session.readonly(bundle.handle):
            bundle.put_hidden(0, self.hidden if hidden is None else hidden)
            bundle.put_rows("current", self.scored if current is None else current)
            bundle.put_rows("past", {} if past is None else past)
        bundle.seal()
        return bundle


class EndpointTests(Fixture):
    def test_header_and_bytes_digest_matches_legacy_convention(self):
        header = f"{tuple(self.weight.shape)}|{self.weight.dtype}|".encode("ascii")
        expected = hashlib.sha256(header + self.weight.numpy().tobytes()).hexdigest()
        self.assertEqual(tensor_sha256(self.weight), expected)
        self.assertEqual(self.native.identity.weight_sha256, expected)
        self.assertNotEqual(expected, hashlib.sha256(self.weight.numpy().tobytes()).hexdigest())

    def test_identity_keys_cannot_silently_coerce_ints_to_strings(self):
        with self.assertRaisesRegex(TypeError, "STRING_KEYS"):
            identity_sha256({"nested": {1: "alias"}})

    def test_one_transfer_per_endpoint_and_no_per_sequence_hash_or_finite_scan(self):
        candidate = self.session.bind_candidate(self.weight + .125, {"trial": 1})
        with self.session.readonly(candidate) as weight:
            before = self.session.work.copy()
            for _ in range(25):
                self.assertIs(self.session.evaluate_handle(candidate), weight)
            for name in ("hash_calls", "full_validations", "finite_validations",
                         "inference_transfer_calls", "inference_transfer_bytes"):
                self.assertEqual(before[name], self.session.work[name], name)
        self.assertEqual(len(self.transfers), 2)
        self.assertEqual(self.session.work["inference_transfer_calls"], 2)
        self.assertEqual(self.session.work["inference_transfer_bytes"], 2 * 12 * 4)
        self.assertEqual(self.session.work["inference_h2d_calls"], 0)
        self.assertEqual(self.session.work["hash_d2h_calls"], 0)
        self.assertEqual(self.session.resident_slots, 2)
        self.assertEqual(self.session.resident_inference_bytes, 2 * 12 * 4)

    def test_direct_unscoped_extraction_is_rejected(self):
        with self.assertRaisesRegex(EndpointSessionError, "READONLY_SCOPE_REQUIRED"):
            self.session.evaluate_handle(self.native)

    def test_cpu_snapshot_and_transfer_output_do_not_alias_owners(self):
        snapshot = self.session.cpu_snapshot(self.native)
        snapshot.numpy()[0, 0] = 30
        self.transfers[0].numpy()[0, 1] = 40
        with self.session.readonly(self.native) as weight:
            torch.testing.assert_close(weight, self.weight, atol=0, rtol=0)
            self.assertNotEqual(weight.data_ptr(), self.weight.data_ptr())
        self.assertEqual(float(self.session.cpu_snapshot(self.native)[0, 0]), 0)

    def test_injected_transfer_input_alias_is_also_isolated(self):
        retained = []

        def retain_input(value, device):
            retained.append(value)
            return value

        with EndpointSession("model", "batch", replace(self.policy, transfer=retain_input)) as session:
            handle = session.bind_native(self.weight)
            retained[0].numpy()[0, 0] = 999
            with session.readonly(handle) as weight:
                self.assertEqual(float(weight[0, 0]), 0)

    def test_external_source_numpy_alias_mutation_fails_next_boundary(self):
        version = self.weight._version
        self.weight.numpy()[0, 0] = 999
        self.assertEqual(version, self.weight._version)
        with self.assertRaisesRegex(EndpointCorruptionError, "SOURCE_BYTES_MUTATED"):
            with self.session.readonly(self.native):
                self.fail("must fail before evaluation")
        self.assertTrue(self.session.closed)
        self.assertEqual(self.session.resident_slots, 0)

    def test_private_cpu_owner_numpy_corruption_fails_full_check(self):
        owner = self.session._slots["native"].cpu  # Deliberate fault injection only.
        owner.numpy()[0, 0] = 999
        with self.assertRaisesRegex(EndpointCorruptionError, "CPU_OWNER_BYTES_MUTATED"):
            self.session.validate_handle(self.native)

    def test_source_version_mutation_fails_inside_scope(self):
        with self.assertRaisesRegex(EndpointCorruptionError, "SOURCE_METADATA_OR_VERSION"):
            with self.session.readonly(self.native):
                self.weight.add_(1)
                self.session.evaluate_handle(self.native)

    def test_device_inplace_mutation_fails_immediately(self):
        with self.assertRaisesRegex(EndpointCorruptionError, "DEVICE_OWNER_METADATA_OR_VERSION"):
            with self.session.readonly(self.native) as weight:
                weight.add_(1)
                self.session.evaluate_handle(self.native)

    def test_device_numpy_mutation_fails_at_scope_exit_without_version_claim(self):
        with self.assertRaisesRegex(EndpointCorruptionError, "DEVICE_OWNER_BYTES_MUTATED"):
            with self.session.readonly(self.native) as weight:
                version = weight._version
                weight.numpy()[0, 0] = 23
                self.assertEqual(version, weight._version)
                # Per-call checks cannot see NumPy writes. The scope is not
                # trusted/committed until the full exit check below succeeds.
                self.session.evaluate_handle(self.native)
        self.assertTrue(self.session.closed)

    def test_retained_data_alias_is_detected_at_next_scope_entry(self):
        with self.session.readonly(self.native) as weight:
            retained = weight.data
        retained[0, 0] = 18
        with self.assertRaisesRegex(EndpointCorruptionError, "DEVICE_OWNER_BYTES_MUTATED"):
            with self.session.readonly(self.native):
                pass

    def test_epoch_pack_mask_position_and_source_changes_are_stale(self):
        for field, changed in (("epoch", {"nonselected": 8}),
                               ("inputs", {"mask": "changed"}),
                               ("inputs", {"positions": "changed"}),
                               ("source", {"runtime": "different"})):
            with self.subTest(field=field, changed=changed):
                original = self.state[field]
                with EndpointSession("model", "batch", self.policy) as session:
                    handle = session.bind_native(self.weight)
                    self.state[field] = changed
                    with self.assertRaises(StaleEndpointError):
                        with session.readonly(handle):
                            pass
                    self.assertTrue(session.closed)
                self.state[field] = original

    def test_candidate_replacement_keeps_only_two_slots_and_rejects_stale_handle(self):
        first = self.session.bind_candidate(self.weight + .1, "trial-1")
        second = self.session.bind_candidate(self.weight + .2, "trial-2")
        self.assertEqual(self.session.resident_slots, 2)
        self.assertNotEqual(first.identity, second.identity)
        with self.assertRaises(StaleEndpointError):
            self.session.validate_handle(first)

    def test_foreign_and_reconstructed_handles_fail_closed(self):
        for foreign in (replace(self.native),):
            with self.assertRaises(StaleEndpointError):
                self.session.validate_handle(foreign)
        with EndpointSession("model", "other-batch", self.policy) as other:
            foreign = other.bind_native(self.weight)
            with self.assertRaises(StaleEndpointError):
                other.validate_handle(self.native)
            self.assertNotEqual(self.native.identity.session_id, foreign.identity.session_id)

    def test_gradient_is_separate_leaf_and_is_not_inference_residency(self):
        inference_size = self.session.resident_inference_bytes
        with self.session.gradient_leaf(self.native) as leaf:
            self.assertTrue(leaf.is_leaf and leaf.requires_grad)
            self.assertEqual(self.session.gradient_leaf_count, 1)
            self.assertNotEqual(leaf.data_ptr(), self.session.evaluate_handle(self.native).data_ptr())
            (leaf.square().sum()).backward()
            gradient = leaf.grad.detach().clone()
            self.assertEqual(self.session.resident_inference_bytes, inference_size)
        torch.testing.assert_close(gradient, 2 * self.weight, atol=0, rtol=0)
        self.assertEqual(self.session.gradient_leaf_count, 0)
        self.assertIsNone(leaf.grad)
        self.assertFalse(leaf.requires_grad)
        self.assertEqual(self.session.work["inference_transfer_calls"], 1)
        self.assertEqual(self.session.work["gradient_leaf_clones"], 1)
        self.assertEqual(self.session.work["gradient_leaf_bytes"], 12 * 4)

    def test_gradient_exception_closes_and_clears_graph_references(self):
        with self.assertRaisesRegex(RuntimeError, "fixture backward failure"):
            with self.session.gradient_leaf(self.native) as leaf:
                (leaf.square().sum()).backward()
                raise RuntimeError("fixture backward failure")
        self.assertTrue(self.session.closed)
        self.assertIsNone(leaf.grad)
        self.assertFalse(leaf.requires_grad)
        self.assertEqual(self.session.gradient_leaf_count, 0)

    def test_bad_shape_dtype_finite_or_graph_is_technical_failure(self):
        bad_weights = [torch.ones(2, 4), self.weight.double(),
                       torch.full_like(self.weight, float("nan")),
                       self.weight.clone().requires_grad_()]
        for bad in bad_weights:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                session = EndpointSession("model", "batch", self.policy)
                with self.assertRaises((ValueError, FloatingPointError)):
                    session.bind_native(bad)
                self.assertTrue(session.closed)

    def test_transfer_corruption_and_oom_are_not_native_fallback(self):
        def oom(value, device):
            raise RuntimeError("fixture OOM")

        for transfer, error in ((lambda value, device: value + 1, EndpointCorruptionError),
                                (oom, RuntimeError)):
            session = EndpointSession("model", "batch", replace(self.policy, transfer=transfer))
            with self.assertRaises(error):
                session.bind_native(self.weight)
            self.assertTrue(session.closed)
            self.assertEqual(session.resident_slots, 0)

    def test_exception_releases_both_slots_and_observation(self):
        candidate = self.session.bind_candidate(self.weight + .1, "trial-1")
        bundle = self.populate(self.bundle(candidate))
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with self.session.readonly(candidate):
                raise RuntimeError("fixture failure")
        self.assertTrue(bundle.closed)
        self.assertIsNone(bundle.complete_marker)
        self.assertEqual(self.session.resident_slots, 0)
        self.session.close()  # Idempotent.


class ObservationTests(Fixture):
    def test_partial_observation_never_hits_or_gets_complete_marker(self):
        bundle = self.bundle()
        bundle.put_hidden(0, self.hidden)
        self.assertIsNone(bundle.lookup_hidden(0, key=bundle.key))
        self.assertIsNone(bundle.lookup_rows("current", key=bundle.key))
        self.assertIsNone(bundle.complete_marker)
        with self.assertRaisesRegex(ObservationError, "PARTIAL_COVERAGE"):
            bundle.seal()

    def test_missing_empty_past_marker_is_still_partial(self):
        bundle = self.bundle()
        bundle.put_hidden(0, self.hidden)
        bundle.put_rows("current", self.scored)
        with self.assertRaisesRegex(ObservationError, "PARTIAL_COVERAGE"):
            bundle.seal()

    def test_seal_requires_successful_scope_exit(self):
        bundle = self.bundle()
        with self.session.readonly(self.native):
            bundle.put_hidden(0, self.hidden)
            bundle.put_rows("current", self.scored)
            bundle.put_rows("past", {})
            with self.assertRaisesRegex(ObservationError, "SEAL_AFTER_READONLY"):
                bundle.seal()
        self.assertFalse(bundle.complete_coverage)
        self.assertEqual(bundle.seal(), bundle.key)

    def test_hidden_detached_cpu_and_defensive_copies_remove_graph_and_aliases(self):
        source = self.hidden.clone().requires_grad_()
        graph_hidden = source.square()
        bundle = self.populate(self.bundle(), hidden=graph_hidden)
        stored = bundle.lookup_hidden(0, key=bundle.key)
        self.assertEqual(stored.device.type, "cpu")
        self.assertEqual(stored.dtype, torch.float32)
        self.assertFalse(stored.requires_grad)
        self.assertIsNone(stored.grad_fn)
        original = stored.clone()
        stored.numpy()[0, 1, 1] = 777
        graph_hidden.data.fill_(44)
        torch.testing.assert_close(bundle.lookup_hidden(0, key=bundle.key), original, atol=0, rtol=0)

    def test_score_records_defensively_copy_and_preserve_values_exactly(self):
        bundle = self.populate(self.bundle())
        result = bundle.lookup_rows("current", key=bundle.key)
        self.assertEqual(result, self.scored)
        result[self.rows[0]["sequence_id"]]["predictions"][0] = 99
        self.scored[self.rows[0]["sequence_id"]]["nll"] = 99
        result2 = bundle.lookup_rows("current", key=bundle.key)
        self.assertEqual(result2[self.rows[0]["sequence_id"]]["nll"], .25)
        self.assertEqual(result2[self.rows[0]["sequence_id"]]["predictions"], [1, 2])
        self.assertEqual(bundle.lookup_rows("past", key=bundle.key), {})

    def test_label_position_teacher_input_epoch_trial_and_coverage_mismatch_miss(self):
        bundle = self.populate(self.bundle())
        fields = ("input_manifest_sha256", "rows_identity_sha256", "teacher_identity_sha256",
                  "coverage_sha256", "model_epoch_sha256", "trial_identity_sha256")
        for field in fields:
            with self.subTest(field=field):
                key = replace(bundle.key, **{field: "different"})
                self.assertIsNone(bundle.lookup_rows("current", key=key))
                self.assertIsNone(bundle.lookup_hidden(0, key=key))
        # Actual row label/position changes produce distinct keys without a
        # caller manufacturing a digest or relying only on sequence labels.
        changed_rows = [dict(self.rows[0], positions=[2, 3], labels=[2, 1]), self.rows[1]]
        other_coverage = CoverageSpec.from_rows(changed_rows, hidden_shapes={0: (1, 4, 3)},
                                                 valid_positions={0: [1, 2, 3]})
        key = ObservationKey.create(self.native, input_manifest=self.state["inputs"],
                                    teacher_identity="other-teacher", coverage=other_coverage)
        self.assertIsNone(bundle.lookup_rows("current", key=key))

    def test_same_bytes_other_trial_is_not_same_endpoint(self):
        candidate = self.session.bind_candidate(self.weight.clone(), "trial-1")
        bundle = self.populate(self.bundle(candidate))
        key = bundle.key
        next_candidate = self.session.bind_candidate(self.weight.clone(), "trial-2")
        self.assertTrue(bundle.closed)
        self.assertEqual(candidate.identity.weight_sha256, next_candidate.identity.weight_sha256)
        self.assertNotEqual(key.endpoint_identity, next_candidate.identity)
        with self.assertRaises(ObservationError):
            bundle.lookup_rows("current", key=key)

    def test_candidate_rejection_invalidates_only_candidate_observations(self):
        native_bundle = self.populate(self.bundle())
        candidate = self.session.bind_candidate(self.weight + .1, "trial-1")
        candidate_bundle = self.populate(self.bundle(candidate))
        self.session.release_candidate()
        self.assertTrue(candidate_bundle.closed)
        self.assertEqual(candidate_bundle.work["hidden_bytes"], 12 * 4)
        self.assertFalse(native_bundle.closed)
        self.assertEqual(native_bundle.lookup_rows("current", key=native_bundle.key), self.scored)
        self.assertEqual(self.session.resident_slots, 1)

    def test_no_update_fallback_and_close_release_all_observations(self):
        native_bundle = self.populate(self.bundle())
        self.session.release_candidate()  # Native-only no-update is legal.
        self.assertFalse(native_bundle.closed)
        self.session.close()
        self.assertTrue(native_bundle.closed)
        self.assertFalse(native_bundle.complete_coverage)
        self.assertEqual(native_bundle._hidden, {})

    def test_hidden_numpy_corruption_is_not_a_silent_miss(self):
        bundle = self.populate(self.bundle())
        value, _, _ = next(iter(bundle._hidden.values()))  # Fault injection.
        version = value._version
        value.numpy()[0, 0, 0] = 99
        self.assertEqual(value._version, version)
        with self.assertRaisesRegex(ObservationError, "HIDDEN_BYTES_MUTATED"):
            bundle.lookup_hidden(0, key=bundle.key)
        self.assertTrue(bundle.closed)

    def test_private_hidden_graph_attachment_is_rejected(self):
        bundle = self.populate(self.bundle())
        value, _, _ = next(iter(bundle._hidden.values()))  # Fault injection.
        value.requires_grad_(True)
        with self.assertRaisesRegex(ObservationError, "GRAPH_ATTACHED"):
            bundle.lookup_hidden(0, key=bundle.key)

    def test_direct_coverage_construction_cannot_retain_mutable_aliases(self):
        with self.assertRaisesRegex(TypeError, "IMMUTABLE_COVERAGE"):
            replace(self.coverage, hidden_shapes=list(self.coverage.hidden_shapes))
        with self.assertRaisesRegex(TypeError, "IMMUTABLE_COVERAGE"):
            replace(self.coverage, hidden_shapes=(("int:0", [1, 4, 3]),))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_COVERAGE"):
            replace(self.coverage, hidden_shapes=self.coverage.hidden_shapes * 2)

    def test_nonfinite_hidden_bad_dtype_and_hidden_shape_are_rejected(self):
        for value in (self.hidden.double(), self.hidden[:, :2],
                      torch.full_like(self.hidden, float("inf"))):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                with self.bundle() as bundle:
                    with self.assertRaises((ValueError, FloatingPointError)):
                        bundle.put_hidden(0, value)
                    self.assertFalse(bundle.complete_coverage)

    def test_score_metadata_missing_row_and_graph_payload_rejected(self):
        changed = {sid: dict(row) for sid, row in self.scored.items()}
        changed[self.rows[0]["sequence_id"]]["labels"] = [9, 9]
        missing = {self.rows[0]["sequence_id"]: self.scored[self.rows[0]["sequence_id"]]}
        for rows in (changed, missing):
            with self.bundle() as bundle:
                with self.assertRaises(ObservationError):
                    bundle.put_rows("current", rows)
        with self.bundle() as bundle:
            with self.assertRaises(TypeError):
                bundle.put_stats("invariant_stats", {"graph": self.hidden.requires_grad_()})

    def test_nonempty_past_rows_and_own_anchor_metadata_are_preserved(self):
        past = dict(self.rows[0], sequence_id="past:3:v2:new", cache=9,
                    history_version=2, own_native_sha256="past-own-WN")
        coverage = CoverageSpec.from_rows(self.rows, hidden_shapes={0: (1, 4, 3)},
                                          valid_positions={0: [1, 2, 3]}, past_rows=[past])
        score = {past["sequence_id"]: dict(past, nll=.75, strict=True, predictions=[1, 2])}
        bundle = self.populate(self.bundle(coverage=coverage), past=score)
        self.assertEqual(bundle.lookup_rows("past", key=bundle.key), score)

    def test_partial_scope_exception_never_becomes_reusable(self):
        bundle = self.bundle()
        with self.assertRaisesRegex(RuntimeError, "suffix failed"):
            with self.session.readonly(self.native):
                bundle.put_hidden(0, self.hidden)
                raise RuntimeError("suffix failed")
        self.assertTrue(bundle.closed)
        self.assertIsNone(bundle.complete_marker)

    def test_hit_within_scope_does_not_hash_weight_for_each_sequence(self):
        bundle = self.populate(self.bundle())
        with self.session.readonly(self.native):
            before = self.session.work["hash_calls"]
            for _ in range(10):
                bundle.lookup_hidden(0, key=bundle.key)
                bundle.lookup_rows("current", key=bundle.key)
            self.assertEqual(before, self.session.work["hash_calls"])
        self.assertEqual(bundle.work["hidden_hits"], 10)
        self.assertEqual(bundle.work["row_hits"], 10)

    def test_coverage_valid_positions_row_duplicates_and_missing_cache_rejected(self):
        for positions in ([0, 1, 4], [2, 1], [1, 1], []):
            with self.assertRaises(ValueError):
                CoverageSpec.from_rows(self.rows, hidden_shapes={0: (1, 4, 3)},
                                       valid_positions={0: positions})
        with self.assertRaises(ValueError):
            CoverageSpec.from_rows(self.rows * 2, hidden_shapes={0: (1, 4, 3)},
                                   valid_positions={0: [1, 2, 3]})
        with self.assertRaises(ValueError):
            CoverageSpec.from_rows(self.rows, hidden_shapes={}, valid_positions={})

    def test_byte_exact_cpu_scoring_strict_pair_and_chunked_max_rms(self):
        """Synthetic linear CPU fixture only; no Llama/suffix/model claim."""
        keys = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 31
        residual = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 19
        head = torch.tensor([[.3, .6, .2], [.7, -.2, .3], [.1, .5, -.4]])

        def hidden(weight):
            return residual + F.linear(keys, weight)

        def score(value):
            positions = sorted({p for row in self.rows for p in row["positions"]})
            logits = F.linear(value[0, positions], head)
            lp, pred = logits.log_softmax(-1), logits.argmax(-1)
            result = {}
            for row in self.rows:
                idx = [positions.index(p) for p in row["positions"]]
                label = torch.tensor(row["labels"])
                result[row["sequence_id"]] = dict(row, nll=float(-lp[idx, label].double().mean()),
                    strict=bool((pred[idx] == label).all()), predictions=pred[idx].tolist())
            return result

        legacy_hidden = hidden(self.weight)
        legacy_rows = score(legacy_hidden)
        bundle = self.bundle()
        with self.session.readonly(self.native) as weight:
            candidate_hidden = hidden(weight)
            bundle.put_hidden(0, candidate_hidden)
            bundle.put_rows("current", score(candidate_hidden))
            bundle.put_rows("past", {})
        bundle.seal()
        observed_rows = bundle.lookup_rows("current", key=bundle.key)
        self.assertEqual(observed_rows, legacy_rows)
        sid_new, sid_old = [row["sequence_id"] for row in self.rows]
        self.assertEqual(observed_rows[sid_new]["nll"] < observed_rows[sid_old]["nll"],
                         legacy_rows[sid_new]["nll"] < legacy_rows[sid_old]["nll"])
        with self.session.readonly(self.native):
            cached_hidden = bundle.lookup_hidden(0, key=bundle.key)
            positions = torch.tensor([1, 2, 3])
            diffs = [F.linear(cached_hidden[0, part], head).double()
                     - F.linear(legacy_hidden[0, part], head).double()
                     for part in positions.split(16)]
        diff = torch.cat(diffs)
        self.assertEqual(float(diff.abs().max()), 0.)
        self.assertEqual(float(diff.square().mean().sqrt()), 0.)


if __name__ == "__main__":
    unittest.main()
