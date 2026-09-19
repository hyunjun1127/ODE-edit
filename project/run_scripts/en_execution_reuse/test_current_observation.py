"""CPU toy oracle integration, no model load/GPU/pretrained-parity claim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullTokenCache, FullWeightLlamaOracle
from project.run_scripts.single_layer_edit_preserving_correction.binding import quality_ok, score_rows
from project.run_scripts.single_layer_edit_preserving_correction import geometry

from .current_observation import CurrentObservationController
from .endpoint_session import EndpointSession, RuntimePolicy, tensor_sha256
from .observation import ObservationError


def _legacy_invariant():
    # Execute the actual unchanged runtime.invariant function without importing
    # Runtime's model/asset loader dependencies. This does not synthesize a
    # replacement expected formula or modify source files.
    source = Path(__file__).parents[1] / "single_layer_edit_preserving_correction/runtime.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "invariant")
    namespace = dict(geometry=geometry, score_rows=score_rows)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["invariant"]


class ToyOracle:
    """Only suffix arithmetic is synthetic; reuse actual head/comparison APIs."""
    hidden = FullWeightLlamaOracle.hidden
    logits_at = FullWeightLlamaOracle.logits_at
    logits_from_hidden = FullWeightLlamaOracle.logits_from_hidden
    compare_logits = FullWeightLlamaOracle.compare_logits
    _head = FullWeightLlamaOracle._head

    def __init__(self):
        self.device, self.shape, self.head_chunk_positions = torch.device("cpu"), (3, 4), 16
        self.work = dict(cached_suffix_forwards=0, head_positions=0,
                         comparison_logit_elements=0, weight_checks=0)
        self.head_shapes = []
        self.fail_suffix = False
        self.head = torch.tensor([[.3, .6, .2], [.7, -.2, .3], [.1, .5, -.4],
                                  [-.2, .1, .3], [.4, -.1, .2], [.2, .2, .2], [.3, .3, -.2]])

        def lm_head(value):
            self.head_shapes.append(tuple(value.shape))
            return F.linear(value, self.head)

        self.model = SimpleNamespace(lm_head=lm_head)
        self.caches = []
        for count, mask in ((20, [0]+[1]*19), (6, [1]*5+[0])):
            ids = torch.arange(count)[None]
            packed = dict(input_ids=ids, attention_mask=torch.tensor([mask]),
                          position_ids=torch.arange(count)[None])
            keys = torch.arange(count*4, dtype=torch.float32).reshape(1, count, 4)/80
            keys[:, :, 2:] = 0  # Exact two-dimensional nullspace for fixture.
            residual = torch.arange(count*3, dtype=torch.float32).reshape(1, count, 3)/90
            self.caches.append(FullTokenCache(packed, keys, residual, {}))

    def _guard(self):
        pass

    def _weight(self, weight):
        self.work["weight_checks"] += 1
        if not torch.isfinite(weight).all():
            raise FloatingPointError("NONFINITE_TOY_WEIGHT")
        return weight.to(self.device)

    def suffix_hidden(self, index, weight):
        if self.fail_suffix:
            raise RuntimeError("fixture suffix failure")
        self._guard()
        weight = self._weight(weight)
        self.work["cached_suffix_forwards"] += 1
        cache = self.caches[index]
        return cache.residual + F.linear(cache.keys, weight)


class CurrentObservationTests(unittest.TestCase):
    def setUp(self):
        self.oracle = ToyOracle()
        self.WN = torch.arange(12, dtype=torch.float32).reshape(3, 4)/16
        self.rows = [
            dict(cache=0, positions=[2, 5, 17], labels=[1, 2, 1], sequence_id="1:canonical:new", branch="new", kind="canonical"),
            dict(cache=0, positions=[2], labels=[3], sequence_id="1:canonical:old", branch="old", kind="canonical"),
            dict(cache=1, positions=[0, 3], labels=[0, 1], sequence_id="2:native:new", branch="new", kind="native"),
        ]
        self.identity = dict(epoch=1, inputs="toy-packs", source="test-current-v1")
        policy = RuntimePolicy(shape=(3, 4), device="cpu", model_epoch=1,
            input_identity="toy-packs", source_identity="test-current-v1",
            epoch_getter=lambda: self.identity["epoch"],
            input_identity_getter=lambda: self.identity["inputs"],
            source_identity_getter=lambda: self.identity["source"],
            transfer=lambda value, device: value.clone())
        self.session = EndpointSession("toy-model", dict(batch=1, arm="reuse"), policy)
        self.addCleanup(self.session.close)
        self.native = self.session.bind_native(self.WN)
        self.controller = CurrentObservationController(self.WN, self.oracle, self.rows,
            self.session, self.native, teacher_identity="teacher-v1", input_identity="toy-packs")
        self.addCleanup(self.controller.close)
        self.legacy_invariant = _legacy_invariant()
        self.K = torch.cat([c.valid_keys() for c in self.oracle.caches], dim=1)
        self.allowed = geometry.RightSpace(np.eye(4), np.empty((4, 0)), "RESOLVED")

    def test_native_anchor_byte_exact_and_once(self):
        legacy = ToyOracle()
        expected = score_rows(legacy, self.WN, self.rows)
        self.assertEqual(self.controller.anchor_rows, expected)
        self.assertEqual(self.controller.work["native_observations"], 1)
        self.assertEqual(self.oracle.work["cached_suffix_forwards"], 2)
        self.assertEqual(self.oracle.work["weight_checks"], 0)
        self.assertEqual(self.oracle.head_shapes, [(3, 3), (2, 3)])
        self.assertEqual(self.session.work["inference_transfer_calls"], 1)

    def test_exact_legacy_guard_invariant_and_four_to_one_candidate_suffix(self):
        candidate = self.WN.clone()
        candidate[:, 2:] += .125
        ideal = candidate.double()-self.WN.double()
        legacy = ToyOracle()
        anchor = score_rows(legacy, self.WN, self.rows)
        legacy_start = legacy.work["cached_suffix_forwards"]
        legacy_quality = quality_ok(score_rows(legacy, candidate, self.rows), anchor)
        expected = self.legacy_invariant(legacy, self.rows, anchor, candidate,
                                         self.WN, ideal, self.K, self.allowed)
        self.assertEqual(legacy.work["cached_suffix_forwards"]-legacy_start, 4*2)
        handle = self.session.bind_candidate(candidate, dict(trial=1))
        start = self.oracle.work["cached_suffix_forwards"]
        self.assertEqual(self.controller.guard(handle), legacy_quality)
        self.assertTrue(legacy_quality[0])
        got = self.controller.invariant(handle, ideal, self.K, self.allowed)
        self.assertEqual(got, expected)
        self.assertEqual(self.oracle.work["cached_suffix_forwards"]-start, 2)
        self.assertEqual(self.oracle.work["weight_checks"], 0)
        self.assertEqual(self.controller.work["score_rows_calls"], 2)  # Native + candidate.
        self.assertEqual(got["logit_elements"], 24*7)
        self.assertEqual(self.session.work["inference_transfer_calls"], 2)
        self.assertEqual(self.session.work["inference_h2d_calls"], 0)
        # Two unchanged target-union schedules, then interleaved native/
        # candidate full-valid-token head chunks 16, 3, and 5.
        self.assertEqual(self.oracle.head_shapes,
            [(3, 3), (2, 3)]*2 + [(16, 3), (16, 3), (3, 3), (3, 3), (5, 3), (5, 3)])

    def test_nonzero_logit_rounding_and_nll_scalar_exactness(self):
        candidate = self.WN.clone()
        candidate[:, 0] += 1e-6
        ideal = torch.zeros_like(candidate, dtype=torch.float64)
        legacy = ToyOracle()
        anchor = score_rows(legacy, self.WN, self.rows)
        expected_quality = quality_ok(score_rows(legacy, candidate, self.rows), anchor)
        self.assertTrue(expected_quality[0])
        expected = self.legacy_invariant(legacy, self.rows, anchor, candidate,
                                         self.WN, ideal, self.K, self.allowed)
        handle = self.session.bind_candidate(candidate, 1)
        self.assertEqual(self.controller.guard(handle), expected_quality)
        got = self.controller.invariant(handle, ideal, self.K, self.allowed)
        self.assertEqual(got, expected)
        self.assertGreater(got["logit_max"], 0)
        self.assertGreater(got["max_NLL_difference"], 0)

    def test_repeated_guard_uses_complete_score_and_no_suffix(self):
        handle = self.session.bind_candidate(self.WN.clone(), 1)
        expected = self.controller.guard(handle)
        before = self.oracle.work.copy()
        self.assertEqual(self.controller.guard(handle), expected)
        self.assertEqual(self.oracle.work, before)
        self.assertEqual(self.controller.work["guard_reuses"], 1)

    def test_armijo_rejected_candidate_binding_does_not_trigger_current_work(self):
        before = self.oracle.work.copy()
        self.session.bind_candidate(self.WN+.01, dict(trial=1, reaches_guard=False))
        self.session.bind_candidate(self.WN+.005, dict(trial=2, reaches_guard=False))
        self.assertEqual(self.oracle.work, before)
        self.assertEqual(self.controller.work["candidate_observations"], 0)

    def test_new_trial_invalidates_old_candidate_and_no_cross_trial_hit(self):
        first = self.session.bind_candidate(self.WN.clone(), 1)
        self.controller.guard(first)
        old = self.controller._candidate
        second = self.session.bind_candidate(self.WN.clone(), 2)
        self.assertTrue(old.closed)
        self.controller.guard(second)
        self.assertEqual(self.controller.work["candidate_observations"], 2)
        self.assertEqual(self.oracle.work["cached_suffix_forwards"], 6)
        self.assertEqual(self.controller.work["native_observations"], 1)

    def test_invariant_before_guard_is_technical_failure(self):
        handle = self.session.bind_candidate(self.WN.clone(), 1)
        with self.assertRaisesRegex(ObservationError, "REQUIRES_SAME_PASSED_GUARD"):
            self.controller.invariant(handle, torch.zeros_like(self.WN).double(), self.K, self.allowed)
        self.assertTrue(self.session.closed)

    def test_suffix_failure_closes_session_and_partial_observations(self):
        handle = self.session.bind_candidate(self.WN.clone(), 1)
        self.oracle.fail_suffix = True
        with self.assertRaisesRegex(RuntimeError, "fixture suffix failure"):
            self.controller.guard(handle)
        self.assertTrue(self.session.closed)
        self.assertTrue(self.controller._native.closed)

    def test_numpy_pack_key_or_residual_mutation_fails_boundary(self):
        # A fresh fixture per subcase ensures each failure closes only its own
        # session, and shows that _version alone is not the source check.
        for name in ("input_ids", "attention_mask", "position_ids", "keys", "residual"):
            oracle = ToyOracle()
            policy = self.session.policy
            with EndpointSession("model", name, policy) as session:
                native = session.bind_native(self.WN)
                controller = CurrentObservationController(self.WN, oracle, self.rows,
                    session, native, teacher_identity="teacher", input_identity="input")
                cache = oracle.caches[0]
                value = cache.packed[name] if name in cache.packed else getattr(cache, name)
                version = value._version
                value.numpy().flat[0] += 1
                self.assertEqual(version, value._version)
                with self.assertRaisesRegex(ObservationError, "PACK_KEY_RESIDUAL_BYTES_CHANGED"):
                    _ = controller.anchor_rows
                self.assertTrue(session.closed)

    def test_geometry_failure_keeps_legacy_error_and_closes(self):
        handle = self.session.bind_candidate(self.WN.clone(), 1)
        self.assertTrue(self.controller.guard(handle)[0])
        with self.assertRaisesRegex(RuntimeError, "FP64_NULLSPACE_PROPOSAL_FAILURE"):
            self.controller.invariant(handle, torch.ones_like(self.WN).double(), self.K, self.allowed)
        self.assertTrue(self.session.closed)

    def test_nonlegacy_head_chunk_is_not_silently_accepted(self):
        oracle = ToyOracle()
        oracle.head_chunk_positions = 8
        with self.assertRaisesRegex(ValueError, "HEAD_CHUNK_16_REQUIRED"):
            CurrentObservationController(self.WN, oracle, self.rows, self.session,
                self.native, teacher_identity="teacher", input_identity="input")
        self.assertTrue(self.session.closed)

    def test_head_chunk_policy_cannot_change_after_native_anchor(self):
        handle = self.session.bind_candidate(self.WN.clone(), 1)
        self.oracle.head_chunk_positions = 8
        with self.assertRaisesRegex(ObservationError, "RUNTIME_POLICY_CHANGED"):
            self.controller.guard(handle)
        self.assertTrue(self.session.closed)

    def test_oracle_guard_failure_at_anchor_access_closes_session(self):
        def fail_guard():
            raise RuntimeError("fixture stale model guard")
        self.oracle._guard = fail_guard
        with self.assertRaisesRegex(RuntimeError, "stale model guard"):
            _ = self.controller.anchor_rows
        self.assertTrue(self.session.closed)

    def test_controller_close_does_not_claim_ownership_of_session(self):
        self.controller.close()
        self.assertTrue(self.controller._native.closed)
        self.assertFalse(self.session.closed)
        with self.session.readonly(self.native) as weight:
            self.assertEqual(tensor_sha256(weight), self.native.identity.weight_sha256)


if __name__ == "__main__":
    unittest.main()
