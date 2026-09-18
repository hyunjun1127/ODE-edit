"""B1 current-only guard/invariant reuse; no optimizer or physical-path edits.

The objective owner binds endpoints and calls ``guard`` ONLY after Armijo.
This controller builds one native observation, then one per candidate reaching
the guard. It shares hidden and score rows, not head GEMMs: guard uses unchanged
binding.score_rows target unions and invariant uses unchanged 16-position head
chunks. Geometry is recomputed through the original NumPy/SciPy route.

This module handles current inputs and explicitly empty Past (cold B1 only).
It does not implement nonempty-history guard or claim actual GPU/Llama parity.
All endpoint/input byte checks occur at phase boundaries. No original oracle
instance/class methods or selected model Parameters are monkeypatched.
"""
from contextlib import ExitStack
import json
from types import MethodType

import torch

from project.run_scripts.single_layer_edit_preserving_correction import geometry
from project.run_scripts.single_layer_edit_preserving_correction.binding import quality_ok, score_rows

from .endpoint_session import EndpointSessionError, identity_sha256, tensor_sha256
from .observation import CoverageSpec, ObservationBundle, ObservationError


class _ResidentOracleView:
    """Rebind Python oracle methods to a view overriding only ``_weight``.

    This follows the existing oracle's exact suffix implementation, including
    subclass methods, but lets its _weight call use an already borrowed owner.
    It is used only for inference hidden; physical methods are never invoked.
    FullWeightLlamaOracle uses Python methods and no isinstance/super reliance
    on self in this route. A different oracle architecture needs its own test.
    """
    def __init__(self, oracle, session, handle):
        self._oracle, self._session, self._handle = oracle, session, handle

    def __getattr__(self, name):
        value = getattr(self._oracle, name)
        if isinstance(value, MethodType) and value.__self__ is self._oracle:
            return MethodType(value.__func__, self)
        return value

    def _weight(self, weight):
        owned = self._session.evaluate_handle(self._handle)
        if weight is not owned:
            raise EndpointSessionError("CURRENT_RESIDENT_WEIGHT_OBJECT_REQUIRED")
        return owned


class _ScoreProducer:
    def __init__(self, controller, handle, bundle):
        self.controller, self.handle, self.bundle = controller, handle, bundle
        self.view = _ResidentOracleView(controller.oracle, controller.session, handle)
        self.seen = set()

    def logits_at(self, index, handle, positions):
        if handle is not self.handle or index in self.seen:
            raise ObservationError("CURRENT_SCORE_ENDPOINT_OR_DUPLICATE_SUFFIX")
        self.seen.add(index)
        weight = self.controller.session.evaluate_handle(handle)
        hidden = self.view.hidden(index, weight, route="cached")
        self.bundle.put_hidden(index, hidden)
        self.controller.work["current_suffix_forwards"] += 1
        # Head runs on the original suffix output, with the legacy union shape.
        return self.controller.oracle.logits_from_hidden(hidden, positions)


class CurrentObservationController:
    """Own native/candidate observations, but not the caller's endpoint session.

    API::

        current = CurrentObservationController(
            WN, oracle, rows, session, native_handle,
            teacher_identity=teacher_id, input_identity=input_id)
        # Only after full-reference candidate Armijo passes:
        ok, reasons = current.guard(candidate_handle)
        if ok:
            diagnostics = current.invariant(candidate_handle, ideal, K, allowed)

    Constructor performs native anchor scoring once. ``anchor_rows`` returns
    an owned scalar copy. A repeated guard on the same candidate reuses rows.
    Invariant before guard/on rejected/stale candidates is a technical error.
    Closing this object releases observations; the caller must still close the
    session before any physical observer/commit/next native.
    """

    def __init__(self, WN, oracle, rows, session, native_handle, *,
                 teacher_identity, input_identity):
        self.oracle, self.session, self.native_handle = oracle, session, native_handle
        self._closed = False
        self._native = self._candidate = None
        self._candidate_quality = None
        self.work = {name: 0 for name in (
            "native_observations", "candidate_observations", "score_rows_calls",
            "current_suffix_forwards", "guard_calls", "guard_reuses", "invariant_calls",
            "hidden_staging_calls", "hidden_staging_bytes", "input_byte_check_calls",
            "input_hash_bytes", "comparison_head_calls")}
        try:
            if native_handle.identity.kind != "native":
                raise ValueError("CURRENT_NATIVE_HANDLE_REQUIRED")
            if (WN.device.type != "cpu" or WN.dtype != torch.float32
                    or tensor_sha256(WN) != native_handle.identity.weight_sha256):
                raise ValueError("CURRENT_NATIVE_FP32_BYTES_MISMATCH")
            if oracle.head_chunk_positions != 16:
                raise ValueError("CURRENT_LEGACY_INVARIANT_HEAD_CHUNK_16_REQUIRED")
            if torch.device(oracle.device) != session.device or tuple(oracle.shape) != session.shape:
                raise ValueError("CURRENT_ORACLE_SESSION_DEVICE_SHAPE_MISMATCH")
            self.rows = json.loads(json.dumps(rows, allow_nan=False))
            if not self.rows or {r["cache"] for r in self.rows} != set(range(len(oracle.caches))):
                raise ValueError("CURRENT_NONEMPTY_ALL_CACHE_ROW_COVERAGE_REQUIRED")
            self.teacher_identity = json.loads(json.dumps(teacher_identity, allow_nan=False))
            self.input_identity = json.loads(json.dumps(input_identity, allow_nan=False))
            self.coverage = CoverageSpec.from_rows(
                self.rows,
                hidden_shapes={i: tuple(c.residual.shape) for i, c in enumerate(oracle.caches)},
                valid_positions={i: c.valid_positions().tolist() for i, c in enumerate(oracle.caches)})
            self._manifest = self._input_manifest()
            self._manifest_sha = identity_sha256(self._manifest)
            self._native = self._build(native_handle)
            self.work["native_observations"] += 1
        except BaseException:
            self.session.close()
            self.close()
            raise

    def _input_manifest(self):
        self.oracle._guard()
        manifest = []
        for index, cache in enumerate(self.oracle.caches):
            fields = dict(cache.packed, keys=cache.keys, residual=cache.residual)
            self.work["input_hash_bytes"] += sum(t.numel()*t.element_size() for t in fields.values())
            manifest.append(dict(index=index, tensors={name: tensor_sha256(t) for name, t in fields.items()}))
        self.work["input_byte_check_calls"] += 1
        return dict(caller_identity=self.input_identity, caches=manifest)

    def _check_inputs(self):
        try:
            if self._closed:
                raise ObservationError("CURRENT_CONTROLLER_CLOSED")
            if (self.oracle.head_chunk_positions != 16
                    or torch.device(self.oracle.device) != self.session.device
                    or tuple(self.oracle.shape) != self.session.shape):
                raise ObservationError("CURRENT_ORACLE_RUNTIME_POLICY_CHANGED")
            if identity_sha256(self._input_manifest()) != self._manifest_sha:
                raise ObservationError("CURRENT_PACK_KEY_RESIDUAL_BYTES_CHANGED")
        except BaseException:
            self.session.close()
            raise

    def _build(self, handle):
        self._check_inputs()
        bundle = ObservationBundle(self.session, handle, input_manifest=self._manifest,
                                   teacher_identity=self.teacher_identity, coverage=self.coverage)
        try:
            with self.session.readonly(handle):
                producer = _ScoreProducer(self, handle, bundle)
                rows = score_rows(producer, handle, self.rows)
                self.work["score_rows_calls"] += 1
                if producer.seen != set(range(len(self.oracle.caches))):
                    raise ObservationError("CURRENT_SUFFIX_COVERAGE_INCOMPLETE")
                bundle.put_rows("current", rows)
                bundle.put_rows("past", {})  # Cold B1: actual history N/A.
                self._check_inputs()
            bundle.seal()
            return bundle
        except BaseException:
            bundle.close()
            self.session.close()
            raise

    @property
    def anchor_rows(self):
        self._check_inputs()
        return self._native.lookup_rows("current", key=self._native.key)

    def guard(self, candidate_handle):
        try:
            self._check_inputs()
            if candidate_handle.identity.kind != "candidate":
                raise ValueError("CURRENT_CANDIDATE_HANDLE_REQUIRED")
            self.work["guard_calls"] += 1
            if self._candidate is None or self._candidate.handle is not candidate_handle:
                if self._candidate is not None:
                    self._candidate.close()
                self._candidate = self._build(candidate_handle)
                self.work["candidate_observations"] += 1
            else:
                self.work["guard_reuses"] += 1
            with self.session.readonly(self.native_handle), self.session.readonly(candidate_handle):
                anchor = self._native.lookup_rows("current", key=self._native.key)
                current = self._candidate.lookup_rows("current", key=self._candidate.key)
                result = quality_ok(current, anchor)
            self._candidate_quality = (candidate_handle.identity, result[0])
            return result
        except BaseException:
            self.session.close()
            raise

    @property
    def candidate_rows(self):
        self._check_inputs()
        if self._candidate is None:
            raise ObservationError("CURRENT_CANDIDATE_NOT_OBSERVED")
        with self.session.readonly(self._candidate.handle):
            return self._candidate.lookup_rows("current", key=self._candidate.key)

    def _compare_hidden(self, index, left, right):
        # Same loop, head call order, dtypes and scalar reductions as
        # FullWeightLlamaOracle.compare_logits. Only hidden production changes.
        stat = dict(max_abs=0., squared_error=0., squared_reference=0.,
                    logit_elements=0, valid_positions=0,
                    left_route="cached", right_route="cached")
        with torch.no_grad():
            for positions in self.oracle.caches[index].valid_positions().split(self.oracle.head_chunk_positions):
                l = self.oracle._head(left, positions)
                r = self.oracle._head(right, positions)
                self.work["comparison_head_calls"] += 2
                if not torch.isfinite(l).all() or not torch.isfinite(r).all():
                    raise FloatingPointError("NONFINITE_COMPARISON_LOGITS")
                diff = r.double()-l.double()
                stat["max_abs"] = max(stat["max_abs"], float(diff.abs().max()))
                stat["squared_error"] += float(diff.square().sum())
                stat["squared_reference"] += float(l.double().square().sum())
                stat["logit_elements"] += diff.numel()
                stat["valid_positions"] += positions.numel()
        stat["rms"] = (stat["squared_error"]/stat["logit_elements"])**.5
        stat["relative_l2"] = (stat["squared_error"]/max(stat["squared_reference"], 1e-300))**.5
        self.oracle.work["comparison_logit_elements"] += stat["logit_elements"]
        self.oracle._guard()
        return stat

    def invariant(self, candidate_handle, ideal, K, allowed):
        try:
            self._check_inputs()
            if (self._candidate is None or self._candidate.handle is not candidate_handle
                    or self._candidate_quality != (candidate_handle.identity, True)):
                raise ObservationError("CURRENT_INVARIANT_REQUIRES_SAME_PASSED_GUARD")
            self.work["invariant_calls"] += 1
            weight = self.session.cpu_snapshot(candidate_handle)
            WN = self.session.cpu_snapshot(self.native_handle)
            actual = weight.double()-WN.double()
            # No geometry-cache optimization: identical original numerical route.
            result = geometry.invariant_diagnostics(ideal, actual, WN, K, allowed)
            if not result['ideal_pass']:
                raise RuntimeError('FP64_NULLSPACE_PROPOSAL_FAILURE')
            with ExitStack() as stack:
                stack.enter_context(self.session.readonly(self.native_handle))
                stack.enter_context(self.session.readonly(candidate_handle))
                anchor = self._native.lookup_rows("current", key=self._native.key)
                current = self._candidate.lookup_rows("current", key=self._candidate.key)
                # Exact scalar assembly from legacy runtime.invariant.
                result['max_NLL_difference'] = max(abs(current[k]['nll']-anchor[k]['nll']) for k in anchor)
                strict = lambda x: {k for k, r in x.items() if r['branch'] == 'new' and r['strict']}
                pair = lambda x: {k for k, r in x.items() if r['kind'] == 'canonical' and r['branch'] == 'new'
                    and k.removesuffix('new')+'old' in x and r['nll'] < x[k.removesuffix('new')+'old']['nll']}
                result['strict_symmetric_difference'] = sorted(strict(current)^strict(anchor))
                result['pair_symmetric_difference'] = sorted(pair(current)^pair(anchor))
                stats = []
                for index in range(len(self.oracle.caches)):
                    left = self._native.lookup_hidden(index, key=self._native.key)
                    right = self._candidate.lookup_hidden(index, key=self._candidate.key)
                    self.work["hidden_staging_calls"] += 2
                    self.work["hidden_staging_bytes"] += (left.numel()+right.numel())*4
                    left, right = left.to(self.oracle.device), right.to(self.oracle.device)
                    stats.append(self._compare_hidden(index, left, right))
                result['logit_max'] = max(r['max_abs'] for r in stats)
                result['logit_rms'] = (sum(r['squared_error'] for r in stats)/sum(r['logit_elements'] for r in stats))**.5
                result['logit_elements'] = sum(r['logit_elements'] for r in stats)
                result['pass'] = bool(result['actual_response_pass'] and result['actual_leakage_pass'] and result['max_NLL_difference'] <= 1e-4
                    and not result['strict_symmetric_difference'] and not result['pair_symmetric_difference']
                    and result['logit_max'] <= 1e-3 and result['logit_rms'] <= 1e-4)
                self._check_inputs()
            return result
        except BaseException:
            self.session.close()
            raise

    def close(self):
        self._closed = True
        for bundle in (self._candidate, self._native):
            if bundle is not None:
                bundle.close()

    def __enter__(self):
        self._check_inputs()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
