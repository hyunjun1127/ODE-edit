"""Ragged full-bank KL using sealed W0 upstream caches and the old L4 suffix.

Both timed schedules use this adapter and identical unchunked [T_i,V] heads.
The only endpoint-session difference is bypassing repeated _weight validation
on an already checked resident owner/gradient leaf. Full reference sweeps never
accept indices/subsets. Bounded physical AD checks are a separate diagnostic,
with separate receipts/counters and no claim of full-bank model validation.

Instantiate at the sealed W0, retain one oracle across schedules, and explicitly
acknowledge_selected_write after a caller-owned selected-parameter installation.
CPU cache payload is bounded at construction (default18GiB); this is not a
claim that total process RSS/model/gradient/geometry fit without caller preflight.
"""
from contextlib import nullcontext
from copy import deepcopy
import time
from types import MethodType

import numpy as np
import torch

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import (
    FullTokenCache, FullWeightLlamaOracle, WEIGHT, signed_forward_kl, tensor_sha256,
)
from .endpoint_session import EndpointSessionError
from .generated_teacher import GeneratedTeacherStore, canonical_sha256


class FiniteTrialModelOverflow(FloatingPointError):
    """Incomplete finite-weight trial, pending caller's independent bank check.

    This is not optimizer.TrialNumericalOverflow: the caller must separately
    revalidate the entire fixed teacher bank before permitting backtracking.
    """
    def __init__(self, receipt):
        super().__init__("FINITE_TRIAL_MODEL_NONFINITE_FIXED_DOCUMENT_TEACHER_VERIFIED")
        self.receipt = deepcopy(receipt)


class _ResidentReferenceView:
    """Execute unchanged Python suffix methods on one validated borrowed leaf."""
    def __init__(self, oracle, session, handle, leaf):
        self._oracle, self._session, self._handle, self._leaf = oracle, session, handle, leaf

    def __getattr__(self, name):
        value = getattr(self._oracle, name)
        if isinstance(value, MethodType) and value.__self__ is self._oracle:
            return MethodType(value.__func__, self)
        return value

    def _weight(self, weight):
        self._session.validate_handle(self._handle, full=False)
        if weight is not self._leaf or not self._session.has_active_borrow(self._handle):
            raise EndpointSessionError("REFERENCE_RESIDENT_BORROWED_LEAF_REQUIRED")
        return weight


class GeneratedReferenceOracle(FullWeightLlamaOracle):
    """One resident CPU prefix bank, one document graph, independent sweeps.

    kl(...) returns (signed_document_mean, CPU_FP64_gradient_or_None, rows).
    last_sweep includes complete coverage and per-call work/session deltas.
    The CPU caller owns the59GiB process budget; resident_cache_bytes reports
    the exact tensor payload without including Python/mmap allocator overhead.
    """
    def __init__(self, model, store, *, max_cache_bytes=18 * 2**30):
        setup_started = time.perf_counter()
        if not isinstance(store, GeneratedTeacherStore) or not store.has_upstream_cache:
            raise ValueError("SEALED_GENERATED_TEACHER_AND_UPSTREAM_CACHE_REQUIRED")
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError("POSITIVE_CPU_PREFIX_CACHE_BUDGET_REQUIRED")
        selected = dict(model.named_parameters())[WEIGHT]
        if tensor_sha256(selected) != store._binding["w0_sha256"]:
            raise ValueError("INITIAL_MODEL_MUST_MATCH_SEALED_W0")
        if (tuple(selected.shape) != (store.hidden_size, store.key_size) or
                model.config.vocab_size != store.vocabulary_size):
            raise ValueError("MODEL_TEACHER_DIMENSIONS")
        self.store = store
        self._store_identity = canonical_sha256(store.receipt)
        self._capsules = tuple(store.capsule(i) for i in range(640))
        self._capsule_shas = tuple(canonical_sha256(c) for c in self._capsules)
        estimate = sum(len(c["tf_input_ids"]) * (4 * (store.key_size + store.hidden_size) + 3 * 8)
                       for c in self._capsules)
        if estimate > max_cache_bytes:
            raise MemoryError("CPU_PREFIX_CACHE_BUDGET_EXCEEDED")
        self._preparing_index = 0
        self._last_sweep = None
        packs = [dict(input_ids=torch.tensor([c["tf_input_ids"]], dtype=torch.long),
                      attention_mask=torch.tensor([c["attention_mask"]], dtype=torch.long),
                      position_ids=torch.tensor([c["position_ids"]], dtype=torch.long))
                 for c in self._capsules]
        # _prepare is overridden: no new model-prefix forward or generated data.
        super().__init__(model, packs, require_reference_length=None)
        self._prefix_digests = tuple(self._digest_cache(c) for c in self.caches)
        self.resident_cache_bytes = sum(t.numel() * t.element_size() for c in self.caches
                                       for t in (*c.packed.values(), c.keys, c.residual))
        if self.resident_cache_bytes != estimate:
            raise RuntimeError("CPU_PREFIX_PAYLOAD_ESTIMATE_MISMATCH")
        self.setup_receipt = dict(store=store.receipt, documents=640,
                                  resident_cache_bytes=self.resident_cache_bytes,
                                  cache_budget_bytes=max_cache_bytes, prefix_forwards=0,
                                  source="SEALED_W0_GENERATED_UPSTREAM_FILES",
                                  current_model_validation="NOT_ESTABLISHED_BY_CACHE_LOADING",
                                  setup_wall_seconds=time.perf_counter() - setup_started,
                                  work=deepcopy(self.work))
        self._initialize_extra_counters()

    def _initialize_extra_counters(self):
        for key in ("weight_validation_calls", "weight_to_device_calls", "weight_h2d_calls",
                    "weight_h2d_bytes", "teacher_h2d_calls", "teacher_h2d_bytes",
                    "gradient_d2h_calls", "gradient_d2h_bytes", "prefix_byte_checks",
                    "prefix_hash_bytes", "head_calls", "autograd_calls", "live_document_graphs",
                    "peak_live_document_graphs", "technical_documents", "technical_checks",
                    "finite_trial_model_overflows"):
            self.work.setdefault(key, 0)
        for key in ("weight_prepare_host_seconds", "teacher_transfer_seconds", "prefix_hash_seconds", "head_loss_seconds"):
            self.work.setdefault(key, 0.0)

    def _prepare(self, source):
        index = self._preparing_index
        self._preparing_index += 1
        packed = self._pack(source)
        started = time.perf_counter()
        with self.store.document(index) as document:
            if canonical_sha256(document.capsule) != self._capsule_shas[index]:
                raise RuntimeError("PREFIX_CAPSULE_IDENTITY_CHANGED")
            keys = torch.from_numpy(np.array(document.keys, copy=True)).unsqueeze(0)
            residual = torch.from_numpy(np.array(document.residual, copy=True)).unsqueeze(0)
        self.work.setdefault("retained_prefix_load_seconds", 0.0)
        self.work["retained_prefix_load_seconds"] += time.perf_counter() - started
        self.work.setdefault("retained_prefix_documents_loaded", 0)
        self.work["retained_prefix_documents_loaded"] += 1
        identity = {name: tensor_sha256(value) for name, value in packed.items()}
        identity.update(keys_sha256=tensor_sha256(keys), residual_sha256=tensor_sha256(residual),
                        capsule_sha256=self._capsule_shas[index], store_identity=self._store_identity,
                        valid_tokens=packed["input_ids"].numel(), padded_tokens=packed["input_ids"].numel())
        return FullTokenCache(packed, keys, residual, identity)

    @staticmethod
    def _digest_cache(cache):
        return tuple((name, tensor_sha256(value)) for name, value in
                     (*cache.packed.items(), ("keys", cache.keys), ("residual", cache.residual)))

    def _check_prefix_bytes(self):
        self._guard()
        started = time.perf_counter()
        if canonical_sha256(self.store.receipt) != self._store_identity:
            raise RuntimeError("TEACHER_STORE_IDENTITY_CHANGED")
        if tuple(self._digest_cache(c) for c in self.caches) != self._prefix_digests:
            raise RuntimeError("CPU_PREFIX_CACHE_BYTES_CHANGED")
        if tuple(canonical_sha256(c) for c in self._capsules) != self._capsule_shas:
            raise RuntimeError("GENERATED_CAPSULE_METADATA_CHANGED")
        self.work["prefix_byte_checks"] += 1
        self.work["prefix_hash_bytes"] += self.resident_cache_bytes
        self.work["prefix_hash_seconds"] += time.perf_counter() - started

    def _weight(self, weight):
        self.work["weight_validation_calls"] += 1
        self.work["weight_to_device_calls"] += 1
        if weight.device.type == "cpu" and self.device.type == "cuda":
            self.work["weight_h2d_calls"] += 1
            self.work["weight_h2d_bytes"] += weight.numel() * weight.element_size()
        started = time.perf_counter()
        result = FullWeightLlamaOracle._weight(self, weight)
        # No additional synchronization in the legacy-only route. This host
        # interval includes old finite checking and transfer-call latency; it
        # is not a separately measured CUDA-transfer duration.
        self.work["weight_prepare_host_seconds"] += time.perf_counter() - started
        return result

    def _teacher(self, index):
        started = time.perf_counter()
        with self.store.document(index) as document:
            if canonical_sha256(document.capsule) != self._capsule_shas[index]:
                raise RuntimeError("TEACHER_CAPSULE_IDENTITY_CHANGED")
            copied = torch.from_numpy(np.array(document.logp, copy=True))
        self.work["teacher_read_seconds"] += time.perf_counter() - started
        self.work["teacher_documents_read"] += 1
        self.work["teacher_bytes_read"] += copied.numel() * copied.element_size()
        self._sync()
        started = time.perf_counter()
        teacher = copied.to(self.device)
        self._sync()
        self.work["teacher_transfer_seconds"] += time.perf_counter() - started
        if self.device.type == "cuda":
            self.work["teacher_h2d_calls"] += 1
            self.work["teacher_h2d_bytes"] += copied.numel() * copied.element_size()
        return teacher

    @property
    def last_sweep(self):
        return deepcopy(self._last_sweep)

    def reset_counters(self):
        """Preserves caches; return prior costs including the reset's hash checks."""
        if self.work["live_document_graphs"]:
            raise RuntimeError("CANNOT_RESET_ACTIVE_REFERENCE_GRAPH")
        self._check_prefix_bytes()
        prior = deepcopy(self.work)
        for key, value in self.work.items():
            self.work[key] = 0.0 if isinstance(value, float) else 0
        self.events.clear()
        self._last_sweep = None
        return prior

    @staticmethod
    def _delta(before, after):
        return {key: value - before.get(key, 0) for key, value in after.items()}

    def _evaluate(self, weight, selected, *, gradient, route, coverage=None,
                  session=None, handle=None, allow_trial_overflow=False):
        before, session_before = deepcopy(self.work), deepcopy(session.work) if session else {}
        started = time.perf_counter()
        self._check_prefix_bytes()
        if (session is None) != (handle is None):
            raise ValueError("REFERENCE_SESSION_AND_HANDLE_REQUIRED_TOGETHER")
        if session is not None:
            if route != "cached" or session.shape != self.shape or session.device != self.device:
                raise ValueError("REFERENCE_SESSION_ROUTE_DEVICE_OR_SHAPE")
            if weight.dtype != torch.float32 or tensor_sha256(weight) != handle.identity.weight_sha256:
                session.close()
                raise EndpointSessionError("REFERENCE_CALLER_WEIGHT_HANDLE_BYTES_MISMATCH")
        accumulation = torch.zeros(self.shape, dtype=torch.float64, device="cpu") if gradient else None
        rows = []
        overflow = None
        if session is not None:
            context = session.gradient_leaf(handle) if gradient else session.readonly(handle)
        else:
            prepared = self._weight(weight)
            context = (self.physical_weight(prepared, gradient=gradient) if route == "physical" else
                       nullcontext(prepared.detach().requires_grad_(gradient)))
        with context as leaf, torch.set_grad_enabled(gradient):
            suffix = _ResidentReferenceView(self, session, handle, leaf) if session else self
            for ordinal, index in enumerate(selected):
                cap = self._capsules[index]
                teacher = self._teacher(index)
                try:
                    hidden = (suffix.suffix_hidden(index, leaf) if route == "cached" else self._physical_hidden(index))
                    if gradient:
                        self.work["live_document_graphs"] += 1
                        self.work["peak_live_document_graphs"] = max(self.work["peak_live_document_graphs"],
                                                                    self.work["live_document_graphs"])
                    self._sync()
                    head_started = time.perf_counter()
                    try:
                        logits = self._head(hidden, torch.tensor(cap["score_positions"], dtype=torch.long))
                        self.work["head_calls"] += 1
                        try:
                            loss = signed_forward_kl(logits, teacher)
                        except FloatingPointError as exc:
                            # Do not convert arbitrary arithmetic errors/OOM or
                            # a fixed-teacher error into a usable trial signal.
                            if not allow_trial_overflow or str(exc) != "NONFINITE_LOGITS_OR_TEACHER":
                                raise
                            if not bool(torch.isfinite(teacher).all()):
                                raise FloatingPointError("NONFINITE_FIXED_TEACHER") from exc
                            if bool(torch.isfinite(logits).all()):
                                raise
                            overflow = "MODEL_LOGITS_NONFINITE"
                        else:
                            if not bool(torch.isfinite(loss)):
                                if not allow_trial_overflow:
                                    raise FloatingPointError("NONFINITE_GENERATED_REFERENCE_KL")
                                if not bool(torch.isfinite(teacher).all()):
                                    raise FloatingPointError("NONFINITE_FIXED_TEACHER")
                                overflow = "MODEL_LOSS_NONFINITE"
                    finally:
                        self._sync()
                        self.work["head_loss_seconds"] += time.perf_counter() - head_started
                    if overflow is not None:
                        overflow = dict(reason=overflow, index=index, ordinal=cap["ordinal"], role=cap["role"],
                                        source_row_id=cap["source_row_id"], scored_positions=cap["actual_length"],
                                        input_tokens=len(cap["tf_input_ids"]),
                                        fixed_document_teacher_finite=True)
                        # Normal exit through the borrowed endpoint context is
                        # essential: SHA/state verification still runs and the
                        # healthy session is available for the next trial.
                        break
                    if coverage is not None:
                        coverage.record_forward(ordinal, cap["score_positions"], vocabulary_size=self.store.vocabulary_size)
                    if gradient:
                        self._sync()
                        backward_started = time.perf_counter()
                        grad, = torch.autograd.grad(loss, leaf)
                        self.work["autograd_calls"] += 1
                        if grad.dtype != torch.float32 or not bool(torch.isfinite(grad).all()):
                            raise FloatingPointError("NONFINITE_OR_NONFP32_REFERENCE_GRADIENT")
                        # Exact old EN route: document FP32 -> FP64 -> CPU, add
                        # in document order, then one final division by512.
                        accumulation.add_(grad.detach().double().cpu())
                        if self.device.type == "cuda":
                            self.work["gradient_d2h_calls"] += 1
                            self.work["gradient_d2h_bytes"] += grad.numel() * 8
                        self._sync()
                        self.work["backward_seconds"] += time.perf_counter() - backward_started
                        self.work[route + "_backward_documents"] += 1
                        if coverage is not None:
                            coverage.record_backward(ordinal, cap["score_positions"], vocabulary_size=self.store.vocabulary_size)
                        del grad
                    rows.append(dict(index=index, role=cap["role"], ordinal=cap["ordinal"], source_row_id=cap["source_row_id"],
                                     loss=float(loss.detach()), input_tokens=len(cap["tf_input_ids"]),
                                     valid_tokens=len(cap["tf_input_ids"]), scored_positions=cap["actual_length"]))
                    self.work["kl_documents"] += 1
                    self.work["kl_scored_positions"] += cap["actual_length"]
                    self.work["kl_input_tokens"] += len(cap["tf_input_ids"])
                finally:
                    if gradient:
                        self.work["live_document_graphs"] = 0
                    # No previous document graph/logits survive the next loop.
                    teacher = hidden = logits = loss = None
        if accumulation is not None:
            accumulation.div_(len(selected))
            if not bool(torch.isfinite(accumulation).all()):
                raise FloatingPointError("NONFINITE_REFERENCE_GRADIENT_ACCUMULATION")
        self._check_prefix_bytes()
        if overflow is not None:
            if not bool(torch.isfinite(weight).all()):
                raise FloatingPointError("NONFINITE_CANDIDATE_WEIGHT")
            self.work["finite_trial_model_overflows"] += 1
            receipt = dict(scope="INTERRUPTED_FULL_BANK_TRIAL", complete=False, route=route,
                           gradient=False, role=coverage.role, documents=len(rows),
                           positions=sum(row["scored_positions"] for row in rows),
                           attempted_documents=len(rows) + 1,
                           attempted_positions=sum(row["scored_positions"] for row in rows) + overflow["scored_positions"],
                           vocabulary_size=self.store.vocabulary_size, failed_document=overflow,
                           partial_rows=deepcopy(rows), expected_documents=len(selected),
                           expected_positions=sum(self._capsules[i]["actual_length"] for i in selected),
                           teacher_sha256=self.store.receipt["manifest_sha256"],
                           parent_full_teacher_recheck_required=True,
                           entire_fixed_teacher_bank_rechecked=False,
                           finite_candidate_rechecked=True,
                           endpoint_borrow_exit_validated=session is not None,
                           prefix_bytes_exit_validated=True,
                           coverage=dict(data_id=self._capsules[selected[0]]["data_id"], role=coverage.role,
                                         documents=len(rows), positions=sum(r["scored_positions"] for r in rows),
                                         backward_documents=0, vocabulary_size=self.store.vocabulary_size, complete=False),
                           work=self._delta(before, self.work),
                           session_work=self._delta(session_before, session.work) if session else {},
                           seconds=time.perf_counter() - started, actual_Llama_pass_assigned=False)
            self._last_sweep = receipt
            self.events.append(dict(event="finite_trial_model_overflow", complete_coverage=False,
                                    documents=len(rows), failed_index=overflow["index"]))
            # Raised outside readonly/gradient_leaf, after all normal endpoint
            # and prefix checks. No partial mean or completed sweep is emitted.
            raise FiniteTrialModelOverflow(receipt)
        coverage_receipt = coverage.finish() if coverage is not None else None
        if coverage is not None:
            self.work["kl_sweeps"] += 1
            self.work["gradient_sweeps"] += int(gradient)
        receipt = dict(scope="FULL_BANK_SWEEP" if coverage else "BOUNDED_TECHNICAL_CHECK",
                       complete=True, route=route, gradient=gradient, documents=len(rows),
                       positions=sum(row["scored_positions"] for row in rows),
                       vocabulary_size=self.store.vocabulary_size, teacher_sha256=self.store.receipt["manifest_sha256"],
                       dtypes=dict(logits="float32", log_softmax="float32", loss="float64",
                                   per_document_gradient="float32" if gradient else None,
                                   gradient_accumulation="CPU_float64_document_order" if gradient else None),
                       work=self._delta(before, self.work), session_work=self._delta(session_before, session.work) if session else {},
                       seconds=time.perf_counter() - started, head_position_chunking=False,
                       weight_timing="HOST_CALL_LATENCY_NO_ADDED_CUDA_SYNCHRONIZATION",
                       cached_prefix_forward_calls=0, actual_Llama_pass_assigned=False)
        if coverage is not None:
            receipt["coverage"] = coverage_receipt
        return sum(row["loss"] for row in rows) / len(rows), accumulation, rows, receipt

    def kl(self, weight, *, gradient=False, role="R512", session=None, handle=None, route="cached",
           allow_trial_overflow=False):
        """Full bank; explicit finite-trial opt-in is never for base/accepted/AD.

        The caller must leave allow_trial_overflow=False for native/accepted
        evaluations. Only R512 non-gradient trials may opt in, and a caught
        FiniteTrialModelOverflow still requires independent full-bank teacher
        revalidation before an optimizer backtrack is justified.
        """
        self._last_sweep = None
        if type(allow_trial_overflow) is not bool or (allow_trial_overflow and (gradient or role != "R512")):
            raise ValueError("TRIAL_OVERFLOW_ONLY_FOR_NONGRADIENT_REFERENCE_TRIALS")
        if route != "cached":
            raise ValueError("USE_CHECK_DOCUMENTS_FOR_BOUNDED_PHYSICAL_DIAGNOSTICS")
        if role == "Dev128" and gradient:
            raise ValueError("DEV_GRADIENT_FORBIDDEN")
        selected = self.store.indices(role)
        coverage = self.store.coverage(role, require_backward=gradient)
        result = self._evaluate(weight, selected, gradient=gradient, route=route,
                                coverage=coverage, session=session, handle=handle,
                                allow_trial_overflow=allow_trial_overflow)
        self._last_sweep = result[3]
        self.events.append(dict(event="generated_full_bank_KL", role=role, documents=len(selected),
                                gradient=gradient, mean=result[0], complete_coverage=True))
        return result[:3]

    def check_documents(self, weight, indices, *, gradient=True):
        """Independent cached/physical AD on1..16 explicit indices, not512PASS.

        Returns full detached gradients for independent caller analysis plus
        exactness and relative-error summaries; no tolerance is auto-assigned.
        Physical selected Parameter is restored through the original oracle's
        copy/restore context, including exceptions. No endpoint session used.
        """
        indices = tuple(indices)
        if (not 1 <= len(indices) <= 16 or len(set(indices)) != len(indices) or
                any(type(i) is not int or not 0 <= i < 640 for i in indices)):
            raise ValueError("BOUNDED_UNIQUE_EXPLICIT_DOCUMENT_INDICES_REQUIRED")
        if gradient and any(i >= 512 for i in indices):
            raise ValueError("DEV_GRADIENT_FORBIDDEN")
        self._last_sweep = None
        cached = self._evaluate(weight, indices, gradient=gradient, route="cached")
        physical = self._evaluate(weight, indices, gradient=gradient, route="physical")
        self.work["technical_checks"] += 1
        self.work["technical_documents"] += len(indices) * 2
        result = dict(scope="BOUNDED_REFERENCE_PHYSICAL_AD", indices=list(indices),
                      full_bank_sweep=False, full_bank_validation_pass=False,
                      cached=dict(mean=cached[0], gradient=cached[1], rows=cached[2], receipt=cached[3]),
                      physical=dict(mean=physical[0], gradient=physical[1], rows=physical[2], receipt=physical[3]),
                      scalar_exact=cached[0] == physical[0], rows_exact=cached[2] == physical[2],
                      loss_absolute_difference=abs(cached[0] - physical[0]))
        if gradient:
            diff = cached[1] - physical[1]
            result.update(gradient_exact=torch.equal(cached[1], physical[1]),
                          cached_gradient_norm=float(cached[1].norm()),
                          physical_gradient_norm=float(physical[1].norm()),
                          gradient_max_abs=float(diff.abs().max()),
                          gradient_relative_l2=float(diff.norm() / physical[1].norm().clamp_min(1e-300)))
        return result
