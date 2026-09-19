"""Full-bank choice-risk oracle and factor reuse for mechanism-first v1.

The reference oracle is a read-only source of sealed W0 input keys/residuals and
token capsules.  This module never reads its full-distribution teacher.  A
native scan takes one graph/backward per document, retaining only the gradient
with respect to every valid L4 output token.  Dense document weight gradients
are transient on the device; their FP64 accumulator crosses to CPU once.

Production scan has no subset argument.  ``check_pair_document`` is an
explicit, separately counted technical diagnostic and cannot establish full
512 coverage.  All scalar tolerances below are the published v1 contract.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import tensor_sha256


WORST_DEFICIT_TOLERANCE = 1e-4
RISK_ABSOLUTE_TOLERANCE = 1e-10
RISK_RELATIVE_TOLERANCE = 1e-6
PROJECTED_GRADIENT_RELATIVE_ZERO = 1e-12


class DecisionError(RuntimeError):
    """Technical identity/numeric failure, never a usable native fallback."""


def _require(value, message):
    if not value:
        raise DecisionError(message)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _guard_tensor(t):
    return (t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))


class EndpointBinding:
    """Owned immutable CPU endpoint plus one H2D copy, no external tensor alias.

    Initial finite/hash checks bind bytes once.  Later guards detect mutations
    to the owned tensors without repeating full-byte device readback.  The
    owner must not expose/modify ``cpu`` or ``device_weight``; tensor guards are
    not claimed to detect arbitrary external native-code memory corruption.
    """
    def __init__(self, weight, device, *, endpoint_id, source_identity):
        _require(isinstance(endpoint_id, str) and bool(endpoint_id), "ENDPOINT_ID_REQUIRED")
        _require(weight.ndim == 2 and weight.dtype == torch.float32, "ENDPOINT_FP32_MATRIX")
        _require(bool(torch.isfinite(weight).all()), "NONFINITE_ENDPOINT")
        self.endpoint_id = endpoint_id
        self.source_identity = deepcopy(source_identity)
        self.cpu = weight.detach().cpu().contiguous().clone()
        self.weight_sha256 = tensor_sha256(self.cpu)
        self.identity = _json_sha(dict(endpoint_id=endpoint_id, weight_sha256=self.weight_sha256,
                                       source_identity=self.source_identity))
        self.device = torch.device(device)
        self.device_weight = self.cpu.to(self.device, copy=True)
        self.shape = tuple(weight.shape)
        self._cpu_guard = _guard_tensor(self.cpu)
        self._device_guard = _guard_tensor(self.device_weight)
        self._identity_guard = _json_sha(self.source_identity)
        self.closed = False
        self.work = dict(endpoint_H2D_calls=int(self.device.type == "cuda"),
                         endpoint_H2D_bytes=self.cpu.numel() * self.cpu.element_size()
                         if self.device.type == "cuda" else 0,
                         immutable_endpoint_hash_calls=1)

    def guard(self):
        _require(not self.closed, "ENDPOINT_CLOSED")
        _require(_guard_tensor(self.cpu) == self._cpu_guard and
                 _guard_tensor(self.device_weight) == self._device_guard,
                 "OWNED_ENDPOINT_MUTATED")
        _require(_json_sha(self.source_identity) == self._identity_guard, "ENDPOINT_IDENTITY_MUTATED")
        _require(not self.device_weight.requires_grad, "ENDPOINT_GRAPH_LEAF_FORBIDDEN")

    def close(self):
        self.guard()
        self.closed = True
        self.device_weight = None


class FactorArchive:
    """Create-once CPU A factors, one document resident at a time.

    Keys remain in the already sealed prefix bank, not duplicated here.
    ``identity`` must contain the exact endpoint and input-bank identities.
    Partial files/archives are preserved and cannot be used as complete scans.
    """
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.identity = deepcopy(identity)
        self.identity_sha256 = _json_sha(self.identity)
        self.members = []
        self.sealed = False

    def put(self, index, activation_gradient, metadata):
        _require(not self.sealed and index == len(self.members), "FACTOR_APPEND_ORDER")
        a = activation_gradient.detach().cpu().contiguous()
        _require(a.ndim == 2 and a.dtype == torch.float32 and bool(torch.isfinite(a).all()),
                 "FACTOR_FINITE_FP32_VALID_TOKEN_MATRIX")
        path = self.root / f"factor-{index:04d}.pt"
        temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        payload = dict(A=a, metadata=deepcopy(metadata), archive_identity=self.identity_sha256)
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)  # Only this just-published temporary hardlink.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        item = dict(index=index, path=str(path), bytes=path.stat().st_size,
                    shape=list(a.shape), A_sha256=tensor_sha256(a), metadata=deepcopy(metadata))
        self.members.append(item)
        return deepcopy(item)

    def seal(self, *, expected_count):
        _require(not self.sealed and len(self.members) == expected_count, "FACTOR_COMPLETE_COUNT")
        self.sealed = True
        return dict(identity=deepcopy(self.identity), identity_sha256=self.identity_sha256,
                    count=len(self.members), bytes=sum(x["bytes"] for x in self.members),
                    members=deepcopy(self.members), complete=True)

    def get(self, index):
        _require(self.sealed, "PARTIAL_FACTOR_ARCHIVE")
        item = self.members[index]
        payload = torch.load(item["path"], map_location="cpu", weights_only=True, mmap=True)
        _require(payload["archive_identity"] == self.identity_sha256 and
                 payload["metadata"] == item["metadata"], "FACTOR_ENDPOINT_OR_INPUT_IDENTITY")
        a = payload["A"]
        _require(list(a.shape) == item["shape"] and a.dtype == torch.float32 and
                 bool(torch.isfinite(a).all()) and tensor_sha256(a) == item["A_sha256"],
                 "FACTOR_BYTES_CHANGED")
        return a, deepcopy(item["metadata"])


def margins_from_logits(logits, labels):
    """Exact lowest-ID argmax; alternatives exclude the declared W0 label."""
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    _require(logits.ndim == 2 and logits.dtype == torch.float32 and
             logits.shape[0] == labels.numel() and logits.shape[1] > 1,
             "FULL_VOCABULARY_LOGIT_LABEL_SHAPE")
    _require(bool(torch.isfinite(logits).all()), "NONFINITE_DECISION_LOGITS")
    _require(bool(((labels >= 0) & (labels < logits.shape[1])).all()), "LABEL_VOCABULARY")
    chosen = logits.gather(1, labels[:, None]).squeeze(1)
    alternative = logits.detach().clone()
    alternative.scatter_(1, labels[:, None], -torch.inf)
    values, competitors = alternative.max(-1)
    predictions = logits.detach().argmax(-1)
    return dict(margins=(chosen - values).detach().double().cpu().tolist(),
                competitors=competitors.cpu().tolist(), predictions=predictions.cpu().tolist(),
                correct=predictions.eq(labels).cpu().tolist())


def risk_from_margins(margins):
    _require(bool(margins) and all(math.isfinite(float(m)) for m in margins), "FINITE_NONEMPTY_MARGINS")
    try:
        result = math.fsum(max(0.0, -float(m)) ** 2 for m in margins) / len(margins)
    except OverflowError as exc:
        raise DecisionError("NONFINITE_DECISION_RISK") from exc
    _require(math.isfinite(result), "NONFINITE_DECISION_RISK")
    return result


def risk_tolerance(phi_reference, phi_history=0.0):
    _require(min(phi_reference, phi_history) >= 0 and
             math.isfinite(phi_reference + phi_history), "FINITE_NONNEGATIVE_RISK")
    return RISK_ABSOLUTE_TOLERANCE + RISK_RELATIVE_TOLERANCE * max(
        phi_reference + phi_history, phi_reference)


def no_direction_reason(phi_reference, phi_history, mismatches, raw_norm=None, projected_norm=None,
                        *, required_guards_valid=True):
    """Pure numerical classification, not an efficacy/capacity conclusion."""
    psi = phi_reference + phi_history
    tau = risk_tolerance(phi_reference, phi_history)
    if psi == 0:
        if mismatches:
            return "TIE_ONLY_NO_DIRECTION"
        return "ZERO_RISK_NO_OP" if required_guards_valid else "ZERO_RISK_GUARD_FAILURE"
    if psi <= tau:
        return "BELOW_RISK_RESOLUTION"
    if raw_norm is not None:
        _require(math.isfinite(raw_norm) and raw_norm >= 0 and projected_norm is not None and
                 math.isfinite(projected_norm) and projected_norm >= 0, "FINITE_GRADIENT_NORMS")
        if raw_norm == 0 or projected_norm <= PROJECTED_GRADIENT_RELATIVE_ZERO * raw_norm:
            return "NO_PROJECTED_DIRECTION"
    return None


def suffix_from_output(oracle, index, module_output):
    """Pinned alltoken operation order, with an explicit full-token L4 leaf."""
    cache = oracle.caches[index]
    packed = oracle._on_device(cache.packed)
    hidden = cache.residual.to(oracle.device) + module_output
    args = oracle._args(hidden, packed)
    for layer in oracle.decoder.layers[5:]:
        hidden = layer(hidden, **args)[0]
    return oracle.decoder.norm(hidden)


@dataclass
class DecisionObservation:
    endpoint_identity: str
    input_identity: str
    role: str
    rows: list
    phi_reference: float
    mismatches: int
    gradient: torch.Tensor | None
    coverage: dict
    work: dict
    factors: object | None = None

    def compact(self):
        return dict(endpoint_identity=self.endpoint_identity, input_identity=self.input_identity,
                    role=self.role, phi_reference=self.phi_reference, mismatches=self.mismatches,
                    coverage=deepcopy(self.coverage), work=deepcopy(self.work),
                    gradient_norm=None if self.gradient is None else float(self.gradient.norm()),
                    rows=deepcopy(self.rows))


class DecisionOracle:
    """Consume GeneratedReferenceOracle caches; never call its KL/teacher API."""
    def __init__(self, reference_oracle, *, head_chunk_positions=16):
        self.oracle = reference_oracle
        self.device = reference_oracle.device
        self.shape = tuple(reference_oracle.shape)
        _require(head_chunk_positions == 16, "FIXED_FULL_VOCABULARY_HEAD_CHUNK_16")
        self.chunk = head_chunk_positions
        self.capsules = tuple(deepcopy(c) for c in reference_oracle._capsules)
        _require(len(self.capsules) == len(reference_oracle.caches) == 640, "COMPLETE_640_CAPSULE_BANK")
        for index, cap in enumerate(self.capsules):
            role, ordinal = ("R512", index) if index < 512 else ("Dev128", index - 512)
            _require(cap["role"] == role and cap["ordinal"] == ordinal and
                     len(cap["y0"]) == cap["actual_length"] == len(cap["score_positions"]),
                     "CAPSULE_ORDER_ROLE_OR_POSITION_COUNT")
            _require(cap["score_positions"] == list(range(128, 128 + cap["actual_length"])) and
                     len(cap["tf_input_ids"]) == 128 + cap["actual_length"], "CAPSULE_SHIFT")
        self.input_identity = _json_sha(dict(capsules=self.capsules,
            store_manifest_sha256=reference_oracle.store.receipt["manifest_sha256"]))
        self._cache_guard = reference_oracle._cache_guard()
        self._center_identities = set()
        self.work = dict(scans=0, technical_checks=0, full_reference_backward_sweeps=0,
                         reference_teacher_reads=0, gradient_D2H_calls=0)

    def _guard(self, endpoint):
        endpoint.guard()
        _require(endpoint.shape == self.shape and endpoint.device == self.device,
                 "ENDPOINT_ORACLE_SHAPE_DEVICE")
        self.oracle._guard()
        _require(self.oracle._cache_guard() == self._cache_guard, "REFERENCE_PREFIX_CACHE_MUTATION")

    def _document(self, endpoint, index, *, derivative=False):
        cap, cache = self.capsules[index], self.oracle.caches[index]
        self.oracle._sync()
        clock = time.perf_counter()
        valid = cache.packed["attention_mask"][0].bool().to(self.device)
        keys = cache.keys.to(self.device)
        module = F.linear(keys, endpoint.device_weight).detach().requires_grad_(derivative)
        self.oracle._sync()
        timings = dict(key_transfer_and_L4_linear_seconds=time.perf_counter() - clock,
                       suffix_seconds=0., full_vocabulary_head_seconds=0.,
                       pair_two_row_head_seconds=0., activation_backward_seconds=0.)
        with torch.set_grad_enabled(derivative):
            clock = time.perf_counter()
            hidden = suffix_from_output(self.oracle, index, module)
            self.oracle._sync()
            timings["suffix_seconds"] = time.perf_counter() - clock
            chunks = []
            clock = time.perf_counter()
            with torch.no_grad():
                for begin in range(0, cap["actual_length"], self.chunk):
                    pos = cap["score_positions"][begin:begin + self.chunk]
                    labels = cap["y0"][begin:begin + self.chunk]
                    logits = self.oracle._head(hidden, torch.tensor(pos, dtype=torch.long))
                    chunks.append(margins_from_logits(logits, labels))
                    del logits
            self.oracle._sync()
            timings["full_vocabulary_head_seconds"] = time.perf_counter() - clock
            data = {key: sum((part[key] for part in chunks), [])
                    for key in ("margins", "competitors", "predictions", "correct")}
            worst = min(range(len(data["margins"])), key=lambda j: (data["margins"][j], j))
            position = cap["score_positions"][worst]
            label, competitor = cap["y0"][worst], data["competitors"][worst]
            a = None
            pair_scalar = None
            if derivative:
                # Full-vocab scan chooses the fixed exposed pair.  Two output
                # rows suffice for its scalar backward, not competitor search.
                head = self.oracle.model.lm_head
                clock = time.perf_counter()
                pair = F.linear(hidden[0, position:position + 1], head.weight[[label, competitor]],
                                None if head.bias is None else head.bias[[label, competitor]])
                scalar = pair[0, 0] - pair[0, 1]
                _require(bool(torch.isfinite(scalar)), "NONFINITE_EXPOSED_PAIR")
                self.oracle._sync()
                timings["pair_two_row_head_seconds"] = time.perf_counter() - clock
                clock = time.perf_counter()
                a, = torch.autograd.grad(scalar, module)
                _require(a.dtype == torch.float32 and bool(torch.isfinite(a).all()),
                         "NONFINITE_ACTIVATION_MARGIN_GRADIENT")
                a = a[0, valid].detach()
                pair_scalar = float(scalar.detach())
                self.oracle._sync()
                timings["activation_backward_seconds"] = time.perf_counter() - clock
            row = dict(index=index, ordinal=cap["ordinal"], role=cap["role"],
                       source_row_id=cap["source_row_id"], capsule_sha256=_json_sha(cap),
                       positions=list(cap["score_positions"]), labels=list(cap["y0"]), **data,
                       mu=data["margins"][worst], worst_offset=worst, worst_position=position,
                       worst_label=label, worst_competitor=competitor,
                       exposed_pair_backward_scalar=pair_scalar,
                       phase_seconds=timings,
                       input_tokens=int(valid.sum()), scored_positions=cap["actual_length"],
                       mismatches=sum(not c for c in data["correct"]))
        return row, a, keys[0, valid]

    def scan(self, endpoint, *, gradient=False, factor_sink=None, role="R512", selection_seal=None):
        """Complete512 or observer128 only; gradient always all512 once/center."""
        self._guard(endpoint)
        _require(role in ("R512", "Dev128"), "REFERENCE_OR_OBSERVER_ROLE")
        _require(not gradient or role == "R512", "DEV_GRADIENT_FORBIDDEN")
        _require(role != "Dev128" or (isinstance(selection_seal, str) and bool(selection_seal)),
                 "DEV_REQUIRES_POSTSELECTION_SEAL")
        _require(not gradient or factor_sink is not None, "ALL_FACTORS_MUST_BE_RETAINED")
        key = (endpoint.identity, self.input_identity)
        _require(not gradient or key not in self._center_identities, "SECOND_CENTER_BACKWARD_SWEEP_FORBIDDEN")
        if gradient:
            self._center_identities.add(key)  # A failed sweep cannot silently restart here.
        selected = range(512) if role == "R512" else range(512, 640)
        started = time.perf_counter()
        total = torch.zeros(self.shape, dtype=torch.float64, device=self.device) if gradient else None
        rows, factor_members = [], []
        work = dict(documents=0, input_tokens=0, scored_positions=0, suffix_forwards=0,
                    full_vocab_head_rows=0, full_vocab_head_calls=0, pair_two_row_head_calls=0,
                    backward_documents=0, factor_D2H_bytes=0, dense_gradient_D2H_bytes=0,
                    key_transfer_and_L4_linear_seconds=0., suffix_seconds=0.,
                    full_vocabulary_head_seconds=0., pair_two_row_head_seconds=0.,
                    activation_backward_seconds=0., gradient_contraction_seconds=0.,
                    factor_write_D2H_seconds=0., final_gradient_D2H_seconds=0.,
                    full_teacher_reads=0, accumulation_dtype="GPU_FP64" if self.device.type == "cuda" else "CPU_FP64")
        with torch.set_grad_enabled(gradient):
            for index in selected:
                row, a, keys = self._document(endpoint, index, derivative=gradient)
                rows.append(row)
                work["documents"] += 1
                work["input_tokens"] += row["input_tokens"]
                work["scored_positions"] += row["scored_positions"]
                work["suffix_forwards"] += 1
                work["full_vocab_head_rows"] += row["scored_positions"]
                work["full_vocab_head_calls"] += math.ceil(row["scored_positions"] / self.chunk)
                for name, seconds in row["phase_seconds"].items():
                    work[name] += seconds
                if gradient:
                    # Gi FP32 on device, fixed-order accumulation FP64; only A
                    # (not dense Gi) goes to CPU at each document boundary.
                    self.oracle._sync()
                    clock = time.perf_counter()
                    gi = a.T @ keys
                    _require(bool(torch.isfinite(gi).all()), "NONFINITE_FACTOR_WEIGHT_PRODUCT")
                    total.add_(gi.double(), alpha=-2.0 * max(0.0, -row["mu"]) / 512.0)
                    self.oracle._sync()
                    work["gradient_contraction_seconds"] += time.perf_counter() - clock
                    meta = dict(endpoint_identity=endpoint.identity, input_identity=self.input_identity,
                                cache_index=index, valid_tokens=row["input_tokens"],
                                capsule_sha256=row["capsule_sha256"], position=row["worst_position"],
                                label=row["worst_label"], competitor=row["worst_competitor"])
                    clock = time.perf_counter()
                    factor_members.append(factor_sink.put(index, a, meta))
                    self.oracle._sync()
                    work["factor_write_D2H_seconds"] += time.perf_counter() - clock
                    work["factor_D2H_bytes"] += a.numel() * a.element_size()
                    work["pair_two_row_head_calls"] += 1
                    work["backward_documents"] += 1
                    del gi
                del a, keys
        self._guard(endpoint)
        if gradient:
            _require(bool(torch.isfinite(total).all()), "NONFINITE_DECISION_GRADIENT_ACCUMULATION")
            clock = time.perf_counter()
            total = total.cpu()
            self.oracle._sync()
            work["final_gradient_D2H_seconds"] = time.perf_counter() - clock
            work["dense_gradient_D2H_bytes"] = total.numel() * total.element_size() if self.device.type == "cuda" else 0
            self.work["gradient_D2H_calls"] += int(self.device.type == "cuda")
            factor_sink.seal(expected_count=512)
            self.work["full_reference_backward_sweeps"] += 1
        self.work["scans"] += 1
        work["wall_seconds"] = time.perf_counter() - started
        expected = sum(self.capsules[i]["actual_length"] for i in selected)
        coverage = dict(complete=True, role=role, documents=len(rows), expected_documents=len(selected),
                        positions=work["scored_positions"], expected_positions=expected,
                        full_vocabulary=self.oracle.model.config.vocab_size,
                        backward_documents=work["backward_documents"], input_identity=self.input_identity,
                        all_valid_input_gradient=gradient, endpoint_identity=endpoint.identity)
        if role == "Dev128":
            coverage["postselection_seal"] = selection_seal
        _require(coverage["positions"] == expected, "FULL_REFERENCE_POSITION_COVERAGE")
        return DecisionObservation(endpoint.identity, self.input_identity, role, rows,
            risk_from_margins([r["mu"] for r in rows]), sum(r["mismatches"] for r in rows),
            total, coverage, work, factor_sink if gradient else None)

    def jacobian(self, observation, directions):
        """Only stored A/K contractions; no neural forward or backward."""
        _require(observation.factors is not None and observation.coverage["complete"] and
                 observation.input_identity == self.input_identity and observation.role == "R512",
                 "COMPLETE_MATCHED_CENTER_FACTORS_REQUIRED")
        _require(1 <= len(directions) <= 5, "DIRECTION_COUNT_1_TO_5")
        for d in directions:
            _require(tuple(d.shape) == self.shape and bool(torch.isfinite(d).all()), "FINITE_DIRECTION_SHAPE")
        started = time.perf_counter()
        self.oracle._guard()
        # At most five dense directions live on the device. They are transferred
        # once; each document's A/K is streamed, and no neural graph is built.
        resident = tuple(d.detach().to(self.device, dtype=torch.float64) for d in directions)
        j_device = torch.empty((512, len(directions)), dtype=torch.float64, device=self.device)
        for i in range(512):
            a, meta = observation.factors.get(i)
            _require(meta["endpoint_identity"] == observation.endpoint_identity and
                     meta["input_identity"] == self.input_identity and meta["cache_index"] == i,
                     "FACTOR_CENTER_OR_DOCUMENT_MISMATCH")
            k = self.oracle.caches[i].valid_keys().to(self.device, dtype=torch.float64)
            a = a.to(self.device, dtype=torch.float64)
            for c, d in enumerate(resident):
                j_device[i, c] = (a * (d @ k).T).sum()
        j = j_device.cpu()
        self.oracle._guard()
        _require(bool(torch.isfinite(j).all()), "NONFINITE_COEFFICIENT_JACOBIAN")
        self.last_jacobian_work = dict(documents=512, directions=len(directions),
            device=str(self.device), contraction_dtype="float64", neural_forwards=0, neural_backwards=0,
            resident_direction_bytes=sum(d.numel() * d.element_size() for d in resident),
            wall_seconds=time.perf_counter() - started, factor_read_hash_seconds="NOT_SEPARATED")
        return j

    def check_panel(self, endpoint, indices):
        """Exactly four preselected technical docs; never a selector subset."""
        self._guard(endpoint)
        _require(len(indices) == 4 and len(set(indices)) == 4 and
                 all(type(i) is int and 0 <= i < 512 for i in indices), "FIXED_FOUR_REFERENCE_TECHNICAL_PANEL")
        rows = []
        with torch.no_grad():
            for index in indices:
                row, _, _ = self._document(endpoint, index)
                rows.append(row)
        self._guard(endpoint)
        self.work["technical_checks"] += 1
        return dict(scope="FIXED_FOUR_REFERENCE_TECHNICAL_PANEL", indices=list(indices), rows=rows,
                    endpoint_identity=endpoint.identity, input_identity=self.input_identity,
                    panel_phi=risk_from_margins([r["mu"] for r in rows]),
                    full_bank_pass=False, numerical_pass_not_assigned=True)

    def check_pair_document(self, endpoint, index, direction, scales):
        """One fixed technical document: cached factor vs physical AD and FD.

        Returns measurements only, never widens or assigns tolerances. Scales
        must be frozen by the caller before observing these results (max12).
        Competitor/position is fixed from the endpoint's full-vocabulary scan.
        FD is on *actual FP32 weights* and reports perturbation/rounding signal.
        """
        self._guard(endpoint)
        _require(type(index) is int and 0 <= index < 512, "TECHNICAL_R512_INDEX")
        _require(1 <= len(scales) <= 12 and all(math.isfinite(s) and s > 0 for s in scales) and
                 len(set(scales)) == len(scales), "FROZEN_FD_SCALES_1_TO_12")
        _require(tuple(direction.shape) == self.shape and bool(torch.isfinite(direction).all()),
                 "TECHNICAL_DIRECTION_SHAPE_FINITE")
        row, a, keys = self._document(endpoint, index, derivative=True)
        cached_gradient = (a.T @ keys).detach().double().cpu()
        position, label, competitor = row["worst_position"], row["worst_label"], row["worst_competitor"]
        oracle = self.oracle
        with oracle.physical_weight(endpoint.cpu, gradient=True) as leaf:
            with torch.enable_grad():
                hidden = oracle._physical_hidden(index)
                logits = oracle._head(hidden, torch.tensor([position]))
                scalar = logits[0, label] - logits[0, competitor]
                physical_gradient, = torch.autograd.grad(scalar, leaf)
                physical_gradient = physical_gradient.detach().double().cpu()
                physical_scalar = float(scalar.detach())
        d = direction.detach().cpu().double()
        fd = []
        for scale in scales:
            plus = (endpoint.cpu.double() + scale * d).float()
            minus = (endpoint.cpu.double() - scale * d).float()
            values = []
            for w in (plus, minus):
                with torch.no_grad(), oracle.physical_weight(w):
                    hidden = oracle._physical_hidden(index)
                    logits = oracle._head(hidden, torch.tensor([position]))
                    value = logits[0, label] - logits[0, competitor]
                    _require(bool(torch.isfinite(value)), "NONFINITE_TECHNICAL_FD")
                    values.append(float(value))
            actual = plus.double() - minus.double()
            predicted = float((physical_gradient * actual).sum())
            observed = values[0] - values[1]
            fd.append(dict(scale=scale, plus_margin=values[0], minus_margin=values[1],
                           observed_difference=observed, actual_FP32_AD_prediction=predicted,
                           derivative=observed / (2 * scale),
                           requested_AD_direction=float((physical_gradient * d).sum()),
                           actual_perturbation_norm=float(actual.norm()),
                           actual_FP32_moved=bool(torch.count_nonzero(actual)),
                           plus_sha256=tensor_sha256(plus), minus_sha256=tensor_sha256(minus)))
        error = cached_gradient - physical_gradient
        self._guard(endpoint)
        self.work["technical_checks"] += 1
        return dict(scope="ONE_BOUNDED_TECHNICAL_DOCUMENT_NOT_FULL_BANK", index=index,
                    endpoint_identity=endpoint.identity, input_identity=self.input_identity,
                    full_bank_pass=False, tolerance_pass_not_assigned=True,
                    worst_position=position, label=label, competitor=competitor,
                    tie_at_center=row["mu"] == 0,
                    full_scan_margin=row["mu"], two_row_backward_margin=row["exposed_pair_backward_scalar"],
                    physical_full_head_margin=physical_scalar,
                    gradient_relative_l2=float(error.norm() / physical_gradient.norm().clamp_min(1e-300)),
                    gradient_max_abs=float(error.abs().max()),
                    cached_gradient_norm=float(cached_gradient.norm()),
                    physical_gradient_norm=float(physical_gradient.norm()), FD=fd)


def acceptance(native, candidate, *, native_phi_history=0.0, candidate_phi_history=0.0,
               current_guard_pass, invariant_pass, history_guard_pass):
    """Fixed v1 full-bank finite acceptance; no official observer inputs."""
    _require(native.role == candidate.role == "R512" and
             native.input_identity == candidate.input_identity, "ACCEPTANCE_REFERENCE_IDENTITY")
    for obs in (native, candidate):
        _require(obs.coverage.get("complete") is True and obs.coverage["documents"] == 512 and
                 obs.coverage["positions"] == obs.coverage["expected_positions"] and
                 len(obs.rows) == 512, "ACCEPTANCE_FULL512_COVERAGE")
    reasons = []
    if not current_guard_pass:
        reasons.append("CURRENT_GUARD")
    if not invariant_pass:
        reasons.append("CURRENT_RESPONSE_INVARIANT")
    if not history_guard_pass:
        reasons.append("ACTIVE_HISTORY_GUARD")
    new_flips = repaired = 0
    worst_regressions = []
    for n, c in zip(native.rows, candidate.rows):
        _require(all(n[k] == c[k] for k in ("index", "source_row_id", "capsule_sha256", "positions", "labels")),
                 "ACCEPTANCE_DOCUMENT_TOKEN_IDENTITY")
        _require(len(n["correct"]) == len(c["correct"]) == n["scored_positions"], "ACCEPTANCE_POSITION_IDS")
        new_flips += sum(a and not b for a, b in zip(n["correct"], c["correct"]))
        repaired += sum(not a and b for a, b in zip(n["correct"], c["correct"]))
        if max(0., -c["mu"]) > max(0., -n["mu"]) + WORST_DEFICIT_TOLERANCE:
            worst_regressions.append(n["index"])
    if new_flips:
        reasons.append("REFERENCE_NEW_FLIP")
    if worst_regressions:
        reasons.append("REFERENCE_WORST_DEFICIT")
    tau = risk_tolerance(native.phi_reference, native_phi_history)
    if candidate.phi_reference > native.phi_reference:
        reasons.append("REFERENCE_RISK_INCREASE")
    gain = native.phi_reference + native_phi_history - candidate.phi_reference - candidate_phi_history
    if not gain > tau:
        reasons.append("NO_RESOLVED_COMBINED_RISK_DECREASE")
    if candidate.mismatches > native.mismatches:
        reasons.append("TOTAL_CHOICE_MISMATCH_INCREASE")
    if candidate_phi_history != 0:
        reasons.append("ACTIVE_HISTORY_NEGATIVE_SLACK")
    return dict(accepted=not reasons, reasons=reasons, new_flips=new_flips, repaired_choices=repaired,
                worst_regression_documents=worst_regressions, tau_risk=tau, combined_risk_gain=gain,
                label=("CHOICE_REPAIRED" if repaired else "MARGIN_ONLY") if not reasons else "REJECTED")
