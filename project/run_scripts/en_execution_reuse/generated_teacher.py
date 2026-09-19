"""Strict, read-only EN-R512-G256-v1 teacher and coverage contracts.

This module never generates tokens, loads a model, selects documents, or opens
Report256. Production inputs are the *original bytes* of the pinned inputs.json.
Each complete manifest owns 640 capsule JSONs and FP32 NPY logp [T,V] files.
An optional complete upstream cache adds L4 keys [129+T-1,14336] and pre-MLP
residual [129+T-1,4096]. A teacher-only seal is not an upstream-cache seal;
runners needing it must set require_upstream_cache=True. All paths are relative
to the artifact root; hashes cover whole
files, including NPY headers. Token hashes cover contiguous little-endian int64.

No payload-hash bypass or partial-store promotion is supported. Initialization
verifies every document, one mmap at a time; document access rechecks immutable
seals and bytes. CPU fixtures require explicit opt-in, full 512/128 membership,
and production_ready=false. CPU fixture validation is not model validation.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


DATA_ID = "EN-R512-G256-v1"
SCHEMA = "GeneratedTeacherStore-v1"
CAPSULE_SCHEMA = "GeneratedTeacherCapsule-v1"
PINNED_INPUTS_SHA256 = "507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb"
PRODUCTION_VOCABULARY = 128256
ROLE_COUNTS = {"R512": 512, "Dev128": 128}


class GeneratedTeacherError(ValueError):
    """Technical evidence/data failure; never a native-success fallback."""


def _require(condition, message):
    if not condition:
        raise GeneratedTeacherError(message)


def _keys(value, expected, label):
    _require(isinstance(value, dict) and set(value) == set(expected), label + "_SCHEMA")


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), label + "_SHA256")


def canonical_sha256(value):
    """SHA of sorted compact UTF-8 JSON; NaN/Infinity are forbidden."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_sha256(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def invalid_constant(_):
        raise GeneratedTeacherError("NONFINITE_JSON")

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique, parse_constant=invalid_constant)


def generation_policy(eos_token_ids, *, kv_cache=False):
    """Preparation must bind the *configured* EOS IDs, not tokenizer guesses."""
    return dict(raw_argmax=True, tie_rule="lowest_token_id", logits_processors=[],
                max_new_tokens=256, min_new_tokens=0, eos_token_ids=list(eos_token_ids),
                append_fake_eos=False, kv_cache=kv_cache, canonical_tf_use_cache=False)


def validate_binding(binding, vocabulary_size):
    _keys(binding, ("source_sha256", "model_revision", "model_config_sha256",
                    "model_weights_sha256", "tokenizer_revision", "tokenizer_sha256",
                    "w0_sha256", "bos_token_id", "generation", "mask_policy",
                    "position_policy", "teacher_policy", "selected_parameter",
                    "runtime"), "BINDING")
    for name in ("source_sha256", "model_config_sha256", "model_weights_sha256",
                 "tokenizer_sha256", "w0_sha256"):
        _sha(binding[name], name)
    for name in ("model_revision", "tokenizer_revision"):
        _require(isinstance(binding[name], str) and bool(binding[name]), name + "_REVISION")
    _require(type(binding["bos_token_id"]) is int and
             0 <= binding["bos_token_id"] < vocabulary_size, "BOS_ID")
    policy = binding["generation"]
    _require(isinstance(policy, dict), "GENERATION_POLICY")
    eos = policy.get("eos_token_ids")
    _require(isinstance(eos, list) and bool(eos) and
             all(type(x) is int and 0 <= x < vocabulary_size for x in eos) and
             len(set(eos)) == len(eos), "CONFIGURED_EOS_IDS")
    _require(type(policy.get("kv_cache")) is bool, "GENERATION_KV_POLICY")
    _require(canonical_sha256(policy) == canonical_sha256(generation_policy(eos, kv_cache=policy["kv_cache"])),
             "GENERATION_POLICY")
    _require(binding["mask_policy"] == "all_ones_int64", "MASK_POLICY")
    _require(binding["position_policy"] == "contiguous_zero_based_int64", "POSITION_POLICY")
    _require(binding["teacher_policy"] == "canonical_W0_TF_FP32_full_vocab_log_softmax", "TEACHER_POLICY")
    _require(binding["selected_parameter"] == "model.layers.4.mlp.down_proj.weight", "L4_PARAMETER")
    _require(isinstance(binding["runtime"], dict) and bool(binding["runtime"]), "RUNTIME_BINDING")
    canonical_sha256(binding)  # Also rejects unhashable/nonfinite provenance.


@dataclass(frozen=True)
class CPUFixture:
    """Explicit nonproduction dimensions and input identity for CPU tests only."""
    vocabulary_size: int
    key_size: int
    hidden_size: int
    inputs_sha256: str

    def __post_init__(self):
        _require(all(type(x) is int and x > 0 for x in
                     (self.vocabulary_size, self.key_size, self.hidden_size)), "FIXTURE_DIMENSIONS")
        _sha(self.inputs_sha256, "FIXTURE_INPUTS")


def _tokens(value, vocabulary_size, label):
    _require(isinstance(value, list) and bool(value) and
             all(type(x) is int and 0 <= x < vocabulary_size for x in value), label + "_TOKENS")


def validate_inputs(rows, binding, vocabulary_size):
    """Check exact sealed list order, with no subset or sorting repair."""
    _require(isinstance(rows, list) and len(rows) == 640, "INPUT_MEMBERSHIP_COUNT")
    ids, prompts = set(), set()
    required = {"role", "ordinal", "source_row_id", "input_ids", "prompt_token_sha256",
                "window_token_sha256", "source_role", "source_text_sha256", "window_start"}
    for index, row in enumerate(rows):
        _keys(row, required, "INPUT_ROW")
        role = "R512" if index < 512 else "Dev128"
        ordinal = index if index < 512 else index - 512
        _require(row["role"] == role and type(row["ordinal"]) is int and
                 row["ordinal"] == ordinal, "INPUT_MEMBERSHIP_ORDER")
        source_role = ("S64" if index < 64 else "Reserve320" if index < 384 else
                       "AdditionalTrain128" if index < 512 else "Dev128")
        _require(row["source_role"] == source_role, "INPUT_SOURCE_ROLE")
        rid = row["source_row_id"]
        _require(isinstance(rid, str) and bool(rid) and rid not in ids, "INPUT_DUPLICATE_OR_DEV_LEAK")
        ids.add(rid)
        tokens = row["input_ids"]
        _tokens(tokens, vocabulary_size, "PROMPT")
        _require(len(tokens) == 129 and tokens[0] == binding["bos_token_id"], "PROMPT129_BOS")
        digest = token_sha256(tokens)
        _require(digest == row["prompt_token_sha256"] and digest not in prompts, "INPUT_PROMPT_SHA_OR_DUPLICATE")
        prompts.add(digest)
        for key in ("window_token_sha256", "source_text_sha256"):
            _sha(row[key], key)
        _require(type(row["window_start"]) is int and row["window_start"] >= 0, "INPUT_WINDOW_START")


def make_capsule(input_row, y0, binding):
    """Build metadata, not generation evidence; store still verifies TF argmax.

    No padding, EOS insertion, truncation, or token repair is performed.
    """
    prompt, y0 = list(input_row["input_ids"]), list(y0)
    tf = prompt + y0[:-1]
    ended = bool(y0) and y0[-1] in binding["generation"]["eos_token_ids"]
    return dict(schema=CAPSULE_SCHEMA, data_id=DATA_ID, status="COMPLETE",
                role=input_row["role"], ordinal=input_row["ordinal"],
                source_row_id=input_row["source_row_id"], input_row_sha256=canonical_sha256(input_row),
                binding_sha256=canonical_sha256(binding), input_ids=prompt,
                prompt_token_sha256=token_sha256(prompt), y0=y0, y0_sha256=token_sha256(y0),
                actual_length=len(y0), tf_input_ids=tf, tf_input_sha256=token_sha256(tf),
                full_sequence_sha256=token_sha256(prompt + y0), score_positions=list(range(128, 128 + len(y0))),
                attention_mask=[1] * len(tf), position_ids=list(range(len(tf))),
                stop_reason="eos" if ended else "max_new_tokens", length_censored=not ended)


def validate_capsule(capsule, input_row, binding, vocabulary_size):
    _require(isinstance(capsule, dict), "CAPSULE_SCHEMA")
    y0 = capsule.get("y0")
    _tokens(y0, vocabulary_size, "GENERATED")
    _require(1 <= len(y0) <= 256, "ACTUAL_GENERATION_LENGTH")
    eos = set(binding["generation"]["eos_token_ids"])
    _require(not any(x in eos for x in y0[:-1]), "TOKENS_AFTER_ACTUAL_EOS")
    _require(y0[-1] in eos or len(y0) == 256, "SHORT_NON_EOS_GENERATION")
    expected = make_capsule(input_row, y0, binding)
    _require(capsule == expected, "CAPSULE_IDENTITY_SHIFT_OR_COMPLETENESS")
    # Equality alone would allow booleans to impersonate integer JSON IDs.
    for key in ("input_ids", "tf_input_ids", "score_positions", "attention_mask", "position_ids"):
        _require(all(type(x) is int for x in capsule[key]), "CAPSULE_INTEGER_FIELDS")
    _require(type(capsule["actual_length"]) is int and
             type(capsule["ordinal"]) is int and type(capsule["length_censored"]) is bool,
             "CAPSULE_SCALAR_TYPES")
    return deepcopy(capsule)


@dataclass(frozen=True)
class TeacherDocument:
    index: int
    capsule: Mapping
    logp: np.ndarray
    keys: np.ndarray | None = None
    residual: np.ndarray | None = None
    canonical_tf_argmax_verified: bool = True

    def chunks(self, chunk_size=16):
        """Yield (absolute score positions, FP32 full-vocabulary logp view)."""
        _require(type(chunk_size) is int and chunk_size > 0, "CHUNK_SIZE")
        positions = self.capsule["score_positions"]
        for start in range(0, len(positions), chunk_size):
            yield tuple(positions[start:start + chunk_size]), self.logp[start:start + chunk_size]


class GeneratedTeacherStore:
    def __init__(self, root, manifest_path, *, expected_manifest_sha256,
                 inputs_path, expected_binding, cpu_fixture=None,
                 require_upstream_cache=False):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path)
        self.inputs_path = Path(inputs_path)
        self._manifest_sha = expected_manifest_sha256
        _sha(expected_manifest_sha256, "MANIFEST")
        _require(cpu_fixture is None or isinstance(cpu_fixture, CPUFixture), "CPU_FIXTURE_OPT_IN")
        self.production_ready = cpu_fixture is None
        self.vocabulary_size = PRODUCTION_VOCABULARY if cpu_fixture is None else cpu_fixture.vocabulary_size
        self.key_size = 14336 if cpu_fixture is None else cpu_fixture.key_size
        self.hidden_size = 4096 if cpu_fixture is None else cpu_fixture.hidden_size
        self._inputs_sha = PINNED_INPUTS_SHA256 if cpu_fixture is None else cpu_fixture.inputs_sha256
        self._binding = deepcopy(expected_binding)
        validate_binding(self._binding, self.vocabulary_size)
        self._check_seals()
        manifest = _json(self.manifest_path)
        _keys(manifest, ("schema", "data_id", "status", "production_ready", "vocabulary_size",
                         "key_size", "hidden_size", "inputs_sha256", "binding", "binding_sha256",
                         "documents", "document_counts", "position_counts", "upstream_cache_status"), "MANIFEST")
        _require(manifest["schema"] == SCHEMA and manifest["data_id"] == DATA_ID and
                 manifest["status"] == "COMPLETE", "MANIFEST_SCHEMA_OR_PARTIAL")
        _require(manifest["production_ready"] is self.production_ready, "PRODUCTION_FIXTURE_BOUNDARY")
        _require(manifest["upstream_cache_status"] in ("COMPLETE", "NOT_BUILT"), "UPSTREAM_CACHE_PARTIAL")
        self.has_upstream_cache = manifest["upstream_cache_status"] == "COMPLETE"
        _require(not require_upstream_cache or self.has_upstream_cache, "REQUIRED_UPSTREAM_CACHE_MISSING")
        self._payload_kinds = ("logp", "keys", "residual") if self.has_upstream_cache else ("logp",)
        self._file_kinds = ("capsule",) + self._payload_kinds
        for name in ("vocabulary_size", "key_size", "hidden_size"):
            _require(type(manifest[name]) is int and manifest[name] == getattr(self, name), "PAYLOAD_DIMENSIONS")
        _require(manifest["inputs_sha256"] == self._inputs_sha, "PINNED_INPUTS_SHA")
        _require(canonical_sha256(manifest["binding"]) == canonical_sha256(self._binding) and
                 manifest["binding_sha256"] == canonical_sha256(self._binding), "SOURCE_MODEL_POLICY_BINDING")
        self._inputs = _json(self.inputs_path)
        validate_inputs(self._inputs, self._binding, self.vocabulary_size)
        documents = manifest["documents"]
        _require(isinstance(documents, list) and len(documents) == 640 and
                 manifest["document_counts"] == ROLE_COUNTS, "TEACHER_MEMBERSHIP_COUNT")
        self._documents, self._capsules = [], []
        used_paths, used_inodes = set(), set()
        for index, member in enumerate(documents):
            _keys(member, ("index", "role", "ordinal", "source_row_id") + self._file_kinds, "DOCUMENT")
            row = self._inputs[index]
            _require(type(member["index"]) is int and member["index"] == index and
                     type(member["ordinal"]) is int and
                     all(member[key] == row[key] for key in ("role", "ordinal", "source_row_id")), "TEACHER_MEMBERSHIP_ORDER")
            for kind in self._file_kinds:
                descriptor = member[kind]
                fields = ("path", "bytes", "sha256") if kind == "capsule" else ("path", "bytes", "sha256", "shape", "dtype")
                _keys(descriptor, fields, "FILE_DESCRIPTOR")
                _sha(descriptor["sha256"], "PAYLOAD")
                _require(type(descriptor["bytes"]) is int and descriptor["bytes"] > 0, "PAYLOAD_BYTES")
                path = self._path(descriptor["path"])
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                _require(path not in used_paths and inode not in used_inodes, "DUPLICATE_PAYLOAD_FILE")
                used_paths.add(path)
                used_inodes.add(inode)
                self._check_file(descriptor)
            cap = validate_capsule(_json(self._path(member["capsule"]["path"])), row,
                                   self._binding, self.vocabulary_size)
            length = cap["actual_length"]
            for kind, shape in (("logp", [length, self.vocabulary_size]),
                                ("keys", [128 + length, self.key_size]),
                                ("residual", [128 + length, self.hidden_size])):
                if kind not in self._payload_kinds:
                    continue
                desc = member[kind]
                _require(desc["dtype"] == "float32" and desc["shape"] == shape and
                         all(type(x) is int for x in desc["shape"]), "PAYLOAD_SCHEMA")
            self._documents.append(deepcopy(member))
            self._capsules.append(cap)
            with self._open_payloads(index):
                pass
        counts = {role: sum(c["actual_length"] for c in self._capsules if c["role"] == role)
                  for role in ROLE_COUNTS}
        _require(manifest["position_counts"] == counts, "POSITION_COVERAGE_COUNT")
        self._receipt = dict(data_id=DATA_ID, schema=SCHEMA, manifest_sha256=self._manifest_sha,
                             inputs_sha256=self._inputs_sha, binding_sha256=canonical_sha256(self._binding),
                             production_ready=self.production_ready, document_counts=dict(ROLE_COUNTS),
                             position_counts=counts, vocabulary_size=self.vocabulary_size,
                             upstream_cache_status=manifest["upstream_cache_status"],
                             payload_mode="READ_ONLY_DOCUMENT_MMAP", all_payload_sha256_verified=True,
                             canonical_tf_argmax_verified=True, model_execution_performed=False)
        self._check_seals()

    def _path(self, name):
        _require(isinstance(name, str) and bool(name), "PAYLOAD_PATH")
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts and
                 "." not in relative.parts, "PAYLOAD_PATH_ESCAPE")
        path = self.root
        for part in relative.parts:
            path = path / part
            _require(not path.is_symlink(), "PAYLOAD_SYMLINK")
        _require(path.is_file(), "PAYLOAD_MISSING")
        return path

    def _check_seals(self):
        for path, digest, label in ((self.manifest_path, self._manifest_sha, "MANIFEST"),
                                    (self.inputs_path, self._inputs_sha, "PINNED_INPUTS")):
            _require(path.is_file() and not path.is_symlink(), label + "_NOT_REGULAR")
            _require(file_sha256(path) == digest, label + "_SHA")

    def _check_file(self, descriptor):
        path = self._path(descriptor["path"])
        _require(path.stat().st_size == descriptor["bytes"], "PAYLOAD_SIZE")
        _require(file_sha256(path) == descriptor["sha256"], "PAYLOAD_SHA")

    @contextmanager
    def _open_payloads(self, index):
        member, cap = self._documents[index], self._capsules[index]
        arrays = {}
        try:
            for kind in self._payload_kinds:
                desc = member[kind]
                array = np.load(self._path(desc["path"]), mmap_mode="r", allow_pickle=False)
                arrays[kind] = array
                _require(isinstance(array, np.memmap) and list(array.shape) == desc["shape"] and
                         array.dtype == np.dtype("<f4") and array.flags.c_contiguous and
                         not array.flags.writeable, "NPY_PAYLOAD_SCHEMA")
                for start in range(0, len(array), 16):
                    block = array[start:start + 16]
                    _require(bool(np.isfinite(block).all()), "NONFINITE_" + kind.upper())
                    if kind == "logp":
                        _require(np.array_equal(np.argmax(block, axis=-1), cap["y0"][start:start + 16]),
                                 "CANONICAL_TF_ARGMAX_MISMATCH")
                        values = block.astype(np.float64)
                        maximum = values.max(axis=-1)
                        normalizer = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=-1))
                        _require(bool((np.abs(normalizer) <= 5e-5).all()), "TEACHER_LOGP_NORMALIZATION")
            yield TeacherDocument(index, deepcopy(cap), **arrays)
        finally:
            for array in arrays.values():
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def indices(self, role):
        _require(role in ROLE_COUNTS, "UNAPPROVED_ROLE")
        return tuple(range(512)) if role == "R512" else tuple(range(512, 640))

    def _index(self, index):
        _require(type(index) is int and 0 <= index < 640, "DOCUMENT_INDEX")

    def capsule(self, index):
        self._index(index)
        self._check_seals()
        self._check_file(self._documents[index]["capsule"])
        return deepcopy(self._capsules[index])

    @contextmanager
    def document(self, index):
        """Bounded lifetime: copy needed arrays before leaving this context."""
        self._index(index)
        self._check_seals()
        for kind in self._file_kinds:
            self._check_file(self._documents[index][kind])
        with self._open_payloads(index) as document:
            yield document
        # Catch changes while consuming mmap; an exception invalidates the sweep.
        for kind in self._file_kinds:
            self._check_file(self._documents[index][kind])

    def coverage(self, role="R512", *, require_backward=False):
        return CoverageTracker([self._capsules[i] for i in self.indices(role)],
                               vocabulary_size=self.vocabulary_size, role=role,
                               require_backward=require_backward)


class CoverageTracker:
    """Exact ordered all-document/position/vocabulary sweep receipts.

    R512 gradient sweeps additionally record each document's suffix backward
    once over its full actual positions. This records caller events; it is not
    independent proof that an autograd operation was actually executed.
    """
    def __init__(self, capsules: Sequence[Mapping], *, vocabulary_size,
                 role="R512", require_backward=False):
        _require(role in ROLE_COUNTS, "UNAPPROVED_ROLE")
        _require(not require_backward or role == "R512", "DEV_GRADIENT_FORBIDDEN")
        _require(len(capsules) == ROLE_COUNTS[role], "SWEEP_MEMBERSHIP_COUNT")
        _require(type(vocabulary_size) is int and vocabulary_size > 0, "SWEEP_VOCABULARY")
        self.role, self.vocabulary_size = role, vocabulary_size
        self.require_backward = require_backward
        self._capsules = deepcopy(list(capsules))
        ids = set()
        for ordinal, cap in enumerate(self._capsules):
            _require(cap["data_id"] == DATA_ID and cap["schema"] == CAPSULE_SCHEMA and
                     cap["status"] == "COMPLETE" and cap["role"] == role and
                     type(cap["ordinal"]) is int and cap["ordinal"] == ordinal,
                     "SWEEP_MEMBERSHIP_ORDER")
            _require(cap["source_row_id"] not in ids, "SWEEP_DUPLICATE_DOCUMENT")
            ids.add(cap["source_row_id"])
            _require(type(cap["actual_length"]) is int and 1 <= cap["actual_length"] <= 256 and
                     cap["score_positions"] == list(range(128, 128 + cap["actual_length"])), "SWEEP_POSITIONS")
        self._forward_document = self._forward_offset = self._backward_document = 0
        self._position_count = 0
        self._finished = False

    def record_forward(self, ordinal, positions, *, vocabulary_size):
        _require(not self._finished, "SWEEP_ALREADY_COMPLETE")
        _require(type(vocabulary_size) is int and vocabulary_size == self.vocabulary_size, "FULL_VOCABULARY_REQUIRED")
        _require(type(ordinal) is int and ordinal == self._forward_document and
                 ordinal < len(self._capsules), "SWEEP_FORWARD_ORDER_OR_DUPLICATE")
        positions = list(positions)
        expected = self._capsules[ordinal]["score_positions"]
        _require(bool(positions) and all(type(p) is int for p in positions) and
                 positions == expected[self._forward_offset:self._forward_offset + len(positions)],
                 "SWEEP_POSITION_GAP_DUPLICATE_OR_ORDER")
        self._forward_offset += len(positions)
        self._position_count += len(positions)
        if self._forward_offset == len(expected):
            self._forward_document += 1
            self._forward_offset = 0

    def record_backward(self, ordinal, positions, *, vocabulary_size):
        _require(not self._finished and self.require_backward, "UNEXPECTED_BACKWARD")
        _require(type(vocabulary_size) is int and vocabulary_size == self.vocabulary_size, "FULL_VOCABULARY_REQUIRED")
        _require(type(ordinal) is int and ordinal == self._backward_document and
                 ordinal < self._forward_document, "SWEEP_BACKWARD_ORDER_OR_DUPLICATE")
        _require(list(positions) == self._capsules[ordinal]["score_positions"], "BACKWARD_ALL_POSITIONS_REQUIRED")
        self._backward_document += 1

    def finish(self):
        _require(not self._finished, "SWEEP_ALREADY_COMPLETE")
        _require(self._forward_document == len(self._capsules) and self._forward_offset == 0,
                 "PARTIAL_FORWARD_SWEEP")
        _require(not self.require_backward or self._backward_document == len(self._capsules),
                 "PARTIAL_BACKWARD_SWEEP")
        self._finished = True
        return dict(data_id=DATA_ID, role=self.role, documents=self._forward_document,
                    positions=self._position_count, vocabulary_size=self.vocabulary_size,
                    backward_documents=self._backward_document, complete=True,
                    coverage_sha256=canonical_sha256([(c["source_row_id"], c["score_positions"])
                                                     for c in self._capsules]))


class DocumentMeanAccumulator:
    """Signed FP64 chunk-sum / actual T_i, then fixed-order document mean.

    add_chunk accepts a full-vocabulary KL *sum*, never a chunk mean. Sums are
    reduced in arrival order and no negative KL is clamped. Gradient math stays
    in the caller; this CPU helper makes denominators and coverage explicit.
    """
    def __init__(self, coverage: CoverageTracker):
        self.coverage = coverage
        self._document_sum = np.float64(0)
        self._bank_sum = np.float64(0)
        self._document_means = []

    def add_chunk(self, ordinal, positions, kl_sum, *, vocabulary_size):
        _require(isinstance(kl_sum, (float, int, np.floating)) and math.isfinite(kl_sum), "NONFINITE_KL_SUM")
        self.coverage.record_forward(ordinal, positions, vocabulary_size=vocabulary_size)
        self._document_sum = np.float64(self._document_sum + np.float64(kl_sum))
        _require(bool(np.isfinite(self._document_sum)), "NONFINITE_DOCUMENT_KL")
        if self.coverage._forward_document == ordinal + 1:
            mean = np.float64(self._document_sum / self.coverage._capsules[ordinal]["actual_length"])
            self._document_means.append(float(mean))
            self._bank_sum = np.float64(self._bank_sum + mean)
            _require(bool(np.isfinite(self._bank_sum)), "NONFINITE_BANK_KL")
            self._document_sum = np.float64(0)

    def finish(self):
        receipt = self.coverage.finish()
        _require(len(self._document_means) == ROLE_COUNTS[self.coverage.role], "PARTIAL_DOCUMENT_MEANS")
        return float(self._bank_sum / len(self._document_means)), tuple(self._document_means), receipt
