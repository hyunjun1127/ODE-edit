"""W0 raw-greedy G256 preparation, one bounded document at a time.

This is the model-call adapter, not a launcher or a new editing policy. Generation
uses the physical decoder without KV reuse/processors. The canonical completed
prefix forward jointly produces the teacher and the raw L4 key/residual cache.
Only a full 640-document seal is a GeneratedTeacherStore; per-document evidence
and failed/partial files do not imply a ready teacher. All writes are create-once.
CPU fixtures exercise the adapter with tiny fake models, not actual Llama parity.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import (
    WEIGHT, model_guard, tensor_sha256,
)
from .generated_teacher import (
    CPUFixture, DATA_ID, PINNED_INPUTS_SHA256, PRODUCTION_VOCABULARY, ROLE_COUNTS,
    SCHEMA, canonical_sha256, file_sha256, generation_policy, make_capsule,
    validate_binding, validate_capsule, validate_inputs,
)


class ReferencePreparationError(RuntimeError):
    """Technical preparation failure; never interpreted as a usable fallback."""


def _require(value, message):
    if not value:
        raise ReferencePreparationError(message)


def _atomic_create(path, writer):
    """Durable no-clobber publication; failed temporary bytes remain evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".partial-", dir=path.parent)
    with os.fdopen(fd, "wb") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    # A same-filesystem hard link makes a fully written file visible atomically,
    # and fails instead of replacing an existing target. Remove only our own
    # successfully published temporary link, never an old artifact.
    os.link(temporary, path)
    os.unlink(temporary)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def atomic_json(path, value):
    content = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
    return _atomic_create(path, lambda handle: handle.write(content))


def atomic_array(path, value):
    value = np.asarray(value)
    _require(value.dtype == np.float32 and value.flags.c_contiguous, "ARRAY_FP32_CONTIGUOUS")
    _require(bool(np.isfinite(value).all()), "NONFINITE_PREPARATION_ARRAY")
    return _atomic_create(path, lambda handle: np.save(handle, value, allow_pickle=False))


def descriptor(root, path, *, array=None):
    path, root = Path(path), Path(root)
    result = dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                  sha256=file_sha256(path))
    if array is not None:
        result.update(shape=list(array.shape), dtype="float32")
    return result


def configured_eos(model):
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is None:
        configured = getattr(model.config, "eos_token_id", None)
    values = [configured] if type(configured) is int else configured
    _require(isinstance(values, (tuple, list)) and bool(values) and
             all(type(x) is int and x >= 0 for x in values) and len(set(values)) == len(values),
             "CONFIGURED_EOS_REQUIRED")
    return list(values)


class GeneratedReferenceBuilder:
    """Generate and seal exactly the prebound R512 then Dev128 input order.

    ``binding.w0_sha256`` uses alltoken.tensor_sha256's header+bytes convention.
    Source/model/tokenizer manifest SHA fields are caller-verified provenance;
    selected-W0 bytes, configured EOS, dimensions and runtime are checked here.
    Full model tensor rehash is not silently claimed by this adapter.

    ``build_document(index)`` is useful for bounded preparation tests/repair.
    ``build()`` traverses all 640 and publishes manifest.json only after every
    document succeeds. Existing complete documents may be reused only with
    ``reuse_completed=True`` and exact per-file evidence; incomplete documents
    are never overwritten or treated as complete.
    """
    def __init__(self, model, binding, root, inputs_path, *, cpu_fixture=None,
                 retain_upstream_cache=True, reuse_completed=False):
        self.model, self.binding = model, deepcopy(binding)
        self.root, self.inputs_path = Path(root).resolve(), Path(inputs_path).resolve()
        self.fixture = cpu_fixture
        _require(cpu_fixture is None or isinstance(cpu_fixture, CPUFixture), "CPU_FIXTURE_OPT_IN")
        self.production = cpu_fixture is None
        self.vocabulary = PRODUCTION_VOCABULARY if self.production else cpu_fixture.vocabulary_size
        self.key_size = 14336 if self.production else cpu_fixture.key_size
        self.hidden_size = 4096 if self.production else cpu_fixture.hidden_size
        self.inputs_sha = PINNED_INPUTS_SHA256 if self.production else cpu_fixture.inputs_sha256
        self.retain_cache = bool(retain_upstream_cache)
        self.reuse_completed = bool(reuse_completed)
        validate_binding(self.binding, self.vocabulary)
        _require(self.binding["generation"] == generation_policy(configured_eos(model), kv_cache=False),
                 "PINNED_NO_KV_GENERATION_POLICY")
        _require(self.inputs_path.is_file() and file_sha256(self.inputs_path) == self.inputs_sha,
                 "PINNED_INPUT_BYTES")
        self.inputs = json.loads(self.inputs_path.read_text())
        validate_inputs(self.inputs, self.binding, self.vocabulary)
        self._check_runtime()
        self.parameter = dict(model.named_parameters())[WEIGHT]
        self.device = self.parameter.device
        _require(tuple(self.parameter.shape) == (self.hidden_size, self.key_size), "SELECTED_WEIGHT_SHAPE")
        _require(tensor_sha256(self.parameter) == self.binding["w0_sha256"], "W0_WEIGHT_SHA")
        self.guard = model_guard(model)
        self.root.mkdir(parents=True, exist_ok=True)
        self.preparation = dict(data_id=DATA_ID, status="PARTIAL_NOT_READY",
                                binding_sha256=canonical_sha256(self.binding),
                                inputs_sha256=self.inputs_sha, production_ready=self.production,
                                expected_documents=dict(ROLE_COUNTS), generation_KV=False,
                                retain_upstream_cache=self.retain_cache,
                                w0_sha_convention="alltoken.header_shape_dtype_and_contiguous_bytes",
                                model_manifest_verification="CALLER_OWNED_NOT_FULL_REHASHED_HERE")
        prep_path = self.root / "preparation.json"
        if prep_path.exists():
            _require(self.reuse_completed and json.loads(prep_path.read_text()) == self.preparation,
                     "EXISTING_PREPARATION_BINDING")
        else:
            atomic_json(prep_path, self.preparation)

    def _check_runtime(self):
        if self.production:
            import transformers
            _require(transformers.__version__ == "4.44.2", "TRANSFORMERS_4_44_2_REQUIRED")
            _require(self.model.config.model_type == "llama" and
                     self.model.config.pretraining_tp == 1 and
                     self.model.config._attn_implementation == "eager", "LLAMA_EAGER_SINGLE_TP")
        _require(not self.model.training and len(self.model.model.layers) >= 5, "EVAL_PHYSICAL_LAYER4")
        params = list(self.model.parameters())
        _require(bool(params) and all(p.dtype == torch.float32 and not p.requires_grad for p in params),
                 "FROZEN_FP32_PARAMETERS")
        _require(all(p.device == params[0].device for p in params), "SINGLE_MODEL_DEVICE")
        _require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                 "TF32_OFF_REQUIRED")
        _require(self.model.model.layers[4].mlp.down_proj.bias is None, "BIASLESS_DOWN_PROJ")
        _require(self.model.lm_head.weight.shape == (self.vocabulary, self.hidden_size), "FULL_VOCABULARY_HEAD")

    def _guard(self, *, full_weight=False):
        _require(model_guard(self.model) == self.guard, "MODEL_STATE_HOOK_OR_MODE_CHANGED")
        _require(all(p.grad is None for p in self.model.parameters()), "UNEXPECTED_MODEL_GRADIENT")
        if full_weight:
            _require(tensor_sha256(self.parameter) == self.binding["w0_sha256"], "W0_WEIGHT_SHA")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _pack(self, ids):
        ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(ids.shape[1], device=self.device).unsqueeze(0))

    @torch.no_grad()
    def generate(self, input_ids):
        """Raw last-row physical decoder/head, no candidate/policy processors."""
        _require(len(input_ids) == 129, "GENERATION_PROMPT129")
        self._guard()
        eos = set(self.binding["generation"]["eos_token_ids"])
        sequence, answer = list(input_ids), []
        input_tokens = 0
        self._sync()
        started = time.perf_counter()
        for _ in range(256):
            packed = self._pack(sequence)
            hidden = self.model.model(**packed, use_cache=False, return_dict=True).last_hidden_state
            logits = self.model.lm_head(hidden[0, -1])
            _require(logits.shape == (self.vocabulary,) and logits.dtype == torch.float32,
                     "RAW_FULL_VOCABULARY_FP32_LOGITS")
            _require(bool(torch.isfinite(logits).all()), "NONFINITE_GENERATION_LOGITS")
            token = int(logits.argmax())  # first (= lowest ID) exact maximum
            input_tokens += len(sequence)
            sequence.append(token)
            answer.append(token)
            if token in eos:
                break
        self._sync()
        self._guard()
        return answer, dict(generation_seconds=time.perf_counter() - started,
                            generation_decoder_forwards=len(answer), generation_head_rows=len(answer),
                            generation_input_tokens=input_tokens, generation_valid_tokens=input_tokens,
                            KV_cache=False, backwards=0)

    @torch.no_grad()
    def teacher_and_cache(self, capsule):
        """One canonical physical TF decoder forward for teacher + upstream cache."""
        self._guard()
        packed = self._pack(capsule["tf_input_ids"])
        layer = self.model.model.layers[4]
        keys, residual = [], []
        handles = [layer.mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs: keys.append(inputs[0].detach())),
            layer.post_attention_layernorm.register_forward_pre_hook(
            lambda _module, inputs: residual.append(inputs[0].detach()))]
        self._sync()
        started = time.perf_counter()
        try:
            hidden = self.model.model(**packed, use_cache=False, return_dict=True).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        _require(len(keys) == len(residual) == 1, "L4_KEY_RESIDUAL_CAPTURE_ONCE")
        positions = capsule["score_positions"]
        logits = self.model.lm_head(hidden[0, positions])
        _require(logits.shape == (capsule["actual_length"], self.vocabulary) and
                 logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()),
                 "CANONICAL_TF_FULL_VOCABULARY_FP32_FINITE")
        argmax = logits.argmax(-1).cpu().tolist()
        logp = logits.log_softmax(-1).detach().cpu().contiguous()
        k = keys[0][0].cpu().contiguous().clone()
        r = residual[0][0].cpu().contiguous().clone()
        length = len(capsule["tf_input_ids"])
        _require(k.shape == (length, self.key_size) and r.shape == (length, self.hidden_size),
                 "CAPTURED_FULL_TOKEN_DIMENSIONS")
        _require(k.dtype == r.dtype == torch.float32 and bool(torch.isfinite(k).all()) and
                 bool(torch.isfinite(r).all()) and bool(torch.isfinite(logp).all()),
                 "NONFINITE_TEACHER_OR_UPSTREAM_CACHE")
        self._sync()
        timing = dict(canonical_TF_seconds=time.perf_counter() - started,
                      canonical_TF_decoder_forwards=1, canonical_TF_input_tokens=length,
                      canonical_TF_head_rows=capsule["actual_length"], key_capture_forwards=0,
                      teacher_and_key_capture_share_TF_forward=True, backwards=0)
        self._guard()
        return logp.numpy(), k.numpy(), r.numpy(), argmax, timing

    def _reuse_document(self, path, row, index):
        evidence = json.loads(path.read_text())
        _require(evidence["status"] == "COMPLETE" and evidence["binding_sha256"] ==
                 canonical_sha256(self.binding) and evidence["inputs_sha256"] == self.inputs_sha,
                 "COMPLETED_DOCUMENT_BINDING")
        member = evidence["member"]
        expected_keys = {"index", "role", "ordinal", "source_row_id", "capsule", "logp"}
        if self.retain_cache:
            expected_keys.update(("keys", "residual"))
        _require(set(member) == expected_keys and member["index"] == index and
                 all(member[k] == row[k] for k in ("role", "ordinal", "source_row_id")),
                 "COMPLETED_DOCUMENT_MEMBERSHIP")
        for kind in ("capsule", "logp", "keys", "residual"):
            if kind not in member:
                continue
            item = member[kind]
            relative = Path(item["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts, "REUSED_PATH_ESCAPE")
            payload = self.root / relative
            _require(payload.is_file() and not payload.is_symlink() and
                     payload.stat().st_size == item["bytes"] and file_sha256(payload) == item["sha256"],
                     "COMPLETED_DOCUMENT_PAYLOAD_IDENTITY")
        cap = json.loads((self.root / member["capsule"]["path"]).read_text())
        validate_capsule(cap, row, self.binding, self.vocabulary)
        return member, cap, evidence["work"]

    def build_document(self, index):
        _require(type(index) is int and 0 <= index < 640, "DOCUMENT_INDEX")
        row = self.inputs[index]
        folder = self.root / row["role"] / f'{row["ordinal"]:03d}'
        completed = folder / "completion.json"
        if completed.exists():
            _require(self.reuse_completed, "REUSE_MUST_BE_EXPLICIT")
            return self._reuse_document(completed, row, index)
        _require(not folder.exists(), "PARTIAL_DOCUMENT_PRESERVED_NEW_ATTEMPT_REQUIRED")
        folder.mkdir(parents=True)
        started = time.perf_counter()
        try:
            answer, work = self.generate(row["input_ids"])
            capsule = make_capsule(row, answer, self.binding)
            validate_capsule(capsule, row, self.binding, self.vocabulary)
            atomic_json(folder / "generation-evidence.json", dict(
                status="GENERATION_COMPLETE_TF_UNRESOLVED", data_id=DATA_ID,
                input_row_sha256=canonical_sha256(row), binding_sha256=canonical_sha256(self.binding),
                y0=answer, work=work))
            logp, keys, residual, argmax, timing = self.teacher_and_cache(capsule)
            work.update(timing)
            writing = time.perf_counter()
            logp_path = atomic_array(folder / "logp.npy", logp)
            parity = dict(canonical_TF_argmax=argmax, generated_y0=answer,
                          exact=argmax == answer, comparisons=len(answer),
                          mismatch_positions=[128 + i for i, (a, b) in enumerate(zip(argmax, answer)) if a != b])
            atomic_json(folder / "generation-TF-parity.json", parity)
            _require(parity["exact"], "GENERATION_CANONICAL_TF_ARGMAX_MISMATCH")
            capsule_path = atomic_json(folder / "capsule.json", capsule)
            member = dict(index=index, role=row["role"], ordinal=row["ordinal"],
                          source_row_id=row["source_row_id"], capsule=descriptor(self.root, capsule_path),
                          logp=descriptor(self.root, logp_path, array=logp))
            if self.retain_cache:
                for name, array in (("keys", keys), ("residual", residual)):
                    path = atomic_array(folder / (name + ".npy"), array)
                    member[name] = descriptor(self.root, path, array=array)
            work.update(write_hash_seconds=time.perf_counter() - writing,
                        document_wall_seconds=time.perf_counter() - started,
                        logical_payload_bytes=int(logp.nbytes + (keys.nbytes + residual.nbytes if self.retain_cache else 0)),
                        generated_tokens=len(answer), actual_TF_positions=len(answer),
                        generation_TF_exact=True, model_execution="ACTUAL_LLAMA" if self.production else "CPU_FAKE_MODEL")
            self._guard()
            atomic_json(completed, dict(status="COMPLETE", binding_sha256=canonical_sha256(self.binding),
                                       inputs_sha256=self.inputs_sha, member=member, work=work))
            return member, capsule, work
        except BaseException as error:
            # IO failure may prevent even a failure receipt. Never hide the
            # original exception or declare an incomplete document complete.
            try:
                atomic_json(folder / "failure.json", dict(status="TECHNICAL_FAILURE",
                    index=index, source_row_id=row["source_row_id"], exception_type=type(error).__name__,
                    exception=str(error), elapsed_seconds=time.perf_counter() - started,
                    complete=False, fallback=False))
            except Exception:
                pass
            raise

    def build(self, *, progress=None):
        _require(not (self.root / "manifest.json").exists(), "COMPLETE_STORE_ALREADY_EXISTS")
        members, position_counts, work_rows = [], dict(R512=0, Dev128=0), []
        newly_built = reused_completed = 0
        self._guard(full_weight=True)
        for index in range(640):
            row = self.inputs[index]
            was_complete = (self.root / row["role"] / f'{row["ordinal"]:03d}' / "completion.json").exists()
            member, capsule, work = self.build_document(index)
            newly_built += int(not was_complete)
            reused_completed += int(was_complete)
            members.append(member)
            position_counts[capsule["role"]] += capsule["actual_length"]
            work_rows.append(dict(index=index, role=capsule["role"], ordinal=capsule["ordinal"],
                                  execution_this_invocation=not was_complete,
                                  cost_lineage="PRIOR_COMPLETE_DOCUMENT" if was_complete else "NEW_PREPARATION",
                                  **work))
            if progress is not None:
                progress(index, capsule, work)
        self._guard(full_weight=True)
        _require(file_sha256(self.inputs_path) == self.inputs_sha, "INPUT_BYTES_CHANGED_DURING_PREPARATION")
        manifest = dict(schema=SCHEMA, data_id=DATA_ID, status="COMPLETE", production_ready=self.production,
                        vocabulary_size=self.vocabulary, key_size=self.key_size, hidden_size=self.hidden_size,
                        inputs_sha256=self.inputs_sha, binding=self.binding,
                        binding_sha256=canonical_sha256(self.binding), documents=members,
                        document_counts=dict(ROLE_COUNTS), position_counts=position_counts,
                        upstream_cache_status="COMPLETE" if self.retain_cache else "NOT_BUILT")
        atomic_json(self.root / "work.json", dict(data_id=DATA_ID, documents=work_rows,
                    newly_built_documents=newly_built, reused_complete_documents=reused_completed,
                    prior_cost_is_new_cost=False,
                    timer_note="document_wall includes generation, TF, write/hash; do not sum nested timers",
                    numerical_validation="ACTUAL_GENERATION_TF_AND_CAPTURE_ONLY; NO_CORRECTION_PARITY_CLAIM"))
        path = atomic_json(self.root / "manifest.json", manifest)
        return dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size,
                    document_counts=dict(ROLE_COUNTS), position_counts=position_counts,
                    newly_built_documents=newly_built, reused_complete_documents=reused_completed,
                    production_ready=self.production)
