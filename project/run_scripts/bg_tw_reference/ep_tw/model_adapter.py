"""Actual-model EP-TW-1 observer and residual-coordinate differentiation.

No native source, model parameter, or shared helper is patched.  Scientific
gradient calls have one canonical-current sweep and one S64 sweep per episode.
Technical finite differences are separate forward-only probes, never a policy
gradient refresh.  All payloads returned here are LOCAL-ONLY.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


_TOKEN_CONTRACTS = None


def _token_contracts():
    """Read exact tiny contracts module without unrelated package __init__."""
    global _TOKEN_CONTRACTS
    if _TOKEN_CONTRACTS is None:
        path = Path(__file__).resolve().parents[2] / 'alphaedit_strength_neutral_barrier/contracts.py'
        name = 'project.run_scripts.bg_tw_reference.ep_tw._readonly_token_contracts'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _TOKEN_CONTRACTS = module
    return _TOKEN_CONTRACTS


class ModelBoundary(RuntimeError):
    pass


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _finite(value, label):
    if not torch.isfinite(value).all().item():
        raise ModelBoundary('NONFINITE_' + label)


class TeacherStore:
    """Read-only teacher mmap bound to completed teacher192 and token identities.

    Full file SHA verification is optional only when parent has already bound
    an exact full-SHA verification receipt.  Size/schema checks remain here.
    No teacher is generated and no reporting/audit data is selected by this API.
    """
    def __init__(self, reference_root, teacher_manifest, *, expected_manifest_sha=None,
                 verify_payload_hashes=True):
        self.root = Path(reference_root)
        self.manifest_path = Path(teacher_manifest)
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise ModelBoundary('TEACHER_MANIFEST_NOT_REGULAR')
        self.manifest_sha = _file_sha(self.manifest_path)
        if expected_manifest_sha and self.manifest_sha != expected_manifest_sha:
            raise ModelBoundary('TEACHER_MANIFEST_SHA')
        manifest = json.loads(self.manifest_path.read_text())
        if (manifest['status'] != 'TEACHER192_READY' or manifest['vocab'] != 128256
                or manifest['cache_dtype'] != 'float32'
                or manifest['model_revision'] != '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
                or manifest['shape_per_document'] != [128, 128256]
                or manifest['microbatch'] != 1):
            raise ModelBoundary('TEACHER_CONTRACT')
        token_path = self.root / 'reference-tokens.npz'
        if _file_sha(token_path) != manifest['reference_tokens']['sha256']:
            raise ModelBoundary('REFERENCE_TOKEN_SHA')
        data = np.load(token_path, allow_pickle=False)
        self.ids = data['input_ids'].copy()
        self.source_ids = list(map(str, data['source_row_ids']))
        if (self.ids.shape != (768, 257) or self.ids.dtype != np.int64
                or not np.array_equal(data['score_input_indices'], np.arange(129, 257))
                or not np.array_equal(data['score_logits_indices'], np.arange(128, 256))):
            raise ModelBoundary('REFERENCE_SCORE_SHIFT')
        self._shards = {}
        self._metadata = {}
        members = manifest['cache_shards']
        if len(members) != 24 or len({m['path'] for m in members}) != 24:
            raise ModelBoundary('TEACHER_SHARD_INVENTORY')
        for member in members:
            path = Path(member['path'])
            if path.is_symlink() or not path.is_file() or path.stat().st_size != member['bytes']:
                raise ModelBoundary('TEACHER_SHARD_TYPE_SIZE')
            if verify_payload_hashes and _file_sha(path) != member['sha256']:
                raise ModelBoundary('TEACHER_SHARD_SHA')
            role = path.parent.name
            if role not in ('S64', 'Dev128'):
                raise ModelBoundary('UNAPPROVED_TEACHER_ROLE')
            number = int(path.stem.split('-')[-1])
            begin = (0 if role == 'S64' else 64) + 8 * number
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (8, 128, 128256) or array.dtype != np.float32:
                raise ModelBoundary('TEACHER_SHARD_SCHEMA')
            for offset in range(8):
                index = begin + offset
                if index in self._shards:
                    raise ModelBoundary('TEACHER_DUPLICATE_DOCUMENT')
                self._shards[index] = (array, offset)
                self._metadata[index] = member
        if set(self._shards) != set(range(192)):
            raise ModelBoundary('TEACHER_DOCUMENT_COVERAGE')
        self.receipt = dict(manifest_sha256=self.manifest_sha,
                            token_sha256=manifest['reference_tokens']['sha256'],
                            full_sha_this_process=verify_payload_hashes,
                            model_revision=manifest['model_revision'], documents=192,
                            scored_positions_per_document=128, teacher_refresh=0,
                            payload_mode='READ_ONLY_NUMPY_MMAP')

    def indices(self, role):
        if role == 'S64':
            return list(range(64))
        if role == 'Dev128':
            return list(range(64, 192))
        raise ModelBoundary('UNAPPROVED_GENERIC_ROLE')

    def document(self, index, device):
        if index not in self._shards:
            raise ModelBoundary('TEACHER_DOCUMENT_NOT_AVAILABLE')
        array, offset = self._shards[index]
        # Copy from readonly mmap: no writable shared mmap tensor alias.
        teacher = torch.from_numpy(np.array(array[offset], copy=True)).to(device)
        inputs = torch.tensor(self.ids[index:index+1], dtype=torch.long, device=device)
        return inputs, teacher.unsqueeze(0), self.source_ids[index]


class _RawAnchoredWeight(torch.autograd.Function):
    """C=0 forward copies RAW bytes; linear backward is exact fixed-map VJP.

    The custom node avoids even adding signed zero to RAW at C0.  Nonzero C
    follows FP32(raw + C @ A).  No gradient flows into raw or the solver map.
    """
    @staticmethod
    def forward(ctx, correction, raw, fixed_a):
        ctx.save_for_backward(fixed_a)
        if torch.count_nonzero(correction).item() == 0:
            return raw.clone()
        return raw + correction @ fixed_a

    @staticmethod
    def backward(ctx, weight_gradient):
        (fixed_a,) = ctx.saved_tensors
        return weight_gradient @ fixed_a.T, None, None


def anchored_weight(correction, raw, fixed_a):
    if (correction.dtype != torch.float32 or raw.dtype != torch.float32
            or fixed_a.dtype != torch.float32 or correction.shape[0] != raw.shape[0]
            or fixed_a.shape != (correction.shape[1], raw.shape[1])):
        raise ModelBoundary('RESIDUAL_MAP_SCHEMA')
    return _RawAnchoredWeight.apply(correction, raw.detach(), fixed_a.detach())


def capture_native_fit(fitter, model, writer_tok, hp, history, projector, requests,
                       *, layer=4):
    """Read return-frame locals of the *unmodified* native compute_z function.

    No Python tracing/rewrite of optimizer operations, no global monkeypatch,
    no extra target evaluation.  Only return frames copy anchor/delta/target
    and report native Adam step/loss counters.  Parent owns disk persistence.
    """
    if sys.getprofile() is not None:
        raise ModelBoundary('EXISTING_PROFILE_OWNER')
    code = fitter.module.compute_z.__code__
    captures = []

    def observe(frame, event, result):
        if event != 'return' or frame.f_code is not code:
            return
        local = frame.f_locals
        if result is None or not isinstance(result, torch.Tensor):
            return  # Exception propagates; no successful capture is invented.
        anchor = local['target_init'].detach().cpu().clone()
        delta = local['delta'].detach().cpu().clone()
        state = local['opt'].state.get(local['delta'], {})
        step = state.get('step', 0)
        step = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        radius = float((hp.clamp_norm_factor * local['target_init'].norm()).item())
        captures.append(dict(case_id=int(local['request']['case_id']), anchor=anchor,
            delta=delta, target=result.detach().cpu().clone(), radius=radius,
            adam_updates=step, loss_evaluations=int(local['it']) + 1,
            lookup_indices=list(local['lookup_idxs']),
            teacher_sha256=tensor_sha(local['kl_distr_init']),
            loss_final=float(local['loss'].detach()),
            source_code_filename=frame.f_code.co_filename))

    sys.setprofile(observe)
    try:
        fit = fitter.fit(model, writer_tok, hp, history, projector, requests,
                         layer=layer, capture=True)
    finally:
        sys.setprofile(None)
    if len(captures) != len(requests) or [r['case_id'] for r in captures] != [int(r['case_id']) for r in requests]:
        raise ModelBoundary('NATIVE_ANCHOR_CAPTURE_ORDER')
    z = torch.stack(fit['captures']['compute_z'], dim=1)
    observed_z = torch.stack([r['target'] for r in captures], dim=1)
    if not torch.equal(z, observed_z):
        raise ModelBoundary('NATIVE_RETURN_CAPTURE_BRIDGE')
    if any(r['adam_updates'] > 24 or r['loss_evaluations'] > 25 for r in captures):
        raise ModelBoundary('NATIVE_TARGET_QUOTA')
    fit.update(anchors=torch.stack([r['anchor'] for r in captures], dim=1),
               radii=torch.tensor([r['radius'] for r in captures], dtype=torch.float32),
               target=z, target_observations=captures)
    fit['receipt']['anchor_capture'] = dict(
        method='READ_ONLY_NATIVE_COMPUTE_Z_RETURN_FRAME_PROFILE',
        calls=len(captures), numeric_source_modified=False,
        adam_updates=sum(r['adam_updates'] for r in captures),
        loss_evaluations=sum(r['loss_evaluations'] for r in captures),
        target_return_exact=True, extra_target_calls=0)
    return fit


class EpisodeAdapter:
    """CounterFact current and fixed-C4 observers at actual or residual weights."""
    def __init__(self, model, tokenizer, weight_name, teacher_store, *, current_microbatch=16):
        self.model, self.tok = model, tokenizer
        self.weight_name, self.teacher = weight_name, teacher_store
        self.parameters = dict(model.named_parameters())
        if weight_name not in self.parameters:
            raise ModelBoundary('SELECTED_WEIGHT_MISSING')
        self.weight = self.parameters[weight_name]
        self.device = self.weight.device
        if current_microbatch != 16:
            raise ModelBoundary('CANONICAL_MICROBATCH16_REQUIRED')
        self.microbatch = current_microbatch
        if model.training or any(p.requires_grad or p.grad is not None for p in self.parameters.values()):
            raise ModelBoundary('EVAL_FROZEN_MODEL_REQUIRED')
        if any(p.dtype != torch.float32 for p in self.parameters.values()):
            raise ModelBoundary('MODEL_FP32_REQUIRED')
        self.raw = self.fixed_a = None
        self._sweeps = 0

    def set_episode(self, raw, fixed_a):
        if (raw.shape != self.weight.shape or raw.dtype != torch.float32
                or fixed_a.dtype != torch.float32 or fixed_a.ndim != 2
                or fixed_a.shape[1] != raw.shape[1]):
            raise ModelBoundary('EPISODE_SCHEMA')
        self.raw = raw.detach().to(self.device).clone()
        self.fixed_a = fixed_a.detach().to(self.device).clone()
        _finite(self.raw, 'RAW'); _finite(self.fixed_a, 'FIXED_A')
        self._versions = (self.raw._version, self.fixed_a._version)
        self._sweeps = 0

    def _assert_episode(self):
        if self.raw is None or self._versions != (self.raw._version, self.fixed_a._version):
            raise ModelBoundary('EPISODE_NOT_SET_OR_MUTATED')

    @contextmanager
    def _state_guard(self):
        versions = [(p, p.data_ptr(), p._version, p.requires_grad) for p in self.parameters.values()]
        training = {name: module.training for name, module in self.model.named_modules()}
        hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks),
                        tuple(module._backward_hooks)) for name, module in self.model.named_modules()}
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(self.device).clone() if self.device.type == 'cuda' else None
        try:
            yield
        finally:
            if any(p.data_ptr() != ptr or p._version != version or p.requires_grad != req or p.grad is not None
                   for p, ptr, version, req in versions):
                raise ModelBoundary('EVALUATOR_PARAMETER_MUTATION')
            if any(module.training != training[name] or hooks[name] != (
                tuple(module._forward_hooks), tuple(module._forward_pre_hooks), tuple(module._backward_hooks))
                for name, module in self.model.named_modules()):
                raise ModelBoundary('EVALUATOR_MODE_OR_HOOK_MUTATION')
            if not torch.equal(cpu_rng, torch.random.get_rng_state()):
                raise ModelBoundary('EVALUATOR_CPU_RNG_MUTATION')
            if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.device)):
                raise ModelBoundary('EVALUATOR_CUDA_RNG_MUTATION')

    def _forward(self, inputs, attention, correction):
        kwargs = dict(input_ids=inputs, attention_mask=attention, use_cache=False)
        if correction is None:
            return self.model(**kwargs).logits.float()
        self._assert_episode()
        replacement = anchored_weight(correction, self.raw, self.fixed_a)
        return torch.func.functional_call(self.model, {self.weight_name: replacement}, (), kwargs,
                                          strict=False).logits.float()

    def _current_groups(self, records):
        records = list(records)
        if not records or len({int(r['case_id']) for r in records}) != len(records):
            raise ModelBoundary('CURRENT_REQUEST_INVENTORY')
        encoded = []
        for record in records:
            rw = record['requested_rewrite']
            prompt = rw['prompt'].format(rw['subject'])
            prompt_ids = _token_contracts().prompt_token_ids(self.tok, prompt)
            target_ids = _token_contracts().target_token_ids(self.tok, rw['target_new']['str'])
            encoded.append((record, prompt, prompt_ids, target_ids))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        if pad is None:
            raise ModelBoundary('CANONICAL_PAD_TOKEN')
        for begin in range(0, len(encoded), self.microbatch):
            group = encoded[begin:begin+self.microbatch]
            length = max(len(p)+len(t)-1 for _, _, p, t in group)
            ids = torch.full((len(group), length), int(pad), dtype=torch.long, device=self.device)
            attention = torch.zeros_like(ids)
            positions = []
            for row, (_, _, prompt_ids, target_ids) in enumerate(group):
                full = prompt_ids + target_ids
                offset = length - len(full) + 1
                ids[row, offset:] = torch.tensor(full[:-1], device=self.device)
                attention[row, offset:] = 1
                positions.append(list(range(offset+len(prompt_ids)-1, length)))
            yield group, ids, attention, positions

    def _current_values(self, logits, group, positions):
        # Match historical source: FP32 log_softmax all positions, argmax ties
        # choose the first token.  Canonical NLL preference tie failure is a
        # distinct R/P/N observer rule, not a new strict tie rule here.
        logp = torch.log_softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        losses, rows = [], []
        for i, (record, prompt, _, targets) in enumerate(group):
            target = torch.tensor(targets, device=self.device)
            selected = logp[i, positions[i], :].gather(1, target[:, None])
            loss = -selected.mean()
            losses.append(loss)
            predicted = predictions[i, positions[i]]
            correct = predicted == target
            rows.append(dict(case_id=int(record['case_id']), kind='rewrite_target_new',
                prompt_index=0, prompt=prompt, target=record['requested_rewrite']['target_new']['str'],
                target_token_ids=list(targets), nll=float(loss.detach()),
                token_predictions=predicted.detach().cpu().tolist(),
                token_correct=correct.detach().cpu().tolist(),
                all_tokens_correct=bool(correct.all().item())))
        return torch.stack(losses), rows

    def _current(self, records, correction, *, backward=False):
        records = list(records)
        rows, input_count, forwards, backward_count = [], 0, 0, 0
        gradient = torch.zeros_like(correction) if backward else None
        start = time.monotonic()
        with self._state_guard(), torch.set_grad_enabled(backward):
            for group, ids, attention, positions in self._current_groups(records):
                logits = self._forward(ids, attention, correction)
                losses, values = self._current_values(logits, group, positions)
                _finite(losses, 'CURRENT_NLL')
                if backward:
                    scalar = losses.sum() / len(records)
                    gradient.add_(torch.autograd.grad(scalar, correction)[0].detach())
                    backward_count += 1
                rows.extend(values)
                input_count += int(attention.sum())
                forwards += 1
                del logits, losses
        _sync(self.device)
        result = dict(E=math.fsum(r['nll'] for r in rows)/len(rows), denominator=len(rows), rows=rows,
            strict_ids=[r['case_id'] for r in rows if r['all_tokens_correct']],
            layout='HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE',
            reduction='TOKEN_MEAN_PER_REQUEST_THEN_ALL_REQUEST_MEAN',
            counts=dict(forwards=forwards, backwards=backward_count, input_tokens=input_count,
                        scored_tokens=sum(len(r['target_token_ids']) for r in rows)),
            seconds=time.monotonic()-start, parameter_hook_rng_nonmutation=True)
        return result, gradient

    def current(self, records, correction=None):
        return self._current(records, correction, backward=False)[0]

    def _generic(self, role, correction, *, backward=False):
        if backward and role != 'S64':
            raise ModelBoundary('ONLY_S64_GRADIENT_ALLOWED')
        indices = self.teacher.indices(role)
        rows, forward_seconds, io_seconds, backward_count = [], 0., 0., 0
        gradient = torch.zeros_like(correction) if backward else None
        start = time.monotonic()
        with self._state_guard(), torch.set_grad_enabled(backward):
            for index in indices:
                io_begin = time.monotonic()
                inputs, logp0, source_id = self.teacher.document(index, self.device)
                io_seconds += time.monotonic()-io_begin
                begin = time.monotonic()
                if inputs.shape[1] != 257 or logp0.shape[1] != 128:
                    raise ModelBoundary('GENERIC_SCORE_SCHEMA')
                logits = self._forward(inputs, torch.ones_like(inputs), correction)
                logpw = torch.log_softmax(logits[:, 128:256, :].float(), dim=-1)
                if logpw.shape != logp0.shape:
                    raise ModelBoundary('FULL_VOCAB_TEACHER_SHAPE')
                # Exact teacher direction; do not clamp negative FP32 residual.
                by_position = (logp0.exp() * (logp0-logpw)).sum(-1)
                loss = by_position.mean()
                _finite(loss, 'GENERIC_KL')
                natural = -logpw.gather(-1, inputs[:, 129:257, None]).mean()
                if backward:
                    gradient.add_(torch.autograd.grad(loss/len(indices), correction)[0].detach())
                    backward_count += 1
                rows.append(dict(index=index, role=role, source_row_id=source_id,
                    kl=float(loss.detach()), natural_nll=float(natural.detach()),
                    scored_positions=128, input_tokens=257,
                    teacher_normalizer_max_abs=float(torch.logsumexp(logp0, -1).abs().max())))
                _sync(self.device)
                forward_seconds += time.monotonic()-begin
                del logits, logpw, logp0, inputs, loss, natural, by_position
        result = dict(D=math.fsum(r['kl'] for r in rows)/len(rows), denominator=len(rows), rows=rows,
            role=role, reduction='VOCAB_SUM_POSITION128_MEAN_DOCUMENT_MEAN',
            counts=dict(forwards=len(rows), backwards=backward_count, input_tokens=257*len(rows),
                        scored_tokens=128*len(rows), teacher_reads=len(rows)),
            seconds=dict(total=time.monotonic()-start, forward_backward=forward_seconds,
                         teacher_read=io_seconds), parameter_hook_rng_nonmutation=True)
        return result, gradient

    def generic(self, role='S64', correction=None):
        return self._generic(role, correction, backward=False)[0]

    def gradient_sweeps(self, records):
        self._assert_episode()
        if self._sweeps:
            raise ModelBoundary('EPISODE_GRADIENT_SWEEPS_ALREADY_CONSUMED')
        if len(records) != self.fixed_a.shape[0]:
            raise ModelBoundary('FIXED_A_CURRENT_REQUEST_COUNT')
        self._sweeps += 1  # Never silently repeat a partly failed sweep.
        correction = torch.zeros((self.raw.shape[0], self.fixed_a.shape[0]),
                                 device=self.device, dtype=torch.float32, requires_grad=True)
        current, g_e = self._current(records, correction, backward=True)
        generic, g_d = self._generic('S64', correction, backward=True)
        return dict(gE=g_e, gD=g_d, current=current, generic=generic,
            receipt=dict(current_sweeps=1, generic_sweeps=1,
                gradients_finite=bool(torch.isfinite(g_e).all() and torch.isfinite(g_d).all()),
                separate_gradients=True, mixed_scalar_backward=False,
                C0_native_bytes_exact=torch.equal(anchored_weight(correction.detach(), self.raw, self.fixed_a), self.raw),
                raw_sha256=tensor_sha(self.raw), map_sha256=tensor_sha(self.fixed_a),
                residual_shape=list(correction.shape), current_microbatch=16, generic_microbatch=1,
                gradient_through_solver=False, all_token_weight_application=True))
