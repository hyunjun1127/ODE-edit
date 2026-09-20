"""Immutable target teachers and fresh all-position history gradient factors.

This module never saves W, M, optimizer state, or an equivalent update.  Only
per-version at-write distributions and input-dependent prefix caches enter the
cold store.  A replay sweep retains at most 612 CPU activation factors and one
microbatch graph.  Its only dense gradient is the aggregate selected KL block.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from project.run_scripts.en_adaptive_nullspace.json_io import save
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import (
    FullTokenCache, FullWeightLlamaOracle, signed_forward_kl, tensor_sha256,
)
from project.run_scripts.single_layer_edit_preserving_correction.binding import (
    pack, _token_contracts,
)


def _reference():
    path = Path(__file__).resolve().parents[3] / 'audits/global/2026-09-20-en-adapt-gss-history-design-v1/history_reference.py'
    spec = importlib.util.spec_from_file_location('en_gss_history_frozen_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _reference()
CAPACITY = 512
POOL_MAX = 612
MAP_DIM = 32
REPLICAS = 2


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_arrays(path, **arrays):
    """Lossless typed arrays, prevalidated, create once; no pickle payloads."""
    path = Path(path)
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise TypeError(f'ARRAY_WITHOUT_OBJECTS_REQUIRED:{name}')
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f'NONFINITE_COLD_ARRAY:{name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **arrays)  # Uncompressed, exact FP32 teacher bytes.
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha_file(temporary)
        size = os.stat(temporary).st_size
        os.link(temporary, path)
        return dict(path=str(path), bytes=size, sha256=digest)
    finally:
        os.unlink(temporary)  # Only this call's newly owned temporary name.


def make_maps(output_dim, input_dim, basis, *, seed=20260920):
    """Fixed independent Gaussian maps; FP64 projection/algebra, FP32 model.

    The explicit PCG64 generator and draw order are sealed, and do not consume
    the native fitting RNG.  Pstar is supplied by its orthonormal FP64 basis.
    """
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[0] != input_dim or basis.dtype != np.float64:
        raise ValueError('FP64_PSTAR_BASIS_SHAPE')
    rng = np.random.Generator(np.random.PCG64(seed))
    maps = []
    for _ in range(REPLICAS):
        output = rng.normal(0., 1. / math.sqrt(MAP_DIM), (output_dim, MAP_DIM))
        raw_input = rng.normal(0., 1. / math.sqrt(MAP_DIM), (input_dim, MAP_DIM))
        projected_input = basis @ (basis.T @ raw_input)
        maps.append((np.ascontiguousarray(output), np.ascontiguousarray(raw_input),
                     np.ascontiguousarray(projected_input)))
    return maps


def sketch_factors(activation, projected_keys, output_maps):
    """A[output,valid input positions]; keys are [replica,32,positions]."""
    if activation.ndim != 2 or projected_keys.shape != (REPLICAS, MAP_DIM, activation.shape[1]):
        raise ValueError('ALL_VALID_INPUT_FACTOR_POSITIONS')
    values = [((out.T @ activation.double()) @ projected_keys[r].T).flatten()
              for r, out in enumerate(output_maps)]
    result = torch.cat(values) / math.sqrt(REPLICAS)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('NONFINITE_NLL_SKETCH')
    return result


def activation_vjps(logits, activation, labels, teacher, *, selection, timings=None):
    """One fact graph, per-target mean losses, NLL then KL VJP exactly once."""
    kl = signed_forward_kl(logits, teacher)
    a_nll = None
    if selection:
        started = time.monotonic()
        nll = -logits.log_softmax(-1).gather(1, labels[:, None]).double().mean()
        a_nll, = torch.autograd.grad(nll, activation, retain_graph=True)
        if not bool(torch.isfinite(a_nll).all()):
            raise FloatingPointError('NONFINITE_NLL_ACTIVATION_FACTOR')
        if timings is not None:
            timings['NLL_VJP_seconds'] += time.monotonic()-started
    started = time.monotonic()
    a_kl, = torch.autograd.grad(kl, activation)
    if not bool(torch.isfinite(a_kl).all()):
        raise FloatingPointError('NONFINITE_KL_ACTIVATION_FACTOR')
    if timings is not None:
        timings['KL_VJP_seconds'] += time.monotonic()-started
    return float(kl.detach()), a_nll, a_kl


def aggregate_factors(factors, keys, weights, shape, device):
    """FP64 single aggregate; no per-fact dense temporary and one final D2H."""
    aggregate = torch.zeros(shape, dtype=torch.float64, device=device)
    for a, k, weight in zip(factors, keys, weights, strict=True):
        if a.ndim != 2 or k.ndim != 2 or a.shape[1] != k.shape[1]:
            raise ValueError('FACTOR_KEY_VALID_POSITION_MISMATCH')
        aggregate.addmm_(a.to(device, torch.float64), k.to(device, torch.float64).T,
                         alpha=float(weight))
    if not bool(torch.isfinite(aggregate).all()):
        raise FloatingPointError('NONFINITE_AGGREGATE_HISTORY_GRADIENT')
    return aggregate


class HistoryReplay:
    """One independent arm's cold store and bounded hot replay implementation.

    Rows supplied by the ledger contain ``version_id``, ``record`` and
    ``created_batch``.  ``capture`` returns ``{'bindings': {id: metadata},
    'receipt': ...}``.  ``prepare_pool`` returns selected IDs, normalized weights,
    CPU FP64 gradient, native L_H, and a compact receipt.  ``evaluate`` returns
    ``(L_H, receipt)`` using exactly that fixed selected bank and weights.
    """
    def __init__(self, model, tokenizer, root, basis, seed=20260920, expected_map_seal=None):
        self.model, self.tokenizer = model, tokenizer
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.basis = np.load(basis, mmap_mode='r', allow_pickle=False) if isinstance(basis, (str, Path)) else basis
        self.seed = int(seed)
        if isinstance(expected_map_seal,(str,Path)):
            seal_bytes = Path(expected_map_seal).read_bytes()
            self.expected_map_seal = json.loads(seal_bytes)
            self.expected_map_seal_sha256 = hashlib.sha256(seal_bytes).hexdigest()
        else:
            self.expected_map_seal = expected_map_seal
            self.expected_map_seal_sha256 = None
        self.oracle = None
        self.hot = OrderedDict()
        self.maps = None
        self.map_tensors = None
        self.projected_input = None
        self.verified_payloads = {}
        self.factor_diagnostic_done = False
        self.sketch_diagnostic_done = False
        self.counts = dict(capture_requests=0, capture_reused=0, teacher_bytes=0,
                           cold_bytes=0, prefix_forwards=0, pool_forwards=0,
                           NLL_VJPs=0, KL_VJPs=0, candidate_forwards=0,
                           gradient_D2H=0, activation_D2H_bytes=0,
                           diagnostic_forwards=0, diagnostic_backwards=0)

    @property
    def device(self):
        return dict(self.model.named_parameters())['model.layers.4.mlp.down_proj.weight'].device

    @property
    def shape(self):
        return tuple(dict(self.model.named_parameters())['model.layers.4.mlp.down_proj.weight'].shape)

    def _directory(self, identity):
        return self.root / 'versions' / hashlib.sha256(str(identity).encode()).hexdigest()

    def _canonical(self, row):
        rec = row['record']
        request = rec['requested_rewrite']
        contract = _token_contracts()
        prefix = contract.prompt_token_ids(self.tokenizer, request['prompt'].format(request['subject']))
        labels = contract.target_token_ids(self.tokenizer, request['target_new']['str'])
        if not prefix or not labels:
            raise ValueError('EMPTY_CANONICAL_PREFIX_OR_FULL_TARGET')
        return pack(prefix + labels[:-1]), list(range(len(prefix)-1, len(prefix)+len(labels)-1)), labels

    def rebind(self, expected_weight):
        if self.oracle is not None:
            return self.oracle.acknowledge_selected_write(expected_weight)
        return dict(status='NO_HISTORY_ORACLE_YET')

    def _use_cache(self, cache):
        if self.oracle is None:
            # One bounded bootstrap prefix, then reuse the established guard.
            self.oracle = FullWeightLlamaOracle(self.model, [cache.packed], require_reference_length=None)
            self.counts['prefix_forwards'] += 1
        self.oracle.caches = [cache]
        self.oracle.cache_guard = self.oracle._cache_guard()
        self.oracle._guard()
        return self.oracle

    def _remember(self, identity, item):
        self.hot[identity] = item
        self.hot.move_to_end(identity)
        while len(self.hot) > POOL_MAX:
            self.hot.popitem(last=False)

    def _load(self, identity):
        identity = str(identity)
        if identity in self.hot:
            self.hot.move_to_end(identity)
            return self.hot[identity]
        directory = self._directory(identity)
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['version_id'] != identity:
            raise ValueError('COLD_VERSION_IDENTITY_MISMATCH')
        path = directory / 'payload.npz'
        stat = path.stat()
        stable = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stat.st_size != metadata['payload']['bytes']:
            raise ValueError('COLD_PAYLOAD_SIZE_OR_SHA_MISMATCH')
        if self.verified_payloads.get(str(path)) != stable:
            if _sha_file(path) != metadata['payload']['sha256']:
                raise ValueError('COLD_PAYLOAD_SIZE_OR_SHA_MISMATCH')
            self.verified_payloads[str(path)] = stable
        with np.load(path, allow_pickle=False) as payload:
            expected = {'keys', 'residual', 'input_ids', 'attention_mask', 'position_ids', 'teacher_logp'}
            if set(payload.files) != expected:
                raise ValueError('COLD_PAYLOAD_MEMBER_CLOSURE')
            arrays = {k: torch.from_numpy(payload[k].copy()) for k in expected}
        packed = {k: arrays[k] for k in ('input_ids', 'attention_mask', 'position_ids')}
        cache = FullTokenCache(packed, arrays['keys'], arrays['residual'], metadata['input_identity'])
        teacher = arrays['teacher_logp']
        if teacher.dtype != torch.float32 or list(teacher.shape) != metadata['teacher_shape'] or not bool(torch.isfinite(teacher).all()):
            raise ValueError('COLD_TEACHER_FP32_FULLVOCAB_SHAPE')
        item = dict(metadata=metadata, cache=cache, teacher=teacher)
        self._remember(identity, item)
        return item

    def capture(self, versions, weight):
        started = time.monotonic()
        self.rebind(weight)
        device_weight = weight.to(self.device)
        bindings, new_count, reused, new_bytes = {}, 0, 0, 0
        for row in versions:
            identity = str(row['version_id'])
            directory = self._directory(identity)
            packed, positions, labels = self._canonical(row)
            if (directory / 'metadata.json').exists():
                item = self._load(identity)
                metadata = item['metadata']
                if (metadata['created_batch'] != int(row['created_batch']) or
                    metadata['positions'] != positions or metadata['labels'] != labels or
                    any(not torch.equal(item['cache'].packed[k], packed[k]) for k in packed)):
                    raise ValueError('EXISTING_VERSION_TEACHER_REFRESH_ATTEMPT')
                bindings[identity] = metadata
                reused += 1
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError('INCOMPLETE_COLD_VERSION_PRESERVED_NO_OVERWRITE')
            if not 1 <= int(row['created_batch']) < 100:
                raise ValueError('NEW_COLD_TEACHER_MUST_HAVE_FUTURE_REPLAY_CONSUMER')
            if self.oracle is None:
                self.oracle = FullWeightLlamaOracle(self.model, [packed], require_reference_length=None)
                cache = self.oracle.caches[0]
            else:
                cache = self.oracle._prepare(packed)
            self.counts['prefix_forwards'] += 1
            oracle = self._use_cache(cache)
            with torch.no_grad():
                hidden = oracle.suffix_hidden(0, device_weight)
                teacher_logits = oracle._head(hidden, torch.tensor(positions))
                correct = teacher_logits.argmax(-1).cpu().eq(torch.tensor(labels)).tolist()
                teacher = teacher_logits.log_softmax(-1).cpu().contiguous()
            if teacher.dtype != torch.float32 or not bool(torch.isfinite(teacher).all()):
                raise FloatingPointError('INVALID_FINAL_AT_WRITE_TEACHER')
            payload = _atomic_arrays(directory / 'payload.npz', keys=cache.keys.numpy(),
                                     residual=cache.residual.numpy(), teacher_logp=teacher.numpy(),
                                     **{k: v.numpy() for k, v in packed.items()})
            metadata = dict(version_id=identity, case_id=row['record']['case_id'],
                            created_batch=int(row['created_batch']), positions=positions, labels=labels,
                            input_identity=cache.input_identity, teacher_shape=list(teacher.shape),
                            teacher_dtype='torch.float32', teacher_sha256=tensor_sha256(teacher),
                            at_write_TF_token_correct=correct, at_write_TF_valid_tokens=len(labels),
                            at_write_TF_strict=all(correct),
                            at_write_desired_NLL=float(-teacher[torch.arange(len(labels)),torch.tensor(labels)].double().mean()),
                            payload=payload, checkpoint=False, save_checkpoints=False,
                            policy='FINAL_AT_WRITE_SUPPLIED_TARGET_FULL_VOCAB_FP32_TF',
                            prefix_cache_selected_weight_independent=True)
            save(directory / 'metadata.json', metadata)
            stat = (directory / 'payload.npz').stat()
            self.verified_payloads[str(directory / 'payload.npz')] = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            bindings[identity] = metadata
            self._remember(identity, dict(metadata=metadata, cache=cache, teacher=teacher))
            new_count += 1
            new_bytes += payload['bytes']
            self.counts['teacher_bytes'] += teacher.numel()*teacher.element_size()
            del hidden, teacher_logits
        self.counts['capture_requests'] += new_count
        self.counts['capture_reused'] += reused
        self.counts['cold_bytes'] += new_bytes
        return dict(bindings=bindings, receipt=dict(new_versions=new_count, reused_versions=reused,
                    new_cold_bytes=new_bytes, seconds=time.monotonic()-started,
                    terminal_unused_teacher_forward=False, checkpoint=False,
                    observer_forward_shared=False, reason='CANONICAL_FULL_TARGET_TEACHER_EXPLICIT_SEPARATE_COST'))

    def _initialize_maps(self):
        if self.maps is not None:
            return
        started = time.monotonic()
        self.maps = make_maps(*self.shape, self.basis, seed=self.seed)
        payload = {f'replica_{i}_{kind}': array for i, triples in enumerate(self.maps)
                   for kind, array in zip(('output', 'input', 'Pstar_input'), triples)}
        logical = {name:dict(shape=list(array.shape),dtype=str(array.dtype),
                            sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
                   for name,array in payload.items()}
        if self.expected_map_seal is not None:
            expected = self.expected_map_seal['arrays']
            if set(expected) != set(logical) or any(
                any(expected[name][field] != actual[field] for field in ('shape','dtype','sha256'))
                for name,actual in logical.items()):
                self.maps = None
                raise ValueError('PRESEALED_FIXED_MAP_LOGICAL_IDENTITY_MISMATCH')
        # No device materialization or persistence precedes the logical seal.
        self.map_tensors = [torch.from_numpy(o).to(self.device) for o, _, _ in self.maps]
        self.projected_input = [torch.from_numpy(p).to(self.device) for _, _, p in self.maps]
        path = self.root / 'fixed-maps.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                if set(previous.files) != set(payload) or any(not np.array_equal(previous[k], v) for k, v in payload.items()):
                    raise ValueError('FIXED_MAP_IDENTITY_MISMATCH')
            info = dict(path=str(path), bytes=path.stat().st_size, sha256=_sha_file(path))
        else:
            info = _atomic_arrays(path, **payload)
        self.map_receipt = dict(seed=self.seed, generator='numpy.Generator(PCG64)',
                                draw_order='replica0_output_input_then_replica1_output_input',
                                mean=0., std=1/math.sqrt(32), dimension=2048,
                                output_dim=32, input_dim=32, replicas=2,
                                algebra_dtype='float64', model_hidden_head_dtype='float32',
                                logical_arrays=logical,
                                expected_map_seal_sha256=self.expected_map_seal_sha256,
                                presealed_logical_verification='PASS' if self.expected_map_seal is not None else 'NOT_REQUESTED_CPU_FIXTURE_OR_UNSEALED_CALLER',
                                payload=info, seconds=time.monotonic()-started)
        if not (self.root / 'fixed-maps.json').exists():
            save(self.root / 'fixed-maps.json', self.map_receipt)

    def _projected_keys(self, identity, cache):
        self._initialize_maps()
        path = self._directory(identity) / 'projected-keys.npz'
        metadata_path = self._directory(identity) / 'projected-keys.json'
        if path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata['input_identity'] != cache.input_identity or metadata['map_sha256'] != self.map_receipt['payload']['sha256']:
                raise ValueError('PROJECTED_KEY_MAP_INPUT_IDENTITY')
            stat = path.stat()
            stable = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if stat.st_size != metadata['payload']['bytes'] or (self.verified_payloads.get(str(path)) != stable and _sha_file(path) != metadata['payload']['sha256']):
                raise ValueError('PROJECTED_KEY_PAYLOAD_SIZE_OR_SHA')
            self.verified_payloads[str(path)] = stable
            with np.load(path, allow_pickle=False) as data:
                value = torch.from_numpy(data['projected_keys'].copy())
        else:
            keys = cache.valid_keys().to(self.device, torch.float64)
            value = torch.stack([v.T @ keys for v in self.projected_input]).cpu()
            payload = _atomic_arrays(path, projected_keys=value.numpy())
            save(metadata_path,dict(input_identity=cache.input_identity,
                                    map_sha256=self.map_receipt['payload']['sha256'],payload=payload))
            stat = path.stat()
            self.verified_payloads[str(path)] = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if value.shape != (REPLICAS, MAP_DIM, cache.valid_positions().numel()) or not bool(torch.isfinite(value).all()):
            raise ValueError('PROJECTED_KEY_SHAPE_FINITE')
        return value.to(self.device)

    def _activation_logits(self, item, weight):
        oracle = self._use_cache(item['cache'])
        cache = item['cache']
        # Exact inherited residual + F.linear order; the leaf is the L4 output,
        # before residual addition.  All prefix positions remain in this leaf.
        cut = F.linear(cache.keys.to(self.device), weight.to(self.device)).detach().requires_grad_(True)
        hidden = cache.residual.to(self.device) + cut
        args = oracle._args(hidden, oracle._on_device(cache.packed))
        for layer in oracle.decoder.layers[5:]:
            hidden = layer(hidden, **args)[0]
        hidden = oracle.decoder.norm(hidden)
        logits = oracle._head(hidden, torch.tensor(item['metadata']['positions']))
        return cut, logits

    def _factor_diagnostic(self, weight, ids, selected_weights, factors):
        start = time.monotonic()
        chosen = sorted(range(len(ids)), key=lambda j: REFERENCE.priority(ids[j], self.seed))[:4]
        probabilities = np.asarray([selected_weights[j] for j in chosen], dtype=np.float64)
        probabilities /= probabilities.sum()
        items = [self._load(ids[j]) for j in chosen]
        leaf = weight.to(self.device).detach().requires_grad_(True)
        total = None
        for item, omega in zip(items, probabilities):
            oracle = self._use_cache(item['cache'])
            hidden = oracle.suffix_hidden(0, leaf)
            logits = oracle._head(hidden, torch.tensor(item['metadata']['positions']))
            loss = signed_forward_kl(logits, item['teacher'].to(self.device))*float(omega)
            total = loss if total is None else total+loss
        direct, = torch.autograd.grad(total, leaf)
        factored = aggregate_factors([factors[ids[j]] for j in chosen],
                                    [x['cache'].valid_keys() for x in items],
                                    probabilities, self.shape, self.device)
        difference = factored-direct.double()
        result = dict(version_ids=[ids[j] for j in chosen], weights=probabilities.tolist(),
                      direct_route='INDEPENDENT_ABSOLUTE_W_CACHED_AUTOGRAD_SINGLE_WEIGHTED_AGGREGATE',
                      maximum_absolute_error=float(difference.abs().max()),
                      relative_frobenius_error=float(difference.norm()/direct.double().norm().clamp_min(1e-300)),
                      direct_norm=float(direct.double().norm()), factor_norm=float(factored.norm()),
                      all_valid_prefix_positions_included=True, tolerance_gate=False,
                      precision='NOT_ESTABLISHED', seconds=time.monotonic()-start)
        self.counts['diagnostic_forwards'] += len(chosen)
        self.counts['diagnostic_backwards'] += 1
        self.factor_diagnostic_done = True
        save(self.root / 'factor-diagnostic-once.json', result)
        return result

    def _sketch_diagnostic(self, diag_factors, sketches, ids):
        start = time.monotonic()
        chosen = sorted(diag_factors, key=lambda i: REFERENCE.priority(i, self.seed))
        lengths = [diag_factors[i].shape[1] for i in chosen]
        activations = torch.cat([diag_factors[i] for i in chosen], dim=1).to(self.device, torch.float64)
        keys = np.concatenate([self._load(i)['cache'].valid_keys().numpy() for i in chosen], axis=1).astype(np.float64)
        projected = torch.from_numpy(np.asarray(self.basis.T @ keys)).to(self.device)
        a_gram = activations.T @ activations
        k_gram = projected.T @ projected
        offsets = np.cumsum([0]+lengths)
        exact = np.empty((len(chosen), len(chosen)), dtype=np.float64)
        for i in range(len(chosen)):
            si = slice(offsets[i], offsets[i+1])
            for j in range(i, len(chosen)):
                sj = slice(offsets[j], offsets[j+1])
                exact[i,j] = exact[j,i] = float((a_gram[si,sj]*k_gram[si,sj]).sum())
        norms = np.sqrt(np.maximum(0., np.diag(exact)))
        z = np.asarray([sketches[ids.index(i)] for i in chosen])
        zn = np.linalg.norm(z, axis=1)
        nonzero = (norms>1e-12)&(zn>1e-12)
        pairs = [(i,j) for i in range(len(chosen)) for j in range(i+1,len(chosen)) if nonzero[i] and nonzero[j]]
        truth = np.asarray([exact[i,j]/(norms[i]*norms[j]) for i,j in pairs])
        approximate = np.asarray([z[i]@z[j]/(zn[i]*zn[j]) for i,j in pairs])
        if len(pairs)>1 and np.ptp(truth)>0 and np.ptp(approximate)>0:
            from scipy.stats import spearmanr
            correlation = float(spearmanr(truth, approximate).statistic)
        else:
            correlation = None
        first = set(REFERENCE.gss_prune(np.asarray(sketches)[:,:1024], ids, cap=CAPACITY, seed=self.seed)['selected'])
        second = set(REFERENCE.gss_prune(np.asarray(sketches)[:,1024:], ids, cap=CAPACITY, seed=self.seed)['selected'])
        result = dict(version_ids=chosen, pairs=len(pairs),
                      exact_self_inner_products=np.diag(exact).tolist(),
                      negative_exact_self_inner_products=int((np.diag(exact)<0).sum()),
                      exact_factor_norms=norms.tolist(), sketch_norms=zn.tolist(),
                      zero_exact=int((norms<=1e-12).sum()), zero_sketch=int((zn<=1e-12).sum()),
                      sign_disagreements=int((np.sign(truth)!=np.sign(approximate)).sum()),
                      mean_absolute_cosine_error=float(np.abs(truth-approximate).mean()) if len(pairs) else None,
                      maximum_absolute_cosine_error=float(np.abs(truth-approximate).max()) if len(pairs) else None,
                      pair_rank_spearman=correlation, replica_selected_intersection=len(first&second),
                      replica_selected_union=len(first|second), replica_jaccard=len(first&second)/len(first|second),
                      exact_formula='sum((A_i.T A_j) * (K_i.T Pstar K_j))',
                      per_fact_dense_gradient=False, status='SKETCH_SELECTION_UNRESOLVED',
                      interpretation='BOUNDED_MEASUREMENT_NO_FIDELITY_TOLERANCE_OR_PERFORMANCE_GATE',
                      precision='NOT_ESTABLISHED', seconds=time.monotonic()-start)
        self.sketch_diagnostic_done = True
        save(self.root / 'sketch-diagnostic-once.json', result)
        return result

    def prepare_pool(self, weight, version_rows, *, select_gss, current_batch, recency):
        started = time.monotonic()
        rows = list(version_rows)
        ids = [str(row['version_id']) for row in rows]
        if len(ids)>POOL_MAX or len(ids)!=len(set(ids)) or (not select_gss and len(ids)>CAPACITY):
            raise ValueError('UNIQUE_BOUNDED_HISTORY_POOL_REQUIRED')
        self.rebind(weight)
        if not ids:
            receipt = dict(pool_size=0,pool_ids=[],selected_ids=[],weights=[],L_H=0.,
                           recency=bool(recency),current_batch=int(current_batch),created_batches=[],
                           selection_exercised=False,NLL_VJPs=0,KL_VJPs=0,microbatch_size=1,
                           forward_graphs=0,two_VJPs_share_graph=False,microbatch_extra_mean=False,
                           activation_host_bytes=0,dense_gradient_D2H=0,per_fact_dense_gradient=False,
                           hot_cache_versions=len(self.hot),pruning=dict(selected=[],removed=[],trace=[],selection_exercised=False),
                           diagnostics={},timings={},seconds=time.monotonic()-started,
                           precision='NOT_ESTABLISHED')
            return dict(selected_ids=[],weights=[],gradient=torch.zeros(self.shape,dtype=torch.float64),
                        L_H=0.,receipt=receipt)
        device_weight = weight.to(self.device)
        selection = bool(select_gss and len(ids)>CAPACITY)
        factors, losses, sketches, diag_factors = {}, [], [], {}
        timings = dict(pool_forward_seconds=0., NLL_VJP_seconds=0., KL_VJP_seconds=0.,
                       activation_D2H_seconds=0., sketch_and_key_map_seconds=0.,
                       pruning_seconds=0., selected_gradient_assembly_seconds=0.)
        diagnostic_ids = set(sorted(ids, key=lambda i: REFERENCE.priority(i,self.seed))[:64]) if selection and not self.sketch_diagnostic_done else set()
        activation_bytes = 0
        for identity, row in zip(ids,rows):
            item = self._load(identity)
            if (item['metadata']['created_batch'] != int(row['created_batch']) or
                item['metadata']['case_id'] != row['record']['case_id']):
                raise ValueError('POOL_VERSION_BIRTH_OR_CANONICAL_RECORD_IDENTITY')
            phase = time.monotonic()
            cut, logits = self._activation_logits(item, device_weight)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            timings['pool_forward_seconds'] += time.monotonic()-phase
            labels = torch.tensor(item['metadata']['labels'], device=self.device)
            value, nll, kl = activation_vjps(logits, cut, labels, item['teacher'].to(self.device), selection=selection, timings=timings)
            valid = item['cache'].packed['attention_mask'].to(self.device).bool()
            phase = time.monotonic()
            a_kl = kl[valid].T.contiguous().cpu()
            timings['activation_D2H_seconds'] += time.monotonic()-phase
            factors[identity] = a_kl
            activation_bytes += a_kl.numel()*a_kl.element_size()
            losses.append(value)
            if selection:
                phase = time.monotonic()
                a_nll = nll[valid].T.contiguous()
                sketches.append(sketch_factors(a_nll, self._projected_keys(identity,item['cache']), self.map_tensors).cpu().numpy())
                if identity in diagnostic_ids:
                    diag_factors[identity] = a_nll.cpu()
                del a_nll
                timings['sketch_and_key_map_seconds'] += time.monotonic()-phase
            del cut, logits, labels, nll, kl
        phase = time.monotonic()
        pruning = REFERENCE.gss_prune(np.asarray(sketches), ids, cap=CAPACITY, seed=self.seed) if selection else dict(selected=list(range(len(ids))), removed=[], trace=[], selection_exercised=False)
        timings['pruning_seconds'] = time.monotonic()-phase
        selected = [ids[i] for i in pruning['selected']]
        selected_rows = [rows[i] for i in pruning['selected']]
        probabilities = REFERENCE.history_weights([r['created_batch'] for r in selected_rows], current_batch, recency=recency)
        phase = time.monotonic()
        aggregate = aggregate_factors([factors[i] for i in selected],
                                      [self._load(i)['cache'].valid_keys() for i in selected],
                                      probabilities, self.shape, self.device)
        gradient = aggregate.cpu()
        timings['selected_gradient_assembly_seconds'] = time.monotonic()-phase
        native_loss = float(sum(probabilities[j]*losses[i] for j,i in enumerate(pruning['selected'])))
        diagnostics = {}
        if len(selected)>=4 and not self.factor_diagnostic_done:
            diagnostics['factor'] = self._factor_diagnostic(device_weight, selected, probabilities, factors)
        if diagnostic_ids:
            diagnostics['sketch'] = self._sketch_diagnostic(diag_factors, sketches, ids)
        self.counts['pool_forwards'] += len(ids)
        self.counts['NLL_VJPs'] += len(ids) if selection else 0
        self.counts['KL_VJPs'] += len(ids)
        self.counts['gradient_D2H'] += 1
        self.counts['activation_D2H_bytes'] += activation_bytes
        receipt = dict(pool_size=len(ids), pool_ids=ids, selected_ids=selected,
                       weights=probabilities.tolist(), L_H=native_loss,
                       recency=bool(recency), current_batch=int(current_batch),
                       created_batches=[r['created_batch'] for r in selected_rows],
                       selection_exercised=selection, NLL_VJPs=len(ids) if selection else 0,
                       KL_VJPs=len(ids), microbatch_size=1, forward_graphs=len(ids),
                       two_VJPs_share_graph=selection, microbatch_extra_mean=False,
                       activation_host_bytes=activation_bytes, dense_gradient_D2H=1,
                       per_fact_dense_gradient=False, hot_cache_versions=len(self.hot),
                       pruning=pruning, diagnostics=diagnostics, timings=timings,
                       seconds=time.monotonic()-started,
                       precision='NOT_ESTABLISHED')
        return dict(selected_ids=selected, weights=probabilities.tolist(), gradient=gradient,
                    L_H=native_loss, receipt=receipt)

    def evaluate(self, weight, selected_ids, weights):
        start = time.monotonic()
        ids, probabilities = list(selected_ids), np.asarray(weights,dtype=np.float64)
        if len(ids)>CAPACITY or len(ids)!=len(set(ids)) or probabilities.shape!=(len(ids),):
            raise ValueError('FROZEN_CANDIDATE_HISTORY_BANK_SHAPE')
        if len(ids) and (not np.all(np.isfinite(probabilities)) or np.any(probabilities<0) or abs(probabilities.sum()-1)>1e-12):
            raise ValueError('NORMALIZED_FIXED_HISTORY_WEIGHTS_REQUIRED')
        rows = []
        device_weight = weight.to(self.device)
        with torch.no_grad():
            for identity in ids:
                item = self._load(identity)
                oracle = self._use_cache(item['cache'])
                hidden = oracle.suffix_hidden(0, device_weight)
                logits = oracle._head(hidden, torch.tensor(item['metadata']['positions']))
                value = float(signed_forward_kl(logits,item['teacher'].to(self.device)))
                rows.append(dict(version_id=identity, loss=value, positions=len(item['metadata']['positions'])))
        value = float(sum(w*r['loss'] for w,r in zip(probabilities,rows)))
        self.counts['candidate_forwards'] += len(ids)
        return value, dict(L_H=value, selected_ids=ids, weights=probabilities.tolist(),
                           rows=rows, candidate_fact_forwards=len(ids),
                           seconds=time.monotonic()-start, bank_teacher_weights_fixed=True)
