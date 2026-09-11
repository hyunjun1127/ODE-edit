"""Read-only native keys and same-entry reconstructed history.

No model loader, target optimizer, or process/environment setup lives here.
The caller binds CUMRISK_ASSET_ROOT before importing its native kernel and
validates the supplied entry/source identities. Reconstruction retains ALL
chronological raw requests, including retired facts. FP32 CPU ``M += K K.T``
uses the original native B100 append grouping, not physical forward chunks.
"""
import copy
import importlib
import json
from pathlib import Path
import stat

import torch

from .contracts import canonical, digest, sha, save, tensor_save, tensor_sha
from project.run_scripts.single_layer_cumulative_risk.microbatch import bounded_reader


class HistoryBoundary(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise HistoryBoundary(message)


def _directory(path):
    """Reject symlink components before creating any local artifact."""
    p = Path(path).absolute()
    for part in [*reversed(p.parents), p]:
        if part.exists() or part.is_symlink():
            _require(stat.S_ISDIR(part.lstat().st_mode), f'UNSAFE_DIRECTORY: {part}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path):
    _require(stat.S_ISREG(path.lstat().st_mode), f'NONREGULAR_RECEIPT: {path}')
    return json.loads(path.read_text())


def _model_guard(model):
    """Cheap all-parameter/buffer pointer/version guard, not a new byte seal."""
    return tuple((kind, name, id(t), t.data_ptr(), t._version, tuple(t.shape),
                  str(t.dtype), str(t.device))
                 for kind, iterator in [('parameter', model.named_parameters()),
                                        ('buffer', model.named_buffers())]
                 for name, t in iterator)


def _requests(records):
    return [dict(copy.deepcopy(r['requested_rewrite']), case_id=r['case_id']) for r in records]


def capture_native_keys(model, tok, native, hp, contexts, records, layer, ledger,
                        *, input_width, physical_batch=8, repr_module=None):
    """Return FP32 CPU K[din,B]; unchanged compute_ks context-type averaging.

    This scoped monkeypatch only bounds representation *forward* batches. It
    must run in one task-owned process, not concurrently with another reader.
    No native wrapper/solve/compute_z/history finalizer is invoked.
    """
    _require(physical_batch > 0 and input_width > 0, 'INVALID_KEY_DIMENSION')
    _require(getattr(tok, 'padding_side', None) == 'right', 'TOKENIZER_NOT_RIGHT_PADDED')
    _require(not model.training, 'MODEL_NOT_EVAL')
    _require(bool(contexts) and all(contexts), 'EMPTY_NATIVE_CONTEXT_TYPE')
    if not records:
        return torch.empty((input_width, 0), dtype=torch.float32)
    state = _model_guard(model)
    record_hash, context_hash = digest(records), digest(contexts)
    reader = repr_module if repr_module is not None else importlib.import_module('rome.repr_tools')
    original = reader.get_reprs_at_idxs
    reader.get_reprs_at_idxs = bounded_reader(original, ledger, physical_batch)
    try:
        with torch.no_grad(), ledger.time('native_key_capture'):
            result = native.compute_ks(model, tok, _requests(records), hp, layer, copy.deepcopy(contexts))
        ledger.add('native_key_capture')
        ledger.add('native_key_requests', len(records))
        ledger.add('native_key_context_rows', len(records) * sum(map(len, contexts)))
        _require(result.shape == (len(records), input_width), 'NATIVE_KEY_SHAPE')
        _require(result.dtype == torch.float32 and bool(torch.isfinite(result).all()), 'NATIVE_KEY_DTYPE_NONFINITE')
        return result.detach().T.to(device='cpu').contiguous().clone()
    finally:
        reader.get_reprs_at_idxs = original
        _require(_model_guard(model) == state, 'NATIVE_KEY_MUTATED_LIVE_STATE')
        _require(digest(records) == record_hash and digest(contexts) == context_hash,
                 'NATIVE_KEY_MUTATED_INPUT')


def _key_member(path, key):
    return dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path),
                tensor_sha256=tensor_sha(key), shape=list(key.shape), dtype=str(key.dtype))


def _load_key(path, member, shape):
    _require(path.name == member['name'] and stat.S_ISREG(path.lstat().st_mode), 'KEY_MEMBER_PATH')
    _require(path.stat().st_size == member['bytes'] and sha(path) == member['sha256'], 'KEY_MEMBER_HASH')
    key = torch.load(path, map_location='cpu', weights_only=True)
    _require(isinstance(key, torch.Tensor) and list(key.shape) == list(shape)
             and key.dtype == torch.float32 and bool(torch.isfinite(key).all()), 'KEY_MEMBER_SCHEMA')
    _require(list(key.shape) == member['shape'] and str(key.dtype) == member['dtype']
             and tensor_sha(key) == member['tensor_sha256'], 'KEY_TENSOR_HASH')
    return key


def reconstruct_history(model, tok, native, hp, contexts, records, offset, ledger, out,
                        *, entry_identity, source_identity, input_width, layer=8,
                        physical_batch=8, capture=None, verify_entry=None, progress=None):
    """Build same-We M8 (or missing native baseline layer) without touching M4.

    ``records[:offset]`` is the unfiltered sealed raw stream. ``capture`` is a
    CPU-fixture seam with signature ``capture(batch_records)->K[din,B]``;
    production uses capture_native_keys. Completed key+receipt pairs are
    create-once resumable checkpoints. An unreceipted/orphan/partial file is
    never silently consumed or overwritten. Resume replays CPU Gram additions
    from verified keys, avoiding every completed model forward.

    Entry/source identity are caller-validated hash dictionaries. Optional
    verify_entry rechecks actual binding before any capture (including resume).
    progress(event_dict) runs after each SHA-sealed reused/new key chunk and
    finite Gram accumulation; it is suitable for task-local first-valid output.
    Returned M is newly allocated; this API never receives or mutates M4.
    """
    _require(layer in (5, 6, 7, 8), 'ORIGINAL_M4_RECONSTRUCTION_FORBIDDEN')
    _require(0 <= offset <= len(records) and offset % 100 == 0, 'HISTORY_NOT_CHRONOLOGICAL_B100')
    _require(input_width > 0 and physical_batch > 0, 'INVALID_HISTORY_DIMENSION')
    _require(bool(entry_identity) and bool(source_identity), 'UNBOUND_HISTORY_SOURCE_ENTRY')
    if verify_entry is not None:
        verify_entry()
    out = _directory(out)
    lock = dict(schema='same-entry-native-history-v1', layer=layer, input_width=input_width,
                offset=offset, append_batch_size=100, physical_batch=physical_batch,
                gram_device='cpu', gram_dtype='torch.float32', operator='chronological M += K @ K.T',
                entry_identity=entry_identity, source_identity=source_identity,
                request_sha256=[digest(r) for r in records[:offset]],
                case_ids=[r['case_id'] for r in records[:offset]], context_sha256=digest(contexts),
                rewrite_module_tmp=hp.rewrite_module_tmp, fact_token=hp.fact_token,
                selection='ALL_PRIOR_RAW_REQUESTS_NO_ACTIVE_FILTER',
                origin='reconstructed at the fixed current entry; not historical original M8',
                original_M4_unchanged=True)
    lock_path = out / 'input.lock.json'
    if lock_path.exists() or lock_path.is_symlink():
        _require(_read_json(lock_path) == lock, 'HISTORY_RESUME_INPUT_MISMATCH')
    else:
        _require(not any(out.iterdir()), 'UNBOUND_EXISTING_HISTORY_NAMESPACE')
        save(lock_path, lock)
    lock_sha = sha(lock_path)
    state = _model_guard(model)
    gram = torch.zeros((input_width, input_width), dtype=torch.float32)
    chunks, previous = [], digest(lock)
    encountered_new = False
    # Reject a hole before performing any new model work. An unfinished first
    # missing chunk is fine; later completed or orphan chunks are not.
    missing = False
    for start in range(0, offset, 100):
        stem = f'batch-{start // 100 + 1:03d}'
        kp, rp = out / (stem + '-keys.pt'), out / (stem + '.json')
        if rp.exists() or rp.is_symlink():
            _require(not missing, 'NONCONTIGUOUS_FINISHED_HISTORY')
        else:
            missing = True
            _require(not kp.exists() and not kp.is_symlink()
                     and not kp.with_name(kp.name + '.partial').exists(), 'UNRECEIPTED_HISTORY_KEY_PRESERVED')
    for start in range(0, offset, 100):
        batch_records = records[start:start + 100]
        stem = f'batch-{start // 100 + 1:03d}'
        kp, rp = out / (stem + '-keys.pt'), out / (stem + '.json')
        expected = dict(input_lock_sha256=lock_sha, batch_index=start // 100 + 1,
                        start_ordinal=start, stop_ordinal=start + 100,
                        requests_sha256=digest(batch_records), context_sha256=digest(contexts),
                        entry_identity=entry_identity, previous_receipt_identity=previous)
        if rp.exists() or rp.is_symlink():
            reused = True
            _require(not encountered_new, 'NONCONTIGUOUS_FINISHED_HISTORY')
            receipt = _read_json(rp)
            _require(all(receipt.get(k) == v for k, v in expected.items()), 'HISTORY_CHUNK_IDENTITY')
            unsigned = {k: v for k, v in receipt.items() if k != 'identity'}
            _require(receipt.get('identity') == digest(unsigned), 'HISTORY_RECEIPT_IDENTITY')
            key = _load_key(kp, receipt['key'], (input_width, 100))
            ledger.add('history_reused_requests', 100)
        else:
            reused = False
            encountered_new = True
            _require(not kp.exists() and not kp.is_symlink()
                     and not kp.with_name(kp.name + '.partial').exists(), 'UNRECEIPTED_HISTORY_KEY_PRESERVED')
            if verify_entry is not None:
                verify_entry()
            key = (capture(batch_records) if capture is not None else
                   capture_native_keys(model, tok, native, hp, contexts, batch_records, layer, ledger,
                                       input_width=input_width, physical_batch=physical_batch))
            _require(key.device.type == 'cpu' and key.dtype == torch.float32
                     and key.shape == (input_width, 100) and bool(torch.isfinite(key).all()),
                     'HISTORY_CAPTURE_SCHEMA')
            _require(_model_guard(model) == state, 'HISTORY_ENTRY_STATE_CHANGED')
            tensor_save(kp, key)
            receipt = dict(expected, key=_key_member(kp, key), completed_requests=100,
                           history_mutation_count=0, actual_history_append_count=0,
                           ledger=ledger.receipt())
            receipt['identity'] = digest(receipt)
            save(rp, receipt)
            ledger.add('history_new_requests', 100)
        # Native AlphaEdit_main appends this exact CPU FP32 Gram once per batch.
        gram += key @ key.T
        _require(bool(torch.isfinite(gram).all()), 'NONFINITE_HISTORY_GRAM')
        chunks.append(dict(receipt=rp.name, sha256=sha(rp), identity=receipt['identity']))
        previous = receipt['identity']
        if progress is not None:
            progress(dict(stage='same-entry-history', layer=layer, completed_requests=start + 100,
                          total_requests=offset, completed_batches=start // 100 + 1,
                          reused=reused, receipt_path=str(rp), receipt_sha256=sha(rp),
                          receipt_identity=receipt['identity'], key_shape=list(key.shape),
                          gram_finite=True, source_state='SAME_WE_RECONSTRUCTED'))
    _require(_model_guard(model) == state, 'HISTORY_ENTRY_STATE_CHANGED')
    final = dict(schema='same-entry-native-history-complete-v1', input_lock_sha256=lock_sha,
                 completed_requests=offset, completed_batches=offset // 100, layer=layer,
                 chunks=chunks, chunks_root=digest(chunks), shape=list(gram.shape), dtype=str(gram.dtype),
                 gram_sha256=tensor_sha(gram), source_state='SAME_WE_RECONSTRUCTED',
                 original_M4_unchanged=True, production_history_append_count=0)
    final['identity'] = digest(final)
    final_path = out / 'complete.json'
    if final_path.exists() or final_path.is_symlink():
        _require(_read_json(final_path) == final, 'HISTORY_COMPLETE_MISMATCH')
    else:
        save(final_path, final)
    return gram, final


def finalize_history_once(model, tok, native, hp, contexts, records, histories, ledger, out,
                          *, endpoint_identity, source_identity, physical_batch=8,
                          capture=None, verify_endpoint=None):
    """Return fresh finalized histories; the supplied histories/model stay read-only.

    Caller invokes this only at its accepted endpoint and owns a unique out
    marker. A second invocation (even after failure) rejects. A partial attempt
    remains immutable and needs an explicitly distinct technical namespace.
    ``capture(layer, records)`` is a CPU fixture seam.
    """
    _require(bool(endpoint_identity) and bool(source_identity), 'UNBOUND_FINALIZATION')
    out = _directory(out)
    _require(not any(out.iterdir()), 'FINALIZATION_ALREADY_ATTEMPTED')
    if verify_endpoint is not None:
        verify_endpoint()
    before = _model_guard(model)
    inputs = {str(layer): tensor_sha(m) for layer, m in histories.items()}
    save(out / 'attempt.json', dict(endpoint_identity=endpoint_identity, source_identity=source_identity,
         request_sha256=digest(records), context_sha256=digest(contexts), history_sha256=inputs))
    states, members = {}, []
    for layer, m in histories.items():
        _require(m.ndim == 2 and m.shape[0] == m.shape[1] and m.dtype == torch.float32
                 and bool(torch.isfinite(m).all()), 'FINAL_HISTORY_INPUT_SCHEMA')
        key = (capture(layer, records) if capture is not None else capture_native_keys(
            model, tok, native, hp, contexts, records, layer, ledger,
            input_width=m.shape[0], physical_batch=physical_batch))
        _require(key.shape == (m.shape[0], len(records)) and key.device.type == 'cpu'
                 and key.dtype == torch.float32 and bool(torch.isfinite(key).all()), 'FINAL_KEY_SCHEMA')
        new = m.detach().to('cpu').clone()
        if records:
            new += key @ key.T
            ledger.add('terminal_history_layer_appends')
        _require(bool(torch.isfinite(new).all()), 'FINAL_HISTORY_NONFINITE')
        states[layer] = new
        path = out / f'layer-{layer}-keys.pt'
        tensor_save(path, key)
        members.append(dict(layer=layer, key=_key_member(path, key), final_history_sha256=tensor_sha(new)))
    _require(_model_guard(model) == before, 'FINALIZATION_MUTATED_LIVE_STATE')
    _require({str(l): tensor_sha(m) for l, m in histories.items()} == inputs, 'FINALIZATION_MUTATED_INPUT_HISTORY')
    receipt = dict(status='FINALIZED_KEYS_CAPTURED_INPUT_HISTORY_UNCHANGED',
                   endpoint_identity=endpoint_identity, source_identity=source_identity,
                   members=members, completed_requests=len(records),
                   terminal_batch_finalizations=int(bool(records)),
                   terminal_layer_appends=len(histories) if records else 0,
                   inner_history_appends=0, ledger=ledger.receipt())
    receipt['identity'] = digest(receipt)
    save(out / 'complete.json', receipt)
    return states, receipt
