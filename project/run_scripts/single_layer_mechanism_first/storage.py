"""CPU native scientific evidence, explicitly excluding resume equivalents.

The actual CapturedHookedFitter adds ``captures.solve_update`` to the ordinary
NativeSingletonFitter result. Dropping only root ``weight`` is insufficient:
that solve tensor is a full writer update. This helper removes it recursively,
along with W/M/RNG/optimizer states and full endpoint/delta payloads. Native z,
keys, readouts, small z optimization vectors and source receipts remain usable
scientific evidence. It performs no file I/O, model work or checkpoint restore.

This is an explicit serialization boundary for the source-bound native result,
not a claim that allowed target/key evidence cannot support later scientific
recomputation. No exact endpoint reconstruction or crash-resume is established.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re

import numpy as np
import torch


class StorageBoundary(ValueError):
    pass


_POLICY_KEY = 'native_evidence_storage_policy'
_STATE_NAMES = frozenset({
    'weight', 'weights', 'w', 'w0', 'wn', 'w4', 'w8', 'm', 'm0', 'm4', 'm8',
    'model', 'modelstate', 'modelstatedict', 'statedict', 'memorystate',
    'memory', 'memories', 'history', 'historystate', 'cachec',
    'rng', 'rngstate', 'rngstates', 'randomstate', 'pythonrngstate',
    'numpyrngstate', 'torchrngstate', 'cpurngstate', 'cudarngstate',
    'optimizer', 'optimizerstate', 'optimizerstatedict', 'schedulerstate',
    'checkpoint', 'checkpointstate', 'resumestate', 'resumecheckpoint',
    'endpoint', 'selectedendpoint', 'nativeendpoint', 'finalendpoint',
    'entryweight', 'selectedweight', 'nativeweight', 'finalweight', 'endpointweight',
    'entryweights', 'selectedweights', 'nativeweights', 'finalweights',
    'solveupdate', 'solveupdates', 'nativeupdate', 'nativeupdates',
    'weightupdate', 'weightupdates', 'fullupdate', 'fullupdates',
    'actualdelta', 'idealdelta', 'fulldelta', 'weightdelta', 'deltaweight',
    'nativedelta', 'deltanative', 'accumulateddelta', 'deltaacc', 'dacc',
    'dacc64', 'actuald', 'ideald', 'fullmodel',
})
_SCIENTIFIC_MATRIX_NAMES = frozenset({
    'gradient', 'gradients', 'rawgradient', 'projectedgradient',
    'neuralgradient', 'activationgradient', 'activationgradients',
    'g', 'gq', 'h', 'jacobian', 'computez', 'computeks',
    'getmoduleinputoutputatwords', 'target', 'targets', 'anchor', 'anchors',
    'keys', 'k', 'currentoutput', 'residual', 'r',
})


def _name(key):
    return re.sub('[^a-z0-9]', '', str(key).lower())


def _primitive(value):
    return value is None or isinstance(value, (str, bool, int, float, np.generic))


def _tensor_like(value):
    return isinstance(value, (torch.Tensor, np.ndarray))


def _cpu(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided:
            raise StorageBoundary('NATIVE_EVIDENCE_CPU_STRIDED_TENSOR_REQUIRED')
    elif value.dtype.hasobject:
        raise StorageBoundary('NATIVE_EVIDENCE_OBJECT_ARRAY_FORBIDDEN')


def _identity(value):
    """Header+bytes SHA streamed by first dimension; no device transfer."""
    _cpu(value)
    shape = list(value.shape)
    dtype = str(value.dtype)
    count = value.numel() if isinstance(value, torch.Tensor) else value.size
    width = value.element_size() if isinstance(value, torch.Tensor) else value.itemsize
    h = hashlib.sha256(json.dumps(dict(shape=shape, dtype=dtype), sort_keys=True,
                                  separators=(',', ':')).encode() + b'\n')
    blocks = [value] if not shape else (value[i:i+64] for i in range(0, shape[0], 64))
    for block in blocks:
        if isinstance(block, torch.Tensor):
            # uint8 view also supports CPU bfloat16 without numpy conversion of
            # the floating dtype; hashing does not round or normalize bytes.
            h.update(block.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(np.ascontiguousarray(block).tobytes())
    return dict(shape=shape, dtype=dtype, logical_bytes=int(count*width),
                sha256_header_and_bytes=h.hexdigest())


def _summary(value):
    """Excluded-member receipt containing identities, never tensor payloads."""
    if _tensor_like(value):
        return dict(type=type(value).__name__, **_identity(value))
    if isinstance(value, Mapping):
        return dict(type='mapping', members={str(k): _summary(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, members=[_summary(v) for v in value])
    if _primitive(value):
        # Do not preserve RNG/optimizer integer lists as an accidental resume
        # payload. A scalar's JSON hash suffices inside an excluded subtree.
        raw = value.item() if isinstance(value, np.generic) else value
        return dict(type=type(raw).__name__, sha256=json_hash(raw))
    raise StorageBoundary('UNSUPPORTED_EXCLUDED_NATIVE_VALUE:' + type(value).__name__)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=True).encode()).hexdigest()


def _torch_storage(value):
    if isinstance(value, torch.Tensor):
        _cpu(value)
        return ('torch', value.untyped_storage().data_ptr()) if value.numel() else None
    if isinstance(value, np.ndarray):
        _cpu(value)
        base = value
        while isinstance(base.base, np.ndarray):
            base = base.base
        return ('numpy', int(base.__array_interface__['data'][0])) if value.size else None
    return None


def native_evidence(result):
    """Return an owned CPU payload plus scalar exclusion receipt, without W/M.

Input dictionaries and tensors are not mutated. Retained tensors are cloned
so later runtime updates do not mutate evidence awaiting atomic serialization.
Small ``target_observations[*].delta`` and hook z-batch ``deltas`` are activation
optimization evidence, not writer-weight deltas, and are retained explicitly.
Scientific gradients are not treated as checkpoint state merely by their size.
"""
    if not isinstance(result, Mapping) or 'weight' not in result or 'receipt' not in result:
        raise StorageBoundary('NATIVE_RESULT_SCHEMA_REQUIRED')
    if _POLICY_KEY in result:
        raise StorageBoundary('NATIVE_EVIDENCE_ALREADY_FILTERED')
    weight = result['weight']
    if not _tensor_like(weight) or len(weight.shape) != 2:
        raise StorageBoundary('NATIVE_WRITER_MATRIX_REQUIRED')
    _cpu(weight)
    writer_shape = tuple(weight.shape)
    writer_elements = int(np.prod(writer_shape))
    forbidden_storage = set()
    excluded = []

    def small_z(path, value):
        if not _tensor_like(value):
            return False
        shape = tuple(value.shape)
        return (('target_observations' in path and shape == (writer_shape[0],)) or
                ('z_hook' in path and 'batches' in path and len(shape) == 2 and
                 shape[-1] == writer_shape[0] and shape[0] <= 100 and
                 int(np.prod(shape)) < writer_elements))

    def reason(path, value):
        key = _name(path[-1]) if path else ''
        # Hashes, source names and scalar norms remain useful provenance even
        # when named W/M/rng. Structured state under that name is prohibited.
        if not _primitive(value) and key in _STATE_NAMES:
            return 'RESUME_OR_FULL_WRITER_STATE_FIELD'
        if key in ('delta', 'deltas', 'd', 'updates') and not _primitive(value) and not small_z(path, value):
            return 'WEIGHT_DELTA_OR_UNCLASSIFIED_UPDATE'
        if _tensor_like(value):
            _cpu(value)
            pointer = _torch_storage(value)
            if pointer is not None and pointer in forbidden_storage:
                return 'ALIASED_EXCLUDED_STATE_STORAGE'
            count = value.numel() if isinstance(value, torch.Tensor) else value.size
            if count == writer_elements and not any(_name(p) in _SCIENTIFIC_MATRIX_NAMES for p in path):
                return 'UNCLASSIFIED_FULL_WRITER_SIZED_TENSOR'
        return None

    def collect_forbidden(value, path=()):
        if reason(path, value):
            def pointers(v):
                if _tensor_like(v):
                    pointer = _torch_storage(v)
                    if pointer is not None:
                        forbidden_storage.add(pointer)
                elif isinstance(v, Mapping):
                    for x in v.values(): pointers(x)
                elif isinstance(v, (list, tuple)):
                    for x in v: pointers(x)
            pointers(value)
            return
        if isinstance(value, Mapping):
            for k, v in value.items(): collect_forbidden(v, path+(str(k),))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value): collect_forbidden(v, path+(str(i),))

    collect_forbidden(result)

    def filtered(value, path=()):
        why = reason(path, value)
        if why:
            excluded.append(dict(path='/'.join(path), reason=why, identity=_summary(value)))
            return None, False
        if _tensor_like(value):
            _cpu(value)
            return (value.detach().clone() if isinstance(value, torch.Tensor) else value.copy()), True
        if isinstance(value, Mapping):
            out = {}
            for k, v in value.items():
                if not isinstance(k, (str, int)):
                    raise StorageBoundary('NATIVE_EVIDENCE_MAPPING_KEY')
                kept, included = filtered(v, path+(str(k),))
                if included: out[k] = kept
            return out, True
        if isinstance(value, (list, tuple)):
            out = []
            for i, v in enumerate(value):
                kept, included = filtered(v, path+(str(i),))
                if included: out.append(kept)
            return (tuple(out) if isinstance(value, tuple) else out), True
        if _primitive(value):
            return (value.item() if isinstance(value, np.generic) else value), True
        raise StorageBoundary('UNSUPPORTED_NATIVE_EVIDENCE_VALUE:' + type(value).__name__)

    payload, included = filtered(result)
    if not included:
        raise StorageBoundary('NATIVE_RESULT_ROOT_EXCLUDED')
    payload[_POLICY_KEY] = dict(schema='native-no-checkpoint-evidence-v1',
        disk_checkpoint='SKIPPED_USER_DIRECTED', endpoint_weight_saved=False,
        full_solve_update_saved=False, W_M_RNG_optimizer_state_saved=False,
        exact_resume_from_payload=False, endpoint_reconstruction='NOT_ESTABLISHED',
        retained_scope='native targets, keys, current readouts, small z vectors and source/scalar receipts',
        original_result_mutated=False, writer_shape=list(writer_shape), excluded=excluded,
        excluded_field_paths=[r['path'] for r in excluded])
    return payload
