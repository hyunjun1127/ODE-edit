"""Read-only observation of the pinned BLUE dense native singleton writer.

The module-local torch proxy mirrors the historical observer. Actual K/R are
read from the original apply frame, not reconstructed by a replacement solve.
No tracing, global torch changes, optimizer hooks, or RNG calls are introduced.
Tensor members are local raw artifacts, not JSON/Git publication members.
"""
from contextlib import contextmanager
import sys
import time

import torch

from .fixtures import FixtureBoundary, tensor_sha


class _Proxy:
    def __init__(self, original, **overrides):
        self._original, self._overrides = original, overrides

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._original, name)


def _cpu(value):
    return value.detach().cpu().clone()


def _identity(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))


@contextmanager
def observe_native(module, hp, weights, state, requests, model, *, on_event=None):
    """Context factory accepted directly by native_runner.run_native_batch.

    Native compute_z internals are not public results: final training NLL,
    iterations, stop reason and clamp flag remain NOT_OBSERVED. Target tensors
    themselves are captured. Observer timing is host wall time, not CUDA-synced
    kernel timing; additional copies/history verification are separately timed.
    """
    names = ('torch', 'compute_z', 'compute_ks', 'upd_matrix_match_shape')
    if len(hp.layers) != 1 or not hp.blue:
        raise FixtureBoundary('OBSERVER_SINGLETON_BLUE_ONLY')
    if getattr(module, '_e01_observer_active', False):
        raise FixtureBoundary('OBSERVER_ALREADY_ACTIVE')
    original = {name: getattr(module, name) for name in names}
    apply_code = module.apply_AlphaEdit_to_model.__code__
    start_history = _cpu(state)
    receipt = dict(status='OBSERVING', z=[], keys=[], solves=[], physical_updates=[],
                   compute_z=0, solve_calls=0, key_calls=0,
                   history_append_passes=0, target_seconds=0., key_seconds=0., solve_seconds=0.,
                   observer_copy_seconds=0., history_verification_seconds=0.,
                   timing_policy='HOST_WALL_WITH_COPY_SYNCHRONIZATION_NOT_CUDA_KERNEL_TIME',
                   compute_z_final_training_nll='NOT_OBSERVED',
                   compute_z_iterations='NOT_OBSERVED', compute_z_stop_reason='NOT_OBSERVED',
                   compute_z_clamp='NOT_OBSERVED',
                   _tensors=dict(targets=[], keys=[], solves=[], physical_updates=[]))

    def emit(label, row):
        if on_event is not None:
            on_event(label, row)

    def target(*args, **kwargs):
        started = time.monotonic()
        value = original['compute_z'](*args, **kwargs)
        receipt['target_seconds'] += time.monotonic() - started
        copied = time.monotonic()
        saved = _cpu(value)
        request = args[2] if len(args) > 2 else kwargs['request']
        layer = args[4] if len(args) > 4 else kwargs['layer']
        row = dict(case_id=request['case_id'], layer=layer, **_identity(saved))
        receipt['z'].append(row); receipt['_tensors']['targets'].append(saved)
        receipt['compute_z'] += 1
        receipt['observer_copy_seconds'] += time.monotonic() - copied
        emit('NATIVE_TARGET_RETURNED', row)
        return value

    def keys(*args, **kwargs):
        started = time.monotonic()
        value = original['compute_ks'](*args, **kwargs)
        receipt['key_seconds'] += time.monotonic() - started
        copied = time.monotonic()
        saved = _cpu(value)
        layer = args[4] if len(args) > 4 else kwargs['layer']
        phase = 'writer' if receipt['key_calls'] == 0 else 'post_write_history'
        row = dict(layer=layer, phase=phase, **_identity(saved))
        receipt['keys'].append(row); receipt['_tensors']['keys'].append(saved)
        receipt['key_calls'] += 1
        receipt['observer_copy_seconds'] += time.monotonic() - copied
        emit('NATIVE_KEY_RETURNED', row)
        return value

    def solve(matrix, rhs, *args, **kwargs):
        # Native direct dense call remains the only source of its solution.
        frame = sys._getframe(1)
        if frame.f_code is not apply_code:
            raise FixtureBoundary('OBSERVER_SOLVE_CALLSITE_UNVERIFIED')
        local = frame.f_locals
        if not {'layer_ks', 'resid', 'layer'}.issubset(local):
            raise FixtureBoundary('OBSERVER_NATIVE_LOCALS_UNVERIFIED')
        actual_key, actual_residual, layer = local['layer_ks'], local['resid'], local['layer']
        del frame, local
        copied = time.monotonic()
        saved_key, saved_residual, saved_rhs = _cpu(actual_key), _cpu(actual_residual), _cpu(rhs)
        # Matrix is potentially d_in^2: only its hash/schema, not a second raw
        # copy, is retained. K/R/RHS and original returned dense solve suffice.
        matrix_identity = _identity(matrix)
        receipt['observer_copy_seconds'] += time.monotonic() - copied
        started = time.monotonic()
        value = original['torch'].linalg.solve(matrix, rhs, *args, **kwargs)
        receipt['solve_seconds'] += time.monotonic() - started
        copied = time.monotonic()
        saved_output = _cpu(value)
        row = dict(layer=layer, matrix=matrix_identity, rhs=_identity(saved_rhs),
                   actual_K=_identity(saved_key), actual_R=_identity(saved_residual),
                   returned_dense_solution=_identity(saved_output),
                   capture='PINNED_NATIVE_APPLY_FRAME_ACTUAL_TENSORS')
        receipt['solves'].append(row)
        receipt['_tensors']['solves'].append(dict(K=saved_key, R=saved_residual,
                                                  rhs=saved_rhs, solution=saved_output))
        receipt['solve_calls'] += 1
        receipt['observer_copy_seconds'] += time.monotonic() - copied
        emit('NATIVE_DENSE_SOLVE_RETURNED', row)
        return value

    def orient(value, shape):
        result = original['upd_matrix_match_shape'](value, shape)
        copied = time.monotonic()
        saved = _cpu(result)
        row = _identity(saved)
        receipt['physical_updates'].append(row)
        receipt['_tensors']['physical_updates'].append(saved)
        receipt['observer_copy_seconds'] += time.monotonic() - copied
        return result

    module._e01_observer_active = True
    module.compute_z, module.compute_ks = target, keys
    module.upd_matrix_match_shape = orient
    module.torch = _Proxy(original['torch'], linalg=_Proxy(original['torch'].linalg, solve=solve))
    try:
        yield receipt
        expected = [(hp.layers[0], row['case_id']) for row in requests]
        if [(r['layer'], r['case_id']) for r in receipt['z']] != expected:
            raise FixtureBoundary('OBSERVER_Z_ORDER_OR_COUNT')
        if not requests:
            if receipt['key_calls'] or receipt['solve_calls']:
                raise FixtureBoundary('OBSERVER_EMPTY_BATCH_CALLS')
            exact = tensor_sha(state) == tensor_sha(start_history)
        else:
            if (receipt['solve_calls'] != 1 or receipt['key_calls'] != 2 or
                    len(receipt['physical_updates']) != 1 or
                    [r['layer'] for r in receipt['keys']] != hp.layers * 2):
                raise FixtureBoundary('OBSERVER_NATIVE_CALL_COUNTS')
            started = time.monotonic()
            # Original source transposes returned [request,d_in] keys, then
            # appends CPU float32 Gram exactly once after the physical write.
            post_key = receipt['_tensors']['keys'][1].T
            expected_history = start_history.clone()
            expected_history[0] += post_key @ post_key.T
            exact = torch.equal(state.detach().cpu(), expected_history)
            receipt['history_verification_seconds'] = time.monotonic() - started
            receipt['history_append_passes'] = 1
            receipt['history_verification_extra_cpu_gram_calls'] = 1
        receipt.update(status='OBSERVED_NATIVE_VALID' if exact else 'HISTORY_APPEND_MISMATCH',
                       history_append_exact=bool(exact), history_endpoint=_identity(state),
                       projector_physical_layers=list(hp.layers),
                       projector_asset_indices=[hp.layers[0] - 4], projector_local_indices=[0],
                       global_torch_modified=False, native_inputs_replaced=False,
                       observer_rng_calls=0)
        if not exact:
            raise FixtureBoundary('OBSERVER_HISTORY_APPEND_MISMATCH')
        emit('NATIVE_HISTORY_APPEND_VERIFIED', {k: v for k, v in receipt.items() if k != '_tensors'})
    except BaseException as exc:
        receipt.update(status='OBSERVATION_FAILED', original_exception=repr(exc))
        raise
    finally:
        for name, value in original.items():
            setattr(module, name, value)
        delattr(module, '_e01_observer_active')
        receipt['wrappers_restored'] = all(getattr(module, name) is value for name, value in original.items())
