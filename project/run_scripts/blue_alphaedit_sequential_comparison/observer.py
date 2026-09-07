"""Call-scoped counters/capture around unmodified BLUE module globals."""
import time
from contextlib import contextmanager
from .integrity import tensor_sha, digest


@contextmanager
def observe(module, hp, weights, cache, requests):
    originals = {k: getattr(module, k) for k in ('compute_z', 'compute_ks')}
    receipt = dict(z=[], keys=[], target_seconds=0.0, key_seconds=0.0)
    def z(*args, **kwargs):
        start = time.monotonic()
        value = originals['compute_z'](*args, **kwargs)
        receipt['target_seconds'] += time.monotonic()-start
        receipt['z'].append(dict(case_id=args[2]['case_id'], layer=args[4], sha256=tensor_sha(value), norm=float(value.detach().double().norm())))
        return value
    def ks(*args, **kwargs):
        start = time.monotonic()
        value = originals['compute_ks'](*args, **kwargs)
        receipt['key_seconds'] += time.monotonic()-start
        # Capture only identity/scalars. The second pass is the native history append.
        receipt['keys'].append(dict(layer=args[4], sha256=tensor_sha(value), shape=list(value.shape)))
        return value
    module.compute_z, module.compute_ks = z, ks
    try:
        yield receipt
        expected = [(layer, r['case_id']) for layer in hp.layers for r in requests]
        if [(r['layer'], r['case_id']) for r in receipt['z']] != expected:
            raise RuntimeError('NATIVE_LAYER_Z_ORDER_COUNT')
        if [r['layer'] for r in receipt['keys']] != hp.layers * 2:
            raise RuntimeError('NATIVE_KEY_APPEND_PASS_COUNT')
        receipt.update(compute_z=len(expected), writer_layers=len(hp.layers),
                       history_append_passes=1, cold_reset_inside_batch=0, z_hash_order=digest(receipt['z']))
    finally:
        for k, v in originals.items():
            setattr(module, k, v)
        assert all(getattr(module, k) is v for k, v in originals.items())
