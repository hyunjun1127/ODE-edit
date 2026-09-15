"""Module-scoped no-op observer: no new forwards, no tensor persistence."""
from contextlib import contextmanager
import time


class Proxy:
    def __init__(self, base, **changes):
        self.base, self.changes = base, changes

    def __getattr__(self, key):
        return self.changes[key] if key in self.changes else getattr(self.base, key)


def nonselected(model, selected):
    return {n: (p.data_ptr(), p._version, tuple(p.shape), str(p.dtype))
            for n, p in model.named_parameters() if n not in selected}


@contextmanager
def observe(module, hp, weights, requests, model):
    from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import tensor_sha
    original = {k: getattr(module, k) for k in ('torch', 'compute_z', 'compute_ks')}
    before = nonselected(model, weights)
    receipt = dict(z=[], keys=[], solves=[], target_seconds=0., key_seconds=0., solve_seconds=0.)

    def z(*args, **kwargs):
        start = time.monotonic()
        v = original['compute_z'](*args, **kwargs)
        receipt['target_seconds'] += time.monotonic() - start
        receipt['z'].append(dict(case_id=int(args[2]['case_id']), layer=args[4],
                                 sha256=tensor_sha(v), norm=float(v.detach().float().norm())))
        return v

    def keys(*args, **kwargs):
        start = time.monotonic()
        v = original['compute_ks'](*args, **kwargs)
        receipt['key_seconds'] += time.monotonic() - start
        assert tuple(v.shape) == (len(requests), 14336), 'CAKE_KEY_CARDINALITY_OR_LAYER_SHAPE'
        receipt['keys'].append(dict(layer=args[4], sha256=tensor_sha(v), shape=list(v.shape)))
        return v

    def solve(a, b, *args, **kwargs):
        start = time.monotonic()
        v = original['torch'].linalg.solve(a, b, *args, **kwargs)
        # GPU operation is synchronized only for timing, never altered.
        original['torch'].cuda.synchronize()
        receipt['solve_seconds'] += time.monotonic() - start
        receipt['solves'].append(dict(shape=list(a.shape), rhs_shape=list(b.shape), dtype=str(a.dtype)))
        return v

    module.compute_z, module.compute_ks = z, keys
    module.torch = Proxy(original['torch'], linalg=Proxy(original['torch'].linalg, solve=solve))
    try:
        yield receipt
        assert hp.layers == [4, 5, 6, 7, 8]
        assert [(x['layer'], x['case_id']) for x in receipt['z']] == [(8, int(r['case_id'])) for r in requests], 'NATIVE_Z_ORDER'
        assert [x['layer'] for x in receipt['keys']] == hp.layers * 2, 'NATIVE_WRITE_HISTORY_KEYS'
        assert len(receipt['solves']) == 5, 'NATIVE_SOLVE_COUNT'
        assert before == nonselected(model, weights), 'NONSELECTED_WEIGHT_MUTATION'
        receipt.update(compute_z=len(requests), solve_calls=5, history_append_passes=1,
                       layer_history_updates=5, nonselected_pointer_version_exact=True,
                       nonselected_full_bytes_checked=False, extra_forward_count=0,
                       tensor_persistence=False, timer_components_nonadditive_to_outer=True)
    finally:
        for k, v in original.items():
            setattr(module, k, v)
