"""CPU comparison of immutable replay endpoints with migrated original CPs."""
import argparse
import json
import time
from pathlib import Path

from .contracts import digest, member, save


def compare(actual, reference, reference_sha, output):
    import torch
    from .fixtures import tensor_sha
    start = time.monotonic()
    bindings = dict(actual=member(actual), reference=member(reference, expected=reference_sha))
    a = torch.load(actual, map_location='cpu', weights_only=False, mmap=True)
    b = torch.load(reference, map_location='cpu', weights_only=False, mmap=True)
    assert set(a['weights']) == set(b['weights'])
    pairs = [(name, a['weights'][name], b['weights'][name]) for name in a['weights']]
    pairs.append(('cache_c', a['cache_c'], b['cache_c']))
    rows = []
    for name, x, y in pairs:
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        # Chunk reductions avoid retaining a second dense FP64 history matrix.
        xn, yn = x.reshape(-1), y.reshape(-1)
        sq = refsq = 0.; maximum = 0.; nonzero = 0
        for begin in range(0, xn.numel(), 1 << 20):
            left = xn[begin:begin+(1 << 20)].double()
            right = yn[begin:begin+(1 << 20)].double()
            delta = left-right
            sq += float(delta.square().sum()); refsq += float(right.square().sum())
            maximum = max(maximum, float(delta.abs().max())); nonzero += int(torch.count_nonzero(delta))
        rows.append(dict(name=name, shape=list(x.shape), dtype=str(x.dtype), finite=True,
                         actual_sha256=tensor_sha(x), reference_sha256=tensor_sha(y),
                         bitexact=torch.equal(x, y), nonzero_difference_elements=nonzero,
                         max_abs=maximum, difference_norm=sq**.5,
                         relative_norm=(sq/refsq)**.5 if refsq else None))
    identity = {}
    for key in ('batch','seen_ids','base_model_revision','contexts','rng','covariance'):
        identity[key] = dict(present_actual=key in a['metadata'], present_reference=key in b['metadata'],
            exact=key in a['metadata'] and key in b['metadata'] and digest(a['metadata'][key])==digest(b['metadata'][key]))
    exact = all(r['bitexact'] for r in rows) and all(r['exact'] for r in identity.values())
    result = dict(status='ENDPOINT_STATE_EXACT' if exact else 'NONEXACT_CAUSE_UNRESOLVED',
                  bindings=bindings, tensors=rows, metadata=identity,
                  endpoint_state_exact=exact, original_trajectory_equivalence=False,
                  trajectory_scope='ENDPOINT_COMPARISON_ONLY_INTERMEDIATE_TARGETS_SEPARATE',
                  hardware_cause_attribution=False, target_recomputation_equivalence='SEPARATE_PER_BATCH_RECEIPT',
                  original_metrics_re_evaluated=False, model_GPU_action=0,
                  elapsed_seconds=time.monotonic()-start, scientific_promotion=False)
    return save(output, result)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--reference-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(compare(a.actual,a.reference,a.reference_sha,a.output)))
