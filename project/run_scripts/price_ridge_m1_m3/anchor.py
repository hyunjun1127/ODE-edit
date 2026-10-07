"""M1 uses existing same-forward hidden values, never an extra model call."""
import torch


def guarded_anchor(found, rows, canonical_rows, canonical_index, enabled):
    row = rows[canonical_index]
    h = found[canonical_index, row['lookup']].detach()
    if not enabled or row['lookup'] != 0:
        return h.norm()
    canonical = set(canonical_rows)
    prefix = [(i, r) for i, r in enumerate(rows)
              if r['kind'] == 'rewrite' and r['request'] == row['request']
              and r['global_row'] not in canonical]
    if row['kind'] != 'rewrite' or len(prefix) != 5:
        raise ValueError('M1_EXACT_FIVE_PREFIX_REWRITE_ROWS')
    norms = torch.stack([found[i, r['lookup']].detach().norm() for i, r in prefix])
    if norms.dtype != torch.float32 or not bool(torch.isfinite(norms).all()):
        raise ValueError('M1_FINITE_NATIVE_FP32_PREFIX_NORMS')
    return norms.mean()
