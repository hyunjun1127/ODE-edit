"""Whole logical batch actual-key ridge; rewrite + KL share lower writes."""
import time
import torch
import torch.nn.functional as F
from .profile import move
from .geometry import ridge, mean_keys
from .common import require
from .physical_linear import materialize, linear


def build(a, entry, R, candidate, route='direct', stop_geometry=False):
    start = time.monotonic()
    groups = [dict(rows=g['rows'], **move(g['cache'], a.device)) for g in entry['groups']]
    rows = [r for g in groups for r in g['rows']]
    ids = [r['global_row'] for r in rows]
    require(sorted(ids) == list(range(len(ids))), 'WHOLE_NATIVE_ROW_COVERAGE')
    order = torch.tensor(sorted(range(len(ids)), key=lambda i: ids[i]), device=a.device)
    rw = [j for j, r in enumerate(rows) if r['kind'] == 'rewrite']
    rwrows = [rows[j] for j in rw]
    geometry = {}; weights = {}; v = {}; raw = {}; prebase = {}
    for index, l in enumerate(a.sites):
        subjects = []; bases = []
        for g in groups:
            ix = torch.arange(len(g['rows']), device=a.device)
            pos = torch.tensor([r['lookup'] for r in g['rows']], device=a.device)
            k = g['key'][ix, pos]
            subjects.append(k)
            bases.append(g['residual'][ix, pos] + F.linear(k, entry['entry_weights'][l]))
        key = torch.cat(subjects)
        K = mean_keys(key[rw].T, rwrows, entry['pack']).double()
        if l == a.first and 'ridge' in entry['first_geometry']:
            geo = entry['first_geometry']['ridge']
        else:
            geo = ridge(K.detach() if stop_geometry else K, entry['factors'][l])
            if l == a.first: entry['first_geometry']['ridge'] = geo
        P = geo['P'].detach() if stop_geometry else geo['P']
        geo = dict(geo, P=P)
        with torch.set_grad_enabled(torch.is_grad_enabled() and route == 'dense'):
            W = materialize(entry['entry_weights'][l], R[l], P)
        # Both input VJPs are retained. W is materialized once per candidate.
        effective = linear(key, R[l], P, W, route) - F.linear(key, entry['entry_weights'][l])
        v[l] = effective[order]; raw[l] = key[order]; prebase[l] = torch.cat(bases)[order]
        geometry[l] = geo; weights[l] = W
        if index+1 < len(a.sites):
            for g in groups:
                g['key'], g['residual'] = a.stage(l, a.sites[index+1], g['key'], g['residual'],
                    R[l], P, W, g['kwargs'], route)
    return dict(geometry=geometry, P={l:g['P'] for l,g in geometry.items()}, weights=weights,
                v=v, actual_keys=raw, prebase=prebase, rows=sorted(rows,key=lambda r:r['global_row']),
                candidate=candidate, seconds=time.monotonic()-start, writer_kind='ridge',
                first_solve_this_candidate=candidate==1, upper_solve_count=len(a.sites)-1,
                whole_B=True, KL_in_builder=True, KL_in_ridge=False, subject_hook_in_builder=False)
