"""Whole-B layer barriers and all-token lower writes; no subject hooks."""
import time
import torch
from .profile import move
from .geometry import ridge, exact, mean_keys
from .common import require
from .physical_linear import materialize

def build(a, entry, D, candidate, route='direct', qualification_stop_P=False, writer_kind='ridge'):
    start = time.monotonic(); B = entry['pack']['n_requests']
    # Include all RW rows, with full original rows (no builder crop enabled).
    groups = []
    for group in entry['groups']:
        selected = [j for j,r in enumerate(group['rows']) if r['kind']=='rewrite']
        if not selected: continue
        cache = move(group['cache'],a.device)
        kw = dict(cache['kwargs'])
        # RoPE/position can be shared [1,...]; attention mask can be per row.
        for name in ('attention_mask','position_ids','position_embeddings'):
            v = kw.get(name)
            def select(t):
                return t[selected] if isinstance(t,torch.Tensor) and t.ndim>0 and t.shape[0]==len(group['rows']) and t.shape[0]!=1 else t
            kw[name] = tuple(select(t) for t in v) if isinstance(v,tuple) else select(v)
        groups.append(dict(rows=[group['rows'][j] for j in selected],key=cache['key'][selected],
                           residual=cache['residual'][selected],kwargs=kw))
    geometry={}; weights={}
    for index,l in enumerate(a.sites):
        key_parts=[]
        for g in groups:
            ix=torch.arange(len(g['rows']),device=a.device)
            pos=torch.tensor([r['lookup'] for r in g['rows']],device=a.device)
            key_parts.append(g['key'][ix,pos].T)
        raw=torch.cat(key_parts,1)
        rows=[r for g in groups for r in g['rows']]
        K=mean_keys(raw,rows,entry['pack']).double()
        cache=entry['first_geometry']
        if l==a.first and writer_kind in cache:
            geo=cache[writer_kind]
        elif writer_kind=='ridge':
            geo=ridge(K,entry['factors'][l])
            if l==a.first: cache[writer_kind]=geo
        else:
            operator,verdict=exact(K,entry['factors'][l],D[l],entry['entry_weights'][l])
            geometry[l]=dict(metadata=verdict,K=K)
            if operator is None:
                return dict(geometry=geometry,qualified=False,unsupported_layer=l,
                            seconds=time.monotonic()-start,writer_kind=writer_kind)
            geo=dict(operator,metadata=verdict)
            # Exact D is fixed only within this one shadow; never cached as ridge.
        if qualification_stop_P:
            p=geo['P'].detach()
            from .geometry import PriorProduct
            r=p.T@K-torch.eye(B,device=K.device,dtype=K.dtype)
            geo=dict(geo,P=p,M=p.T@K,G=p.T@PriorProduct.apply(entry['factors'][l]['A'],p),E=r@r.T)
        geo['raw_keys']=raw
        geometry[l]=geo
        # Custom direct backward owns both D/P paths; do not also differentiate W.
        with torch.set_grad_enabled(torch.is_grad_enabled() and route=='dense'):
            weights[l]=materialize(entry['entry_weights'][l],D[l],geo['P'])
        if index+1<len(a.sites):
            next_l=a.sites[index+1]
            for g in groups:
                g['key'],g['residual']=a.stage(l,next_l,g['key'],g['residual'],D[l],geo['P'],weights[l],g['kwargs'],route)
    return dict(geometry=geometry,P={l:g['P'] for l,g in geometry.items()},weights=weights,
                seconds=time.monotonic()-start,candidate=candidate,all_current_columns=True,
                qualified=True,writer_kind=writer_kind,
                key_source='actual_lower_writer',builder_delta_hook=False)
