"""Whole-B layer barriers and all-token lower writes; no subject hooks."""
import time
import torch
from .profile import move
from .dynamic_solve import solve
from .physical_linear import materialize

def build(a, entry, D, candidate, route='direct', qualification_stop_P=False):
    start = time.monotonic(); B = entry['pack']['n_requests']
    alpha = torch.tensor(entry['pack']['key_context_weights'],dtype=torch.float64,device=a.device)
    owners = torch.tensor(entry['pack']['key_request'],device=a.device)
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
        K=torch.cat(key_parts,1)
        if l==a.first and entry['first_geometry'] is not None:
            geo=entry['first_geometry']
        else:
            geo=solve(K,entry['factors'][l],alpha,owners,B)
            if l==a.first: entry['first_geometry']=geo
        if qualification_stop_P:
            # Negative control only, never selected by the production fit caller.
            p=geo['P'].detach();z=torch.nn.functional.one_hot(owners,B).T.double()
            t=entry['factors'][l].T@p;r=p.T@K.double()-z
            geo=dict(geo,P=p,T=t,G=t.T@t,E=(r*alpha)@r.T)
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
                key_source='actual_lower_writer',builder_delta_hook=False)
