"""Small CPU mathematical reference, not a model runner or original GSS-Greedy.

All caller features must use one fixed map and the SAME current native state.
"""
import hashlib
import numpy as np


def priority(identity, seed=20260920):
    return hashlib.sha256(f'{seed}|{identity}'.encode()).hexdigest(),str(identity)


def history_weights(created_batches, current_batch, *, recency=True, cap=512, batch_size=100):
    created = np.asarray(created_batches,dtype=float)
    ages = current_batch-1-created
    if np.any(ages<0):
        raise ValueError('current/future version is not past history')
    if len(ages)==0:
        return np.empty(0)
    if not recency or np.all(ages==ages[0]):
        return np.full(len(ages),1/len(ages))
    w = 1+np.exp2(-ages/(cap/batch_size))
    return w/w.sum()


def factor_sketch(activation_gradient, fixed_keys, projector, maps):
    """Dense P only in this small reference; runtime caches V.T @ P @ K."""
    a,k,p = map(np.asarray,(activation_gradient,fixed_keys,projector))
    if a.shape[1]!=k.shape[1]:
        raise ValueError('factor positions must match, including valid prefix positions')
    values = [(o.T@a)@(v.T@p@k).T for o,v in maps]
    return np.concatenate([x.ravel() for x in values])/np.sqrt(len(values))


def gss_prune(features, identities, cap=512, *, seed=20260920, zero=1e-12):
    z = np.asarray(features,dtype=float)
    if z.ndim!=2 or len(z)!=len(identities) or len(set(identities))!=len(identities):
        raise ValueError('unique identity per feature row required')
    if cap<=0 or not np.all(np.isfinite(z)):
        raise ValueError('positive capacity and finite features required')
    n = len(z)
    if n<=cap:
        return dict(selected=list(range(n)),removed=[],trace=[],selection_exercised=False)
    norms = np.linalg.norm(z,axis=1)
    active = list(np.flatnonzero(norms>zero))
    zeros = list(np.flatnonzero(norms<=zero))
    trace,removed = [],[]
    if len(active)<=cap:
        extras = sorted(zeros,key=lambda j:priority(identities[j],seed))[:cap-len(active)]
        selected = sorted(active+extras)
        return dict(selected=selected,removed=[i for i in range(n) if i not in selected],
                    trace=trace,selection_exercised=True,zero_count=len(zeros))
    u = z[active]/norms[active,None]
    gram = u@u.T
    local = list(range(len(active)))
    row_sum = gram.sum(axis=1)
    # At most pool-cap removals; Gram reused for all comparisons.
    while len(local)>cap:
        best = max(row_sum[j] for j in local)
        tol = 64*np.finfo(float).eps*max(1,abs(best))
        candidates = [j for j in local if best-row_sum[j]<=tol]
        chosen = min(candidates,key=lambda j:priority(identities[active[j]],seed))
        before = float(row_sum[local].sum())
        trace.append(dict(active=[active[j] for j in local],removed=active[chosen],
                          objective_before=before,
                          predicted_objective_after=before-2*float(row_sum[chosen])+1))
        local.remove(chosen)
        row_sum -= gram[:,chosen]
        removed.append(active[chosen])
    selected = sorted(active[j] for j in local)
    return dict(selected=selected,removed=removed+zeros,trace=trace,
                selection_exercised=True,zero_count=len(zeros))


def active_versions(events):
    """Ledger toy: identical-target repeats preserve version birth/teacher identity."""
    active = {}
    for event in events:
        fact = tuple(event['fact'])
        old = active.get(fact)
        if old is None or old['target']!=event['target']:
            active[fact] = dict(event)
    return active


def shared_cells(final_batch=100):
    rows = []
    for batch in range(1,final_batch+1):
        groups = ['ALL'] if batch<=2 else ['UNIFORM','REC'] if batch<=6 else ['RES','GSS','REC']
        for group in groups:
            previous = (batch-1)*100
            selected = min(512,previous)
            overflow_gss = batch>=7 and group in ('GSS','REC')
            pool = min(612,previous) if overflow_gss else selected
            rows.append(dict(batch=batch,stage='B2_CHECK' if batch<=2 else 'LIFELONG',state_group=group,native_passes=1,
                reference_gradient_sweeps=1,reference_candidate_sweeps_max=2,
                history_forward_pool=pool,history_KL_fact_VJPs=pool,
                history_NLL_fact_VJPs=pool if overflow_gss else 0,
                history_candidate_fact_forwards_max=2*selected,
                selection_exercised=int(overflow_gss),extra_z=0))
    return rows
