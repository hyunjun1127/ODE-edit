"""Explicit native-row identities and exact cached nonsymmetric ridge adjoint."""
import torch
from official.ours.config import require_config
from official.ours.common import require

def annotate(entry,config):
    config=require_config(config)
    counts={};rows=[];groups=[]
    for gi,g in enumerate(entry['groups']):
        for row in g['rows']:
            owner=row['request'];kind=row['kind']
            index=counts.get((owner,kind),0);counts[owner,kind]=index+1
            row['reduction_index']=index
            row['role_weight']=1/entry['pack']['n_rw'] if kind=='rewrite' else config['lambda_KL']
            rows.append(row)
        tok=g['tokens'];groups.append(dict(index=gi,global_rows=[r['global_row'] for r in g['rows']],
            owners=[r['request'] for r in g['rows']],roles=[r['kind'] for r in g['rows']],
            valid_tokens=int(tok['attention_mask'].sum()),padded_tokens=tok['input_ids'].numel(),
            boundary_bytes=sum(g['cache'][k].numel()*g['cache'][k].element_size() for k in ('key','residual'))))
    require([r['global_row'] for r in rows]==list(range(len(rows))),'ROW_ORDER')
    for r in range(entry['pack']['n_requests']):
        require(counts.get((r,'rewrite'))==entry['pack']['n_rw'] and counts.get((r,'kl'))==1,'COMPLETE_OWNER_SCHEMA')
    entry['row_schedule']=groups
    entry['first_site_causal_proof']='entry prefix before first editable site, no u/R inputs; immutable cached boundary'
    return entry

def cached_vjp(K,P,GP,prior):
    """One transpose solve after all P cotangents, using the stored A unchanged."""
    require(K.dtype==P.dtype==GP.dtype==torch.float64 and K.shape==P.shape==GP.shape,'VJP_SHAPE')
    rhs=GP-K@(P.T@GP)
    if prior['LU'] is not None:
        Z=torch.linalg.lu_solve(prior['LU'],prior['pivots'],rhs,adjoint=True)
        backend='cached_same_A_LU_transpose'
    else:
        Z=torch.cholesky_solve(rhs,prior['L']);backend='cached_same_A_cholesky'
    def residual(z):
        AZ=(prior['A'].T@z.cpu()).to(z.device)
        return float((AZ+K@(K.T@z)-GP).norm()/GP.norm().clamp_min(1e-30))
    initial=residual(Z);fallback=False
    if not bool(torch.isfinite(Z).all()) or initial>1e-8:
        Z=torch.linalg.solve((prior['A'].to(K.device)+K@K.T).T,GP)
        backend='native_B_transpose_fallback';fallback=True
    final=residual(Z)
    require(bool(torch.isfinite(Z).all()) and final<=1e-8,'CACHED_TRANSPOSE_RESIDUAL:'+str(final))
    g=Z-Z@(P.T@K)-P@(Z.T@K)
    require(bool(torch.isfinite(g).all()),'CACHED_GRADIENT_FINITE')
    return g,dict(backend=backend,initial_relative_residual=initial,relative_residual=final,
        fallback=fallback,primal_recomputed=False,transpose_solve=1)
