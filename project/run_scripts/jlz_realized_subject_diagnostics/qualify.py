"""Same-entry fixed-candidate comparison, not an additional optimization fit."""
import gc
import time
import torch
from project.run_scripts.jlz_realized_subject.qualification import fixed,compare
from project.run_scripts.jlz_realized_subject.causal_builder import build
from project.run_scripts.jlz_realized_subject.subject import evaluate
from project.run_scripts.jlz_realized_subject.writer import rng_snapshot,rng_equal
from .common import require,state,write,tensor_sha
from .instrument import extra_components,replace


def qualify(a,entry,H,out):
    before=state(a,H);rng=rng_snapshot();hooks=a.hook_signature();guard=a.guard();results=[]
    start=time.monotonic()
    for observed in (False,True):
        R=fixed(a,entry);built=build(a,entry,R,0)
        if observed:
            component,cost=extra_components(a,entry,R,built)
            del component
        result=evaluate(a,entry,R,built,'A',components=False)
        weights={l:tensor_sha(w) for l,w in built['weights'].items()}
        if observed:
            # Detached RAM capture/replacement/reset, exactly evaluated tensors.
            w0={l:w.detach().cpu().clone() for l,w in a.weights.items()}
            captured={l:w.detach().cpu().clone() for l,w in built['weights'].items()}
            try:replace(a,captured)
            finally:replace(a,w0)
            require({l:tensor_sha(w) for l,w in captured.items()}==weights,'QUAL_SNAPSHOT_HASH')
            del w0,captured
        results.append(dict(loss=result['total_mean'],nll=result['nll'].cpu(),kl=result['kl'].cpu(),
            gradients={l:g.detach().cpu() for l,g in result['gradient'].items()},weights=weights,
            solve={str(l):g['metadata']['relative_residual'] for l,g in built['geometry'].items()}))
        del built,result,R;gc.collect()
    left,right=results;gradients=compare(left['gradients'],right['gradients'])
    loss_error=abs(left['loss']-right['loss']);loss_limit=1e-5+1e-4*abs(left['loss'])
    checks={name:bool(((left[name]-right[name]).abs()<=1e-5+1e-4*left[name].abs()).all()) for name in ('nll','kl')}
    valid=loss_error<=loss_limit and all(r['passed'] for r in gradients.values()) and all(checks.values()) and left['weights']==right['weights']
    require(state(a,H)==before and a.guard()==guard and a.hook_signature()==hooks and rng_equal(rng),'QUAL_STATE_MUTATION')
    receipt=dict(passed=valid,logical_B=entry['pack']['n_requests'],fixed_candidate_comparisons=2,
        scientific_fits=0,optimizer_updates=0,history_appends=0,loss_error=loss_error,loss_limit=loss_limit,
        gradient=gradients,per_context=checks,materialization_exact=left['weights']==right['weights'],
        original_solve=left['solve'],instrumented_solve=right['solve'],seconds=time.monotonic()-start,
        snapshot_replacement_restore_exact=True,nonselected_guard='pointer/version, not fullmodel byte hash',
        CUDA_allocated_peak=torch.cuda.max_memory_allocated() if a.device.type=='cuda' else None,
        extra_component_cost=cost,actual_model=True)
    write(out/'instrumentation-parity.json',receipt)
    require(valid,'INSTRUMENTATION_FORWARD_GRADIENT_PARITY')
    return receipt


def mask_parity(actual,expected,label):
    ref={r['identity']:r for r in expected};rows=[]
    for row in actual:
        require(row['identity'] in ref,'MASK_PARITY_IDENTITY');old=ref[row['identity']]
        for tag in ('new','true'):
            require(all(row[tag+'_'+k]==old[tag+'_'+k] for k in ('token_identity','token_count')),'MASK_PARITY_TOKEN_ID')
            error=abs(row[tag+'_nll']-old[tag+'_nll']);limit=1e-5+1e-4*abs(old[tag+'_nll'])
            rows.append(dict(identity=row['identity'],target=tag,error=error,limit=limit,passed=error<=limit,
                TF_equal=all(row[tag+'_'+k]==old[tag+'_'+k] for k in ('token_correct','strict'))))
    # TF ties may be discontinuous, but masks of identical weight/materialization
    # and identical MB1 grouping must still pass the declared equality boundary.
    return dict(label=label,rows=rows,passed=all(r['passed'] and r['TF_equal'] for r in rows),
        denominator=len(actual),max_NLL_error=max((r['error'] for r in rows),default=0.))
