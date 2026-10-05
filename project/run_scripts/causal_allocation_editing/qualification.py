"""Bounded fixed-candidate technical qualification; no fits or new GPU jobs."""
import time
import torch
from project.run_scripts.jlz_realization.common import write, tensor_sha
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_restore, rng_equal
from .engine import CausalObjective, require, native_terms, selected_logits, positions
from .geometry import mean_keys


def comparison(actual, reference, atol, rtol):
    x,y=actual.detach().double().cpu(),reference.detach().double().cpu()
    require(x.shape==y.shape, 'QUALIFICATION_SHAPE')
    require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),
            'QUALIFICATION_NONFINITE_COMPARISON')
    error=(x-y).abs();limit=atol+rtol*y.abs();fail=error>limit
    return dict(passed=not bool(fail.any()),atol=atol,rtol=rtol,
                failed_elements=int(fail.sum()),elements=x.numel(),max_absolute=float(error.max()),
                error_RMS=float(error.square().mean().sqrt()),reference_RMS=float(y.square().mean().sqrt()),
                max_error_minus_limit=float((error-limit).max()))


@torch.no_grad()
def _physical_probe(a,entry,payload):
    """Real committed-parameter forward, then unconditional exact RAM restore."""
    original_weights={l:w.detach().clone() for l,w in a.weights.items()}
    rng=rng_snapshot();guard=a.guard();raw={l:[] for l in a.sites};rwrows=[]
    task=0.;nll=torch.zeros_like(payload['nll']);kl=torch.zeros_like(payload['kl'])
    weight_identity=False;logit_checks=[];action_checks={str(l):[] for l in a.sites}
    try:
        for l,w in a.weights.items():w.copy_(payload['weights'][l].to(w.device))
        weight_identity=all(torch.equal(w.detach().cpu(),payload['weights'][l]) for l,w in a.weights.items())
        require(weight_identity,'QUALIFICATION_EXACT_PAYLOAD_COPY')
        for gi,group in enumerate(entry['groups']):
            captured={};outputs={};handles=[]
            for l in a.sites:
                handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(
                    lambda m,args,l=l:captured.update({l:args[0].detach()})))
                handles.append(a.blocks[l].mlp.down_proj.register_forward_hook(
                    lambda m,args,value,l=l:outputs.update({l:value.detach()})))
            try:
                # The task-local entry preserves the complete native rewrite
                # and KL sentences.  Verify the original token bytes at the
                # SAME owner graph shape; no KL-future crop is reused.
                global_rows=[r['global_row'] for r in group['rows']]
                width=group['tokens']['input_ids'].shape[1]
                original_tokens=entry['pack']['tokens']
                require(all(int(original_tokens['attention_mask'][row].sum())<=width for row in global_rows),
                        'NATIVE_UNCROPPED_OWNER_WIDTH_IDENTITY')
                tokens={k:v[global_rows,:width].to(a.device) for k,v in original_tokens.items()}
                nh,fh=a.full(tokens)
                readout={}
                terms,values=native_terms(a,entry,group,nh,fh,capture=readout);task+=float(terms)
                reference_logits,_=selected_logits(a,group['rows'],payload['nll_hidden'][gi].to(a.device),
                                                   payload['final_hidden'][gi].to(a.device))
                logit_checks.append(comparison(readout['selected_logits'],reference_logits,2e-5,2e-4))
                ix,pos,_=positions(group,a.device)
                for l in a.sites:
                    key=captured[l][ix,pos]
                    # Actual local output subtraction vs EXACT effective FP32
                    # parameter subtraction, not the ideal cast(U) surrogate.
                    # Preserve original native [owner rows, padded T] linear
                    # shape for the reference too; selected-row GEMM regrouping
                    # is not silently certified by this local-action check.
                    baseline=torch.nn.functional.linear(captured[l],original_weights[l])[ix,pos]
                    actual=outputs[l][ix,pos]-baseline
                    delta=payload['weights'][l].double()-original_weights[l].cpu().double()
                    predicted=(delta@key.cpu().double().T).T
                    action_checks[str(l)].append(comparison(actual,predicted,2e-5,2e-4))
                for kind,owner,column,value in values:
                    if kind=='rewrite':nll[owner,column]=value
                    else:kl[owner]=value
                for j,row in enumerate(group['rows']):
                    if row['kind']=='rewrite':
                        rwrows.append(row)
                        for l in a.sites:raw[l].append(captured[l][j,row['lookup']].cpu())
            finally:
                for handle in handles:handle.remove()
    finally:
        for l,w in a.weights.items():w.copy_(original_weights[l])
        rng_restore(rng)
    require(all(torch.equal(w,original_weights[l]) for l,w in a.weights.items()) and a.guard()==guard and rng_equal(rng),
            'QUALIFICATION_RAM_RESTORE')
    keys={l:mean_keys(torch.stack(raw[l]).T,rwrows,entry['pack']).double() for l in a.sites}
    comparisons={str(l):comparison(keys[l],payload['K'][l],2e-5,2e-4) for l in a.sites}
    comparisons['nll']=comparison(nll,payload['nll'],2e-5,2e-4)
    comparisons['kl']=comparison(kl,payload['kl'],2e-5,2e-4)
    passed=all(v['passed'] for v in comparisons.values()) and all(v['passed'] for v in logit_checks)
    passed=passed and all(v['passed'] for parts in action_checks.values() for v in parts)
    return dict(passed=passed,checks=comparisons,selected_full_vocabulary_logits=logit_checks,
                effective_local_action=action_checks,
                task_sum=task,accepted_payload_exact=True,RAM_restore_exact=True,
                native_complete_KL_prompt=True,causal_prefix_crop_reused=False,
                native_full_KL_input_verified=True,
                original_owner_forward_shape_preserved=True,
                persistent_history_appends=0,model_forward_groups=len(entry['groups']),
                extra_selected_head_calls=len(entry['groups']),extra_model_forwards=0,
                action_reference_linear_calls=len(entry['groups'])*len(a.sites),
                full_prompt_vocabulary_logits_retained=False)


def check(a,entry,out=None):
    """One fixed nonzero candidate on 2--4 original owner-shaped graphs.

    Calls are declared here before target-model execution: one streamed primal,
    one complete dense graph, three streamed reverse channels (combined, task,
    Q), one stopped-Q reverse negative control, and one native parameter probe.
    The nonzero synthetic plan is not an optimizer trajectory/calibration fit.
    """
    B=entry['pack']['n_requests'];require(2<=B<=4,'QUALIFICATION_REQUEST_BUDGET')
    start=time.monotonic();engine=CausalObjective(a,entry,route='direct')
    guard=a.guard();rng=rng_snapshot();u={}
    for index,l in enumerate(a.sites):
        x=torch.arange(a.dims[l][0]*B,device=a.device,dtype=torch.float32).reshape(a.dims[l][0],B)
        x=torch.sin(x*.013+l+.3);x=x/x.double().norm(dim=0).float()
        u[l]=x*.025
        if index==0:u[l][:,0]=0 # zero block must retain a live Jacobian.
    primal=engine.evaluate(u,gradient=False,lambda_Q=.2)
    dense=engine.dense_evaluate(u,lambda_Q=.2)
    direct=engine.backward(primal['payload'],lambda_Q=.2)
    grad={str(l):comparison(direct['gradient'][l],dense['gradient'][l],1e-6,2e-4) for l in a.sites}
    scalar=dict(task=comparison(torch.tensor(primal['task_sum']),torch.tensor(dense['task_sum']),2e-5,2e-4),
                Q=comparison(torch.tensor(primal['Q'],dtype=torch.float64),torch.tensor(dense['Q'],dtype=torch.float64),1e-7,1e-5))
    key={str(l):comparison(primal['payload']['K'][l],dense['K'][l],2e-5,2e-4) for l in a.sites}
    selected='direct';fallback=None
    if not all(v['passed'] for v in grad.values()):
        # Optional direct contraction is OFF if unqualified; same FP32 dense dW
        # at original group shapes is the independent reference route, not a
        # numerical tolerance change or a different scientific objective.
        reference=engine.backward(primal['payload'],lambda_Q=.2,route='dense')
        fallback={str(l):comparison(reference['gradient'][l],dense['gradient'][l],1e-6,2e-4) for l in a.sites}
        selected='dense'
        require(all(v['passed'] for v in fallback.values()),'DENSE_REPLAY_FULL_GRADIENT_PARITY')
    require(all(v['passed'] for v in list(scalar.values())+list(key.values())), 'ACTUAL_CAUSAL_PRIMAL_PARITY')
    channels=engine.backward(primal['payload'],lambda_Q=.2,channels=True,route=selected)
    stopped=engine._reverse(primal['payload'],0.,1.,route=selected,stop_solve=True)[0]
    difference={str(l):float((channels['Q_gradient'][l]-stopped[l]).norm()) for l in a.sites}
    # A numerically zero fixture is inconclusive, never an invented GPU PASS.
    negative_control_detected=any(v>1e-8 for l,v in difference.items() if int(l)!=a.sites[-1])
    require(negative_control_detected,'DOWNSTREAM_Q_NEGATIVE_CONTROL_INCONCLUSIVE')
    physical=_physical_probe(a,entry,primal['payload'])
    require(physical['passed'],'ACTUAL_COMMITTED_MODEL_PRIMAL_PARITY')
    require(a.guard()==guard and rng_equal(rng),'QUALIFICATION_STATE_NONMUTATION')
    record=dict(status='TECHNICAL_READY',actual_model=True,main_complete=False,
        selected_gradient_route=selected,direct_gradient_qualified=all(v['passed'] for v in grad.values()),
        direct_gradient_checks=grad,dense_fallback_checks=fallback,scalar_checks=scalar,key_checks=key,
        downstream_Q_negative_control=dict(detected=negative_control_detected,difference_norm=difference),
        physical_model_probe=physical,fixed_candidates=1,extra_fits=0,optimizer_updates=0,
        native_owner_group_shape_preserved=True,writer_logical_B=B,
        geometry_reference=dense['geometry_reference'],norm_in_smooth_gradient=False,
        predeclared_call_policy='one fixed candidate; original owner groups; optional dense route only on direct contraction failure',
        physical_calls=dict(engine.calls),seconds=time.monotonic()-start,
        no_tensor_checkpoint=True,CPU_synthetic_not_actual_model=True)
    if out is not None:write(out/'qualification.json',record)
    return record


qualify=check
