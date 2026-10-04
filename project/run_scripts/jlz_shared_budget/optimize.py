"""Independent request stopping, a*u injection, one F backward, no terminal call."""
import time
import torch
from project.run_scripts.jlz_realization.subject import row_logprobs
from .optimizer import EfficiencyAdam,norm_gradient
from .telemetry import diagnostics,validate_relations
from .common import require,tensor_sha,digest,write

def forward(a,entry,group,u):
    D={l:entry['anchors'][l][None,:]*v for l,v in u.items()}
    nh,fh,hidden=a.native(group,D,True)
    rows=group['rows'];logps=row_logprobs(a,rows,nh,fh);requests=sorted({r['request'] for r in rows})
    nll={r:[] for r in requests};kl={};captures={r:{} for r in requests}
    for j,(row,lp) in enumerate(zip(rows,logps)):
        r=row['request']
        if row['kind']=='rewrite':
            t=row['target'][row['target']!=-100].to(a.device)
            nll[r].append(-lp.gather(1,t[:,None]).mean())
            if row['global_row'] in entry['pack']['canonical_rows']:
                captures[r]={l:h[j].detach().cpu().clone() for l,h in hidden.items()}
        else:
            teacher=entry['teachers'][r].to(a.device)
            kl[r]=(lp.exp()*(lp-teacher)).sum()
    require(all(len(nll[r])==entry['pack']['n_rw'] and set(captures[r])==set(a.sites) and r in kl for r in requests),'COMPLETE_REQUEST_GRAPH')
    return {r:(torch.stack(nll[r]).mean(),kl[r]) for r in requests},captures

def stop_reason(value,candidate,threshold=.05):
    require(0<=candidate<=24,'CANDIDATE_RANGE')
    if value<threshold:return 'ZERO_STEP' if candidate==0 else 'OBJECTIVE_THRESHOLD'
    return 'EVALUATION_BUDGET' if candidate==24 else None

def fit(a,entry,events,out):
    started=time.monotonic();B=entry['pack']['n_requests'];layers=a.sites;ids=entry['pack']['record_ids']
    u={l:torch.zeros((a.dims[l][0],B),device=a.device,dtype=torch.float32,requires_grad=True) for l in layers}
    star=entry['anchors'][a.profile['anchor_layer']]
    opts=[EfficiencyAdam([u[l][:,r] for l in layers],star[r]) for r in range(B)]
    terminal={};z={l:torch.empty((a.dims[l][0],B),dtype=torch.float32) for l in layers};Fcount=Bcount=0
    for candidate in range(25):
        for v in u.values():v.grad=None
        pending={}
        for group in entry['groups']:
            group_requests=sorted({row['request'] for row in group['rows']})
            if all(r in terminal for r in group_requests):continue
            # Default group is one complete request. Multi-request test groups
            # preserve stopped states; only continuing F enters backward.
            values,captures=forward(a,entry,group,u);Fcount+=1
            continuing=[]
            for r in group_requests:
                if r in terminal:continue
                nll,kl=values[r];F=nll+a.profile['kl_factor']*kl
                blocks=[u[l][:,r].detach() for l in layers]
                norms=[float(v.norm()) for v in blocks];budget=sum(norms);price=.5/float(star[r])
                J=float(F.detach())+price*budget
                require(bool(torch.isfinite(F)) and all(bool(torch.isfinite(v).all()) for v in captures[r].values()),'FIT_NONFINITE')
                reason=stop_reason(J,candidate)
                pending[r]=dict(nll=float(nll.detach()),kl=float(kl.detach()),F=float(F.detach()),J=J,
                    budget=budget,price=price,norms=norms,reason=reason,capture=captures[r])
                if reason is None:continuing.append(F)
                else:
                    terminal[r]=dict(candidate=candidate,reason=reason,J=J)
                    for l in layers:z[l][:,r].copy_(captures[r][l])
            if continuing:torch.stack(continuing).sum().backward();Bcount+=1
            del values,captures,continuing
        for r,p in pending.items():
            blocks=[u[l][:,r].detach().clone() for l in layers]
            grads=None if p['reason'] else [u[l].grad[:,r].detach().clone() for l in layers]
            kkt,gs=diagnostics(blocks,grads,p['price'])
            events.emit('candidate_request',dict(evaluation_ordinal=candidate+1,updates_completed=opts[r].t,
                nll=p['nll'],kl_raw=p['kl'],kl_weighted=a.profile['kl_factor']*p['kl'],task_loss=p['F'],
                norm_weighted=p['price']*p['budget'],objective_J=p['J'],anchor_native_l2=float(star[r]),norm_price=p['price'],
                budget_used=p['budget'],budget_radius=.75,budget_utilization=p['budget']/.75,will_backward=p['reason'] is None,
                terminal_reason=p['reason'],kkt=kkt,terminal_z_capture_candidate=candidate),ids[r],candidate)
            for l,n,g in zip(layers,p['norms'],gs):
                events.emit('candidate_layer',dict(entry_anchor_l2=float(entry['anchors'][l][r]),u_l2=n,
                    delta_l2=float((entry['anchors'][l][r]*u[l][:,r]).detach().norm()),plan_active_exact=n>0,
                    plan_share=n/p['budget'] if p['budget'] else 0.,plan_share_status='DEFINED' if p['budget'] else 'NO_PLANNED_EDIT',
                    virtual_post_injection_canonical_l2=float(p['capture'][l].norm()),gradient=g),ids[r],candidate,l)
            if p['reason']:
                events.emit('terminal_request',dict(accepted_candidate_index=candidate,logical_evaluations=candidate+1,
                    backward_calls=opts[r].t,optimizer_updates=opts[r].t,reason=p['reason'],accepted_objective_J=p['J'],
                    accepted_budget_used=p['budget'],accepted_z_identity=digest({l:tensor_sha(z[l][:,r]) for l in layers}),
                    accepted_z_capture_stage='CANONICAL_POST_INJECTION_EXISTING_EVALUATION_FORWARD',
                    gradient_kkt_status='NO_BACKWARD_TERMINAL',frozen_after_terminal=True,kept_in_full_batch_writer=True),ids[r],candidate)
            else:
                total=[g+norm_gradient(v,p['price']) for v,g in zip(blocks,grads)]
                new,s=opts[r].step(blocks,total)
                with torch.no_grad():
                    for i,l in enumerate(layers):
                        u[l][:,r].copy_(new[i])
                        events.emit('optimizer_layer',dict(update_ordinal=s['update'],learning_rate_u=s['lr'],epsilon_u=s['eps'],
                            gamma=s['gamma'][i],s_hat=s['s_hat'][i],all_eligible_layer_mean_s_hat=s['mean_s_hat'],
                            proposal_u_l2=s['proposal_norm'][i],stored_projected_u_l2=s['stored_norm'][i],
                            actual_stored_step_l2=s['step_norm'][i],tau_projection=s['tau'],postcast_request_budget_used=s['stored_budget'],
                            postcast_primal_violation=s['violation'],moments_preserved_after_projection=True,eligible_for_reentry=True),ids[r],candidate,l)
        if len(terminal)==B:break
    require(len(terminal)==B,'TERMINAL_COMPLETENESS')
    counts=validate_relations(events.path,layers,ids)
    result=dict(**counts,physical_forward_calls=Fcount,physical_backward_calls=Bcount,seconds=time.monotonic()-started,
                candidates={ids[r]:v for r,v in terminal.items()},terminal_extra_forward=0,terminal_extra_backward=0)
    write(out/'fit.json',result)
    return dict(u={l:v.detach() for l,v in u.items()},D={l:(entry['anchors'][l][None,:]*v).detach() for l,v in u.items()},
                z=z,terminal=terminal),result
