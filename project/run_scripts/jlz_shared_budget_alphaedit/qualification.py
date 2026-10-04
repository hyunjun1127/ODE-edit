"""Bounded fixed-candidate actual-model comparisons; no extra fit."""
import importlib
import torch
from .common import require,write
from .optimize import forward
from .optimizer import EfficiencyAdam,project
from .writer import capture
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_two_arm.baseline_pilot import _import_native

def compare(x,y):
    e=x.double()-y.double();n=float(y.double().norm())
    value=float(e.norm())/max(n,1e-8) if n>1e-8 else float(e.abs().max())
    limit=2e-3 if n>1e-8 else 1e-6
    return dict(error=value,limit=limit,reference_norm=n,pass_=value<=limit)

def evaluated(a,entry,u,route,reverse=False):
    a.native_route=route;value=0.;captures={}
    groups=list(reversed(entry['groups'])) if reverse else entry['groups']
    for group in groups:
        values,z=forward(a,entry,group,u)
        F=sum(n+.0625*k for n,k in values.values());value+=float(F.detach());F.backward()
        captures.update(z)
    return value,{l:v.grad.clone() for l,v in u.items()},captures

def variables(a,entry,scale):
    B=entry['pack']['n_requests']
    return {l:(torch.sin(torch.arange(a.dims[l][0]*B,device=a.device).float()+l).reshape(a.dims[l][0],B)
        *scale/(len(a.sites)*a.dims[l][0]**.5)).requires_grad_() for l in a.sites}

def qualify(a,bench,entry,config,out):
    write(out/'predeclared.json',dict(fixed_scales=[0.,.025],routes=['cached','full'],
        reversed_group_nonzero_comparison=True,anchor_only_same_candidate=True,
        native_operator_reference=True,loss_atol=2e-5,loss_rtol=2e-4,gradient_relative=2e-3,
        tiny_gradient_absolute=1e-6,fit_calls=0,new_baseline_fits=0,quality_gate=False))
    try:
        for i,scale in enumerate((0.,.025)):
            results=[evaluated(a,entry,variables(a,entry,scale),route) for route in ('cached','full')]
            own,ref=results;checks={l:compare(own[1][l],ref[1][l]) for l in a.sites}
            loss_error=abs(own[0]-ref[0]);limit=2e-5+2e-4*abs(ref[0])
            capture_checks=[bool(torch.allclose(own[2][r][l],ref[2][r][l],atol=2e-5,rtol=2e-4)) for r in own[2] for l in a.sites]
            write(out/f'fixed-{i}.json',dict(loss_error=loss_error,limit=limit,gradients=checks,canonical_capture=capture_checks))
            require(loss_error<=limit and all(c['pass_'] for c in checks.values()) and all(capture_checks),'ACTUAL_CACHE_REFERENCE')
        normal=evaluated(a,entry,variables(a,entry,.025),'cached')
        rev=evaluated(a,entry,variables(a,entry,.025),'cached',True)
        checks={l:compare(rev[1][l],normal[1][l]) for l in a.sites}
        write(out/'microbatch.json',dict(gradient=checks,request_complete_groups=True,reverse_order=True,
            loss_error=abs(normal[0]-rev[0])))
        require(all(c['pass_'] for c in checks.values()),'ACTUAL_MICROBATCH')
        # One actual anchor-only gradient; explicit absolute-delta chain rule,
        # then original torch Adam with tiny-scaled gradient as well.
        star=a.profile['anchor_layer'];u=variables(a,entry,0.)
        loss_u,gu,_=evaluated(a,entry,u,'full');anchor=entry['anchors'][star][0]
        D={l:torch.zeros_like(v,requires_grad=True) for l,v in u.items()};loss_D=0.
        for group in entry['groups']:
            nh,fh,_=a.native(group,D,False);lps=row_logprobs(a,group['rows'],nh,fh);terms=[]
            for row,lp in zip(group['rows'],lps):
                if row['kind']=='rewrite':
                    target=row['target'][row['target']!=-100].to(a.device)
                    terms.append(-lp.gather(1,target[:,None]).mean()/entry['pack']['n_rw'])
                else:
                    teacher=entry['teachers'][row['request']].to(a.device)
                    terms.append(.0625*torch.nn.functional.kl_div(teacher,lp,log_target=True,reduction='sum'))
            value=sum(terms);loss_D+=float(value.detach());value.backward()
        chain={l:compare(gu[l],D[l].grad*entry['anchors'][l][None,:]) for l in a.sites}
        write(out/'absolute-delta-gradient.json',dict(gradient=chain,loss_error=abs(loss_D-loss_u),
            reference='absolute fullblock D leaf and native functional KL',new_fit=0))
        require(abs(loss_D-loss_u)<=2e-5+2e-4*abs(loss_D) and all(x['pass_'] for x in chain.values()),'NATIVE_ABSOLUTE_CHAIN')
        alignment=[]
        for factor in (1.,1e-9):
            g=gu[star][:,0]*factor/anchor
            d=torch.zeros_like(g,requires_grad=True);adam=torch.optim.Adam([d],lr=.1,eps=1e-8,foreach=False)
            d.grad=g;adam.step()
            with torch.no_grad():
                if d.norm()>.75*anchor:d.mul_(.75*anchor/d.norm())
            v,r=EfficiencyAdam([torch.zeros_like(g)],anchor).step([torch.zeros_like(g)],[g*anchor])
            error=float((anchor*v[0]-d).abs().max());ok=bool(torch.allclose(anchor*v[0],d,atol=2e-5,rtol=2e-4))
            alignment.append(dict(factor=factor,maxabs=error,pass_=ok,epsilon_u=r['eps'],gamma=r['gamma']))
        write(out/'native-scaled-update.json',dict(checks=alignment,extra_target_forward=0))
        require(all(c['pass_'] for c in alignment),'NATIVE_SCALED_UPDATE')
        measured=capture(a,entry,a.sites);checks={}
        with _import_native(config['native_root']):
            ks=importlib.import_module('memit.compute_ks').compute_ks
            hp=importlib.import_module('memit.memit_hparams').MEMITHyperParams.from_json(config['native_hparams'])
            for l in a.sites:
                ref=ks(a.model,bench.tokenizer,entry['pack']['requests'],hp,l,bench.contexts).T.cpu()
                own=measured['mean'][l];diff=(own-ref).abs();limit=2e-5+2e-4*ref.abs()
                checks[l]=dict(maxabs=float(diff.max()),max_excess=float((diff-limit).max()),pass_=bool((diff<=limit).all()))
        write(out/'native-keys.json',dict(checks=checks,original_function=True,baseline_fit=0))
        require(all(c['pass_'] for c in checks.values()),'NATIVE_KEYS')
        write(out/'ready.json',dict(status='QUALIFIED_BOUNDED_ACTUAL_MODEL',main_initial=False,quality_gate=False))
    finally:a.native_route='cached'
