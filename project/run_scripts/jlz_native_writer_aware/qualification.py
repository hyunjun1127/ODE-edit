"""Fixed B1-subset operators, no optimization or extra full-B fit."""
import torch
from project.run_scripts.jlz_realized_subject.qualification import fixed,compare,full_masked,singletons
from project.run_scripts.jlz_realized_subject.causal_builder import build as dense_build
from project.run_scripts.jlz_realized_subject.geometry import ridge
from .builder import build,reverse
from .subject import evaluate
from .common import require,write

def check(a,entry,root,zero=False):
    R=fixed(a,entry,0. if zero else .025);built=build(a,entry,R,0)
    result=evaluate(a,entry,built,backward=True);g,receipt=reverse(a,entry,R,built,result['adjoint'])
    original=dense_build(a,dict(entry,first_geometry={}),R,0,route='dense')
    reference=evaluate(a,entry,original,backward=True,dense_R=R)
    gradients=compare(g,{l:x.to(a.device) for l,x in reference['adjoint'].items()})
    loss=(result['F']-reference['F']).abs()
    forward={};solves={}
    for l in a.sites:
        ref=original['v'][l].detach().cpu();err=(built['v'][l]-ref).abs()
        forward[l]=dict(max_error=float(err.max()),passed=bool((err<=1e-5+1e-4*ref.abs()).all()))
        # Same stored A direct native reference, no symmetrization.
        k=built['K'][l];A=entry['factors'][l]['A'].to(a.device)
        direct=torch.linalg.solve(A+k@k.T,k);err=(direct-built['P'][l]).norm()/direct.norm().clamp_min(1e-30)
        solves[l]=dict(relative_error=float(err),passed=float(err)<=1e-8)
        del A,direct
    with full_masked(a):full=evaluate(a,entry,built,backward=True)
    fullg,_=reverse(a,entry,R,built,full['adjoint'])
    fullcmp=compare(g,fullg)
    stopped,_=reverse(a,entry,R,built,result['adjoint'],stop_solve=True)
    split=singletons(entry)
    split['groups'].sort(key=lambda g:g['rows'][0]['global_row'])
    splitbuilt=build(a,split,R,0);splitvalue=evaluate(a,split,splitbuilt,backward=True)
    splitg,_=reverse(a,split,R,splitbuilt,splitvalue['adjoint'])
    splitcmp=compare(g,splitg)
    record=dict(B=entry['pack']['n_requests'],zero_candidate=zero,additional_fit=0,updates=0,
        full_dense_gradient=gradients,full_native_hooks_gradient=fullcmp,microbatch1_gradient=splitcmp,action=forward,solve=solves,
        loss_error_max=float(loss.max()),native_loss_pass=bool((loss<=1e-5+1e-4*reference['F'].abs()).all()),
        full_hook_loss_error_max=float((full['F']-result['F']).abs().max()),
        stop_solve_negative_control=compare(stopped,g),reverse=receipt,
        thresholds=dict(loss_atol=1e-5,loss_rtol=1e-4,gradient_atol=1e-6,gradient_rtol=1e-3,solve=1e-8))
    write(root/'qualification.json',record)
    require(record['native_loss_pass'] and all(x['passed'] for d in (gradients,fullcmp,splitcmp,forward,solves) for x in d.values()),'V14_ACTUAL_OPERATOR_GRADIENT_PARITY')
    require(bool(((full['F']-result['F']).abs()<=1e-5+1e-4*full['F'].abs()).all()),'V14_NATIVE_FULL_LOSS')
    return record
