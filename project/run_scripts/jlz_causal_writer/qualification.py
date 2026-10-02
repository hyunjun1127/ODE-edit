"""Fixed small development comparisons; no efficacy selection or extra fits."""
from contextlib import contextmanager
import torch
from .common import require,write,state
from .subject import native
from .causal_builder import build
from .allocation import loss as allocation
from .physical_aux import actual

def compare(left,right):
    rows={}
    for l in left:
        x,y=left[l].double(),right[l].double()
        rms=float((x-y).square().mean().sqrt());reference=float(y.square().mean().sqrt())
        rows[str(l)]=dict(error_RMS=rms,reference_RMS=reference,limit=1e-6+2e-3*reference,pass_=rms<=1e-6+2e-3*reference)
    return rows

@contextmanager
def full_native_reference(a):
    original=a.native
    def full(group,D,capture=False):
        handles=[];subjects={}
        rows=group['rows'];ix=torch.arange(len(rows),device=a.device)
        pos=torch.tensor([r['lookup'] for r in rows],device=a.device)
        owners=torch.tensor([r['request'] for r in rows],device=a.device)
        for l in a.sites:
            def hook(m,args,output,l=l):
                y=output.clone();y[ix,pos]=y[ix,pos]+D[l][:,owners].T
                if capture:subjects[l]=y[ix,pos]
                return y
            handles.append(a.blocks[l].register_forward_hook(hook))
        try:
            nh,fh=a.full({k:v.to(a.device) for k,v in group['tokens'].items()})
            return nh,fh,subjects
        finally:
            for h in handles:h.remove()
    a.native=full
    try:yield
    finally:a.native=original

def fixed(a,entry,scale):
    B=entry['pack']['n_requests'];result={}
    for l in a.sites:
        x=torch.arange(a.dims[l][0]*B,device=a.device,dtype=torch.float32).reshape(a.dims[l][0],B)
        x=torch.sin(x+l+1);x=x/x.norm(dim=0)
        result[l]=(x*(scale*entry['anchors'][l])).detach().requires_grad_(True)
    return result

def singleton_permuted(entry):
    """Same full rows in reverse order, MB1; retain request-column identity."""
    groups=[];weights={}
    rw=[r for g in entry['groups'] for r in g['rows'] if r['kind']=='rewrite']
    for r,w in zip(rw,entry['pack']['key_context_weights']):weights[r['global_row']]=w
    for g in reversed(entry['groups']):
        for i in reversed(range(len(g['rows']))):
            def select(t):
                return t[i:i+1] if isinstance(t,torch.Tensor) and t.ndim and t.shape[0]==len(g['rows']) and t.shape[0]!=1 else t
            kw={k:tuple(select(t) for t in v) if isinstance(v,tuple) else select(v) for k,v in g['cache']['kwargs'].items()}
            groups.append(dict(rows=[g['rows'][i]],tokens={k:v[i:i+1] for k,v in g['tokens'].items()},
                cache=dict(key=g['cache']['key'][i:i+1],residual=g['cache']['residual'][i:i+1],kwargs=kw)))
    rows=[r for g in groups for r in g['rows'] if r['kind']=='rewrite']
    pack=dict(entry['pack'],key_context_weights=[weights[r['global_row']] for r in rows],key_request=[r['request'] for r in rows])
    return dict(entry,pack=pack,groups=groups,first_geometry=None)

def qualify(a,entry,out):
    # Two native pairs + one causal direct/dense pair + one stop-P control.
    # Full-row builder is used in production; subject-prefix builder crop OFF.
    B=entry['pack']['n_requests'];records=[];teachers=None
    for i,scale in enumerate((0.,.025),1):
        D=fixed(a,entry,scale);left=native(a,entry,D,True,range(B))
        gl={l:d.grad.clone() for l,d in D.items()}
        D=fixed(a,entry,scale)
        with full_native_reference(a):right=native(a,entry,D,True,range(B))
        gr={l:d.grad.clone() for l,d in D.items()}
        difference=dict(pair=i,scale=scale,loss_mean_abs=abs(left['native_sum']-right['native_sum'])/B,
            gradients=compare(gl,gr),cached='first_modified_linear_prefix_only',reference='full_native_hooks')
        write(out/f'native-pair-{i}.json',difference)
        require(difference['loss_mean_abs']<=5e-5 and all(r['pass_'] for r in difference['gradients'].values()),'NATIVE_FULL_CACHE_PARITY')
        records.append(difference);teachers=right['teachers']
    results=[]
    for route,checkpoint in [('direct',True),('dense',False)]:
        a.checkpoint_enabled=checkpoint;D=fixed(a,entry,.025)
        b=build(a,entry,D,0,route);policy,_=allocation(D,b['geometry'],entry['anchors'],'B')
        aux=actual(a,entry,b,D,teachers,range(B),[],[],[],True,route=route)
        outputs=[policy];adjoints=[policy.new_tensor(B)]
        for l,p in b['P'].items():
            if p.requires_grad:outputs.append(p);adjoints.append(aux['grad_P'][l])
        torch.autograd.backward(outputs,adjoints)
        gradients={l:d.grad+aux['grad_D'][l] for l,d in D.items()}
        results.append(dict(grad={l:g.detach().clone() for l,g in gradients.items()},
            loss=float(policy.detach())+aux['loss_sum']/B,
            keys={l:g['K'].detach().clone() for l,g in b['geometry'].items()},
            P={l:p.detach().clone() for l,p in b['P'].items()}))
        del b,policy,aux,D
    a.checkpoint_enabled=True
    left,right=results;difference=dict(pair=3,loss_mean_abs=abs(left['loss']-right['loss']),
        gradients=compare(left['grad'],right['grad']),direct_checkpoint_vs_dense_no_checkpoint=True,
        key_max={str(l):float((left['keys'][l]-right['keys'][l]).abs().max()) for l in a.sites})
    write(out/'causal-pair-3.json',difference)
    require(difference['loss_mean_abs']<=5e-5 and all(r['pass_'] for r in difference['gradients'].values()),'CAUSAL_D_P_CHECKPOINT_PARITY')
    # Runtime actual evidence of lower-write changes, not a third scientific arm.
    with torch.no_grad():zero=build(a,entry,fixed(a,entry,0.),0)
    movement={str(l):dict(K_delta=float((left['keys'][l]-zero['geometry'][l]['K']).norm()),
        P_delta=float((left['P'][l]-zero['P'][l]).norm())) for l in a.sites}
    write(out/'dependency-movement.json',dict(layers=movement,quality_gate=False))
    D=fixed(a,entry,.025);b=build(a,entry,D,0,qualification_stop_P=True)
    policy,_=allocation(D,b['geometry'],entry['anchors'],'B')
    aux=actual(a,entry,b,D,teachers,range(B),[],[],[],True)
    (B*policy).backward()
    stopped={l:d.grad+aux['grad_D'][l] for l,d in D.items()}
    write(out/'stop-P-control.json',dict(forward_mean_difference=abs(float(policy.detach())+aux['loss_sum']/B-left['loss']),
        gradient_difference=compare(stopped,left['grad']),negative_control_only=True,third_science_arm=False))
    del b,policy,aux,D,zero
    shuffled=singleton_permuted(entry);D=fixed(a,entry,.025)
    b=build(a,shuffled,D,0);policy,_=allocation(D,b['geometry'],entry['anchors'],'B')
    aux=actual(a,shuffled,b,D,teachers,range(B),[],[],[],True)
    outputs=[policy];adjoints=[policy.new_tensor(B)]
    for l,p in b['P'].items():
        if p.requires_grad:outputs.append(p);adjoints.append(aux['grad_P'][l])
    torch.autograd.backward(outputs,adjoints)
    permutation=dict(loss_mean_abs=abs(float(policy.detach())+aux['loss_sum']/B-left['loss']),
        gradients=compare({l:d.grad+aux['grad_D'][l] for l,d in D.items()},left['grad']),
        full_RW_row_order_reversed=True,wholeB_solve=True,microbatch=1,logical_B=B)
    write(out/'wholeB-microbatch-permutation.json',permutation)
    require(permutation['loss_mean_abs']<=5e-5 and all(v['pass_'] for v in permutation['gradients'].values()),'WHOLEB_MICROBATCH_PERMUTATION')
    result=dict(status='QUALIFIED_BOUNDED',pair_count=4,native_physical_F=4,native_physical_B=4,
        causal_candidate_F=5,causal_candidate_B=4,builder_crop=False,scope='BS2 development candidates only',
        tolerances=dict(loss_mean_abs=5e-5,gradient='RMS <= 1e-6+2e-3RMSreference'),
        actual_stop_P_negative_control=True,main_GPU_PASS=False)
    write(out/'receipt.json',result)
    return result
