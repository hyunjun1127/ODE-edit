"""FE equations in an explicit no-cache/no-save Llama adapter.

Upstream z_methods/compute_z.py non-LTI and FE-memit_main.py are the
authority. Physical MB1 checkpoint recomputes are counted, not extra fits.
"""
import ast
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from . import *

def lookup_function(upstream):
    path=Path(upstream)/'locate_edit_utils/repr_tools.py'
    tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='get_words_idxs_in_templates')
    # Only the exact upstream token-index function, no package-wide imports.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns={};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    return ns[fn.name]

def pack(tok,record,contexts,lookup):
    r=record['requested_rewrite'];target=tok(' '+r['target_new']['str'],return_tensors='pt')['input_ids'][0]
    if len(target) and int(target[0]) in (tok.bos_token_id,tok.unk_token_id):target=target[1:]
    require(len(target)>0 and [len(g) for g in contexts]==[1,5],'TARGET_CONTEXT')
    templates=[s.format(r['prompt'])+tok.decode(target[:-1]) for g in contexts for s in g]+['{} is a']
    positions=[v[0] for v in lookup(tok,templates,[r['subject']]*7,'last')]
    tokens=dict(tok([s.format(r['subject']) for s in templates],return_tensors='pt',padding=True))
    labels=torch.full_like(tokens['input_ids'],-100)
    for i in range(6):
        n=int(tokens['attention_mask'][i].sum());require(n>=len(target),'TARGET_UNDERFLOW');labels[i,n-len(target):n]=target
    require(all(0<=p<int(tokens['attention_mask'][i].sum()) for i,p in enumerate(positions)),'LOOKUP_RANGE')
    identity=digest(dict(tokens={k:v.tolist() for k,v in tokens.items()},lookup=positions,target=target.tolist()))
    return dict(tokens=tokens,labels=labels,lookup=positions,identity=identity,target=target)

def stop_reason(total,iteration):
    require(torch.isfinite(torch.as_tensor(total)).all().item(),'NONFINITE_TOTAL')
    return 'TOTAL_BELOW_005' if total<.05 else ('BUDGET_EXHAUSTED' if iteration==34 else None)

def hidden(output):
    return output[0] if isinstance(output,tuple) else output

def replace(output,value):
    return (value,*output[1:]) if isinstance(output,tuple) else value

class Adapter:
    def __init__(self,model,tok,contexts,upstream):
        self.model,self.tok,self.contexts=model,tok,contexts;self.device=next(model.parameters()).device
        self.weights={l:model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
        self.lookup=lookup_function(upstream);self.calls=dict(fit_logical=0,fit_physical=0,recompute=0,replay=0,key=0,canonical=0)
        model.requires_grad_(False)
        require(all(p.dtype==torch.float32 for p in model.parameters()),'MODEL_FP32')
    def guard(self):
        selected={id(w) for w in self.weights.values()}
        return [(n,p.data_ptr(),p._version) for n,p in self.model.named_parameters() if id(p) not in selected]
    def versions(self):return [(n,p.data_ptr(),p._version) for n,p in self.model.named_parameters()]
    def hook_signature(self):return [(n,tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in self.model.named_modules() if m._forward_hooks or m._forward_pre_hooks]
    def fit(self,record,index,out,expected_pack):
        started=time.monotonic();before=self.versions();hooks=self.hook_signature();p=pack(self.tok,record,self.contexts,self.lookup)
        require(p['identity']==expected_pack,'FIT_TOKEN_IDENTITY')
        delta=torch.zeros(self.model.config.hidden_size,device=self.device,requires_grad=True)
        opt=torch.optim.Adam([delta],lr=.1);initial=None;teacher=None;trace=[];row_calls={};qualification={}
        for iteration in range(35):
            opt.zero_grad();parts=[]
            for row in range(7):
                # Default arguments freeze row/iteration for delayed recomputation.
                def row_loss(d,row=row,iteration=iteration):
                    nonlocal initial,teacher
                    key=(iteration,row);row_calls[key]=row_calls.get(key,0)+1
                    self.calls['fit_physical']+=1
                    if row_calls[key]>1:self.calls['recompute']+=1
                    pos=p['lookup'][row]
                    def inject(module,args,output):
                        nonlocal initial
                        value=hidden(output);require(value.ndim==3 and value.shape[0]==1,'L4_HOOK_LAYOUT')
                        if initial is None:initial=value[0,pos].detach().clone();require(float(initial.norm())>0,'ZERO_INITIAL')
                        value=value.clone();value[0,pos]+=d;return replace(output,value)
                    captured={}
                    def final_hook(module,args,output):captured['h']=hidden(output)
                    h=self.model.model.layers[4].register_forward_hook(inject)
                    h31=self.model.model.layers[31].register_forward_hook(final_hook) if index==0 and iteration==0 and row==0 else None
                    try:
                        tokens={k:v[row:row+1].to(self.device) for k,v in p['tokens'].items()}
                        logits=self.model(**tokens,use_cache=False).logits.float()
                        if row<6:
                            lab=p['labels'][row].to(self.device);mask=lab!=-100
                            loss=-logits[0,mask].log_softmax(-1).gather(1,lab[mask,None]).mean()
                            if h31 is not None and row_calls[key]==1:
                                # Observational head check must not add a forward-only
                                # autograd branch inside activation checkpointing.
                                with torch.no_grad():
                                    ref=self.model.lm_head(self.model.model.norm(captured['h'].detach()))
                                    qualification['same_forward_head_max_abs']=float((ref-logits.detach()).abs().max())
                                    ref_loss=-ref[0,mask].log_softmax(-1).gather(1,lab[mask,None]).mean()
                                    qualification['same_forward_nll_abs']=float((ref_loss-loss.detach()).abs())
                                qualification['loss_layer']=31
                            return loss
                        logp=logits[0,pos].log_softmax(-1)
                        if teacher is None:teacher=logp.detach().clone()
                        return F.kl_div(teacher[None,:],logp[None,:],log_target=True,reduction='batchmean')
                    finally:
                        h.remove()
                        if h31 is not None:h31.remove()
                parts.append(checkpoint(row_loss,delta,use_reentrant=False,preserve_rng_state=True))
            nll=torch.stack(parts[:6]).mean();kl=.0625*parts[6];norm=.5*delta.norm()/initial.norm().square()
            total=nll+kl+norm;self.calls['fit_logical']+=1
            vals=dict(iteration=iteration,nll=float(nll.detach()),kl=float(kl.detach()),norm=float(norm.detach()),total=float(total.detach()),delta_norm=float(delta.detach().norm()))
            reason=stop_reason(vals['total'],iteration);vals['stop']=reason
            require(all(torch.isfinite(x).all().item() for x in (delta,nll,kl,norm)),'NONFINITE_FIT')
            if reason:
                trace.append(vals);break
            total.backward();require(delta.grad is not None and bool(torch.isfinite(delta.grad).all()),'FIT_GRADIENT')
            vals['gradient_norm']=float(delta.grad.norm());opt.step()
            with torch.no_grad():
                cap=.75*initial.norm()
                if delta.norm()>cap:delta.mul_(cap/delta.norm())
            trace.append(vals)
        result=(initial+delta.detach()).detach().cpu();require(bool(torch.isfinite(result).all()),'NONFINITE_TARGET')
        require(self.versions()==before and self.hook_signature()==hooks,'FIT_W0_MUTATION')
        receipt=dict(index=index,case_id=record['case_id'],record=digest(record),pack=p['identity'],iterations=trace,
            evaluations=len(trace),Adam_updates=len(trace)-1,stop=reason,target_hash=tensor_sha(result),target_norm=float(result.norm()),
            initial_norm=float(initial.norm()),delta_norm=float(delta.detach().norm()),teacher_hash=tensor_sha(teacher),
            seconds=time.monotonic()-started,physical_forwards=sum(row_calls.values()),checkpoint_recompute=sum(max(n-1,0) for n in row_calls.values()),
            physical_MB=1,logical_contexts=7,qualification=qualification,W0_unchanged=True,table_reused=True)
        write(out,receipt);return result

    @torch.no_grad()
    def replay(self,record,z4):
        r=record['requested_rewrite'];tok=dict(self.tok(r['prompt'].format(r['subject']),return_tensors='pt'))
        pos=self.lookup(self.tok,[r['prompt']],[r['subject']],'last')[0][0];found={};handles=[]
        for layer in LAYERS:
            def hook(module,args,output,layer=layer):
                v=hidden(output);require(v.shape[0]==1 and 0<=pos<v.shape[1],'REPLAY_LAYOUT')
                if layer==4:v=v.clone();v[0,pos]=z4.to(self.device)
                found[layer]=v[0,pos].detach().cpu().clone();return replace(output,v)
            handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        try:self.model(**{k:v.to(self.device) for k,v in tok.items()},use_cache=False);self.calls['replay']+=1
        finally:
            for h in handles:h.remove()
        require(set(found)==set(LAYERS) and torch.equal(found[4],z4),'ABSOLUTE_REPLAY_IDENTITY')
        require(all(bool(torch.isfinite(v).all()) for v in found.values()),'NONFINITE_REPLAY');return found

    @torch.no_grad()
    def capture(self,records,layer):
        raw=[];canonical=[]
        for record in records:
            r=record['requested_rewrite'];templates=[t.format(r['prompt']) for group in self.contexts for t in group]
            positions=self.lookup(self.tok,templates,[r['subject']]*6,'last');keys=[]
            for i,(template,where) in enumerate(zip(templates,positions)):
                pos=where[0];got={};tokens=dict(self.tok(template.format(r['subject']),return_tensors='pt'))
                def kh(module,args):got['k']=args[0][0,pos].detach().clone()
                def hh(module,args,output):got['h']=hidden(output)[0,pos].detach().clone()
                h1=self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(kh)
                h2=self.model.model.layers[layer].register_forward_hook(hh)
                try:self.model(**{k:v.to(self.device) for k,v in tokens.items()},use_cache=False);self.calls['key']+=1
                finally:h1.remove();h2.remove()
                keys.append(got['k'])
                if i==0:canonical.append(got['h'])
            # Exact original nested mean, not a six-context equal mean.
            raw.append(torch.stack((keys[0],torch.stack(keys[1:]).mean(0))).mean(0))
        return torch.stack(raw,1),torch.stack(canonical,1)

@torch.no_grad()
def solve_write(weight,H,C0,K,R):
    require(K.dtype==R.dtype==weight.dtype==H.dtype==C0.dtype==torch.float32,'WRITER_DTYPES')
    def stamp():
        if K.is_cuda:torch.cuda.synchronize(K.device)
        return time.monotonic()
    started=stamp();k=K.double();r=R.double();gram=k@k.T
    A=gram+H.to(K.device).double()+15000*C0.to(K.device).double()
    rhs=k@r.T;prepared=stamp();solution=torch.linalg.solve(A,rhs);solved=stamp()
    require(bool(torch.isfinite(solution).all()),'NONFINITE_SOLVE')
    residual=float((A@solution-rhs).norm()/rhs.norm().clamp_min(1e-300));delta=solution.T
    old=weight.detach().clone();before_H=tensor_sha(H);verified=stamp()
    # add_ computes FP64 sum and casts to the CPU FP32 destination, as upstream +=.
    H.add_(gram.cpu());history_done=stamp();weight.copy_(weight+delta);write_done=stamp()
    require(bool(torch.isfinite(H).all()) and bool(torch.isfinite(weight).all()),'NONFINITE_WRITE')
    actual=weight.double()-old.double()
    return dict(seconds=time.monotonic()-started,prepare_seconds=prepared-started,solve_seconds=solved-prepared,
        residual_and_hash_seconds=verified-solved,history_seconds=history_done-verified,write_seconds=write_done-history_done,
        relative_solve_residual=residual,residual_policy='RECORD_ONLY_IF_FINITE',
        residual_norm=float(R.norm()),update_FP64_norm=float(delta.norm()),actual_norm=float(actual.norm()),
        rounding_max_abs=float((actual-delta).abs().max()),realization_max_abs=float((actual@k-r).abs().max()),
        H_before=before_H,H_after=tensor_sha(H),H_norm=float(H.norm()),history_appends=1,
        H_timing='PREWRITE_KEY_FP64_GRAM_TO_CPU_FP32',weight_add='FP64_SUM_THEN_FP32_DESTINATION',divisor=1)
