"""Pinned singleton compute_z + explicit 4.57 tensor/tuple interface shim.

No writer, history, HJ solver, or native scientific hyperparameter is changed.
First singleton candidate/Adam state is reused in RAM, never recomputed for
cached routes. Qualified independent batching keeps one Adam per request.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from project.run_scripts.jlz_pilot.run import detached
from project.run_scripts.single_layer_mechanism_first.z_hook import (
    prepare_batch,capture_prefix,suffix_hidden,native_losses,normalize_requests)

SOURCE=Path('/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-22-alphaedit-original-failure-audit/official')

class TraceDict(dict):
    """Only adapt tensor block return to original source's tuple[0] interface."""
    def __init__(self,module,layers,edit_output,**kw):
        super().__init__();self.handles=[]
        for name in dict.fromkeys(layers):
            item=SimpleNamespace();self[name]=item
            def hook(mod,args,out,name=name,item=item):
                tensor=isinstance(out,torch.Tensor)
                wrapped=(out,) if tensor else out
                changed=edit_output(wrapped,name)
                item.output=changed
                return changed[0] if tensor else changed
            self.handles.append(module.get_submodule(name).register_forward_hook(hook))
    def __enter__(self):return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()

def parameter(model,name):
    try:return model.get_parameter(name)
    except AttributeError as exc:raise LookupError(name) from exc

def load_native(event,iteration):
    source=(SOURCE/'AlphaEdit/compute_z.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f'
    reprsource=(SOURCE/'rome/repr_tools.py').read_text()
    assert hashlib.sha256(reprsource.encode()).hexdigest()=='b3f7f07358437e7927d556b64666bf0e67d03eb6489c00613a8865e83372849f'
    ns=dict(torch=torch,np=np,_event=event,_iteration=iteration,
        nethook=SimpleNamespace(get_module=lambda m,n:m.get_submodule(n),get_parameter=parameter,
            set_requires_grad=lambda flag,m:[p.requires_grad_(flag) for p in m.parameters()],TraceDict=TraceDict))
    def extract(text,names):
        nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
        return ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    rep=extract(reprsource,{'get_words_idxs_in_templates'})
    exec(compile(ast.fix_missing_locations(rep),str(SOURCE/'rome/repr_tools.py'),'exec'),ns)
    ns['repr_tools']=SimpleNamespace(get_words_idxs_in_templates=ns['get_words_idxs_in_templates'])
    tree=extract(source,{'compute_z','find_fact_lookup_idx'});hits=[0,0,0]
    for node in ast.walk(tree):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='it':
            body=[ast.parse('_iteration(it)').body[0]]
            for statement in node.body:
                if isinstance(statement,ast.If) and ast.unparse(statement.test)=='delta.norm() > max_norm':
                    body.append(ast.parse("_event('pre_clamp',locals())").body[0])
                body.append(statement)
                if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in statement.targets) and isinstance(statement.value,ast.BinOp):
                    body.append(ast.parse("_event('loss',locals())").body[0]);hits[0]+=1
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='backward':
                    body.append(ast.parse("_event('gradient',locals())").body[0]);hits[1]+=1
            body.append(ast.parse("_event('after',locals())").body[0]);node.body=body;hits[2]+=1
    assert hits==[1,1,1]
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE/'AlphaEdit/compute_z.py')+':scalar-observer-and-tensor-interface','exec'),ns)
    hp=SimpleNamespace(**json.loads((SOURCE/'hparams/AlphaEdit/Llama3-8B.json').read_text()))
    assert (hp.layers,hp.v_loss_layer,hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay,hp.clamp_norm_factor,hp.kl_factor)==([4,5,6,7,8],31,25,.1,.5,.75,.0625)
    return ns['compute_z'],ns['find_fact_lookup_idx'],hp

def original(model,tok,requests,contexts):
    states=[];reports=[];current={};it=[0]
    def event(kind,local):
        delta=local['delta'];n=local['it']
        if kind=='loss':
            current['final_delta']=delta.detach().clone()
            current['trace'].append(dict(iteration=n,total=float(local['loss']),nll=float(local['nll_loss']),kl=float(local['kl_loss']),decay=float(local['weight_decay']),delta_norm=float(delta.norm())))
            if n==0:
                current.update(initial=local['target_init'].detach().clone(),teacher=local['kl_distr_init'].detach().clone(),first_delta=delta.detach().clone())
        if kind=='gradient':
            current['updates']+=1
            if n in (0,1):current.setdefault('gradients',{})[n]=delta.grad.detach().clone()
        if kind=='pre_clamp':
            current['clamp'].append(bool(delta.norm()>local['max_norm']))
            current['near_clamp'] |= abs(float(delta.norm()/local['max_norm'])-1)<=1e-4
        if kind=='after':
            if n==0:
                current['first_delta']=delta.detach().clone();current['first_adam']=copy.deepcopy(local['opt'].state_dict())
    native,lookup,hp=load_native(event,lambda n:it.__setitem__(0,n));layer=hp.layers[-1]
    for request in normalize_requests(requests):
        current.clear();current.update(trace=[],updates=0,clamp=[],near_clamp=False);prefix={}
        module=model.get_submodule(hp.layer_module_tmp.format(layer))
        def before(mod,args,kwargs):
            if any(kwargs.get(k) is not None for k in ('past_key_values','past_key_value')):
                raise RuntimeError('MUTABLE_NATIVE_KV_CACHE_FORBIDDEN')
            if it[0]==0 and 'hidden' not in prefix:prefix.update(args=detached(args[1:]),kwargs=detached(kwargs))
        def after(mod,args,out):
            if it[0]==0 and 'hidden' not in prefix:prefix['hidden']=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
        handles=[module.register_forward_pre_hook(before,with_kwargs=True),module.register_forward_hook(after)]
        it[0]=0;start=time.monotonic()
        try:target=native(model,tok,request,hp,layer,contexts)
        finally:
            for handle in handles:handle.remove()
        state=copy.deepcopy(current);state.update(prefix=prefix,target=target.detach().clone(),request=request)
        states.append(state)
        reports.append(dict(case_id=request['case_id'],candidates=len(state['trace']),updates=state['updates'],trace=state['trace'],clamp=state['clamp'],seconds=time.monotonic()-start,
            stop='TOTAL_LOSS_LT_0.05' if state['trace'][-1]['total']<.05 else 'MAX_25',first_original_singleton=True))
    return states,reports,hp,lookup

def cached(model,tok,states,hp,lookup,contexts,batched=False):
    device=next(model.parameters()).device;requests=[s['request'] for s in states];layer=hp.layers[-1]
    batches=[prepare_batch(tok,requests,contexts,hp,lookup,device)] if batched else [prepare_batch(tok,[r],contexts,hp,lookup,device) for r in requests]
    prefixes=[capture_prefix(model,hp,layer,batches[0]['tokens'])] if batched else [s['prefix'] for s in states]
    deltas=[s['first_delta'].detach().clone().requires_grad_() for s in states];opts=[]
    done=[s['trace'][0]['total']<.05 for s in states];traces=[[dict(s['trace'][0])] for s in states]
    updates=[int(not d) for d in done];clamps=[s['clamp'][:1] for s in states];fixed_grad=[]
    for s,d in zip(states,deltas):
        opt=torch.optim.Adam([d],lr=hp.v_lr)
        if 'first_adam' in s:opt.load_state_dict(copy.deepcopy(s['first_adam']))
        opts.append(opt)
    start=time.monotonic();physical_f=physical_b=0
    ambiguous=any(s['near_clamp'] or any(abs(r['total']-.05)<=1e-4 for r in s['trace']) for s in states)
    for iteration in range(1,25):
        if all(done):break
        for opt in opts:opt.zero_grad()
        groups=[list(range(len(states)))] if batched else [[i] for i in range(len(states)) if not done[i]]
        for indices in groups:
            b=batches[0] if batched else batches[indices[0]];p=prefixes[0] if batched else prefixes[indices[0]]
            delta=torch.stack([deltas[i] for i in indices]);initial=torch.stack([states[i]['initial'] for i in indices]);teacher=torch.cat([states[i]['teacher'] for i in indices])
            lh,fh=suffix_hidden(model,hp,layer,p,delta,b)
            losses,nll,kl,decay,_=native_losses(model,hp,b,lh,fh,delta,initial,teacher);physical_f+=1
            active=[]
            for j,i in enumerate(indices):
                if done[i]:continue
                traces[i].append(dict(iteration=iteration,total=float(losses[j]),nll=float(nll[j]),kl=float(kl[j]),decay=float(decay[j]),delta_norm=float(deltas[i].norm())))
                ambiguous |= abs(float(losses[j])-.05)<=1e-4
                if losses[j]<.05:done[i]=True
                elif iteration<24:active.append((j,i))
            if active:
                (losses[active[0][0]] if len(active)==1 else torch.stack([losses[j] for j,i in active]).sum()).backward();physical_b+=1
                for j,i in active:
                    g=deltas[i].grad
                    if g is None or not bool(torch.isfinite(g).all()):raise FloatingPointError('NATIVE_CACHE_GRADIENT')
                    if iteration==1 and 1 in states[i].get('gradients',{}):
                        ref=states[i]['gradients'][1];fixed_grad.append(dict(case_id=states[i]['request']['case_id'],relative=float((g-ref).norm()/ref.norm().clamp_min(1)),maxabs=float((g-ref).abs().max())))
                    opts[i].step();updates[i]+=1
                    with torch.no_grad():
                        radius=hp.clamp_norm_factor*states[i]['initial'].norm();hit=bool(deltas[i].norm()>radius)
                        ambiguous |= abs(float(deltas[i].norm()/radius)-1)<=1e-4
                        if hit:deltas[i].copy_(deltas[i]*radius/deltas[i].norm())
                        clamps[i].append(hit)
    reports=[];passed=True
    for i,s in enumerate(states):
        refdelta=s['final_delta'];rel=float((deltas[i].detach()-refdelta).norm()/refdelta.norm().clamp_min(1))
        discrete=len(traces[i])==len(s['trace']) and updates[i]==s['updates'] and clamps[i]==s['clamp'] and (traces[i][-1]['total']<.05)==(s['trace'][-1]['total']<.05)
        passed &= discrete and rel<=1e-4
        reports.append(dict(case_id=s['request']['case_id'],candidates=len(traces[i]),updates=updates[i],trace=traces[i],clamp=clamps[i],returned_delta_relative=rel,discrete_exact=discrete))
    passed &= all(g['relative']<=1e-5 and g['maxabs']<=1e-5 for g in fixed_grad) and not ambiguous
    return dict(status='PASS' if passed else 'UNQUALIFIED',route='qualified_independent_batching' if batched else 'cached_singleton',requests=reports,
        first_candidate_and_Adam='REUSE exact original singleton RAM state, no extra teacher forward',first_fixed_delta_gradients=fixed_grad,
        uncertified_near_stop_or_clamp=ambiguous,near_boundary_policy='UNQUALIFIED, no hidden reference call',
        seconds=time.monotonic()-start,physical_suffix_forwards=physical_f,physical_suffix_backwards=physical_b,
        extra_reference_request_forwards=len(states) if batched else 0,extra_reference_request_backwards=0,
        actual_multi_token='NOT_COVERED: first4 single-token; CPU synthetic only',history_appends=0,weight_writes=0)
