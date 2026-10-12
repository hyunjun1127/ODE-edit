"""Three fixed B6 target cases, passive fit/replay and W0/B5 linear attribution.

CLI: python failure_case_probe.py BINDINGS.json NEW_OUTPUT
Same b255 failure_probe bindings plus reference_probe member pinned below and
mandatory fits member (archived fits/B06.json, 100 original returned-vector norms).
No writer, H restore, eval, new CP, selection or model intervention after restore.
"""
import argparse
import copy
import importlib
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import failure_probe as base
from official.runners.server1.common import read,verify,member,require
from official.experiments.prepare import write_new
from official.experiments.checkpoint import rng_snapshot
from official.diagnostics.fe_qwen_current.storage import state_digest,tensor_sha

INDICES=[2,31,93]
CASES=[7231,761,20473]
REFERENCE_SHA='babbe7371e1eebf3c8f0494387be2a6f7637c262642061ea62a75c63bb9cc7e9'


def compare(a,b):
    x=a.detach().double().cpu();y=b.detach().double().cpu();d=x-y
    return dict(maxabs=float(d.abs().max()),l2=float(d.norm()),relative=float(d.norm()/y.norm().clamp_min(1e-30)))


def norm_check(actual,reference):
    # Reduction-only allowance, fixed before observing this probe's real output.
    a=torch.tensor(actual,dtype=torch.float32);b=torch.tensor(reference,dtype=torch.float32)
    inf=torch.tensor(float('inf'))
    tolerance=2*max(float(torch.nextafter(a,inf)-a),float(torch.nextafter(b,inf)-b))
    error=abs(actual-reference)
    return dict(actual=actual,reference=reference,exact=actual==reference,absolute_error=error,
        allowed_two_fp32_ulps=tolerance,within_reduction_roundoff=error<=tolerance)


def linear_attribution(w0,w5,key,native_output):
    """Row-chunked FP64 arithmetic, no whole FP64 matrix or model mutation."""
    k=key.detach().double().cpu();a=[];b=[];delta=[]
    for start in range(0,w0.shape[0],64):
        x=w0[start:start+64].detach().double().cpu();y=w5[start:start+64].detach().double().cpu()
        a.append(x@k);b.append(y@k);delta.append((y-x)@k)
    a=torch.cat(a);b=torch.cat(b);d=torch.cat(delta);den=b.square().sum().clamp_min(1e-30)
    rounded=torch.nn.functional.linear(key.detach().to(w5.device,w5.dtype),w5.detach())
    indices=k.abs().topk(min(8,k.numel())).indices.tolist();columns=[]
    for j in indices:
        x=w0[:,j].detach().double().cpu();y=w5[:,j].detach().double().cpu()
        columns.append(dict(channel=j,key=float(k[j]),W0_column_norm=float(x.norm()),B5_column_norm=float(y.norm()),
            delta_column_norm=float((y-x).norm()),B5_column_output_norm=float((y*k[j]).norm()),
            B5_column_projection_fraction=float(torch.dot(y*k[j],b)/den)))
    return dict(W0_k_norm=float(a.norm()),B5_k_norm=float(b.norm()),deltaW_k_norm=float(d.norm()),
        decomposition=compare(a+d,b),W0_projection_fraction=float(torch.dot(a,b)/den),
        delta_projection_fraction=float(torch.dot(d,b)/den),
        native_vs_explicit_BF16=compare(native_output,rounded),
        native_vs_FP64=compare(native_output,b),columns=columns,
        counterfactual_linear_only=True,not_model_intervention=True)


def observe(model,tok,requests,cfg,contexts,*,supplier,compute_z,lookup,TraceDict,w0,extra=True):
    require(len(requests)==3 and sys.getprofile() is None,'THREE_CASE_PROFILE')
    names=[cfg.llms.layer_module_tmp.format(l) for l in (4,5)];down=cfg.llms.rewrite_module_tmp.format(5)
    modules=dict(model.named_modules());require(modules[down].bias is None,'NATIVE_BIAS_FREE')
    w5=modules[down].weight;phase='fit';fit_frame=None;captured={};fits=[];replays=[];current=-1
    versions={n:(id(p),p._version) for n,p in model.named_parameters()}
    hooks={n:tuple(m._forward_hooks) for n,m in model.named_modules()};rng=state_digest(rng_snapshot())
    def points(tensor,positions):
        return torch.stack([tensor[i,int(pos)].detach().clone() for i,pos in enumerate(positions)])
    def positions():
        if phase=='fit':return fit_frame.f_locals['lookup_idxs']
        r=requests[current];return [lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)]
    def l4hook(m,args,out):
        nonlocal current,captured
        if phase=='replay':current+=1;captured={}
    def keyhook(m,args,out):
        captured['key']=points(args[0],positions());captured['mlp']=points(base.hidden(out),positions())
    def blockhook(m,args,out):captured['block']=points(base.hidden(out),positions())
    handles=[modules[names[0]].register_forward_hook(l4hook),modules[down].register_forward_hook(keyhook),modules[names[1]].register_forward_hook(blockhook)]
    def replay_end(m,args,out):
        if phase=='replay':replays.append(dict(captured))
    handles.append(modules[names[1]].register_forward_hook(replay_end))
    def fit(*args,**kwargs):
        nonlocal fit_frame,phase,captured
        captured={};receipt={}
        def profile(frame,event,value):
            nonlocal fit_frame
            if frame.f_code is compute_z.__code__:
                if event=='call':fit_frame=frame
                elif event=='return':
                    v=frame.f_locals
                    receipt.update(vectors=dict(captured),lookups=[int(x) for x in v['lookup_idxs']],
                        native_returned_vector_norm=float(value.detach().float().norm()),
                        nll_each=v['nll_loss_each'].detach().float().cpu().tolist(),
                        probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(),iterations=int(v['it'])+1)
        sys.setprofile(profile)
        try:z=compute_z(*args,**kwargs)
        finally:sys.setprofile(None);fit_frame=None
        fits.append(receipt)
        if len(fits)==3:phase='replay'
        return z
    try:targets=supplier(model,tok,requests,cfg,contexts,compute_z=fit,find_fact_lookup_idx=lookup,TraceDict=TraceDict)
    finally:
        for h in handles:h.remove()
    require(len(fits)==len(replays)==3,'ALL_CASES_CAPTURED')
    rows=[]
    for i,r in enumerate(requests):
        f=fits[i];rp=replays[i];allfit=[]
        for row in range(len(f['lookups'])):
            allfit.append(dict(row=row,kind='KL' if row==len(f['lookups'])-1 else 'rewrite',lookup=f['lookups'][row],
                values={k:base.scalar_vector(v[row]) for k,v in f['vectors'].items()},
                linear=linear_attribution(w0,w5,f['vectors']['key'][row],f['vectors']['mlp'][row])))
        entry=dict(case_id=r['case_id'],local_index=INDICES[i],fit_nll_each=f['nll_each'],fit_probabilities=f['probabilities'],iterations=f['iterations'],fit_rows=allfit,
            native_returned_vector_norm=f['native_returned_vector_norm'],
            replay={k:base.scalar_vector(v[0]) for k,v in rp.items()},
            bare_fit_vs_replay={k:compare(f['vectors'][k][0],rp[k][0]) for k in rp},
            replay_linear=linear_attribution(w0,w5,rp['key'][0],rp['mlp'][0]),
            L4_target=base.scalar_vector(targets[4][:,i]),L5_target=base.scalar_vector(targets[5][:,i]),extra=[])
        if extra:
            pos=lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)
            inputs=tok(r['prompt'].format(r['subject']),return_tensors='pt').to(next(model.parameters()).device)
            for inject in (False,True):
                def edit(out,name):
                    if inject and name==names[0]:base.hidden(out)[0,pos]=targets[4][:,i]
                    return out
                with torch.no_grad(),TraceDict(model,layers=names+[down],retain_input=True,clone=True,edit_output=edit) as tr:model(**inputs)
                k=tr[down].input
                if isinstance(k,(tuple,list)):k=k[0]
                key=k[0,pos];mlp=base.hidden(tr[down].output)[0,pos]
                entry['extra'].append(dict(injected=inject,key=base.scalar_vector(key),mlp=base.scalar_vector(mlp),
                    L4=base.scalar_vector(base.hidden(tr[names[0]].output)[0,pos]),L5=base.scalar_vector(base.hidden(tr[names[1]].output)[0,pos]),
                    linear=linear_attribution(w0,w5,key,mlp)))
        rows.append(entry)
    require(versions=={n:(id(p),p._version) for n,p in model.named_parameters()},'WEIGHTS_CHANGED')
    require(hooks=={n:tuple(m._forward_hooks) for n,m in model.named_modules()},'HOOK_LEAK')
    require(rng==state_digest(rng_snapshot()),'RNG_CHANGED')
    return targets,dict(rows=rows,extra_forwards=6 if extra else 0,weights_rng_hooks_unchanged=True,
        writer_calls=0,history_consumed=False,checkpoint_saves=0,causal_historical_zero_attribution=False)


def run(bindings,output):
    b=read(bindings);v=base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    require(b['reference_probe']['sha256']==REFERENCE_SHA and b['reference_probe']['bytes']==556385,'EXACT_REFERENCE_PROBE')
    ref=read(verify(b['reference_probe']));require(ref['checkpoint']==b['checkpoint'],'REFERENCE_CP')
    old_fits=read(verify(b['fits']));require(len(old_fits['rows'])==100,'BOUND_ORIGINAL_FIT100')
    require(not out.exists(),'NEW_OUTPUT_ONLY')
    import transformers
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'NATIVE_RUNTIME')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    payload=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(payload['batch']==5 and payload['cursor']['edits']==500 and payload['identity']==v['pointer']['identity']
        and payload['cursor']['native_config']==v['runtime']['resolved'],'B5_PAYLOAD')
    cfg=OmegaConf.create(payload['cursor']['native_config']);require(int(cfg.gpu)==0,'GPU_INDEX')
    data=read(verify(c['native_data']));stream=read(verify(c['stream']))
    require(len(data)==len(stream)==2000 and payload['cursor']['case_ids']==[r['case_id'] for r in stream[:500]]
        and payload['cursor']['raw']==v['commit']['raw'],'B5_ORDER_RAW')
    assets=read(verify(c['model_assets']));snap=assets['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'ORIGINAL_BACKEND')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    params=dict(model.named_parameters());name=cfg.llms.rewrite_module_tmp.format(5)+'.weight'
    w0=params[name].detach().to('cpu',copy=True)  # ONLY original L5 down_proj, no full model copy.
    w0_sha=tensor_sha(w0);names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names,'SELECTED_SET')
    with torch.no_grad():
        for n,w in payload['weights'].items():
            require(w.dtype==params[n].dtype==torch.bfloat16 and w.shape==params[n].shape,'BF16_RESTORE')
            params[n].copy_(w);require(tensor_sha(params[n])==tensor_sha(w),'RESTORE_SHA')
    contexts=copy.deepcopy(payload['contexts']);rng=payload['rng'];del payload
    require(sum(len(group) for group in contexts)==6 and contexts[0]==['{}'],'ORIGINAL_SIX_CONTEXTS')
    sys.path.insert(0,lock['author_path'])
    z=importlib.import_module('algs.memit.compute_z');supplier=importlib.import_module('algs.memit.native_sequential_z');native=importlib.import_module('algs.memit.memit_main')
    for module in (z,supplier,native):require(Path(module.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE_IMPORT')
    requests=[copy.deepcopy(data[500+i]) for i in INDICES]
    require([r['case_id'] for r in requests]==CASES==[stream[500+i]['case_id'] for i in INDICES],'FIXED_CASES')
    for i,r in enumerate(requests):r['sample_idx']=i;r['target_new']=' '+r['target_new']
    base.restore_native_execution_state(native.set_random_seed,cfg.seed,rng)
    targets,result=observe(model,tok,requests,cfg,contexts,supplier=supplier.current_batch_targets,
        compute_z=z.compute_z,lookup=z.find_fact_lookup_idx,TraceDict=z.nethook.TraceDict,w0=w0)
    comparisons=[]
    for row in result['rows']:
        old=next(r for r in ref['replays'] if r['local_index']==row['local_index'])
        require(old['case_id']==row['case_id'],'REFERENCE_CASE')
        checks=dict(native_returned_L4=norm_check(row['native_returned_vector_norm'],old_fits['rows'][row['local_index']]['target_norm']),
            stacked_L4_different_stride=norm_check(row['L4_target']['norm'],old['L4_target']['norm']),
            L5_target=norm_check(row['L5_target']['norm'],old['L5_target']['norm']))
        tops={}
        for key,oldkey in [('key','L5_key'),('mlp','L5_mlp_output'),('block','L5_block')]:
            checks[key]=norm_check(row['replay'][key]['norm'],old[oldkey]['norm'])
            tops[key]=all(row['replay'][key][field]==old[oldkey][field] for field in ('top_indices','top_values'))
        comparisons.append(dict(case_id=row['case_id'],norm_checks=checks,top8_exact=tops,
            full_vector_historical_bitwise_observed=False))
    exact=all(all(c['exact'] for c in r['norm_checks'].values()) and all(r['top8_exact'].values()) for r in comparisons)
    compatible=all(all(c['within_reduction_roundoff'] for c in r['norm_checks'].values()) and all(r['top8_exact'].values()) for r in comparisons)
    require(tensor_sha(w0)==w0_sha,'W0_COPY_MUTATED');verify(b['checkpoint'])
    result.update(status='FIXED_CASE_PROBE_COMPLETE' if compatible else 'HISTORICAL_PARITY_MISMATCH_NO_ATTRIBUTION',
        comparisons=comparisons,historical_scalar_norms_exact=exact,historical_norms_within_two_fp32_ulps=compatible,
        bindings=member(bindings),reference_probe=b['reference_probe'],
        original_L5_weight_sha256=w0_sha,probe_source=member(__file__),no_benchmark_or_recovery_claim=True,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,attention_backend=model.config._attn_implementation,
        cuda_device_name=torch.cuda.get_device_name(0),cuda_capability=list(torch.cuda.get_device_capability(0)),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        tokenizer_pad_id=tok.pad_token_id,tokenizer_padding_side=tok.padding_side))
    out.mkdir(parents=True);write_new(out/'case-probe.json',result)
    return member(out/'case-probe.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output))
