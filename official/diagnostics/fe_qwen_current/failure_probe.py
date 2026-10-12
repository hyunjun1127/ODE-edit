"""B5->B6 target-only diagnostic, never a writer or benchmark.

CLI: python -m official.diagnostics.fe_qwen_current.failure_probe BINDINGS.json NEW_OUTPUT
Bindings: config, lock, checkpoint, pointer, runtime, commit, writer (path/bytes/sha256
members from the OLD L2-10 attempt). Optional fits member is archived fits/B06.json.
All old absolute member paths must remain valid. batch=6 is fixed, not a CLI override.
The checkpoint SHA/size/source are pinned below. Root runs this on rent only.
No history restore/copy, CP save, selection, evaluation, or scheduler operations.
"""
import argparse
import copy
import importlib
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from official.experiments.prepare import digest,write_new
from official.experiments.checkpoint import rng_restore,rng_snapshot
from official.runners.server1.common import read,verify,member,require
from official.diagnostics.fe_qwen_current.storage import tensor_sha,state_digest

SOURCE='ac05e8482c322607490521fce084d9235bb68e21'
CP_SHA='b4a91af71bac225ecdd44e8547c946a5efbbf8c045c9be6c24453b305dc231cf'
CP_BYTES=7856478169


def hidden(value):
    return value[0] if isinstance(value,(tuple,list)) else value


def scalar_vector(value):
    v=value.detach().float().reshape(-1)
    k=min(8,v.numel());idx=v.abs().topk(k).indices
    return dict(norm=float(v.norm()),top_indices=idx.cpu().tolist(),top_values=v[idx].cpu().tolist())


def diff(a,b):
    return float((a.detach().float()-b.detach().float()).abs().max())


def probe_targets(model,tok,requests,cfg,contexts,*,supplier,compute_z,lookup,TraceDict,extra=True):
    """Observe the actual supplier/fit functions; no substitutes for their math."""
    require(sys.getprofile() is None,'EXISTING_PROFILE')
    layers=list(cfg.llms.layers);first=layers[0];later=layers[1]
    modules=dict(model.named_modules());names=[cfg.llms.layer_module_tmp.format(l) for l in layers]
    state=dict(phase='fit',index=-1,plain=False);fits=[];replays=[];initials=[];fitted=[]
    versions={n:(id(p),p._version) for n,p in model.named_parameters()}
    hooks_before={n:tuple(m._forward_hooks) for n,m in model.named_modules()}
    rng_before=state_digest(rng_snapshot());handles=[]
    def pre_l4(module,args,out):
        if state['phase']!='replay':return
        i=state['index']+1;state['index']=i;r=requests[i]
        pos=lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)
        h=hidden(out)[0,pos].detach();z=fitted[i]
        replays.append(dict(case_id=r['case_id'],local_index=i,lookup=pos,
            pre_L4=scalar_vector(h),fit_init_vs_replay_pre_maxabs=diff(initials[i],h),
            fit_init_vs_replay_pre_l2=float((initials[i].float()-h.float()).norm()),
            fitted_fp32_vs_effective_bf16_maxabs=diff(z,z.to(h.dtype)),
            fitted_fp32_vs_effective_bf16_l2=float((z.float()-z.to(h.dtype).float()).norm()),
            effective_L4=scalar_vector(z.to(h.dtype))))
    def key_hook(module,args,out):
        if state['phase']!='replay':return
        r=replays[-1];pos=r['lookup']
        r['L5_key']=scalar_vector(args[0][0,pos]);r['L5_mlp_output']=scalar_vector(hidden(out)[0,pos])
    def block_hook(module,args,out):
        if state['phase']=='replay':replays[-1]['L5_block']=scalar_vector(hidden(out)[0,replays[-1]['lookup']])
    handles.append(modules[names[0]].register_forward_hook(pre_l4))
    handles.append(modules[cfg.llms.rewrite_module_tmp.format(later)].register_forward_hook(key_hook))
    handles.append(modules[names[1]].register_forward_hook(block_hook))
    def fit(*args,**kwargs):
        captured={}
        def profile(frame,event,value):
            if frame.f_code is compute_z.__code__ and event=='return':
                v=frame.f_locals;init=v['target_init'].detach().clone();delta=v['delta'].detach()
                # Final fit TraceDict is still referenced in native locals.
                actual=hidden(v['tr'][names[0]].output)[0,v['lookup_idxs'][0]].detach()
                captured.update(init=init,receipt=dict(case_id=requests[len(fits)]['case_id'],
                    local_index=len(fits),init=scalar_vector(init),delta=scalar_vector(delta),
                    lookups=[int(x) for x in v['lookup_idxs']],nll_each=v['nll_loss_each'].detach().float().cpu().tolist(),
                    probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(),
                    iterations=int(v['it'])+1,returned_dtype=str(value.dtype),
                    final_fit_L4=scalar_vector(actual),fit_effective_vs_return_cast_maxabs=diff(actual,value.to(actual.dtype))))
        sys.setprofile(profile)
        try:value=compute_z(*args,**kwargs)
        finally:sys.setprofile(None)
        require(bool(captured),'NATIVE_FIT_LOCALS_MISSING')
        initials.append(captured['init']);fitted.append(value.detach().clone());fits.append(captured['receipt'])
        if len(fits)==len(requests):state['phase']='replay'
        return value
    try:
        targets=supplier(model,tok,requests,cfg,contexts,compute_z=fit,find_fact_lookup_idx=lookup,TraceDict=TraceDict)
    finally:
        for handle in handles:handle.remove()
    require(len(fits)==len(replays)==len(requests),'ALL_TARGETS_OBSERVED')
    for i,r in enumerate(replays):
        r['L4_target']=scalar_vector(targets[first][:,i]);r['L5_target']=scalar_vector(targets[later][:,i])
        r['ratio']=r['L5_target']['norm']/max(r['L4_target']['norm'],1e-30)
    target_rng=state_digest(rng_snapshot());extra_rows=[]
    chosen=sorted(range(len(replays)),key=lambda i:replays[i]['ratio'],reverse=True)[:2]
    control=next((i for i,r in enumerate(replays) if r['lookup']==0 and i not in chosen),None)
    if control is None:
        remaining=[i for i in range(len(requests)) if i not in chosen]
        control=min(remaining,key=lambda i:(replays[i]['lookup'],i)) if remaining else None
    if control is not None:chosen.append(control)
    if extra:
        for i in chosen:
            req=requests[i];pos=replays[i]['lookup'];inputs=tok(req['prompt'].format(req['subject']),return_tensors='pt').to(next(model.parameters()).device)
            for inject in (False,True):
                def edit(out,name):
                    if inject and name==names[0]:hidden(out)[0,pos]=targets[first][:,i]
                    return out
                traced_names=[names[0],names[1],cfg.llms.rewrite_module_tmp.format(later)]
                with torch.no_grad(),TraceDict(model,layers=traced_names,retain_input=True,edit_output=edit,clone=True) as tr:
                    model(**inputs)
                k=tr[traced_names[2]].input
                if isinstance(k,(tuple,list)):k=k[0]
                extra_rows.append(dict(case_id=req['case_id'],local_index=i,injected=inject,lookup=pos,
                    L4=scalar_vector(hidden(tr[names[0]].output)[0,pos]),
                    L5=scalar_vector(hidden(tr[names[1]].output)[0,pos]),
                    key=scalar_vector(k[0,pos]),mlp=scalar_vector(hidden(tr[traced_names[2]].output)[0,pos])))
    require(versions=={n:(id(p),p._version) for n,p in model.named_parameters()},'WEIGHTS_CHANGED')
    require(hooks_before=={n:tuple(m._forward_hooks) for n,m in model.named_modules()},'HOOK_LEAK')
    require(rng_before==target_rng==state_digest(rng_snapshot()),'RNG_CHANGED')
    return targets,dict(fits=fits,replays=replays,extra=extra_rows,extra_forwards=len(extra_rows),
        target_norms={str(l):float(v.detach().double().norm()) for l,v in targets.items()},
        target_norms_native_fp32={str(l):float(v.detach().float().norm()) for l,v in targets.items()},
        model_versions_unchanged=True,hooks_unchanged=True,rng_unchanged=True,
        no_writer=True,no_evaluation=True,no_checkpoint_save=True,no_causal_lookup0_attribution=True)


def validate_inputs(b):
    values={k:read(verify(b[k])) for k in ('config','lock','pointer','runtime','commit','writer')}
    c=values['config'];lock=values['lock'];ptr=values['pointer'];rt=values['runtime'];commit=values['commit']
    require(lock['source_commit']==c['source']==SOURCE and c['L2']==10,'EXACT_OLD_SOURCE_ARM')
    require(c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'})
        and b['config'] in lock['configs'],'CONFIG_BINDING')
    cp=b['checkpoint'];require(cp['sha256']==CP_SHA and cp['bytes']==CP_BYTES,'EXACT_B5_CP')
    require(all(ptr[k]==cp[k] for k in ('path','bytes','sha256')) and ptr['batch']==5
        and ptr['identity']==rt['identity']==commit['identity'] and commit['batch']==5
        and ptr['identity']['config_sha256']==c['config_sha256'],'B5_POINTER_RAW_BINDING')
    require(commit['checkpoint']['sha256']==CP_SHA,'B5_COMMIT_CP')
    require(values['writer']['batch']==6,'EXACT_B6_WRITER')
    for m in commit['raw']:verify(m)
    for m in lock['source_members']+lock['author_members']+lock['model_members']:verify(m)
    for key in ('stream','native_data','model_assets'):verify(c[key])
    native=rt['resolved']
    require(native['model_dtype']=='bfloat16' and native['algs']['L2']==10
        and native['llms']['clamp_norm_factor']==1 and native['bs']==100
        and native['num_edits']==2000 and native['seed']==0,'ORIGINAL_NATIVE_CONFIG')
    return values


def run(bindings,output):
    b=read(bindings);v=validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    require(not out.exists(),'NEW_OUTPUT_ONLY');out.mkdir(parents=True)
    import transformers
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'NATIVE_RUNTIME')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1_ONLY')
    cp=b['checkpoint'];payload=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    require(payload['batch']==5 and payload['cursor']['edits']==500 and payload['identity']==v['pointer']['identity']
        and payload['cursor']['native_config']==v['runtime']['resolved'],'EXACT_PAYLOAD_IDENTITY')
    native_rows=read(verify(c['native_data']));rows=read(verify(c['stream']))
    require(len(rows)==len(native_rows)==2000 and payload['cursor']['case_ids']==[r['case_id'] for r in rows[:500]],'B5_ORDER')
    require(payload['cursor']['raw']==v['commit']['raw'],'B5_PAYLOAD_RAW')
    cfg=OmegaConf.create(payload['cursor']['native_config'])
    require(int(cfg.gpu)==0,'ORIGINAL_GPU_INDEX')
    assets=read(verify(c['model_assets']));snap=assets['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    params=dict(model.named_parameters());names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names,'SELECTED_WEIGHT_SET')
    untouched={n:(id(p),p._version) for n,p in params.items() if n not in names}
    with torch.no_grad():
        for n,w in payload['weights'].items():
            require(w.dtype==params[n].dtype==torch.bfloat16 and w.shape==params[n].shape,'EXACT_BF16_RESTORE')
            params[n].copy_(w);require(tensor_sha(params[n])==tensor_sha(w),'RESTORE_SHA')
    contexts=copy.deepcopy(payload['contexts']);rng=payload['rng']
    context_sha=state_digest(contexts);restored_rng_sha=state_digest(rng)
    del payload  # no history tensor is traversed, cloned or restored
    sys.path.insert(0,lock['author_path'])
    zmod=importlib.import_module('algs.memit.compute_z');supplier=importlib.import_module('algs.memit.native_sequential_z')
    require(Path(zmod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve())
        and Path(supplier.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'ACTUAL_OLD_NATIVE_IMPORT')
    requests=copy.deepcopy(native_rows[500:600])
    require([r['case_id'] for r in requests]==[r['case_id'] for r in rows[500:600]],'B6_NATIVE_ORDER')
    for i,r in enumerate(requests):r['sample_idx']=i;r['target_new']=' '+r['target_new']
    rng_restore(rng)
    targets,result=probe_targets(model,tok,requests,cfg,contexts,supplier=supplier.current_batch_targets,
        compute_z=zmod.compute_z,lookup=zmod.find_fact_lookup_idx,TraceDict=zmod.nethook.TraceDict)
    require(untouched=={n:(id(p),p._version) for n,p in params.items() if n not in names},'NONSELECTED_MUTATED')
    verify(cp)
    comparisons=[]
    for layer in v['writer']['layers']:
        key=str(layer['layer']);actual=result['target_norms_native_fp32'][key];archived=layer['target_norm']
        comparisons.append(dict(layer=layer['layer'],actual_fp32=actual,archived_fp32=archived,
            signed_difference=actual-archived,exact_scalar_match=actual==archived))
    result['historical_target_norm_comparison']=comparisons
    if 'fits' in b:
        archived=read(verify(b['fits']));require(archived['batch']==6 if 'batch' in archived else len(archived['rows'])==100,'B6_FITS')
        result['historical_fit_norm_comparison']=[dict(local_index=i,actual=r['norm'],archived=old['target_norm'],
            signed_difference=r['norm']-old['target_norm'])
            for i,(r,old) in enumerate(zip([x['L4_target'] for x in result['replays']],archived['rows']))]
        require(len(result['historical_fit_norm_comparison'])==100,'B6_FIT_COUNT')
    result.update(status='TARGET_PROBE_ONLY_NOT_BENCHMARK',checkpoint=cp,bindings=member(bindings),
        frozen_source=SOURCE,probe_source=member(__file__),archived_writer=b['writer'],
        contexts_sha256=context_sha,checkpoint_rng_sha256=restored_rng_sha,
        checkpoint_history_consumed=False,first_failure_localized=False,batch=6,
        archived_writer_values=v['writer'],runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        attention_backend=model.config._attn_implementation,training=model.training,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        tokenizer_padding_side=tok.padding_side,tokenizer_pad_id=tok.pad_token_id),historical_parity_claim=False)
    write_new(out/'probe.json',result)
    return member(out/'probe.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output))
