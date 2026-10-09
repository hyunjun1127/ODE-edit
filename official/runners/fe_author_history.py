"""Shared author-hparams history-FE CF/zsRE cold chains; no qualification/generation."""
import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from official.baselines import registry
from official.baselines.memit_fe_history import native
from official.evaluation.factual import evaluate_counterfact
from official.experiments import checkpoint
from official.experiments.prepare import digest, write_new, file_sha
from official.tracking import init,official_zsre_metrics
from official.evaluation import zsre_paper
from official.baselines.fe_author_profile import INSTRUCTION,TASK,METHOD,PROFILE,profile,resolve
from .server1.common import member,verify,read,seed_edit,factual_payload
from .server1.native import tensor_sha
from .server1.assets import member as asset_member
from .server1.fe_history_prepare import runtime

def validate_config(c):
    assert c['schema']=='official-fe-author-history-v1' and c['instruction_id']==INSTRUCTION
    assert c['task_id']==TASK and c['method']==METHOD and c['model'] in ('llama3','qwen25')
    assert c['server'] in ('server1','server2') and c['dataset'] in ('cf','zsre')
    assert (c['batch_size'],c['batches'],c['requests'],c['edit_seed'])==(100,20,2000,0)
    assert c['qualification']=='NOT_RUN_USER_DISABLED' and c['milestones']==[5,10,15,20]
    assert c['hparams']==profile(c['model'])['hparams']
    assert c['author_profile_sha256']==file_sha(PROFILE)
    assert type(c['storage_min_free_bytes']) is int and c['storage_min_free_bytes']>=0
    assert c['output'] and c['arm'] and c['tracking_env_file']
    if c['dataset']=='cf':assert c['generation_schedule']=='DEFERRED_CHECKPOINT_EVALUATION'
    else:assert 'generation_schedule' not in c
    assert digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256']
    return c

def state(weights,history,calls):
    return dict(selected_weights={k:tensor_sha(v) for k,v in weights.items()},
                history={k:tensor_sha(v) for k,v in history.items()},successful_calls=calls,
                context_sha256=digest(native.CONTEXT_TEMPLATES_CACHE))

def tracking_values(c,lock):
    values=dict(server=c['server'],task_id=TASK,model=c['model'],model_family={'llama3':'llama','qwen25':'qwen2'}[c['model']],
        writer='memit_fe_history',baseline=METHOD,role='scientific',arm=c['arm'],attempt=Path(c['output']).parent.parent.name+'-'+c['model']+'-'+c['dataset'],
        source_sha=lock['source_commit'],config_sha=c['config_sha256'],dataset=c['dataset'],metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1')
    if c['dataset']=='cf':values['generation_schedule']=c['generation_schedule']
    return values

def evaluate_endpoint(model,tok,rows,c,external):
    if c['dataset']=='cf':return evaluate_counterfact(model,tok,rows,identity=external,batch_size=16,device='cuda:0')
    return zsre_paper.evaluate(model,tok,rows,model_family=c['model'],identity=external,batch_size=16,device='cuda:0')

def run(config_path,lock_path,resume=False):
    c=validate_config(read(config_path)); lock=read(lock_path)
    assert member(config_path) in lock['configs']
    source=Path(lock['source_directory']).resolve()
    assert Path(__file__).resolve().is_relative_to(source)
    for m in lock['source_members']: verify(m)
    assert runtime()==c['runtime']
    assert os.environ['OFFICIAL_CODE_COMMIT']==lock['source_commit']
    assets=read(verify(c['assets'])); records=read(verify(c['stream_member']))
    if isinstance(records,dict): records=records['records']
    assert len(records)==2000 and digest(records)==c['stream_sha256']
    assert digest({k:v for k,v in assets.items() if k!='assets_sha256'})==assets['assets_sha256']
    for m in assets['model']['members']:
        a=asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
        assert a['real_path']==m['real_path']
        if m['path'].endswith(('.safetensors','.bin')):
            assert Path(a['real_path']).name==a['sha256'],'HF_CONTENT_ADDRESSED_PAYLOAD_MISMATCH'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    desc=assets['model']['identity']; snapshot=assets['model']['snapshot']
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    safe=any(m['path'].endswith('.safetensors') for m in assets['model']['members'])
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,use_safetensors=safe,
                    dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').to('cuda:0').eval()
    expected_type={'gptj':'gptj','llama3':'llama','qwen25':'qwen2'}[c['model']]
    assert model.config.model_type==expected_type and model.config.hidden_size==desc['hidden']
    model.config.use_cache=False
    for p in model.parameters(): p.requires_grad_(False)
    hp=resolve(c['model'],device=0)
    weights={f'{hp.rewrite_module_tmp.format(layer)}.weight':native.nethook.get_parameter(model,f'{hp.rewrite_module_tmp.format(layer)}.weight') for layer in hp.layers}
    for p in weights.values(): assert p.dtype==torch.float32 and list(p.shape)==[desc['hidden'],desc['intermediate']]
    H={str(layer):torch.zeros(desc['intermediate'],desc['intermediate'],dtype=torch.float32) for layer in hp.layers}
    native.COV_CACHE={};native.CONTEXT_TEMPLATES_CACHE=None
    for layer in hp.layers:
        row=assets['C0'][str(layer)];m=row['member'];verify(m)
        with np.load(m['path'],allow_pickle=False) as a:
            assert int(a['sample_size'])==hp.mom2_n_samples
            assert int(a['mom2.count'])==row['validation']['masked_token_vector_count']
            cov=torch.from_numpy(a['mom2.mom2'])/int(a['mom2.count'])
        assert cov.dtype==torch.float32 and cov.shape==H[str(layer)].shape and torch.isfinite(cov).all()
        native.COV_CACHE[(model.config._name_or_path.replace('/','_'),row['module'])]=cov
    def no_recompute(*a,**kw): raise ValueError('FE_HISTORY_C0_CACHE_MISS_FORBIDDEN')
    native.layer_stats=no_recompute
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    identity=dict(config_sha256=c['config_sha256'],stream_sha256=c['stream_sha256'],code_commit=lock['source_commit'],
        official_tree_sha256=lock['official_tree'],model_revision=desc['revision'],tokenizer_sha256=assets['tokenizer_sha256'],assets_sha256=assets['assets_sha256'])
    values=tracking_values(c,lock)
    tracker=init(env_file=c['tracking_env_file'],spool=output/('tracking-resume' if resume else 'tracking'),config=values)
    def log(v):
        if tracker.log(v) is False: raise ValueError('TRACKING_REJECTED')
    external=dict(identity,model=c['model'],method=METHOD,instruction_id=INSTRUCTION)
    def factual(rows,batch,label):
        endpoint=evaluate_endpoint(model,tok,rows,c,external)
        path=output/'factual'/f'{label}.json';write_new(path,endpoint)
        prefix='W0_first2000' if batch==0 else 'all_seen/post'
        if c['dataset']=='cf':log(factual_payload(endpoint,prefix,batch*100))
        else:log(official_zsre_metrics(endpoint['summary'],config_values=values,endpoint=prefix,edits=batch*100,post_state_edits=batch*100))
        return member(path)
    def save(batch,cursor):
        disk=os.statvfs(output)
        assert disk.f_bavail*disk.f_frsize>=c['storage_min_free_bytes'],'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE'
        return checkpoint.save(output/'checkpoint',batch=batch,weights=weights,cache_c=H,
            contexts=dict(method=METHOD,successful_calls=batch,context_templates=native.CONTEXT_TEMPLATES_CACHE),
            evaluation_cursor=cursor,identity=identity,method=METHOD,evaluation_complete=True)
    exit_code=1
    try:
        seed_edit(); start=0
        if resume:
            saved=checkpoint.load(output/'checkpoint',identity)
            assert saved['method']==METHOD and set(saved['weights'])==set(weights) and set(saved['cache_c'])==set(H)
            assert saved['contexts']['successful_calls']==saved['batch']==saved['evaluation_cursor']['completed_batch']
            with torch.no_grad():
                for k,v in weights.items():
                    old=saved['weights'][k];assert old.shape==v.shape and old.dtype==v.dtype and torch.isfinite(old).all();v.copy_(old)
                for k,v in H.items():
                    old=saved['cache_c'][k];assert old.shape==v.shape and old.dtype==v.dtype and torch.isfinite(old).all();H[k]=old
            native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(saved['contexts']['context_templates'])
            checkpoint.rng_restore(saved['rng']);start=saved['batch'];del saved
        else:
            assert not (output/'checkpoint/latest.json').exists(),'NO_IMPLICIT_RESUME'
            cursor=dict(completed_batch=0,factual=factual(records,0,'W0'))
            save(0,cursor)
        assert start<20,'ALREADY_COMPLETE_NO_DUPLICATE'
        for batch in range(start+1,21):
            entry_history=dict(H)
            before=state(weights,H,batch-1);started=time.monotonic()
            requests=registry.requests(records[(batch-1)*100:batch*100],METHOD,c['model'])
            request_hash=digest(requests)
            _,_,apply=registry.implementation(METHOD,c['model'])
            apply(model,tok,requests,hp,history=H,copy=False,return_orig_weights=False,cache_template=None)
            assert digest(requests)==request_hash
            model.zero_grad(set_to_none=True)
            for p in model.parameters():p.requires_grad_(False)
            model.eval()
            after=state(weights,H,batch)
            if batch==1:
                write_new(output/'native-context-identity.json',dict(context_sha256=after['context_sha256'],
                    source=lock['source_commit'],config_sha256=c['config_sha256'],
                    author_profile_sha256=c['author_profile_sha256'],old_context_reused=False,request_sha256=request_hash))
            cursor=dict(completed_batch=batch,edit=dict(method=METHOD,request_sha256=request_hash,requests=100,
                before=before,after=after,history_appends_per_layer=1,H0_native_matrix_exact_runtime=batch==1,
                elapsed_sec=time.monotonic()-started),generation_status='DEFERRED_TO_SAVED_W20_CHECKPOINT' if c['dataset']=='cf' else 'NOT_APPLICABLE')
            try:
                if batch in c['milestones']: cursor['factual']=factual(records[:batch*100],batch,f'batch-{batch:02d}')
                pointer=save(batch,cursor)
            except BaseException:
                H.clear();H.update(entry_history)
                raise
            del entry_history
            write_new(output/'commits'/f'batch-{batch:02d}.json',dict(batch=batch,cursor=cursor,checkpoint=pointer,identity=identity))
            log(dict(batch=batch,step=batch,status_code=1))
        write_new(output/'COMPLETE.json',dict(method=METHOD,model=c['model'],dataset=c['dataset'],identity=identity,
            requests=2000,native_apply_calls=20,history_appends_per_layer=20,final_cursor=cursor,
            qualification='NOT_RUN_USER_DISABLED',checkpoint_W20_preserved=True,future_flucon_consumer_pending=c['dataset']=='cf',author_profile_sha256=c['author_profile_sha256'],variant='MEMIT_FE_HISTORY_FE_AUTHOR_HPARAMS'))
        exit_code=0
    except BaseException as error:
        write_new(output/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            identity=identity,checkpoint_keep=True,no_automatic_retry=True))
        raise
    finally: tracker.finish(exit_code=exit_code,timeout=45)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--lock',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.lock,a.resume)
