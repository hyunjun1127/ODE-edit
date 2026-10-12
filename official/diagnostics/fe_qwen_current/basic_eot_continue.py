"""Conditional exact BASIC-EOT B5 resume through B20, not default FE.

CLI BINDINGS NEW_OUTPUT. Old b255 bindings + prefix_result, prefix_pointer,
prefix_commits[5], execution{source_commit,job_id,server,attempt,source_run_id},
storage as failure_eot_prefix. Completed NEW EOT B5 only; no old-failed restore.
Fresh ORIGINAL W0 first2000 before restore, B6..20 only edits. Current100 each,
allseen/atomic latest at1000/1500/2000; collapse stop AFTER saved current raw.
"""
import argparse
import copy
import importlib
import mmap
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_prefix_probe as prefix
from official.diagnostics.fe_qwen_current.rescue import quality_stop
from official.diagnostics.fe_qwen_current.storage import tensor_sha,state_digest,save_latest,finite_tensor
from official.experiments.checkpoint import rng_snapshot,rng_restore
from official.experiments.prepare import digest,write_new
from official.evaluation.factual import _plan,validate_retained_observation
from official.runners.server1.common import read,verify,member,require,factual_payload
from official.tracking import init
from official.tracking.schema import bind_job_identity,job_identity


class QualityEarlyStop(Exception):pass


def requests_for(data,batch):
    require(len(data)==2000 and 6<=batch<=20,'FUTURE_BATCH_RANGE')
    result=copy.deepcopy(data[(batch-1)*100:batch*100]);require(len(result)==100,'FUTURE100')
    for i,r in enumerate(result):r['sample_idx']=i;r['target_new']=' '+r['target_new']
    return result


def validate_checkpoint(payload,pointer,proof,native_config,stream):
    require(payload['batch']==pointer['batch']==5 and payload['cursor']['edits']==500,'NEW_EOT_B5_ONLY')
    require(payload['identity']==pointer['identity']==proof['identity'],'EOT_CP_IDENTITY')
    require(payload['identity']['input_policy']=='OURS_BASIC_SUBJECT_LAST_ZERO_EOT' and payload['identity']['L2']==10
        and payload['identity']['clamp_norm_factor']==1. and payload['identity']['no_mask'] is True,'EXACT_EOT_POLICY')
    require(payload['cursor']['native_config']==native_config and payload['cursor']['case_ids']==[r['case_id'] for r in stream[:500]],'CP_NATIVE_ORDER')
    require(payload['cursor']['input_adapter']['sha256']==member(eot.__file__)['sha256']==proof['adapter']['sha256'],'ADAPTER_BYTES')
    cfg=OmegaConf.create(native_config);names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names and all(t.dtype==torch.bfloat16 for t in payload['weights'].values()),'SELECTED_BF16')
    H=payload['cache_c'];require(H.dtype==torch.float32 and H.device.type=='cpu' and H.is_contiguous()
        and H.shape==(len(cfg.llms.layers),next(iter(payload['weights'].values())).shape[1],next(iter(payload['weights'].values())).shape[1]),'FP32_NATIVE_HISTORY_SHAPE')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'CP_CONTENT_INTEGRITY')


def restore(model,payload,native):
    params=dict(model.named_parameters());unselected={n:(id(p),p._version) for n,p in params.items() if n not in payload['weights']}
    with torch.no_grad():
        for n,w in payload['weights'].items():
            require(n in params and params[n].shape==w.shape and params[n].dtype==w.dtype==torch.bfloat16,'RESTORE_TYPE_SHAPE')
            params[n].copy_(w);require(tensor_sha(params[n])==tensor_sha(w),'EXACT_SELECTED_RESTORE')
    require(unselected=={n:(id(p),p._version) for n,p in params.items() if n not in payload['weights']},'NONSELECTED_CHANGED')
    contexts=copy.deepcopy(payload['contexts']);native.CONTEXT_TEMPLATES_CACHE=contexts
    native.set_random_seed(payload['cursor']['native_config']['seed']);rng_restore(payload['rng'])
    H=payload['cache_c'] # MAP_PRIVATE mapping; mutate private pages, no full H clone.
    receipt=dict(selected={n:tensor_sha(params[n]) for n in payload['weights']},history=tensor_sha(H),
        contexts=state_digest(contexts),rng=state_digest(rng_snapshot()),nonselected_unchanged=True)
    require(receipt['contexts']==state_digest(payload['contexts']) and receipt['rng']==state_digest(payload['rng']),'CONTEXT_RNG_RESTORE')
    return H,contexts,receipt


class Continuation(parent.EOTObserver):
    def __init__(self,out,b,c,cfg,model,tok,native,z,s,stream,payload,proof):
        self.out=out;self.b=b;self.cfg=cfg;self.stream=stream;self.ref=payload;self.raw=[];self.batch=5;self.writers=[];self.targets_rows=[]
        self.adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);self.z=z;self.native=native
        self.fit=self.adapter.fit(z.compute_z);self.supply=self.adapter.supplier(s.current_batch_targets);self.writer=self.adapter.writer(native.batch_edit,native.compute_ks)
        self.identity=dict(stage='BASIC_EOT_HELDOUT_B5_CONTINUATION',dataset='cf',input_policy='OURS_BASIC_SUBJECT_LAST_ZERO_EOT',
            config_sha256=digest(b),source=b['execution']['source_commit'],job_id=b['execution']['job_id'],parent_checkpoint=proof['checkpoint'],parent_result=b['prefix_result'],
            source_run_id=b['execution']['source_run_id'],L2=10,clamp_norm_factor=1.,stream=c['stream'],adapter=member(eot.__file__),no_mask=True)
        values=parent.tracking_values(b,c);values.update(task_id='fe-qwen-basic-eot-continuation-20261012',arm='DIAGNOSTIC-B6-B20-L2-10-clamp1-basicEOT',
            source_run_id=b['execution']['source_run_id'],checkpoint_sha256=proof['checkpoint']['sha256'],evaluation_profile='FE_PLUS_OURS_BASIC_POSITION0_EOT_CONTINUATION')
        values=bind_job_identity(values);actual=job_identity(values);require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_JOB')
        self.tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=values)
        write_new(out/'runtime.json',dict(identity=self.identity,native_config=OmegaConf.to_container(cfg,resolve=True),tracking=values,
            model_backend=model.config._attn_implementation,W0='FRESH_COLD_FIRST2000_NOT_RELABELLED500',quality_stop=dict(R=80,P=70,N=65)))
    def targets(self,supplier,model,tok,requests,cfg,contexts,**kw):
        require(len(requests)==100 and 5<=self.batch<20,'RESUME_TARGET_BATCH');fits=[]
        targets=self.supply(model,tok,requests,cfg,contexts,compute_z=parent.fit_receipt(self.fit,fits),find_fact_lookup_idx=self.adapter.lookup,TraceDict=self.z.nethook.TraceDict)
        require(len(fits)==100,'FIT_ALL100');counts=dict(rewrite=0,KL=0);kl_only=[];rows=[]
        for i,(r,f) in enumerate(zip(requests,fits)):
            prompts=[c.format(r['prompt']) for group in contexts for c in group]+['{} is a']
            old=[self.z.find_fact_lookup_idx(p,r['subject'],tok,cfg.llms.fact_token,verbose=False) for p in prompts]
            for j,(p,row) in enumerate(zip(old,f['input_rows'])):
                require(row['lookup']==(1 if p==0 else p) and all(row['mask']),'BASIC_EOT_ROWS')
                if p==0:counts['rewrite' if j<6 else 'KL']+=1
            if old[-1]==0 and all(p!=0 for p in old[:6]):kl_only.append(r['case_id'])
            rows.append(dict(case_id=r['case_id'],fit=f,target_norms={str(l):float(t[:,i].detach().float().norm()) for l,t in targets.items()}))
        write_new(self.out/'targets'/f'B{self.batch+1:02d}.json',dict(rows=rows,changed_rows=counts,KL_only_cases=kl_only));return targets
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1 and 6<=batch<=20,'RESUME_WRITE_ORDER')
        r=parent.all_write(self.writer,cfg,model,tok,requests,device,H,targets,batch)
        self.batch=batch;write_new(self.out/'writers'/f'B{batch:02d}.json',r)
    def commit(self,model,tok,H,contexts,batch):
        current=self.evaluate(model,tok,H,self.stream[(batch-1)*100:batch*100],'current/post',batch,f'B{batch:02d}-current')
        bad=quality_stop(self.raw[-1]['summary'])
        if bad:
            write_new(self.out/'QUALITY_EARLY_STOP.json',dict(status='QUALITY_EARLY_STOP',batch=batch,edits=batch*100,violations=bad,
                current_raw=current,identity=self.identity,numerical_failure=False,W20_complete=False,previous_checkpoint=self.latest))
            write_new(self.out/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,current_raw=current,status='QUALITY_EARLY_STOP',checkpoint=None))
            raise QualityEarlyStop('QUALITY_EARLY_STOP')
        raws=[current];cp=None
        if batch in (10,15,20):
            raws.append(self.evaluate(model,tok,H,self.stream[:batch*100],'all_seen/post',batch,f'B{batch:02d}-all-seen'))
            parameters=dict(model.named_parameters());require(finite_tensor(H) and all(finite_tensor(parameters[n]) for n in self.ref['weights']),'NONFINITE_CHECKPOINT')
            cp=parent.preserved(model,H,lambda:save_latest(self.out/'checkpoints',{n:parameters[n] for n in self.ref['weights']},H,contexts,
                dict(policy='BASIC_EOT_CURRENT_ENTRY',durable_target_cache=False),batch,self.identity,
                dict(edits=batch*100,case_ids=[r['case_id'] for r in self.stream[:batch*100]],raw=raws,native_config=OmegaConf.to_container(self.cfg,resolve=True),input_adapter=member(eot.__file__),parent_B5=self.b['prefix_result']),
                lock_path=self.b['storage']['lock_path'],reserve_bytes=self.b['storage']['reserve_bytes']))
            self.latest=cp
        write_new(self.out/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,raw=raws,checkpoint=cp,identity=self.identity))


def run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    proof=read(verify(b['prefix_result']));require(proof['status']=='EOT_ALL500_DIAGNOSTIC_COMPLETE' and proof['edits']==500 and proof['batches']==5 and proof['masked_requests']==0,'COMPLETED_NEW_EOT_PREFIX')
    pb=read(verify(proof['bindings']));require(pb['config']==b['config'] and pb['checkpoint']==b['checkpoint'],'PARENT_INPUT_BINDING')
    require(proof['source']['sha256']==member(parent.__file__)['sha256'],'EXACT_FROZEN_PREFIX_IMPLEMENTATION')
    parent.validate_paired(read(verify(pb['eot_probe'])),pb)
    require(proof['adapter']['sha256']==member(eot.__file__)['sha256'],'SAME_C36_ADAPTER')
    pointer=read(verify(b['prefix_pointer']));require(all(pointer[k]==proof['checkpoint'][k] for k in ('path','sha256','bytes','batch','identity')),'ACTUAL_PREFIX_POINTER')
    require(len(b['prefix_commits'])==5 and not out.exists(),'FIVE_COMMITS_NEW_OUTPUT')
    commits=[read(verify(m)) for m in b['prefix_commits']]
    for i,receipt in enumerate(commits):require(receipt['batch']==i+1 and receipt['edits']==(i+1)*100 and receipt['identity']==proof['identity'],'PREFIX_COMMIT_CHAIN')
    require(commits[4]['checkpoint']==proof['checkpoint'],'B5_COMMIT_CP')
    require(torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE,'PRIVATE_CHECKPOINT_MAPPING')
    payload=torch.load(verify(proof['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    data=read(verify(c['native_data']));stream=read(verify(c['stream']));require(len(data)==len(stream)==2000 and [r['case_id'] for r in data]==[r['case_id'] for r in stream],'ORDER2000')
    validate_checkpoint(payload,pointer,proof,v['runtime']['resolved'],stream);cfg=OmegaConf.create(payload['cursor']['native_config'])
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1');require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    for m in c['C0']:verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else '';p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    out.mkdir(parents=True);write_new(out/'admission.json',parent.admission(out,b['storage']))
    snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    # Independent CPU re-reduction and exact original query/order binding BEFORE restore.
    expected={(0,'W0_first500',500),(5,'all_seen/post',500)}|{(i,'current/post',100) for i in range(1,6)}
    require({(e['batch'],e['endpoint'],e['requests']) for e in proof['evaluations']}==expected and len(proof['evaluations'])==7,'EXACT_PREFIX_ENDPOINTS')
    for i,receipt in enumerate(commits):
        expected_raw=[e['raw'] for e in proof['evaluations'] if e['batch']==i+1]
        require(receipt['raw']==expected_raw,'COMMIT_RAW_COVERAGE')
    for ev in proof['evaluations']:
        batch=ev['batch'];rows=stream[:500] if batch==0 or ev['endpoint']=='all_seen/post' else stream[(batch-1)*100:batch*100]
        raw=read(verify(ev['raw']));queries=read(verify(ev['queries']))
        require(queries==_plan(rows,'cf',tok)[2] and raw['summary']==ev['summary'],'PREFIX_ORIGINAL_QUERY_SUMMARY')
        audit=validate_retained_observation(raw,'cf',input_signature=queries)
        write_new(out/'parent-audits'/f"{batch}-{ev['endpoint'].replace('/','-')}.json",audit)
        if batch==5 and ev['endpoint']=='current/post':require(not quality_stop(raw['summary']),'B5_QUALITY_STOPPED_NO_RESUME')
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'BACKEND')
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main');z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE_IMPORT')
    require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[],'FRESH_NATIVE_GLOBALS')
    obs=Continuation(out,b,c,cfg,model,tok,native,z,s,stream,payload,proof);success=False;status='TECHNICAL_FAILURE'
    try:
        # W0 is independently measured on cold HF, not an enlarged W0-500 label.
        obs.evaluate(model,tok,torch.empty(0),stream,'W0_first2000',0,'W0-first2000')
        w0=read(verify(obs.raw[-1]['raw']));require(obs.tracker.log(factual_payload(w0,'W0_first2000',0)) is not False,'W0_TRACKING')
        native.set_random_seed(cfg.seed);native.load_cov(cfg,model,tok)
        H,contexts,restored=restore(model,payload,native);write_new(out/'restore.json',dict(status='EXACT_B5_RESTORE',**restored,source_checkpoint=proof['checkpoint']))
        obs.latest=proof['checkpoint'];nonselected={n:(id(p),p._version) for n,p in model.named_parameters() if n not in payload['weights']}
        for batch in range(6,21):
            requests=requests_for(data,batch)
            targets=obs.targets(None,model,tok,requests,cfg,contexts)
            obs.write(native.batch_edit,cfg,model,tok,requests,next(model.parameters()).device,H,targets,batch)
            require(nonselected=={n:(id(p),p._version) for n,p in model.named_parameters() if n not in payload['weights']},'NONEDITED_STATE')
            obs.commit(model,tok,H,contexts,batch)
        success=True;status='BASIC_EOT_HELDOUT_W20_COMPLETE'
        write_new(out/'continuation-result.json',dict(status=status,parent=b['prefix_result'],bindings=member(bindings),source=member(__file__),
            identity=obs.identity,edits=2000,parent_batches=5,continuation_batches=15,checkpoint=obs.latest,evaluations=obs.raw,
            recovery_claim=False,not_author_default_L2=True,not_main_benchmark=True))
    except QualityEarlyStop:
        status='QUALITY_EARLY_STOP';raise
    finally:
        verify(proof['checkpoint']) # MAP_PRIVATE writes must not mutate parent CP.
        finish=obs.tracker.finish(exit_code=0 if success else 2 if status=='QUALITY_EARLY_STOP' else 1,timeout=45)
        write_new(out/'tracking-finish.json',dict(status=status,finish=finish,remote_readback='ROOT_TO_VERIFY'))
    return member(out/'continuation-result.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args();print(run(a.bindings,a.output))
