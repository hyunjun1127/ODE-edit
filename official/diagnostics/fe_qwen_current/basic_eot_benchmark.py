"""Conditional cold main CF/zsRE; BASIC EOT, native L2=10 (NOT default0).

run BINDINGS NEW_OUTPUT | collect CF_RESULT ZSRE_RESULT NEW_REPORT.
No scheduler operations. See basic_eot_benchmark.md for immutable bindings.
"""
import argparse
import copy
import importlib
import inspect
import math
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import basic_eot_continue as cont
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_prefix_probe as prefix
from official.diagnostics.fe_qwen_current import observer
from official.diagnostics.fe_qwen_current.benchmark import native_records,reduce_zsre,verify_query_subset
from official.diagnostics.fe_qwen_current.storage import save_latest,state_digest,finite_tensor
from official.diagnostics.fe_qwen_current.rescue import quality_stop
from official.evaluation import zsre_paper
from official.evaluation.zsre_query_parity import compare_queries
from official.evaluation.factual import _plan,validate_retained_observation,evaluate_counterfact
from official.evaluation.reduce import counterfact
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require,factual_payload
from official.runners.fe_original import cf_payload
from official.tracking import init,official_zsre_metrics
from official.tracking.schema import bind_job_identity,job_identity

POLICY='OURS_BASIC_SUBJECT_LAST_ZERO_EOT'
PROFILE='FE_PLUS_OURS_BASIC_POSITION0_EOT_FINAL_MAIN'


def settings(identity):
    require(identity['input_policy']==POLICY and identity['L2']==10 and identity['clamp_norm_factor']==1.
        and identity['no_mask'] is True,'EXACT_BASIC_EOT_L210_CLAMP1')


def endpoint_rows(stream,ev):
    b=ev['batch'];ep=ev['endpoint']
    if b==0:rows=stream[:ev['requests']]
    elif ep=='current/post':rows=stream[(b-1)*100:b*100]
    else:require(ep=='all_seen/post','ENDPOINT');rows=stream[:b*100]
    require(len(rows)==ev['requests'],'ENDPOINT_COUNT');return rows


def audit_endpoint(ev,stream,dataset,tok,identity):
    rows=endpoint_rows(stream,ev);raw=read(verify(ev['raw']));q=read(verify(ev['queries']))
    external=raw['identity'].get('external_identity',raw['identity'])
    require(external==dict(identity,endpoint=ev['endpoint'],edits=ev['batch']*100),'RAW_EXECUTION_IDENTITY')
    require(raw['summary']==ev['summary'] and len(raw['cases'])==len(rows),'RAW_SUMMARY_COUNT')
    if dataset=='cf':
        require(q==_plan(rows,'cf',tok)[2],'ORIGINAL_CF_QUERIES')
        validate_retained_observation(raw,'cf',input_signature=q);metrics=counterfact(raw['cases'])
    else:
        require(q==zsre_paper.build_queries(tok,rows,model_family='qwen25'),'ORIGINAL_PUBLIC_QUERIES')
        metrics=reduce_zsre(raw,q)['metrics']
    for k in ('Efficacy','Generalization','Specificity','Score'):
        # Public-query raw intentionally omits Score; derive it independently,
        # as official_zsre_metrics does, without modifying retained raw.
        require(math.isfinite(metrics[k]),'FINITE_REDUCED_METRIC')
        if dataset=='zsre' and k=='Score' and k not in raw['summary']:continue
        require(abs(metrics[k]-raw['summary'][k])<=1e-10,'INDEPENDENT_METRICS')
    return metrics


def audit_chain(proof,commits,stream,dataset,tok,batches,w0):
    expected={(b,'current/post',100) for b in batches}
    expected|={(b,'all_seen/post',b*100) for b in batches if b%5==0}
    expected.add((0,w0[0],w0[1]))
    events=proof['evaluations']
    require(len(events)==len(expected) and {(e['batch'],e['endpoint'],e['requests']) for e in events}==expected,'ENDPOINT_COVERAGE')
    require(len(commits)==len(batches),'COMMIT_COVERAGE')
    for b,m in zip(batches,commits):
        r=read(verify(m));require(r['batch']==b and r['edits']==100*b and r['identity']==proof['identity'],'COMMIT_IDENTITY_ORDER')
        require(r['raw']==[e['raw'] for e in events if e['batch']==b],'COMMIT_RAW_LINK')
        if b==batches[-1]:require(r['checkpoint']==proof['checkpoint'],'FINAL_COMMIT_CHECKPOINT')
    return {(e['batch'],e['endpoint']):audit_endpoint(e,stream,dataset,tok,proof['identity']) for e in events}


def audit_cp(proof,pointer_member,stream,batch):
    pointer=read(verify(pointer_member));cp=proof['checkpoint']
    require(all(pointer[k]==cp[k] for k in ('path','sha256','bytes','batch','identity')) and cp['batch']==batch,'CP_POINTER')
    payload=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    require(payload['identity']==proof['identity']==cp['identity'] and payload['batch']==batch,'CP_IDENTITY')
    settings(payload['identity']);cur=payload['cursor'];cfg=OmegaConf.create(cur['native_config'])
    require(cur['edits']==batch*100 and cur['case_ids']==[r['case_id'] for r in stream[:batch*100]],'CP_ORDER_CURSOR')
    require(cur['input_adapter']['sha256']==member(eot.__file__)['sha256'],'CP_ADAPTER')
    require(cfg.algs.L2==10 and cfg.llms.clamp_norm_factor==1. and cfg.bs==100 and cfg.num_edits==2000 and cfg.seed==0,'CP_NATIVE_CONFIG')
    names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names and all(w.dtype==torch.bfloat16 for w in payload['weights'].values()),'CP_SELECTED_BF16')
    H=payload['cache_c'];width=next(iter(payload['weights'].values())).shape[1]
    require(H.dtype==torch.float32 and H.shape==(len(names),width,width) and H.is_contiguous(),'CP_NATIVE_H')
    require(finite_tensor(H) and all(finite_tensor(w) for w in payload['weights'].values()),'CP_FINITE')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'CP_INTEGRITY')
    return cur['native_config']


def pass_gate(metrics):
    require(all(type(metrics[k]) in (int,float) and math.isfinite(metrics[k]) and metrics[k]>=v
        for k,v in (('Efficacy',95),('Generalization',85),('Specificity',70))),'HELDOUT_W20_GATE_FAILED')


def audit_heldout(b,tok):
    result=read(verify(b['heldout_result']))
    require(result['status']=='BASIC_EOT_HELDOUT_W20_COMPLETE' and result['edits']==2000
        and result['parent_batches']==5 and result['continuation_batches']==15,'ACTUAL_HELDOUT_W20_REQUIRED')
    require(result['source']['sha256']==member(cont.__file__)['sha256'],'FROZEN36_SOURCE')
    cb=read(verify(result['bindings']));require(cb['config']==b['config'] and cb['lock']==b['lock'],'HELDOUT_ASSETS')
    require(result['identity']['config_sha256']==digest(cb) and result['identity']['source']==cb['execution']['source_commit']
        and result['identity']['job_id']==cb['execution']['job_id'],'HELDOUT_EXECUTION_BINDING')
    proof=read(verify(result['parent']));require(result['parent']==cb['prefix_result'],'PARENT_LINK')
    require(proof['status']=='EOT_ALL500_DIAGNOSTIC_COMPLETE' and proof['masked_requests']==0
        and proof['source']['sha256']==member(parent.__file__)['sha256'] and proof['adapter']['sha256']==member(eot.__file__)['sha256'],'ACTUAL_C36_8F6_PARENT')
    pb=read(verify(proof['bindings']));parent.validate_paired(read(verify(pb['eot_probe'])),pb)
    require(pb['config']==cb['config'] and result['identity']['parent_checkpoint']==proof['checkpoint'],'EXACT_PARENT_CP')
    c=read(verify(b['config']));stream=read(verify(c['stream']));require(len(stream)==2000,'HELDOUT2000')
    settings(proof['identity']);settings(result['identity'])
    audit_chain(proof,cb['prefix_commits'],stream,'cf',tok,list(range(1,6)),('W0_first500',500))
    reduced=audit_chain(result,b['heldout_commits'],stream,'cf',tok,list(range(6,21)),('W0_first2000',2000))
    cfg=audit_cp(proof,cb['prefix_pointer'],stream,5)
    require(audit_cp(result,b['heldout_pointer'],stream,20)==cfg,'PARENT_FINAL_NATIVE_CONFIG')
    final=reduced[(20,'all_seen/post')];pass_gate(final)
    return dict(status='HELDOUT_RAW_AUDITED_PASS',result=b['heldout_result'],parent=result['parent'],metrics=final,
        adapter=member(eot.__file__),native_config=cfg,stream=c['stream'],GPU_parity_claim=False)


def main_stream(b,heldout):
    provenance=read(verify(b['original_main_config']));m=b['main_stream'];rows=read(verify(m))
    require(b['dataset'] in ('cf','zsre') and provenance['dataset']==b['dataset'] and provenance['stream']==m,'ORIGINAL_MAIN_PROVENANCE')
    require(len(rows)==2000 and len({r['case_id'] for r in rows})==2000,'MAIN2000_UNIQUE')
    if b['dataset']=='cf':require(not {r['case_id'] for r in rows}&{r['case_id'] for r in heldout},'MAIN_HELDOUT_DISJOINT')
    native_records(rows) # Unexpanded native conversion, including zsRE.
    return rows


def tracking(b,c):
    values=parent.tracking_values(b,dict(c,stream=b['main_stream']))
    values.update(task_id='fe-qwen-basic-eot-final-main-20261012',arm='FINAL-'+b['dataset']+'-L2-10-clamp1-basicEOT',
        baseline='FE-OURS-basic-EOT-L2-10-NOT-author-default',dataset=b['dataset'],evaluation_profile=PROFILE,
        source_run_id=b['execution']['source_run_id'])
    if b['dataset']=='zsre':values.pop('generation_schedule')
    return bind_job_identity(values)


class FinalObserver:
    def __init__(self,out,b,c,cfg,model,tok,native,z,s,stream,gate):
        self.out=out;self.b=b;self.cfg=cfg;self.stream=stream;self.batch=0;self.raw=[];self.latest=None
        self.adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);self.fit=self.adapter.fit(z.compute_z)
        self.supply=self.adapter.supplier(s.current_batch_targets);self.writer=self.adapter.writer(native.batch_edit,native.compute_ks);self.z=z
        self.names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        self.nonedited={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.names}
        self.identity=dict(stage='BASIC_EOT_FINAL_MAIN',dataset=b['dataset'],input_policy=POLICY,L2=10,clamp_norm_factor=1.,no_mask=True,
            source=b['execution']['source_commit'],job_id=b['execution']['job_id'],config_sha256=digest(b),stream=b['main_stream'],
            heldout_result=b['heldout_result'],adapter=member(eot.__file__),z_reference='batch_entry')
        self.values=tracking(b,c);actual=job_identity(self.values)
        require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_JOB')
        self.tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=self.values)
        write_new(out/'runtime.json',dict(identity=self.identity,native_config=OmegaConf.to_container(cfg,resolve=True),tracking=self.values,
            attention_backend=model.config._attn_implementation,selection=gate,not_author_default_L2=True,zsre_quality_stop=None))
    def evaluate(self,model,tok,H,rows,ep,batch,label):
        identity=dict(self.identity,endpoint=ep,edits=batch*100);dataset=self.b['dataset']
        if dataset=='cf':
            value=parent.preserved(model,H,lambda:evaluate_counterfact(model,tok,rows,identity=identity,batch_size=16,device=str(next(model.parameters()).device)))
            queries=_plan(rows,'cf',tok)[2];validate_retained_observation(value,'cf',input_signature=queries)
            payload=factual_payload(value,ep,0) if batch==0 else cf_payload(value,ep,batch*100)
        else:
            queries=zsre_paper.build_queries(tok,rows,model_family='qwen25');verify_query_subset(self.full_queries,queries,rows)
            value=parent.preserved(model,H,lambda:zsre_paper.evaluate(model,tok,rows,model_family='qwen25',identity=identity,batch_size=16,device=str(next(model.parameters()).device)))
            reduce_zsre(value,queries)
            payload=official_zsre_metrics(value['summary'],config_values=self.values,endpoint=ep,edits=batch*100,post_state_edits=batch*100)
        p=self.out/'factual'/f'{label}.json';q=self.out/'queries'/f'{label}.json';write_new(p,value);write_new(q,queries)
        ev=dict(endpoint=ep,batch=batch,requests=len(rows),raw=member(p),queries=member(q),summary=value['summary'])
        audit=audit_endpoint(ev,self.stream,dataset,tok,self.identity);write_new(self.out/'raw-audits'/f'{label}.json',audit)
        self.raw.append(ev);require(self.tracker.log(payload) is not False,'TRACKING');return ev['raw']
    def targets(self,supplier,model,tok,requests,cfg,contexts,**kw):
        require(len(requests)==100 and self.batch<20,'ALL100_FITS');fits=[]
        result=self.supply(model,tok,requests,cfg,contexts,compute_z=parent.fit_receipt(self.fit,fits),find_fact_lookup_idx=self.adapter.lookup,TraceDict=self.z.nethook.TraceDict)
        require(len(fits)==100,'ALL_FITS');write_new(self.out/'targets'/f'B{self.batch+1:02d}.json',dict(case_ids=[r['case_id'] for r in requests],fits=fits));return result
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1,'BATCH_ORDER');r=parent.all_write(self.writer,cfg,model,tok,requests,device,H,targets,batch)
        self.batch=batch;write_new(self.out/'writers'/f'B{batch:02d}.json',r)
    def snapshot(self,cfg,model,tok,H,contexts,batch,requests):
        require(batch==self.batch,'SNAPSHOT_ORDER')
        require(self.nonedited=={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.names},'NONEDITED_STATE')
        if batch==0:
            require(not bool(torch.count_nonzero(H)),'COLD_H0');write_new(self.out/'contexts.json',dict(contexts=contexts,sha256=digest(contexts)))
            self.evaluate(model,tok,H,self.stream,'W0_first2000',0,'W0-first2000');return
        raws=[self.evaluate(model,tok,H,self.stream[(batch-1)*100:batch*100],'current/post',batch,f'B{batch:02d}-current')]
        bad=quality_stop(self.raw[-1]['summary']) if self.b['dataset']=='cf' else None
        if bad:
            write_new(self.out/'QUALITY_EARLY_STOP.json',dict(status='QUALITY_EARLY_STOP',batch=batch,violations=bad,raw=raws,numerical_failure=False))
            write_new(self.out/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,identity=self.identity,raw=raws,
                checkpoint=None,status='QUALITY_EARLY_STOP',previous_checkpoint=self.latest))
            raise cont.QualityEarlyStop('QUALITY_EARLY_STOP')
        cp=None
        if batch%5==0:
            raws.append(self.evaluate(model,tok,H,self.stream[:batch*100],'all_seen/post',batch,f'B{batch:02d}-all-seen'))
            params=dict(model.named_parameters());require(finite_tensor(H) and all(finite_tensor(params[n]) for n in self.names),'FINITE_CP')
            cp=parent.preserved(model,H,lambda:save_latest(self.out/'checkpoints',{n:params[n] for n in self.names},H,contexts,
                dict(policy='BASIC_EOT_CURRENT_ENTRY',durable_target_cache=False),batch,self.identity,
                dict(edits=batch*100,case_ids=[r['case_id'] for r in self.stream[:batch*100]],raw=raws,native_config=OmegaConf.to_container(cfg,resolve=True),input_adapter=member(eot.__file__)),
                lock_path=self.b['storage']['lock_path'],reserve_bytes=self.b['storage']['reserve_bytes']))
            self.latest=cp
        write_new(self.out/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,identity=self.identity,raw=raws,checkpoint=cp))


def run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    require(not out.exists() and len(b['execution']['source_commit'])==40,'NEW_OUTPUT_SOURCE')
    import transformers
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    gate=audit_heldout(b,tok);require(gate['native_config']==v['runtime']['resolved'],'UNCHANGED_NATIVE_CONFIG')
    stream=main_stream(b,read(verify(c['stream'])));cfg=OmegaConf.create(gate['native_config'])
    out.mkdir(parents=True);write_new(out/'selection-audit.json',gate);write_new(out/'admission.json',parent.admission(out,b['storage']))
    full=None
    if b['dataset']=='zsre':
        parity=compare_queries(tok,stream,model_family='qwen25');require(parity['status']=='PASS_CPU_QUERY_ONLY','PUBLIC_QUERY_PARITY')
        full=zsre_paper.build_queries(tok,stream,model_family='qwen25');require(parity['query_sha256']==full['query_sha256'],'FULL_QUERY_BINDING');write_new(out/'query-parity.json',parity)
    for m in c['C0']:verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else '';p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'BACKEND')
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main');z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'NATIVE_IMPORT')
    prefix.check_callback_binding(inspect.getsource(native));require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[] and observer._STATE is None,'FRESH_COLD_NATIVE')
    obs=FinalObserver(out,b,c,cfg,model,tok,native,z,s,stream,gate);obs.full_queries=full;success=False;status='TECHNICAL_FAILURE'
    try:
        with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
            native.apply_memit_to_model(model,tok,native_records(stream),cfg)
        require(obs.batch==20 and obs.latest['batch']==20,'W20_COMPLETE')
        success=True;status='BASIC_EOT_FINAL_W20_COMPLETE'
        write_new(out/'benchmark-result.json',dict(status=status,dataset=b['dataset'],identity=obs.identity,bindings=member(bindings),source=member(__file__),
            edits=2000,batches=20,checkpoint=obs.latest,pointer=member(out/'checkpoints/latest.json'),
            commits=[member(out/'commits'/f'B{i:02d}.json') for i in range(1,21)],evaluations=obs.raw,recovery_claim=False))
    except cont.QualityEarlyStop:status='QUALITY_EARLY_STOP';raise
    finally:write_new(out/'tracking-finish.json',dict(status=status,finish=obs.tracker.finish(exit_code=0 if success else 2 if status=='QUALITY_EARLY_STOP' else 1,timeout=45),remote_readback='ROOT_TO_VERIFY'))
    return member(out/'benchmark-result.json')


def collect(cf,zsre,output):
    import transformers
    results=[];selection=None
    for dataset,path in (('cf',cf),('zsre',zsre)):
        proof=read(path);require(proof['status']=='BASIC_EOT_FINAL_W20_COMPLETE' and proof['dataset']==dataset and proof['edits']==2000 and proof['batches']==20,'ACTUAL_FINAL_W20')
        require(proof['source']['sha256']==member(__file__)['sha256'],'FINAL_SOURCE');b=read(verify(proof['bindings']));c=read(verify(b['config']))
        snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        gate=audit_heldout(b,tok)
        if selection is None:selection=gate
        else:require(selection==gate,'SAME_HELDOUT_SETTING_NO_RETUNING')
        settings(proof['identity']);require(proof['identity']['dataset']==dataset and proof['identity']['heldout_result']==b['heldout_result']
            and proof['identity']['config_sha256']==digest(b) and proof['identity']['source']==b['execution']['source_commit']
            and proof['identity']['job_id']==b['execution']['job_id'] and proof['identity']['stream']==b['main_stream'],'FINAL_DATASET_PROOF')
        stream=main_stream(b,read(verify(c['stream'])))
        if dataset=='zsre':require(compare_queries(tok,stream,model_family='qwen25')['status']=='PASS_CPU_QUERY_ONLY','FINAL_PUBLIC_QUERY_PARITY')
        reduced=audit_chain(proof,proof['commits'],stream,dataset,tok,list(range(1,21)),('W0_first2000',2000))
        require(audit_cp(proof,proof['pointer'],stream,20)==gate['native_config'],'FINAL_NATIVE_CONFIG')
        results.append(dict(dataset=dataset,result=member(path),checkpoint=proof['checkpoint'],W20=reduced[(20,'all_seen/post')],W0=reduced[(0,'W0_first2000')],
            Loc_delta_pp=reduced[(20,'all_seen/post')]['Specificity']-reduced[(0,'W0_first2000')]['Specificity']))
    value=dict(status='BOTH_FINAL_W20_CPU_AUDITED',both_W20_complete=True,results=results,selection=selection,recovery_claim=False,generation='DEFERRED')
    write_new(Path(output),value);return member(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('bindings');r.add_argument('output')
    r=sub.add_parser('collect');r.add_argument('cf');r.add_argument('zsre');r.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output) if a.command=='run' else collect(a.cf,a.zsre,a.output))
