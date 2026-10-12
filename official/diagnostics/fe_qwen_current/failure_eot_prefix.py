"""All-request basic-EOT cold B1..B5 diagnostic with ORIGINAL CF evaluation.

CLI BINDINGS NEW_OUTPUT. Old b255 bindings + control_result, five ordered
control_targets/control_writers, completed eot_probe, execution{source_commit,
job_id}, storage{lock_path,reserve_bytes,payload_bound,raw_margin_bytes}.
Original averaged K/L2=10/clamp1/native math; no mask, sink list or tuning.
W0 first500/current100 x5/allseen500, one B5 latest CP, then three B6 targets.
Not a W20 benchmark or author-bitwise FE. No submission; diagnostic W&B group.
"""
import argparse
import copy
import fcntl
import importlib
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import failure_prefix_probe as prefix
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_case_probe as cases
from official.diagnostics.fe_qwen_current import observer
from official.diagnostics.fe_qwen_current.storage import save_latest,state_digest,finite_tensor
from official.evaluation.factual import evaluate_counterfact,_plan,validate_retained_observation
from official.experiments.prepare import write_new,digest
from official.runners.server1.common import read,verify,require,member
from official.runners.fe_original import cf_payload
from official.tracking import init
from official.tracking.schema import bind_job_identity,job_identity

_TRACKER=None


def tracking_values(b,c):
    values=dict(server=b['execution']['server'],task_id='fe-qwen-basic-eot-prefix-20261012',model='qwen25',model_family='qwen2',
        writer='fe_native_batch_entry_z',baseline='DIAGNOSTIC-FE-OURS-basic-EOT-not-benchmark',role='scientific',
        arm='DIAGNOSTIC-first500-L2-10-clamp1-basicEOT',attempt=b['execution']['attempt'],source_sha=b['execution']['source_commit'],
        config_sha=digest(b),observation_identity=digest(b),stream_sha256=c['stream']['sha256'],
        evaluation_profile='FE_PLUS_OURS_BASIC_POSITION0_EOT_INPUT_PREPROCESSING',dataset='cf',metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',generation_schedule='DEFERRED_CHECKPOINT_EVALUATION')
    return values


def validate_paired(result,b):
    require(result['status']=='PAIRED_EOT_NO_WRITE_COMPLETE' and result['fits']==result['replays']==14
        and result['writer']==result['history']==result['checkpoint_save']==result['evaluation']==0,'PAIRED_COMPLETE')
    rb=read(verify(result['bindings']));require(rb['checkpoint']==b['checkpoint'] and rb['config']==b['config'] and rb['control_result']==b['control_result'],'PAIRED_BINDING')
    require([r['case_id'] for r in result['rows']]==eot.CASES and result['context_generation']=='ORIGINAL_NATIVE_W0_EQUAL','PAIRED_CONTEXT_CASES')
    require(all(v['non_input_ast_exact'] for v in result['overlay_proofs'].values()),'PAIRED_AST_PROOF')
    require(result['source']['sha256']==member(eot.__file__)['sha256'],'EXACT_PAIRED_ADAPTER_SOURCE')
    for pair in result['rows']:
        original=pair['variants']['original']['fit']['input_rows'];new=pair['variants']['eot']['fit']['input_rows']
        require(len(original)==len(new)==7,'PAIRED_ROW_COUNT')
        for a,d in zip(original,new):
            changed=a['lookup']==0
            require(d['ids']==([151643]+a['ids'] if changed else a['ids']) and all(d['mask']) and len(d['mask'])==len(d['ids']),'PAIRED_IDS_MASK')
            require(d['lookup']==(1 if changed else a['lookup']),'PAIRED_LOOKUP')
            if a['labels'] is not None:require(d['labels']==([-100]+a['labels'] if changed else a['labels']),'PAIRED_LABELS')


def preserved(model,H,fn):
    modes={n:m.training for n,m in model.named_modules()}
    def call():
        try:model.eval();return fn()
        finally:
            for n,m in model.named_modules():m.training=modes[n]
    return observer.preserved_evaluation(model,H,call)


def fit_receipt(function,receipts):
    def call(*args,**kw):
        require(sys.getprofile() is None,'PROFILE_BUSY')
        def profile(frame,event,value):
            if frame.f_code is function.__code__ and event=='return':
                v=frame.f_locals;rows=[];n=len(v['rewriting_prompts'])
                for i,pos in enumerate(v['lookup_idxs']):
                    size=int(v['input_tok']['attention_mask'][i].sum())
                    rows.append(dict(lookup=int(pos),kind='rewrite' if i<n else 'KL',
                        ids=v['input_tok']['input_ids'][i,:size].tolist(),mask=v['input_tok']['attention_mask'][i,:size].tolist(),
                        labels=v['rewriting_targets'][i,:size].tolist() if i<n else None))
                receipts.append(dict(native_triplet=dict(init=float(v['target_init'].norm()),delta=float(v['delta'].norm()),target=float(value.detach().float().norm())),
                    probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(),input_rows=rows))
        sys.setprofile(profile)
        try:return function(*args,**kw)
        finally:sys.setprofile(None)
    return call


def admission(out,s):
    require(s['reserve_bytes']>=32*1024**3 and s['payload_bound']>0 and s['raw_margin_bytes']>=0,'STORAGE_BOUNDS')
    lock=Path(s['lock_path']);lock.parent.mkdir(parents=True,exist_ok=True)
    require(not lock.is_symlink() and lock.parent.stat().st_dev==out.stat().st_dev,'HOST_LOCK_FS')
    with lock.open('a+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:
            st=os.statvfs(out);free=st.f_bavail*st.f_frsize;need=2*s['payload_bound']+s['raw_margin_bytes']+s['reserve_bytes']
            require(free>=need,'FRESH_STORAGE_ADMISSION')
            return dict(free_bytes=free,required_free_bytes=need,**s)
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def all_write(writer,cfg,model,tok,requests,device,H,targets,batch,channels=(10654,3267)):
    require(sys.gettrace() is None,'TRACE_BUSY');rows=[];seen=set();pre=None
    def trace(frame,event,arg):
        nonlocal pre
        if frame.f_code is not writer.__code__:return None
        v=frame.f_locals
        if event=='line' and all(k in v for k in ('layer','weight_name','weights','upd_matrix','resid')):
            l=int(v['layer']);name=cfg.llms.rewrite_module_tmp.format(l)+'.weight'
            if v['weight_name']==name and l not in seen and v['upd_matrix'].shape==v['weights'][name].shape:
                seen.add(l);W=v['weights'][name];D=v['upd_matrix'];K=v['layer_ks']
                require(K.shape[1]==len(requests)==targets[l].shape[1],'ALL_REQUEST_WRITES')
                row=dict(layer=l,weight_norm=prefix.norm(W),update_norm=prefix.norm(D),residual_norm=prefix.norm(v['resid']),
                    target_norm=float(targets[l].detach().float().norm()),key_norms=K.detach().norm(dim=0).cpu().tolist())
                if l==5:
                    pre=W[:,list(channels)].detach().clone()
                    row['columns']=[dict(channel=ch,pre_column_norm=prefix.norm(W[:,ch]),delta_column_norm=prefix.norm(D[:,ch]),selected_K_channel=K[ch].detach().cpu().tolist()) for ch in channels]
                rows.append(row)
        return trace
    sys.settrace(trace)
    try:writer(cfg,model,tok,requests,device,H,targets)
    finally:sys.settrace(None)
    require(seen==set(cfg.llms.layers),'ALL_LAYERS')
    W=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(5)+'.weight']
    for i,col in enumerate(next(r for r in rows if r['layer']==5)['columns']):
        col.update(post_column_norm=prefix.norm(W[:,col['channel']]),actual_delta_column_norm=prefix.norm(W[:,col['channel']].detach().double()-pre[:,i].double()))
    return dict(batch=batch,case_ids=[r['case_id'] for r in requests],layers=rows,masked_requests=0,additional_solves=0)


class EOTObserver:
    def __init__(self,out,ref,adapter,native,z,s,stream,cfg,b,ct,cw):
        self.out=out;self.ref=ref;self.adapter=adapter;self.native=native;self.z=z;self.s=s;self.stream=stream;self.cfg=cfg;self.b=b;self.ct=ct;self.cw=cw
        self.batch=0;self.final=False;self.contexts=None;self.H=None;self.raw=[];self.writers=[];self.targets_rows=[]
        self.fit=adapter.fit(z.compute_z);self.supply=adapter.supplier(s.current_batch_targets);self.writer=adapter.writer(native.batch_edit,native.compute_ks)
        self.identity=dict(stage='B5_EOT_ALL_REQUEST_DIAGNOSTIC',dataset='cf',input_policy='OURS_BASIC_SUBJECT_LAST_ZERO_EOT',
            config_sha256=digest(b),parent_config_sha256=read(verify(b['config']))['config_sha256'],source=b['execution']['source_commit'],job_id=b['execution']['job_id'],
            adapter_source=member(eot.__file__),stream=b['stream_binding'],L2=10,clamp_norm_factor=1.,z_reference='batch_entry',no_mask=True)
        global _TRACKER
        c=read(verify(b['config']));values=bind_job_identity(tracking_values(b,c))
        actual=job_identity(values);require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_DIAGNOSTIC_JOB')
        self.tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=values);_TRACKER=self.tracker
        write_new(out/'runtime.json',dict(identity=self.identity,tracking_config=values,native_config=OmegaConf.to_container(cfg,resolve=True),
            transport_role_note='scientific is existing schema editing role; task/group/baseline/profile explicitly DIAGNOSTIC',
            W0_tracking='LOCAL_ONLY_FIRST500_NOT_RELABELLED_FIRST2000',paired_proof=b['eot_probe']))
    def evaluate(self,model,tok,H,rows,endpoint,batch,label):
        value=preserved(model,H,lambda:evaluate_counterfact(model,tok,rows,identity=dict(self.identity,endpoint=endpoint,edits=batch*100),batch_size=16,device=str(next(model.parameters()).device)))
        require(len(value['cases'])==value['summary']['requests']==len(rows),'EVALUATION_COVERAGE')
        queries=_plan(rows,'cf',tok)[2];audit=validate_retained_observation(value,'cf',input_signature=queries)
        raw=self.out/'factual'/f'{label}.json';qp=self.out/'queries'/f'{label}.json'
        write_new(raw,value);write_new(qp,queries);write_new(self.out/'raw-audits'/f'{label}.json',audit)
        receipt=dict(endpoint=endpoint,requests=len(rows),batch=batch,raw=member(raw),queries=member(qp),summary=value['summary'],input_policy='ORIGINAL_NO_EOT_EVALUATOR')
        write_new(self.out/'evaluations'/f'{label}.json',receipt);self.raw.append(receipt)
        if batch>0:
            payload=cf_payload(value,endpoint,batch*100);payload['batch']=batch
            require(self.tracker.log(payload) is not False,'TRACKING_NOT_ACCEPTED')
        return member(raw)
    def snapshot(self,cfg,model,tok,H,contexts,batch,requests):
        require(batch==self.batch,'BATCH_ORDER')
        if batch==0:
            require(contexts==self.ref['contexts'] and not bool(torch.count_nonzero(H)),'COLD_CONTEXT_H')
            self.contexts=contexts;self.H=H
            self.nonedited={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.ref['weights']}
            self.evaluate(model,tok,H,self.stream[:500],'W0_first500',0,'W0-first500');return
        require(self.nonedited=={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.ref['weights']},'NONEDITED_CHANGED')
        receipts=[self.evaluate(model,tok,H,self.stream[(batch-1)*100:batch*100],'current/post',batch,f'B{batch:02d}-current')]
        if batch==5:
            receipts.append(self.evaluate(model,tok,H,self.stream[:500],'all_seen/post',batch,'B05-all-seen'))
            params=dict(model.named_parameters());require(finite_tensor(H) and all(finite_tensor(params[n]) for n in self.ref['weights']),'NONFINITE_B5_STATE')
            self.cp=preserved(model,H,lambda:save_latest(self.out/'checkpoints',{n:params[n] for n in self.ref['weights']},H,contexts,
                dict(policy='current_entry_EOT',durable_target_cache=False),5,self.identity,
                dict(edits=500,case_ids=[r['case_id'] for r in self.stream[:500]],raw=receipts,native_config=OmegaConf.to_container(cfg,resolve=True),input_adapter=member(eot.__file__)),
                lock_path=self.b['storage']['lock_path'],reserve_bytes=self.b['storage']['reserve_bytes']))
            self.final=True
        write_new(self.out/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,identity=self.identity,raw=receipts,checkpoint=self.cp if batch==5 else None))
        if batch==5:raise prefix.PrefixComplete()
    def targets(self,supplier,model,tok,requests,cfg,contexts,**kw):
        require(len(requests)==100 and self.batch<5,'FIT_ALL100');receipts=[]
        result=self.supply(model,tok,requests,cfg,contexts,compute_z=fit_receipt(self.fit,receipts),find_fact_lookup_idx=self.adapter.lookup,TraceDict=self.z.nethook.TraceDict)
        rows=[];counts=dict(rewrite=0,KL=0);kl_only=[]
        for i,(r,f) in enumerate(zip(requests,receipts)):
            rawprompts=[c.format(r['prompt']) for g in contexts for c in g]+['{} is a']
            old=[self.z.find_fact_lookup_idx(p,r['subject'],tok,cfg.llms.fact_token,verbose=False) for p in rawprompts]
            for j,(pos,row) in enumerate(zip(old,f['input_rows'])):
                require(row['lookup']==(1 if pos==0 else pos) and all(row['mask']),'EOT_ROW_LOOKUP_MASK')
                if pos==0:counts['rewrite' if j<6 else 'KL']+=1
            if old[-1]==0 and not any(x==0 for x in old[:6]):kl_only.append(r['case_id'])
            rows.append(dict(case_id=r['case_id'],fit=f,target_norms={str(l):float(t[:,i].detach().float().norm()) for l,t in result.items()}))
        require(len(rows)==100,'ALL_FITS_OBSERVED')
        if self.batch==0:
            require(counts==dict(rewrite=0,KL=0),'B1_UNAFFECTED')
            for r,o in zip(rows,self.ct[0]):
                require(r['case_id']==o['case_id'],'B1_CASES')
                require(all(r['fit']['native_triplet'][k]==o['fit'][k] for k in ('init','delta','target')),'B1_SCALAR_PARITY')
        self.targets_rows.append(rows);write_new(self.out/'targets'/f'B{self.batch+1:02d}.json',dict(rows=rows,changed_rows=counts,KL_only_cases=kl_only));return result
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1 and batch<=5,'WRITE_ORDER')
        r=all_write(self.writer,cfg,model,tok,requests,device,H,targets,batch)
        if batch==1:
            for a,o in zip(r['layers'],self.cw[0]['layers']):require(all(a[k]==o[k] for k in ('layer','weight_norm','update_norm','residual_norm','target_norm')),'B1_WRITER_SCALARS')
        self.writers.append(r);self.batch=batch;write_new(self.out/'writers'/f'B{batch:02d}.json',r)


def _run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    proof=read(verify(b['control_result']));require(proof['status']=='PREFIX_STATE_EXACT','EXACT_CONTROL');cb=read(verify(proof['bindings']))
    require(cb['checkpoint']==b['checkpoint'] and cb['config']==b['config'],'CONTROL_BINDING');validate_paired(read(verify(b['eot_probe'])),b)
    require(len(b['control_targets'])==len(b['control_writers'])==5,'FIVE_CONTROL_BATCHES')
    ct=[read(verify(m)) for m in b['control_targets']];cw=[read(verify(m)) for m in b['control_writers']]
    require(not out.exists() and len(b['execution']['source_commit'])==40 and str(b['execution']['job_id']),'NEW_EXECUTION')
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1');require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    ref=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(ref['cursor']['native_config']==v['runtime']['resolved'] and ref['identity']==v['pointer']['identity'],'REFERENCE_CONFIG')
    cfg=OmegaConf.create(ref['cursor']['native_config'])
    for m in c['C0']:verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else '';p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    data=read(verify(c['native_data']));stream=read(verify(c['stream']));require(len(data)==len(stream)==2000 and [r['case_id'] for r in data[:500]]==ref['cursor']['case_ids']==[r['case_id'] for r in stream[:500]],'ORDER')
    for i in range(5):require([r['case_id'] for r in ct[i]]==[r['case_id'] for r in data[i*100:(i+1)*100]]==cw[i]['case_ids'],'CONTROL_ORDER')
    b=dict(b,stream_binding=c['stream']);out.mkdir(parents=True);write_new(out/'admission.json',admission(out,b['storage']))
    snap=read(verify(c['model_assets']))['model']['snapshot'];model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'BACKEND');tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    w0=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(5)+'.weight'].detach().to('cpu',copy=True)
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main');z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE_IMPORT')
    prefix.check_callback_binding(inspect.getsource(native));require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[] and observer._STATE is None,'FRESH_NATIVE')
    adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);obs=EOTObserver(out,ref,adapter,native,z,s,stream,cfg,b,ct,cw)
    with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
        try:native.apply_memit_to_model(model,tok,copy.deepcopy(data[:500]),cfg)
        except prefix.PrefixComplete:pass
        else:raise ValueError('DEDICATED_STOP_REQUIRED')
    require(obs.final and obs.batch==5,'ALL500_COMPLETE')
    requests=[copy.deepcopy(data[500+i]) for i in cases.INDICES];require([r['case_id'] for r in requests]==cases.CASES,'B6_FIXED_THREE')
    for i,r in enumerate(requests):r['target_new']=' '+r['target_new'];r['sample_idx']=i
    # Existing three-case observer plain forwards tokenize unmodified prompt; all
    # three fixed bare lookups are nonzero. Verify rather than silently force EOT.
    require(all(adapter.prompt(r['prompt'],r['subject'])==r['prompt'] for r in requests),'THREE_BARE_UNCHANGED')
    targets,three=cases.observe(model,tok,requests,cfg,obs.contexts,supplier=obs.supply,compute_z=obs.fit,lookup=adapter.lookup,TraceDict=z.nethook.TraceDict,w0=w0)
    require(obs.cp['batch']==5,'B5_CP_ONLY');verify(b['checkpoint'])
    result=dict(status='EOT_ALL500_DIAGNOSTIC_COMPLETE',bindings=member(bindings),source=member(__file__),adapter=member(eot.__file__),identity=obs.identity,
        batches=5,edits=500,current_requests=100,W0_requests=500,all_seen_requests=500,checkpoint=obs.cp,evaluations=obs.raw,three_B6_cases=three,
        B5_focus=[r for r in obs.targets_rows[4] if r['case_id'] in (11721,5351)],B1_parity='SAVED_SCALARS_ONLY_NOT_FULL_STATE',
        masked_requests=0,additional_solves=0,B6_write=0,W20=False,recovery_claim=False,overlay_proofs=adapter.proofs)
    write_new(out/'eot-prefix.json',result);return member(out/'eot-prefix.json')


def run(bindings,output):
    global _TRACKER
    require(_TRACKER is None,'NEW_PROCESS_ONLY');success=False
    try:
        result=_run(bindings,output);success=True;return result
    finally:
        if _TRACKER is not None:
            receipt=_TRACKER.finish(exit_code=0 if success else 1,timeout=45)
            write_new(Path(output)/'tracking-finish.json',dict(accepted_finish=receipt,diagnostic_complete=success,W20_complete=False,remote_readback='ROOT_TO_VERIFY'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args();print(run(a.bindings,a.output))
