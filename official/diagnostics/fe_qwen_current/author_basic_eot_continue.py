"""USER-directed original author zsRE B15->W20 resume, no metric stop.

run BINDINGS NEW_OUTPUT | collect RESULT NEW_AUDIT.
Root execution only; no scheduler. Frozen B15 remains read-only MAP_PRIVATE.
Original first15 commits/W0/public queries are audited, never relabeled.
"""
import argparse
import copy
import importlib
import mmap
import os
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_zsre_case_probe as binding
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import basic_eot_continue as resume
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current.storage import tensor_sha,state_digest,finite_tensor
from official.experiments.prepare import digest,write_new
from official.experiments.checkpoint import rng_snapshot
from official.evaluation import zsre_paper
from official.tracking import init,official_zsre_metrics
from official.tracking.schema import job_identity
from official.runners.server1.common import read,verify,member,require

SCHEMA='author-basic-eot-zsre-B15-resume-v1'
STATUS='AUTHOR_BASIC_EOT_ZSRE_RESUMED_W20_COMPLETE'
STAGE='AUTHOR_BASIC_EOT_ZSRE_B15_CONTINUATION'
PARENT_WANDB='a2615463bd5d4b75'


def requests_for(data,batch):
    require(16<=batch<=20,'AUTHOR_TAIL_B16_B20_ONLY')
    return resume.requests_for(data,batch)  # exact native space/local sample_idx


def validate_bindings(b,*,live=False):
    require(b['schema']==SCHEMA and b['dataset']=='zsre','AUTHOR_ZSRE_RESUME_SCOPE')
    e=b['execution'];require(len(e['source_commit'])==40 and all(c in '0123456789abcdef' for c in e['source_commit'])
        and str(e['job_id'])!='121243' and e['source_run_id']==PARENT_WANDB and e['attempt'],'NEW_EXECUTION_PARENT_LINK')
    if live:require(str(e['job_id'])==os.environ.get('ODEEDIT_RENT_JOB_ID'),'ACTUAL_RENT_JOB')
    require(member(verify(b['resume_source']))['sha256']==member(__file__)['sha256'],'RESUME_SOURCE')
    require(member(verify(b['author_source']))['sha256']==member(author.__file__)['sha256']==binding.AUTHOR_MODULE_SHA
        and member(eot.__file__)['sha256']==binding.ADAPTER_SHA,'FROZEN_AUTHOR_C36')
    pb=read(verify(b['author_bindings']));old=binding.base.validate_inputs(pb)
    rt,audit,commit,pointer=[read(verify(b[k])) for k in ('runtime','author_settings','commit','pointer')]
    cfg=binding.validate_parent(pb,rt,audit,commit,pointer,b['checkpoint'],author.author_profiles(pb))
    _,expected=author.execution_config(old['runtime']['resolved'],pb)
    require(audit==expected and rt['attention_backend']==old['runtime']['attention_backend'],'EXACT_ORIGINAL_AUTHOR_CONFIG')
    stream=author.common.main_stream(pb,read(verify(old['config']['stream'])))
    require(len(b['parent_commits'])==15 and b['parent_commits'][-1]==b['commit'],'FIFTEEN_PARENT_COMMITS')
    verify(rt['identity']['adapter']);verify(b['checkpoint'])
    return pb,old,rt,audit,commit,pointer,cfg,stream


def validate_checkpoint(payload,pointer,identity,resolved,stream,commit):
    require(payload['batch']==pointer['batch']==15 and payload['identity']==pointer['identity']==identity,'EXACT_AUTHOR_B15')
    author.settings(identity);require(identity['dataset']=='zsre' and identity['source']==binding.SOURCE and str(identity['job_id'])=='121243','PARENT_ONLY')
    cur=payload['cursor'];require(cur['native_config']==resolved and cur['edits']==1500
        and cur['case_ids']==[r['case_id'] for r in stream[:1500]] and cur['raw']==commit['raw']
        and cur['input_adapter']['sha256']==binding.ADAPTER_SHA,'B15_CURSOR_ADAPTER_ORDER')
    cfg=OmegaConf.create(resolved);names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(cfg.algs.L2==0 and cfg.llms.clamp_norm_factor==1. and cfg.diagnostic_z_reference=='batch_entry'
        and cfg.bs==100 and cfg.num_edits==2000 and cfg.seed==0,'AUTHOR_RESUME_CONFIG')
    require(set(payload['weights'])==names and all(w.dtype==torch.bfloat16 and w.is_contiguous() for w in payload['weights'].values()),'SELECTED_BF16')
    H=payload['cache_c'];width=next(iter(payload['weights'].values())).shape[1]
    require(H.dtype==torch.float32 and H.device.type=='cpu' and H.is_contiguous() and H.shape==(len(names),width,width),'NATIVE_FP32_H')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'PAYLOAD_INTEGRITY')


def audit_parent(b,rt,pointer,stream,tok):
    proof=dict(identity=rt['identity'],checkpoint=pointer,evaluations=b['parent_evaluations'])
    metrics=author.common.audit_chain(proof,b['parent_commits'],stream,'zsre',tok,list(range(1,16)),('W0_first2000',2000))
    full=zsre_paper.build_queries(tok,stream,model_family='qwen25')
    parity=read(verify(b['parent_query_parity']))
    require(parity['status']=='PASS_CPU_QUERY_ONLY' and parity['query_sha256']==full['query_sha256'],'PARENT_PUBLIC_QUERY_PROOF')
    w0=next(e for e in b['parent_evaluations'] if e['batch']==0)
    require(read(verify(w0['queries']))==full,'ACTUAL_ORIGINAL_W0_PLAN')
    return dict(status='PARENT_15_COMMITS_19_ENDPOINTS_CPU_AUDITED',parent_identity=rt['identity'],
        endpoint_metrics=[dict(batch=k[0],endpoint=k[1],metrics=v) for k,v in metrics.items()],W0=w0,
        parent_bindings=b['author_bindings'],parent_checkpoint=b['checkpoint'],fresh_forward=False),full


def expected_restore(payload):
    return dict(selected={n:tensor_sha(w) for n,w in payload['weights'].items()},history=tensor_sha(payload['cache_c']),
        contexts=state_digest(payload['contexts']),rng=state_digest(payload['rng']),nonselected_unchanged=True)


def restore(model,payload,native):
    require(torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE,'PRIVATE_MMAP_REQUIRED')
    expected=expected_restore(payload)
    H,contexts,actual=resume.restore(model,payload,native)
    require(actual==expected,'EXACT_B15_W_H_CONTEXT_RNG_RESTORE')
    return H,contexts,actual


def child_identity(b,pb,rt,audit):
    return dict(stage=STAGE,dataset='zsre',input_policy=author.common.POLICY,L2=0,clamp_norm_factor=1.,no_mask=True,
        source=b['execution']['source_commit'],job_id=b['execution']['job_id'],config_sha256=digest(b),stream=pb['main_stream'],
        author_commit=author.AUTHOR,author_settings_sha256=audit['resolved_sha256'],adapter=member(eot.__file__),z_reference='batch_entry',
        parent_checkpoint=b['checkpoint'],parent_identity=rt['identity'],source_run_id=PARENT_WANDB)


def tracking(b,pb,c,env=None):
    values=author.tracking(dict(b,main_stream=pb['main_stream']),c,env)
    values.update(task_id='fe-qwen-author-basic-eot-zsre-resume-20261012',arm='AUTHOR-zsRE-B16-B20-L2-0-clamp1-basicEOT',
        evaluation_profile='FE_AUTHOR_HPARAMS_CURRENT_Z_BASIC_EOT_RESUME_B15',checkpoint_sha256=b['checkpoint']['sha256'])
    return values


class Continuation(author.AuthorObserver):
    """Inherit exact original zsRE evaluate/targets/write/checkpoint callbacks.

    The inherited snapshot's quality gate applies only to dataset=cf. This
    class requires zsre and never changes that frozen implementation.
    """
    def __init__(self,out,b,pb,c,cfg,model,tok,native,z,s,stream,rt,audit):
        require(b['dataset']=='zsre','ZSRE_NO_CF_STOP')
        self.out=out;self.b=b;self.cfg=cfg;self.stream=stream;self.batch=15;self.raw=[];self.latest=b['checkpoint']
        self.adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);self.fit=self.adapter.fit(z.compute_z)
        self.supply=self.adapter.supplier(s.current_batch_targets);self.writer=self.adapter.writer(native.batch_edit,native.compute_ks);self.z=z
        self.names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        self.nonedited={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.names}
        self.identity=child_identity(b,pb,rt,audit);self.values=tracking(b,pb,c);actual=job_identity(self.values)
        require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_TRACKING_JOB')
        self.tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=self.values)
        write_new(out/'runtime.json',dict(identity=self.identity,native_config=audit['resolved'],author_settings=audit,tracking=self.values,
            attention_backend=model.config._attn_implementation,argv=sys.argv,quality_stop=None,
            W0='VERIFIED_ORIGINAL_PARENT_2000_OBSERVATION',source_run_id=PARENT_WANDB,partial_B16_fits_discarded_no_committed_write=True))


def combined_coverage(parent_events,child_events):
    expected={(0,'W0_first2000',2000)}|{(i,'current/post',100) for i in range(1,21)}|{(i,'all_seen/post',i*100) for i in (5,10,15,20)}
    events=parent_events+child_events
    require(len(parent_events)==19 and len(child_events)==6 and len(events)==25
        and {(e['batch'],e['endpoint'],e['requests']) for e in events}==expected,'FULL20_25_ENDPOINT_CHAIN')
    require(all(e['batch']<=15 for e in parent_events) and all(16<=e['batch']<=20 for e in child_events),'PARENT_TAIL_SEPARATION')


def run(bindings,output):
    b=read(bindings);out=Path(output);require(not out.exists() and not out.is_symlink(),'NEW_OUTPUT_ONLY')
    pb,old,rt,audit,commit,pointer,cfg,stream=validate_bindings(b,live=True);c=old['config'];lock=old['lock']
    require(torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE,'PRIVATE_CHECKPOINT_MAPPING')
    payload=torch.load(b['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
    validate_checkpoint(payload,pointer,rt['identity'],audit['resolved'],stream,commit)
    import transformers,numpy,tokenizers
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers']
        and numpy.__version__=='1.26.4' and tokenizers.__version__=='0.21.4','EXACT_EDITING_RUNTIME')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    for m in c['C0']:verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else ''
        p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    out.mkdir(parents=True);write_new(out/'admission.json',parent.admission(out,b['storage']))
    snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    require(tok.pad_token_id==151643 and tok.padding_side=='right','ORIGINAL_TOKENIZER')
    parent_audit,full=audit_parent(b,rt,pointer,stream,tok);write_new(out/'parent-audit.json',parent_audit)
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==rt['attention_backend'],'ORIGINAL_BACKEND')
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main')
    z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):
        require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve())
            and any(m['sha256']==member(mod.__file__)['sha256'] for m in lock['author_members']),'PINNED_NATIVE_IMPORT')
    require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[],'FRESH_NATIVE_GLOBALS')
    obs=Continuation(out,b,pb,c,cfg,model,tok,native,z,s,stream,rt,audit);obs.full_queries=full
    success=False;status='TECHNICAL_FAILURE'
    try:
        w0=parent_audit['W0'];raw=read(verify(w0['raw']))
        require(obs.tracker.log(official_zsre_metrics(raw['summary'],config_values=obs.values,endpoint='W0_first2000',edits=0)) is not False,'PARENT_W0_TRACKING')
        write_new(out/'reused-W0.json',dict(provenance='ORIGINAL_PARENT_OBSERVATION_NOT_NEW_FORWARD',event=w0,parent_identity=rt['identity']))
        native.set_random_seed(cfg.seed);native.load_cov(cfg,model,tok)
        H,contexts,restored=restore(model,payload,native)
        write_new(out/'restore.json',dict(status='EXACT_AUTHOR_B15_RESTORE',source_checkpoint=b['checkpoint'],**restored))
        require(restored['rng']==state_digest(rng_snapshot()),'PRE_B16_EXACT_RNG')
        data=binding.native_records(stream)
        for batch in range(16,21):
            req=requests_for(data,batch);targets=obs.targets(None,model,tok,req,cfg,contexts)
            obs.write(native.batch_edit,cfg,model,tok,req,next(model.parameters()).device,H,targets,batch)
            params=dict(model.named_parameters());require(finite_tensor(H) and all(finite_tensor(params[n]) for n in obs.names),'FINITE_NATIVE_STATE')
            obs.snapshot(cfg,model,tok,H,contexts,batch,req) # original zsRE metrics + W20 CP; no quality stop
        combined_coverage(b['parent_evaluations'],obs.raw)
        require(obs.latest['batch']==20 and obs.batch==20,'REAL_W20')
        require(author.audit_settings(cfg,author.author_profiles(pb))==audit['resolved'],'AUTHOR_CONFIG_UNCHANGED')
        success=True;status=STATUS
        write_new(out/'continuation-result.json',dict(status=status,source=member(__file__),bindings=member(bindings),identity=obs.identity,
            parent_identity=rt['identity'],parent_bindings=b['author_bindings'],parent_checkpoint=b['checkpoint'],
            parent_commits=b['parent_commits'],parent_evaluations=b['parent_evaluations'],parent_audit=member(out/'parent-audit.json'),
            commits=[member(out/'commits'/f'B{i:02d}.json') for i in range(16,21)],evaluations=obs.raw,
            checkpoint=obs.latest,pointer=member(out/'checkpoints/latest.json'),runtime=member(out/'runtime.json'),restore=member(out/'restore.json'),
            parent_batches=15,continuation_batches=5,edits=2000,batches=20,total_endpoints=25,recovery_claim=False,
            W0_reused=True,quality_stop=None,author_settings=b['author_settings']))
    finally:
        try:verify(b['checkpoint']) # MAP_PRIVATE writes must never change original B15.
        finally:write_new(out/'tracking-finish.json',dict(status=status,finish=obs.tracker.finish(exit_code=0 if success else 1,timeout=45),remote_readback='ROOT_TO_VERIFY'))
    return member(out/'continuation-result.json')


def collect(result_path,output):
    proof=read(result_path);require(proof['status']==STATUS and proof['parent_batches']==15 and proof['continuation_batches']==5
        and proof['edits']==2000 and proof['batches']==20 and proof['quality_stop'] is None,'ACTUAL_RESUMED_W20')
    require(proof['source']['sha256']==member(__file__)['sha256'],'RESUME_RESULT_SOURCE')
    b=read(verify(proof['bindings']));pb,old,rt,audit,commit,pointer,cfg,stream=validate_bindings(b)
    require(proof['identity']==child_identity(b,pb,rt,audit) and proof['parent_identity']==rt['identity']
        and proof['parent_checkpoint']==b['checkpoint'] and proof['parent_commits']==b['parent_commits']
        and proof['parent_evaluations']==b['parent_evaluations'] and proof['parent_bindings']==b['author_bindings'],'EXACT_RESUME_ANCESTRY')
    import transformers
    snap=read(verify(old['config']['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    parent_audit,_=audit_parent(b,rt,pointer,stream,tok);require(read(verify(proof['parent_audit']))==parent_audit,'PARENT_REAUDIT')
    combined_coverage(proof['parent_evaluations'],proof['evaluations'])
    require(len(proof['commits'])==5,'TAIL_COMMITS')
    reduced={}
    for batch,m in zip(range(16,21),proof['commits']):
        c=read(verify(m));require(c['batch']==batch and c['edits']==100*batch and c['identity']==proof['identity']
            and c['raw']==[e['raw'] for e in proof['evaluations'] if e['batch']==batch],'TAIL_COMMIT')
        if batch==20:require(c['checkpoint']==proof['checkpoint'],'TAIL_W20_CP')
    for ev in proof['evaluations']:
        reduced[(ev['batch'],ev['endpoint'])]=author.common.audit_endpoint(ev,stream,'zsre',tok,proof['identity'])
    runtime=read(verify(proof['runtime']));require(runtime['identity']==proof['identity'] and runtime['native_config']==audit['resolved']
        and runtime['author_settings']==audit and runtime['quality_stop'] is None
        and runtime['attention_backend']==rt['attention_backend']
        and runtime['tracking']==tracking(b,pb,old['config'],{'ODEEDIT_RENT_JOB_ID':str(b['execution']['job_id'])}),'FINAL_RUNTIME')
    restore_receipt=read(verify(proof['restore']));require(restore_receipt['status']=='EXACT_AUTHOR_B15_RESTORE'
        and restore_receipt['source_checkpoint']==b['checkpoint'],'RESTORE_RECEIPT')
    parent_cp=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    validate_checkpoint(parent_cp,pointer,rt['identity'],audit['resolved'],stream,commit)
    require(all(restore_receipt[k]==v for k,v in expected_restore(parent_cp).items()),'INDEPENDENT_RESTORE_HASHES')
    cp=proof['checkpoint'];ptr=read(verify(proof['pointer']))
    require(ptr==cp and cp['batch']==20,'FINAL_POINTER');payload=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    cur=payload['cursor'];require(payload['batch']==20 and payload['identity']==cp['identity']==proof['identity']
        and cur['native_config']==audit['resolved'] and cur['edits']==2000 and cur['case_ids']==[r['case_id'] for r in stream]
        and cur['raw']==read(verify(proof['commits'][-1]))['raw'] and cur['input_adapter']['sha256']==binding.ADAPTER_SHA,'FINAL_CP_IDENTITY_CURSOR')
    names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names and all(w.dtype==torch.bfloat16 and finite_tensor(w) for w in payload['weights'].values()),'FINAL_SELECTED_BF16')
    H=payload['cache_c'];width=next(iter(payload['weights'].values())).shape[1]
    require(H.dtype==torch.float32 and H.is_contiguous() and H.shape==(5,width,width) and finite_tensor(H),'FINAL_NATIVE_H')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'FINAL_CP_INTEGRITY')
    result=dict(status='AUTHOR_BASIC_EOT_ZSRE_RESUMED_W20_CPU_AUDITED',result=member(result_path),checkpoint=cp,
        parent_batches=15,continuation_batches=5,total_commits=20,total_endpoints=25,
        W0=next(x['metrics'] for x in parent_audit['endpoint_metrics'] if x['batch']==0),W20=reduced[(20,'all_seen/post')],
        parent_source_run_id=PARENT_WANDB,recovery_claim=False,generation='NOT_APPLICABLE_ZSRE')
    write_new(Path(output),result);return member(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('bindings');r.add_argument('output')
    r=sub.add_parser('collect');r.add_argument('result');r.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output) if a.command=='run' else collect(a.result,a.output))
