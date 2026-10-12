"""USER-directed cold author numeric settings + current-z + BASIC EOT.

No L2-10 selection prerequisite, checkpoint restore, scheduler, or generation.
run BINDINGS NEW_OUTPUT | collect CF_RESULT ZSRE_RESULT NEW_REPORT.
"""
import argparse
import copy
import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import basic_eot_benchmark as common
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current import failure_prefix_probe as prefix
from official.diagnostics.fe_qwen_current import observer
from official.diagnostics.fe_qwen_current.storage import state_digest,finite_tensor
from official.diagnostics.fe_qwen_current.benchmark import native_records
from official.evaluation import zsre_paper
from official.evaluation.zsre_query_parity import compare_queries
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require
from official.tracking import init
from official.tracking.schema import bind_job_identity,job_identity

AUTHOR='478134dfb24b43f4e18b47e8500893ce3f9cc50f'
YAML_SHA={'algs':'4799ac1df4e726e3dd2abcc54201e9f87f1c76cb35680059c0684806df8809d3',
    'llms':'0e1481d06d530746b13aea6df5dec2a4449aef8928120a2a12c4d7da72b50183'}
PROFILE='FE_AUTHOR_HPARAMS_CURRENT_Z_BASIC_EOT'


def author_profiles(bindings):
    profiles={}
    for name,sha in YAML_SHA.items():
        m=bindings['author_yaml'][name];path=verify(m);require(m['sha256']==sha,'PINNED_AUTHOR_YAML')
        profiles[name]=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    require(profiles['algs']==dict(name='memit',add_old_keys=True,L2=0),'AUTHOR_ALGS')
    return profiles


def audit_settings(cfg,profiles):
    actual=OmegaConf.to_container(cfg,resolve=True)
    require(actual['algs']==profiles['algs'] and actual['llms']==profiles['llms'],'ALL_AUTHOR_MODEL_ALGS_FIELDS')
    require(actual['model_dtype']=='bfloat16' and actual['bs']==100 and actual['num_edits']==2000 and actual['seed']==0,'USER_SEQUENTIAL_PROTOCOL')
    require(actual['cov_mode']=='default' and actual['lti'] is False and actual['negetive_prompt_test'] is False,'NO_OTHER_METHOD_MODES')
    for key in ('target_true_test','inversion_tf','unlearning_ab','subject_output_relation_test1','subject_output_relation_test2','target_auto_corrected_test','c_edit'):
        require(not actual.get(key,False),'NO_HIDDEN_NATIVE_MODE_'+key)
    return actual


def execution_config(historical,bindings):
    """Only exact pinned model/algs profiles replace historical numeric input.

    No arbitrary CLI/OmegaConf overrides. Paths and native user protocol stay
    historical; actual main requests are supplied directly, not load_data.
    """
    profiles=author_profiles(bindings);resolved=copy.deepcopy(historical)
    resolved.update(copy.deepcopy(profiles));cfg=OmegaConf.create(resolved)
    actual=audit_settings(cfg,profiles)
    changes={group:{k:dict(before=historical[group].get(k),after=value) for k,value in values.items()
        if historical[group].get(k)!=value} for group,values in profiles.items()}
    receipt=dict(status='AUTHOR_SETTINGS_CPU_VERIFIED',author_commit=AUTHOR,author_yaml=bindings['author_yaml'],
        profiles_sha256=digest(profiles),resolved=actual,resolved_sha256=digest(actual),historical_changes=changes,
        input_policy=common.POLICY,z_reference='batch_entry',user_protocol=dict(bs=100,num_edits=2000,seed=0),
        pure_unmodified_author_repo=False,selection_required=False)
    return cfg,receipt


def settings(identity):
    require(identity['stage']=='AUTHOR_BASIC_EOT_MAIN' and identity['input_policy']==common.POLICY and identity['L2']==0
        and identity['clamp_norm_factor']==1. and identity['no_mask'] is True and identity['z_reference']=='batch_entry','AUTHOR_MAIN_IDENTITY')


def tracking(bindings,c,env=None):
    values=parent.tracking_values(bindings,dict(c,stream=bindings['main_stream']))
    values.update(task_id='fe-qwen-author-basic-eot-main-20261012',arm='AUTHOR-'+bindings['dataset']+'-L2-0-clamp1-basicEOT',
        baseline='FE-author-hparams-current-z-basicEOT',dataset=bindings['dataset'],evaluation_profile=PROFILE,
        source_run_id=bindings['execution']['source_run_id'])
    if bindings['dataset']=='zsre':values.pop('generation_schedule')
    return bind_job_identity(values,env)


class AuthorObserver(common.FinalObserver):
    """Reuse unchanged evaluation/native writer/atomic checkpoint callbacks."""
    def __init__(self,out,b,c,cfg,model,tok,native,z,s,stream,audit):
        self.out=out;self.b=b;self.cfg=cfg;self.stream=stream;self.batch=0;self.raw=[];self.latest=None
        self.adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);self.fit=self.adapter.fit(z.compute_z)
        self.supply=self.adapter.supplier(s.current_batch_targets);self.writer=self.adapter.writer(native.batch_edit,native.compute_ks);self.z=z
        self.names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        self.nonedited={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.names}
        self.identity=dict(stage='AUTHOR_BASIC_EOT_MAIN',dataset=b['dataset'],input_policy=common.POLICY,L2=0,clamp_norm_factor=1.,no_mask=True,
            source=b['execution']['source_commit'],job_id=b['execution']['job_id'],config_sha256=digest(b),stream=b['main_stream'],
            author_commit=AUTHOR,author_settings_sha256=audit['resolved_sha256'],adapter=member(eot.__file__),z_reference='batch_entry')
        self.values=tracking(b,c);actual=job_identity(self.values)
        require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_JOB')
        self.tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=self.values)
        write_new(out/'runtime.json',dict(identity=self.identity,native_config=audit_settings(cfg,author_profiles(b)),tracking=self.values,
            author_settings=audit,attention_backend=model.config._attn_implementation,argv=sys.argv,
            pure_unmodified_author_repo=False,zsre_quality_stop=None,selection_required=False))


def run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    require(not out.exists() and len(b['execution']['source_commit'])==40,'NEW_OUTPUT_SOURCE')
    # Historical inputs remain references, not the execution configuration.
    # No heldout_result or failed/partial checkpoint is loaded or promoted.
    cfg,audit=execution_config(v['runtime']['resolved'],b)
    stream=common.main_stream(b,read(verify(c['stream'])))
    import transformers
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    out.mkdir(parents=True);write_new(out/'author-settings.json',audit);write_new(out/'admission.json',parent.admission(out,b['storage']))
    snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
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
    obs=AuthorObserver(out,b,c,cfg,model,tok,native,z,s,stream,audit);obs.full_queries=full;success=False;status='TECHNICAL_FAILURE'
    try:
        require(audit_settings(cfg,author_profiles(b))==audit['resolved'],'ACTUAL_LAUNCH_CONFIG')
        with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
            native.apply_memit_to_model(model,tok,native_records(stream),cfg)
        require(obs.batch==20 and obs.latest['batch']==20,'W20_COMPLETE')
        require(audit_settings(cfg,author_profiles(b))==audit['resolved'],'CONFIG_UNCHANGED_AFTER_EDIT')
        success=True;status='AUTHOR_BASIC_EOT_W20_COMPLETE'
        write_new(out/'benchmark-result.json',dict(status=status,dataset=b['dataset'],identity=obs.identity,bindings=member(bindings),source=member(__file__),
            edits=2000,batches=20,checkpoint=obs.latest,pointer=member(out/'checkpoints/latest.json'),author_settings=member(out/'author-settings.json'),runtime=member(out/'runtime.json'),
            commits=[member(out/'commits'/f'B{i:02d}.json') for i in range(1,21)],evaluations=obs.raw,recovery_claim=False,selection_required=False))
    except common.cont.QualityEarlyStop:status='QUALITY_EARLY_STOP';raise
    finally:write_new(out/'tracking-finish.json',dict(status=status,finish=obs.tracker.finish(exit_code=0 if success else 2 if status=='QUALITY_EARLY_STOP' else 1,timeout=45),remote_readback='ROOT_TO_VERIFY'))
    return member(out/'benchmark-result.json')


def audit_checkpoint(proof,stream,resolved):
    cp=proof['checkpoint'];pointer=read(verify(proof['pointer']))
    require(all(pointer[k]==cp[k] for k in ('path','sha256','bytes','batch','identity')) and cp['batch']==20,'CP_POINTER_W20')
    payload=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    require(payload['identity']==proof['identity']==cp['identity'] and payload['batch']==20,'CP_IDENTITY');settings(payload['identity'])
    cur=payload['cursor'];require(cur['native_config']==resolved and cur['edits']==2000 and cur['case_ids']==[r['case_id'] for r in stream],'CP_CONFIG_ORDER')
    require(cur['input_adapter']['sha256']==member(eot.__file__)['sha256'],'CP_ADAPTER')
    cfg=OmegaConf.create(resolved);names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names and all(w.dtype==torch.bfloat16 for w in payload['weights'].values()),'CP_BF16')
    H=payload['cache_c'];width=next(iter(payload['weights'].values())).shape[1]
    require(H.dtype==torch.float32 and H.shape==(len(names),width,width) and H.is_contiguous(),'CP_NATIVE_H')
    require(finite_tensor(H) and all(finite_tensor(w) for w in payload['weights'].values()),'CP_FINITE')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'CP_INTEGRITY')


def collect(cf,zsre,output):
    import transformers
    results=[];profiles=None
    for dataset,path in (('cf',cf),('zsre',zsre)):
        proof=read(path);require(proof['status']=='AUTHOR_BASIC_EOT_W20_COMPLETE' and proof['dataset']==dataset and proof['edits']==2000 and proof['batches']==20,'ACTUAL_AUTHOR_W20')
        require(proof['source']['sha256']==member(__file__)['sha256'],'AUTHOR_SOURCE');b=read(verify(proof['bindings']));v=prefix.base.validate_inputs(b);c=v['config']
        cfg,audit=execution_config(v['runtime']['resolved'],b);require(read(verify(proof['author_settings']))==audit,'SAVED_AUTHOR_AUDIT')
        if profiles is None:profiles=audit['profiles_sha256']
        else:require(profiles==audit['profiles_sha256'],'SAME_AUTHOR_SETTINGS')
        settings(proof['identity']);i=proof['identity']
        require(i['dataset']==dataset and i['config_sha256']==digest(b) and i['source']==b['execution']['source_commit'] and i['job_id']==b['execution']['job_id']
            and i['stream']==b['main_stream'] and i['author_commit']==AUTHOR and i['author_settings_sha256']==audit['resolved_sha256'],'FINAL_BINDING')
        runtime=read(verify(proof['runtime']));require(runtime['identity']==i and runtime['native_config']==audit['resolved'] and runtime['author_settings']==audit,'RUNTIME_AUTHOR_BINDING')
        require(runtime['attention_backend']==v['runtime']['attention_backend'] and runtime['tracking']==tracking(b,c,{'ODEEDIT_RENT_JOB_ID':str(b['execution']['job_id'])}),'RUNTIME_TRACKING_BACKEND')
        snap=read(verify(c['model_assets']))['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        stream=common.main_stream(b,read(verify(c['stream'])))
        if dataset=='zsre':require(compare_queries(tok,stream,model_family='qwen25')['status']=='PASS_CPU_QUERY_ONLY','FINAL_QUERY_PARITY')
        reduced=common.audit_chain(proof,proof['commits'],stream,dataset,tok,list(range(1,21)),('W0_first2000',2000))
        audit_checkpoint(proof,stream,audit['resolved'])
        results.append(dict(dataset=dataset,result=member(path),checkpoint=proof['checkpoint'],W0=reduced[(0,'W0_first2000')],W20=reduced[(20,'all_seen/post')],
            Loc_delta_pp=reduced[(20,'all_seen/post')]['Specificity']-reduced[(0,'W0_first2000')]['Specificity']))
    value=dict(status='BOTH_AUTHOR_BASIC_EOT_W20_CPU_AUDITED',both_W20_complete=True,results=results,author_commit=AUTHOR,
        profiles_sha256=profiles,recovery_claim=False,selection_required=False,pure_unmodified_author_repo=False,generation='DEFERRED')
    write_new(Path(output),value);return member(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('bindings');r.add_argument('output')
    r=sub.add_parser('collect');r.add_argument('cf');r.add_argument('zsre');r.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output) if a.command=='run' else collect(a.cf,a.zsre,a.output))
