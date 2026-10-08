"""Explicit USER-disabled standalone GPU qualification; actual science unchanged."""
import argparse
from pathlib import Path
import uuid
from official.experiments.prepare import digest,write_new
from .common import (read,require,verify,member,validate_config,source_binding,bindings,
                     local_output,Tracking)

AUTHORITY='USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1'
STATUS='NOT_RUN_USER_DISABLED'
POLICY=dict(instruction_id=AUTHORITY,qualification=STATUS,checkpoint_resume_GPU_equivalence=STATUS)
CF_METHODS=('FT','MEMIT','MEMIT_FE','ALPHAEDIT','SPHERE')
ZSRE_METHODS=('FT','MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE')

def validate_overlay(config):
    require(config.get('qualification_policy')==POLICY,'EXACT_USER_DISABLED_POLICY_REQUIRED')
    require(not any(k in config for k in ('qualification_outputs','qualification_plan','zsre_smoke_output')),
            'OLD_QUALIFICATION_GATE_FORBIDDEN')
    require(config['method'] in (CF_METHODS if config['dataset']=='cf' else ZSRE_METHODS),
            'NOQUAL_ACTUAL_REGISTERED_SUBSET_ONLY')
    require(config.get('actual_GPU_qualification') is False,'NO_FAKE_GPU_PASS')

def disabled(config):
    if 'qualification_policy' not in config:return False
    validate_overlay(config)
    return True

def collect(args,config,lock,output):
    from transformers import AutoTokenizer
    from .audit import audit_commits,audit_factual
    from . import run,zsre_run
    from .costs import collect_costs
    from official.evaluation.generation.assets import load_assets
    manifest=read(args.job_manifest)
    require(manifest['source']==lock['source'] and manifest['profiles']==lock['profiles']
        and manifest['execution_lock']==member(args.source_lock)
        and manifest['plan_sha256']==lock['plan_sha256'],'COLLECTOR_IMMUTABLE_NOQUAL_LOCK')
    plan=read(verify(lock['plan']))
    require(digest(plan)==lock['plan_sha256'] and plan['jobs']==manifest['profiles'],'COLLECTOR_PLAN_DIGEST')
    rows=[]
    for profile in manifest['profiles']:
        if profile['mode']=='collect':continue
        require(profile['mode'] in ('base_w0','chain'),'QUALIFICATION_PROFILE_FORBIDDEN')
        cfg=validate_config(read(verify(profile['config'])));validate_overlay(cfg)
        require(profile['config'] in lock['job_configs'],'COLLECTOR_CONFIG_MEMBER')
        folder=Path(profile['output']);row=dict(key=profile['key'],job_id=manifest['jobs'][profile['key']],
            dataset=profile['dataset'],method=profile['method'],mode=profile['mode'],complete=False,
            qualification=STATUS,checkpoint_resume_GPU_equivalence=STATUS)
        try:
            assets,records,identity,external=bindings(cfg,lock)
            tok=AutoTokenizer.from_pretrained(assets['model']['tokenizer_path'],local_files_only=True)
            tok.pad_token=tok.eos_token;tok.padding_side='right'
            if profile['dataset']=='cf' and profile['mode']=='chain':
                run._collect_profile(profile,cfg,lock,folder,assets,records,identity,external,tok,{})
            elif profile['dataset']=='cf':
                ready=run.read_w0(cfg,lock,assets)
                audit_factual(read(verify(ready['cf_factual'])),records,'cf',tok,external)
                refs=load_assets(verify(assets['generation_reference']['manifest']))
                run._collect_generation(verify(ready['cf_generation']),records,tok,assets,lock,'W0',references=refs,folder=folder)
                bundle=read(verify(cfg['stream_bundle_member']))
                zr=read(verify(bundle['datasets']['zsre']['stream']))
                ref=read(verify(ready['zsre_reference']))
                audit_factual(ref['evaluation'],zr,'zsre',tok,ready['zsre_external_identity'],ref)
            else:
                ref=zsre_run.reference(cfg,lock,external)
                audit_factual(ref['evaluation'],records,'zsre',tok,external,ref)
                if profile['mode']=='chain':
                    terminal=read(folder/'COMPLETE.json')
                    require(terminal['identity']==identity and terminal['requests']==2000 and
                        terminal['native_apply_calls']==20 and terminal['generation_calls']==0 and
                        terminal['final_checkpoint']==member(folder/'checkpoint/latest.json'), 'ZSRE_TERMINAL_IDENTITY')
                    audit_commits(folder,profile['method'],identity,20,records=records)
                    for batch in range(1,21):
                        for endpoint in ('pre','post'):
                            audit_factual(read(folder/'current'/f'{endpoint}-{batch:02d}.json'),
                                records[(batch-1)*100:batch*100],'zsre',tok,external,ref)
                    for batch in (5,10,15,20):
                        audit_factual(read(folder/'factual'/f'batch-{batch:02d}.json'),records[:batch*100],'zsre',tok,external,ref)
            row.update(complete=True,status='CPU_ORIGINAL_RECEIPTS_VERIFIED')
        except Exception as error:
            row.update(status='NOT_OBSERVED_OR_INVALID',error_type=type(error).__name__)
        rows.append(row)
    write_new(output/'costs.json',collect_costs(manifest))
    write_new(output/'summary.json',dict(rows=rows,qualification=STATUS,checkpoint_resume_GPU_equivalence=STATUS,
        expected_chains=sum(r['mode']=='chain' for r in rows),
        complete_chains=sum(r['mode']=='chain' and r['complete'] for r in rows),
        coverage='COMPLETE' if all(r['complete'] for r in rows) else 'PARTIAL_OR_NOT_OBSERVED',
        source=lock['source'],actual_model_forward_calls=0,model_loaded=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('base_w0','chain','collect'),required=True)
    p.add_argument('--dataset',choices=('cf','zsre'),required=True);p.add_argument('--method')
    for name in ('config','output','source-lock'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--job-manifest',type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();config=validate_config(read(args.config));validate_overlay(config)
    require(config['dataset']==args.dataset and (args.method is None or config['method']==args.method),'PROFILE_IDENTITY')
    lock=source_binding(args.source_lock,args.config);output=local_output(args.output)
    if args.mode=='collect':collect(args,config,lock,output);return
    _,_,identity,_=bindings(config,lock);tracker=None
    try:
        write_new(output/'qualification-user-disabled.json',dict(POLICY,source=lock['source'],config_sha256=config['config_sha256']))
        tracker=Tracking(config,output,identity,mode=args.mode,method=args.method,dataset=args.dataset)
        from . import run,zsre_run
        if args.dataset=='cf':
            (run.base_w0 if args.mode=='base_w0' else run.chain)(args,config,lock,output,tracker)
        elif args.mode=='base_w0':zsre_run.base_w0(config,lock,output,tracker)
        else:zsre_run.chain(args,config,lock,output,tracker)
    except BaseException as error:
        write_new(output/'failures'/(uuid.uuid4().hex+'.json'),dict(error_type=type(error).__name__,
            mode=args.mode,method=args.method,source=lock['source'],automatic_retry=False))
        if tracker:tracker.finish(exit_code=1)
        raise
    else:
        if tracker:tracker.finish(exit_code=0)

if __name__=='__main__':main()
