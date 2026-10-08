"""Six native zsRE chains; one cold reference, actual B1/B3-resume gates, no generation."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import uuid

from official.experiments import checkpoint
from official.experiments.prepare import digest, write_new
from official.evaluation.factual import build_zsre_w0_reference
from .common import (SUPPORTED_METHODS, HISTORY_METHODS, Tracking, bindings, factual_payload,
    immutable_observation, load_model, local_output, member, read, require, restore_checkpoint,
    rng_digest, seed_edit, source_binding, validate_config, verify, ZSRE_PLAN)
from .run import factual, checkpoint_save, compare_qualification, recover_committed_ledger


def reference(config, lock, external):
    folder = Path(config['zsre_W0_output'])
    ready = read(folder/'READY.json')
    require(ready.get('schema') == 'official-server1-zsre-W0-v1' and
        ready.get('source') == lock['source'] and ready.get('external_identity') == external and
        ready.get('requests') == 2000 and ready.get('actual_model_edits') == 0 and
        ready.get('generation_calls') == 0, 'ZSRE_COLD_REFERENCE_IDENTITY')
    value = read(verify(ready['reference']))
    require(value['evaluation']['identity']['external_identity'] == external,
            'ZSRE_REFERENCE_ORIGINAL_EXTERNAL_IDENTITY')
    return value


def base_w0(config, lock, output, tracker):
    assets, records, identity, external = bindings(config, lock)
    model, tokenizer = load_model(assets)
    seed_edit()
    value = build_zsre_w0_reference(model, tokenizer, records, identity=external,
                                  batch_size=16, device='cuda:0')
    from .audit import audit_factual
    audit_factual(value['evaluation'],records,'zsre',tokenizer,external,value)
    write_new(output/'reference-local.json', value)
    tracker.log(factual_payload(value['evaluation'], 'W0_first2000', 0))
    write_new(output/'READY.json', dict(schema='official-server1-zsre-W0-v1', source=lock['source'],
        external_identity=external, requests=2000, actual_model_edits=0, generation_calls=0,
        reference=member(output/'reference-local.json'), identity=identity))


def state_check(state, method, calls):
    from official.baselines.registry import hparams
    hp = hparams(method,'llama3')
    unsigned = {k:v for k,v in state.items() if k != 'identity_sha256'}
    require(state.get('identity_sha256') == digest(unsigned) and state.get('method') == method
        and state.get('successful_calls') == calls, 'ZSRE_NATIVE_STATE_IDENTITY')
    names = {hp.rewrite_module_tmp.format(layer)+'.weight' for layer in hp.layers}
    weights = state.get('selected_weights',{})
    require(set(weights) == names, 'ZSRE_NATIVE_STATE_SELECTED_LAYERS')
    def tensor(row, shape):
        return set(row) == {'sha256','shape','dtype'} and row['shape'] == shape and \
            row['dtype'] == 'torch.float32' and isinstance(row['sha256'],str) and \
            len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256'])
    require(all(tensor(v,[4096,14336]) for v in weights.values()), 'ZSRE_NATIVE_WEIGHT_SCHEMA')
    history = state.get('cache_c',{})
    require(set(history) == ({str(i) for i in hp.layers} if method in HISTORY_METHODS else set())
        and all(tensor(v,[14336,14336]) for v in history.values()), 'ZSRE_NATIVE_HISTORY_SCHEMA')


def stage(args, config, lock, output):
    from .native import NativeEngine
    assets, records, identity, external = bindings(config, lock)
    ref = reference(config, lock, external)
    model, tokenizer = load_model(assets)
    from .audit import audit_factual
    audit_factual(ref['evaluation'],records,'zsre',tokenizer,external,ref)
    engine = NativeEngine(model,tokenizer,args.method,assets,source_verified=True)
    seed_edit()
    start=0
    if args.qualification_stage == 'resume':
        payload=checkpoint.load(output/'checkpoint',identity)
        require(payload['batch']==2,'ZSRE_RESUME_B2_REQUIRED')
        start=restore_checkpoint(model,engine,payload,identity)
    else:
        require(not (output/'checkpoint'/'latest.json').exists(),'ZSRE_COLD_DUPLICATE')
        checkpoint_save(output,0,engine,identity,dict(completed_batch=0))
    stop=2 if args.qualification_stage=='stop' else 3
    started=time.monotonic(); receipts=[]
    for batch in range(start+1,stop+1):
        chunk=records[(batch-1)*100:batch*100]
        if batch==1:
            immutable_observation(output/'B1-pre-local.json',
                factual(model,tokenizer,chunk,'zsre',external,reference=ref))
        edit=engine.apply(chunk);receipts.append(edit)
        cursor=dict(completed_batch=batch,edit=edit,evaluation='QUALIFICATION')
        if batch in (1,3):
            endpoint=factual(model,tokenizer,records[:batch*100],'zsre',external,reference=ref)
            write_new(output/f'B{batch}-factual-proof-local.json',endpoint)
            cursor['factual']=member(output/f'B{batch}-factual-proof-local.json')
        pointer=checkpoint_save(output,batch,engine,identity,cursor)
        write_new(output/'commits'/f'batch-{batch:02d}.json',dict(batch=batch,cursor=cursor,
            checkpoint=pointer,actual_applied_requests=batch*100,identity=identity))
    payload=checkpoint.load(output/'checkpoint',identity)
    state=engine.state_identity();state_check(state,args.method,stop)
    value=dict(schema='official-server1-zsre-qualification-stage-v1',stage=args.qualification_stage,
        completed_batch=stop,selected_state=state,contexts_sha256=digest(engine.contexts()),
        RNG_sha256=rng_digest(checkpoint.rng_snapshot()),checkpoint_RNG_sha256=rng_digest(payload['rng']),
        identity=identity,method=args.method,actual_model_loaded=True,GPU_actual=True,
        actual_native_fit_calls=len(receipts),edit_receipts=receipts,seconds=time.monotonic()-started,
        factual_member=member(output/'B3-factual-proof-local.json') if stop==3 else None,
        generation_calls=0,actual_B1_smoke=args.qualification_stage!='resume')
    write_new(output/('stage-stop.json' if stop==2 else 'stage-complete.json'),value)


def verify_qualification(config, identity, value=None):
    folder=Path(config['qualification_outputs'][config['method']])
    value=read(folder/'READY.json') if value is None else value
    require(value.get('schema')=='official-server1-zsre-resume-READY-v1' and value.get('identity')==identity
        and value.get('method')==config['method'] and value.get('plan')==ZSRE_PLAN
        and value.get('passed') is True and value.get('actual_fit_calls')==6
        and value.get('generation_calls')==0,'ZSRE_ACTUAL_QUALIFICATION_READY')
    stages={k:read(verify(value[k])) for k in ('continuous','stopped','resumed')}
    for key,phase,calls,batch in (('continuous','continuous',3,3),('stopped','stop',2,2),('resumed','resume',1,3)):
        row=stages[key]
        require(row.get('schema')=='official-server1-zsre-qualification-stage-v1' and
            row.get('stage')==phase and row.get('actual_native_fit_calls')==calls and row.get('completed_batch')==batch
            and row.get('identity')==identity and row.get('method')==config['method']
            and row.get('GPU_actual') is True and row.get('actual_model_loaded') is True
            and isinstance(row.get('seconds'),(int,float)) and 0 < row['seconds'] < float('inf'),
            'ZSRE_QUALIFICATION_STAGE_IDENTITY')
        state_check(row['selected_state'],config['method'],batch)
    require(value['checks']==compare_qualification(stages['continuous'],stages['resumed']),
            'ZSRE_ACTUAL_RESUME_CHECKS')
    return stages


def qualification(args,config,lock,output,tracker):
    write_new(output/'qualification-plan.json',ZSRE_PLAN)
    for phase,folder in (('continuous',output/'continuous'),('stop',output/'split'),('resume',output/'split')):
        argv=[sys.executable,'-B','-u','-m','official.runners.server1.zsre_run','--mode','qualification_stage',
            '--qualification-stage',phase,'--method',args.method,'--dataset','zsre','--config',str(args.config),
            '--output',str(folder),'--source-lock',str(args.source_lock)]
        require(subprocess.run(argv,check=False).returncode==0,'ZSRE_NATIVE_QUALIFICATION_STAGE_FAILED')
    continuous=read(output/'continuous'/'stage-complete.json');resumed=read(output/'split'/'stage-complete.json')
    value=dict(schema='official-server1-zsre-resume-READY-v1',identity=continuous['identity'],method=args.method,
        plan=ZSRE_PLAN,passed=True,checks=compare_qualification(continuous,resumed),actual_fit_calls=6,
        continuous=member(output/'continuous'/'stage-complete.json'),stopped=member(output/'split'/'stage-stop.json'),
        resumed=member(output/'split'/'stage-complete.json'),generation_calls=0)
    verify_qualification(config,continuous['identity'],value=value)
    from transformers import AutoTokenizer
    from .audit import audit_commits,audit_factual
    assets,records,_,external=bindings(config,lock)
    tok=AutoTokenizer.from_pretrained(assets['model']['tokenizer_path'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    ref=reference(config,lock,external)
    for name in ('continuous','split'):
        audit_commits(output/name,args.method,continuous['identity'],3,records=records)
        audit_factual(read(output/name/'B3-factual-proof-local.json'),records[:300],'zsre',tok,external,ref)
        audit_factual(read(output/name/'B1-factual-proof-local.json'),records[:100],'zsre',tok,external,ref)
        audit_factual(read(output/name/'B1-pre-local.json'),records[:100],'zsre',tok,external,ref)
    write_new(output/'READY.json',value)
    tracker.log({'setup_ok':1,'step':6})


def chain(args,config,lock,output,tracker):
    from .native import NativeEngine
    assets,records,identity,external=bindings(config,lock)
    from .noqual import disabled
    if not disabled(config):
        verify_qualification(config,identity)
    ref=reference(config,lock,external)
    model,tokenizer=load_model(assets)
    from .audit import audit_factual
    audit_factual(ref['evaluation'],records,'zsre',tokenizer,external,ref)
    engine=NativeEngine(model,tokenizer,args.method,assets,source_verified=True);seed_edit()
    start=0
    if args.resume:
        payload=checkpoint.load(output/'checkpoint',identity)
        start=restore_checkpoint(model,engine,payload,identity)
        recover_committed_ledger(output,payload,identity)
    else:
        require(not (output/'checkpoint'/'latest.json').exists(),'ZSRE_CHAIN_DUPLICATE_CHECKPOINT')
        checkpoint_save(output,0,engine,identity,dict(completed_batch=0))
        tracker.log(factual_payload(ref['evaluation'],'W0_first2000',0))
    require(start<20,'ZSRE_ALREADY_COMPLETE')
    for batch in range(start+1,21):
        edits=batch*100;chunk=records[edits-100:edits]
        pre=factual(model,tokenizer,chunk,'zsre',external,reference=ref)
        immutable_observation(output/'current'/f'pre-{batch:02d}.json',pre)
        payload=factual_payload(pre,'current/pre',edits)
        payload.update(pre_state_edits=edits-100,post_state_edits=edits);tracker.log(payload)
        edit=engine.apply(chunk)
        post=factual(model,tokenizer,chunk,'zsre',external,reference=ref)
        immutable_observation(output/'current'/f'post-{batch:02d}.json',post)
        tracker.log(factual_payload(post,'current/post',edits))
        cursor=dict(completed_batch=batch,edit=edit,evaluation='CURRENT_COMPLETE',
            current_pre=member(output/'current'/f'pre-{batch:02d}.json'),
            current_post=member(output/'current'/f'post-{batch:02d}.json'))
        if batch in (5,10,15,20):
            endpoint=factual(model,tokenizer,records[:edits],'zsre',external,reference=ref)
            immutable_observation(output/'factual'/f'batch-{batch:02d}.json',endpoint)
            cursor.update(evaluation='COMPLETE',factual=member(output/'factual'/f'batch-{batch:02d}.json'))
            tracker.log(factual_payload(endpoint,'all_seen/post',edits))
        pointer=checkpoint_save(output,batch,engine,identity,cursor)
        write_new(output/'commits'/f'batch-{batch:02d}.json',dict(batch=batch,cursor=cursor,
            checkpoint=pointer,actual_applied_requests=edits,identity=identity))
    write_new(output/'COMPLETE.json',dict(schema='official-server1-zsre-chain-v1',identity=identity,
        method=args.method,requests=2000,native_apply_calls=20,generation_calls=0,
        final_checkpoint=member(output/'checkpoint'/'latest.json'),checkpoint_W20_preserved=True))


def collect(args,config,lock,output):
    from transformers import AutoTokenizer
    from .audit import audit_commits,audit_factual
    from .costs import collect_costs
    manifest=read(args.job_manifest)
    require(manifest['source']==lock['source'] and manifest['profiles']==lock['profiles']
        and manifest['execution_lock']==member(args.source_lock),'ZSRE_COLLECTOR_LOCK')
    rows=[]
    for profile in manifest['profiles']:
        if profile['mode']=='collect':continue
        folder=Path(profile['output']);row=dict(key=profile['key'],job_id=manifest['jobs'][profile['key']],
            mode=profile['mode'],method=profile['method'],complete=False)
        try:
            require(profile['config'] in lock['job_configs'],'ZSRE_COLLECTOR_CONFIG_MEMBER')
            cfg=validate_config(read(verify(profile['config'])));assets,records,identity,external=bindings(cfg,lock)
            tok=AutoTokenizer.from_pretrained(assets['model']['tokenizer_path'],local_files_only=True)
            tok.pad_token=tok.eos_token;tok.padding_side='right'
            ref=reference(cfg,lock,external)
            audit_factual(ref['evaluation'],records,'zsre',tok,external,ref)
            if profile['mode']=='qualification':
                stages=verify_qualification(cfg,identity)
                for key in ('continuous','resumed'):
                    audit_factual(read(verify(stages[key]['factual_member'])),records[:300],'zsre',tok,external,ref)
                for name,maximum in (('continuous',3),('split',3)):
                    audit_commits(folder/name,profile['method'],identity,maximum,records=records)
                    audit_factual(read(folder/name/'B1-factual-proof-local.json'),records[:100],'zsre',tok,external,ref)
                    audit_factual(read(folder/name/'B1-pre-local.json'),records[:100],'zsre',tok,external,ref)
            elif profile['mode']=='chain':
                terminal=read(folder/'COMPLETE.json')
                require(terminal['identity']==identity and terminal['requests']==2000
                    and terminal['native_apply_calls']==20 and terminal['generation_calls']==0,'ZSRE_TERMINAL_IDENTITY')
                audit_commits(folder,profile['method'],identity,20,records=records)
                for batch in range(1,21):
                    for endpoint in ('pre','post'):
                        audit_factual(read(folder/'current'/f'{endpoint}-{batch:02d}.json'),records[(batch-1)*100:batch*100],
                            'zsre',tok,external,ref)
                for batch in (5,10,15,20):
                    audit_factual(read(folder/'factual'/f'batch-{batch:02d}.json'),records[:batch*100],'zsre',tok,external,ref)
            row.update(complete=True,status='CPU_RECEIPTS_VERIFIED_NOT_GPU_CERTIFICATION')
        except Exception as error:
            row.update(status='MISSING_OR_INVALID',error_type=type(error).__name__)
        rows.append(row)
    write_new(output/'costs.json',collect_costs(manifest))
    write_new(output/'summary.json',dict(rows=rows,expected_chains=sum(r['mode']=='chain' for r in rows),
        actual_complete_chains=sum(r['complete'] and r['mode']=='chain' for r in rows),
        source=lock['source'],model_loaded=False,actual_model_forward_calls=0,
        coverage='COMPLETE' if all(r['complete'] for r in rows) else 'PARTIAL_OR_NOT_OBSERVED'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('base_w0','qualification','qualification_stage','chain','collect'))
    p.add_argument('--qualification-stage',choices=('continuous','stop','resume'));p.add_argument('--method',choices=SUPPORTED_METHODS)
    p.add_argument('--dataset',choices=('zsre',),required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--job-manifest',type=Path);p.add_argument('--resume',action='store_true');args=p.parse_args()
    config=validate_config(read(args.config));require(config.get('zsre_six') is True and
        (args.method is None or args.method==config['method']),'ZSRE_SIX_RUNNER_SCOPE')
    lock=source_binding(args.source_lock,args.config);output=local_output(args.output)
    if args.mode=='collect':collect(args,config,lock,output);return
    if args.mode=='qualification_stage':stage(args,config,lock,output);return
    _,_,identity,_=bindings(config,lock);tracker=None
    try:
        tracker=Tracking(config,output,identity,mode=args.mode,method=args.method,dataset='zsre')
        if args.mode=='base_w0':base_w0(config,lock,output,tracker)
        elif args.mode=='qualification':qualification(args,config,lock,output,tracker)
        else:chain(args,config,lock,output,tracker)
    except BaseException as error:
        write_new(output/'failures'/(uuid.uuid4().hex+'.json'),dict(error_type=type(error).__name__,
            mode=args.mode,method=args.method,source=lock['source'],automatic_retry=False))
        if tracker:tracker.finish(exit_code=1)
        raise
    else:
        if tracker:tracker.finish(exit_code=0)


if __name__=='__main__':main()
