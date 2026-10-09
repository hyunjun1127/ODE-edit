"""Saved CF W20 evaluation only. Native shared generation, immutable old weights."""
import argparse
import math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from official.experiments.prepare import digest,write_new,file_sha
from official.baselines import registry
from official.evaluation.generation.assets import load_assets
from official.evaluation.generation.native_profile import PROFILE
from official.evaluation.generation.native_observer import NativeGenerationObserver,read_observed
from official.evaluation.generation.metrics import generation_payload
from official.evaluation.generation.paper_display import paper_cell
from official.experiments.checkpoint import rng_snapshot
from official.tracking import init,official_generation_progress
from official.tracking.schema import config as tracking_config
from .common import read,member,verify,load_model,rng_content
from .native import tensor_sha
from .assets import member as asset_member
from .zsre_reeval_restore import restore
from .fe_history_prepare import runtime

INSTRUCTION='USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1'
TASK='flucon-paper-scale-20261010'
BASE=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
LOCAL=BASE/TASK
REFERENCE=Path('/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json')

def display(raw, unit):
    canonical='cosine_0_to_1' if unit=='cosine' else unit
    return dict(raw_unit=canonical,raw_value=raw,paper_display_x100=paper_cell(raw,metric='Flu' if unit=='bits' else 'Con',raw_unit=canonical),display_unit=canonical+' x100')

def tracking_values(c,source):
    o=c['original']
    return tracking_config(dict(server='server1',task_id=TASK,arm=c['key'],attempt='r1-'+o['job_id'],
        source_sha=source,config_sha=c['config_sha256'],model=c['model'],model_family={'llama3':'llama','gptj':'gptj'}[c['model']],
        writer=o['method'].lower(),baseline=o['method'],role='eval_only',metric_schema='official-baselines-scalar-v1',dataset='cf',
        instruction_id=INSTRUCTION,evaluation_profile='cf-native-generation-W20-only-v1',
        checkpoint_sha256=o['checkpoint']['sha256'],evaluator_sha256=c['evaluator']['sha256'],
        stream_sha256=c['stream']['sha256'],tokenizer_sha256=o['identity']['tokenizer_sha256'],source_run_id=o['job_id'],
        generation_metric_schema='counterfact-cake-generation-metrics-v1',generation_profile=PROFILE,generation_eval_seed=20261007,
        reference_assets_sha256=c['reference_identity'],generation_source_sha=source,
        generation_repair_instruction=INSTRUCTION,generation_schedule='W20_ONLY_FIRST2000'))

def prepare(output):
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    refs=load_assets(REFERENCE)
    candidates=[('llama3','FT','61771',BASE/'cf-display-score-repair-r1/preparation-r1/configs/cf-ft.json'),
      ('llama3','SPHERE','61770',BASE/'cf-display-score-repair-r1/preparation-r1/configs/cf-sphere.json'),
      ('llama3','MEMIT_FE','61773',BASE/'cf-display-score-repair-r1/preparation-r1/configs/cf-memit_fe.json'),
      ('gptj','MEMIT_FE_HISTORY','61927',BASE/'memit-fe-history-three-model-2k/preparation-r1/configs/gptj.json'),
      ('llama3','MEMIT_FE_HISTORY','61928',BASE/'memit-fe-history-three-model-2k/preparation-r1/configs/llama3.json')]
    configs=[];inventory=[];seen=set();verified_base=set()
    for tag,method,jid,oldpath in candidates:
        row=dict(model=tag,method=method,job_id=jid)
        try:
            old=read(oldpath)
            folder=Path(old['output']) if 'output' in old else oldpath.parent.parent/'runs'/('cf-'+method.lower())
            done=read(folder/'COMPLETE.json')
            assert done['method']==method and done['dataset']=='cf'
            assert done.get('native_apply_calls',done.get('actual_native_apply_calls'))==20
            assert done.get('requests',done.get('actual_applied_requests'))==2000
            identity=done['identity'];assert identity['config_sha256']==old['config_sha256']
            pointer=read(folder/'checkpoint/latest.json');assert pointer['batch']==20 and pointer['final_W20']
            assert pointer['identity_sha256']==digest(identity) and Path(pointer['file']).name==pointer['file']
            cp=member(folder/'checkpoint'/pointer['file']);assert cp['sha256']==pointer['sha256']
            assert cp['sha256'] not in seen;seen.add(cp['sha256'])
            history=method=='MEMIT_FE_HISTORY'
            assert (old['generation_schedule']=='DEFERRED_CHECKPOINT_EVALUATION' if history else old['cf_W20_generation']=='DEFERRED_TO_SAVED_W20_CHECKPOINT')
            am=old['assets'] if history else old['assets_member'];assets=read(verify(am))
            stream=old['stream_member'] if history else read(verify(old['stream_bundle_member']))['datasets']['cf']['stream']
            records=read(verify(stream));records=records['records'] if isinstance(records,dict) else records
            assert len(records)==2000
            if history:assert digest([r['case_id'] for r in records])==old['ordered_case_ids_sha256']
            previous=None
            for b in range(1,21):
                commit=read(folder/'commits'/f'batch-{b:02d}.json')
                assert commit['batch']==b and commit['identity']==identity and commit['cursor']['completed_batch']==b
                assert commit['cursor']['edit']['requests']==100
                req=registry.requests(records[(b-1)*100:b*100],method,tag)
                assert commit['cursor']['edit']['request_sha256']==digest(req)
                if history:
                    edit=commit['cursor']['edit'];assert edit['history_appends_per_layer']==1
                    if previous is not None:assert edit['before']==previous
                    previous=edit['after']
            assert commit['checkpoint']==pointer and done['final_cursor']==commit['cursor']
            endpoint=read(verify(commit['cursor']['factual']));assert endpoint['summary']['requests']==len(endpoint['cases'])==2000
            files=assets['model']['members'] if history else assets['model']['weights']+list(assets['model']['tokenizer_files'].values())
            if am['sha256'] not in verified_base:
                for m in files:asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
                verified_base.add(am['sha256'])
            original=dict(job_id=jid,method=method,identity=identity,checkpoint=cp,pointer=member(folder/'checkpoint/latest.json'),
                terminal=member(folder/'COMPLETE.json'),config=member(oldpath),hparams=old['hparams'],
                expected_history=method in ('SPHERE','MEMIT_FE_HISTORY'),ordered_case_ids_sha256=digest([r['case_id'] for r in records]))
            key=tag+'-'+method.lower()
            c=dict(instruction=INSTRUCTION,key=key,model=tag,original=original,assets=am,asset_schema='history' if history else 'official',
                stream=stream,reference=member(REFERENCE),reference_identity=refs.sha,
                evaluator=member(Path(__file__).parents[2]/'evaluation/generation/native_observer.py'),
                runtime=runtime(),
                output=str(output/'runs'/key),tracking_env_file=old.get('tracking_env_file') if history else old['tracking']['env_file'])
            c['config_sha256']=digest(c);tracking_values(c,'a'*40)
            path=output/'configs'/f'{key}.json';write_new(path,c);configs.append(member(path))
            row.update(status='VALID_LOCAL_W20_CP',checkpoint=cp,identity=identity,ordered_case_ids_sha256=original['ordered_case_ids_sha256'])
        except Exception as e:row.update(status='EXCLUDED',reason=str(e),error_type=type(e).__name__)
        inventory.append(row);print(key if 'key' in locals() else jid,row['status'],row.get('reason',''),flush=True)
    # Read-only W0 CPU rescore, no new generation. Preserve original source/path/runtime.
    w0=read(BASE/'cf-display-score-repair-r1/preparation-r1/runs/cf-w0/cf-generation-local.json')
    observed=read_observed(w0['rows_path'],assets=refs)
    assert observed['summary']['planned_count']==2000
    w0row=dict(model='llama3',endpoint='W0',raw=member(w0['rows_path']),identity=observed['identity'],
        summary=observed['summary'],Flu=display(observed['summary']['ngram_entropy'],'bits'),Con=display(observed['summary']['reference_score'],'cosine'))
    exclusions=[dict(model='qwen25',job_id='61975',reason='OLD_CONTEXT_MASK_EXCLUDED'),
        dict(model='qwen25',job_id='62061',reason='NOT_COMPLETE_AT_INVENTORY'),
        dict(model='llama3',method='MEMIT/ALPHAEDIT/ALPHAEDIT_BLUE',reason='HISTORICAL_REMOTE_CP_NOT_LOCAL_CURRENT_OFFICIAL_COHORT_NO_RESTORE_AUTHORITY')]
    result=dict(instruction=INSTRUCTION,configs=configs,inventory=inventory,excluded=exclusions,W0=w0row,reference=member(REFERENCE))
    write_new(output/'preparation.json',result);return result

def run(config_path,lock_path):
    c=read(config_path);lock=read(lock_path);assert c['instruction']==INSTRUCTION
    assert digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256'] and member(config_path) in lock['configs']
    assert runtime()==c['runtime']
    assert Path(__file__).resolve().is_relative_to(Path(lock['source_directory']))
    for m in lock['source_members']:verify(m)
    o=c['original'];verify(o['checkpoint']);verify(o['pointer']);verify(o['terminal']);verify(o['config'])
    assets=read(verify(c['assets']));refs=load_assets(verify(c['reference']));assert refs.sha==c['reference_identity']
    records=read(verify(c['stream']));records=records['records'] if isinstance(records,dict) else records
    assert len(records)==2000 and digest([r['case_id'] for r in records])==o['ordered_case_ids_sha256']
    files=assets['model']['members'] if c['asset_schema']=='history' else assets['model']['weights']+list(assets['model']['tokenizer_files'].values())
    for m in files:
        s=Path(m['path']).stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'],m['ctime_ns'])
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);assert not (out/'COMPLETE.json').exists()
    cfg=tracking_values(c,lock['source_commit']);tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=cfg);exit_code=1
    def log(v):
        if tracker.log(v) is False:raise ValueError('FLUCON_TRACKING_REJECTED')
    try:
        if c['asset_schema']=='official':model,tok=load_model(assets)
        else:
            assert torch.cuda.is_available() and torch.cuda.device_count()==1
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            snap=assets['model']['snapshot'];tok=AutoTokenizer.from_pretrained(snap,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
            safe=any(m['path'].endswith('.safetensors') for m in files)
            model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,use_safetensors=safe,dtype=torch.float32,
                low_cpu_mem_usage=True,attn_implementation='eager').to('cuda:0').eval()
            model.config.use_cache=False
            for p in model.parameters():p.requires_grad_(False)
        receipt=restore(model,o);write_new(out/'restore.json',receipt)
        hp=o['hparams'];selected={n:p for n,p in model.named_parameters() if any(hp['rewrite_module_tmp'].format(l) in n for l in hp['layers'])}
        before={n:tensor_sha(p) for n,p in selected.items()};rng=rng_content(rng_snapshot())
        binding=dict(checkpoint=o['checkpoint'],original_identity=o['identity'],source=lock['source_commit'],config_sha256=c['config_sha256'])
        def state():return dict(binding=binding,selected_weights={n:tensor_sha(p) for n,p in selected.items()})
        observer=NativeGenerationObserver(model,tok,refs,dict(model_identity=dict(model=c['model'],revision=o['identity']['model_revision'],tokenizer_sha256=o['identity']['tokenizer_sha256']),
            profile=PROFILE,eval_seed=20261007,generation_source_sha=dict(code_commit=lock['source_commit'],official_tree=lock['official_tree'])),out/'generation',
            state_callback=state,progress_callback=lambda v:log(official_generation_progress(v,endpoint='W20')))
        result=observer.observe(records,'W20',cohort='first2000',state_identity=binding)
        assert result['summary']['planned_count']==len(result['rows'])==2000
        assert before=={n:tensor_sha(p) for n,p in selected.items()} and rng==rng_content(rng_snapshot())
        verify(o['checkpoint']);verify(o['pointer'])
        payload=generation_payload('all_seen/post',result['summary']);payload.update(edits=2000,post_state_edits=2000)
        log(payload)
        write_new(out/'COMPLETE.json',dict(instruction=INSTRUCTION,config=member(config_path),source=lock['source_commit'],
            original_checkpoint=o['checkpoint'],endpoint=member(result['rows_path']),summary=result['summary'],requests=2000,
            Flu=display(result['summary']['ngram_entropy'],'bits'),Con=display(result['summary']['reference_score'],'cosine'),
            weights_unchanged=True,RNG_unchanged=True,checkpoint_unchanged=True,edits=0))
        exit_code=0
    except BaseException as e:
        write_new(out/'FAILURE.json',dict(type=type(e).__name__,error=str(e),checkpoint_keep=True));raise
    finally:tracker.finish(exit_code=exit_code,timeout=45)

def collect(preparation,output,lock_path):
    prep=read(preparation);lock=read(lock_path);refs=load_assets(verify(prep['reference']));rows=[]
    for cm in prep['configs']:
        c=read(verify(cm));row=dict(key=c['key'],original_job=c['original']['job_id'])
        try:
            done=read(Path(c['output'])/'COMPLETE.json');assert done['config']==cm and done['source']==lock['source_commit']
            result=read_observed(verify(done['endpoint']),assets=refs);assert result['summary']==done['summary'] and result['summary']['planned_count']==2000
            row.update(status='CPU_RAW_VERIFIED',summary=result['summary'],endpoint=done['endpoint'],
                Flu=display(result['summary']['ngram_entropy'],'bits'),Con=display(result['summary']['reference_score'],'cosine'))
        except Exception as e:row.update(status='INCOMPLETE_OR_FAILED',reason=str(e))
        rows.append(row)
    write_new(output,dict(instruction=INSTRUCTION,rows=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect']);p.add_argument('--output');p.add_argument('--config');p.add_argument('--lock');p.add_argument('--preparation');a=p.parse_args()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='run':run(a.config,a.lock)
    else:collect(a.preparation,a.output,a.lock)
