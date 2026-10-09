"""CPU-only exact three-model preparation, using existing assets without copying."""
import argparse
import json
import platform
import sys
from pathlib import Path
from importlib.metadata import version
from official.experiments.prepare import digest, file_sha, write_new, load_plan
from .assets import member as asset_member, validate_c0
from .common import member, verify, read

INSTRUCTION='USER-GH-SH1-MEMIT-FE-HISTORY-THREE-MODEL-2K-20261009-R1'
TASK='memit-fe-history-three-model-2k'
METHOD='MEMIT_FE_HISTORY'
MODELS=('gptj','llama3','qwen25')
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')/TASK
HUB=Path('/mnt/raid5/janghj/.cache/huggingface/hub')
STATS=Path('/mnt/raid5/janghj/EasyEdit/examples/data/stats')

def runtime():
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                packages={k:version(k) for k in ('torch','transformers','numpy','safetensors')})

def prepare(output):
    output=Path(output).absolute()
    assert output.is_relative_to(LOCAL) and not output.exists()
    contract,_=load_plan()
    oldcfg=read('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/cf-display-score-repair-r1/preparation-r1/configs/cf-ft.json')
    stream=read(verify(oldcfg['stream_bundle_member']))
    # Reuse the exact official normalized first2000 stream, not PRICE reordered data.
    stream_member=stream['datasets']['cf']['stream']
    records=read(verify(stream_member))
    if isinstance(records,dict): records=records['records']
    assert len(records)==2000
    configs=[]
    for tag in MODELS:
        desc=contract['models'][tag]
        snapshot=HUB/('models--'+desc['model_id'].replace('/','--'))/'snapshots'/desc['revision']
        assert snapshot.is_dir(),str(snapshot)
        model_config=read(snapshot/'config.json')
        model_files=[]
        for path in sorted(snapshot.iterdir()):
            if path.is_file() and (path.suffix in ('.json','.safetensors','.model','.bin') or path.name in ('vocab.json','merges.txt','tokenizer.model')):
                model_files.append(asset_member(path,allow_symlink=True))
        assert any(x['path'].endswith(('.safetensors','.bin')) for x in model_files),'MODEL_PAYLOAD_MISSING'
        # Fully hashed model members; no source manifest may invent a cache hit.
        hp=read(Path(__file__).parents[2]/'hparams/MEMIT_FE'/f'{tag}.json')
        covariance={}
        for layer in hp['layers']:
            module=hp['rewrite_module_tmp'].format(layer)
            path=STATS/desc['model_id'].split('/')[-1]/'wikipedia_stats'/f'{module}_float32_mom2_100000.npz'
            covariance[str(layer)]=dict(member=asset_member(path),validation=validate_c0(path,desc['intermediate']),module=module)
        assets=dict(model=dict(snapshot=str(snapshot),identity=desc,config=model_config,members=model_files),C0=covariance,
                    tokenizer_sha256=digest([x for x in model_files if not x['path'].endswith(('.safetensors','.bin'))]),runtime=runtime())
        assets['assets_sha256']=digest(assets)
        asset_path=output/'assets'/f'{tag}.json';write_new(asset_path,assets)
        config=dict(schema='official-server1-fe-history-v1',instruction_id=INSTRUCTION,task_id=TASK,
            model=tag,method=METHOD,dataset='cf',hparams=hp,assets=member(asset_path),stream_member=stream_member,
            stream_sha256=digest(records),ordered_case_ids_sha256=digest([r['case_id'] for r in records]),
            batch_size=100,batches=20,requests=2000,edit_seed=0,milestones=[5,10,15,20],
            generation_schedule='DEFERRED_CHECKPOINT_EVALUATION',qualification='NOT_RUN_USER_DISABLED',
            history=dict(solve='lambda_C*C0+H_entry+K@K.T',dtype='float32',solver_dtype='float64',
                         append='final_model_native_mean_key_once_per_layer_per_committed_batch'),
            tracking_env_file=oldcfg['tracking']['env_file'],runtime=runtime(),
            checkpoint_policy='W0_THEN_LATEST_EACH_BATCH_KEEP_W20_FUTURE_CONSUMER_PENDING',
            output=str(output/'runs'/tag),automatic_retry=False)
        config['config_sha256']=digest(config)
        path=output/'configs'/f'{tag}.json';write_new(path,config);configs.append(member(path))
        print(tag,'ASSET_CPU_BINDING_READY',flush=True)
    result=dict(instruction=INSTRUCTION,task_id=TASK,configs=configs,models=list(MODELS),GPU_observed=False)
    write_new(output/'preparation.json',result)
    return result

def prepare_failed_replacement(output, job_id):
    old=read(LOCAL/'registration-r1/submission.json')
    rows=[(m,j) for m,j in old['jobs'].items() if j['job_id']==str(job_id)]
    assert len(rows)==1
    tag,job=rows[0];config=read(verify(job['config']))
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    verify(config['assets']);verify(config['stream_member'])
    config.pop('config_sha256')
    config['output']=str(output/'runs'/tag)
    config['replaces_failed_job']=str(job_id)
    config['repair']='FP64_SYSTEM_BUFFER_LIFETIME_AND_CPU_ROLLBACK'
    config['config_sha256']=digest(config)
    path=output/'configs'/f'{tag}.json';write_new(path,config)
    result=dict(instruction=INSTRUCTION,task_id=TASK,configs=[member(path)],models=[tag],
                replaces_failed_job=str(job_id),GPU_observed=False)
    write_new(output/'preparation.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--replace-failed')
    a=p.parse_args()
    prepare_failed_replacement(a.output,a.replace_failed) if a.replace_failed else prepare(a.output)
