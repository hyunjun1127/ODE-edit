"""Server3 (H200) rerun of the Llama CF author-hparams history-FE chain of server1 job 62529.

The science is the 62529 config unchanged (author clamp 0.75 / 35 steps, per-batch z at the batch-entry model,
persistent history, cold native contexts, same stream and C0 content); only server-bound fields change: model/C0
paths of the server3 copies (verified by SHA-256 against the template manifest), the stream copy, the runtime, the
output and the tracking env file. The W20 checkpoint stays local for the deferred Flu/Con evaluation.

    python -m official.runners.server3.fe_author_llama_cf prepare --template-config CF.json \
        --template-assets LLAMA3_ASSETS.json --stream CF_STREAM.json --output OUT/preparation-r1 [--wandb-env ENV]
    python -m official.runners.server3.fe_author_llama_cf register --preparation OUT/preparation-r1/preparation.json \
        --attempt OUT/registration-r1 --cap-authority TEXT
"""
import argparse
import getpass
import os
import re
import shlex
from pathlib import Path
from official.experiments.prepare import digest,write_new,file_sha
from official.baselines.fe_author_profile import INSTRUCTION,PROFILE,profile
from official.runners.fe_author_history import validate_config,tracking_values
from official.tracking.schema import config as tracking_config, load_env
from ..server1.common import read,verify,member
from ..server1.assets import member as asset_member
from ..server1.fe_history_prepare import runtime
from ..server1 import submit as control

LOCAL=Path('/data/janghj/ODE-edit/local/fe-author-llama-cf-s3-20261011')
PATH_MAP=(('/mnt/raid5/janghj/.cache/huggingface/hub/','/data/janghj/.cache/huggingface/hub/'),
          ('/mnt/raid5/janghj/EasyEdit/examples/data/stats/','/data/janghj/EasyEdit/examples/data/stats/'))
TASK='fe-author-llama-cf-s3-20261011'
NODE,QOS,PARTITION='ubuntu','lab_gpu_s3','gpu'
PYTHON='/data/janghj/ODE-edit/local/runtime/price-s4-mirror-v1/venv/bin/python'
WANDB_ENV='/data/janghj/ODE-edit/servers/local/wandb.env'

def mapped(path):
    for old,new in PATH_MAP:
        if path.startswith(old):return new+path[len(old):]
    raise ValueError('UNMAPPED_TEMPLATE_PATH '+path)

def server3_assets(template):
    """Same content, server3 paths; every member re-hashed here and required equal to the template SHA/bytes."""
    a=dict(template);a.pop('assets_sha256')
    model=dict(a['model']);model['snapshot']=mapped(model['snapshot']);members=[]
    for m in model['members']:
        got=asset_member(mapped(m['path']),allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
        if got['path'].endswith(('.safetensors','.bin')):
            assert Path(got['real_path']).name==got['sha256'],'HF_CONTENT_ADDRESSED_PAYLOAD_MISMATCH'
        members.append(got)
    model['members']=members;a['model']=model;c0={}
    for layer,row in a['C0'].items():
        got=asset_member(mapped(row['member']['path']),allow_symlink=False,expected_sha=row['member']['sha256'],
                         expected_bytes=row['member']['bytes'])
        c0[layer]=dict(row,member=got)
    a['C0']=c0;a['runtime']=runtime();a['template_assets_sha256']=template['assets_sha256']
    a['assets_sha256']=digest(a);return a

def prepare(template_config,template_assets,stream,output,wandb_env=WANDB_ENV):
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    load_env(wandb_env)  # the runner's tracker accepts only the schema ENV_KEYS
    t=read(template_config);tmpl=read(template_assets)
    assert (t['model'],t['dataset'],t['server'],t['method'])==('llama3','cf','server1','MEMIT_FE_HISTORY')
    assert 'native_context' not in t and 'context_replay' not in t,'TEMPLATE_USES_COLD_NATIVE_CONTEXTS'
    assert t['assets']['sha256']==file_sha(template_assets) and t['hparams']==profile('llama3')['hparams']
    records=read(stream);records=records['records'] if isinstance(records,dict) else records
    assert len(records)==2000 and digest(records)==t['stream_sha256'],'STREAM_NOT_TEMPLATE'
    assert digest([r['case_id'] for r in records])==t['ordered_case_ids_sha256']
    output.mkdir(parents=True);ap=output/'assets-llama3.json';write_new(ap,server3_assets(tmpl))
    c=dict(t);c.pop('config_sha256')
    c.update(server='server3',task_id=TASK,assets=member(ap),stream_member=member(Path(stream).absolute()),
        runtime=runtime(),output=str(output/'runs'/'cf'),tracking_env_file=str(Path(wandb_env).absolute()),
        author_profile_sha256=file_sha(PROFILE),covariance_policy='strict',
        rerun_of=dict(job='62529',config_sha256=t['config_sha256'],config=member(Path(template_config).absolute()),
                      reason='W20 checkpoint of 62529 was deleted; rerun keeps W20 for the deferred Flu/Con evaluation'))
    c['config_sha256']=digest(c);validate_config(c);tracking_config(tracking_values(c,{'source_commit':'a'*40}))
    cp=output/'configs'/'cf.json';write_new(cp,c)
    result=dict(instruction=INSTRUCTION,server='server3',model='llama3',datasets=['cf'],configs=[member(cp)],
        assets=member(ap),author_profile=member(PROFILE),template_config=member(Path(template_config).absolute()),
        GPU_qualification='NOT_RUN_USER_DISABLED')
    write_new(output/'preparation.json',result);print(result)

def register(preparation,attempt,cap_authority):
    preparation=Path(preparation).absolute();prep=read(preparation)
    assert prep['instruction']==INSTRUCTION and prep['server']=='server3' and len(prep['configs'])==1
    attempt=Path(attempt).absolute();assert attempt.is_relative_to(LOCAL) and not attempt.exists(),'EXISTING_ATTEMPT_RECONCILE'
    assert cap_authority,'SERVER3_REQUIRES_USER_DIRECT_AUTHORITY'
    source=control.command(['git','rev-parse','HEAD']);tree=control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    assert not control.command(['git','status','--porcelain','--','official'])
    cm=prep['configs'][0];c=read(verify(cm));validate_config(c);assert c['server']=='server3'
    verify(c['assets']);verify(c['stream_member']);assert c['author_profile_sha256']==file_sha(PROFILE)
    tracking_config(tracking_values(c,{'source_commit':source}));load_env(c['tracking_env_file'])
    assert runtime()==c['runtime'],'REGISTER_WITH_THE_RUN_PYTHON'
    resources=dict(control.DEFAULT_RESOURCES,node=NODE,qos=QOS,partition=PARTITION,wall_seconds=48*3600,memory_MiB=98304)
    plan=dict(cap=2,source=dict(main_commit=source,official_tree=tree),resources=resources)
    disk=os.statvfs(LOCAL);assert disk.f_bavail*disk.f_frsize>=c['storage_min_free_bytes']+(32<<30)
    attempt.mkdir(parents=True);(attempt/'logs').mkdir();(attempt/'scripts').mkdir()
    write_new(attempt/'admission.json',dict(cap_authority=cap_authority,resources=resources,
        disk_available_bytes=disk.f_bavail*disk.f_frsize,queue=control.command(['squeue','-w',NODE,'-h','-o','%i %u %T %b'])))
    frozen=control.freeze_source(plan,attempt)
    lock=dict(instruction=INSTRUCTION,source_commit=source,official_tree=tree,source_directory=frozen['directory'],
              source_members=frozen['members'],configs=[cm],preparation=member(preparation),author_profile=member(PROFILE),
              GPU_qualification='NOT_RUN_USER_DISABLED')
    lp=attempt/'execution-lock.json';write_new(lp,lock)
    key='cf-llama3-memit-fe-author-history'
    argv=[PYTHON,'-B','-u','-m','official.runners.fe_author_history','--config',cm['path'],'--lock',str(lp)]
    env=dict(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OFFICIAL_CODE_COMMIT=source,OFFICIAL_TREE_SHA256=tree,
        WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
    text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
    text+='cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
    script=attempt/'scripts'/'cf.sh'
    with script.open('x') as f:f.write(text)
    cmd=['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--partition='+PARTITION,'--qos='+QOS,
         '--nodelist='+NODE,'--cpus-per-task=8','--mem=98304M','--time=2880','--gres=gpu:1','--job-name=official-s3-'+key,
         '--chdir='+frozen['directory'],'--output='+str(attempt/'logs'/(key+'-%j.out')),'--error='+str(attempt/'logs'/(key+'-%j.err')),str(script)]
    response=control.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
    write_new(attempt/'submitted-cf.json',dict(response=response,command=cmd,job_id=jid,config=cm,argv=argv,script=member(script)))
    d=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
    assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser' and d['UserId'].split('(')[0]==getpass.getuser()
    assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']=='official-s3-'+key
    assert d['ReqNodeList']==NODE and d['QOS']==QOS and d['Partition']==PARTITION and d['Requeue']=='0'
    assert control.gpu_count(d['ReqTRES'])==1 and control.memory_MiB(d['MinMemoryNode'])==98304
    assert control.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
    write_new(attempt/'held-inspection.json',dict(job=d,source_lock=member(lp)))
    control.command(['scontrol','release',jid])
    initial=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
    write_new(attempt/'submission.json',dict(instruction=INSTRUCTION,source=source,official_tree=tree,job_id=jid,
        name='official-s3-'+key,cap_authority=cap_authority,execution_lock=member(lp),initial_snapshot=initial,released=True))
    print(dict(job_id=jid,state=initial['JobState'],reason=initial.get('Reason')))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--template-config',required=True);a.add_argument('--template-assets',required=True)
    a.add_argument('--stream',required=True);a.add_argument('--output',required=True);a.add_argument('--wandb-env',default=WANDB_ENV)
    b=sub.add_parser('register');b.add_argument('--preparation',required=True);b.add_argument('--attempt',required=True)
    b.add_argument('--cap-authority',required=True)
    x=p.parse_args()
    if x.cmd=='prepare':prepare(x.template_config,x.template_assets,x.stream,x.output,x.wandb_env)
    else:register(x.preparation,x.attempt,x.cap_authority)
