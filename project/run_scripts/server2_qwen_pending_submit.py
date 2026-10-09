"""One USER-authorized pending-only refresh; keep two original FT chains."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from official.experiments.prepare import file_sha,read,write_new
from official.runners.server2.qwen_plan import rows
from official.runners.server2.qwen_eval_refresh import planned
from project.run_scripts import server2_qwen_submit as base
from project.run_scripts import server2_qwen_archive as archive
from project.run_scripts.server2_qwen_pending_control import OLD,EVAL,OUT,state

def prepare_inputs(prep):
    base.check(not prep.exists(),'PREPARATION_EXISTS')
    cancellation=read(OUT/'cancellation.json')
    base.check({j['job']['job_id'] for j in cancellation['kept']}=={'61898','61900'},'KEPT_SCOPE')
    kept={}
    for j in read(OLD/'released.json')['jobs']:
        if j['job_id'] not in ('61898','61900'):continue
        state(j,OLD,'qwen')
        p=OLD/'configs'/(j['cell']+'.json')
        kept[j['cell']]=dict(root=str(OLD),job_id=j['job_id'],source=j['source'],
            config_path=str(p),config_file_sha256=file_sha(p))
    write_new(prep/'kept.json',kept)
    write_new(prep/'w0-parent.json',dict(root=str(OLD),producer_source=read(OLD/'source-lock.json')['code_commit'],
        source_lock=dict(path=str(OLD/'source-lock.json'),sha256=file_sha(OLD/'source-lock.json')),
        asset_preflight=dict(path=str(OLD/'asset-preflight.json'),sha256=file_sha(OLD/'asset-preflight.json'))))
    for row in rows():write_new(prep/'configs'/(row['logical_main_row']+'.json'),row['config'])
    shutil.copytree(OLD/'streams',prep/'streams')
    write_new(prep/'assets.candidate.json',read(OLD/'assets.json'))
    # Actual tokenizer only; no model or GPU qualification.
    from transformers import AutoTokenizer
    from official.evaluation.zsre_query_parity import compare_queries
    assets=read(OLD/'asset-preflight.json')
    tokenizer=AutoTokenizer.from_pretrained(assets['assets']['model_snapshot']['path'],local_files_only=True,use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    _,records=__import__('official.runners.server2.qwen_run',fromlist=['validate_stream']).validate_stream(prep/'streams/zsre-stream.json','zsre')
    proof=compare_queries(tokenizer,records,model_family='qwen25')
    write_new(prep/'query-parity.json',proof)
    base.check(proof.get('status')=='PASS_CPU_QUERY_ONLY','PUBLIC_QUERY_PARITY_FAILED:'+str(proof.get('status')))

def admission():
    # Check every same-owner requested/allocated server2 GPU, not job-name only.
    known={'61898','61900',*map(str,range(61942,61948))};observed=[]
    for raw in base.cmd(['scontrol','show','job','-o']).splitlines():
        f=dict(x.split('=',1) for x in raw.split() if '=' in x)
        if not f.get('UserId','').startswith('janghj('):continue
        if 'server2' not in (f.get('NodeList'),f.get('ReqNodeList')):continue
        if f.get('JobState') not in ('RUNNING','COMPLETING','CONFIGURING','PENDING'):continue
        if 'gres/gpu' not in f.get('ReqTRES',''):continue
        base.check(f['JobId'] in known,'UNBOUND_CURRENT_GPU_FRONTIER:'+f['JobId'])
        observed.append(f)
    for j in read(EVAL/'released.json')['jobs']:
        f=state(j,EVAL,'eval')['fields']
        base.check(f['JobState']=='PENDING' and f['Reason']=='JobHeldUser','EVAL_HOLD_LOST')
    for j in read(OLD/'released.json')['jobs']:
        if j['job_id'] in ('61898','61900'):state(j,OLD,'qwen')
    qos=base.cmd(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxTRESPU'])
    base.check('gres/gpu=4' in qos,'QOS_CAP_NOT_FOUR')
    return dict(at=base.now(),cap=4,authority='Latest direct USER server2 cap4; pending Qwen refresh',
        jobs=observed,node=base.cmd(['scontrol','show','node','server2','-o']),qos=qos,
        local_caps=base.cmd(['sed','-n','1,8p','/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv']))

def submit(root):
    base.verify(root,True)
    base.check(not (root/'registration-started.json').exists(),'NO_DUPLICATE_REGISTRATION')
    write_new(root/'registration-started.json',admission())
    kept=read(root/'kept.json');jobs=[];ids={};previous_archive=None
    order=[r for m in ('MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE') for r in rows() if r['config']['method']==m]
    for cell,old in kept.items():
        deps=[old['job_id']]+([previous_archive] if previous_archive else [])
        item=base.register(root,cell,'archive',deps,None);jobs.append(item);previous_archive=item['job_id']
    for row,lane,logical_deps in planned(order,{'cf':'61898','zsre':'61900'}):
        cell=row['logical_main_row'];deps=[ids.get(d,d) for d in logical_deps]
        item=base.register(root,cell,'gpu',deps,row['config']);jobs.append(item);ids[cell]=item['job_id']
        archive.registered(root,cell,item['job_id'],item['submitted_at'])
        item=base.register(root,cell,'archive',[item['job_id'],previous_archive],None)
        jobs.append(item);previous_archive=item['job_id']
    jobs.append(base.register(root,'collector','collector',list(dict.fromkeys(['61898','61900']+[j['job_id'] for j in jobs])),None))
    write_new(root/'submission.json',dict(stage='ALL_HELD',jobs=jobs,kept=kept,qualification='NOT_RUN_USER_DISABLED'))
    base.verify(root)
    for j in jobs:base.inspect(root,j)
    # Preserve existing eval science IDs/source; replace only cancelled resource edges.
    mapping={'61942':ids['qwen25-cf-sphere'],'61943':ids['qwen25-zsre-sphere'],
             '61944':ids['qwen25-cf-memit_fe'],'61945':ids['qwen25-zsre-memit_fe']}
    ejobs=read(EVAL/'released.json')['jobs']
    for j in ejobs:
        if j['job_id'] not in mapping:continue
        before=state(j,EVAL,'eval');base.check(before['fields']['JobState']=='PENDING','EVAL_STATE')
        dep=mapping[j['job_id']]
        base.cmd(['scontrol','update','JobId='+j['job_id'],'Dependency=afterany:'+dep])
        after=state(j,EVAL,'eval')
        base.check(after['fields']['Reason']=='JobHeldUser' and ('afterany:'+dep) in after['fields']['Dependency'],'EVAL_REBIND_FAILED')
        write_new(root/'eval-resource-rebind'/ (j['job_id']+'.json'),dict(before=before,after=after,science_source_changed=False))
    # Release dependents first; all resource dependencies remain in place.
    for j in reversed(ejobs):
        state(j,EVAL,'eval');base.cmd(['scontrol','release',j['job_id']])
        write_new(root/'eval-releases'/(j['job_id']+'.json'),dict(at=base.now(),job_id=j['job_id']))
    for j in reversed(jobs):
        base.cmd(['scontrol','release',j['job_id']])
        write_new(root/'releases'/(j['job_id']+'.json'),dict(at=base.now(),job_id=j['job_id']))
    snapshot=base.cmd(['squeue','-j',','.join(['61898','61900']+[j['job_id'] for j in jobs+ejobs]),'-h','-o','%i|%j|%T|%E'])
    write_new(root/'released.json',dict(stage='RELEASED',at=base.now(),jobs=jobs,kept=kept,eval_resource_rebind=mapping,snapshot=snapshot))
    print(json.dumps(dict(stage='RELEASED',jobs=jobs,snapshot=snapshot)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','prepare','submit']);p.add_argument('--root',type=Path,required=True);p.add_argument('--preparation',type=Path);a=p.parse_args()
    if a.action=='inputs':prepare_inputs(a.root)
    elif a.action=='prepare':base.prepare(a.root,a.preparation)
    else:submit(a.root)
