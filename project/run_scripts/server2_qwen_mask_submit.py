"""Single deliberate eight-cold + one FT evaluation registration, four lanes."""
import argparse
import json
import os
from pathlib import Path
import shutil
from official.experiments.prepare import read,write_new,file_sha
from official.runners.server2.qwen_mask_profile import rows,INSTRUCTION,GENERATOR_SHA,DEFERRED
from project.run_scripts import server2_qwen_submit as base

OLD=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-native-eval-r2')
FIRST=OLD.parent/'registration-r1'
EVAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server2/zsre-2k-reeval-20261009/registration-r1')
OLD_IDS={'qwen25-cf-memit':'61954','qwen25-zsre-memit':'61956',
 'qwen25-cf-alphaedit':'61958','qwen25-zsre-alphaedit':'61960',
 'qwen25-cf-memit_fe':'61966','qwen25-zsre-memit_fe':'61968',
 'qwen25-cf-sphere':'61970','qwen25-zsre-sphere':'61972'}

def prepare_inputs(out):
    from official.runners.server2.zsre_reeval import member,runtime
    from official.runners.server2.qwen_run import validate_config,validate_stream
    from official.evaluation.zsre_query_parity import compare_queries
    from transformers import AutoTokenizer
    base.check(not out.exists(),'PREPARATION_EXISTS_NO_RETRY')
    base.verify(OLD)
    for row in rows():
        c=row['config'];validate_config(c)
        old=read(OLD/'configs'/(row['logical_main_row']+'.json'))
        stripped={k:v for k,v in c.items() if k not in ('config_sha256','mask_repair_instruction','context_generator_sha256','generation_schedule')}
        base.check(stripped=={k:v for k,v in old.items() if k!='config_sha256'},'NATIVE_SCIENCE_CHANGED')
        write_new(out/'configs'/(row['logical_main_row']+'.json'),c)
    shutil.copytree(OLD/'streams',out/'streams')
    for ds in ('cf','zsre'):validate_stream(out/'streams'/f'{ds}-stream.json',ds)
    write_new(out/'assets.candidate.json',read(OLD/'assets.json'))
    write_new(out/'w0-parent.json',read(OLD/'w0-parent.json'))
    kept={}
    for cell,jid,root in [('qwen25-cf-ft','61898',FIRST),('qwen25-zsre-ft','61900',FIRST),
            ('qwen25-cf-alphaedit_blue','61962',OLD),('qwen25-zsre-alphaedit_blue','61964',OLD)]:
        cp=root/'configs'/f'{cell}.json'
        kept[cell]=dict(root=str(root),job_id=jid,config_path=str(cp),config_file_sha256=file_sha(cp),
            source=read(root/'source-lock.json')['code_commit'],unchanged=True)
    write_new(out/'kept.json',kept)
    write_new(out/'mask-profile.json',dict(instruction_id=INSTRUCTION,generator_sha256=GENERATOR_SHA,
        cold_start=True,old_CP_resume=False,CF_generation=DEFERRED,old_to_cells=OLD_IDS,
        qualification='NOT_RUN_USER_DISABLED',context_receipt='ACTUAL_FIRST_CALL_PENDING',
        no_raw_broadcast='NO_BROADCAST_NOT_REQUIRED'))
    # Original FT terminal, 20 committed batches and final physical checkpoint.
    original=FIRST/'runs/qwen25-zsre-ft';t=read(original/'terminal.json')
    base.check(t['status']=='W20_COMPLETE' and t['completed_edits']==2000 and t['actual_job_id']=='61900','FT_FINAL_ENDPOINT')
    base.check(len(t['commits'])==20,'FT_COMMITS')
    for i,m in enumerate(t['commits'],1):
        base.check(file_sha(m['path'])==m['sha256'] and read(m['path'])['completed_batch']==i,'FT_COMMIT_IDENTITY')
    pointer=member(original/'checkpoint/latest.json');p=read(pointer['path'])
    checkpoint=member(original/'checkpoint'/p['file'])
    base.check(p['batch']==20 and p['final_W20'] and p['sha256']==checkpoint['sha256']==t['checkpoint']['sha256'],'FT_CP_SHA')
    assets=read(FIRST/'asset-preflight.json');snapshot=assets['assets']['model_snapshot']['path']
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    tok.padding_side='right';tok.pad_token=tok.eos_token
    stream=member(out/'streams/zsre-stream.json');proof=compare_queries(tok,read(stream['path']),model_family='qwen25')
    from official.runners.server2.qwen_mask_ft_eval import COUNTS
    base.check(proof['requests']==2000 and proof['token_denominators']==COUNTS,'QWEN_QUERY_PARITY')
    write_new(out/'query-parity.json',proof)
    row=dict(method='FT',original_job_id='61900',identity=t['checkpoint_identity'],checkpoint=checkpoint,
        latest=pointer,original_config=member(FIRST/'configs/qwen25-zsre-ft.json'))
    write_new(out/'ft-eval-inputs.json',dict(row=row,stream=stream,runtime=runtime(),query_proof=proof,
        model_snapshot=snapshot,model_revision=t['checkpoint_identity']['model_revision'],
        tokenizer_sha256=t['checkpoint_identity']['tokenizer_sha256'],source_terminal=member(original/'terminal.json')))
    print(json.dumps(dict(stage='CPU_INPUTS_PREPARED',cells=8,FT_checkpoint=checkpoint,query_proof=proof)))

def submit(root):
    base.verify(root,True)
    base.check(not (root/'registration-started.json').exists(),'NO_DUPLICATE_REGISTRATION_PASS')
    # Exact protected eval source/command frontier, not job-name guessing.
    from project.run_scripts.server2_qwen_mask_control import exact
    ejobs=read(EVAL/'released.json')['jobs'];frontier=['61946','61947','61944','61945']
    evidence={jid:exact(next(j for j in ejobs if j['job_id']==jid),EVAL,True) for jid in frontier}
    queue=base.cmd(['scontrol','show','job','-o'])
    own=[]
    for line in queue.splitlines():
        f=dict(x.split('=',1) for x in line.split() if '=' in x)
        if f.get('UserId','').startswith('janghj(') and (f.get('ReqNodeList')=='server2' or f.get('NodeList')=='server2') and 'gres/gpu' in f.get('ReqTRES','') and f.get('JobState') in ('RUNNING','PENDING','CONFIGURING','COMPLETING'): own.append(f)
    allowed={'61898','61962','61964',*map(str,range(61942,61948))}
    base.check(all(f['JobId'] in allowed for f in own),'UNKNOWN_OWN_GPU_FRONTIER')
    qos=base.cmd(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxTRESPU'])
    base.check('gres/gpu=4' in qos,'CURRENT_QOS_CAP_CHANGED')
    write_new(root/'registration-started.json',dict(at=base.now(),authority=INSTRUCTION,protected_frontier=evidence,
        owned_queue=own,node=base.cmd(['scontrol','show','node','server2','-o']),qos=qos,
        local_caps=base.cmd(['sed','-n','1,12p','/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv']),
        free_bytes=shutil.disk_usage(root).free,cap=4))
    jobs=[];lanes=list(frontier)
    ft=base.register(root,'qwen25-zsre-ft-eval','gpu',[lanes[3]],None)
    jobs.append(ft);lanes[3]=ft['job_id']
    prior_archive=None
    for i,row in enumerate(rows()):
        cell=row['logical_main_row'];lane=i%4
        gpu=base.register(root,cell,'gpu',[lanes[lane]],row['config']);jobs.append(gpu);lanes[lane]=gpu['job_id']
        base.archive.registered(root,cell,gpu['job_id'],gpu['submitted_at'])
        arc=base.register(root,cell,'archive',[gpu['job_id']]+([prior_archive] if prior_archive else []),row['config'])
        jobs.append(arc);prior_archive=arc['job_id']
    jobs.append(base.register(root,'collector','collector',[j['job_id'] for j in jobs],None))
    write_new(root/'submission.json',dict(stage='ALL_HELD',jobs=jobs,lanes=lanes,qualification='NOT_RUN_USER_DISABLED'))
    base.verify(root)
    for j in jobs:base.inspect(root,j)
    for j in reversed(jobs):
        base.cmd(['scontrol','release',j['job_id']])
        write_new(root/'releases'/(j['job_id']+'.json'),dict(job_id=j['job_id'],at=base.now()))
    snapshot=base.cmd(['squeue','-j',','.join(j['job_id'] for j in jobs),'-h','-o','%i|%j|%T|%E'])
    write_new(root/'released.json',dict(at=base.now(),jobs=jobs,snapshot=snapshot,lanes=lanes))
    print(json.dumps(dict(stage='RELEASED',jobs=jobs,snapshot=snapshot)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inputs','submit']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();prepare_inputs(a.root) if a.command=='inputs' else submit(a.root)
