"""One deliberate held registration pass; no retries or existing-job mutation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .run import TASK


def job(j):
    text=subprocess.check_output(['scontrol','show','job',str(j),'-o'],text=True)
    keys=('JobId','JobName','UserId','JobState','Priority','Dependency','ReqNodeList',
          'Command','WorkDir','NumCPUs','MinMemoryNode','TimeLimit','Requeue','TresPerNode','SubmitLine')
    result={k:(m.group(1) if (m:=re.search(r'(?:^| )'+k+r'=(\S*)',text)) else None) for k in keys}
    line=re.search(r'(?:^| )SubmitLine=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)',text)
    result['SubmitLine']=line.group(1).strip() if line else None
    return result


def submit(attempt,frontier):
    from project.run_scripts.jlz_interference_l1.cap_common import verify,sha
    receipt=attempt/'submission.json'
    if receipt.exists():raise RuntimeError('DUPLICATE_REGISTRATION_RECEIPT')
    lock=json.loads((attempt/'execution.lock.json').read_text())
    cfg=json.loads((attempt/'config.json').read_text())
    if sha(attempt/'config.json')!=lock['config_sha256']:raise RuntimeError('CONFIG_LOCK')
    for row in lock['source_members']+lock['input_members']+lock['launchers']:verify(row)
    # Actual owner queue must consist only of preserved OURS, held baselines and
    # this new graph on server4. Unknown same-owner server4 jobs block admission.
    ids=subprocess.check_output(['squeue','-h','-u','janghj','-o','%i'],text=True).split()
    before=[]
    for j in ids:
        row=job(j)
        if row['ReqNodeList']!='server4':continue
        if row['JobName'].startswith(TASK):raise RuntimeError('DUPLICATE_TASK_JOB')
        before.append(row)
        if row['JobId'] in ('60106','60107','60108'):continue
        if row['JobId'] in tuple(str(x) for x in range(60917,60924)) and row['Priority']=='0':continue
        raise RuntimeError('UNKNOWN_SERVER4_ADMITTED_JOB_'+row['JobId'])
    if frontier!=['60107']:raise RuntimeError('FRONTIER_NOT_REVIEWED')
    f=job('60107')
    if f['UserId']!='janghj(1025)' or f['ReqNodeList']!='server4' or f['JobName']!='jlz-price-alpha-writer-2k-LLAMA_AE_FREE100':
        raise RuntimeError('FRONTIER_IDENTITY')
    if f['JobState']=='PENDING' and 'afterany:60106' not in f['Dependency']:
        raise RuntimeError('FRONTIER_COVERAGE')
    result=dict(task_id=TASK,stage='REGISTERING_HELD',jobs={},before=before,
        project_cap=2,new_graph_width=2,dependencies={},held_inspection={},source=lock['source_commit'])
    def persist():
        receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    persist()
    roles=list(cfg['cells'])+['collector']
    for role in roles:
        dep='afterany:'+':'.join(frontier if role!='collector' else list(result['jobs'].values()))
        mem='59392M' if role!='collector' else '24576M';wall='2-00:00:00' if role!='collector' else '04:00:00'
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
              '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',
              '--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),'--mem='+mem,'--time='+wall,
              '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err')),
              '--dependency='+dep]
        if role!='collector':argv.append('--gres=gpu:1')
        argv.append(str(attempt/(role+'.sh')))
        completed=subprocess.run(argv,text=True,capture_output=True)
        if completed.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s]+)?\s*',completed.stdout):
            result.update(stage='REGISTRATION_BLOCKED',error=dict(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr));persist()
            raise RuntimeError('REGISTRATION_FAILED_NO_AUTORETRY')
        j=completed.stdout.strip().split(';')[0];result['jobs'][role]=j;result['dependencies'][role]=dep;persist()
        r=job(j);result['held_inspection'][role]=r
        assert r['UserId']=='janghj(1025)' and r['ReqNodeList']=='server4' and r['JobState']=='PENDING' and r['Priority']=='0'
        assert r['JobName']==TASK+'-'+role
        assert '--export=NONE' in (r['SubmitLine'] or '') and '--no-requeue' in r['SubmitLine']
        assert r['Command']==str(attempt/(role+'.sh')) and r['WorkDir']==str(attempt/'source') and r['Requeue']=='0'
        assert r['NumCPUs'] in ('8','8-14') and r['MinMemoryNode']==('58G' if role!='collector' else '24G')
        assert r['TresPerNode']==('gres/gpu:1' if role!='collector' else None)
        assert r['TimeLimit']==wall
        actual_ids=set(re.findall(r'afterany:(\d+)',r['Dependency']))
        assert actual_ids==set(dep.split(':')[1:])
        persist()
    for j in result['jobs'].values():subprocess.run(['scontrol','release',j],check=True)
    result['stage']='RELEASED';result['initial']={k:job(j) for k,j in result['jobs'].items()};persist()
    print(json.dumps(dict(stage=result['stage'],jobs=result['jobs'],dependencies=result['dependencies'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--frontier',nargs='+',required=True)
    a=p.parse_args();submit(a.attempt.resolve(),a.frontier)
