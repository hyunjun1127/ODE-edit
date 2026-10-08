"""One exact replacement registration; explicitly bound resource lanes."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from .run import TASK
from .submit import job


def submit(attempt):
    from project.run_scripts.jlz_interference_l1.cap_common import verify,sha
    target=attempt/'submission.json'
    if target.exists():raise RuntimeError('DUPLICATE_SUBMISSION_NO_AUTORETRY')
    cfg=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    cap=cfg.get('project_gpu_cap',2)
    if cap not in (2,3) or lock['project_gpu_cap']!=cap:raise RuntimeError('PROJECT_CAP_BINDING')
    if sha(attempt/'config.json')!=lock['config_sha256']:raise RuntimeError('CONFIG_SHA')
    for row in lock['source_members']+lock['input_members']+lock['launchers']:verify(row)
    before=[];frontier=[]
    for j in subprocess.check_output(['squeue','-h','-u','janghj','-o','%i'],text=True).split():
        r=job(j)
        if r['ReqNodeList']!='server4':continue
        before.append(r)
        if r['JobId'] in [str(x) for x in range(60917,60924)]:
            if r['Priority']!='0' or r['JobState']!='PENDING':raise RuntimeError('BASELINE_HOLD_NOT_PRESERVED')
            continue
        if r['JobId']=='60108' and r['TresPerNode'] is None:continue
        if r['JobId']=='60107' and r['JobName']=='jlz-price-alpha-writer-2k-LLAMA_AE_FREE100':
            frontier.append('60107');continue
        raise RuntimeError('UNEXPECTED_OWN_SERVER4_JOB_'+j)
    roles=[x for x in ('LLAMA_REPRO','GPTJ_M1','GPT2XL_M1_M2','GPT2XL_M1_M3','GPT2XL_M1_M2_M3') if x in cfg['cells']]
    if set(roles)!=set(cfg['cells']):raise RuntimeError('UNKNOWN_CELL')
    data=dict(task_id=TASK,repair='USER_KV_GENERATION_AND_GPT2_INPUT',source=lock['source_commit'],
        before=before,project_cap=cap,stage='REGISTERING_HELD',jobs={},dependencies={},held_inspection={})
    def persist():target.write_text(json.dumps(data,indent=2)+'\n')
    persist();tails=list(frontier)*cap if frontier else [None]*cap
    for index,role in enumerate(roles+['collector']):
        collector=role=='collector'
        dependencies=list(data['jobs'].values()) if collector else ([tails[index%cap]] if tails[index%cap] else [])
        dep='afterany:'+':'.join(dependencies) if dependencies else None
        wall='04:00:00' if collector else '2-00:00:00';mem='24576M' if collector else '59392M'
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
            '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',
            '--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),'--mem='+mem,'--time='+wall,
            '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
        if dep:argv.append('--dependency='+dep)
        if not collector:argv.append('--gres=gpu:1')
        argv.append(str(attempt/(role+'.sh')))
        p=subprocess.run(argv,text=True,capture_output=True)
        if p.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s]+)?\s*',p.stdout):
            data.update(stage='SUBMISSION_BLOCKED',error=dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr));persist()
            raise RuntimeError('REGISTRATION_BLOCKED_NO_AUTORETRY')
        j=p.stdout.strip().split(';')[0];data['jobs'][role]=j;data['dependencies'][role]=dep;persist()
        r=job(j);data['held_inspection'][role]=r
        assert r['UserId']=='janghj(1025)' and r['ReqNodeList']=='server4'
        assert r['JobState']=='PENDING' and r['Priority']=='0' and r['JobName']==TASK+'-'+role
        assert r['Command']==str(attempt/(role+'.sh')) and r['WorkDir']==str(attempt/'source')
        assert r['Requeue']=='0' and '--export=NONE' in r['SubmitLine'] and '--no-requeue' in r['SubmitLine']
        assert r['NumCPUs'] in ('8','8-14') and r['MinMemoryNode']==('24G' if collector else '58G')
        assert r['TresPerNode']==(None if collector else 'gres/gpu:1') and r['TimeLimit']==wall
        assert set(re.findall(r'afterany:(\d+)',r['Dependency']))==set(dependencies)
        if not collector:tails[index%cap]=j
        persist()
    for j in data['jobs'].values():subprocess.run(['scontrol','release',j],check=True)
    data.update(stage='RELEASED',initial={k:job(j) for k,j in data['jobs'].items()});persist()
    print(json.dumps({k:data[k] for k in ('stage','jobs','dependencies')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    submit(p.parse_args().attempt.resolve())
