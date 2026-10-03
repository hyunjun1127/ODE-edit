"""One GPU + exact afterany CPU collector. No recurring monitor/retry."""
import argparse
import datetime
import getpass
import json
import re
import shlex
import subprocess
from pathlib import Path
from .common import *


def command(argv):
    p=subprocess.run(argv,text=True,capture_output=True)
    require(p.returncode==0,str(argv)+':'+p.stderr);return p.stdout.strip()


def fields(text):
    matches=list(re.finditer(r'(?:^| )([A-Za-z][A-Za-z0-9:/]*)=',text))
    return {m.group(1):text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip() for i,m in enumerate(matches)}


def inspect(text,jid,stage,script,argv,gpu_id):
    f=fields(text);gpu=stage=='gpu'
    require(f['JobId']==jid and f['UserId'].startswith(getpass.getuser()+'('),'HELD_OWNER')
    require(f['JobName']=='odeedit_jlz_v10_a_b1_diag_s1_'+stage and f['ReqNodeList']=='devbox' and f['Partition']=='gpu','HELD_TASK_NODE')
    require(f['JobState']=='PENDING' and f['Priority']=='0' and f['Requeue']=='0','HELD_STATE_RESOURCE')
    # Slurm may expose a pre-allocation topology range (e.g. 8-14).
    # Contract binds requested CPU count, not the pending upper topology bound.
    require(f.get('CPUs/Task')=='8' and re.search(r'(?:^|,)cpu=8(?:,|$)',f['ReqTRES']) is not None
            and int(f['NumCPUs'].split('-')[0])==8,'HELD_REQUESTED_CPU')
    require(f['Command'].split()[0]==str(script),'HELD_LAUNCHER')
    require(('gres/gpu=1' in f['ReqTRES'])==gpu and ('mem=96G' if gpu else 'mem=24G') in f['ReqTRES'],'HELD_GPU_MEMORY')
    require(f['TimeLimit']==('06:00:00' if gpu else '02:00:00'),'HELD_WALL')
    require(f['Dependency']=='(null)' if gpu else 'afterany:'+gpu_id in f['Dependency'],'HELD_DEPENDENCY')
    require('--export=NONE' in f['SubmitLine'] and '--no-requeue' in f['SubmitLine'],'HELD_EXPORT')
    for arg in argv[1:]:require(arg in f['SubmitLine'],'HELD_ARG:'+arg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--resume-held-registration',action='store_true');a=p.parse_args()
    if not a.resume_held_registration:require(not (a.attempt/'submission.started').exists(),'NONCE_ALREADY_REGISTERED')
    else:
        require((a.attempt/'submission.started').exists() and (a.attempt/'registered-gpu.json').exists()
                and not (a.attempt/'submission.json').exists() and not (a.attempt/'release-gpu.json').exists(),'ONLY_INCOMPLETE_HELD_REGISTRATION')
    admission=json.loads(a.admission.read_text());now=datetime.datetime.now(datetime.timezone.utc)
    require(admission['instruction']==NONCE and admission['project_GPU_after']<=admission['effective_cap'] and admission['task_existing_GPU']==0,'ADMISSION')
    require(0<=(now-datetime.datetime.fromisoformat(admission['observed_at'])).total_seconds()<300,'STALE_ADMISSION')
    lock=json.loads((a.attempt/'execution.lock.json').read_text());launchers=json.loads((a.attempt/'launchers.json').read_text())
    require(lock['instruction']==NONCE,'AUTHORITY')
    for row in lock['members']+lock['runtime_sources']:
        require(sha(row['path'])==row['sha256'],'PRE_SUBMIT_CLOSURE_CHANGED')
    if not a.resume_held_registration:write(a.attempt/'submission.started',dict(instruction=NONCE,time=now.isoformat()))
    jobs={}
    for stage in ('gpu','collector'):
        gpu=stage=='gpu';script=Path(launchers[stage]['file']['path'])
        require(sha(script)==launchers[stage]['file']['sha256'],'LAUNCHER_IDENTITY')
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
            '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',
            '--job-name=odeedit_jlz_v10_a_b1_diag_s1_'+stage,'--mem='+('98304M' if gpu else '24576M'),
            '--time='+('06:00:00' if gpu else '02:00:00'),'--chdir='+lock['source_root'],
            '--output='+str(a.attempt/(stage+'-%j.out')),'--error='+str(a.attempt/(stage+'-%j.err'))]
        if gpu:argv.append('--gres=gpu:1')
        else:argv.append('--dependency=afterany:'+jobs['gpu']['id'])
        argv.append(str(script))
        if not gpu:argv+=['--gpu-job',jobs['gpu']['id']]
        registered=a.attempt/('registered-'+stage+'.json')
        if registered.exists():
            require(a.resume_held_registration,'DUPLICATE_REGISTERED_JOB')
            jobs[stage]=json.loads(registered.read_text());require(jobs[stage]['argv']==argv,'RESUME_EXACT_ARGS')
            jid=jobs[stage]['id']
        else:
            jid=command(argv).split(';')[0];require(jid.isdigit(),'JOB_ID')
            jobs[stage]=dict(id=jid,argv=argv,dependency=None if gpu else 'afterany:'+jobs['gpu']['id'])
            write(registered,jobs[stage])
        text=command(['scontrol','show','job','-o',jid])
        write(a.attempt/('inspection-'+stage+('-resume' if a.resume_held_registration else '')+'.json'),dict(raw=text))
        inspect(text,jid,stage,script,argv,jobs['gpu']['id'])
    write(a.attempt/'submission.json',dict(instruction=NONCE,source=lock['source'],lock=member(a.attempt/'execution.lock.json'),
        jobs=jobs,all_held_inspected=True,admission=member(a.admission),actual_GPU='NOT_YET_OBSERVED',
        submission_controller=member(Path(__file__)),resume_held_registration=a.resume_held_registration))
    for stage in ('collector','gpu'):
        result=command(['scontrol','release',jobs[stage]['id']])
        write(a.attempt/('release-'+stage+'.json'),dict(job_id=jobs[stage]['id'],exit=0,response=result,
            time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(jobs=jobs,source=lock['source'],lock_sha=sha(a.attempt/'execution.lock.json'),released=True)))


if __name__=='__main__':main()
