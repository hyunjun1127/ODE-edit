"""Explicit one-lane upfront Slurm registration. Held inspect before release."""
import argparse
import datetime
import getpass
import json
import os
import re
import subprocess
from pathlib import Path
from .common import require,write,sha,member,INSTRUCTION


def run(argv):
    p=subprocess.run(argv,text=True,capture_output=True)
    require(p.returncode==0,str(argv)+':'+p.stderr);return p.stdout.strip()


def inspect_text(text):
    matches=list(re.finditer(r'(?:^| )([A-Za-z][A-Za-z0-9:/]*)=',text));return {
        m.group(1):text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip() for i,m in enumerate(matches)}


def validate_inspection(fields,job,script,node='ubuntu'):
    require(fields['JobId']==job['job_id'] and fields['UserId'].startswith(getpass.getuser()+'('),'HELD_OWNER')
    require(fields['JobName']==job['name'] and fields['ReqNodeList']==node and fields['Partition']=='gpu','HELD_NAME_NODE')
    require(fields['JobState']=='PENDING' and fields['Priority']=='0','HELD_STATE')
    require(fields['Command']==str(script) and fields['Requeue']=='0','HELD_SOURCE_REQUEUE')
    require(int(fields['NumCPUs'])==8,'HELD_CPU')
    tres=fields['ReqTRES'];require(('gres/gpu=1' in tres)==bool(job['gpu']),'HELD_GPU')
    require('mem='+('24G' if not job['gpu'] else '59G') in tres,'HELD_MEMORY')
    for dep in job['dependency_job_ids']:require(dep in fields['Dependency'],'HELD_DEPENDENCY')
    submitline=fields.get('SubmitLine','');require('--export=NONE' in submitline and '--no-requeue' in submitline,'EXPORT_REQUEUE')
    for arg in job['sbatch_argv'][1:]:require(arg in submitline,'HELD_FULL_ARG:'+arg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);a=p.parse_args()
    require(not (a.attempt/'submission.json').exists(),'DUPLICATE_SUBMISSION_RECEIPT')
    lock=json.loads((a.attempt/'execution.lock.json').read_text());launchers=json.loads((a.attempt/'launchers.json').read_text())
    admission=json.loads(a.admission.read_text());require(admission['instruction']==INSTRUCTION and admission['cap']==1 and admission['host_memory_mib']==60416,'ADMISSION_BOUNDARY')
    observed=datetime.datetime.fromisoformat(admission['observed_at'])
    age=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()
    require(0<=age<=300,'STALE_ADMISSION')
    require(lock['instruction']==INSTRUCTION,'AUTHORITY')
    # Validate local prerequisites before consuming the one-shot registration lock.
    # The lock still excludes a second concurrent invocation before any Slurm write.
    with (a.attempt/'submission.started').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    jobs={};external=admission['predecessor_job_ids']
    for stage in ('q1','A','B','collector'):
        script=Path(launchers[stage]['file']['path']);require(sha(script)==launchers[stage]['file']['sha256'],'LAUNCHER_CHANGED')
        dependencies=[];ids=[]
        if stage=='q1' and external:dependencies=['afterany:'+':'.join(external)];ids=external
        elif stage=='A':dependencies=['afterok:'+jobs['q1']['job_id']];ids=[jobs['q1']['job_id']]
        elif stage=='B':dependencies=['afterok:'+jobs['q1']['job_id'],'afterany:'+jobs['A']['job_id']];ids=[jobs['q1']['job_id'],jobs['A']['job_id']]
        elif stage=='collector':ids=[j['job_id'] for j in jobs.values()];dependencies=['afterany:'+':'.join(ids)]
        gpu=stage!='collector';name='odeedit_jlz_v10_s3_'+stage
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s3','--nodelist=ubuntu','--nodes=1','--ntasks=1','--cpus-per-task=8',
            '--export=NONE','--no-requeue','--chdir='+lock['source_root'],'--job-name='+name,'--mem='+('60416M' if gpu else '24576M'),
            '--time='+('24:00:00' if stage=='q1' else '7-00:00:00' if gpu else '04:00:00'),
            '--output='+str(a.attempt/(stage+'-%j.out')),'--error='+str(a.attempt/(stage+'-%j.err'))]
        if gpu:argv+=['--gres=gpu:1']
        if dependencies:argv+=['--dependency='+','.join(dependencies)]
        if stage in ('A','B'):argv+=['--kill-on-invalid-dep=yes']
        argv+=[str(script)]
        jid=run(argv).split(';')[0];require(jid.isdigit(),'SUBMIT_JOB_ID')
        jobs[stage]=dict(job_id=jid,name=name,gpu=int(gpu),sbatch_argv=argv,dependency_job_ids=ids,dependencies=dependencies)
        write(a.attempt/('registered-'+stage+'.json'),jobs[stage])
        text=run(['scontrol','show','job','-o',jid]);write(a.attempt/('inspection-'+stage+'.json'),dict(job_id=jid,raw=text))
        validate_inspection(inspect_text(text),jobs[stage],script)
    # Every ID is known and inspected before any release; successors depend on this lane.
    write(a.attempt/'submission.json',dict(instruction=INSTRUCTION,source=lock['source_commit'],lock=member(a.attempt/'execution.lock.json'),
        jobs=jobs,admission=member(a.admission),cap=1,all_held_inspected=True,actual_initial='NOT_OBSERVED'))
    for stage in ('collector','B','A','q1'):
        output=run(['scontrol','release',jobs[stage]['job_id']])
        write(a.attempt/('release-'+stage+'.json'),dict(job_id=jobs[stage]['job_id'],exit_code=0,stdout=output,
            released_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(jobs))


if __name__=='__main__':main()
