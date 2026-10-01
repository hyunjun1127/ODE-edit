"""One held/inspected/released DAG. No polling, implicit retry or job mutation."""
import argparse
import fnmatch
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
from .common import write, sha, require

def command(argv):
    r=subprocess.run(argv,text=True,capture_output=True)
    require(r.returncode==0,repr(argv)+': '+r.stderr+r.stdout)
    return r.stdout.strip()

def resource_snapshot(exclude=()):
    raw=command(['squeue','-h','-r','-w','server4','-o','%i|%u|%j|%T|%b|%R'])
    rows=[]
    for line in raw.splitlines():
        job,user,name,state,gres,reason=line.split('|',5)
        if job in exclude:continue
        if user!=getpass.getuser() and not any(fnmatch.fnmatch(name,p) for p in ('odeedit_*','bfode_*','motivation_*','session01_*')):continue
        details=command(['scontrol','show','job',job,'--oneliner'])
        match=re.search(r'\bReqTRES=([^ ]+)',details);require(match,'UNKNOWN_TRES')
        g=re.search(r'gres/gpu=(\d+)',match[1]);count=int(g[1]) if g else 0
        if count:rows.append(dict(job=job,owner=user,name=name,state=state,gpus=count,reason=reason))
    return dict(project_jobs=rows,resource_only=True,raw=raw)

def specs(ids,external=()):
    prep_dep='afterany:'+':'.join(external) if external else None
    return [
        ('prep',True,prep_dep,[]),
        ('pilot-A',True,'afterok:'+ids['prep'] if 'prep' in ids else None,[]),
        ('pilot-B',True,'afterok:'+ids['prep'] if 'prep' in ids else None,[]),
        ('main-A',True,'afterok:'+':'.join(ids[x] for x in ('pilot-A','pilot-B')) if 'pilot-B' in ids else None,[]),
        ('main-B',True,'afterok:'+':'.join(ids[x] for x in ('pilot-A','pilot-B')) if 'pilot-B' in ids else None,[]),
        ('collector',False,'afterany:'+':'.join(ids.values()) if len(ids)==5 else None,[','.join(ids.values())] if len(ids)==5 else []),
    ]

def job_name(name):
    return 'odeedit_jlz_twoarm_collect_s4' if name=='collector' else 'odeedit_jlz_twoarm_s4_'+name.replace('-','_')

def dependency_members(value):
    # Slurm repeats dependency type for each ID in its oneline representation.
    return set(re.findall(r'(afterok|afterany):([0-9]+(?:_[0-9]+)?)',value))

def inspect(job,spec,root,argv):
    name,gpu,dep,args=spec
    text=command(['scontrol','show','job',job,'--oneliner'])
    require(f'JobId={job} ' in text and f'JobName={job_name(name)} ' in text,'HELD_IDENTITY')
    require('UserId='+getpass.getuser()+'(' in text and 'JobState=PENDING ' in text and 'Reason=JobHeldUser ' in text,'HELD_OWNER_STATE')
    for value in ('Requeue=0 ','NumCPUs=8 ','ReqNodeList=server4 ','Partition=gpu ','QOS=lab_gpu_s4 '):require(value in text,'HELD_RESOURCE '+value)
    memory=60416 if gpu else 24576
    require(f'mem={memory}M' in text or f'mem={memory//1024}G' in text,'HELD_MEMORY')
    require(('TresPerNode=gres/gpu:1' in text) if gpu else 'gres/gpu' not in text,'HELD_GPU')
    require('TimeLimit='+('7-00:00:00' if gpu else '04:00:00') in text,'HELD_WALL')
    script=str(root/(name+'.sh'))
    require('Command='+script in text and 'WorkDir='+str(root/'source') in text,'HELD_COMMAND')
    field=re.search(r'\bDependency=([^ ]+)',text);require(field,'HELD_DEPENDENCY_FIELD')
    expected={(dep.split(':')[0],x) for x in dep.split(':')[1:]} if dep else set()
    require(dependency_members(field[1])==expected,'HELD_DEPENDENCY_EXACT')
    command_field=re.search(r'\bCommand=(.*?)(?= [A-Z][A-Za-z]+=|$)',text)
    require(command_field and command_field[1]==script,'HELD_COMMAND_PATH')
    submit_field=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',text)
    require(submit_field and shlex.split(submit_field[1])==argv,'HELD_FULL_ARGV')
    # Scheduler text may omit export/argv details; its stored script must match.
    stored=command(['scontrol','write','batch_script',job,'-'])
    require(stored.strip()==Path(script).read_text().strip(),'HELD_STORED_SCRIPT')
    require('--export=NONE' in argv and '--no-requeue' in argv,'HELD_EXPORT')
    return dict(job=job,name=name,requested_argv=argv,scontrol=text,stored_script_sha256=sha(script))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--resume-held',action='store_true');a=p.parse_args();root=a.run.resolve()
    require(not (root/'submission.json').exists(),'NO_DUPLICATE_SUBMISSION')
    existing={p.stem.removeprefix('submitted-'):json.loads(p.read_text()) for p in root.glob('submitted-*.json')}
    require((not existing and not a.resume_held) or (a.resume_held and len(existing)==6),'EXACT_EXISTING_HELD_GRAPH_REQUIRED')
    lock=json.loads((root/'execution.lock.json').read_text())
    require(lock['resource']['cap']==2 and lock['resource']['mem_mib']==60416,'RESOURCE_LOCK')
    require(sha(root/'config.json')==lock['config_sha256'],'CONFIG_CHANGED')
    for row in lock['launchers']:require(sha(row['path'])==row['sha256'],'LAUNCHER_CHANGED')
    require(shutil.disk_usage(root).free>=lock['resource']['storage_reserve'],'STORAGE_RESERVE')
    check=subprocess.run(['bash',str(Path(lock['worktree'])/'scripts/check-slurm-resource-cap.sh'),'server4','1','60416M'],
          env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),capture_output=True,text=True)
    require(check.returncode in (0,4),'RESOURCE_HELPER_ERROR '+check.stdout+check.stderr)
    helper=dict(exit_code=check.returncode,output=check.stdout+check.stderr,
                limitation='exit4 permits dependency-pending only; no helper PASS relabel')
    before=resource_snapshot(exclude=tuple(row['job'] for row in existing.values()))
    if existing:
        dependencies=[x.removeprefix('--dependency=afterany:') for x in existing['prep']['argv'] if x.startswith('--dependency=afterany:')]
        external=dependencies[0].split(':') if dependencies else []
        require({r['job'] for r in before['project_jobs']}<=set(external),'NEW_EXTERNAL_ADMISSION_KEEP_OWN_HELD')
    else:external=[r['job'] for r in before['project_jobs']]
    # Wait for existing admitted project capacity as a dependency, never alter it.
    # Graph's maximum antichain is 2; prep waits all prior project allocations.
    ids={};submitted=[]
    common=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
            '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',f'--chdir={root}/source']
    for i in range(6):
        spec=specs(ids,external)[i];name,gpu,dependency,args=spec
        argv=common+[f'--job-name={job_name(name)}',
                     '--mem='+('60416M' if gpu else '24576M'),'--time='+('7-00:00:00' if gpu else '04:00:00'),
                     f'--output={root}/{name}-%j.out',f'--error={root}/{name}-%j.err']
        if gpu:argv+=['--gres=gpu:1']
        if dependency:argv+=['--dependency='+dependency]
        if gpu and dependency and dependency.startswith('afterok:'):argv+=['--kill-on-invalid-dep=yes']
        argv+=[str(root/(name+'.sh')),*args]
        if existing:
            require(existing[name]['argv']==argv,'RESUMED_SUBMISSION_ARGV_IDENTITY')
            job=existing[name]['job']
        else:
            job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
            write(root/('submitted-'+name+'.json'),dict(job=job,argv=argv,status='HELD'))
        ids[name]=job
        submitted.append(inspect(job,spec,root,argv))
    later=resource_snapshot(exclude=tuple(ids.values()))
    require({r['job'] for r in later['project_jobs']}<=set(external),'NEW_ADMISSION_RACE_KEEP_OWN_HELD')
    write(root/'held-inspection.json',dict(jobs=submitted,prior_admission=before,prerelease=later,helper=helper,
           execution_lock_sha256=sha(root/'execution.lock.json'),max_simultaneous_task_gpu=2,
           external_afterany=external,graph='prep -> pilot A/B -> main A/B; collector afterany all five',
           dependency_fail='kill-on-invalid-dep yes for science; afterany collector sees missing/failure',
           controller=__file__,controller_sha256=sha(__file__),resumed_exact_held_graph=a.resume_held))
    for name in reversed(list(ids)):
        result=command(['scontrol','release',ids[name]])
        write(root/('released-'+name+'.json'),dict(job=ids[name],release_command_succeeded=True,output=result))
    write(root/'submission.json',dict(status='RELEASED',jobs=ids,execution_lock_sha256=sha(root/'execution.lock.json'),
           source=lock['source_commit'],initial='NOT_OBSERVED',max_simultaneous_task_gpu=2,
           dependency_graph=submitted,no_unrelated_job_mutations=True))
    print(json.dumps(dict(status='RELEASED',jobs=ids,initial='NOT_OBSERVED')))

if __name__=='__main__':main()
