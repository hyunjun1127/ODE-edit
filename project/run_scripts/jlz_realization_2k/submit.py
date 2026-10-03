"""Create-once main A/B plus afterany collector, held audit then release. No retry."""
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from project.run_scripts.jlz_writer_coupled.submit import command, _copy_once, _write_script, launcher, admission, dependencies, dependency_members, expected_dependencies
from .common import *
from .schedule import validate_config

SOURCE_PATHS=['project/run_scripts/jlz_realization_2k','project/run_scripts/jlz_realization',
 'project/run_scripts/jlz_writer_coupled','project/run_scripts/jlz_pilot/prompts.py','project/run_scripts/jlz_pilot/__init__.py',
 ENVELOPE,CONTRACT,EXCEPTION,CASE,EVALUATION,'control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv',
 'scripts/check-slurm-resource-cap.sh','scripts/check-slurm-gpu-cap.sh','scripts/slurm_memory_policy.py']

def freeze(config_path,attempt):
    require(attempt.parent==LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(),'NEW_CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCE_PATHS],cwd=ROOT),'SOURCE_MUST_BE_COMMITTED')
    c=json.loads(config_path.read_text());validate_config(c)
    require(c['instruction_id']==INSTRUCTION and c['task_id']==TASK and c['authority']==AUTHORITY,'TASK_AUTHORITY')
    require(shutil.disk_usage(LOCAL).free>=c['resources']['reserve_bytes'],'RESOURCE_BLOCKED_STORAGE')
    source_commit=command(['git','rev-parse','HEAD'],cwd=ROOT)
    source_tree=command(['git','rev-parse','HEAD^{tree}'],cwd=ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),source_commit,*SOURCE_PATHS],cwd=ROOT)
    with tarfile.open(archive) as tar:
        members=tar.getmembers();require(len({m.name for m in members})==len(members),'ARCHIVE_DUPLICATE')
        for m in members:require((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'ARCHIVE_PATH')
        tar.extractall(source,filter='data')
    _copy_once(config_path,attempt/'config.json')
    for arm in ('A','B'):
        _write_script(attempt/('main-'+arm+'.sh'),launcher(source,source_commit,'project.run_scripts.jlz_realization_2k.run',
            ['--arm',arm,'--config',str(attempt/'config.json'),'--attempt',str(attempt)]))
    _write_script(attempt/'collector.sh',launcher(source,source_commit,'project.run_scripts.jlz_realization_2k.collect',
        ['--attempt',str(attempt),'--report',str(attempt/'report')],cpu_only=True))
    sources=[member(p) for p in sorted(source.rglob('*')) if p.is_file()]
    write(attempt/'execution.lock.json',dict(instruction_id=INSTRUCTION,task_id=TASK,source_commit=source_commit,source_tree=source_tree,
        archive=member(archive),source_members=sources,config_sha256=sha(attempt/'config.json'),
        runtime_sources=c['runtime']['source_members'],native_reference=c['native_reference'],
        launchers=[member(p) for p in sorted(attempt.glob('*.sh'))],resources=c['resources'],
        qualification_reuse=c['qualification_reuse'],owner=getpass.getuser(),host='server4',
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',noCP=True,flow='reused Q1 -> two cold20 main lanes -> afterany collector'))

def verify_frozen(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());c=json.loads((attempt/'config.json').read_text())
    require(lock['instruction_id']==c['instruction_id']==INSTRUCTION and lock['owner']==getpass.getuser(),'OWNER_TASK')
    require(sha(attempt/'config.json')==lock['config_sha256'],'CONFIG_HASH')
    for r in lock['source_members']+lock['runtime_sources']+lock['native_reference']+lock['launchers']+[lock['archive']]:verify(r)
    for r in c['assets']:
        s=Path(r['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'ASSET_STAT')
    bridge=json.loads(verify(c['qualification_reuse']['receipt']).read_text())
    for r in bridge['evidence']:verify(r)
    verify(c['w0_reuse']['receipt']);verify(c['observer_identity'])
    require(shutil.disk_usage(LOCAL).free>=c['resources']['reserve_bytes'],'STORAGE_RESERVE')
    return lock,c

def jobname(name):return 'odeedit_jlz_v9_2k_s4_'+name.replace('-','_')
def wall(name,r):return r['collector_wall'] if name=='collector' else r['wall']

def arguments(name,dep,attempt,r):
    gpu=name!='collector';mem=r['host_mib'] if gpu else r['collector_host_mib']
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4','--nodes=1',
        '--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue','--job-name='+jobname(name),
        '--chdir='+str(attempt/'source'),'--mem='+str(mem)+'M','--time='+wall(name,r),
        '--output='+str(attempt/(name+'-%j.out')),'--error='+str(attempt/(name+'-%j.err'))]
    if gpu:argv.append('--gres=gpu:1')
    if dep:argv.append('--dependency='+dep)
    return argv+[str(attempt/(name+'.sh'))]

def inspect(job,name,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);gpu=name!='collector';script=attempt/(name+'.sh')
    for term in [f'JobId={job} ',f'JobName={jobname(name)} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','CPUs/Task=8 ','ReqTRES=cpu=8,','ReqNodeList=server4 ','Partition=gpu ','QOS=lab_gpu_s4 ']:
        require(term in detail,'HELD:'+term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ',detail),'HELD_CPU_REQUEST_RANGE')
    require(('TresPerNode=gres/gpu:1' in detail) if gpu else 'gres/gpu' not in detail,'HELD_GPU')
    mem=r['host_mib'] if gpu else r['collector_host_mib']
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+wall(name,r)+' ' in detail,'HELD_WALL')
    depfield=re.search(r'\bDependency=([^ ]+)',detail)
    require(depfield and dependency_members(depfield[1])==expected_dependencies(dep),'HELD_DEPENDENCY')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'HELD_SOURCE_COMMAND')
    line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(line and shlex.split(line[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,name=name,argv=argv,scontrol=detail,launcher=member(script))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    args=p.parse_args();attempt=args.attempt.resolve()
    # Before create/submit, exact own task prefix must have no active registration.
    queue=command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%j|%T'])
    require(not any('odeedit_jlz_v9_2k_s4_' in line for line in queue.splitlines()),'NO_DUPLICATE_TASK_JOB')
    freeze(args.config.resolve(),attempt);lock,c=verify_frozen(attempt)
    local=int(next(x for x in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[2])
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    cap=min(2,local,tracked);require(cap>=1,'NO_ADMITTED_CAP')
    helper=subprocess.run(['bash',str(ROOT/'scripts/check-slurm-resource-cap.sh'),'server4','1',str(c['resources']['host_mib'])+'M'],
        env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),capture_output=True,text=True)
    require(helper.returncode in (0,4),'RESOURCE_HELPER:'+helper.stdout+helper.stderr)
    partition=command(['scontrol','show','partition','gpu']);node=command(['scontrol','show','node','server4'])
    require('MaxTime=30-00:00:00' in partition,'PARTITION_LIMIT_CHANGED')
    before=admission();external=[r['job'] for r in before['jobs']]
    names=('main-A','main-B','collector');ids={};mapping={};inspection=[]
    for name in names:
        dep=dependencies(('afterany',list(ids.values()))) if name=='collector' else dependencies(('afterany',external+([ids['main-A']] if name=='main-B' and cap==1 else [])))
        argv=arguments(name,dep,attempt,c['resources']);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[name]=job;mapping[name]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+name+'.json'),dict(instruction=INSTRUCTION,status='HELD',**mapping[name]))
        inspection.append(inspect(job,name,dep,argv,attempt,c['resources']))
    after=admission(tuple(ids.values()));require({r['job'] for r in after['jobs']}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=inspection,pre_admission=before,prerelease=after,cap=cap,node=node,partition=partition,
        helper=dict(exit_code=helper.returncode,output=helper.stdout+helper.stderr,exit4_is_pending_not_PASS=True),
        all_jobs_held_before_release=True,source_lock=member(attempt/'execution.lock.json')))
    for name in reversed(names):
        output=command(['scontrol','release',ids[name]])
        write(attempt/('released-'+name+'.json'),dict(job=ids[name],command_succeeded=True,output=output))
    write(attempt/'submission.json',dict(instruction_id=INSTRUCTION,status='RELEASED',jobs=ids,mapping=mapping,cap=cap,
        lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        initial='NOT_OBSERVED',no_other_job_mutation=True,automatic_resume=False))
    print(json.dumps(dict(status='RELEASED',jobs=ids,initial='NOT_OBSERVED')))

if __name__=='__main__':main()
