"""All-scope held DAG registration; exact inspection, then release once."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tarfile
from project.run_scripts.jlz_writer_coupled.submit import command,_copy_once,_write_script,launcher,admission,dependencies,dependency_members,expected_dependencies
from .common import ROOT,LOCAL,INSTRUCTION,TASK,require,write,member,sha,digest
from .prepare import AUTHORITY,DESIGN,ENVELOPE,EXCEPTION

SOURCE_PATHS=['project/run_scripts/jlz_realization','project/run_scripts/jlz_writer_coupled',
 'project/run_scripts/jlz_pilot/prompts.py','project/run_scripts/jlz_pilot/__init__.py',
 DESIGN,'plans/global/2026-10-03-jlz-native-writer-v8','docs/methods/jlz-realization-v9.tex',ENVELOPE,EXCEPTION,
 'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv',
 'control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv',
 'scripts/check-slurm-resource-cap.sh','scripts/check-slurm-gpu-cap.sh','scripts/slurm_memory_policy.py']

def freeze(config_path,attempt):
    require(attempt.parent==LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(),'NEW_ATTEMPT_CREATE_ONCE')
    require(not command(['git','status','--porcelain','--',*SOURCE_PATHS],cwd=ROOT),'SOURCE_MUST_BE_COMMITTED')
    config=json.loads(config_path.read_text());require(config['instruction_id']==INSTRUCTION,'CONFIG_TASK')
    require(shutil.disk_usage(LOCAL).free>=config['resources']['reserve_bytes'],'STORAGE_RESERVE')
    commit=command(['git','rev-parse','HEAD'],cwd=ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],cwd=ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCE_PATHS],cwd=ROOT)
    with tarfile.open(archive) as tar:
        members=tar.getmembers();require(len({m.name for m in members})==len(members),'ARCHIVE_DUPLICATE')
        for m in members:require((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'ARCHIVE_PATH')
        tar.extractall(source,filter='data')
    _copy_once(config_path,attempt/'config.json')
    common=['--config',str(attempt/'config.json'),'--attempt',str(attempt)]
    for phase in ('prep','main'):
        for arm in ('A','B'):
            name=phase+'-'+arm
            _write_script(attempt/(name+'.sh'),launcher(source,commit,'project.run_scripts.jlz_realization.run',
                          ['--phase',phase,'--arm',arm,*common]))
    _write_script(attempt/'collector.sh',launcher(source,commit,'project.run_scripts.jlz_realization.collect',
        ['--attempt',str(attempt),'--report',str(attempt/'report')],cpu_only=True))
    sources=[member(p) for p in sorted(source.rglob('*')) if p.is_file()]
    write(attempt/'execution.lock.json',dict(instruction_id=INSTRUCTION,task_id=TASK,source_commit=commit,source_tree=tree,
        archive=member(archive),source_members=sources,source_root_sha256=digest(sources),
        config_sha256=sha(attempt/'config.json'),runtime_sources=config['runtime']['source_members'],native_reference=config['native_reference'],
        launchers=[member(p) for p in sorted(attempt.glob('*.sh'))],resources=config['resources'],
        owner=getpass.getuser(),host='server4',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',noCP=True,
        flow='Q1A/Q1B; both READY/afterok -> independent cold mainA(Q2 in B1)/mainB; afterany collector'))

def verify(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());config=json.loads((attempt/'config.json').read_text())
    require(lock['instruction_id']==config['instruction_id']==INSTRUCTION and sha(attempt/'config.json')==lock['config_sha256'],'LOCK')
    for row in lock['source_members']+lock['runtime_sources']+lock['native_reference']+lock['launchers']+[lock['archive']]:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'FROZEN_MEMBER')
    for row in config['assets']:
        s=Path(row['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT')
    return lock,config

def wall(name,r):return r['collector_wall'] if name=='collector' else r['qualification_wall'] if name.startswith('prep') else r['wall']
def jobname(name):return 'odeedit_jlz_v9_s4_'+name.replace('-','_')

def arguments(name,dep,attempt,r):
    gpu=name!='collector';mem=r['host_mib'] if gpu else r['collector_host_mib']
    args=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
          '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',
          '--job-name='+jobname(name),'--chdir='+str(attempt/'source'),'--mem='+str(mem)+'M','--time='+wall(name,r),
          '--output='+str(attempt/(name+'-%j.out')),'--error='+str(attempt/(name+'-%j.err'))]
    if gpu:args.append('--gres=gpu:1')
    if dep:
        args.append('--dependency='+dep)
        if 'afterok:' in dep:args.append('--kill-on-invalid-dep=yes')
    return args+[str(attempt/(name+'.sh'))]

def inspect(job,name,dep,args,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);script=attempt/(name+'.sh');gpu=name!='collector'
    for t in [f'JobId={job} ',f'JobName={jobname(name)} ','UserId='+getpass.getuser()+'(',
              'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','NumCPUs=8 ',
              'ReqNodeList=server4 ','Partition=gpu ','QOS=lab_gpu_s4 ']:require(t in detail,'HELD:'+t)
    require(('TresPerNode=gres/gpu:1' in detail) if gpu else 'gres/gpu' not in detail,'GPU_REQUEST')
    mem=r['host_mib'] if gpu else r['collector_host_mib']
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'MEMORY_REQUEST')
    requested_wall=wall(name,r);accepted={requested_wall,'1-00:00:00'} if requested_wall=='24:00:00' else {requested_wall}
    require(any('TimeLimit='+v+' ' in detail for v in accepted),'WALL')
    field=re.search(r'\bDependency=([^ ]+)',detail)
    require(field and dependency_members(field[1])==expected_dependencies(dep),'DEPENDENCY')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'COMMAND_CWD')
    line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(line and shlex.split(line[1])==args,'FULL_SUBMIT_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'SCRIPT_BYTES')
    return dict(job=job,name=name,argv=args,scontrol=detail,launcher=member(script))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    attempt=a.attempt.resolve();freeze(a.config.resolve(),attempt);lock,config=verify(attempt)
    cap_file=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    local_cap=int(next(x for x in cap_file.read_text().splitlines() if x.startswith('server4\t')).split('\t')[2])
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    cap=min(2,local_cap,tracked);require(cap>=1,'NO_ADMITTED_CAPACITY')
    before=admission();external=[r['job'] for r in before['jobs']]
    partition=command(['scontrol','show','partition','gpu']);require('MaxTime=30-00:00:00' in partition,'PARTITION_LIMIT_CHANGED_REVIEW')
    ids={};inspected=[];mapping={}
    for name in ('prep-A','prep-B','main-A','main-B','collector'):
        if name.startswith('prep'):
            dep=dependencies(('afterany',external+([ids['prep-A']] if cap==1 and name=='prep-B' else [])))
        elif name.startswith('main'):
            dep=dependencies(('afterok',[ids['prep-A'],ids['prep-B']]),
                ('afterany',external+([ids['main-A']] if cap==1 and name=='main-B' else [])))
        else:dep=dependencies(('afterany',list(ids.values())))
        argv=arguments(name,dep,attempt,config['resources']);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[name]=job;mapping[name]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+name+'.json'),dict(instruction=INSTRUCTION,status='HELD',**mapping[name]))
        inspected.append(inspect(job,name,dep,argv,attempt,config['resources']))
    after=admission(tuple(ids.values()));require({r['job'] for r in after['jobs']}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    verify(attempt)
    write(attempt/'held-inspection.json',dict(jobs=inspected,pre_admission=before,pre_release=after,
        cap=cap,partition=partition,all_held_before_release=True,source_lock=member(attempt/'execution.lock.json')))
    for name in reversed(list(ids)):
        output=command(['scontrol','release',ids[name]])
        write(attempt/('released-'+name+'.json'),dict(job=ids[name],command_succeeded=True,output=output))
    write(attempt/'submission.json',dict(instruction_id=INSTRUCTION,status='RELEASED',jobs=ids,mapping=mapping,
        initial='NOT_OBSERVED',cap=cap,lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        no_other_job_mutation=True,automatic_resume=False))
    print(json.dumps(dict(status='RELEASED',jobs=ids,initial='NOT_OBSERVED')))

if __name__=='__main__':main()
