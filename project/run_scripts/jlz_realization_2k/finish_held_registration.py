"""One explicit same-attempt held-registration repair, not a science retry."""
import argparse
import json
import getpass
from .common import *
from .submit import command,verify_frozen,admission,dependencies,arguments,inspect

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args()
    attempt=args.attempt.resolve()
    require(attempt==LOCAL/'attempt-r2' and not (attempt/'submission.json').exists(),'EXACT_UNRELEASED_ATTEMPT')
    lock,c=verify_frozen(attempt)
    require(lock['source_commit']=='ab6f1bda4688ffc275dd5ff47649899d5a80093c','SEALED_EXECUTION_SOURCE')
    require(not command(['git','status','--porcelain','--','project/run_scripts/jlz_realization_2k'],cwd=ROOT),'COMMITTED_REPAIR')
    existing=json.loads((attempt/'submitted-main-A.json').read_text())
    require(existing['job']=='57899' and existing['dependency'] is None,'EXACT_HELD_A')
    q=command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%j|%T'])
    require([line.split('|')[0] for line in q.splitlines() if 'odeedit_jlz_v9_2k_s4_' in line]==['57899'],'NO_DUPLICATE_NEW_TASK')
    local=int(next(x for x in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[2])
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    require(min(local,tracked,2)==2,'CAP_CHANGED_KEEP_HELD')
    before=admission(('57899',));require(not before['jobs'],'OTHER_PROJECT_ADMISSION_CHANGED_KEEP_HELD')
    ids={'main-A':'57899'};mapping={'main-A':{k:existing[k] for k in ('job','dependency','argv')}}
    expected=arguments('main-A',None,attempt,c['resources']);require(existing['argv']==expected,'ORIGINAL_ARGV')
    inspections=[inspect('57899','main-A',None,expected,attempt,c['resources'])]
    for name in ('main-B','collector'):
        dep=dependencies(('afterany',list(ids.values()))) if name=='collector' else None
        argv=arguments(name,dep,attempt,c['resources']);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[name]=job;mapping[name]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+name+'.json'),dict(instruction=INSTRUCTION,status='HELD',**mapping[name]))
        inspections.append(inspect(job,name,dep,argv,attempt,c['resources']))
    after=admission(tuple(ids.values()));require(not after['jobs'],'ADMISSION_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=inspections,pre_admission=before,prerelease=after,cap=2,
        node=command(['scontrol','show','node','server4']),partition=command(['scontrol','show','partition','gpu']),
        all_jobs_held_before_release=True,source_lock=member(attempt/'execution.lock.json'),
        registration_repair_source=command(['git','rev-parse','HEAD'],cwd=ROOT),
        registration_repair=member(Path(__file__)),inspection_repair=member(ROOT/'project/run_scripts/jlz_realization_2k/submit.py'),
        original_A_preserved=True,no_science_retry=True))
    for name in ('collector','main-B','main-A'):
        output=command(['scontrol','release',ids[name]])
        write(attempt/('released-'+name+'.json'),dict(job=ids[name],command_succeeded=True,output=output))
    write(attempt/'submission.json',dict(instruction_id=INSTRUCTION,status='RELEASED',jobs=ids,mapping=mapping,cap=2,
        lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        initial='NOT_OBSERVED',no_other_job_mutation=True,automatic_resume=False))
    print(json.dumps(dict(status='RELEASED',jobs=ids,initial='NOT_OBSERVED')))

if __name__=='__main__':main()
