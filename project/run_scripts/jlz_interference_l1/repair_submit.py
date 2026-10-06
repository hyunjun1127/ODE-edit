"""One explicit USER-recalled PRICE repair; no automatic retry or controls restart."""
import argparse,getpass,json,os,re,subprocess
from pathlib import Path
from . import ROOT,LOCAL,TASK,NONCE,require,write,member,sha,verify
from .submit import command,freeze,verify_frozen,resource_inventory,arguments,inspect,job_name

ORIGINAL_SOURCE='a9905b9fccbdb48b9b17e768368afa9e663f0bf5'
ORIGINAL_CONFIG='ad6ca0f4987ecfd400c4b637682ba2dc1a8a2626f22de8594b8d7cadb1b7511d'
ORIGINAL_IDS={'PRICE':'59721','FLAT':'59722','REVERSE':'59723','collector':'59724'}

def evidence(prior):
    require(prior==LOCAL/'attempt','EXACT_FAILED_ATTEMPT')
    submission=json.loads((prior/'submission.json').read_text())
    lock=json.loads(verify(submission['lock']).read_text())
    require(submission['jobs']==ORIGINAL_IDS and submission['source']==lock['source_commit']==ORIGINAL_SOURCE
        and lock['config_sha256']==sha(prior/'config.json')==ORIGINAL_CONFIG,'FAILED_SOURCE_REGISTRATION_IDENTITY')
    terminal=json.loads((prior/'PRICE/terminal.json').read_text())
    failure=json.loads((prior/'PRICE/first-error.json').read_text())
    rollback=json.loads((prior/'PRICE/batch-01/rollback.json').read_text())
    require(terminal['job']=='59721' and terminal['commits']==0 and terminal['source']==ORIGINAL_SOURCE
        and failure['error']=='PROJECTION_SORTED_BREAKPOINT_ROOT_UNAVAILABLE'
        and rollback['verified'] is True and rollback['committed_prefix']==0,'EXACT_REPAIR_FAILURE')
    rows=command(['sacct','-n','-P','-j','59721','--format=JobIDRaw,User,JobName%120,State,ExitCode,ElapsedRaw,AllocTRES'])
    parent=next(line.split('|') for line in rows.splitlines() if line.startswith('59721|'))
    require(parent[1]==getpass.getuser() and parent[2]==job_name('PRICE') and parent[3]=='FAILED','FAILED_PARENT_TERMINAL')
    return dict(original_source=ORIGINAL_SOURCE,original_jobs=ORIGINAL_IDS,parent_accounting=parent,
        failure=member(prior/'PRICE/first-error.json'),rollback=member(prior/'PRICE/batch-01/rollback.json'),
        terminal=member(prior/'PRICE/terminal.json'),original_archive_KEEP=True,checkpoint_resume=False)

def submit(config,attempt,prior):
    config,attempt,prior=map(lambda p:Path(p).resolve(),(config,attempt,prior))
    require(not attempt.exists() and attempt.parent==LOCAL and attempt.name=='repair-59721','CREATE_ONCE_USER_REPAIR')
    require(not list(LOCAL.glob('repair-59721*/submission.json')),'NO_REPAIR_DUPLICATE')
    proof=evidence(prior);c=json.loads(config.read_text())
    require(c.get('execution_arms')==['PRICE'] and c.get('repair',{}).get('failed_job')==59721
        and c['W0_reuse']['status']=='QUALIFIED_EXACT_REUSE','USER_PRICE_REPAIR_SCOPE')
    regression=json.loads(verify(c['repair']['regression']).read_text())
    require(regression['status']=='PASS_REAL_FAILURE_SCALAR_REGRESSION'
        and not regression['tolerances_changed'] and not regression['method_changed'],'ACTUAL_FAILURE_REGRESSION')
    verify(regression['repaired_source'])
    before=resource_inventory()
    require(not any(j['name']==job_name('PRICE') for j in before['jobs']),'ACTIVE_PRICE_DUPLICATE')
    localrow=next(x for x in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
        if x.startswith('server4\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines()
        if x.startswith('server4\t')).split('\t')[1]);cap=min(3,int(localrow[2]),tracked)
    require(cap>=1,'NO_GPU_CAP')
    # Never change original controls. Wait for their exact currently admitted
    # resource lane(s), and any other occupied project lanes needed for cap.
    barriers=[j['job'] for j in before['jobs'] if j['name'].startswith(TASK+'-')]
    for j in before['jobs']:
        if j['name'].startswith(TASK+'-'):
            require(j['job'] in (ORIGINAL_IDS['FLAT'],ORIGINAL_IDS['REVERSE'])
                and 'Command='+str(prior/('FLAT.sh' if j['job']==ORIGINAL_IDS['FLAT'] else 'REVERSE.sh'))+' ' in j['resource_detail'],
                'EXACT_OLD_CONTROL_RESOURCE_IDENTITY')
    if sum(j['gpus'] for j in before['jobs'])+1>cap:barriers=[j['job'] for j in before['jobs']]
    node=command(['scontrol','show','node','server4']);partition=command(['scontrol','show','partition','gpu'])
    lock,c=freeze(config,attempt,roles=('PRICE','collector'));r=c['resources']
    require(r['gpu']==1 and r['host_mib']==59392 and r['hard_host_mib']==60416 and r['task_cap']==2
        and 1<=r['cpu']<=8 and 1<=r['collector_cpu']<=8 and r['collector_host_mib']==24576
        and r['wall']=='2-00:00:00' and r['collector_wall']=='04:00:00','FIXED_REPAIR_RESOURCES')
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server4',
        '--gpus','1','--mem','59392M','--local-limit-mib-per-gpu',localrow[3]])
    require('ALLOW_MEMORY_POLICY' in memory,'MEMORY_POLICY')
    checks=[]
    for role in ('PRICE','collector'):
        argv=[x for x in arguments(role,None,attempt,r) if x not in ('--parsable','--hold')];argv.insert(1,'--test-only')
        result=subprocess.run(argv,text=True,capture_output=True)
        checks.append(dict(role=role,argv=argv,returncode=result.returncode,stderr=result.stderr,stdout=result.stdout,allocation_created=False))
        require(result.returncode==0,'SCHEDULER_RESOURCE_REJECTED:'+result.stderr)
    write(attempt/'scheduler-resource-check.json',dict(checks=checks))
    ids={};mapping={};held=[]
    for role in ('PRICE','collector'):
        parents=barriers if role=='PRICE' else [ids['PRICE']]
        dep='afterany:'+':'.join(parents) if parents else None
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[role]=job;mapping[role]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+role+'.json'),dict(nonce=NONCE,status='HELD',role=role,**mapping[role]))
        held.append(inspect(job,role,dep,argv,attempt,r))
    now=resource_inventory(tuple(ids.values()));previous={j['job'] for j in before['jobs']}
    require({j['job'] for j in now['jobs']}<=previous,'ADMISSION_RACE_KEEP_HELD')
    overlap=sum(j['gpus'] for j in now['jobs'] if j['job'] not in barriers)+1
    require(overlap<=cap,'PROJECT_CAP_KEEP_HELD')
    task_overlap=sum(j['gpus'] for j in now['jobs'] if j['name'].startswith(TASK+'-') and j['job'] not in barriers)+1
    require(task_overlap<=2,'TASK_CAP_KEEP_HELD');verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=held,before=before,prerelease=now,effective_project_cap=cap,
        task_cap=2,maximum_repair_GPU=1,project_overlap_bound=overlap,task_overlap_bound=task_overlap,
        resource_barriers=barriers,node=node,partition=partition,memory_policy=memory,scheduler_checks=checks,
        failure_provenance=proof,all_held_inspected_before_release=True,original_job_mutations=0))
    for role in ('collector','PRICE'):
        result=command(['scontrol','release',ids[role]])
        write(attempt/('released-'+role+'.json'),dict(job=ids[role],command_succeeded=True,result=result))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(nonce=NONCE,task_id=TASK,repair_of=59721,execution_arms=['PRICE'],status='RELEASED',jobs=ids,mapping=mapping,
        source=lock['source_commit'],lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        original_job_provenance=ORIGINAL_IDS,original_job_mutations_by_this_launcher=0,
        user_control_cancellation=c['repair'].get('control_cancellation'),
        W0_reuse=c['W0_reuse'],failure_provenance=proof,
        bounded_initial_snapshot=snapshot,actual_repaired_B1='NOT_OBSERVED',W20='NOT_OBSERVED',
        monitoring_active=False,automatic_resume=False,automatic_retry=False)
    write(attempt/'submission.json',receipt)
    return dict(status='RELEASED',jobs=ids,snapshot=snapshot,W0_new_forward=0,original_job_mutations=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--prior',required=True);a=p.parse_args();print(json.dumps(submit(a.config,a.attempt,a.prior)))
