"""Owner-only exact never-started pending replacement; no experiment retries.

This coordinator never mutates 60001. Every scheduler write has an exact
source/config/script/owner binding and an immediately preceding state check.
Original archives and receipts remain in place. No scientific tensor is read.
"""
import argparse
import ast
import copy
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from project.run_scripts.jlz_interference_l1 import ROOT, member, sha, verify, write, require

AUTHORITY='USER-GH-PRICE-MODEL-RUNS-TRACKING-20261007'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
LOCAL=Path('/data/janghj/ODE-edit/local/price-model-runs-tracking/20261007')
OLD_SOURCE='2440e548be39e55a99747d7847d21a88df419b93'
MEMIT=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/logging-repair-20261007')
ALPHA=Path('/data/janghj/ODE-edit/local/jlz-price-alpha-writer-2k/logging-repair-20261007')
# Collector first, then reverse topological GPU successors. 60001 absent.
TARGETS=(('60017',ALPHA,'collector'),('60007',MEMIT,'collector'),
    ('60013',ALPHA,'LLAMA_AE_FREE100'),('60012',ALPHA,'LLAMA_AE_CAP100'),
    ('60011',ALPHA,'LLAMA_AE_CAP075'),('60003',MEMIT,'LLAMA_FREE100'),
    ('60002',MEMIT,'LLAMA_CAP100'))
MEMIT_NEW=MEMIT.parent/'method-metrics-20261007'
ALPHA_NEW=ALPHA.parent/'method-metrics-20261007'

def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    require(p.returncode==0,'COMMAND_FAILED:'+shlex.join(argv)+':'+p.stderr)
    return p.stdout.strip()

def boundary():
    require(socket.gethostname()=='server4' and os.environ.get('CODEX_THREAD_ID')==SESSION,'OWNER_BOUNDARY')
    require(command(['git','-C',str(ROOT),'branch','--show-current'])=='codex/server4-price-gptj-2k','OWN_NON_MAIN_BRANCH')
    require('hyunjun1127/ODE-edit' in command(['git','-C',str(ROOT),'remote','get-url','origin']),'ORIGIN')
    envelope=json.loads((ROOT/'messages/head/2026-10-07-price-model-runs-tracking.json').read_text())
    require(envelope['instruction_id']==AUTHORITY
        and envelope['job_reconciliation']['explicit_protected_job']['job_id']=='60001','CURRENT_AUTHORITY_KEEP')
    LOCAL.mkdir(parents=True,exist_ok=True)

def detail(job):
    text=command(['scontrol','show','job',job,'--oneliner'])
    return text,{k:v for k,v in re.findall(r'(?:^| )([A-Za-z/][A-Za-z0-9/]*)=([^ ]+)',text)}

def bind(job,attempt,role,pending=True):
    require(job!='60001' or not pending,'EXPLICIT_KEEP60001')
    submission=json.loads((attempt/'submission.json').read_text())
    lock=json.loads((attempt/'execution.lock.json').read_text())
    config=json.loads((attempt/'config.json').read_text())
    require(submission['jobs'][role]==job and lock['source_commit']==submission['source']==OLD_SOURCE,'JOB_SOURCE_BINDING')
    require(sha(attempt/'config.json')==lock['config_sha256'],'OLD_CONFIG_SHA')
    launcher=next(r for r in lock['launchers'] if r['path']==str(attempt/(role+'.sh')))
    verify(launcher);verify(lock['archive'])
    text,f=detail(job)
    require(f['UserId'].startswith(getpass.getuser()+'(') and f['ReqNodeList']=='server4','EXACT_OWNER_NODE')
    require(f['JobName']==config['task_id']+'-'+role and f['Command']==launcher['path']
        and f['WorkDir']==str(attempt/'source'),'JOB_TASK_SOURCE_ARGV')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',text)
    require(submit and shlex.split(submit[1])==submission['mapping'][role]['argv'],'EXACT_REGISTERED_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==Path(launcher['path']).read_text().strip(),'ORIGINAL_SCRIPT_BYTES')
    accounting=command(['sacct','-X','-n','-P','-j',job,
        '--format=JobIDRaw,User,State,ElapsedRaw,Start,AllocTRES'])
    if pending:
        require(f['JobState']=='PENDING' and f['RunTime']=='00:00:00'
            and f['StartTime']=='Unknown' and f['AllocTRES']=='(null)'
            and f.get('NodeList','') in ('','(null)') and f['Restarts']=='0','NOT_NEVER_STARTED_PENDING')
        rows=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job]
        require(len(rows)==1 and rows[0][1]==getpass.getuser() and rows[0][2]=='PENDING'
            and rows[0][3]=='0' and rows[0][4] in ('Unknown','') and not rows[0][5],'ACCOUNTING_NEVER_STARTED')
    return dict(job=job,role=role,detail=text,accounting=accounting,
        original_config=member(attempt/'config.json'),original_lock=member(attempt/'execution.lock.json'),
        source=OLD_SOURCE,launcher=launcher)

def hold():
    boundary();path=LOCAL/'hold-receipt.json'
    if path.exists():
        previous=json.loads(path.read_text())
        require(previous['status']=='IN_PROGRESS' and not previous['held'], 'NO_DUPLICATE_HOLD_TURN')
        write(LOCAL/'hold-prewrite-parser-error.json',dict(original_receipt=previous,
            cause='Slurm empty NodeList field; no scheduler write performed',repair='accept absent empty pendingNodeList only with no AllocTRES/start/runtime'))
    receipt=dict(authority=AUTHORITY,protected_job='60001',protected_job_mutations=0,held=[],status='IN_PROGRESS')
    write(path,receipt)
    # Bind every exact target before any write. Preserve running source unchanged.
    before=[bind(j,p,r) for j,p,r in TARGETS]
    receipt['before']=before;write(LOCAL/'hold-bound-before.json',receipt)
    for job,attempt,role in TARGETS:
        prior=bind(job,attempt,role)
        command(['scontrol','hold',job])
        after=bind(job,attempt,role)
        require('Reason=JobHeldUser ' in after['detail'],'HOLD_NOT_OBSERVED')
        step=dict(before=prior,after=after)
        receipt['held'].append(step);write(LOCAL/('held-'+job+'.json'),step)
    receipt['status']='EXACT_PENDING_DOWNSTREAM_HELD';write(LOCAL/'hold-complete.json',receipt)
    return dict(status=receipt['status'],jobs=[j for j,_,_ in TARGETS],protected_job='60001')

def cancel():
    boundary();held=json.loads((LOCAL/'hold-complete.json').read_text())
    require(held['status']=='EXACT_PENDING_DOWNSTREAM_HELD','HOLD_RECEIPT_REQUIRED')
    path=LOCAL/'cancellation.json';require(not path.exists(),'NO_DUPLICATE_CANCEL')
    receipt=dict(authority=AUTHORITY,protected_job='60001',protected_job_mutations=0,cancelled=[],status='IN_PROGRESS')
    write(LOCAL/'cancel-before.json',receipt)
    for job,attempt,role in TARGETS:
        observed=bind(job,attempt,role,pending=False)
        if 'JobState=CANCELLED ' in observed['detail']:
            # Explicit resume after a metadata parser repair. Do not issue a
            # second cancellation or treat a terminal job as pending.
            prior=next(x['before'] for x in held['held'] if x['before']['job']==job)
            action='ALREADY_CANCELLED_TERMINAL_OBSERVED_NO_SECOND_WRITE'
        else:
            prior=bind(job,attempt,role)
            require('Reason=JobHeldUser ' in prior['detail'],'MUST_STILL_BE_HELD_PENDING')
            command(['scancel',job]);action='EXACT_HELD_PENDING_CANCELLED'
        after=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,User,State,ElapsedRaw,Start,AllocTRES'])
        row=next(line.split('|') for line in after.splitlines() if line.split('|')[0]==job)
        require(row[1]==getpass.getuser() and row[2].startswith('CANCELLED')
            and row[3]=='0' and row[4] in ('Unknown','None','') and not row[5],'CANCELLATION_TERMINAL_UNALLOCATED')
        step=dict(before=prior,after_accounting=after,action=action,
            accounting_start_semantics='None is terminal unstarted; scontrol cancellation timestamp is not an allocation/start of execution')
        receipt['cancelled'].append(step);write(LOCAL/('cancelled-'+job+'.json'),step)
    receipt['status']='EXACT_NEVER_STARTED_PENDING_CANCELLED';write(path,receipt)
    return dict(status=receipt['status'],jobs=[j for j,_,_ in TARGETS],protected_job='60001')

def prepare(preflight):
    boundary();review=json.loads(verify(member(preflight)).read_text())
    require(review['passed'] is True,'CPU_SOURCE_REVIEW')
    configs={};suffix='' if preflight.name=='source-cpu.json' else '-'+preflight.stem
    for old,new,selected in ((MEMIT,MEMIT_NEW,['LLAMA_CAP100','LLAMA_FREE100']),
            (ALPHA,ALPHA_NEW,['LLAMA_AE_CAP075','LLAMA_AE_CAP100','LLAMA_AE_FREE100'])):
        require(not new.exists(),'NEW_ATTEMPT_CREATE_ONCE')
        c=json.loads((old/'config.json').read_text())
        # Original scientific input/profiles/assets/W0 identity are not remapped.
        c['attempt']=str(new);c['run_instance']['attempt']=new.name
        c['selected_cells']=selected;c['execution_authority']=AUTHORITY
        c['cpu_preflight']=member(preflight)
        c['authority_members'].extend(member(ROOT/x) for x in
            ('messages/head/2026-10-07-price-model-runs-tracking.json','control/wandb-method-metric-schema.json','control/wandb-policy.json'))
        c['dependency_sources']=[member(x['path']) for x in c['dependency_sources']]
        c['tracking']['metric_schema']='price-first2k-scalar-v1'
        c['tracking']['parent_runs']={} # Do not invent UUIDs from job numbers.
        c['resources'].update(project_cap=3,effective_user_cap_override=3,
            maximum_new_GPU_concurrency=1,resource_order='retained60001 then one Llama lane',
            free_bytes=shutil.disk_usage(LOCAL).free)
        c['resource_flow']='60001 KEEP -> new MEMIT CAP100 -> FREE100 -> Alpha CAP075 -> CAP100 -> FREE100; afterany resource only'
        c['replacement_provenance']=dict(original_config=member(old/'config.json'),original_lock=member(old/'execution.lock.json'),
            protected_job='60001',old_bytes_unchanged=True,science_changed=False)
        if old==MEMIT:
            c['retained_cells']={'LLAMA_CAP075':dict(attempt=str(old),config_path=str(old/'config.json'),
                lock_path=str(old/'execution.lock.json'),job_id='60001',execution_source=OLD_SOURCE,
                config_sha256=sha(old/'config.json'),lock_sha256=sha(old/'execution.lock.json'))}
        path=LOCAL/(('memit-config' if old==MEMIT else 'alpha-config')+suffix+'.json')
        write(path,c);configs[old.name+('-memit' if old==MEMIT else '-alpha')]=member(path)
    write(LOCAL/('prepared'+suffix+'.json'),dict(configs=configs,protected_job='60001',new_job_ids=[]))
    return dict(status='SOURCE_INPUT_REBOUND_NOT_SUBMITTED',configs=configs)

def register(memit_config=None,alpha_config=None):
    boundary()
    require(json.loads((LOCAL/'cancellation.json').read_text())['status']=='EXACT_NEVER_STARTED_PENDING_CANCELLED','CANCEL_BEFORE_REPLACEMENT')
    from project.run_scripts.jlz_interference_l1 import cap_submit as memit
    from project.run_scripts.jlz_price_alpha_writer import submit as alpha
    from project.run_scripts.jlz_price_gptj.admission import width
    require(not (LOCAL/'submission.json').exists(),'NO_DUPLICATE_NEW_REGISTRATION')
    for attempt in (MEMIT_NEW,ALPHA_NEW):require(not attempt.exists(),'CREATE_ONCE_NEW_REGISTRATION')
    before=memit.resource_inventory()
    localrow=next(x for x in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
        if x.startswith('server4\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines()
        if x.startswith('server4\t')).split('\t')[1])
    cap=min(3,int(localrow[2]),tracked);require(cap>=1,'CAP_BLOCK')
    old_cancelled={j for j,_,_ in TARGETS}
    require(not old_cancelled&{j['job'] for j in before['jobs']},'OLD_CANCELLED_STILL_ADMITTED')
    existing=[j['job'] for j in before['jobs']]
    # New Llama jobs replace60001's resource lane, not an extra concurrent lane.
    parents=['60001']
    if width(before['jobs'])+(0 if '60001' in existing else 1)>cap:
        parents=list(dict.fromkeys(parents+existing))
    ids={};mapping={};held=[];frozen={}
    for name,module,attempt in (('memit',memit,MEMIT_NEW),('alpha',alpha,ALPHA_NEW)):
        config=(memit_config if name=='memit' else alpha_config) or LOCAL/(name+'-config.json')
        c=json.loads(config.read_text());roles=(*c['selected_cells'],'collector')
        lock,c=module.freeze(config,attempt,roles=roles);frozen[name]=(lock,c)
        r=c['resources']
        require(r['cpu']==8 and r['host_mib']==59392 and r['hard_host_mib']==60416
            and r['collector_cpu']==8 and r['collector_host_mib']==24576
            and r['wall']=='2-00:00:00' and r['collector_wall']=='04:00:00','RESOURCE_BINDING')
        policy=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server4',
            '--gpus','1','--mem','59392M','--local-limit-mib-per-gpu',localrow[3]])
        require('ALLOW_MEMORY_POLICY' in policy,'HARD_MEMORY_POLICY')
        own={};ownmap={}
        for role in roles:
            dependencies=(['60001']+[own[x] for x in c['selected_cells']] if name=='memit' else
                [own[x] for x in c['selected_cells']]) if role=='collector' else parents
            dep='afterany:'+':'.join(dependencies)
            argv=module.arguments(role,dep,attempt,r)
            job=command(argv).split(';')[0];require(job.isdigit() and job!='60001','ACTUAL_NEW_JOB_ID')
            own[role]=job;ids[name+'/'+role]=job
            record=dict(job=job,dependency=dep,argv=argv,dependency_role='resource-only afterany; no performance gate')
            ownmap[role]=record;mapping[name+'/'+role]=record
            write(attempt/('submitted-'+role+'.json'),dict(nonce=AUTHORITY,role=role,status='HELD',**record))
            held.append(module.inspect(job,role,dep,argv,attempt,r))
            if role!='collector':parents=[job]
        frozen[name]+=(own,ownmap)
    current=memit.resource_inventory()
    require(width(current['jobs'])<=cap,'CAP_GRAPH_NOT_PROVEN_KEEP_HELD')
    require({j['job'] for j in current['jobs']}<=set(existing)|set(ids.values()),'ADMISSION_RACE_KEEP_HELD')
    inspection=dict(jobs=held,before=before,prerelease=current,effective_cap=cap,
        combined_maximum_GPU_concurrency=width(current['jobs']),maximum_new_Llama_concurrency=1,
        protected_job='60001',protected_job_mutations=0,old_cancelled_ids_not_in_new_dependencies=True,
        all_held_inspected_before_release=True,noCP=True,other_job_mutations=0,
        node=command(['scontrol','show','node','server4']),partition=command(['scontrol','show','partition','gpu']))
    write(LOCAL/'held-inspection.json',inspection)
    for name,module,attempt in (('memit',memit,MEMIT_NEW),('alpha',alpha,ALPHA_NEW)):
        module.verify_frozen(attempt);write(attempt/'held-inspection.json',inspection)
    for job in reversed(list(ids.values())):command(['scontrol','release',job])
    snapshot=command(['squeue','-h','-j',','.join(['60001']+list(ids.values())),
        '-o','%i|%j|%T|%b|%N|%r'])
    for name,attempt in (('memit',MEMIT_NEW),('alpha',ALPHA_NEW)):
        lock,c,own,ownmap=frozen[name]
        write(attempt/'submission.json',dict(nonce=AUTHORITY,task_id=c['task_id'],status='RELEASED',jobs=own,mapping=ownmap,
            retained_jobs={'LLAMA_CAP075':'60001'} if name=='memit' else {},source=lock['source_commit'],
            lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
            bounded_initial_snapshot=snapshot,GPU_qualification='NOT_OBSERVED',W20='NOT_OBSERVED',
            monitoring_active=False,automatic_resume=False,automatic_retry=False))
    receipt=dict(nonce=AUTHORITY,status='RELEASED',jobs=ids,mapping=mapping,protected_job='60001',
        initial_snapshot=snapshot,execution_source=next(iter(frozen.values()))[0]['source_commit'],
        retained_source=OLD_SOURCE,monitoring_active=False,automatic_resume=False,automatic_retry=False)
    write(LOCAL/'submission.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['hold','cancel','prepare','register'])
    p.add_argument('--preflight',type=Path);p.add_argument('--memit-config',type=Path);p.add_argument('--alpha-config',type=Path);a=p.parse_args()
    result=prepare(a.preflight) if a.action=='prepare' else register(a.memit_config,a.alpha_config) if a.action=='register' else globals()[a.action]()
    print(json.dumps(result))
