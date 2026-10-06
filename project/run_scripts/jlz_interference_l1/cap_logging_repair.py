"""Explicit one-shot USER repair of MEMIT6 and Alpha6; no automatic retries."""
import argparse,ast,copy,getpass,json,os,re,shutil,subprocess,sys
from pathlib import Path
from .cap_common import ROOT,require,write,member,sha
from . import cap_submit as memit
from project.run_scripts.jlz_price_alpha_writer import submit as alpha

ATTEMPT='logging-repair-20261007'
EVIDENCE=memit.LOCAL/'logging-repair-evidence'
MODULES=(memit,alpha)
OLD_IDS={'MEMIT':list(range(59931,59938)),'ALPHA':list(range(59949,59956))}

def terminal_previous():
    ids=[str(j) for values in OLD_IDS.values() for j in values]
    raw=memit.command(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,JobName%70,User,State,ExitCode,Elapsed,AllocTRES%80'])
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    require({r[0] for r in rows}==set(ids),'EXACT_PREVIOUS_ACCOUNTING')
    require(all(r[2]=='janghj' and r[3].split()[0] in ('FAILED','CANCELLED','COMPLETED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),'PREVIOUS_ALL_TERMINAL')
    require(not memit.command(['squeue','-h','-j',','.join(ids),'-o','%i']),'PREVIOUS_ALLOCATION_RELEASED')
    return rows

def preflight_and_config():
    require(os.environ.get('CODEX_THREAD_ID')==memit.SESSION,'SESSION')
    require(memit.command(['git','branch','--show-current'],ROOT)=='codex/server4-price-logging-repair-20261007','BRANCH')
    require(not (EVIDENCE/'repair-configs.json').exists(),'PREPARATION_CREATE_ONCE')
    previous=terminal_previous()
    checked=subprocess.run([sys.executable,'-m','unittest','project.run_scripts.jlz_interference_l1.test_cap_tracking_repair','-v'],
        cwd=ROOT,text=True,capture_output=True,timeout=60)
    require(checked.returncode==0,'RECORDED_CANDIDATE_LOGGER_REGRESSION')
    write(EVIDENCE/'logger-regression.json',dict(passed=True,tests=5,stdout=checked.stdout,stderr=checked.stderr,
        evidence='existing failed B1 c0 plus existing pre-current metric summary; caller failure injection only',
        numerical_science_tests=0,model_loads=0,forwards=0,GPU=0,previous_accounting=previous))
    configs=[];reserve=0
    for module in MODULES:
        old=module.LOCAL/'attempt';oldlock,c=module.verify_frozen(old)
        differences=[]
        for item in oldlock['source_members']:
            relative=Path(item['path']).relative_to(old/'source');new=ROOT/relative
            if new.is_file() and sha(new)!=item['sha256']:differences.append(str(relative))
        require(differences==['project/run_scripts/jlz_interference_l1/cap_tracking.py'],'ONLY_LOGGER_DIFF:'+repr(differences))
        paths=sorted((ROOT/'project/run_scripts/jlz_interference_l1').glob('cap_*.py'))
        # MEMIT archive does not include the Alpha package; its caller is statically audited separately.
        if module is alpha:paths+=sorted((ROOT/'project/run_scripts/jlz_price_alpha_writer').glob('*.py'))
        for p in paths:ast.parse(p.read_text(),filename=str(p))
        for runner in (('project.run_scripts.jlz_interference_l1.cap_run','project.run_scripts.jlz_interference_l1.cap_collect') if module is memit else ('project.run_scripts.jlz_price_alpha_writer.run','project.run_scripts.jlz_price_alpha_writer.collect')):
            p=subprocess.run([sys.executable,'-m',runner,'--help'],cwd=ROOT,text=True,capture_output=True,timeout=60)
            require(p.returncode==0,'CLI_IMPORT:'+runner)
        check=EVIDENCE/(module.TASK+'-preflight.json')
        write(check,dict(task=module.TASK,passed=True,numeric_tests=0,source=[member(p) for p in paths],
            regression=member(EVIDENCE/'logger-regression.json'),science_changes=[],changed_existing_source=differences,
            parent_source=oldlock['source_commit'],actual_B1='NOT_OBSERVED_FOR_REPAIR',model_load=False,owner_audit=True,independent_reviewer=False))
        c=copy.deepcopy(c);c['attempt']=str(module.LOCAL/ATTEMPT)
        c['run_instance']=dict(date='2026-10-07',attempt=ATTEMPT,parent_attempt=str(old),parent_source=oldlock['source_commit'],
            reason='USER-approved nested-NLL logging repair; independent cold restart, no checkpoint resume')
        c['cpu_preflight']=member(check)
        c['logging_repair']=dict(changed_existing_source=differences,science_unchanged=True,
            old_config=member(old/'config.json'),cancellation=member(EVIDENCE/'cancellation.json'))
        c['tracking']['parent_runs']={}
        for cell in module.CELLS:
            receipt=old/cell/'tracking/receipt.json'
            if receipt.exists():
                rid=json.loads(receipt.read_text()).get('run_id')
                require(isinstance(rid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',rid),'PARENT_RUN_ID')
                c['tracking']['parent_runs'][cell]=rid
        c['resources']['free_bytes']=shutil.disk_usage(module.LOCAL).free
        c['resources']['free_inodes']=os.statvfs(module.LOCAL).f_favail
        if module is alpha:
            for model,mc in c['models'].items():
                mc['predecessor_W0_roots']=[str(memit.LOCAL/ATTEMPT/(model+'_'+arm)) for arm in ('CAP075','CAP100','FREE100')]
                mc['predecessor_observation_config']=member(configs[0])
        config=EVIDENCE/(module.TASK+'-config.json');write(config,c);configs.append(config)
        reserve+=c['resources']['reserve_bytes']
    import torch
    require(not torch.cuda.is_initialized(),'NO_GPU_PREPARATION')
    require(shutil.disk_usage(memit.LOCAL).free>=reserve,'COMBINED_STORAGE_RESERVE')
    write(EVIDENCE/'repair-configs.json',dict(configs=[member(p) for p in configs],combined_reserve_bytes=reserve,
        parent_sources_preserved=True,all_previous_terminal=True,extra_fit_or_model_calls=0))
    print(json.dumps({'status':'CPU_LOGGING_REPLAY_AND_STATIC_CHECKED','configs':list(map(str,configs)),'reserve_bytes':reserve}))

def submit_all():
    terminal_previous()
    binding=json.loads((EVIDENCE/'repair-configs.json').read_text())
    require(all(not (module.LOCAL/ATTEMPT).exists() for module in MODULES),'NO_DUPLICATE_REPAIR_ATTEMPT')
    names=','.join(module.job_name(r) for module in MODULES for r in module.ROLES)
    require(not memit.command(['squeue','-h','-u',getpass.getuser(),'--name='+names,'-o','%i']),'NO_SAME_TASK_ACTIVE_JOB')
    before=memit.resource_inventory()
    local=next(x for x in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    cap=min(2,int(local[2]),tracked);require(cap>=1,'RESOURCE_CAP');parallel=cap
    require(shutil.disk_usage(memit.LOCAL).free>=binding['combined_reserve_bytes'],'COMBINED_STORAGE_RESERVE')
    node=memit.command(['scontrol','show','node','server4']);partition=memit.command(['scontrol','show','partition','gpu'])
    external=[j['job'] for j in before['jobs']]
    barrier=external if sum(j['gpus'] for j in before['jobs'])+parallel>cap else []
    all_ids=[];stages=[];frontier=[]
    for module,config in zip(MODULES,binding['configs']):
        module.verify(config);attempt=module.LOCAL/ATTEMPT
        lock,c=module.freeze(Path(config['path']),attempt);r=c['resources']
        require(r['host_mib']==59392 and r['hard_host_mib']==60416 and r['collector_host_mib']==24576
            and 1<=r['cpu']<=8 and r['task_cap']==2 and r['wall']=='2-00:00:00','RESOURCE_CONTRACT')
        memory=memit.command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server4',
            '--gpus','1','--mem','59392M','--local-limit-mib-per-gpu',local[3]])
        require('ALLOW_MEMORY_POLICY' in memory,'MEMORY_POLICY')
        dry=[]
        for role in (module.CELLS[0],module.CELLS[3],'collector'):
            argv=[v for v in module.arguments(role,None,attempt,r) if v not in ('--hold','--parsable')];argv.insert(1,'--test-only')
            p=subprocess.run(argv,text=True,capture_output=True)
            dry.append(dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,allocation_created=False))
            require(p.returncode==0,'SCHEDULER_PREFLIGHT:'+p.stderr)
        order=module.resource_order(parallel);ids={};mapping={};held=[]
        for role in module.ROLES:
            parents=[ids[x] for x in order[role]] if order[role] else list(dict.fromkeys(barrier+frontier))
            dep='afterany:'+':'.join(parents) if parents else None
            argv=module.arguments(role,dep,attempt,r);job=memit.command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
            ids[role]=job;all_ids.append(job);mapping[role]=dict(job=job,dependency=dep,argv=argv)
            write(attempt/('submitted-'+role+'.json'),dict(status='HELD',role=role,**mapping[role]))
            held.append(module.inspect(job,role,dep,argv,attempt,r))
        stages.append((module,attempt,lock,c,ids,mapping,held,dry,memory))
        frontier=[ids[module.CELLS[2]],ids[module.CELLS[5]]]
    current=memit.resource_inventory(tuple(all_ids))
    require({j['job'] for j in current['jobs']}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    require(bool(barrier) or sum(j['gpus'] for j in current['jobs'])+parallel<=cap,'CAP_RACE_KEEP_HELD')
    for module,attempt,lock,c,ids,mapping,held,dry,memory in stages:
        module.verify_frozen(attempt)
        write(attempt/'held-inspection.json',dict(jobs=held,before=before,prerelease=current,node=node,partition=partition,
            maximum_combined_GPU_concurrency=parallel,effective_server4_cap=cap,memory_policy=memory,scheduler_dry_checks=dry,
            common_start_dependency='MEMIT two tails before all Alpha; independent afterany cold arms',
            all_14_held_before_any_release=True,resource_external_barrier=barrier,automatic_retry=False))
    # Release successors/collectors first; roots last. No old job modification here.
    for module,attempt,lock,c,ids,mapping,held,dry,memory in reversed(stages):
        for role in reversed(module.ROLES):
            memit.command(['scontrol','release',ids[role]])
            write(attempt/('released-'+role+'.json'),dict(job=ids[role],command_succeeded=True))
    snapshot=memit.command(['squeue','-h','-j',','.join(all_ids),'-o','%i|%j|%T|%b|%N|%r'])
    for module,attempt,lock,c,ids,mapping,held,dry,memory in stages:
        write(attempt/'submission.json',dict(task_id=module.TASK,status='RELEASED',jobs=ids,mapping=mapping,source=lock['source_commit'],
            lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),bounded_initial_snapshot=snapshot,
            parent_submission=member(module.LOCAL/'attempt/submission.json'),actual_B1='NOT_OBSERVED',W20='NOT_OBSERVED',
            monitoring_active=False,automatic_resume=False,automatic_retry=False))
    write(EVIDENCE/'replacement-submissions.json',dict(submissions=[member(a/'submission.json') for _,a,*_ in stages],
        jobs=all_ids,initial_snapshot=snapshot,combined_cap=cap,automatic_retry=False))
    print(json.dumps({'status':'ALL_12_ARMS_RELEASED','jobs':{m.TASK:ids for m,a,l,c,ids,*_ in stages},'snapshot':snapshot}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);args=p.parse_args()
    preflight_and_config() if args.action=='prepare' else submit_all()
