"""One USER-recalled registration pass; frozen scientific bytes stay unchanged."""
import argparse
import copy
import getpass
import json
import os
import shutil
import subprocess
import tarfile

from .common import *
from .submit import ROLES, arguments, inspect, launcher, order, command, dependencies, width
from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory

RECALL='USER-GH-SH1-SH2-SH4-BASELINE-GENERATION-REGISTER-RESUME-20261007-R1'
AUTHORITY='6d6e2fdb531ecb3d6e5bc98db0f188001c6a7aa0'
ENVELOPE_RECALL='messages/head/2026-10-07-baseline-generation-register-resume.json'
RECALL_SHA='43099367d76e1f8a2b57b539c4a18aa17351e2194885db70e3516af19464fc16'
SCIENCE='6bc51602632b5a2dfb4c832479002b30b604b8eb'
OLD_LOCK_SHA='dd1690924118251ab302aeb885ffdd396976b6b863a731b7f54b120e5b135f35'
OLD_CONFIG_SHA='713e164ce793ea7231d3b08b8bfab1a55adc08d74a8de62c290e6bd999f583aa'
OLD_ARCHIVE_SHA='e502cb6a9e2e259aa074944dc834047a49a730750b7f404e90553ad5190f3fa9'
NEW=LOCAL/'attempt-register-r1'


def config_diff(old,new):
    allowed={'attempt','run_instance','registration_recall'}
    changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    require(changed<=allowed and changed==allowed,'REGISTRATION_ONLY_CONFIG_DIFF')
    return sorted(changed)


def deduplicate():
    require(not any(LOCAL.glob('*/submitted-*.json')) and not any(LOCAL.glob('*/submission.json')),
            'EXISTING_ACTUAL_SUBMISSION_KEEP_NO_DUPLICATE')
    names=','.join(TASK+'-'+r for r in ROLES)
    queue=command(['squeue','-h','-u',getpass.getuser(),'--name='+names,'-o','%i|%j|%T|%R'])
    env=os.environ.copy();env['TMPDIR']=str(LOCAL/'cpu-tmp')
    result=subprocess.run(['sacct','-X','-S','2026-10-07T00:00:00','-u',getpass.getuser(),
        '-n','-P','--name='+names,'--format=JobIDRaw,JobName%90,State,Submit'],
        text=True,capture_output=True,env=env)
    require(result.returncode==0,'DEDUP_ACCOUNTING_UNAVAILABLE')
    require(not queue and not result.stdout.strip(),'EXISTING_MATCHING_REGISTERED_KEEP_NO_DUPLICATE')
    return dict(queue=queue,accounting=result.stdout.strip(),checked_names=names.split(','),
                current_user=getpass.getuser(),no_matching_registration=True)


def prepare():
    authority();require(not NEW.exists(),'CREATE_ONCE_RECALLED_ATTEMPT')
    require(not command(['git','status','--porcelain','--',str(Path(__file__).relative_to(ROOT))],ROOT),
            'COMMIT_REGISTRATION_SOURCE_FIRST')
    old=LOCAL/'attempt-v1'
    for path,expected in [('execution.lock.json',OLD_LOCK_SHA),('config.json',OLD_CONFIG_SHA),
                          ('source.tar',OLD_ARCHIVE_SHA)]:
        require(sha(old/path)==expected,'OLD_ATTEMPT_IDENTITY:'+path)
    prior=read(old/'execution.lock.json');original=read(old/'config.json')
    require(prior['source_commit']==SCIENCE,'FROZEN_SCIENTIFIC_COMMIT')
    for field in ('source_members','runtime_sources','launchers','native_closure','source_config_members'):
        for item in prior[field]:verify(item)
    for key in ('archive','tracking_env','generation_reference'):verify(prior[key])
    tests=read(verify(original['cpu_preflight']));require(tests['status']=='PASS','REUSED_CPU_PREFLIGHT')
    for item in original['generation']['source_members']:verify(item)
    shared=Path('/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1')
    require(sha(shared/'manifest.json')=='6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8'
        and sha(shared/'READY.json')=='634ca5c70f54f7af5346849b3594ed92280cfbc3853d0def16a7eef86ae64203',
        'SHARED_READY_EXACT')
    require(not Path(original['generation']['shared_W0_root']).exists(),'FRESH_GENERATION_W0_NOT_PREMEASURED')
    receipt=deduplicate()
    envelope=subprocess.check_output(['git','show',AUTHORITY+':'+ENVELOPE_RECALL],cwd=ROOT)
    import hashlib
    require(hashlib.sha256(envelope).hexdigest()==RECALL_SHA,'RECALL_ENVELOPE_SHA')
    e=json.loads(envelope);target=e['targets']['server1']
    require(e['instruction_id']==RECALL and target['session']==SESSION and target['source']==SCIENCE
        and target['project_GPU_cap']==2 and target['task_id']==TASK,'RECALL_OWNER_SCOPE')
    NEW.mkdir();source=NEW/'source';source.mkdir()
    write_bytes(NEW/'recall-envelope.json',envelope)
    shutil.copyfile(old/'source.tar',NEW/'source.tar')
    require(sha(NEW/'source.tar')==OLD_ARCHIVE_SHA,'SCIENCE_ARCHIVE_BYTE_IDENTICAL')
    with tarfile.open(NEW/'source.tar') as tf:
        items=tf.getmembers()
        require(len(items)==len({x.name for x in items}) and all((x.isfile() or x.isdir())
            and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in items),
            'SAFE_UNCHANGED_ARCHIVE')
        tf.extractall(source,filter='data')
    members=[]
    for item in prior['source_members']:
        bound=member(source/Path(item['path']).relative_to(old/'source'))
        require(bound['sha256']==item['sha256'] and bound['bytes']==item['bytes'],
                'EXACT_FROZEN_SOURCE_MEMBER')
        members.append(bound)
    c=copy.deepcopy(original)
    c['attempt']=str(NEW);c['run_instance']={'attempt':NEW.name}
    c['registration_recall']=dict(instruction_id=RECALL,authority=AUTHORITY,
        envelope=member(NEW/'recall-envelope.json'),old_lock=member(old/'execution.lock.json'),
        scientific_source_unchanged=True,automatic_retry=False)
    changed=config_diff(original,c);write(NEW/'config.json',c)
    for role in ROLES:
        (NEW/'tmp'/role).mkdir(parents=True,mode=0o700)
        script=NEW/(role+'.sh');write_bytes(script,launcher(source,SCIENCE,role,NEW).encode());script.chmod(0o755)
    lock=copy.deepcopy(prior)
    lock.update(config_sha256=sha(NEW/'config.json'),archive=member(NEW/'source.tar'),source_members=members,
        launchers=[member(NEW/(r+'.sh')) for r in ROLES],registration_recall=c['registration_recall'],
        registration_control_commit=command(['git','rev-parse','HEAD'],ROOT),
        registration_control_members=[member(Path(__file__)),member(Path(__file__).with_name('test_resume_registration.py'))])
    write(NEW/'execution.lock.json',lock)
    proof=dict(instruction_id=RECALL,science_source=SCIENCE,old_attempt_preserved=True,
        old_lock=member(old/'execution.lock.json'),new_lock=member(NEW/'execution.lock.json'),
        archive_identical=True,source_members_exact=len(members),config_changed_keys=changed,
        reused_CPU=original['cpu_preflight'],shared_READY=member(shared/'READY.json'),dedup=receipt,
        GPU_model_fit_or_online_tests=0,new_identity_on_actual_start=True)
    write(NEW/'registration-preparation.json',proof)
    return proof


def submit():
    require(NEW.is_dir() and not (NEW/'registration-pass-started.json').exists(),'ONE_RECALLED_PASS_ONLY')
    c=read(NEW/'config.json');lock=read(NEW/'execution.lock.json')
    require(c['instruction_id']==NONCE and lock['source_commit']==SCIENCE
        and sha(NEW/'config.json')==lock['config_sha256'],'RECALLED_FREEZE_IDENTITY')
    config_diff(read(LOCAL/'attempt-v1/config.json'),c);dup=deduplicate()
    before=resource_inventory();existing=before['jobs']
    row=next(x for x in (ROOT/'servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')[1])
    cap=min(2,int(row[2]),tracked);require(cap>=1 and width(existing)<=cap,'CURRENT_COMBINED_CAP')
    frontier=dependencies(existing);dag=order(cap);virtual=list(existing);fake={}
    for i,role in enumerate(ARMS):
        fake[role]=str(999999990+i);dep=frontier+[fake[x] for x in dag[role]]
        virtual.append(dict(job=fake[role],gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(dep) if dep else '(null)')+' '))
    projected=width(virtual);require(projected<=cap,'PROJECTED_COMBINED_CAP')
    r=c['resources'];node=command(['scontrol','show','node','devbox']);part=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s1','format=Name,MaxWall,MaxTRESPerJob,MaxTRESPerUser'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in part and qos,'CURRENT_CANONICAL_RESOURCES')
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1',
        '--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',row[3]])
    root=shutil.disk_usage('/');raid=shutil.disk_usage(NEW);fs=os.statvfs(NEW)
    require('ALLOW_MEMORY_POLICY' in memory and raid.free>=r['reserve_bytes']
        and root.free>0 and fs.f_favail>32,'FRESH_MEMORY_STORAGE_INODES')
    write(NEW/'resource-preflight.json',dict(before=before,node=node,partition=part,qos=qos,
        effective_cap=cap,projected_GPU_width=projected,frontier=frontier,memory_policy=memory,
        root_available_bytes=root.free,RAID_available_bytes=raid.free,available_inodes=fs.f_favail,
        canonical_cap_policy=member(ROOT/'control/gpu-concurrency-policy.tsv'),duplicate_check=dup,
        old_jobs_mutated=False))
    for field in ('source_members','launchers','runtime_sources','native_closure','source_config_members','registration_control_members'):
        for item in lock[field]:verify(item)
    for key in ('archive','tracking_env','generation_reference'):verify(lock[key])
    verify(c['registration_recall']['envelope'])
    write(NEW/'registration-pass-started.json',dict(instruction_id=RECALL,automatic_retry=False,one_deliberate_pass=True))
    ids={};held=[];role=None
    try:
        for role in ROLES:
            dep=list(ids.values()) if role=='collector' else list(dict.fromkeys(frontier+[ids[x] for x in dag[role]]))
            argv=arguments(role,dep,NEW,r)
            result=subprocess.run(argv,text=True,capture_output=True)
            write(NEW/('sbatch-'+role+'.json'),dict(role=role,argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            require(result.returncode==0,'SBATCH_REGISTRATION_FAILED:'+result.stderr.strip())
            job=result.stdout.strip().split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID')
            ids[role]=job;write(NEW/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dep,argv=argv))
            held.append(inspect(job,role,dep,argv,NEW,r))
        write(NEW/'held-inspection.json',dict(source_commit=SCIENCE,config_sha256=lock['config_sha256'],jobs=held,passed=True))
        fresh=resource_inventory(exclude=tuple(ids.values()))
        total=width(fresh['jobs']+[dict(job=x['job'],gpus=1,resource_detail=x['resource_detail']) for x in held if x['role']!='collector'])
        require(total<=cap,'PRE_RELEASE_COMBINED_CAP')
        write(NEW/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=total,effective_cap=cap,passed=True))
        for item in lock['source_members']+lock['launchers']+lock['registration_control_members']:verify(item)
        release=[dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])) for role in reversed(ROLES)]
        write(NEW/'release.json',dict(jobs=release))
        snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
        receipt=dict(instruction_id=NONCE,registration_recall=RECALL,task_id=TASK,jobs=ids,
            source_commit=SCIENCE,source_tree=lock['source_tree'],registration_control_commit=lock['registration_control_commit'],
            lock=member(NEW/'execution.lock.json'),config=member(NEW/'config.json'),held=member(NEW/'held-inspection.json'),
            release=member(NEW/'release.json'),dependencies={x['role']:x['dependency'] for x in held},snapshot=snapshot,
            initial='NOT_OBSERVED',W_B='NOT_YET_RUN; startup/readback inside actual jobs',checkpoint_saved=False,
            exact_resume='NOT_AVAILABLE',combined_GPU_cap=cap,projected_GPU_width=total,
            broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False,automatic_retry=False)
        write(NEW/'submission.json',receipt);return receipt
    except Exception as error:
        write(NEW/'registration-failure.json',dict(instruction_id=RECALL,role=role,jobs_returned=ids,
            error=str(error),automatic_retry=False,existing_jobs_mutated=False,release_complete=(NEW/'release.json').exists()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','submit'));args=p.parse_args()
    print(json.dumps(prepare() if args.phase=='prepare' else submit(),ensure_ascii=False))
