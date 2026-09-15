"""Cap-only sweep: immutable single-chain locks and bounded admission.

No model forwards, scheduler polling loop, or scientific selection. Old assets
are reused by full SHA provenance plus stable stat, never silently substituted.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from .control import REPO, boundary, git, identity, save, sha
from .gate_skip import SWEEP_TASK as TASK, MODE, SKIPPED, verify_skip_members
from .gate_skip_control import copied

PREFIX = 'project/run_scripts/bg_tw_reference/ep_tw/'
ROOT = REPO/'local/ep-tw1-alpha-cap-sweep/20260915-v1'
OLD = REPO/'local/ep-tw1-c4/20260915-v1/gate-skip-r1'
DISPATCH = 'plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-dispatch'
ENVELOPE = 'messages/head/2026-09-15-sh4-ep-tw1-alpha-cap-sweep.md'
ARMS = {'CAP1': ('bounded', 1), 'CAP10': ('bounded', 10),
        'CAP100': ('bounded', 100), 'NORM_ONLY': ('disabled', None)}
NEW_ARMS = ('CAP10', 'CAP100', 'NORM_ONLY')
EXEC = '6d317bdb2660d7e9919bc3a9fb878564e9729e37'
OLD_LOCK = '5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9'
RUN_THROUGH = 'CONTINUE_OWN_SWEEP_TO_COMPLETION_AND_REPORT'


def verify_arm(d):
    from .policy import validate_dispatch, NumericalPolicy
    validate_dispatch(d)
    assert d['instruction_id'] == TASK and d['execution_authorized_by_user'] is True
    arm = d['arm']; assert arm in ARMS
    assert (d['alpha_cap_mode'], d['alpha_cap']) == ARMS[arm]
    assert d['seed'] == 20260915 and d['epsilon_num'] == 1e-12 and d['zeta'] == .25
    assert d['validation_mode'] == MODE and d['numerical_validation'] == 'NOT_ESTABLISHED'
    assert d['agent_stops_after_initial_gate'] is False
    NumericalPolicy(alpha_cap_mode=d['alpha_cap_mode'], alpha_cap=d['alpha_cap'])
    return d


def verify_arm_lock(lock):
    d = verify_arm(json.loads(Path(lock['dispatch']['path']).read_text()))
    assert identity(lock['dispatch']['path']) == lock['dispatch']
    assert lock['instruction_id'] == TASK and lock['sweep_arm'] == d['arm']
    assert d['arm'] in NEW_ARMS, 'CAP1_REUSE_DEFAULT_NO_CONDITIONAL_RERUN_JUSTIFICATION'
    for key in ('alpha_cap', 'alpha_cap_mode'):
        assert lock['numerical_policy'][key] == d[key]
    assert lock['after_gate'] == lock['after_submission'] == RUN_THROUGH
    assert lock['output'] == str(ROOT/d['arm']/'attempt-v1/scientific-v1')
    return d


def initialize(worktree):
    w = Path(worktree).resolve(); b = boundary(w)
    ROOT.mkdir(parents=True, mode=0o700, exist_ok=False)
    inputs = json.loads((w/(DISPATCH+'/dispatch-inputs.json')).read_text())
    documents = []
    for m in inputs['source_documents']:
        p = w/m['path']; r = identity(p)
        assert all(r[k] == m[k] for k in ('bytes', 'sha256')), m['path']
        copied(p, ROOT/'authoritative'/m['path'])
        documents.append(dict(r, lines=len(p.read_bytes().splitlines()),
                              reading='FULL_READ' if m['path'] in inputs['GH_full_read_primary_documents'] else 'SUPPLEMENT_IDENTITY_REUSE'))
    for rel in (ENVELOPE, DISPATCH+'.json', DISPATCH+'/dispatch-inputs.json', 'PROTOCOL.md',
                *(DISPATCH+'/arms/'+a+'.json' for a in ARMS)):
        p=w/rel; copied(p, ROOT/'authoritative'/rel)
        documents.append(dict(identity(p), lines=len(p.read_bytes().splitlines())))
    assert sha(w/'PROTOCOL.md') == 'af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b'
    assert sha(OLD/'execution.lock.json') == OLD_LOCK
    for arm in ARMS: verify_arm(json.loads((w/(DISPATCH+'/arms/'+arm+'.json')).read_text()))
    return save(ROOT/'full-read-receipt.json', dict(instruction_id=TASK,
        nonce='ODEEDIT-GH-SH4-EP-TW1-ALPHA-CAP-SWEEP-20260915-R1',
        time=datetime.now(timezone.utc).isoformat(), boundary=b, documents=documents,
        prior_same_protocol_and_native_source_full_read=identity(OLD/'full-read-receipt.json'),
        original_v3_full_read=identity(OLD.parent/'attempt-v1/full-read-receipt.json'),
        CAP1_reuse='DEFAULT; cap-only change and nonmutating metadata; CPU exact legacy arithmetic check pending',
        new_chains=3, conditional_max=4, unique_requests=1000, new_request_observations=3000,
        numerical_validation='NOT_ESTABLISHED', validation_mode=MODE,
        after_initial_gate=RUN_THROUGH, CAKE_mutation=False,
        estimated_GPU_hours=7694*3/3600, estimate_not_measurement=True,
        no_broadcast='NO_BROADCAST_NOT_REQUIRED'))


def check(worktree):
    w=Path(worktree).resolve(); boundary(w)
    files=[w/(PREFIX+x) for x in ('policy.py','runner.py','gate_skip.py','sweep_control.py','test_sweep.py')]
    for p in files: ast.parse(p.read_text(), filename=str(p))
    r=subprocess.run([sys.executable,'-B','-m','unittest',
        'project.run_scripts.bg_tw_reference.ep_tw.test_sweep',
        'project.run_scripts.bg_tw_reference.ep_tw.test_gate_skip'],cwd=w,text=True,capture_output=True)
    assert r.returncode==0,(r.stdout,r.stderr)
    subprocess.run(['bash','-n',str(w/(PREFIX+'sweep.sbatch'))],check=True)
    return save(ROOT/'minimal-checks.json',dict(status='CAP_CONFIG_LEGACY_ARITHMETIC_ROUTING_CPU_PASS',
        test_output=r.stdout+r.stderr,source_members=[identity(p) for p in files]+[identity(w/(PREFIX+'sweep.sbatch'))],
        model_forwards=0, numerical_validation='NOT_ESTABLISHED',
        CAP1_reuse='APPROVED_SCOPE_DEFAULT: legacy cap1 CPU arithmetic exact; no other scientific function changes',
        red_scope='single EP policy per arm, three fresh chains, no FD/ULP gates, fixed sample/teacher/menu/history retained'))


def freeze(worktree):
    from scripts.fixed_counterfact import load_prefix
    from .runner import validate_batch
    w=Path(worktree).resolve(); b=boundary(w)
    assert not git(w,'status','--porcelain','--',PREFIX),'UNCOMMITTED_RUNTIME'
    checks=json.loads((ROOT/'minimal-checks.json').read_text())
    assert all(identity(m['path'])==m for m in checks['source_members'])
    assert sha(OLD/'execution.lock.json')==OLD_LOCK
    old=json.loads((OLD/'execution.lock.json').read_text())
    records=load_prefix(old['dataset_root'],1000)
    for batch,item in enumerate(old['batches'],1): validate_batch(records,batch,item)
    source=ROOT/'source-v1'; source.mkdir(mode=0o700,exist_ok=False)
    oldsource=Path(old['source_root']); stats={m['path']:m for m in old['member_stats']}
    external=[]; rels=set(); copied_lineage=[]
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldsource):
            rel=str(p.relative_to(oldsource)); rels.add(rel)
        else:
            s=p.stat(); now=dict(path=str(p),dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size)
            assert now==stats[str(p)] and s.st_size==m['bytes'],('ASSET_STAT_DRIFT',str(p))
            if s.st_size<8*(1<<20): assert sha(p)==m['sha256']
            external.append(dict(m))
    rels.update(str(p.relative_to(w)) for p in (w/PREFIX).glob('*.py'))
    rels.update((PREFIX+'sweep.sbatch',ENVELOPE,DISPATCH+'.json',DISPATCH+'/dispatch-inputs.json'))
    rels.update(DISPATCH+'/arms/'+arm+'.json' for arm in ARMS)
    documents=json.loads((w/(DISPATCH+'/dispatch-inputs.json')).read_text())['source_documents']
    rels.update(m['path'] for m in documents)
    members=list(external)
    for rel in sorted(rels):
        # Reuse exact old non-EP closure; current owned source only for EP.
        p=w/rel if rel.startswith(PREFIX) or not (oldsource/rel).exists() else oldsource/rel
        copied(p,source/rel); members.append(identity(source/rel))
        copied_lineage.append(dict(relative=rel,source=str(p),sha256=sha(p)))
    for name in ('ledger.py','model_adapter.py'):
        assert sha(source/(PREFIX+name))==sha(oldsource/(PREFIX+name)),('READONLY_SCIENCE',name)
    archive=ROOT/'source-v1.tar'
    with archive.open('xb') as out,tarfile.open(fileobj=out,mode='w') as tar:
        for rel in sorted(rels):
            info=tar.gettarinfo(str(source/rel),arcname=rel)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as inp:tar.addfile(info,inp)
    for p in (ROOT/'full-read-receipt.json',ROOT/'minimal-checks.json'):
        members.append(identity(p))
    member_stats=[]
    for m in members:
        s=Path(m['path']).stat();member_stats.append(dict(path=m['path'],dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size))
    diff=git(w,'diff',EXEC,'HEAD','--',PREFIX)
    save(ROOT/'source-lineage.json',dict(execution_HEAD=b['source_head'],execution_tree=b['source_tree'],
        previous_execution=EXEC,source_files=copied_lineage,actual_source_diff=diff,
        immutable_helper_imports='original source bytes retained, not current-main mutable imports',
        CAP1='REUSE_47962; bounded cap1 exact fixture; only cap mode and nonmutating scalar/control additions'))
    locks=[]
    for arm in NEW_ARMS:
        a=ROOT/arm/'attempt-v1'; a.mkdir(parents=True,mode=0o700,exist_ok=False)
        d=verify_arm(json.loads((source/(DISPATCH+'/arms/'+arm+'.json')).read_text()))
        lock=copy.deepcopy(old)
        lock.update(instruction_id=TASK,sweep_arm=arm,source_head=b['source_head'],source_tree=b['source_tree'],
            source_root=str(source),source_archive=identity(archive),members=members,member_stats=member_stats,
            dispatch=identity(source/(DISPATCH+'/arms/'+arm+'.json')),output=str(a/'scientific-v1'),
            after_gate=RUN_THROUGH,after_submission=RUN_THROUGH,prior_CAP1_lock=identity(OLD/'execution.lock.json'))
        lock['validation_override']['inherited_by']=TASK
        lock['validation_override']['sweep_envelope']=identity(source/ENVELOPE)
        lock['numerical_policy'].update(alpha_cap_mode=d['alpha_cap_mode'],alpha_cap=d['alpha_cap'])
        lock['cost_plan'].update(new_estimate_gpu_hours=7694/3600,estimate_basis='same-host CAP1 7694 GPU-sec linear estimate; not measurement/budget',
            total_new_disk_reserve_bytes=20*(1<<30),expected_prior_run_total_bytes=13407776774,
            original_model_checkpoint_schedule='10 actual W4/M4/RNG/ledger snapshots including retained W10')
        verify_arm_lock(lock)
        ref=save(a/'execution.lock.json',lock)
        verification=verify_skip_members(lock)
        save(a/'input-verification.json',dict(verification,execution_lock=ref,arm=arm,
            sample_prefix=1000,batches=10,cold_W0_M0=True,teacher_reused='47592_COMPLETE',
            final_weight_retention_required=True))
        locks.append(ref)
    return save(ROOT/'frozen-arms.json',dict(status='SOURCE_INPUT_FROZEN_NOT_SUBMITTED',locks=locks,
        source_archive=identity(archive),source_head=b['source_head'],source_tree=b['source_tree']))


def submit(worktree):
    from .operations import command,count_reservations
    w=Path(worktree).resolve();boundary(w)
    ctl=ROOT/'submission-v1';ctl.mkdir(mode=0o700,exist_ok=False)
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING','-o','%i|%j|%T|%b|%R'])
    reservations=count_reservations(queue,lambda jid:command(['scontrol','show','job',jid,'-o']))
    admitted=sum(r['gpus'] for r in reservations)
    # The task explicitly allows one sequential reservation for multiple jobs.
    # Dependency successors cannot allocate while their predecessor is live.
    assert admitted<2,('CAPACITY_HOLD_NO_FREE_PROJECT_LANE',reservations)
    lanes=2-admitted
    source=ROOT/'source-v1';shell=source/(PREFIX+'sweep.sbatch')
    memory=command([sys.executable,str(w/'scripts/slurm_memory_policy.py'),'audit',str(shell)])
    env=dict(os.environ,AGENT_GPU_CAPS_FILE=str(REPO/'servers/local/gpu-caps.tsv'))
    cap=command(['bash',str(w/'scripts/check-slurm-resource-cap.sh'),'server4','1','60416M'],env=env)
    disk=shutil.disk_usage(ROOT);fs=os.statvfs(ROOT)
    assert disk.free>3*20*(1<<30)+20*(1<<30) and fs.f_favail>5000,'DISK_HOLD'
    resource=save(ctl/'resource-preflight.json',dict(time=datetime.now(timezone.utc).isoformat(),
        reservations=reservations,raw_queue=queue,existing_reserved=admitted,sweep_lanes=lanes,
        maximum_aggregate_GPUs=admitted+lanes,cap_check=cap,memory_audit=memory,
        disk_free=disk.free,inodes_free=fs.f_favail,sweep_disk_reserve=60*(1<<30),safety_free=20*(1<<30),
        reserve_is_not_exclusive=True,estimated_GPU_hours=7694*3/3600,
        CAKE_scope='resource-only admission; no scientific output or mutation',
        node=command(['scontrol','show','node','server4','-o'])))
    held_jobs=[]
    names={'CAP10':'odeedit_ep_tw1_cap10_s4','CAP100':'odeedit_ep_tw1_cap100_s4','NORM_ONLY':'odeedit_ep_tw1_normonly_s4'}
    for index,arm in enumerate(NEW_ARMS):
        a=ROOT/arm/'attempt-v1';lp=a/'execution.lock.json';lock=json.loads(lp.read_text())
        verify_arm_lock(lock);verify_skip_members(lock)
        assert not Path(lock['output']).exists(),'OUTPUT_EXISTS_NO_DUPLICATE'
        dependency=held_jobs[index-lanes]['job_id'] if index>=lanes else None
        args=['sbatch','--parsable','--hold','--no-requeue','--job-name='+names[arm],
            f'--output={ctl}/{arm}-%j.out',f'--error={ctl}/{arm}-%j.err']
        if dependency:args+=['--dependency=afterany:'+dependency]
        args += [str(shell),str(source),str(lp),lock['output']]
        job=command(args).split(';')[0];assert job.isdigit()
        save(ctl/(arm+'-held.json'),dict(job_id=job,arm=arm,args=args,lock=identity(lp),resource=resource))
        command(['scontrol','update','JobId='+job,'Requeue=0'])
        record=command(['scontrol','show','job',job,'-o'])
        for field in ('JobName='+names[arm],'UserId=janghj(','JobState=PENDING','ReqNodeList=server4',
                      'MinMemoryNode=59G','NumCPUs=8','TimeLimit=12:00:00','Requeue=0','gpu:rtx_pro_6000:1',str(shell)):
            assert field in record,('HELD_INSPECTION',arm,field)
        assert ('Dependency=afterany:'+dependency) in record if dependency else 'Dependency=(null)' in record
        held_jobs.append(dict(arm=arm,job_id=job,dependency=dependency,held_record=record,lock=identity(lp),output=lock['output']))
    save(ctl/'all-held-inspected.json',dict(jobs=held_jobs,aggregate_lanes=lanes,existing=admitted))
    # All arms are upfront and frozen; release even if any predecessor will fail.
    for j in held_jobs:command(['scontrol','release',j['job_id']])
    for j in held_jobs:j['last_record']=command(['scontrol','show','job',j['job_id'],'-o'])
    return save(ctl/'release-receipt.json',dict(status='THREE_ARMS_SUBMITTED_NOT_COMPLETED',jobs=held_jobs,
        last_observation=datetime.now(timezone.utc).isoformat(),source=identity(ROOT/'frozen-arms.json'),
        maximum_sweep_concurrent=lanes,aggregate_cap=2,after_initial=RUN_THROUGH,
        quality_selection_of_arms=False,callback=False,numerical_validation='NOT_ESTABLISHED'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','check','freeze','submit']);p.add_argument('--worktree',required=True)
    x=p.parse_args();print(json.dumps({'init':initialize,'check':check,'freeze':freeze,'submit':submit}[x.command](x.worktree)))
