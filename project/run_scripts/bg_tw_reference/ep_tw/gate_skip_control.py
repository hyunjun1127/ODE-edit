"""Create-once direct-science lock, reusing sealed assets; no numerical tests."""
import argparse
import ast
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from .control import REPO,DISPATCH,boundary,git,identity,save,sha,verify_dispatch
from .gate_skip import TASK,MODE,SKIPPED,verify_skip_members

OLD=REPO/'local/ep-tw1-c4/20260915-v1/attempt-v1'
REPAIR=OLD.parent/'repair-r1'
ENVELOPE='messages/head/2026-09-15-sh4-ep-tw1-gate-skip-run.md'
WAIVER='audits/global/2026-09-15-ep-tw1-numerical-gate-user-waiver.md'
PREFIX='project/run_scripts/bg_tw_reference/ep_tw/'

def copied(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as inp,dst.open('xb') as out:shutil.copyfileobj(inp,out)
    dst.chmod(0o400)

def initialize(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve();b=boundary(w)
    assert a==REPO/'local/ep-tw1-c4/20260915-v1/gate-skip-r1'
    a.mkdir(mode=0o700,exist_ok=False)
    prior=json.loads((REPAIR/'full-read-receipt.json').read_text())
    protocol=next(m for m in prior['documents'] if m['path'].endswith('/PROTOCOL.md'))
    assert sha(w/'PROTOCOL.md')==protocol['sha256']
    docs=[]
    for rel in (ENVELOPE,WAIVER,'PROTOCOL.md',DISPATCH):
        p=w/rel;dst=a/'authoritative'/rel;copied(p,dst)
        docs.append(dict(identity(p),lines=len(p.read_bytes().splitlines()),sealed_copy=str(dst)))
    return save(a/'full-read-receipt.json',dict(instruction_id=TASK,
        nonce='ODEEDIT-GH-SH4-EP-TW1-GATES-SKIP-RUN-20260915-R1',time=datetime.now(timezone.utc).isoformat(),
        boundary=b,documents=docs,latest_envelope_and_waiver_full_read=True,
        identical_protocol_and_v3_source_full_read_reused=identity(REPAIR/'full-read-receipt.json'),
        prior_original_full_read=identity(OLD/'full-read-receipt.json'),
        diagnostic_status=MODE,numerical_validation='NOT_ESTABLISHED',
        prior_failed_jobs={'47884':473,'47942':69},teacher_reused_allocated_seconds=98,
        new_diagnostic_jobs=0,new_scientific_chains=1,new_scientific_targets=1000,
        original_source_outputs_preserved=True,no_broadcast='NO_BROADCAST_NOT_REQUIRED'))

def freeze(worktree,attempt):
    from scripts.fixed_counterfact import load_prefix
    from .runner import validate_batch
    w,a=Path(worktree).resolve(),Path(attempt).resolve();b=boundary(w)
    assert not git(w,'status','--porcelain','--',PREFIX),'UNCOMMITTED_RUNTIME'
    oldref=identity(OLD/'execution.lock.json')
    assert oldref['sha256']=='306ad09b56d470ce47562bd3f36230d7f96fe258769dc0fb5b3dc192c33d954b'
    old=json.loads(Path(oldref['path']).read_text());verify_dispatch(w/DISPATCH)
    prior=json.loads((REPAIR/'repair.lock.json').read_text())
    assert sha(REPAIR/'repair.lock.json')=='e6501577e55aa509765d454d3c605f12a7d710f1a07cf83bbd18aecc97c2f43c'
    priorstats={x['path']:x for x in prior['member_stats']}
    rows=load_prefix(old['dataset_root'],1000)
    for batch,item in enumerate(old['batches'],1):validate_batch(rows,batch,item)
    source=a/'source-v1';source.mkdir(mode=0o700,exist_ok=False)
    oldsource=Path(old['source_root']);rels=[];members=[];reused=0
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldsource):
            rel=str(p.relative_to(oldsource));rels.append(rel)
            # Non-EP imported bytes are unchanged, not silently upgraded.
            if not rel.startswith(PREFIX):assert sha(w/rel)==m['sha256'],('READONLY_DEPENDENCY_DRIFT',rel)
        else:
            s=p.stat();st=dict(path=str(p),dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size)
            assert st==priorstats[str(p)] and s.st_size==m['bytes'],('REUSED_ASSET_STAT_DRIFT',str(p))
            if s.st_size<8*(1<<20):assert sha(p)==m['sha256']
            else:reused+=1
            members.append(dict(m))
    rels += [str(p.relative_to(w)) for p in (w/PREFIX).glob('*.py')]
    rels += [ENVELOPE,WAIVER,PREFIX+'nogate.sbatch']
    for rel in sorted(set(rels)):
        dst=source/rel;copied(w/rel,dst);members.append(identity(dst))
    for p in (a/'full-read-receipt.json',a/'minimal-checks.json'):
        members.append(identity(p))
    checks=json.loads((a/'minimal-checks.json').read_text())
    assert checks['status']=='MINIMAL_ROUTING_SYNTAX_PASS_NOT_NUMERICAL_VALIDATION'
    assert all(identity(m['path'])==m for m in checks['source_members'])
    # policy/ledger and native helper bytes are independent invariants.
    for name in ('policy.py','ledger.py'):
        assert sha(w/(PREFIX+name))==sha(oldsource/(PREFIX+name)),name
    archive=a/'source-v1.tar'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w') as tar:
        for rel in sorted(set(rels)):
            info=tar.gettarinfo(str(source/rel),arcname=rel)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as inp:tar.addfile(info,inp)
    stats=[]
    for m in members:
        s=Path(m['path']).stat()
        stats.append(dict(path=m['path'],dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size))
    lock=copy.deepcopy(old)
    lock.pop('technical_numerics')
    lock.update(instruction_id=TASK,validation_mode=MODE,numerical_validation='NOT_ESTABLISHED',
        validation_override=dict(instruction_id=TASK,envelope=identity(source/ENVELOPE),waiver=identity(source/WAIVER),
            skipped=SKIPPED,diagnostic_callbacks_executed=0),
        source_head=b['source_head'],source_tree=b['source_tree'],source_root=str(source),
        source_archive=identity(archive),members=members,member_stats=stats,dispatch=identity(source/DISPATCH),
        output=str(a/'scientific-v1'),prior_failed_lock=oldref,
        asset_reuse=dict(prior_full_sha_with_stable_stat=reused,prior_stat_lock=identity(REPAIR/'repair.lock.json'),
                        teacher_evidence=identity(OLD/'teacher-reuse-verification.json'),new_teacher_generation=False),
        after_submission='WAITING_USER_RESUME',after_gate='WAITING_USER_RESUME',
        resource=dict(gpu=1,cpus=8,mem='60416M',wall='12:00:00',gpu_cap=2,gpu_hour_cap=None,export='NONE',requeue=0))
    lock['cost_plan'].update(old_failed_gpu_seconds={'47884':473,'47942':69},
        old_native_fit_nested_seconds=286.54453050531447,new_diagnostic_jobs=0,
        estimate_not_measurement=True,diagnostics_removed_not_counted_as_pass=True)
    ref=save(a/'execution.lock.json',lock)
    save(a/'input-verification.json',dict(verify_skip_members(lock),sample_prefix=1000,
        batches=10,cold_W0_M0=True,policy_ledger_bytes_unchanged=True,execution_lock=ref))
    return ref

def check(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve();boundary(w)
    paths=sorted((w/PREFIX).glob('*.py'))
    for p in paths:ast.parse(p.read_text(),filename=str(p))
    test=subprocess.run([sys.executable,'-B','-m','unittest',
        'project.run_scripts.bg_tw_reference.ep_tw.test_gate_skip'],cwd=w,text=True,capture_output=True)
    assert test.returncode==0,(test.stdout,test.stderr)
    shell=subprocess.run(['bash','-n',str(w/(PREFIX+'nogate.sbatch'))],text=True,capture_output=True)
    assert shell.returncode==0,shell.stderr
    return save(a/'minimal-checks.json',dict(status='MINIMAL_ROUTING_SYNTAX_PASS_NOT_NUMERICAL_VALIDATION',
        routing_mocks=5,test_output=test.stdout+test.stderr,python_AST_files=len(paths),bash_syntax=True,
        model_import_loads=0,GPU_jobs=0,numerical_tests=0,numerical_validation='NOT_ESTABLISHED',
        source_members=[identity(p) for p in paths]+[identity(w/(PREFIX+'nogate.sbatch'))]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','freeze','check'])
    p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps({'init':initialize,'freeze':freeze,'check':check}[x.command](x.worktree,x.attempt)))
