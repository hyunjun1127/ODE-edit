"""Create-once EP controls. No old BG selector/calibration admission imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

TASK = 'ODEEDIT-S06-EP-TW1-C4-V3-OURS-FIRST-SH4-V1'
SESSION = '01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
REPO = Path('/data/janghj/ODE-edit')
OLD = REPO/'local/bg1-c4-ours-first/20260915-v1/attempt-v1'
DISPATCH = 'plans/global/2026-09-15-ep-tw1-c4-v3-dispatch-contract.json'
ENVELOPE = 'messages/head/2026-09-15-sh4-ep-tw1-c4-v3.md'
INVENTORY = 'audits/global/2026-09-15-ep-tw1-v3-sh4-dispatch-source-inventory.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def identity(path):
    p=Path(path).absolute()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return identity(p)

def git(worktree,*args):
    return subprocess.check_output(['git','-C',str(worktree),*args],text=True).strip()

def boundary(worktree):
    w=Path(worktree).resolve()
    assert socket.gethostname()=='server4'
    assert git(w,'remote','get-url','origin')=='https://github.com/hyunjun1127/ODE-edit.git'
    assert Path(git(w,'rev-parse','--path-format=absolute','--git-common-dir')).resolve()==REPO/'.git'
    registry=(w/'servers/connection-inventory.md').read_text()
    assert SESSION in registry and str(REPO) in registry
    return dict(host='server4',session=SESSION,registered_cwd=str(REPO),worktree=str(w),
        repository='hyunjun1127/ODE-edit',registry=identity(w/'servers/connection-inventory.md'),
        shared_boundary='STALE_PRESERVED; task-local exact current envelope/registry binding',
        source_head=git(w,'rev-parse','HEAD'),source_tree=git(w,'rev-parse','HEAD^{tree}'),
        remote_main=git(w,'rev-parse','origin/main'),shared_root_status=git(REPO,'status','--short','--branch'))

def verify_dispatch(path):
    d=json.loads(Path(path).read_text())
    expected=dict(instruction_id=TASK,policy='EP-TW-1',execution_server='server4',session_id=SESSION,
        new_scientific_policies=['EP-TW-1'],new_scientific_chains=1,initial_model='pre-edit W0',
        layer=4,native_l2=1,batch_size=100,batches_per_chain=10,unique_requests=1000,
        ordinals=[0,1000],new_editing_logical_batches=10,baseline_editing_reruns_allowed=False,
        N4_calibration_required=False,preservation_budget=None,log_barrier=False,
        gradient_sweeps={'current':1,'S64':1},candidate_ids=['RAW','C1','C05','C025'],
        candidate_betas=[0,1,.5,.25],fallback='actual_RAW_native_not_parent')
    for k,v in expected.items(): assert d[k]==v, ('DISPATCH_SCOPE',k)
    assert d['quality']['positive_quality_tolerance'] is None
    assert d['after_gate']['agent_state']=='WAITING_USER_RESUME' and not d['after_gate']['automatic_resume']
    return d

def initialize(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).absolute()
    b=boundary(w);a.mkdir(parents=True,mode=0o700,exist_ok=False)
    inv=json.loads((w/INVENTORY).read_text())
    expected={m['path']:m for m in inv['files']}
    rels=list(expected)+['PROTOCOL.md',DISPATCH,ENVELOPE,INVENTORY,
        'project/run_scripts/low_cost_write_donor_pilot/fitting.py',
        'project/run_scripts/low_cost_write_donor_pilot/evaluation.py',
        'project/run_scripts/low_cost_write_donor_pilot/sequential_runtime.py',
        'project/run_scripts/baseline_mechanism_first/evaluation.py',
        'project/run_scripts/baseline_mechanism_first/fixtures.py',
        'project/run_scripts/bg_tw_reference/native_map.py',
        'project/run_scripts/bg_tw_reference/teacher.py',
        'experiment-reports/servers/server4/bg1-c4-ours-first-2026-09-15-v1/g0-factual-report-ko.md']
    members=[]
    for rel in rels:
        p=w/rel;r=identity(p);r['lines']=len(p.read_bytes().splitlines())
        if rel in expected:
            assert all(r[k]==expected[rel][k] for k in ('bytes','sha256','lines')),rel
        dst=a/'authoritative'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,dst.open('xb') as out:shutil.copyfileobj(src,out)
        dst.chmod(0o400);members.append(dict(r,copy=str(dst)))
    native=REPO/'local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit'
    members += [identity(native/n) for n in ('AlphaEdit_main.py','compute_z.py')]
    members += [identity(OLD/'resume-manifest.json')]
    verify_dispatch(w/DISPATCH)
    return save(a/'full-read-receipt.json',dict(instruction_id=TASK,nonce='ODEEDIT-GH-SH4-EP-TW1-C4-V3-20260915-R1',
        time=datetime.now(timezone.utc).isoformat(),boundary=b,full_read_by_parent=True,members=members,
        policy_scope='EP-TW-1 ONLY / W0 first1000 B100x10 L4',old_BG1_immutable=True,
        N4_calibration=False,baseline_editing_reruns=0,initial_gate='NOT_RUN',gpu_cap=2,gpu_hour_cap=None,
        no_broadcast='NO_BROADCAST_NOT_REQUIRED'))

def verify_teacher(attempt):
    import numpy as np
    a=Path(attempt);p=OLD/'teacher-output-v1/teacher-manifest.json';m=json.loads(p.read_text())
    assert m['status']=='TEACHER192_READY' and len(m['cache_shards'])==24
    assert m['model_revision']=='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
    assert m['cache_dtype']==m['model_weight_dtype']=='float32' and m['vocab']==128256
    assert m['microbatch']==1 and m['shape_per_document']==[128,128256]
    assert m['edit_calls']==m['history_append']==0 and m['scored_position_count']==192*128
    assert m['lock']['sha256']=='174d3c3b738444082b59caf21d7bdd655fdc321b69c64a98d7835ce26accf968'
    verified=[]
    for ref in [m['lock'],m['reference_tokens'],m['splits'],*m['cache_shards']]:
        fp=Path(ref['path']);before=fp.stat()
        assert identity(fp)==ref,('TEACHER_MEMBER_SHA',str(fp))
        if fp.suffix=='.npy':
            x=np.load(fp,mmap_mode='r',allow_pickle=False)
            assert x.shape==(8,128,128256) and x.dtype==np.float32
            assert all(np.isfinite(x[i]).all() for i in range(8));del x
        after=fp.stat()
        assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        verified.append(dict(ref,dev=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns))
    assert sum(x['bytes'] for x in m['cache_shards'])==12608080896
    assert [d['index'] for d in m['documents']]==list(range(192))
    tokens=np.load(m['reference_tokens']['path'],allow_pickle=False)
    assert all(d['source_row_id']==str(tokens['source_row_ids'][i]) for i,d in enumerate(m['documents']))
    built=json.loads((OLD/'reference-v1/build-status.json').read_text())
    return save(a/'teacher-reuse-verification.json',dict(status='TEACHER192_FULLSHA_SCHEMA_FINITE_REUSED',
        teacher_job='47592',scheduler=dict(state='COMPLETED',exit='0:0',allocated_gpu_seconds=98,
            start='2026-09-15T03:53:21',end='2026-09-15T03:54:59',owner='janghj',
            query_scope='single bounded sacct47592; scontrol no longer retained'),
        manifest=identity(p),reference_build=identity(OLD/'reference-v1/build-status.json'),
        members=verified,teacher_tensor_bytes=m['tensor_payload_bytes'],actual_cache_bytes=m['actual_cache_bytes'],
        seconds=m['seconds'],kernel=m['kernel'],new_teacher_jobs=[],old_source_unchanged=True,
        model_selfKL_independent_forward='NOT_YET_RUN',G0_PASS=False,time=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','teacher']);p.add_argument('--worktree');p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps(initialize(x.worktree,x.attempt) if x.command=='init' else verify_teacher(x.attempt)))
