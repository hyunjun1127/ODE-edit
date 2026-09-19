"""Create-once frozen source/locks and read-only verification; no auto-submit."""
import argparse
import io
import os
import socket
import subprocess
import tarfile
from pathlib import Path
from .common import *


def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    tree=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD^{tree}'],text=True).strip()
    paths=['project/run_scripts/checkpoint_mechanism_audit',str(CONTRACT_PATH.relative_to(REPO))]
    data=subprocess.check_output(['git','-C',str(REPO),'archive',head,*paths])
    with (output/'source.tar').open('xb') as f:f.write(data)
    source=output/'source';source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            p=Path(member.name)
            assert not p.is_absolute() and '..' not in p.parts and (member.isfile() or member.isdir())
        tar.extractall(source,filter='data')
    members=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(source.rglob('*')) if p.is_file()]
    receipt=dict(source_head=head,source_tree=tree,source=str(source),members=members,
        source_archive_sha256=sha256(output/'source.tar'),
        inputs_source_map_sha256=sha256(ATTEMPT/'inputs/source-map.json'),
        source_ready_sha256=sha256(ATTEMPT/'inputs/source-ready.json'),
        contract_sha256=sha256(CONTRACT_PATH),hostname=socket.gethostname(),
        session='01a0493a-074c-7f91-9a13-769116326fef',repository='hyunjun1127/ODE-edit',
        boundary_helper=dict(canonical_root_check='PASS',new_worktree_check='MISSING_LOCAL_CONFIG',
                             authority='explicit dedicated-worktree envelope + frozen member verification; no helper relaxation'),
        expected_output=str(ATTEMPT/'results/model-gate-r1'),
        resource=dict(project_cap=2,task_cap=2,gpu_per_job=1,cpu=8,host_mem_mib=60416,
                      walltime_hours=8,export='NONE',requeue=False,
                      estimate='Gate-only conservative reservation, not measured runtime; later costs based on gate',
                      disk_available=os.statvfs(ATTEMPT).f_bavail*os.statvfs(ATTEMPT).f_frsize),
        save_checkpoints=False,editing_chain=0,z_optimization=0,history_append=0,
        global_slurm_audit='6 historical server4 violations, not modified; new launcher scoped audit required',
        source_scope='checkpoint_mechanism_audit only + immutable contract',scientific_promotion=False)
    write_json(output/'execution.lock.json',receipt)
    print(json.dumps(dict(head=head,tree=tree,source=str(source),lock_sha256=sha256(output/'execution.lock.json'))),flush=True)


def verify(lock):
    x=read(lock)
    assert Path.cwd().resolve()==Path(x['source']).resolve()
    for member in x['members']:
        assert Path(member['path']).stat().st_size==member['bytes'] and sha256(member['path'])==member['sha256']
    assert sha256(ATTEMPT/'inputs/source-map.json')==x['inputs_source_map_sha256']
    assert sha256(ATTEMPT/'inputs/source-ready.json')==x['source_ready_sha256']
    assert sha256(CONTRACT_PATH)==x['contract_sha256']
    assert x['resource']['host_mem_mib']<=60416 and x['resource']['project_cap']==2
    print('EXECUTION_LOCK_PASS',x['source_head'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','verify']);p.add_argument('--output');p.add_argument('--lock')
    a=p.parse_args();freeze(a.output) if a.mode=='freeze' else verify(a.lock)
