"""Immutable task entrypoints, driven only by pre-registered Slurm dependencies."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
from .common import save,file_sha

PHASES=('gate','geometry','writers','reduce')

def require_ready(path):
    x=json.loads(Path(path).read_text())
    if x.get('status')!='PASS':raise RuntimeError('ACTUAL_TECHNICAL_READY_REQUIRED')
    return x

def validate_execution(lock_path,phase):
    lock=json.loads(Path(lock_path).read_text())
    assert phase in PHASES and lock['allowed_phases']==list(PHASES)
    assert lock['instruction_id']=='ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1'
    assert lock['task_gpu_cap']==lock['project_gpu_cap']==2
    assert lock['save_new_resume_checkpoints'] is False
    assert lock['followup_submissions']==[]
    assert lock['scientific_contrast_families']==94
    assert os.uname().nodename=='server4'
    for member in lock['execution_source_members']:
        p=Path(lock['repo'])/member['relative_path']
        assert p.stat().st_size==member['bytes'] and file_sha(p)==member['sha256'],('FROZEN_SOURCE_DRIFT',str(p))
    # No minimum-space waiver: actual future reserve is locked separately.
    root=Path(lock['root'])
    assert root.is_dir() and os.access(root,os.W_OK)
    return lock

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True);parser.add_argument('--phase',choices=PHASES,required=True)
    args=parser.parse_args(argv);lock=validate_execution(args.lock,args.phase)
    root=Path(lock['root']);out=root/'execution'/lock['attempt'];out.mkdir(parents=True,exist_ok=True)
    phase=args.phase;t=time.monotonic()
    try:
        if phase=='reduce':
            from .reduce import reduce_package
            reduce_package(root,out,Path(lock['publication_repo']),lock)
            return
        if phase!='gate':require_ready(out/'gate/READY.json')
        from .runtime import Runtime
        rt=Runtime(root,lock['repo'])
        if phase=='gate':
            from .technical import validate_checkpoint_inventory,run_g1,reconstruct_timestamp_bank
            gate=out/'gate';gate.mkdir(exist_ok=False)
            cp_receipt=root/'receipts/preflight-r1/checkpoint-content/checkpoint-content.json'
            checked=json.loads(cp_receipt.read_text());assert checked['status']=='PASS' and checked['checkpoints']==12
            for row in checked['rows']:
                stat=Path(row['path']).stat();oldstat=row['current_stat']
                assert (stat.st_dev,stat.st_ino,stat.st_mtime_ns,stat.st_size)==(oldstat['device'],oldstat['inode'],oldstat['mtime_ns'],oldstat['size']),'INPUT_CP_MUTATED_AFTER_CPU_CONTENT_VERIFICATION'
            save(gate/'checkpoint-content-reuse.json',dict(status='PASS',prior_receipt_sha256=file_sha(cp_receipt),
                reuse_level='fresh pre-admission full W/M tensor SHA plus current exact stat; no repeated GPU-allocated CPU scan'))
            rt.set_state(50)
            run_g1(rt,gate/'prefix-g1')
            bank=reconstruct_timestamp_bank(rt,gate/'timestamp')
            from .common import tensor_file
            tensor_file(gate/'timestamp-bank.pt',{'bank':bank['bank'],'case_ids':bank['case_ids']})
            save(gate/'READY.json',dict(status='PASS',scope='G0/source/input/CP plus G1 prefix/restoration/timestamp',
                full_delta_control='DEFERRED_TO_NATIVE_W50_G2',native100='NOT_RUN',SHAM='NOT_RUN',
                lock_sha256=file_sha(args.lock),source_commit=lock['source_commit'],job_id=os.environ.get('SLURM_JOB_ID'),
                seconds=time.monotonic()-t,model_load_seconds=rt.model_load_seconds))
        elif phase=='geometry':
            from .geometry_runner import run_geometry
            panels=json.loads((rt.design/'geometry-panels.json').read_text())
            run_geometry(rt,panels,rt.byid,out/'geometry')
        elif phase=='writers':
            from .writer_runner import run_writers
            run_writers(rt,out/'writers',out/'gate')
        import torch
        save(out/f'{phase}-program.json',dict(status='COMPLETED',phase=phase,job_id=os.environ.get('SLURM_JOB_ID'),
            wall_seconds=time.monotonic()-t,model_load_seconds=rt.model_load_seconds,
            host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            source_commit=lock['source_commit'],lock_sha256=file_sha(args.lock)))
    except BaseException as exc:
        failure=dict(status='TECHNICAL_FAILED',phase=phase,exception_type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-t,
            source_commit=lock['source_commit'],lock_sha256=file_sha(args.lock),automatic_retry=False)
        try:save(out/f'{phase}-failure.json',failure)
        except BaseException as secondary:
            print('FAILURE_RECEIPT_WRITE_FAILED',repr(secondary),flush=True)
        raise

if __name__=='__main__':main()
