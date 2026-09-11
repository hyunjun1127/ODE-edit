"""Focused raw-free final package sealing and actual PNG rerun verification."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import matplotlib
import numpy
from .identity import ROOT,REPO,save,sha,digest,member
from .analysis import csv_save,read

def main(args):
    report=Path(args.report);repro=Path(args.reproduction);repro.mkdir(parents=True,exist_ok=False)
    commands=[]
    for output in (report,repro):
        cmd=[sys.executable,'-m','project.run_scripts.l4_two_memory_conflict_routing.plotting','--input',str(report),'--output',str(output)]
        subprocess.run(cmd,check=True,cwd=REPO);commands.append(cmd)
    images=[]
    for p in sorted(report.glob('*.png')):
        second=repro/p.name
        if p.read_bytes()!=second.read_bytes():raise ValueError('PNG_REPRODUCTION_MISMATCH')
        images.append(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,rerun_sha256=sha(second)))
    save(report/'plot-reproduction-receipt.json',dict(status='BYTE_STABLE_REPRODUCTION_PASS',commands=commands,images=images,
        inputs=[member(report/n) for n in ('endpoint-summary.csv','trajectory.csv','base-preservation-versus-recovery.csv','compute-ledger.csv')],
        source=member(REPO/'project/run_scripts/l4_two_memory_conflict_routing/plotting.py'),
        python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=numpy.__version__,image_tools_used=False))
    jobs=[]
    for job in args.jobs:
        raw=subprocess.check_output(['sacct','-n','-P','-j',str(job),'--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'],text=True)
        lines=[x.split('|') for x in raw.splitlines() if x.split('|')[0]==str(job)]
        if len(lines)!=1:raise ValueError('SCHEDULER_JOB_IDENTITY')
        jid,status,exitcode,elapsed,tres=lines[0][:5]
        if status in ('RUNNING','PENDING','COMPLETING'):raise ValueError('TASK_ALLOCATION_NOT_TERMINAL')
        gpu=[int(x.split('=')[1]) for x in tres.split(',') if x.startswith('gres/gpu=')]
        if gpu!=[1]:raise ValueError('SCHEDULER_GPU_ACCOUNTING')
        jobs.append(dict(job=jid,status=status,exit=exitcode,elapsed_seconds=int(elapsed),allocated_gpus=1,gpu_seconds=int(elapsed),alloc_tres=tres))
    csv_save(report/'job-gpu-ledger.csv',jobs)
    save(report/'resource-receipt.json',dict(jobs=jobs,total_gpu_seconds=sum(j['gpu_seconds'] for j in jobs),
        total_gpu_hours=sum(j['gpu_seconds'] for j in jobs)/3600,project_cap=2,requested_memory_mib=182272,
        gpu_hour_cap=None,other_task_budget_inherited=False,all_attempts_included=args.jobs))
    from .report import main as make_report
    make_report(argparse.Namespace(root=str(report)))
    forbidden=('.pt','.npz','.safetensors','.bin','.pkl','.out','.err','.log')
    for p in report.rglob('*'):
        if p.is_symlink():raise ValueError('PUBLICATION_SYMLINK')
        if p.is_file() and p.suffix in forbidden:raise ValueError('FORBIDDEN_RAW_PUBLICATION')
    required=['bank-manifest.json','geometry-spectrum.csv','conflict-map.csv','native-response-parity.csv','trajectory.csv',
              'per-context-harm.csv','paired-endpoints.csv','compute-ledger.csv','base-preservation-versus-recovery.csv','diagnostic-report-ko.md',
              'requirements-evidence.csv','coverage-receipt.json','bank-overlap.csv','intermediate-current.csv',
              'structural-actions.csv','physical-path-actions.csv','context-response.csv','job-gpu-ledger.csv',
              'calibration.csv','input-mode-conflict.csv','input-mode-endpoint-change.csv','input-preservation-receipt.json',
              'snapshot-check.json','joint-domain-checks.csv','joint-domain-receipt.json','technical-exclusions.json','peak-memory.csv',
              'all-attempt-compute.csv','all-attempt-compute-receipt.json']
    missing=[p for p in required if not (report/p).is_file()]
    if missing:raise ValueError(f'MISSING_PUBLICATION: {missing}')
    code=[member(p) for p in sorted((REPO/'project/run_scripts/l4_two_memory_conflict_routing').glob('*')) if p.is_file()]
    payload=[member(p) for p in sorted(report.iterdir()) if p.is_file()]
    manifest=dict(status='FULL_REHASH_PASS',members=payload,members_root=digest(payload),source=code,
        source_root=digest(code),analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        analysis_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=REPO,text=True).strip(),
        raw_policy='LOCAL_ONLY; member identities in raw-input-manifest.json',broadcast='NO_BROADCAST_NOT_REQUIRED',scientific_promotion=False)
    save(report/'analysis-manifest.json',manifest)
    receipt=dict(status='REVIEW_READY',manifest_sha256=sha(report/'analysis-manifest.json'),members_root=manifest['members_root'],
        report_sha256=sha(report/'diagnostic-report-ko.md'),source_root=manifest['source_root'],scientific_promotion=False)
    receipt['identity']=digest(receipt);save(report/'rooted-receipt.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--reproduction',required=True)
    p.add_argument('--jobs',nargs='+',type=int,required=True);main(p.parse_args())
