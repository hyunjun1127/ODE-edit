"""Terminal-stage Slurm allocation cost, excluding step double counting."""
import argparse
import csv
import io
import json
import subprocess
from pathlib import Path
from .analysis import write_csv
from .import_assets import sha
from .records import save,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--stage',choices=['A','B','C'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-job',action='append',default=[]);a=p.parse_args();jobs={};inputs=[]
    for entry,config in json.loads(a.registry.read_text()).items():
        roots=[config['native'],*config.get('direct',[])] if a.stage=='A' else ([config[a.stage]] if a.stage in config else [])
        for root in map(Path,roots):
            path=root/'runtime.json';r=json.loads(path.read_text());job=str(r['slurm_job'])
            jobs[job]=dict(entry=entry,output=str(root),source_head=r['source_head'],scope='CANONICAL_EXECUTION')
            inputs.append(dict(path=str(path),sha256=sha(path)))
    for job in a.extra_job:jobs.setdefault(job,dict(scope='ADDITIONAL_TECHNICAL_ATTEMPT'))
    if not jobs:raise ValueError('NO_RECORDED_JOBS')
    raw=subprocess.check_output(['sacct','-j',','.join(sorted(jobs)),
         '--format=JobID,JobName%100,User,State,ExitCode,ElapsedRaw,AllocTRES%200,Start,End','-P'],text=True,timeout=30)
    rows=[]
    for r in csv.DictReader(io.StringIO(raw),delimiter='|'):
        if r['JobID'] not in jobs:continue
        if r['User']!='janghj' or not r['JobName'].startswith('odeedit_cumrisk_'):raise ValueError('JOB_OWNERSHIP_MISMATCH')
        if r['State'] not in ['COMPLETED','FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED'] and not r['State'].startswith('CANCELLED'):raise ValueError('NONTERMINAL_COST_ACCOUNTING')
        tres=dict(x.split('=',1) for x in r['AllocTRES'].split(',') if '=' in x)
        gpu=int(tres.get('gres/gpu','0'));seconds=int(r['ElapsedRaw'])*gpu
        rows.append(dict(**r,**jobs[r['JobID']],gpu_count=gpu,allocated_gpu_seconds=seconds,allocated_gpu_hours=seconds/3600))
    if {r['JobID'] for r in rows}!=set(jobs):raise ValueError('INCOMPLETE_SCHEDULER_ACCOUNTING')
    a.output.mkdir(parents=True,exist_ok=False);member=write_csv(a.output/'job-gpu-hour-ledger.csv',rows)
    save(a.output/'job-accounting-receipt.json',dict(stage=a.stage,inputs=inputs,input_root=digest(inputs),output=member,
         jobs=len(rows),allocated_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in rows),
         allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows),gpu_hour_cap=None,
         inherited_budget=False,scheduler_step_double_count=0,server_time_zone='Asia/Seoul',
         measured_scope='Full allocation elapsed, including preparation/evaluation/generation/idle-in-process, not kernel-only time.'))

if __name__=='__main__':main()
