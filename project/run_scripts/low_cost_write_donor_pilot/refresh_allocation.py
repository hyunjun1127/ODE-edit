"""One bounded own-job allocation capture; no loop or scheduler mutation."""
import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import re
import subprocess

from .runtime import save

FIELDS=['JobID','JobIDRaw','State','ExitCode','ElapsedRaw','Submit','Start','End','AllocTRES','ReqMem','NodeList','MaxRSS']

def allocation_rows(raw):
    rows=list(csv.DictReader(io.StringIO(raw),delimiter='|'))
    jobs=[]
    for r in rows:
        if '.' in r['JobID']:
            continue
        if not re.fullmatch(r'\d+(?:_\d+)?',r['JobID']):
            continue
        tres=dict(p.split('=',1) for p in r['AllocTRES'].split(',') if '=' in p)
        gpus=int(tres.get('gres/gpu','0'))
        jobs.append(dict(job=r['JobID'],raw_id=r['JobIDRaw'],status=r['State'],exit=r['ExitCode'],
                         elapsed_seconds=int(r['ElapsedRaw']),gpus=gpus,
                         allocated_gpu_seconds=int(r['ElapsedRaw'])*gpus,
                         submit=r['Submit'],start=r['Start'],end=r['End'],mem=r['ReqMem'],node=r['NodeList'],
                         max_rss=r.get('MaxRSS') or 'NOT_RECORDED_JOB_LEVEL'))
    return jobs

def concurrency(jobs):
    events=[]
    for r in jobs:
        if r['start'] in ('Unknown','None','') or r['end'] in ('Unknown','None',''):
            return dict(status='NOT_TERMINAL',max_concurrent_GPUs=None)
        if r['gpus']==0:
            continue
        events += [(datetime.fromisoformat(r['start']),r['gpus']),
                   (datetime.fromisoformat(r['end']),-r['gpus'])]
    active=peak=0;gpu_seconds=two_gpu_seconds=0;previous=None
    for t,change in sorted(events):
        if previous is not None:
            dt=(t-previous).total_seconds();gpu_seconds+=dt*active
            if active>=2:two_gpu_seconds+=dt
        active+=change;peak=max(peak,active);previous=t
    return dict(status='RECORDED_ALLOCATION_INTERVALS',max_concurrent_GPUs=peak,
                interval_gpu_seconds=gpu_seconds,at_least_two_GPU_seconds=two_gpu_seconds,
                GPU_compute_occupancy='NOT_MEASURED_ALLOCATION_IS_NOT_UTILIZATION')

def snapshot(jobids,out):
    assert jobids and all(re.fullmatch(r'\d+(?:_\d+)?',j) for j in jobids)
    command=['sacct','-P','-j',','.join(jobids),'--format='+','.join(FIELDS)]
    result=subprocess.run(command,check=True,text=True,capture_output=True)
    jobs=allocation_rows(result.stdout)
    assert len({r['job'] for r in jobs})==len(jobs)
    receipt=dict(command=command,rows=jobs,concurrency=concurrency(jobs),
                 allocated_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in jobs),
                 readonly=True,polling_loop=False,raw_sacct=result.stdout)
    return save(Path(out),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(snapshot(args.jobs,args.out)))
