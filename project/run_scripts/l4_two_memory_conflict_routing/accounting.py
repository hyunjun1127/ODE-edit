"""Actual allocation counters, including excluded and interrupted work."""
import argparse
from pathlib import Path
from .analysis import read,csv_save
from .identity import save,member,digest,ROOT

def main(args):
    out=Path(args.output);spec=read(args.allocations);rows=[];receipts=[]
    for record in spec['allocations']:
        root=Path(record['root']);terminal=root/'terminal.json'
        if terminal.exists():
            data=read(terminal);source=terminal;scope='COMPLETE_ALLOCATION_COUNTERS'
        else:
            progress=sorted(root.rglob('progress/*.json'))
            # Global ns filename is the event timestamp, independent of folder.
            progress.sort(key=lambda p:int(p.stem))
            if progress:source=progress[-1];data=read(source);scope='INTERRUPTED_LAST_SAVED_LOWER_BOUND'
            else:source=root/'run.lock.json';data={};scope='NOT_RECORDED_BEFORE_INTERRUPTION'
        receipts.append(dict(**record,evidence=member(source),counter_scope=scope))
        for kind,values in data.get('compute',{}).items():
            if isinstance(values,dict):
                for name,value in values.items():rows.append(dict(job=record['job'],purpose=record['purpose'],scope=scope,
                    category=kind,component=name,value=value))
    csv_save(out/'all-attempt-compute.csv',rows)
    cpu=[]
    for path in sorted(ROOT.glob('geometry-*/terminal.json')):
        item=read(path)
        if 'parts' in item:continue  # Joined receipts are sums, not extra runs.
        cpu.append(dict(path=str(path),seconds=item['seconds'],evidence_sha=member(path)['sha256'],
            scope='actual CPU geometry pass, including superseded previews',GPU_actions=0))
    csv_save(out/'cpu-geometry-ledger.csv',cpu)
    save(out/'all-attempt-compute-receipt.json',dict(allocations=receipts,members_root=digest(receipts),
        scheduler_gpu_seconds_authority='job-gpu-ledger.csv including interrupted residency',
        cpu_geometry_seconds_sum=sum(r['seconds'] for r in cpu),
        cpu_wall_sum_not_campaign_elapsed=True,interrupted_unrecorded_calls_not_imputed=True,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocations',required=True);p.add_argument('--output',required=True);main(p.parse_args())
