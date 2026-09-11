"""Create-once union of completed CPU geometry parts without reevaluation."""
import argparse
import csv
from pathlib import Path
from .analysis import csv_save,read
from .identity import save,member,digest

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);refs=[];seconds=0
    names=('geometry-spectrum.csv','structural-actions.csv','physical-path-actions.csv','input-mode-conflict.csv','input-mode-endpoint-change.csv')
    data={n:[] for n in names};seen=set()
    for part in map(Path,args.parts):
        t=read(part/'terminal.json')
        if t['status']!='CPU_GEOMETRY_COMPLETE':raise ValueError('INCOMPLETE_GEOMETRY_PART')
        seconds+=t['seconds'];refs.append(member(part/'terminal.json'))
        inventory=set()
        for name in names:
            with (part/name).open() as f:rr=list(csv.DictReader(f))
            data[name]+=rr;refs.append(member(part/name))
            if name=='structural-actions.csv':inventory={(r['entry'],r['batch_raw']) for r in rr}
        if seen&inventory:raise ValueError('DUPLICATE_GEOMETRY_PART')
        seen|=inventory
    if seen!={('Early','100'),('Middle','100'),('Late','100'),('Middle','1'),('Middle','7')}:raise ValueError('GEOMETRY_INVENTORY')
    for name,rr in data.items():csv_save(out/name,rr)
    save(out/'terminal.json',dict(status='CPU_GEOMETRY_COMPLETE',seconds=seconds,parts=args.parts,
        members=refs,members_root=digest(refs),join_model_actions=0,join_GPU_actions=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parts',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
