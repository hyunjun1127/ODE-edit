"""Read-only union of disjoint endpoint-observation processes, no raw copying."""
import argparse
from collections import defaultdict
from pathlib import Path
from .identity import save,member,digest
from .analysis import read

def folders(root):
    root=Path(root);meta=read(root/'terminal.json')
    for part in meta.get('parts',[str(root)]):
        yield from sorted(Path(part).glob('*-B*'))

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    seen=set();runs=[];counts=defaultdict(int);seconds=defaultdict(float);refs=[]
    for part in map(Path,args.parts):
        t=read(part/'terminal.json')
        if t['status']!='OBSERVATIONS_COMPLETE':raise ValueError('OBSERVATION_PART_INCOMPLETE')
        for folder in folders(part):
            if folder.name in seen:raise ValueError('DUPLICATE_OBSERVATION_FIXTURE')
            seen.add(folder.name)
        runs+=t['runs'];refs.append(member(part/'terminal.json'))
        for k,v in t['compute']['counts'].items():counts[k]+=v
        for k,v in t['compute']['seconds'].items():seconds[k]+=v
    if seen!={'Early-B100','Middle-B100','Late-B100','Middle-B1','Middle-B7'}:raise ValueError('OBSERVATION_FIXTURE_COVERAGE')
    save(out/'terminal.json',dict(status='OBSERVATIONS_COMPLETE',parts=[str(Path(p).resolve()) for p in args.parts],runs=runs,
        part_receipts=refs,parts_root=digest(refs),compute=dict(counts=dict(counts),seconds=dict(seconds)),
        model_loads_in_join=0,new_trajectories=0,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parts',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
