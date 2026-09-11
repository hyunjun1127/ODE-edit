"""Create-once regular-file view of valid original and repaired paths.

Hardlinks avoid copying multi-GB immutable tensors. The view is only read by
analysis/evaluation; no writer may run in it. Original files are never edited.
"""
import argparse
import os
from pathlib import Path
from .analysis import read
from .identity import save,member,digest

def main(args):
    original=Path(args.original).resolve();repair=Path(args.repair).resolve();out=Path(args.output)
    old=read(original/'terminal.json');new=read(repair/'terminal.json')
    if new['status']!='REPAIR_TERMINAL_VALID' or Path(new['original_run']).resolve()!=original:raise ValueError('REPAIR_BINDING')
    arms=new['arms'];out.mkdir(parents=True,exist_ok=False)
    omit=set(arms)|{a+'-trajectory' for a in arms}|{'progress','run.lock.json','terminal.json'}
    for parent,skip in ((original,omit),(repair,{'progress','terminal.json'})):
        for source in sorted(parent.rglob('*')):
            rel=source.relative_to(parent)
            if rel.parts[0] in skip:continue
            if source.is_symlink():raise ValueError('SOURCE_SYMLINK')
            target=out/rel
            if source.is_dir():target.mkdir(parents=True,exist_ok=True)
            elif source.is_file():target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    # Remove only failed-path portions from canonical counters. Actual wasted
    # calls/allocation are preserved separately and charged in final accounting.
    counts={k:dict(v) for k,v in old['compute'].items() if isinstance(v,dict)}
    excluded=[]
    for arm in arms:
        predecessor='BF1' if arm=='BF8' else 'BF8'
        before=read(original/predecessor/'terminal.json')['ledger']
        after=read(original/arm/'terminal.json')['ledger']
        delta={kind:{key:value-before[kind].get(key,0) for key,value in after[kind].items()} for kind in ('counts','seconds')}
        for kind,values in delta.items():
            for key,value in values.items():counts[kind][key]=counts[kind].get(key,0)-value
        excluded.append(dict(arm=arm,status='TECHNICAL_EXCLUDED_NEGATIVE_DUAL_SLACK',
            old_path=str(original/arm),old_terminal=member(original/arm/'terminal.json'),compute=delta))
    for kind,values in new['compute'].items():
        for key,value in values.items():counts[kind][key]=counts[kind].get(key,0)+value
    oldlock=read(original/'run.lock.json')
    save(out/'run.lock.json',dict(**{k:v for k,v in oldlock.items() if k not in ('execution_head','job')},
        execution_head=oldlock['execution_head']+' + BF-repair '+new['source_head'],
        job=str(oldlock['job'])+' + BF-repair '+str(new['job']),
        original_lock=member(original/'run.lock.json'),repair_lock=member(repair.parent/'run.lock.json')))
    lineage=dict(status='CANONICAL_COMPOSITE_READ_ONLY',original=str(original),repair=str(repair),
        arms={arm:dict(path=str((repair if arm in arms else original)/arm),source_head=new['source_head'] if arm in arms else oldlock['execution_head']) for arm in old['arms']},
        exclusions=excluded,old_original_terminal_claim_superseded=True,source_files_mutated=0,
        scientific_promotion=False)
    save(out/'repair-lineage.json',lineage)
    save(out/'terminal.json',dict(**{k:v for k,v in old.items() if k not in ('compute','peak_gpu_bytes')},
        compute=counts,peak_gpu_bytes=max(old['peak_gpu_bytes'],new['peak_gpu_bytes']),
        repair_lineage_identity=digest(lineage),canonical_counter_scope='valid reused preparation+paths; technical-excluded costs separate'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--repair',required=True);p.add_argument('--output',required=True);main(p.parse_args())
