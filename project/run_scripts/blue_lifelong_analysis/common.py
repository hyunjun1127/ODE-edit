from pathlib import Path
from project.run_scripts.blue_fivearm_analysis.common import (sha,digest,read,save,csvread,csvwrite,stats,table,fmt)

INSTRUCTION='ODEEDIT-S06-BLUE-L4-L8-LIFELONG-EXHAUSTIVE-REVIEW-SH4-V1'
BASE=Path('/data/janghj/ODE-edit')
LOCAL=BASE/'local/blue-l4-l8-lifelong-review/attempt-v1'
OUT_REL='experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v1'
SAMPLE_ROOT='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
PREFIX_ROOT='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
SCHEDULE=[1,5,10,20,30,40,50,60,70,80,90,100]
ARMS=['MEMIT_ORIGINAL','AlphaEdit_ORIGINAL','MEMIT_L4_ONLY','AlphaEdit_L4_ONLY','MEMIT_L8_ONLY','AlphaEdit_L8_ONLY']
JOBS=['39307','39283_1','39283_2','39283_3','39283_4','39283_5']
MULT={'RS':1,'PS':2,'NS':10}

def attempt(cell):
    return BASE/'local/blue-lifelong-b100x100'/('attempt-cell0-checkpoint-r3' if cell==0 else 'attempt-checkpoint-r2')

def root(cell):return attempt(cell)/'output'/f'main-cell-{cell}'

def sample():
    lock=read(attempt(0)/'execution.lock.json');s=read(lock['sample'])
    assert digest(s['records'])==SAMPLE_ROOT==lock['sample_root']
    assert len(s['records'])==10000 and len({r['case_id'] for r in s['records']})==10000
    assert all((r['ordinal'],r['batch_index'],r['batch_ordinal'])==(i,i//100+1,i%100) for i,r in enumerate(s['records']))
    assert [r['case_id'] for r in s['records']]==[r['case_id'] for r in s['inventory']]
    assert all((r['rewrite'],r['rephrase'],r['locality'])==(1,2,10) for r in s['inventory'])
    return s

def checked_member(r,relative,manifest):
    p=r/relative;m=manifest[relative]
    assert p.is_file() and not p.is_symlink()
    actual=sha(p);assert actual==m['sha256'] and p.stat().st_size==m['bytes'],str(p)
    return dict(path=str(p),bytes=m['bytes'],sha256=actual,status='FULL_REHASH_PASS')

def cli():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);return p.parse_args()

def table(rows,cols):
    # Unlike the historical formatter, never parse a job ID such as 39283_1 as a float.
    def cell(x):
        if isinstance(x,float):return f'{x:.6g}'
        return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']+
        ['|'+'|'.join(cell(r.get(k,'NOT_RECORDED')) for k in cols)+'|' for r in rows])+'\n'
