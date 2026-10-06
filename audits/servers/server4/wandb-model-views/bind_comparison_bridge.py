"""Bind exactly the twelve already registered repair jobs; no scheduler reads."""
import json
from pathlib import Path
from project.run_scripts.jlz_interference_l1.comparison_bridge import sha,read,check

def main():
    rows=[]
    for task,writer in [('jlz-price-cap-base-repair-2k','memit'),('jlz-price-alpha-writer-2k','alphaedit')]:
        p=Path('/data/janghj/ODE-edit/local')/task/'logging-repair-20261007'
        sub=read(p/'submission.json');lock=read(p/'execution.lock.json')
        check(sha(p/'config.json')==lock['config_sha256'],'FROZEN_CONFIG')
        for cell,job in sub['jobs'].items():
            if cell=='collector':continue
            rows.append(dict(attempt=str(p),task_id=task,writer=writer,cell=cell,model=cell.split('_')[0],
                job=str(job),source=lock['source_commit'],config_sha256=lock['config_sha256'],
                lock_sha256=sha(p/'execution.lock.json')))
    check(len(rows)==12 and len({r['job'] for r in rows})==12,'EXACT_TWELVE_JOBS')
    out=Path('audits/servers/server4/wandb-model-views/comparison-bindings.json')
    with out.open('x') as f:
        json.dump(dict(authority='USER 2026-10-07 실시간 자동 동기화',targets=rows,
            GPU=0,Slurm_mutation=False,original_run_writes=False,poll_seconds=60,max_hours=168,
            scope='CPU sealed scalar comparison only; no progress/quality gate or scientific retry'),f,indent=2)
    print(out)

if __name__=='__main__':main()
