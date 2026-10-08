"""One cold official main cell, GPU qualification USER_DISABLED, no retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from official.experiments.prepare import file_sha,write_new
from official.runners.server4.qwen_plan import rows
from official.runners.server4.qwen_submission_plan import require_execution_enabled, QUALIFICATION_STATUS, AUTHORITY


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


def child(root,command,output,*,dataset,config=None,extra=(),label):
    argv=[sys.executable,'-B','-m','official.runners.server4.qwen_run',command,
          '--assets',str(root/'assets.json'),'--stream',str(root/'streams'/f'{dataset}-stream.json'),
          '--output',str(output)]
    argv += ['--dataset',dataset] if command=='w0' else ['--config',str(config)]
    env=dict(os.environ,ODEEDIT_ATTEMPT_ID=f'{root.name}-{label}-job{os.environ["SLURM_JOB_ID"]}')
    started=now()
    with (root/'logs'/f'{label}-{os.environ["SLURM_JOB_ID"]}.out').open('xb') as log:
        result=subprocess.run([*argv,*extra],env=env,stdout=log,stderr=subprocess.STDOUT)
    write_new(root/'processes'/f'{label}-{os.environ["SLURM_JOB_ID"]}.json',
              dict(argv=argv,returncode=result.returncode,started_at_utc=started,
                   stopped_at_utc=now(),job_id=os.environ['SLURM_JOB_ID']))
    if result.returncode:raise RuntimeError(f'CHILD_FAILED:{label}:{result.returncode}')


def run(root,logical):
    require_execution_enabled()
    root=Path(root).resolve(); row=next(r for r in rows() if r['logical_main_row']==logical)
    config=root/'configs'/f'{logical}.json'
    if read(config)!=row['config']:raise ValueError('CELL_CONFIG_CHANGED')
    dataset=row['config']['dataset'];method=row['config']['method']
    job=os.environ['SLURM_JOB_ID']; out=root/'runs'/logical
    if out.exists():raise RuntimeError('NO_DUPLICATE_COLD_CHAIN')
    (root/'logs').mkdir(exist_ok=True)
    write_new(root/'validation'/f'{logical}.json',dict(
        authority=AUTHORITY, qualification=QUALIFICATION_STATUS,
        GPU_smoke=QUALIFICATION_STATUS, GPU_resume_equivalence=QUALIFICATION_STATUS,
        native_oracle_reevaluation=QUALIFICATION_STATUS,
        runtime_identity_finite_shape_commit_guards='PRESERVED'))
    # Every GPU cell runs after the previous archival gate, not merely GPU exit.
    w0=root/'shared-w0'/f'qwen25-{dataset}'
    if not (w0/'w0-receipt.json').exists():
        if method!='FT':raise RuntimeError('SHARED_W0_PREDECESSOR_MISSING')
        child(root,'w0',w0,dataset=dataset,label=f'{dataset}-w0')
    child(root,'execute',out,dataset=dataset,config=config,label=logical)
    pointer=read(out/'checkpoint/latest.json')
    commits=[read(out/'commits'/f'b{b:02d}.json') for b in range(1,21)]
    if pointer['batch']!=20 or not pointer['final_W20'] or [c['completed_batch'] for c in commits]!=list(range(1,21)):
        raise RuntimeError('W20_JOIN_INCOMPLETE')
    write_new(out/'terminal.json',dict(schema='server4-qwen-official-terminal-v1',
        logical_main_row=logical,actual_job_id=job,completed_at_utc=now(),
        status='W20_COMPLETE',completed_edits=2000,checkpoint=pointer,
        commits=[dict(path=str(out/'commits'/f'b{b:02d}.json'),sha256=file_sha(out/'commits'/f'b{b:02d}.json')) for b in range(1,21)],
        config_sha256=row['config']['config_sha256'],dataset=dataset,
        qualification=QUALIFICATION_STATUS,
        archive_pending=True,scientific_job_source_immutable=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cell',required=True);args=parser.parse_args();run(args.root,args.cell)
