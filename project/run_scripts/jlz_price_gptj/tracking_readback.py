"""Bounded CPU-only readback after original logger finish; never init/resume."""
import argparse,json,math
from pathlib import Path
from project.run_scripts.jlz_interference_l1.comparison_bridge import read,check,atomic,ENTITY,PROJECT

def verify(out):
    import wandb
    identity=read(out/'tracking-identity.json');expected=[]
    with (out/'tracking/accepted-scalars.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['op']=='log':expected.append(row['values'])
    check(len(expected)<=1000,'READBACK_BOUNDED_ROWS')
    run=wandb.Api(timeout=20).run(f'{ENTITY}/{PROJECT}/{identity["run_id"]}')
    check(all(run.config.get(k)==v for k,v in identity['config'].items()),'REMOTE_CONFIG')
    check(all(run.config.get(k)==v for k,v in identity['job_identity'].items()),'REMOTE_JOB_CONFIG')
    check('job'+identity['job_identity']['job_display_id'] in run.name,'REMOTE_JOB_NAME')
    actual=list(run.scan_history(page_size=1000,min_step=0,max_step=1001))
    exact=len(actual)==len(expected) and all(all(k in a and math.isclose(a[k],v,rel_tol=1e-12,abs_tol=1e-9)
        for k,v in e.items()) for a,e in zip(actual,expected))
    receipt=dict(status='REMOTE_PAYLOAD_VERIFIED' if exact else 'LOGGING_INCOMPLETE_OR_MISMATCH',
        run_id=identity['run_id'],url=identity['url'],expected_rows=len(expected),remote_rows=len(actual),
        exact_payload=exact,scientific_completion_claim=False,source_sha=identity['source_sha'],
        remote_run_not_modified=True,extra_forward=0)
    atomic(out/'tracking-readback.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:print(json.dumps(verify(args.out)))
    except Exception as error:
        atomic(args.out/'tracking-readback.json',dict(status='LOGGING_READBACK_NOT_VERIFIED',
            error_type=type(error).__name__,scientific_completion_claim=False,remote_run_not_modified=True))
