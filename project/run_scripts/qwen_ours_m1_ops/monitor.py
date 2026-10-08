"""Job-lifetime CPU companion: bounded phase readback and numeric JSON files.

No W&B init/history writer, model, tensor, raw text upload, or Slurm mutation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

CELL='QWEN_M1_CAP075'
M1={2:19,3:47,7:26,10:25,17:6,18:44,20:32}
BLOCK={'case_id','case_ids','record_ids','ids','prompt','prompts','text','path','W','H','R','K','P','M'}

def numbers(value):
    if value is None or type(value) in (bool,int):return value
    if type(value) is float:return value if math.isfinite(value) else None
    if isinstance(value,list):return [numbers(x) for x in value]
    if isinstance(value,dict):return {k:numbers(v) for k,v in value.items()
        if (k not in BLOCK or (k in ('R','P') and isinstance(v,dict)
            and {'numerator','denominator'}<=set(v))) and not isinstance(v,str)}
    return None

def read(path):return json.loads(path.read_text())
def write(path,value):
    text=json.dumps(value,allow_nan=False,sort_keys=True)+'\n'
    if len(text.encode())>512*1024:raise ValueError('COMPACT_RECEIPT_TOO_LARGE')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(text);os.replace(tmp,path)

def expected_row(spool,batch):
    step=0;found=None
    path=spool/'accepted-scalars.jsonl'
    if not path.exists():return None
    with path.open() as f:
        for line in f:
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row.get('op')!='log':continue
            actual=step if row.get('step') is None else row['step'];step=actual+1
            v=row['values']
            if v.get('batch')==batch and 'current/post/N/count' in v:
                found={'step':actual,'values':v}
    return found

def collect(attempt,ops,label):
    env=dict(os.environ,PYTHONPATH=str(attempt/'source'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run(['/data/janghj/EasyEdit/.venv/bin/python','-m',
        'project.run_scripts.qwen_ours_m1.collect','--attempt',str(attempt)],
        env=env,cwd=attempt/'source',capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError('COLLECTOR_EXIT_'+str(p.returncode))
    value=json.loads(p.stdout);write(ops/('collection-'+label+'.json'),value)
    return value

def main(attempt,ops):
    import wandb
    from project.run_scripts.experiment_tracking.schema import load_env,ENTITY,PROJECT
    from project.run_scripts.experiment_tracking.readback import verify_last_rows
    c=read(attempt/'config.json')['cells'][CELL]
    settings=load_env(c['tracking']['env_file']);root=attempt/CELL;spool=root/'tracking'
    done=set();trials={};identity=None;started=time.monotonic();finished_at=None
    while time.monotonic()-started<24*3600+300:
        finished=(ops/'process-finished.json').exists()
        if finished and finished_at is None:finished_at=time.monotonic()
        if identity is None and (spool/'identity.json').exists():identity=read(spool/'identity.json')
        if identity:
            events=[('startup',0)]
            events += [(f'price-{b:02d}',b) for b in range(1,21)
                if (root/f'batch-{b:02d}'/'entry-price.json').exists()]
            events += [(f'commit-{b:02d}',b) for b in range(1,21)
                if (root/f'batch-{b:02d}'/'commit.json').exists()]
            if finished:events.append(('finish',20))
            for label,b in events:
                if label in done:continue
                try:
                    api=wandb.Api(overrides={'base_url':settings['WANDB_BASE_URL']},timeout=12)
                    remote=api.run(ENTITY+'/'+PROJECT+'/'+identity['run_id'])
                    if remote.id!=identity['run_id'] or remote.name!=identity['run_name']:
                        raise ValueError('REMOTE_IDENTITY')
                    if any(remote.config.get(k)!=v for k,v in identity['config'].items()):
                        raise ValueError('REMOTE_CONFIG')
                    evidence=dict(status='REMOTE_IDENTITY_VERIFIED',run_id=remote.id,url=identity['url'],phase=label,
                        observed_remote_batch=remote.summary.get('batch'),observed_remote_edits=remote.summary.get('edits'))
                    payload={'batch':b}
                    folder=root/f'batch-{b:02d}'
                    if label.startswith('price'):
                        v=read(folder/'entry-price.json')['payload']
                        keys=('layers','anchors','anchor_star','computed_pi','effective_pi','raw_kappa',
                            'beta_base','beta_max','local_caps','denominator','layer_checks')
                        payload['entry_price']=numbers({k:v[k] for k in keys if k in v})
                    if label.startswith('commit'):
                        for name in ('fit.json','m1-anchor.json','writer/realization.json'):
                            p=folder/name
                            if p.exists():payload[name]=numbers(read(p))
                        v=read(folder/'commit.json')
                        payload['elapsed_seconds']=v.get('seconds')
                        row=expected_row(spool,b)
                        if b in (1,2,5,10,15,20):
                            if row is None:raise RuntimeError('LOCAL_POINT_NOT_YET_ACCEPTED')
                            check=verify_last_rows(wandb,settings['WANDB_BASE_URL'],remote.id,
                                identity['config'],identity['run_name'],{'evaluation':row})
                            evidence['readback']=check
                            if check['status']!='REMOTE_BOUNDED_ROWS_VERIFIED':raise RuntimeError('REMOTE_POINT_NOT_YET_VERIFIED')
                        if row:
                            for scope in ('current','all_seen'):
                                new=row['values'].get(scope+'/post/N/success_pct')
                                old=row['values'].get('w0/'+scope+'/N/success_pct')
                                if new is not None and old is not None:payload[scope+'_NS_delta_pp']=new-old
                        if b in (5,10,15,20):payload['CPU_reduction']=numbers(collect(attempt,ops,label))
                        if b in M1:
                            owner=M1[b];last=None
                            with (folder/'events.jsonl').open() as f:
                                for line in f:
                                    r=json.loads(line)
                                    if r.get('event')=='candidate':last=r['payload']
                            payload['M1_owner']=owner
                            if last:payload['M1_fit_NLL_by_context']=last['nll'][owner]
                            gap=read(folder/'writer/subject-alltoken-gap.json')
                            if 'actual_rewrite_NLL_by_context' in gap:
                                payload['M1_actual_NLL_by_context']=gap['actual_rewrite_NLL_by_context'][owner]
                    if label=='finish':
                        payload['CPU_reduction']=numbers(collect(attempt,ops,label))
                        if (root/'terminal.json').exists():payload['terminal']=read(root/'terminal.json')['status']
                        if (spool/'receipt.json').exists():evidence['transport_status']=read(spool/'receipt.json')['status']
                    if label!='startup':
                        target=ops/'uploads'/(label+'.json');write(target,payload)
                        uploaded=remote.upload_file(str(target),root=str(ops/'uploads'))
                        evidence['numeric_JSON_file']=uploaded.name
                    write(ops/'readback'/(label+'.json'),evidence);done.add(label)
                except Exception as error:
                    trials[label]=trials.get(label,0)+1
                    write(ops/'readback'/(label+'.json'),dict(status='LOGGING_UNVERIFIED_OR_DEGRADED',
                        phase=label,error_type=type(error).__name__,attempts=trials[label],scientific_retry=False))
                    if trials[label]>=3:done.add(label)
        if finished and ((identity is not None and 'finish' in done) or time.monotonic()-finished_at>90):break
        time.sleep(15)
    if not identity:write(ops/'readback/missing-startup.json',{'status':'NO_ONLINE_IDENTITY_OBSERVED'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--ops',type=Path,required=True)
    a=p.parse_args();main(a.attempt.resolve(),a.ops.resolve())
