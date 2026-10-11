"""Read-only native FE monitor. It never submits/cancels/changes any run."""
import argparse, ast, datetime, hashlib, json, math, os, re, statistics, subprocess, time
from pathlib import Path

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/native-r2')
NUM=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)'
MODELS={'qwen25':('63385',[4,5,6,7,8]),'gptj':('63386',[3,4,5,6,7,8])}

def parse_log(path):
    batches={};batch=None;layer=None;checkpoints=[]
    for line in path.read_text(errors='replace').splitlines() if path.exists() else []:
        match=re.match(r'NATIVE_FE_BATCH_START (\d+) case_ids=(.+)',line)
        if match:
            batch=int(match[1]);layer=None
            batches[batch]=dict(batch=batch,case_ids=ast.literal_eval(match[2]),fit=[],layers={},complete=False)
        elif line.startswith('NATIVE_FE_CHECKPOINT '):
            checkpoints.append(json.loads(line.removeprefix('NATIVE_FE_CHECKPOINT ')))
        elif line.startswith('NATIVE_FE_BATCH_COMPLETE '):
            batches[int(line.split()[-1])]['complete']=True
        elif batch is not None:
            match=re.match(r'Init norm ('+NUM+r') \| Delta norm ('+NUM+r') \| Target norm ('+NUM+r')',line)
            if match:batches[batch]['fit'].append(dict(zip(('anchor','delta','target'),map(float,match.groups()))))
            match=re.match(r'LAYER (\d+)',line)
            if match:
                layer=match[1];batches[batch]['layers'][layer]={}
            if layer is not None:
                for label,key in [('z error','residual'),('orig norm','weight'),('upd norm','update')]:
                    match=re.match(label+r' tensor\(('+NUM+r')',line)
                    if match:batches[batch]['layers'][layer][key]=float(match[1])
    output=[];warnings=[]
    for batch,row in sorted(batches.items()):
        fits=row.pop('fit');row['fit_count']=len(fits)
        row['fit_norms']={key:{'median':statistics.median([f[key] for f in fits]),'max':max(f[key] for f in fits)} for key in ('anchor','delta','target')} if fits else {}
        for layer,values in row['layers'].items():
            for key,value in values.items():
                if not math.isfinite(value):warnings.append(f'B{batch} L{layer} {key} {value}')
                prior=[r['layers'][layer][key] for r in output if r['complete'] and layer in r['layers'] and key in r['layers'][layer]]
                if prior and key in ('residual','update') and value>8*max(statistics.median(prior),1e-3):
                    warnings.append(f'B{batch} L{layer} {key} {value} exceeds 8x prior median {statistics.median(prior)}; investigate, not automatic cancellation')
        row['case_ids_sha256']=hashlib.sha256(json.dumps(row.pop('case_ids')).encode()).hexdigest()
        output.append(row)
    return dict(batches=output,checkpoints=checkpoints,warnings=warnings)

def capture():
    timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    accounting=subprocess.check_output(['sacct','-n','-X','-P','-j',','.join(j[0] for j in MODELS.values()),'-o','JobID,State,ExitCode,Elapsed'],text=True)
    statuses={r.split('|')[0]:r.split('|')[1:] for r in accounting.strip().splitlines()}
    runs={}
    for model,(job,layers) in MODELS.items():
        result=parse_log(ROOT/f'{model}-{job}.out')
        errors=ROOT/f'{model}-{job}.err'
        error_text=errors.read_text(errors='replace') if errors.exists() else ''
        result.update(job=job,slurm=statuses.get(job),error_tail=error_text[-1800:] if any(s in error_text for s in ('Traceback','Error:','Killed','CANCELLED')) else '')
        result['last_completed_batch']=max([r['batch'] for r in result['batches'] if r['complete']],default=0)
        result['current_batch']=result['batches'][-1] if result['batches'] else None
        runs[model]=result
    return dict(observed_at_kst=timestamp,runs=runs,free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)

def summarize(state):
    return {model:dict(job=r['job'],slurm=r['slurm'],last_completed_batch=r['last_completed_batch'],
                     current_fit_count=r['current_batch']['fit_count'] if r['current_batch'] else 0,
                     latest_layers=r['current_batch']['layers'] if r['current_batch'] else {},
                     checkpoints=[c['completed_batches'] for c in r['checkpoints']],warnings=r['warnings'],error=bool(r['error_tail'])) for model,r in state['runs'].items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    folder=ROOT/'monitor';folder.mkdir(exist_ok=True);previous=None
    while True:
        state=capture();tmp=folder/'latest.json.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,folder/'latest.json')
        summary=summarize(state)
        signature=json.dumps({m:(r['slurm'][:2] if r['slurm'] else None,r['last_completed_batch'],r['checkpoints'],r['warnings'],r['error']) for m,r in summary.items()},sort_keys=True)
        if signature!=previous:
            event=dict(observed_at_kst=state['observed_at_kst'],runs=summary)
            with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True);previous=signature
        if not args.watch:break
        if all(r['slurm'] and r['slurm'][0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in summary.values()):break
        time.sleep(30)
