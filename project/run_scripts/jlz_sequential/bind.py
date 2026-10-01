"""Offline CPU asset/source sealing and create-once local launchers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_pilot.run import MODEL,STATS,LAYERS,file_sha
from project.run_scripts.jlz_pilot.prompts import prepare

ROOT=Path(__file__).resolve().parents[3]
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1')

def member(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))

def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='attempt-r1');a=p.parse_args()
    out=LOCAL/a.attempt;out.mkdir(parents=True,exist_ok=False)
    design=ROOT/'plans/global/2026-10-01-jlz-sequential-bs100x10-v1'
    contract=json.loads((design/'contract.json').read_text())
    inputs=[member(contract['data'][key]) for key in ('path','contexts_path')]
    assert inputs[0]['sha256']==contract['data']['sha256'] and inputs[1]['sha256']==contract['data']['contexts_sha256']
    records=load_prefix(Path(contract['data']['path']).parent,1000)
    ids=[r['case_id'] for r in records]
    assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==contract['data']['first1000_case_ids_sha256']
    for cell in csv.DictReader((design/'cells.csv').open()):
        subset=ids[int(cell['start_inclusive']):int(cell['end_exclusive'])]
        assert hashlib.sha256(json.dumps(subset,separators=(',',':')).encode()).hexdigest()==cell['ordered_case_ids_sha256']
    contexts=json.loads(Path(contract['data']['contexts_path']).read_text())
    stats=[]
    for l in LAYERS:
        path=STATS/f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz'
        with np.load(path,allow_pickle=False) as z:
            raw=z['mom2.mom2'];count=int(z['mom2.count'])
            assert count>0 and raw.dtype==np.float32 and raw.shape==(14336,14336) and np.isfinite(raw).all()
            stats.append({'layer':l,'count':count,'shape':list(raw.shape),'dtype':str(raw.dtype),'finite':True})
        del raw
        inputs.append(member(path));print({'event':'C0_verified','layer':l},flush=True)
    index=MODEL/'model.safetensors.index.json'
    info=json.loads(index.read_text());shards=sorted(set(info['weight_map'].values()))
    model_members=[]
    for path in sorted(MODEL.iterdir()):
        if path.is_file() and (path.name in shards or path.suffix=='.json' or path.name in ('tokenizer.model',)):
            model_members.append(member(path));print({'event':'model_asset_verified','file':path.name},flush=True)
    assert all(any(Path(m['path']).name==name for m in model_members) for name in shards)
    inputs+=model_members
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokenizer.padding_side='right';tokenizer.pad_token=tokenizer.eos_token
    packing=[]
    for i in range(10):
        rs=[r['requested_rewrite']|{'case_id':r['case_id']} for r in records[i*100:(i+1)*100]]
        s=prepare(tokenizer,rs,contexts,'cpu')
        assert len(s['row_request'])==700 and len(s['key_lookup'])==600
        packing.append({'batch':i+1,'identity':s['identity'],'shape':list(s['tokens']['input_ids'].shape),
                        'actual_token_count':int(s['tokens']['attention_mask'].sum()),
                        'target_lengths':[len(r['target']) for r in s['specs']]})
    paths=[]
    for package in ('jlz_sequential','jlz_pilot'):
        paths.extend((ROOT/'project/run_scripts'/package).glob('*.py'))
    paths.extend([ROOT/'scripts/fixed_counterfact.py',
        ROOT/'project/run_scripts/blue_alphaedit_sequential_comparison/__init__.py',
        ROOT/'project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py',
        ROOT/'project/run_scripts/blue_alphaedit_sequential_comparison/integrity.py',
        ROOT/'project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py',
        ROOT/'project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py',
        design/'contract.json',design/'design-ko.md',design/'cells.csv'])
    free=shutil.disk_usage(LOCAL).free
    assert free>=16*1024**3,'INSUFFICIENT_STORAGE_RESERVE'
    maxlen=max(r['shape'][1] for r in packing)
    lock={'instruction':contract['instruction_id'],'source_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
          'worktree':str(ROOT),'source_members':[member(p) for p in sorted(set(paths))],
          'inputs':inputs,'stats':stats,'packing':packing,'contract':contract,
          'resource':{'gpu':1,'task_cap':1,'project_cap':2,'CPU':8,'mem_MiB':131072,'wall_hours':48,
                      'hour_budget':None,'node':'devbox','partition':'gpu','export':'NONE','requeue':False,
                      'storage_reserve_bytes':16*1024**3,'available_bytes_at_lock':free,
                      'max_prefix_hidden_bytes':700*maxlen*4096*4,
                      'history_RAM_bytes':5*14336**2*4,'rollback_history_RAM_bytes':5*14336**2*4,
                      'weights_model_FP32_bytes':32121044992,'selected_5_weights_bytes':5*4096*14336*4,
                      'one_FP64_square_bytes':14336**2*8,'ETA':'NOT_MEASURED_BEFORE_BS100_PREFLIGHT'},
          'checkpoint_saved':False,'exact_resume':'NOT_AVAILABLE','broadcast':'NO_BROADCAST_NOT_REQUIRED'}
    once(out/'execution.lock.json',lock)
    common=f'''#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
'''
    once(out/'gpu.sh',common+f'exec {PYTHON} -u -m project.run_scripts.jlz_sequential.run --run {out}\n')
    once(out/'collector.sh',common+f'exec {PYTHON} -u -m project.run_scripts.jlz_sequential.collect --run {out} --job "${{1:?GPU_JOB_REQUIRED}}"\n')
    print(json.dumps({'run':str(out),'lock_sha256':file_sha(out/'execution.lock.json'),'packing_maxlen':maxlen,
                      'input_files':len(inputs),'storage_free':free,'actual_GPU':'NOT_RUN'}))

if __name__=='__main__':main()
