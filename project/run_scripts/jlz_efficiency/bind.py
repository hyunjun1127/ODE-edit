"""Create-once CPU source/input/resource seal, no model or GPU initialization."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_pilot.run import MODEL
from project.run_scripts.jlz_pilot.prompts import prepare
from .measurement import sha
from .native import SOURCE
from .run import CONTEXT,DATA

ROOT=Path(__file__).resolve().parents[3]
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-efficiency/20261001-v1')
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1/attempt-r1/execution.lock.json')

def once(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def member(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='attempt-r1');parser.add_argument('--reuse',type=Path);args=parser.parse_args()
    assert '/' not in args.attempt
    design=ROOT/'plans/global/2026-10-01-jlz-efficiency-execution-v1';original=ROOT/'plans/global/2026-10-01-jlz-efficiency-v1'
    # USER1390b3b9 changes only resource admission + parallel_override path.
    assert sha(design/'contract.json')=='4485c11ffc9b0dbdf04dc5d672e284254fc15a37bc469271aa0166e8ffb48772'
    manifest=ROOT/'audits/global/2026-10-01-jlz-efficiency-sh1-dispatch/input-manifest.json'
    for row in json.loads(manifest.read_text())['members']:
        p=original/row['name'];assert p.stat().st_size==row['size'] and sha(p)==row['sha256']
    evidence=json.loads((original/'evidence.json').read_text())
    for name,digest in evidence['source_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(DATA/'counterfact.json')==evidence['data_sha256'] and sha(CONTEXT)==evidence['contexts_sha256']
    assert sha(OLD)=='3fb6601ac22352f214c17d3b0a3ebcf495c4897fba0e9523a9afa36bf8a396cf'
    old=json.loads(OLD.read_text());reused=[]
    for row in old['inputs']:
        p=Path(row['path']);assert p.is_file() and p.stat().st_size==row['bytes']
        reused.append(row|{'verification':'EXACT_EXISTING_INPUT_SEAL_REUSE + current existence/size; not repeated full model hash'})
    records=load_prefix(DATA,100);contexts=json.loads(CONTEXT.read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right';packing=[]
    for n in (4,100):
        spec=prepare(tok,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records[:n]],contexts,'cpu')
        packing.append(dict(requests=n,case_ids=[r['case_id'] for r in records[:n]],identity=spec['identity'],shape=list(spec['tokens']['input_ids'].shape),valid_tokens=int(spec['tokens']['attention_mask'].sum())))
    paths=set()
    for package in ('jlz_efficiency','jlz_sequential','jlz_pilot'):
        paths.update((ROOT/'project/run_scripts'/package).glob('*.py'))
    for name in evidence['source_sha256']:paths.add(ROOT/name)
    for name in ('blue_alphaedit_sequential_comparison/evaluation.py','blue_alphaedit_sequential_comparison/integrity.py','alphaedit_strength_neutral_barrier/contracts.py','low_cost_write_donor_pilot/fitting.py','single_layer_mechanism_first/z_hook.py'):
        paths.add(ROOT/'project/run_scripts'/name)
    paths.update([ROOT/'scripts/fixed_counterfact.py',DATA/'counterfact.json',CONTEXT,manifest])
    paths.update(original.iterdir());paths.update(design.iterdir())
    paths.update(SOURCE/p for p in ('AlphaEdit/compute_z.py','rome/repr_tools.py','hparams/AlphaEdit/Llama3-8B.json'))
    reuse=None
    if args.reuse:
        prior=args.reuse.resolve()
        assert prior==LOCAL/'attempt-r1','EXACT_REPAIR_SOURCE_ATTEMPT_ONLY'
        oldlock=prior/'execution.lock.json';assert sha(oldlock)=='56e9995639f56380adb9cb97d33e7875c68378b2b14e48f6a3ed1363e5c17757'
        olddata=json.loads(oldlock.read_text());assert olddata['source']=='bbe19548352e1cf543d54fdbb46ead3c6fb40436'
        locked={r['path']:r for r in olddata['members']}
        for name in ('core','native','reference','solver','budget','geometry','measurement','evaluation'):
            rel=f'project/run_scripts/jlz_efficiency/{name}.py';oldpath=str(Path(olddata['worktree'])/rel)
            assert sha(ROOT/rel)==locked[oldpath]['sha256'],'NUMERICAL_SOURCE_CHANGED '+rel
        selected=[prior/'output'/n for n in ('runtime.json','small4-entry.json','terminal.json','native-original.json','native-cached.json','native-batched.json')]
        selected+=sorted((prior/'output').glob('fixed-*.json'))+sorted((prior/'output').glob('short-*.json'))
        assert len(selected)==21 and not (prior/'output'/'kernels.json').exists()
        terminal=json.loads((prior/'output'/'terminal.json').read_text())
        assert terminal['budget']['small_total']==116 and terminal['budget']['separate']['B100']==0
        paths.update(selected);paths.add(oldlock)
        reuse=dict(root=str(prior),source=olddata['source'],lock_sha256=sha(oldlock),members=[member(p) for p in selected],
            fixed='REUSE32',short='REUSE84; only REF12 reconstructed for lost RAM R',native='REUSE_ALL; prior UNQUALIFIED kept',
            kernel='RUN_MISSING12_CASES; unsaved failed-attempt measurements cannot be recovered',
            probe_B100_observer='RUN_MISSING',old_GPU_seconds=401,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    free=shutil.disk_usage(ROOT).free;assert free>=8*1024**3
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'SOURCE_MUST_BE_COMMITTED_CLEAN'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    out=LOCAL/args.attempt;out.mkdir(parents=True,exist_ok=False)
    lock=dict(instruction='ODEEDIT-GH-SH1-JLZ-EFFICIENCY-20261001-R1',source=source,tree=tree,reference='7b4de31d4361caffbbb383fb0cadb8c5258e9cbe',worktree=str(ROOT),members=[member(p) for p in sorted(paths) if p.is_file()],
        reused_asset_seal=member(OLD),reused_assets=reused,packing=packing,old_job_mutation=False,server3_actions=0,
        resource=dict(GPU=1,CPU=8,mem_MiB=131072,hard_ceiling_MiB=183296,project_cap=2,task_cap=1,wall_hours=12,node='devbox',partition='gpu',QOS='lab_gpu_s1',export='NONE',requeue=0,hour_budget=None,wall_not_ETA=True,
            storage_reserve_bytes=8*1024**3,free_bytes=free,model_FP32_bytes=32121044992,selected_each_copy_bytes=5*4096*14336*4,
            GPU_estimated_peak_GiB=44,host_estimated_peak_GiB=40,host_H_cache_rollback_bytes=3*5*14336**2*4,
            one_FP64_square_bytes=14336**2*8,max_prefix_hidden_bytes=packing[-1]['shape'][0]*packing[-1]['shape'][1]*4096*4,
            outputs='scalar/per-case/timing/source only; no tensor persistence'),
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',broadcast='NO_BROADCAST_NOT_REQUIRED',source_manifest_scope='exact new namespace + read-only import closure; model seal reused')
    if reuse:lock['reuse']=reuse;lock['repair_instruction']='ODEEDIT-GH-SH1-JLZ-EFFICIENCY-KERNEL-REPAIR-20261001-R1'
    once(out/'execution.lock.json',lock)
    common=f'''#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
'''
    once(out/'gpu.sh',common+f'exec {PYTHON} -u -m project.run_scripts.jlz_efficiency.run --run {out}\n')
    once(out/'collector.sh',common+'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+f'exec {PYTHON} -u -m project.run_scripts.jlz_efficiency.collect --run {out} --job "${{1:?JOB_REQUIRED}}"\n')
    subprocess.run(['git','archive','--format=tar',f'--output={out}/source.tar',source],cwd=ROOT,check=True)
    print(json.dumps(dict(run=str(out),source=source,tree=tree,lock_sha256=sha(out/'execution.lock.json'),packing=packing,actual_GPU='NOT_RUN')))

if __name__=='__main__':main()
