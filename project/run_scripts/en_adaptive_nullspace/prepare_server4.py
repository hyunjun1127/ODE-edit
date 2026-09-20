"""Create-once S4 bindings; reuse local immutable inputs, never transfer/raw edit.

This CPU admission step binds prior full-SHA evidence and current stat for large
assets, hashes the small source closure, and preserves all original locks.
It creates no edited model/history checkpoint and never initializes CUDA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/data/janghj/ODE-edit')
PRIOR = ROOT/'local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/execution.lock.json'
BASE = ROOT/'local/en-adaptive-nullspace/20260920-v1/server4-migration-r1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def prepare(destination):
    destination=Path(destination).absolute()
    if not destination.is_relative_to(BASE):raise ValueError('S4_TASK_SCOPE')
    prior=json.loads(PRIOR.read_text())
    if sha(PRIOR)!='6a14ebaf32549cc9479f2d112ba1954b06ef00380fffc1091cd70090f9098f61':raise ValueError('PRIOR_ASSET_LOCK')
    from project.run_scripts.en_execution_reuse.model import verify_large_asset_stats
    verify_large_asset_stats(prior)
    inventory=[]
    for row in prior['prior_large_asset_binding']:
        p=Path(row['prior']['path']);s=p.stat()
        inventory.append(dict(model='llama3-8b-inst',consumed_path=str(p),size=s.st_size,
            sha256=row['prior']['sha256'],observed_sha256=row['prior']['sha256'],
            stable_stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],
            kind='weight_shard' if p.suffix=='.safetensors' else 'input',
            verification='PRIOR_FULL_SHA_PLUS_UNCHANGED_STAT'))
    repo=Path(__file__).resolve().parents[3]
    authority=json.loads((repo/'audits/global/2026-09-20-sh3-en-adaptive-nullspace/authority-manifest.json').read_text())
    for row in authority['members']:
        p=repo/row['path']
        if p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:raise ValueError('AUTHORITY_SHA:'+str(p))
    contract=json.loads(Path(__file__).with_name('metrics-contract.json').read_text())
    paths={}
    for name,expect in contract['source_files'].items():
        p=(repo/'project/run_scripts/baseline_mechanism_first/evaluation.py' if name=='contexts/evaluation.py'
           else Path(prior['helper_scripts_root'])/name.split('evaluator_source/project/run_scripts/')[1])
        if sha(p)!=expect['sha256'] or p.stat().st_size!=expect['bytes']:raise ValueError('EVALUATOR_CLOSURE:'+str(p))
        paths[name]=str(p)
    small=[Path(prior['config4']),Path(prior['cold_capsule']['path'])]
    small+=sorted(Path(prior['blue_root']).rglob('*.py'))
    small+=list(map(Path,paths.values()))
    source=[dict(destination=str(p),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(small))]
    generated_ready=json.loads(Path(prior['generated_ready']['path']).read_text())
    gm=generated_ready['manifest'];generated=Path(gm['path'])
    if sha(generated)!=gm['sha256'] or sha(prior['reference_inputs']['path'])!=prior['reference_inputs']['sha256']:raise ValueError('REFERENCE_MANIFEST_IDENTITY')
    gp=json.loads(generated.read_text())
    # Derive paths from the sealed manifest rather than scanning other tasks.
    import torch
    basis=torch.load(prior['P_star_basis']['path'],weights_only=True,mmap=True,map_location='cpu')
    if list(basis['basis'].shape)!=[14336,14326] or basis['basis'].dtype!=torch.float64:raise ValueError('BASIS_SCHEMA')
    if sha(prior['P_star_basis']['path'])!=prior['P_star_basis']['sha256']:raise ValueError('BASIS_SHA')
    import numpy,scipy,transformers
    if torch.cuda.is_initialized():raise RuntimeError('CPU_PREFLIGHT_INITIALIZED_CUDA')
    inputs=destination/'inputs'
    write(inputs/'asset-inventory.json',inventory)
    write(inputs/'source-allowlist.json',source)
    write(inputs/'evaluator-map.json',paths)
    manifest=dict(node='server4',python=sys.executable,python_version=list(sys.version_info[:3]),
        transformers_import=str(Path(transformers.__file__).resolve()),
        dataset=dict(dataset_sha256='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1',ordered_root='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'),
        dataset_root=prior['dataset_root'],context_capsule=prior['cold_capsule']['path'],
        native_source_sha256=prior['editor_sha256'],native_root=prior['blue_root'],native_config=prior['config4'],
        asset_inventory=str(inputs/'asset-inventory.json'),source_allowlist=str(inputs/'source-allowlist.json'),
        models={'llama3-8b-inst':dict(revision=prior['model_revision'],snapshot=prior['snapshot'],experiment_ready=True,assets={'projector':{'path':prior['projector']}})},
        generated_root=str(generated.parent),reference_inputs=prior['reference_inputs']['path'],basis_pt=prior['P_star_basis']['path'],
        provenance=dict(prior_asset_lock=str(PRIOR),prior_asset_lock_sha256=sha(PRIOR),prior_waivers_inherited=False,
            generated_manifest=gm,basis=prior['P_star_basis'],SH3_source='43904c13def0aa06bb37dcd7574df5df101db135',
            SH3_actual_GPU_jobs=0,reference_inverse_copy_bytes=0),
        versions=dict(torch=str(torch.__version__),numpy=numpy.__version__,scipy=scipy.__version__,transformers=transformers.__version__),
        save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',maximum_batch=3)
    write(inputs/'manifest.json',manifest)
    st=os.statvfs(destination)
    write(destination/'cpu-binding.json',dict(status='INPUT_BINDING_ONLY_NOT_MODEL_PASS',authority=authority,
        small_source_members=len(source),large_binding_count=len(inventory),basis_sha256=sha(prior['P_star_basis']['path']),
        disk_free_bytes=st.f_bavail*st.f_frsize,inodes_free=st.f_favail,
        source_inverse_copy=0,teacher_inverse_copy=0,CUDA_initialized=torch.cuda.is_initialized(),
        timestamp=time.time(),noCP=True))
    print(json.dumps(dict(manifest=str(inputs/'manifest.json'),source_members=len(source),free_GiB=st.f_bavail*st.f_frsize/2**30)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);prepare(p.parse_args().destination)
