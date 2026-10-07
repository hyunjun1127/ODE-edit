"""Create-once source archive/launchers; no scheduler calls."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from .run import TASK,NONCE


def freeze(attempt):
    from project.run_scripts.jlz_interference_l1.cap_common import member
    root=Path.cwd();source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('SOURCE_WORKTREE_NOT_CLEAN')
    config=json.loads((attempt/'config.json').read_text())
    evidence=[]
    for name in ('token-binding.json','source-review.json'):
        p=root/'audits/servers/server4'/TASK/name
        value=json.loads(p.read_text())
        if name=='token-binding.json' and value['status']!='PASS_SAVED_PACK_BINDING':
            raise RuntimeError('SAVED_PACK_GATE')
        evidence.append(member(p))
    archive=attempt/'source.tar';dest=attempt/'source'
    if archive.exists() or dest.exists() or (attempt/'execution.lock.json').exists():
        raise RuntimeError('SOURCE_FREEZE_ALREADY_EXISTS')
    subprocess.run(['git','archive','--format=tar','-o',str(archive),source,'project/run_scripts','scripts'],check=True)
    dest.mkdir()
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    members=[member(f) for f in sorted(dest.rglob('*')) if f.is_file()]
    launchers=[]
    for role in list(config['cells'])+['collector']:
        cmd=('/data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.price_ridge_m1_m3.'+
             ('collect' if role=='collector' else 'run')+' --attempt '+str(attempt)+
             ('' if role=='collector' else ' --cell '+role))
        text='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export PYTHONPATH='+str(dest),'export PYTHONDONTWRITEBYTECODE=1',
            'export OMP_NUM_THREADS=8','export OPENBLAS_NUM_THREADS=8','export MKL_NUM_THREADS=8',
            'export TOKENIZERS_PARALLELISM=false','export HF_HUB_OFFLINE=1','export TRANSFORMERS_OFFLINE=1',
            'export NLTK_DATA=/data/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/nltk_data',
            'export PRICE_M1_M3_SOURCE='+source,'cd '+str(dest),'exec '+cmd,''])
        path=attempt/(role+'.sh')
        with path.open('x') as f:f.write(text)
        subprocess.run(['bash','-n',str(path)],check=True);launchers.append(member(path))
    lock=dict(task_id=TASK,instruction_id=NONCE,source_commit=source,archive=member(archive),
        config_sha256=member(attempt/'config.json')['sha256'],source_members=members,
        input_members=config['input_members']+evidence,launchers=launchers,project_gpu_cap=2,
        CPU=8,host_memory_MiB=59392,hard_memory_MiB=60416,new_W0_allowed=False,noCP=True)
    with (attempt/'execution.lock.json').open('x') as f:json.dump(lock,f,ensure_ascii=False,sort_keys=True,indent=2)
    print(json.dumps(dict(source=source,archive=lock['archive'],lock=member(attempt/'execution.lock.json'),source_files=len(members))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    freeze(p.parse_args().attempt.resolve())
