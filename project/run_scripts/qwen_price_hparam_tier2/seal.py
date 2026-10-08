"""Freeze a clean committed source plus exact existing CPU preparation, no submit."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path
from project.run_scripts.jlz_interference_l1.cap_common import member,verify,require
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard
from . import TASK,NONCE


def seal(root,preparation):
    root=Path(root).resolve();repo=Path.cwd().resolve()
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'CLEAN_SOURCE_REQUIRED')
    require(not root.exists(),'NEW_ATTEMPT_ONLY')
    guard(root,12*1024**3);root.mkdir(parents=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    archive=root/'source.tar.gz'
    subprocess.run(['git','archive','--format=tar.gz','-o',str(archive),source],check=True)
    dest=root/'source';dest.mkdir()
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    tracked=subprocess.check_output(['git','ls-files','-z'],text=True).split('\0')
    source_members=[member(dest/p) for p in tracked if p and (dest/p).is_file()]
    prep=json.loads(Path(preparation).read_text())
    inputs=[member(preparation)]
    for key in ('parent_config','dispatch_manifest','stream','contexts','smoke_reference','resolved','selection_rules','observer_identity'):
        verify(prep[key]);inputs.append(prep[key])
    parent=json.loads(verify(prep['parent_config']).read_text())['cells']['QWEN_M1_CAP075']
    for row in parent['runtime']['source_members']:verify(row);inputs.append(row)
    inputs.append(member(parent['tracking']['env_file']))
    write(root/'selection-rules.json',json.loads(verify(prep['selection_rules']).read_text()))
    launchers=[]
    for phase in ('warmup','sweep'):
        text='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export PYTHONDONTWRITEBYTECODE=1','export HF_HUB_OFFLINE=1','export TRANSFORMERS_OFFLINE=1',
            'export TOKENIZERS_PARALLELISM=false','export OMP_NUM_THREADS=8','export MKL_NUM_THREADS=8',
            'export OPENBLAS_NUM_THREADS=8',f'export PYTHONPATH={dest}',f'export QWEN_TUNING_SOURCE={source}',
            f'cd {dest}',
            f'exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.qwen_price_hparam_tier2.pipeline --attempt {root} --phase {phase}',''])
        launcher=root/(phase+'.sh')
        with launcher.open('x') as f:f.write(text)
        launchers.append(member(launcher))
    write(root/'execution.lock.json',dict(task_id=TASK,instruction_id=NONCE,source_commit=source,
        archive=member(archive),source_members=source_members,input_members=inputs,launchers=launchers,
        preparation=member(preparation),selection_rules=member(root/'selection-rules.json'),
        storage_reserve_bytes=12*1024**3,
        storage_budget=dict(max_fit_batches=30,max_candidate_records=750,candidate_max_bytes=1024**2,
            max_metric_chunks=200,metric_chunk_max_bytes=3*1024**2,spool_and_console_reserve_bytes=2*1024**3,
            atomic_error_other_reserve_bytes=8*1024**3,no_durable_tensors=True),
        resources=dict(warmup_GPUs=1,sweep_GPUs=2,per_arm_GPUs=1,CPU_per_arm=8,
            host_memory_MiB_per_GPU=59392,hard_memory_MiB_per_GPU=60416,
            warmup_wall='04:00:00',sweep_wall='24:00:00',model_peak_VRAM_GiB_estimate=74.62,
            per_arm_host_GiB_estimate=49.74,scientific_batch_split=False),
        no_checkpoint=True,archive_policy='NOT_APPLICABLE_NO_CHECKPOINT',automatic_retry=False),limit=8*1024**2)
    print(json.dumps(dict(root=str(root),source=source,files=len(source_members),status='SOURCE_FROZEN_NOT_SUBMITTED')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True)
    a=p.parse_args();seal(a.attempt,a.preparation)
