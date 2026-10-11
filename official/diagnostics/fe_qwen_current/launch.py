"""CPU identity checks then exec unchanged native main entrypoint/model loading."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
from official.experiments.prepare import digest, write_new
from official.runners.server1.common import read, verify
from official.diagnostics.fe_qwen_current.contract import SPEC
from official.diagnostics.fe_qwen_current.storage import admit_remaining

def launch(config,lockfile):
    lock=read(lockfile);c=read(config)
    if c['config_sha256']!=digest({k:v for k,v in c.items() if k!='config_sha256'}):raise ValueError('CONFIG_SHA')
    if c['source']!=lock['source_commit'] or c['selection_contract_sha256']!=digest(SPEC) or c['server']!=lock['server']:
        raise ValueError('SOURCE_SPEC_HOST')
    if not Path(__file__).resolve().is_relative_to(Path(lock['source_directory']).resolve()):raise ValueError('RUNTIME_SOURCE')
    for m in lock['source_members']+lock['author_members']+lock['configs']+lock['model_members']:
        verify(m)
    verify(lock['selection_lock']);verify(c['stream']);verify(c['native_data'])
    for m in c['C0']:
        verify(m['original'])
        if Path(m['link']).resolve()!=Path(m['original']['path']).resolve():raise ValueError('C0_LINK')
    assets=read(verify(c['model_assets']));model=assets['model'];revision=model['identity']['revision']
    if revision!='a09a35458c702b33eeacc393d103063234e8bc28':raise ValueError('QWEN_REVISION')
    hf=Path(lock['hf_cache'])/'models--Qwen--Qwen2.5-7B-Instruct'
    if (hf/'refs/main').read_text().strip()!=revision:raise ValueError('HF_DEFAULT_REF')
    if (hf/'snapshots'/revision).resolve()!=Path(model['snapshot']).resolve():raise ValueError('HF_SNAPSHOT_ALIAS')
    import torch,transformers
    if torch.__version__!=lock['runtime']['torch'] or transformers.__version__!=lock['runtime']['transformers']:
        raise ValueError('RUNTIME_VERSION')
    base=Path(config).resolve().parents[1];out=Path(c['output'])
    lockpath=Path(c['checkpoint_lock']);lockpath.parent.mkdir(parents=True,exist_ok=True)
    if lockpath.is_symlink():raise ValueError('HOST_LOCK_ALIAS')
    with lockpath.open('a+b') as storage_lock:
        fcntl.flock(storage_lock.fileno(),fcntl.LOCK_EX)
        stat=os.statvfs(base)
        budget=admit_remaining(c,[read(m['path']) for m in lock['configs']],stat.f_bavail*stat.f_frsize)
        write_new(out/'startup-admission.json',budget)
        fcntl.flock(storage_lock.fileno(),fcntl.LOCK_UN)
    env=dict(os.environ,FE_WORKING_LOCK=str(Path(lockfile).resolve()),
        PYTHONPATH=lock['source_directory'],PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE=lock['hf_cache'],TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=str(lock['cpus']),MKL_NUM_THREADS=str(lock['cpus']),OPENBLAS_NUM_THREADS=str(lock['cpus']))
    args=[lock['python'],'-B','-u','main.py','llms=qwen2.5-7b','algs=memit',
        f"algs.L2={c['L2']}",'bs=100','num_edits=2000','seed=0','data=heldout-native.json',
        f'data_dir={base}/data',f'cache_dir={base}/cache',f'results_dir={out}/unused-native-results',
        f'zs_cache_dir={out}/unused-target-cache','save_name=not-exported','load_name=not-loaded',
        '+diagnostic_z_reference=batch_entry',f'+working_config={Path(config).resolve()}',
        f'hydra.run.dir={out}/hydra','hydra.job.chdir=False']
    os.chdir(lock['author_path']);os.execve(lock['python'],args,env)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('lock');a=p.parse_args();launch(a.config,a.lock)
