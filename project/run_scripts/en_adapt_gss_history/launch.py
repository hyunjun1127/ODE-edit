"""One-time in-program frozen input validation before independent cold load."""
import argparse
import json
import hashlib
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(attempt):
    attempt=Path(attempt).resolve()
    lock=json.loads((attempt/'execution.lock.json').read_text())
    if os.environ.get('SLURMD_NODENAME')!='ubuntu' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('SH3_SLURM_NODE_REQUIRED')
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=8 or int(os.environ.get('SLURM_MEM_PER_NODE','0'))!=121856:
        raise ValueError('EXACT_JOB_CPU_MEMORY_REQUIRED')
    if os.environ.get('SLURM_GPUS_ON_NODE','1')!='1' or os.environ.get('SAVE_CHECKPOINTS')!='false':
        raise ValueError('GPU1_NOCP_REQUIRED')
    source=Path(lock['execution']['source'])
    if Path(__file__).resolve().parents[3]!=source:
        raise ValueError('IMPORT_ESCAPED_FROZEN_SOURCE')
    for member in lock['execution']['members']:
        p=source/member['path']
        if p.stat().st_size!=member['bytes'] or sha(p)!=member['sha256']:
            raise ValueError('FROZEN_SOURCE_CHANGED:'+member['path'])
    for name,expected in lock['input_seals'].items():
        if sha(attempt/'execution-inputs'/name)!=expected:
            raise ValueError('SEALED_INPUT_CHANGED:'+name)
    refpath=Path(lock['reference_stat_binding']['path'])
    if sha(refpath)!=lock['reference_stat_binding']['sha256']:
        raise ValueError('REFERENCE_STAT_SEAL_CHANGED')
    for row in json.loads(refpath.read_text()):
        st=Path(row['path']).stat()
        if [st.st_dev,st.st_ino,st.st_mtime_ns]!=row['stat'] or st.st_size!=row['bytes']:
            raise ValueError('REFERENCE_MEMBER_STABLE_IDENTITY_CHANGED:'+row['path'])
    mp=Path(lock['map_seal']['path'])
    if sha(mp)!=lock['map_seal']['sha256']:
        raise ValueError('MAP_LOGICAL_SEAL_CHANGED')
    basis=json.loads(mp.read_text())['basis'];st=Path(basis['path']).stat()
    if st.st_size!=basis['size'] or [st.st_dev,st.st_ino,st.st_mtime_ns]!=basis['stat']:
        raise ValueError('PSTAR_BASIS_STABLE_IDENTITY_CHANGED')
    # No source archive/model/reference copies or whole-payload rehash per batch.
    return attempt/'execution-inputs/config.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path);a=p.parse_args()
    config=validate(a.attempt)
    from .runner import run
    run(config)
