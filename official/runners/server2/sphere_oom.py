"""Exact one-cell cold SPHERE repair; same native runner, assets and public scorer."""
import argparse
from pathlib import Path
from official.experiments.prepare import write_new, file_sha, digest
from official.runners.server1.common import read, verify

INSTRUCTION='USER-SH2-QWEN-ZSRE-SPHERE-OOM-RERUN-20261010-R1'
CELL='qwen25-zsre-sphere'
REPAIRED_SHA='29ad88fe08c909242236405a66f011527112ec1710c5045775586c5bcb4f6692'

def validate(root):
    root=Path(root);p=read(root/'sphere-repair.json')
    assert p['instruction_id']==INSTRUCTION and p['old_job_id']=='62087'
    assert p['cell']==CELL and p['cold_start'] is True
    for m in read(root/'input-lock.json')['members']:verify(m)
    source=read(root/'source-lock.json')
    for m in source['members']:
        assert file_sha(root/'source'/m['relative'])==m['sha256']
    assert file_sha(Path(__file__).resolve().parents[2]/'baselines/easyedit/models/SPHERE/SPHERE_main.py')==REPAIRED_SHA
    return p

def bind_context(root,config,native,tokenizer,*,resume):
    from official.runners.fe_author_history import load_native_context
    p=validate(root)
    assert not resume and config['method']=='SPHERE' and config['dataset']=='zsre'
    assert config['config_sha256']==p['original_config_sha256']
    assert native.context_snapshot() is None
    assets=read(verify(p['normalized_assets']))
    contexts=load_native_context(dict(model='qwen25',edit_seed=0,native_context=p['native_context']),assets,tokenizer)
    assert digest(contexts)==p['native_context']['canonical_sha256']
    native.restore_context(contexts)

def run(root):
    from official.runners.server2.qwen_pipeline import child, run as pipeline
    root=Path(root);p=validate(root)
    assert not (root/'runs'/CELL).exists(),'NO_DUPLICATE_COLD_MAIN'
    w0=root/'shared-w0/qwen25-zsre'
    assert not w0.exists(),'NO_IMPLICIT_RETRY'
    child(root,'w0',w0,dataset='zsre',extra=('--public-zsre-w0',),label='zsre-public-W0')
    w0receipt=read(w0/'w0-receipt.json')
    ref=read(w0/'w0-zsre-reference.json')
    assert ref['schema']=='zsre-public-query-W0-observation-v1'
    assert w0receipt['factual']['work']['evaluation_profile']=='zsre-public-query-v1'
    pipeline(root,CELL)
    write_new(root/'completed.json',dict(instruction_id=INSTRUCTION,old_job_id='62087',
        status='W20_COMPLETE',terminal=str(root/'runs'/CELL/'terminal.json'),old_B9_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
