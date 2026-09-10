"""Small publication provenance closure, one-shot own-job accounting and rehash."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from .transfer import DEST,sha
from .analysis import read
from project.run_scripts.single_layer_cumulative_risk.records import save

REPO=Path(__file__).resolve().parents[3]

def prepare(root,runs):
    import platform
    import numpy
    import matplotlib
    import torch
    assert not torch.cuda.is_initialized()
    save(root/'analysis-runtime.json',dict(python=platform.python_version(),numpy=numpy.__version__,
         matplotlib=matplotlib.__version__,torch=torch.__version__,cuda_initialized=False,
         model_calls=0,bootstrap_seed=20260911,bootstrap_replicates=2000))
    jobs=[str(read(r/'runtime.json')['job']) for r in runs]
    assert len(jobs)==len(set(jobs))==4
    result=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES','--parsable2','--noheader'],text=True)
    accounting=[]
    for line in result.splitlines():
        job,state,code,elapsed,tres=line.split('|')[:5]
        if job not in jobs:continue
        assert state=='COMPLETED' and code=='0:0' and 'gres/gpu=1' in tres
        accounting.append(dict(job=job,state=state,exit=code,elapsed_seconds=int(elapsed),allocation=tres))
    assert len(accounting)==4
    logs=[]
    for job in jobs:
        for p in sorted((DEST/'logs').glob('*-'+job+'.*')):
            assert p.suffix in ('.out','.err') and p.is_file() and not p.is_symlink()
            logs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    assert len(logs)==8
    save(root/'job-accounting.json',dict(jobs=accounting,total_gpu_hours=sum(r['elapsed_seconds'] for r in accounting)/3600,logs=logs,other_job_queries=0))
    members=[]
    test=REPO/'audits/servers/server2/2026-09-11-blue-l4-progress-barrier/focused-test-receipt.json'
    with (root/test.name).open('xb') as f:f.write(test.read_bytes())
    members.append(dict(path=test.name,source_path=str(test),sha256=sha(test)))
    for name in ['input.lock.json','execution.lock.json','science.lock.json','resource.lock.json','transfer-receipt.json','native-internal-consistency.json']:
        p=DEST/name;target=root/name
        with target.open('xb') as f:f.write(p.read_bytes())
        members.append(dict(path=name,source_path=str(p),sha256=sha(p)))
    for r in runs:
        for name in ['runtime.json','calibration.json','application-gate.json','terminal.json']:
            p=r/name;target=root/(r.name+'-'+name)
            with target.open('xb') as f:f.write(p.read_bytes())
            members.append(dict(path=target.name,source_path=str(p),sha256=sha(p)))
    for p in (DEST/'geometry-analysis-v1').glob('*.csv'):
        with (root/p.name).open('xb') as f:f.write(p.read_bytes())
    sources=[]
    for p in sorted(Path(__file__).parent.glob('*.py')):
        sources.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)))
    save(root/'analysis-source-manifest.json',dict(members=sources,execution_source_head=read(DEST/'execution.lock.json')['source_head'],analysis_commit='BOUND_BY_CONTAINING_GIT_COMMIT',source_mixing=False))
    save(root/'provenance-copy-receipt.json',dict(members=members,raw_prompts=0,model_tensors=0,raw_logs=0,NO_BROADCAST_NOT_REQUIRED=True))

def verify(root):
    manifest=read(root/'analysis-manifest.json');receipt=read(root/'rooted-receipt.json')
    assert sha(root/'analysis-manifest.json')==receipt['manifest_sha']
    assert sha(root/'diagnostic-report-ko.md')==receipt['report_sha']
    expected={r['path'] for r in manifest['members']}|{'analysis-manifest.json','rooted-receipt.json'}
    assert {p.name for p in root.iterdir()}==expected
    for m in manifest['members']:
        p=root/m['path'];assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],p
        assert p.suffix in {'.csv','.json','.md','.png'},p
    for m in read(root/'analysis-source-manifest.json')['members']:
        assert sha(REPO/m['path'])==m['sha256'],m['path']
    print('PUBLICATION_FULL_REHASH_PASS',len(manifest['members']),receipt['report_sha'])

def reproduce(root):
    target=DEST/'analysis/png-reproduction-v1';target.mkdir(parents=True,exist_ok=False)
    excluded={'diagnostic-report-ko.md','trajectory.png','observed-tradeoff.png','middle-auxiliary.png','analysis-manifest.json','rooted-receipt.json'}
    for p in root.iterdir():
        if p.name not in excluded:shutil.copyfile(p,target/p.name)
    subprocess.run(['/mnt/raid5/janghj/EasyEdit/.venv/bin/python','-m','project.run_scripts.blue_l4_progress_barrier.report','--report',str(target)],cwd=REPO,check=True)
    images=[]
    for name in ['trajectory.png','observed-tradeoff.png','middle-auxiliary.png']:
        assert sha(root/name)==sha(target/name),name
        images.append(dict(name=name,sha256=sha(root/name)))
    save(DEST/'analysis/png-reproduction-receipt.json',dict(status='EXACT_PNG_REPRODUCTION_PASS',images=images,model_calls=0))
    print('EXACT_PNG_REPRODUCTION_PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','verify','reproduce']);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--runs',nargs='*',type=Path);a=ap.parse_args()
    if a.action=='prepare':prepare(a.report,a.runs)
    elif a.action=='verify':verify(a.report)
    else:reproduce(a.report)

if __name__=='__main__':main()
