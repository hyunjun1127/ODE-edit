"""Create a new immutable source archive/closure/lock; no submission."""
import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .config import ROOT, check_lock
from .preflight import resource_snapshot
from project.run_scripts.single_layer_edit_preserving_correction.common import member, write, digest


def freeze(repo, plan, attempt):
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('CLEAN_EXECUTION_COMMIT_REQUIRED')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()
    root=ROOT/'T0'/attempt;root.mkdir(parents=True,exist_ok=False)
    archive=root/'source.tar.gz'
    tar=subprocess.check_output(['git','archive',commit],cwd=repo)
    with archive.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(tar)
    source=root/'source';source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:tf.extractall(source,filter='data')
    # Entire project/scripts closure is compact source, never task raw.
    members=[member(p) for top in ('project/run_scripts','scripts') for p in sorted((source/top).rglob('*'))
             if p.is_file() and p.suffix in ('.py','.json','.sbatch','.sh')]
    lock=json.loads(Path(plan).read_text())
    lock.update(output=str(root/'output'),status='FROZEN_EXECUTION_LOCK',phase='HOOK',
        execution=dict(commit=commit,tree=tree,archive=member(archive),source=str(source),members=members),
        resource_before_freeze=resource_snapshot())
    lock['lock_identity']=digest(lock);check_lock(lock)
    result=write(root/'execution.lock.json',lock)
    print(json.dumps(dict(lock=result,execution={k:v for k,v in lock['execution'].items() if k!='members'})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();freeze(a.repo,a.plan,a.attempt)
