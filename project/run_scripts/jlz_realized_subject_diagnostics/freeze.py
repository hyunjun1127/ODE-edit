"""Committed source freeze: c2d5fb10 closure plus explicit callback/new namespace."""
import argparse
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    require(a.attempt.parent==LOCAL and not a.attempt.exists(),'CREATE_ONCE_TASK_ATTEMPT')
    require(not subprocess.check_output(['git','status','--porcelain','--','project/run_scripts/jlz_realized_subject_diagnostics','project/run_scripts/jlz_realized_subject/optimize.py'],cwd=ROOT),'UNCOMMITTED_SOURCE')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    a.attempt.mkdir();frozen=a.attempt/'source'
    env=json.loads((ROOT/ENVELOPE).read_text());receipt=json.loads((ROOT/env['bindings']['source_member_manifest']).read_text())
    members=[];lineage=[]
    refs={r['path']:FROZEN for r in receipt['source_members']}
    refs['project/run_scripts/jlz_realized_subject/optimize.py']=source
    for rel in subprocess.check_output(['git','ls-tree','-r','--name-only',source,'project/run_scripts/jlz_realized_subject_diagnostics'],cwd=ROOT,text=True).splitlines():refs[rel]=source
    for rel,rev in sorted(refs.items()):
        data=subprocess.check_output(['git','show',rev+':'+rel],cwd=ROOT);dst=frozen/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(data)
        m=member(dst);members.append(m);lineage.append(dict(relative_path=rel,commit=rev,**m))
    for name in ('config.json','lookup-local.json','full-read.json'):
        dst=a.attempt/name
        with dst.open('xb') as f:f.write((a.preparation/name).read_bytes())
        members.append(member(dst))
    # Source/input authorities included by exact hashes, not mutated.
    members+=[member(ROOT/ENVELOPE),member(ROOT/'control/experiment-exceptions/jlz-v10-a-b1-diagnostics-sh1-20261003.json')]
    members += [member(ROOT/r['path']) for r in env['bindings']['files']]
    archive=a.attempt/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for rel in sorted(refs):tf.add(frozen/rel,arcname=rel,recursive=False)
    config=json.loads((a.attempt/'config.json').read_text())
    lock=dict(instruction=NONCE,source=source,tree=tree,frozen_science=FROZEN,source_root=str(frozen),
        members=members,runtime_sources=config['runtime_sources'],source_lineage=lineage,archive=member(archive),
        source_diff='only optimize callback; no native math/schedule edit',resources=config['resources'],
        full_program='fixed-candidate qualification + one B1 25/24 + RAM D1/D2 + CPU collector',
        no_checkpoint=True,exact_resume='NOT_AVAILABLE',other_tasks='STOP_UNCHANGED')
    write(a.attempt/'execution.lock.json',lock)
    launchers={}
    for stage in ('gpu','collector'):
        argv=[config['python'],'-u','-m','project.run_scripts.jlz_realized_subject_diagnostics.'+('run' if stage=='gpu' else 'collect'),'--attempt',str(a.attempt)]
        exports=dict(PYTHONPATH=str(frozen),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ODEEDIT_SOURCE_COMMIT=source)
        text='#!/bin/bash\nset -euo pipefail\n'
        text+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())+'\n'
        text+='cd '+shlex.quote(str(frozen))+'\nexec '+shlex.join(argv)+(' "$@"' if stage=='collector' else '')+'\n'
        path=a.attempt/(stage+'.sh')
        with path.open('x') as f:f.write(text)
        path.chmod(0o755);launchers[stage]=dict(file=member(path),argv=argv,exports=exports)
    write(a.attempt/'launchers.json',launchers)
    print(json.dumps(dict(source=source,tree=tree,lock=member(a.attempt/'execution.lock.json'))))


if __name__=='__main__':main()
