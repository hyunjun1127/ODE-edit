"""Freeze a committed bounded import closure; never snapshot model/edit state."""
import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from .common import ROOT,INSTRUCTION,require,write,member,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    require(not a.attempt.exists(),'ATTEMPT_ALREADY_EXISTS');a.attempt.mkdir(parents=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain','--','project/run_scripts/jlz_realized_subject'],cwd=ROOT,text=True),'UNCOMMITTED_PRODUCTION_SOURCE')
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',source,'project/run_scripts/jlz_realized_subject'],cwd=ROOT,text=True).splitlines()
    files += ['project/run_scripts/jlz_writer_coupled/'+n for n in ['__init__.py','common.py','physical.py','entry.py','geometry.py','inputs.py']]
    files += ['project/run_scripts/jlz_pilot/__init__.py','project/run_scripts/jlz_pilot/prompts.py']
    # Include real regular package initializers when tracked; namespace dirs need none.
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',source],cwd=ROOT,text=True).splitlines())
    files += [x for x in ['project/__init__.py','project/run_scripts/__init__.py'] if x in tracked]
    frozen=a.attempt/'source';members=[]
    for rel in sorted(set(files)):
        data=subprocess.check_output(['git','show',source+':'+rel],cwd=ROOT)
        path=frozen/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);members.append(member(path))
    config=json.loads(a.config.read_text());(a.attempt/'config.json').write_bytes(a.config.read_bytes())
    archive=a.attempt/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for m in members:tf.add(m['path'],arcname=str(Path(m['path']).relative_to(frozen)),recursive=False)
    runtime=[]
    for name in ['transformers.models.llama.modeling_llama','transformers.modeling_attn_mask_utils','torch.utils.checkpoint','torch.optim.adam']:
        mod=importlib.import_module(name);runtime.append(member(mod.__file__))
    python='/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python'
    lock=dict(schema=1,instruction=INSTRUCTION,source_commit=source,source_archive=member(archive),source_root=str(frozen),
        config_sha256=sha(a.attempt/'config.json'),source_members=members,runtime_sources=runtime,
        native_reference=config['native_reference'],resource=config['resource'],noCP=True,exact_resume='NOT_AVAILABLE')
    write(a.attempt/'execution.lock.json',lock)
    launchers={}
    for stage in ('q1','A','B','collector'):
        module='collect' if stage=='collector' else 'run'
        argv=[python,'-u','-m','project.run_scripts.jlz_realized_subject.'+module,'--attempt',str(a.attempt),'--config',str(a.attempt/'config.json')]
        if stage!='collector':argv+=['--phase','q1' if stage=='q1' else 'main']+([] if stage=='q1' else ['--arm',stage])
        text='#!/bin/bash\nset -euo pipefail\n'
        env=dict(PYTHONPATH=str(frozen),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ODEEDIT_SOURCE_COMMIT=source)
        for k,v in env.items():text+='export '+k+'='+shlex.quote(v)+'\n'
        text+='cd '+shlex.quote(str(frozen))+'\nexec '+shlex.join(argv)+'\n'
        path=a.attempt/(stage+'.sh');path.write_text(text);path.chmod(0o755)
        launchers[stage]=dict(file=member(path),argv=argv,environment=env)
    write(a.attempt/'launchers.json',launchers)
    print(json.dumps(dict(source=source,lock=member(a.attempt/'execution.lock.json'),launchers=list(launchers))))


if __name__=='__main__':main()
