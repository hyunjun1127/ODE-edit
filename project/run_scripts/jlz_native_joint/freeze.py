"""Freeze a committed scoped archive/config/launchers into a create-once attempt."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from .common import ROOT,LOCAL,INSTRUCTION,member,sha,write,require

PATHS=['project/run_scripts/jlz_native_joint','project/run_scripts/jlz_pilot/prompts.py',
       'project/run_scripts/jlz_pilot/__init__.py',
       'plans/global/2026-10-02-jlz-native-joint-v4',
       'messages/head/2026-10-02-jlz-v4-compute-r1-2k-sh4.json',
       'control/experiment-exceptions/jlz-v4-compute-r1-2k-20261002.json']
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'

def shell(source,module,args):
    return '\n'.join(['#!/usr/bin/env bash','set -euo pipefail','umask 077',
        'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
        'export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL',
        'export PYTHONPATH='+shlex.quote(str(source)),'cd '+shlex.quote(str(source)),
        'exec '+shlex.join([PYTHON,'-m',module,*args]),''])

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args();dest=args.attempt.resolve()
    require(dest.parent==LOCAL and not dest.exists(),'NEW_ATTEMPT_ONLY')
    require(not subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip(),'COMMIT_SOURCE_BEFORE_FREEZE')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    dest.mkdir();source=dest/'source';source.mkdir();archive=dest/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    with tarfile.open(archive) as t:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers()),'ARCHIVE_PATH')
        t.extractall(source,filter='data')
    config=json.loads((LOCAL/'preparation-v1/configuration.json').read_text())
    config.update(execution_source=commit,execution_tree=tree,archive_sha256=sha(archive))
    write(dest/'config.json',config)
    for phase,arm in [('shared','SHARED'),('pilot','JLZ_A'),('pilot','JLZ_B'),('main','JLZ_A'),('main','JLZ_B')]:
        name=phase+'-'+arm
        launch=shell(source,'project.run_scripts.jlz_native_joint.run',
            ['--config',str(dest/'config.json'),'--phase',phase,'--arm',arm,'--attempt',str(dest)])
        file=dest/(name+'.sh');file.write_text(launch);file.chmod(0o700)
    file=dest/'collector.sh'
    file.write_text(shell(source,'project.run_scripts.jlz_native_joint.collect',['--attempt',str(dest)]).rstrip()+' --jobs "$1"\n');file.chmod(0o700)
    write(dest/'execution.lock.json',dict(instruction_id=INSTRUCTION,source_commit=commit,source_tree=tree,
        archive=member(archive),execution_path=str(source),worktree=str(ROOT),config_sha256=sha(dest/'config.json'),
        source_members=[member(x) for x in sorted(source.rglob('*')) if x.is_file()],
        launchers=[member(x) for x in sorted(dest.glob('*.sh'))],resources=config['resources'],
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',owner='SH4',hostname='server4',
        no_other_task_resume=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        scope='shared W0 -> A/B pilot BS2 commit/B2entry -> A/B cold timing4 + fresh main BS100x20 -> CPU collector'))
    print(json.dumps(dict(attempt=str(dest),source=commit,lock=sha(dest/'execution.lock.json'),job_ids=[])))

if __name__=='__main__':main()
