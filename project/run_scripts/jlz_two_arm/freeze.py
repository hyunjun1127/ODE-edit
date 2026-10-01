"""Create-once local execution bundle from a committed, reviewed source tree."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from .common import ROOT, LOCAL, PYTHON, INSTRUCTION, member, sha, write, require

SOURCE_PATHS = [
    'project/run_scripts/jlz_two_arm', 'project/run_scripts/jlz_pilot',
    'project/run_scripts/jlz_sequential', 'project/run_scripts/jlz_efficiency',
    'project/run_scripts/blue_alphaedit_sequential_comparison',
    'project/run_scripts/alphaedit_strength_neutral_barrier',
    'project/run_scripts/ordered_response_barrier_ode',
    'project/run_scripts/memit_history_lifelong/hparams.json',
    'plans/global/2026-10-02-jlz-twoarm-bs100x20-execution-v1',
    'plans/global/2026-10-02-jlz-two-arm-v2',
    'messages/head/2026-10-02-jlz-twoarm-bs100x20-sh4.json',
    'messages/head/2026-10-02-jlz-twoarm-bs100x20-sh4-use-both-gpus.json',
    'audits/global/2026-10-02-jlz-twoarm-sh4-dispatch/input-manifest.json',
]

def shell(source, module, args):
    return '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail', 'umask 077',
        'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',
        'export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
        'export SLURM_EXPORT_ENV=ALL',
        'export PYTHONPATH='+shlex.quote(str(source)),
        'cd '+shlex.quote(str(source)),
        'exec '+shlex.join([PYTHON, '-m', module, *args]), ''])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reuse-prep',type=Path)
    p.add_argument('--preparation',type=Path,default=LOCAL/'preparation-v1');a=p.parse_args()
    root=a.run.resolve();require(root.parent==LOCAL and not root.exists(),'CREATE_ONCE_TASK_ATTEMPT')
    require(not subprocess.check_output(['git','status','--porcelain','--',*SOURCE_PATHS],cwd=ROOT,text=True).strip(),'SOURCE_NOT_COMMITTED')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    root.mkdir();archive=root/'source.tar';source=root/'source';source.mkdir()
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCE_PATHS],cwd=ROOT,check=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            require((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'SOURCE_ARCHIVE_MEMBER')
        t.extractall(source,filter='data')
    config=json.loads((a.preparation/'configuration.json').read_text())
    config.update(source_commit=commit,source_tree=tree,source=str(source),
                  stream_path=config['stream'],case_ids=[x for c in config['schedules']['main'] for x in c['ids']],
                  report_path=str(root/'report'),source_archive_sha256=sha(archive))
    options=config['baseline_pilot'];native=Path(options['native_root'])
    options.update(context_path=config['contexts'],context_sha256='33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e')
    options['configs']['MEMIT-H']=str(source/'project/run_scripts/memit_history_lifelong/hparams.json')
    native_paths=[]
    for directory in ('AlphaEdit','memit','rome','util','dsets'):
        native_paths.extend(sorted((native/directory).rglob('*.py')))
    native_paths.extend(native/n for n in ('globals.yml','LICENSE'))
    config['native_readonly_members']=[member(p) for p in native_paths]
    options['source_sha256']={n:sha(native/relative) for n,relative in {
        'memit.memit_seq_main':'memit/memit_seq_main.py',
        'memit.memit_main':'memit/memit_main.py',
        'AlphaEdit.AlphaEdit_main':'AlphaEdit/AlphaEdit_main.py'}.items()}
    config['inputs'] += [member(options['projector'])]
    config['inputs'] += [member(path) for path in options['configs'].values()]
    for baseline in config['baselines']:
        baseline['endpoint']='W20'
        commitpath=(Path(baseline['raw_path']).parent/'commit.json')
        baseline['commit_receipt_path']=str(commitpath)
        baseline['commit']=member(commitpath)
        config['inputs'].extend([baseline['receipt'],baseline['commit']])
    config['resources']['free_at_freeze_bytes']=shutil.disk_usage(root).free
    require(config['resources']['free_at_freeze_bytes']>=config['resources']['storage_reserve'],'STORAGE_RESERVE')
    if a.reuse_prep:
        from .prep_reuse import build
        config['prep_reuse']=build(a.reuse_prep,config)
    write(root/'config.json',config)
    for phase,arm in [('prep',None),('pilot','A'),('pilot','B'),('main','A'),('main','B')]:
        name=phase+('-'+arm if arm else '')
        args=['--run',str(root),'--phase',phase]+(['--arm',arm] if arm else [])
        path=root/(name+'.sh')
        path.write_text(shell(source,'project.run_scripts.jlz_two_arm.run',args));path.chmod(0o700)
    # Actual collector IDs are passed as one argv after all held jobs are registered.
    path=root/'collector.sh'
    content=shell(source,'project.run_scripts.jlz_two_arm.collect',['--run',str(root)])
    content=content.rstrip()+' --jobs "$1"\n'
    path.write_text(content);path.chmod(0o700)
    source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()]
    source_members+=config['native_readonly_members']
    write(root/'execution.lock.json',dict(instruction_id=INSTRUCTION,session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
          actor='SH4',hostname='server4',source_commit=commit,source_tree=tree,source_archive=member(archive),
          worktree=str(ROOT),execution_source=str(source),config_sha256=sha(root/'config.json'),
          source_members=source_members,launchers=[member(x) for x in sorted(root.glob('*.sh'))],
          preparation=member(a.preparation/'full-read.json'),resource=config['resources'],
          science_scope={'pilot':'A/B each BS4x2 cap32','main':'A/B each fresh BS100x20 cap120',
                         'shared_B100_technical_cap':6,'baseline_new':'same first8 BS4x2 each; full2k reused'},
          scheduling='shared prep then two independent pilot lanes then two independent main lanes',
          numerical_policy='RECORD_ONLY_USER_DIRECTED',numerical_certification='NOT_ESTABLISHED',
          checkpoint_saved=False,exact_resume='NOT_AVAILABLE'))
    print(json.dumps(dict(run=str(root),source=commit,lock=sha(root/'execution.lock.json'),jobs=[])))

if __name__=='__main__':main()
