"""CPU binding, freeze, held inspection and one-lane admission; no polling."""
import argparse
import copy
import datetime
import difflib
import os
from pathlib import Path
import shutil
import subprocess
from .binding import patch_source
from project.run_scripts.joint_multilayer_bs10.common import read,save,record,sha,digest,require,csv_rows
from project.run_scripts.joint_multilayer_bs10.server2_entry import NATIVE,PYTHON,DEPS,MODEL

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/native-weak-b010/20260930-v1')
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/attempt-s2-r1')
EVAL=OLD.parent/'final-full-eval-20260930-v1'
NONCE='ODEEDIT-GH-SH2-NATIVE-WEAK-B010-NLL1-20260930-R1'
JOINT_FILES=('common.py','runtime.py','observations.py','server2_entry.py','final_eval.py','review_b010.py','reduce.py')

def cmd(argv,cwd=None):return subprocess.check_output(argv,cwd=cwd,text=True).strip()

def stat_reuse(r):
    s=Path(r['path']).stat()
    require((s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'REUSE_STAT:'+r['path'])
    return dict(r,verification='PRIOR_FULL_SHA_PLUS_CURRENT_STAT')

def prepare(repo):
    repo=Path(repo).resolve()
    if ROOT.exists():require(not any(ROOT.iterdir()),'PREPARE_DESTINATION_NOT_EMPTY')
    else:ROOT.mkdir(parents=True)
    cfg=copy.deepcopy(read(OLD/'configuration.json'));oldlock=read(OLD/'execution.lock.json')
    require(sha(OLD/'configuration.json')==oldlock['configuration']['sha256'],'OLD_CONFIG_SHA')
    require(sha(repo/'messages/head/2026-09-30-native-weak-b010-sh2.md')=='96d747a96e681280a2fc5256dea5b5e53e860f3722fd79f9b62f14410dfdd957','NEW_AUTHORITY')
    authority=[]
    for r in cfg['authority']:
        # Prior manifest retains both source-archive and original worktree paths.
        rel=next(Path(prefix+r['path'].split('/'+prefix,1)[1]) for prefix in ('audits/','plans/','messages/','project/') if '/'+prefix in r['path'])
        p=repo/rel;require(sha(p)==r['sha256'],'PRIOR_FULL_READ_IDENTITY');authority.append(record(p))
    for rel in ('PROTOCOL.md','messages/head/2026-09-29-joint-multilayer-bs1-sh2-migration.md',
                'messages/head/2026-09-30-native-weak-b010-sh2.md','tasks/pending/2026-09-30-native-weak-b010-sh2.json',
                'plans/updates/server2/joint-multilayer-bs1-20260929-v1/user-cap3.md',
                'messages/head/2026-09-19-all-sh-default-no-checkpoints.md'):
        p=repo/rel;authority.append(record(p));dest=ROOT/'authoritative'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(p.read_bytes())
    original=NATIVE/'AlphaEdit/compute_z.py';text=original.read_text();patched=patch_source(text)
    (ROOT/'inputs').mkdir();path=ROOT/'inputs/compute_z_weak.py'
    with path.open('x') as f:f.write(patched)
    with (ROOT/'inputs/compute_z.diff').open('x') as f:f.writelines(difflib.unified_diff(text.splitlines(True),patched.splitlines(True),fromfile='original/compute_z.py',tofile='task-local/compute_z_weak.py'))
    for r in cfg['native_dependencies']:require(sha(r['path'])==r['sha256'],'NATIVE_DEPENDENCY')
    require(sha(cfg['official_native']['path'])==cfg['official_native']['sha256'],'OFFICIAL_NATIVE')
    cfg['checkpoints']={'B010':dict(cfg['checkpoints']['B010'],**stat_reuse(cfg['checkpoints']['B010']))}
    stat_reuse(cfg['projector'])
    for r in cfg['model_members']:stat_reuse(r)
    require(cfg['execution_ids']==[int(r['case_id']) for r in csv_rows(Path(cfg['design'])/'continuation-ids.csv')[:100]],'SAME_METADATA100')
    require(digest(cfg['execution_ids'])=='f4214f4a2a94c90a1f4113566fff90b19f0779a12a885aa00d5a2c92ac4106b6','ORDER_SHA')
    cfg['instruction_id']=NONCE
    cfg['trajectories']=[dict(name='B010-NATIVE_WEAK_NLL1',arm='NATIVE_WEAK_NLL1',checkpoint='B010')]
    cfg['weak']=dict(stop_quantity='nll_loss',comparison='<=',threshold=1.0,position='before backward',save_checkpoints=False,exact_resume='NOT_AVAILABLE',final_RPN=[100,200,1000])
    cfg['contract']['arms']=['NATIVE_WEAK_NLL1'];cfg['contract']['checkpoints']=['B010']
    cfg['contract']['weight_snapshots']={'enabled':False,'count':0,'exact_editor_resume':False}
    cfg['contract']['budgets']=dict(trajectories=1,scientific_batch_attempts=100,native_target_fits=100,native_solves=500,layer_history_append=500,max_loss_evaluations=2500,max_adam_updates=2400)
    cfg['resources']=dict(node='server2',project_cap=3,task_cap=1,gpus=1,cpus=8,mem_mib=60416,export='NONE',requeue=0,wall='08:00:00',
        output_reserve_bytes=16*1024**3,estimated_peak_gpu_gib=38,estimated_host_gib=40,
        estimate_basis='Same-host native B010 allocated12778s/34.744GiB VRAM/32.928GiB host plus actual-context600calls and full-final2600calls; no weight saving. Eight-hour wall allowance, not GPU-hour hard cap.',
        cap_authority=NONCE,tracked_cap2_superseded_for_this_submission=True)
    save(ROOT/'configuration.json',cfg)
    save(ROOT/'prepare-receipt.json',dict(instruction_id=NONCE,authority=authority,prior_source=oldlock['source'],old_config=record(OLD/'configuration.json'),
        configuration=record(ROOT/'configuration.json'),original_compute_z=record(original),patched_compute_z=record(path),diff=record(ROOT/'inputs/compute_z.diff'),
        ast_only_predicate_changed=True,parent=stat_reuse(cfg['checkpoints']['B010']),projector=stat_reuse(cfg['projector']),
        ids_sha=digest(cfg['execution_ids']),first_id=cfg['execution_ids'][0],last_id=cfg['execution_ids'][-1],gpu=False,submitted=False,
        prior_full_read_reuse='Original thirteen authority members hash equal; original full read reused; new envelope read fully',disk_free=shutil.disk_usage(ROOT).free))
    print('PREPARED_CPU_ONLY',ROOT)

def token_check():
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import verify
    from project.run_scripts.joint_multilayer_bs10.observations import catalog
    from project.run_scripts.joint_multilayer_bs10.final_eval import rows_for
    cfg=read(ROOT/'configuration.json');verify(Path(cfg['dataset']['path']).parent)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
    data,ids,panels,current,native,neighborhood=catalog(cfg,tok,'B010',cfg['checkpoints']['B010']['metadata']['contexts'])
    rows=panels+sum(current.values(),[])+sum(native.values(),[])+sum(neighborhood.values(),[])
    require(rows==read(OLD/'output/B010-NATIVE/token-catalog.json'),'TOKEN_CATALOG_EXACT')
    full=sum([rows_for(tok,data[c]) for c in ids],[])
    require(len(full)==len({r['row_id'] for r in full})==2600,'FULL_ROWS')
    save(ROOT/'token-check.json',dict(status='CPU_TOKEN_EXACT',rows=len(rows),full_rows=2600,panel_pairs=len(panels)//2,
        ids_sha=digest(ids),full_row_ids_sha=digest([r['row_id'] for r in full]),model_loaded=False,forward=0))

def freeze(repo):
    repo=Path(repo).resolve();require(not cmd(['git','status','--porcelain'],repo),'SOURCE_DIRTY')
    src=ROOT/'source';src.mkdir();members=[]
    rels=['scripts/fixed_counterfact.py']+['project/run_scripts/joint_multilayer_bs10/'+f for f in JOINT_FILES]
    rels += [str(p.relative_to(repo)) for p in sorted((repo/'project/run_scripts/native_weak_b010').glob('*.py'))]
    for rel in rels:
        dest=src/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write((repo/rel).read_bytes())
        members.append(record(dest))
    for f in ('common.py','runtime.py','observations.py'):
        require(sha(src/'project/run_scripts/joint_multilayer_bs10'/f)==sha(OLD/'source/project/run_scripts/joint_multilayer_bs10'/f),'ORIGINAL_RUNTIME_BYTE_REUSE')
    cfg=read(ROOT/'configuration.json');comp={a:record(EVAL/'output'/a/'full-metrics.json') for a in ('NATIVE','JOINT_STEP','JOINT_CUM')}
    for a,r in comp.items():require(r['sha256']==read(EVAL/'output'/a/'terminal.json')['raw']['sha256'],'SEALED_COMPARATOR')
    lock=dict(root=str(ROOT),source_root=str(src),instruction_id=NONCE,source=cmd(['git','rev-parse','HEAD'],repo),tree=cmd(['git','rev-parse','HEAD^{tree}'],repo),
        source_members=members,configuration=record(ROOT/'configuration.json'),patched_compute_z=record(ROOT/'inputs/compute_z_weak.py'),
        old_catalog=record(OLD/'output/B010-NATIVE/token-catalog.json'),old_entry_state=record(OLD/'output/B010-NATIVE/entry-state.json'),
        comparators=comp,resources=cfg['resources'],token_check=record(ROOT/'token-check.json'),NO_BROADCAST_NOT_REQUIRED=True)
    save(ROOT/'execution.lock.json',lock)
    script='''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=server2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60416M
#SBATCH --export=NONE
#SBATCH --no-requeue
#SBATCH --time=08:00:00
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
'''+f'cd {src}\nexec {PYTHON} -B -m project.run_scripts.native_weak_b010.runner --lock {ROOT}/execution.lock.json\n'
    with (ROOT/'run.sbatch').open('x') as f:f.write(script)
    save(ROOT/'freeze.json',dict(lock=record(ROOT/'execution.lock.json'),script=record(ROOT/'run.sbatch')))
    print('FROZEN',lock['source'])

def submit():
    require(not (ROOT/'submitted.json').exists(),'DUPLICATE_SUBMIT')
    for r in read(ROOT/'freeze.json').values():require(sha(r['path'])==r['sha256'],'FREEZE_CHANGED')
    q=cmd(['squeue','-h','-r','-u','janghj','-w','server2','-o','%i|%j|%T|%b|%R'])
    gpu=[r.split('|') for r in q.splitlines() if r and 'gpu' in r.split('|')[3]]
    # Known surviving prior paths only; any other admitted GPU task requires owner review.
    require(all(r[0] in ('55116_3','55116_4') and r[3]=='gres/gpu:1' for r in gpu),'UNKNOWN_ADMITTED_JOB')
    require(len(gpu)+1<=3,'PROJECT_CAP3')
    require(shutil.disk_usage(ROOT).free>=16*1024**3,'DISK_RESERVE')
    save(ROOT/'admission.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=q,
        existing_admitted=len(gpu),new_jobs=1,project_cap=3,task_cap=1,other_jobs_mutated=False,
        node=cmd(['scontrol','show','node','server2']),disk=shutil.disk_usage(ROOT)._asdict()))
    argv=['sbatch','--parsable','--hold','--job-name=odeedit_native_weak_b010_s2',
          '--output='+str(ROOT/'%j.out'),'--error='+str(ROOT/'%j.err'),str(ROOT/'run.sbatch')]
    job=cmd(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
    save(ROOT/'submitted.json',dict(job_id=job,argv=argv,held=True))
    info=cmd(['scontrol','show','job','-o',job])
    for token in ('UserId=janghj(','JobState=PENDING','Reason=JobHeldUser','Requeue=0','NumCPUs=8','gres/gpu=1','MinMemoryNode=59G','ReqNodeList=server2',str(ROOT/'run.sbatch')):
        require(token in info,'HELD_INSPECTION:'+token)
    save(ROOT/'held-inspection.json',dict(job_id=job,info=info,source=read(ROOT/'execution.lock.json')['source']))
    subprocess.run(['scontrol','release',job],check=True)
    save(ROOT/'released.json',dict(job_id=job,status='RELEASED',initial='NOT_OBSERVED',monitoring='bounded firststep->secondentry or verified resource-pending then pause'))
    print(job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','tokens','freeze','submit'));p.add_argument('--repo',default='.')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.repo)
    elif a.action=='tokens':token_check()
    elif a.action=='freeze':freeze(a.repo)
    else:submit()
