"""One explicit USER-authorized B9 descendant; no cold fallback or auto-retry."""
import argparse
import getpass
import os
from pathlib import Path
import re
import shlex
import shutil
import tarfile
from datetime import datetime, timezone
from official.experiments.prepare import write_new, file_sha
from official.runners.server1.common import read, member, verify
from official.runners.server1 import submit as control
from official.runners.server2 import submit as own
from official.runners.server3.submit import AGENT_SEALS, official_tree_sha256, check_wandb
from .sphere_oom import REPAIRED_SHA
from .sphere_b9_resume import INSTRUCTION,CELL,LOCAL,validate_binding
from .fe_author_prepare import OLD,LOCAL as FE_LOCAL,storage_plan
from .fe_author_submit import graph,PYTHON

REPO=Path(__file__).resolve().parents[3]

def prepare(root):
    root=Path(root).absolute()
    assert root==LOCAL/'registration-r1' and not root.exists(),'RECONCILE_EXISTING_ATTEMPT'
    assert not own.command(['git','status','--porcelain','--','official'],cwd=REPO)
    source=own.command(['git','rev-parse','HEAD'],cwd=REPO)
    own.command(['git','merge-base','--is-ancestor',source,'origin/main'],cwd=REPO)
    own.command(['git','merge-base','--is-ancestor','7b5097aa447946e35de42229c22b0c0feabd11ae',source],cwd=REPO)
    old=next(j for j in read(OLD/'submission.json')['jobs'] if j['job_id']=='62087')
    assert old['cell']==CELL and old['source']=='7b5097aa447946e35de42229c22b0c0feabd11ae'
    accounting=own.command(['sacct','-X','-j','62087','-nP','--format=JobID,JobName%64,User,State,ExitCode,WorkDir%240'])
    row=accounting.split('|');assert row[:5]==['62087','s2-qwen25-zsre-sphere-gpu',getpass.getuser(),'FAILED','1:0']
    assert row[5]==str(OLD)
    pointer=read(OLD/'runs'/CELL/'checkpoint/latest.json')
    assert pointer['batch']==9 and pointer['sha256']=='7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8'
    oldcp=member(OLD/'runs'/CELL/'checkpoint'/pointer['file']);assert oldcp['sha256']==pointer['sha256']
    storage=storage_plan(shutil.disk_usage(OLD).free)
    # Include the two newly admitted FE-author final payloads, plus this run's
    # latest + atomic temporary. Existing old B9 is already charged to free space.
    storage['required_free_bytes'] += storage['checkpoint_budget_bytes']
    storage['sufficient']=storage['free_bytes']>=storage['required_free_bytes']
    assert storage['sufficient'],'STORAGE_PENDING_KEEP_SOURCE'
    root.mkdir(parents=True)
    for name in ('logs','scripts','processes','configs','streams'): (root/name).mkdir()
    archive=root/'source.tar'
    own.command(['git','archive','--format=tar','--output='+str(archive),source,'official',*AGENT_SEALS],cwd=REPO)
    with tarfile.open(archive) as t:
        assert all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
        t.extractall(root/'source',filter='data')
    lock=dict(code_commit=source,official_tree_sha256=official_tree_sha256(root/'source'),
        members=[dict(relative=str(p.relative_to(root/'source')),sha256=file_sha(p)) for p in sorted((root/'source').rglob('*')) if p.is_file()])
    write_new(root/'source-lock.json',lock)
    cfg=read(OLD/'configs'/f'{CELL}.json')
    from .qwen_mask_profile import validate
    validate(cfg);assert cfg['method']=='SPHERE' and cfg['dataset']=='zsre'
    write_new(root/'configs'/f'{CELL}.json',cfg)
    for name in ('zsre-stream.json','zsre-stream.lock.json'):shutil.copyfile(OLD/'streams'/name,root/'streams'/name)
    write_new(root/'mask-profile.json',read(OLD/'mask-profile.json'))
    assets=read(OLD/'assets.json');assets['output_root']=str(root)
    write_new(root/'assets.json',assets)
    binding=validate_binding(LOCAL/'parent-binding-r1.json',cfg)
    shutil.copyfile(LOCAL/'parent-binding-r1.json',root/'resume-parent.json')
    shutil.copyfile(OLD/'w0-parent.json',root/'w0-parent.json')
    assert file_sha(root/'source/official/baselines/easyedit/models/SPHERE/SPHERE_main.py')==REPAIRED_SHA
    assert not (root/'sphere-repair.json').exists()  # restore contexts, never cold-inject

    os.environ['ODEEDIT_WANDB_PROJECT_VERIFIED']='1'
    write_new(root/'wandb-project-precheck.json',check_wandb(root/'source',assets['wandb_env']))
    from .qwen_assets import preflight
    receipt=preflight(root/'assets.json',hash_large=True,require_generation=False)
    write_new(root/'asset-preflight.json',receipt)
    assert receipt['ready_to_submit'],str(receipt['blockers'])
    original=read(binding['parent_assets']['path'])
    assert receipt['assets_sha256']==original['assets_sha256'] and receipt['asset_identity']==original['asset_identity']
    assert receipt['runtime']['packages']==original['runtime']['packages']
    from .qwen_eval_refresh import verify_parent
    verify_parent(root,'zsre',read(root/'streams/zsre-stream.lock.json'),lock,receipt)
    proof=member(FE_LOCAL/'configs-r1/zsre-query-parity.json')
    q=read(proof['path']);assert q['requests']==2000 and q['input_mismatches']==q['target_mismatches']==0
    write_new(root/'query-proof-reuse.json',dict(proof=proof,stream=member(root/'streams/zsre-stream.json'),
        parent_stream=binding['stream'],pretrained_forward_parity='NOT_CLAIMED'))
    members=[member(p) for p in [root/'assets.json',root/'resume-parent.json',root/'w0-parent.json',root/'mask-profile.json',
        root/'configs'/f'{CELL}.json',root/'streams/zsre-stream.json',root/'streams/zsre-stream.lock.json']]
    write_new(root/'input-lock.json',dict(members=members))
    write_new(root/'preparation.json',dict(instruction_id=INSTRUCTION,source=source,storage=storage,
        old_accounting=accounting,old_checkpoint=oldcp,old_checkpoint_action='READONLY_RESTORE_EXACT_B9',parent_binding_sha256=binding['binding_sha256'],
        qualification='NOT_RUN_USER_DISABLED',repaired_GPU_OOM_resolution='NOT_YET_OBSERVED',
        current_hparams_unchanged=True,CPU_resume_controls='test_sphere_b9_resume',no_new_fit_or_GPU_pilot=True))

def register(root):
    root=Path(root).absolute();assert root==LOCAL/'registration-r1'
    assert not (root/'submitted.json').exists(),'REGISTERED_RECONCILE_NO_DUPLICATE'
    prep=read(root/'preparation.json');source=read(root/'source-lock.json')
    cap_path=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    row=[l.split('\t') for l in cap_path.read_text().splitlines() if l.startswith('server2\t')]
    assert len(row)==1 and row[0][1]=='server2' and int(row[0][3])>=59392
    cap=min(4,int(row[0][2]));existing=own.inventory();assert not existing['ambiguous']
    assert not any('sphere-resume-b9' in r['name'] for r in existing['project'])
    parent=control.metadata(own.command(['scontrol','show','job','62532','--oneliner']))
    assert parent['UserId'].split('(')[0]==getpass.getuser()
    assert parent['JobName']=='s2-qwen25-zsre-fe-author-history' and parent['ReqNodeList']=='server2'
    fe_job=read(FE_LOCAL/'registration-r1/submission.json')['jobs']['zsre']
    assert fe_job['job_id']=='62532' and parent['Command']==fe_job['script']['path']
    assert file_sha(parent['Command'])==fe_job['script']['sha256']
    cancelled=control.metadata(own.command(['scontrol','show','job','62534','--oneliner']))
    assert cancelled['JobState']=='CANCELLED' and cancelled['JobName']=='s2-qwen25-zsre-sphere-oom-r1'
    parents=['62532']
    nodes=[dict(key=r['job'],gpus=r['gpus'],parents=own.dependency_ids(r['dependency'])) for r in existing['project']]
    width=control.graph_width(nodes+[dict(key='NEW_RESUME_B9',gpus=1,parents=parents)]);assert width<=cap
    assert sum(r['allocated_GPUs'] for r in existing['project'])<=cap
    assert shutil.disk_usage(root).free>=prep['storage']['required_free_bytes']
    node=control.metadata(own.command(['scontrol','show','node','server2','--oneliner']))
    qos=own.command(['sacctmgr','-nP','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPU,MaxTRESPJ,GrpTRES'])
    assert 'gres/gpu=4' in qos and int(node['RealMemory'])>=59392
    cfg=read(root/'configs'/f'{CELL}.json');env=dict(PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NLTK_DATA='/mnt/raid5/janghj/nltk_data',
        ODEEDIT_SOURCE_LOCK=str(root/'source-lock.json'),ODEEDIT_CODE_COMMIT=source['code_commit'],
        ODEEDIT_OFFICIAL_TREE_SHA256=source['official_tree_sha256'],ODEEDIT_WANDB_PROJECT_VERIFIED='1',
        ODEEDIT_WANDB_ENV_FILE=read(root/'assets.json')['wandb_env'],WANDB_MODE='online',WANDB_DISABLED='false')
    argv=[PYTHON,'-B','-u','-m','official.runners.server2.sphere_b9_resume','--root',str(root)]
    text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
    text+='cd '+shlex.quote(str(root/'source'))+'\nexec '+shlex.join(argv)+'\n'
    script=root/'scripts/main.sh'
    with script.open('x') as f:f.write(text)
    name='s2-qwen25-zsre-sphere-resume-b9'
    cmd=['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--partition=gpu','--qos=lab_gpu_s2',
        '--nodelist=server2','--nodes=1','--ntasks=1','--cpus-per-task=6','--gres=gpu:a6000:1',
        '--mem=59392M','--time=2-00:00:00','--job-name='+name,'--chdir='+str(root/'source'),
        '--output='+str(root/'logs/main-%j.out'),'--error='+str(root/'logs/main-%j.err')]
    if parents:cmd.append('--dependency=afterany:'+':'.join(parents))
    cmd.append(str(script))
    write_new(root/'admission.json',dict(existing=existing,parents=parents,DAG_width=width,cap=cap,node=node,qos=qos))
    response=own.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
    item=dict(instruction_id=INSTRUCTION,job_id=jid,job_name=name,argv=argv,sbatch=cmd,dependency=parents,
        source=source['code_commit'],official_tree=source['official_tree_sha256'],config_sha256=cfg['config_sha256'],
        output=str(root/'runs'/CELL),resources=dict(GPUs=1,CPUs=6,memory_MiB=59392,wall_hours=48))
    write_new(root/'submitted.json',item)
    d=control.metadata(own.command(['scontrol','show','job',jid,'--oneliner']))
    assert d['UserId'].split('(')[0]==getpass.getuser() and d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
    assert d['Command']==str(script) and d['WorkDir']==str(root/'source') and d['JobName']==name
    assert d['ReqNodeList']=='server2' and d['QOS']=='lab_gpu_s2' and d['Requeue']=='0'
    assert control.gpu_count(d['ReqTRES'])==1 and control.gpu_count(d.get('AllocTRES',''))==0
    assert control.requested_cpu_matches(d,6) and control.memory_MiB(d['MinMemoryNode'])==59392
    assert control.seconds(d['TimeLimit'])==48*3600 and control.dependencies(d['Dependency'])==sorted(('afterany',p) for p in parents)
    assert own.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
    for m in read(root/'input-lock.json')['members']:verify(m)
    assert graph(own.inventory())[1]<=cap
    write_new(root/'held-inspection.json',dict(detail=d,source=source['code_commit'],input_lock=member(root/'input-lock.json'),
        checkpoint_resume='EXACT_PARENT_B9',parent_checkpoint=read(root/'resume-parent.json')['parent_checkpoint'],start_batch=9,next_batch=10,GPU_qualification='NOT_RUN_USER_DISABLED',actual_GPU_OOM_resolved=False))
    own.command(['scontrol','release',jid])
    snap=control.metadata(own.command(['scontrol','show','job',jid,'--oneliner']))
    result=dict(item,held_inspected=True,released=True,initial_snapshot=snap,
        old_job_id='62087',old_B9_checkpoint_KEEP=True,cancelled_cold_job='62534',start_batch=9,next_batch=10,remaining_batches=list(range(10,21)),parent_checkpoint=read(root/'resume-parent.json')['parent_checkpoint'],monitoring_active=False,
        observed_utc=datetime.now(timezone.utc).isoformat())
    write_new(LOCAL/'submission-complete.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','register']);p.add_argument('--root',required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else register)(a.root)
