"""Fresh CPU metadata/input binding; no model or Wikipedia/C0/P calculation."""
import copy
import getpass
import importlib
import os
import shlex
import subprocess
import sys
import tarfile
import uuid
from .common import *

def prepare(attempt):
    from transformers import AutoTokenizer
    import torch
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    from ..gpt2xl_server1_prepare import runtime
    from ..gpt2xl_server1_common import verify_config_lock
    from .submit import prior_binding
    from .curves import attach_occurrence_ordinals
    authority();attempt=Path(attempt).resolve()
    require(not torch.cuda.is_initialized(),'COHORT_CPU_PREP_ONLY')
    require(attempt.parent==LOCAL and not attempt.exists(),'COHORT_CREATE_ONCE_NAMESPACE')
    require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'COHORT_NONCE_DUPLICATE')
    old,old_lock=verify_config_lock(OLD_ATTEMPT/'config.json',OLD_ATTEMPT/'execution.lock.json')
    require(old_lock['source_commit']==OLD_SOURCE,'COHORT_PRIOR_SOURCE')
    model=Path(old['model'])
    require(subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip()==MODEL_REVISION,'COHORT_MODEL_REVISION')
    payload=next(item for item in old['model_assets'] if Path(item['path']).name=='model.safetensors')
    require(payload['sha256']==PAYLOAD_SHA,'COHORT_MODEL_PAYLOAD_SEAL')
    records=load_prefix(Path(old['stream']).parent,2000)
    require(digest([row['case_id'] for row in records])==ORDER_SHA,'COHORT_ORDERED_FIRST2K')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,read(old['contexts']['path']))
    expected=token_identity(records,bench)
    require(expected==read(old['observer_identity']['path'])['rows'] and len(expected)==26000,'COHORT_FRESH_TOKEN_IDENTITY')
    expected=attach_occurrence_ordinals(expected)
    # Actual prior owner/task/source/state is checked once, only for lane admission.
    prior=prior_binding()
    attempt.mkdir(parents=True)
    write(attempt/'prior-w0-binding.json',prior)
    write(attempt/'observer-identity.json',dict(rows=expected,requests=2000,
        ordered_ids_sha256=ORDER_SHA,occurrence_ordinals='0..1999, no dedup or numeric case-ID selection'))
    c={key:copy.deepcopy(old[key]) for key in ('model','model_revision','model_assets','model_asset_identity',
        'cold_selected_W','stream','contexts','ordered_ids_sha256','seed','denominators',
        'observer_microbatch','observer_request_chunk','evaluator_identity')}
    c.update(schema=1,instruction_id=NONCE,task_id=TASK,attempt=str(attempt),
        run_instance=dict(attempt=attempt.name,run_id=uuid.uuid4().hex[:16]),
        observer_identity=member(attempt/'observer-identity.json'),
        prior_binding=member(attempt/'prior-w0-binding.json'),runtime=runtime(),
        input_members=copy.deepcopy(old['input_members'])+[member(attempt/'observer-identity.json')],
        authority=member(ROOT/ENVELOPE),contract=member(ROOT/CONTRACT),source_reference=member(OLD_ATTEMPT/'config.json'),
        reference_config=REFERENCE_CONFIG,prompt_pairs=26000,candidate_rows=52000,
        tracking=dict(env_file=old['tracking']['env_file'],metric_schema=SCHEMA,
            w0_reference_schema='w0-cohort-reference-v1',startup_readback='immutable identity before model load',
            finish_readback='bounded W0 + 20current + 4allseen coverage; SDK accepted is not remote PASS'),
        resources=dict(gpu=1,cpu=8,host_mib=65536,wall='04:00:00',collector_cpu=8,
            collector_host_mib=24576,collector_wall='02:00:00',dependency=prior['active_job_ids'],
            scoped_project_cap_exception=True,global_cap_changed=False,storage=capacity(LOCAL),ETA='NOT_MEASURED'),
        save_checkpoints=False,exact_resume='NOT_AVAILABLE',fresh_observation_required=True,
        edit_calls=0,target_fits=0,solves=0,history_appends=0,stats_loads=0,projector_loads=0,
        previous_raw_reused_as_measurement=False,broadcast='NO_BROADCAST_NOT_REQUIRED; same host local raw KEEP')
    write(attempt/'config.json',c)
    write(attempt/'cpu-input-preflight.json',dict(status='CPU_PASS_NOT_MODEL_PASS',requests=2000,
        prompt_pairs=26000,candidates=52000,explicit_occurrence_ordinals=True,model_loaded=False,GPU=False,
        C0_P_loaded=False,prior_states=prior['states'],new_run_id=c['run_instance']['run_id'],config_sha256=sha(attempt/'config.json')))
    return c

def source_files():
    from . import run,collect,tracking,schema,curves
    from scripts import fixed_counterfact
    from project.run_scripts.jlz_price_gpt2xl import scores
    from project.run_scripts.jlz_realization import inputs
    from project.run_scripts.jlz_realized_writer_sequential import review_completed
    from project.run_scripts.jlz_interference_l1 import comparison_bridge
    files=set()
    for name,module in list(sys.modules.items()):
        value=getattr(module,'__file__',None)
        if value and (name.startswith('project.run_scripts.') or name=='scripts.fixed_counterfact'):
            path=Path(value).resolve()
            if path.is_relative_to(ROOT) and path.suffix=='.py':files.add(path.relative_to(ROOT).as_posix())
    files.update(path.relative_to(ROOT).as_posix() for path in Path(__file__).parent.glob('*.py'))
    files.update(path.relative_to(ROOT).as_posix() for path in (ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))
    files.update((ENVELOPE,CONTRACT,'control/wandb-policy.json','control/wandb-method-metric-schema.json'))
    return sorted(files)

def freeze(config_path,source_commit):
    c=read(config_path);attempt=Path(c['attempt']);authority()
    require(not (attempt/'source').exists() and not (attempt/'execution.lock.json').exists(),'COHORT_SOURCE_CREATE_ONCE')
    files=source_files()
    require(not subprocess.check_output(['git','status','--porcelain','--',*files],cwd=ROOT,text=True),'COHORT_COMMIT_BEFORE_FREEZE')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source_commit,'COHORT_SOURCE_HEAD')
    archive=attempt/'source.tar';subprocess.run(['git','archive','--format=tar','--output='+str(archive),source_commit,*files],cwd=ROOT,check=True)
    source=attempt/'source';source.mkdir()
    with tarfile.open(archive) as stream:
        items=stream.getmembers()
        require(len(items)==len({item.name for item in items}) and all((item.isfile() or item.isdir()) and
            not Path(item.name).is_absolute() and '..' not in Path(item.name).parts for item in items),'COHORT_SAFE_ARCHIVE')
        stream.extractall(source,filter='data')
    members=[member(source/path) for path in files]
    require(all(sha(ROOT/path)==item['sha256'] for path,item in zip(files,members)),'COHORT_FROZEN_SOURCE_BYTES')
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=source_commit,
        source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
        config_sha256=sha(config_path),source_root=str(source),source_members=members,archive=member(archive),
        owner=getpass.getuser(),session=SESSION,server='server1',resources=c['resources'],
        tracking_env=member(c['tracking']['env_file']),authority=member(ROOT/ENVELOPE),contract=member(ROOT/CONTRACT),
        reference_only=True,actual_model_edits=0,checkpoint_saved=False)
    write(attempt/'execution.lock.json',lock);return lock

def launchers(config_path,lock_path):
    c,lock=verify_lock(config_path,lock_path);attempt=Path(c['attempt']);scripts={}
    for role in ('GPU','collector'):
        env=dict(PYTHONPATH=lock['source_root'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',
            MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            ODEEDIT_W0_COHORT_SOURCE=lock['source_commit'])
        if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
        argv=[PYTHON,'-u','-m','project.run_scripts.base_model_eval.gpt2xl_server1_cohort',
            'collect' if role=='collector' else 'run','--config',str(config_path),'--lock',str(lock_path)]
        script='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in env.items())
        script+='cd '+shlex.quote(lock['source_root'])+'\nexec '+shlex.join(argv)+'\n'
        path=attempt/(role+'.sh');write_bytes(path,script.encode());path.chmod(0o755);scripts[role]=member(path)
    value=dict(scripts=scripts,GPU_dependency=c['resources']['dependency'],collector_dependency='afterany:ACTUAL_NEW_GPU_ID',
        export='NONE',requeue=False,registered=False,resources=c['resources'])
    write(attempt/'launchers.json',value);return value
