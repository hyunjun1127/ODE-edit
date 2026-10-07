"""Reuse exact existing S2 native input closures; CPU metadata only, never tensor load."""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from .generation_common import *

OLD_STOCK = Path('/mnt/raid5/janghj/ODE-edit/local/gptj-easyedit-native-baselines-2k/attempt-r1')
OLD_FOUR = Path('/mnt/raid5/janghj/ODE-edit/local/gptj-cake-blue-prune-rect-2k/attempt-r1')
def bind_old(root,source):
    c,lock=read(root/'config.json'),read(root/'execution.lock.json')
    require(lock['source_commit']==source and sha(root/'config.json')==lock['config_sha256'],
            'PRIOR_NATIVE_CONFIG_SOURCE')
    require(c['model_revision']=='47e169305d2e8376be1d31e765533382721b2cc1'
            and c['seed']==20261002 and c['ordered_ids_sha256']==ORDERED_SHA,
            'PRIOR_NATIVE_COHORT_MODEL')
    for row in c['assets']+c['runtime']['members']+[c['observer_identity']]:
        stat_seal(row)
    rows=c['native']['closure']
    rows=rows if isinstance(rows,list) else [row for armrows in rows.values() for row in armrows]
    for row in rows:verify(row)
    require(c['native']['context_reuse'] is None and c['noCP'] and not c['z_disk_cache'],
            'COLD_NATIVE_OWN_CONTEXT_NO_ZCACHE')
    return c, dict(config=small_member(root/'config.json'),lock=small_member(root/'execution.lock.json'),
                   source=source,closure_members=len(rows),large_asset_validation='PRIOR_FULL_SHA_PLUS_CURRENT_STAT',
                   large_asset_copies=0,tensor_loads=0)
def prepare():
    envelope,contract,policy=authority()
    out=LOCAL/'preparation-r1'
    require(not out.exists() and not list(LOCAL.glob('*/submission.json')),'NEW_NONCE_CREATE_ONCE')
    stock,sp=bind_old(OLD_STOCK,'56d3a445553b60bf1a5e33f0e820e0699364ba0b')
    four,fp=bind_old(OLD_FOUR,'3a4a107b7ae4c66f2f7a7f0cb441d26c5a639f68')
    require(stock['assets']==four['assets'] and stock['runtime']==four['runtime']
            and stock['cold_W']==four['cold_W'] and stock['model']==four['model']
            and stock['observer_identity']==four['observer_identity'],
            'SIX_NATIVE_SAME_MODEL_ASSETS_EVALUATOR')
    records=read(stock['stream'])[:2000]
    chunks=list(batches(records))
    require(len({r['case_id'] for r in records})==2000,'ORDERED_UNIQUE_OCCURRENCES')
    schedule=ROOT/contract['science']['schedule']
    require(sha(schedule)==contract['science']['schedule_sha256'],'FROZEN_SCHEDULE_BYTES')
    with schedule.open(newline='') as f: scheduled=list(csv.DictReader(f))
    require([int(r['case_id']) for r in scheduled]==[r['case_id'] for r in records]
            and [int(r['stream_index0']) for r in scheduled]==list(range(2000)),
            'ORDERED_OCCURRENCE_SCHEDULE')
    require(all(len(r['paraphrase_prompts'])==2 and len(r['neighborhood_prompts'])==10 for r in records),
            'RPN_PROMPT_DENOMINATORS')
    for relative in stock['evaluator_sources']:
        require(sha(ROOT/relative)==sha(OLD_STOCK/'source'/relative)==sha(OLD_FOUR/'source'/relative),
                'REUSED_SCORER_EXACT_SOURCE:'+relative)
    c=copy.deepcopy(four)
    c.update(instruction_id=NONCE,task_id=TASK,attempt=str(LOCAL/'attempt-r1'),
        authority=small_member(ROOT/ENVELOPE),contract=small_member(ROOT/CONTRACT),
        generation_policy=small_member(ROOT/POLICY),arm_layers={a:list(ARM_LAYERS[a]) for a in ARMS},
        arm_configs={a:copy.deepcopy(stock if a.startswith('BASE_') else four) for a in ARMS},
        generation=dict(schema=policy['schema'],profile=policy['generation']['profile'],eval_seed=20261007,
            assets_owner='SH1',common_source_status='NOT_YET_BOUND',
            reference_status='ASSET_NOT_AVAILABLE_PENDING_SH1_READY',
            W0_owner='BASE_MEMIT',W0_cache=str(LOCAL/'attempt-r1'/'W0-generation'),
            W0_reuse='Exact cold state/profile/cohort/seed only; not old RPN evidence',
            generator_route='Bind actual SH1 immutable implementation before source freeze',
            no_GPU_file_poll=True,physical_state_unchanged=True),
        native_repair_labels={'CAKE':'GPTJ_NATIVE_FC_OUT_BIAS_GUARD_COMPAT',
            'ALPHAEDIT_BLUE':'GPTJ_NATIVE_FC_OUT_BIAS_GUARD_COMPAT',
            'PRUNE':'PRUNE_TERMINAL_BASE_FIX; explicit_repair receipt key'},
        resources=dict(cpu=6,gpu=1,host_mib=59392,wall='2-00:00:00',
            collector_cpu=6,collector_host_mib=24576,collector_wall='04:00:00',
            project_cap=2,task_cap=2,reserve_bytes=16*1024**3,wall_is_eta=False,
            fresh_admission_required=True,simultaneous_host_requested_mib=2*59392,
            source_old_defaults_reused=True,generation_peak_not_yet_measured=True),
        noCP=True,z_disk_cache=False,exact_resume='NOT_AVAILABLE',
        generation_reference_or_tokenizer_refit=False,broadcast='NO_BROADCAST_NOT_REQUIRED_RAW_LOCAL_KEEP')
    write(out/'config-provisional.json',c)
    count=sum(len(r.get('generation_prompts',[])) for r in records)
    write(out/'metadata-preparation.json',dict(status='NATIVE_METADATA_BOUND_GENERATION_SOURCE_ASSET_PENDING',
        authority=small_member(ROOT/ENVELOPE),contract=small_member(ROOT/CONTRACT),policy=small_member(ROOT/POLICY),
        old_stock=sp,old_four=fp,schedule=small_member(schedule),requests=2000,batches=20,
        per_current=dict(R=100,P=200,N=1000),final=dict(R=2000,P=4000,N=20000),
        generation_planned_cases=2000,generation_prompts=count,
        missing_generation_prompts=sum(not r.get('generation_prompts') for r in records),
        generation_occurrence_prompts_sha256=digest([(i,r['case_id'],r.get('generation_prompts',[])) for i,r in enumerate(records)]),
        ordered_sha256=ORDERED_SHA,actual_GPU=False,new_fit=0,new_Slurm=0,
        large_asset_loads=0,large_asset_copy=0,prior_sha_plus_current_stat=True,
        generation_shared_SOURCE_READY=False,native_hparams_unchanged=True,noCP=True))
    return out
if __name__=='__main__':
    print(prepare())
