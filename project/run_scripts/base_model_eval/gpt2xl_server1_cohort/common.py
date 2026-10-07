"""Create-once identities for the explicitly authorized W0 cohort rerun."""
import os
from pathlib import Path
from ..gpt2xl_server1_common import (read,write,write_bytes,verify,member,sha,digest,require,
    W0View,check_guard,check_finite_rows,pair_specs,token_identity,PYTHON,SESSION,MODEL_REVISION,
    PAYLOAD_SHA,ORDER_SHA,DENOMINATORS,SCHEMA,capacity)

ROOT=Path(__file__).resolve().parents[4]
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gpt2xl-w0-cohort-curves/20261007-v1')
TASK='base-model-gpt2xl-w0-cohort-curves'
NONCE='USER-GH-SH1-SH2-W0-COHORT-CURVES-RERUN-20261007-R1-SERVER1'
ENVELOPE='messages/head/2026-10-07-w0-cohort-curves-server1.json'
ENVELOPE_SHA='78e0fab99858aa214c34a50c9bca97398cfd2afb80fb44aff30a6cb1a106f8e8'
CONTRACT='plans/global/2026-10-07-w0-cohort-curves-rerun/contract.json'
CONTRACT_SHA='b2f57f86b399ce47ad1c8e5cf777fc99e86aa3bdc812b2b16730c4f19015445c'
OLD_ATTEMPT=Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gpt2xl-w0/20261007-v1/attempt-v1')
OLD_SOURCE='de31540487643b3c4187ffad5c09554d18c69e94'
REFERENCE_CONFIG=dict(reference_only=True,evaluation_model_state='W0',
    edits_axis_semantics='reference_cohort_progress',w0_reference_schema='w0-cohort-reference-v1',
    actual_model_edits=0,actual_applied_edits=0,pre_state_edits=0,post_state_edits=0,writer='none',role='scientific')

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA,'COHORT_AUTHORITY_SHA')
    auth=read(ROOT/ENVELOPE)
    require(auth['instruction_id']==NONCE and auth['owner']['session']==SESSION and
        auth['scope']['actual_model_edits']==0 and auth['resources']['task_GPUs']==1,'COHORT_AUTHORITY_SCOPE')
    return auth

def verify_lock(config_path,lock_path):
    c,lock=read(config_path),read(lock_path)
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==TASK and
        sha(config_path)==lock['config_sha256'],'COHORT_CONFIG_LOCK')
    require(os.environ.get('ODEEDIT_W0_COHORT_SOURCE',lock['source_commit'])==lock['source_commit'],'COHORT_SOURCE_BINDING')
    for item in lock['source_members']+[lock['archive'],lock['tracking_env']]+c['input_members']+c['runtime']['source_members']:
        verify(item)
    for item in c['model_assets']:
        verify(item,metadata_only=Path(item['path']).name=='model.safetensors')
    require(all(c['reference_config'][key]==value for key,value in REFERENCE_CONFIG.items()),'COHORT_REFERENCE_IMMUTABLE')
    require(c['save_checkpoints'] is False and c['fresh_observation_required'] is True and
        all(c[key]==0 for key in ('edit_calls','target_fits','solves','history_appends','stats_loads','projector_loads')) and
        c['observer_microbatch']==2,'COHORT_W0_ONLY')
    return c,lock
