"""Reuse exact native/input/reference assets; bind only new USER generation source."""
import copy
import subprocess
from .generation_native_common import *

PRIOR_CONFIG = LOCAL / 'final-generation-v1/preparation-r2/config.json'
PRIOR_SHA = '10c81ca34d2f3912249e398e4a3ae8b3e355e944d7488c976f6eb506c0e009ae'
SCIENCE_KEYS = ('arm_configs','arm_layers','assets','runtime','model','model_revision','model_assets',
    'packs','cold_W','native','native_bundle','native_provenance','observer_identity',
    'evaluator_sources','seed','stream','ordered_ids_sha256','W0_reference','context_policy','noCP','z_disk_cache')

def science_identity(config):
    return digest({key:config[key] for key in SCIENCE_KEYS if key in config})

def bind():
    authority()
    require(not PREPARATION.exists() and not ATTEMPT.exists(), 'NATIVE_REPO_BIND_CREATE_ONCE')
    require(sha(PRIOR_CONFIG) == PRIOR_SHA, 'NATIVE_PRIOR_CONFIG_EXACT')
    old = read(PRIOR_CONFIG)
    for row in old['assets'] + old['runtime']['members'] + [old['observer_identity']]: stat_seal(row)
    value = copy.deepcopy(old)
    value.update(task_id=TASK,instruction_id=NONCE,attempt=str(ATTEMPT),registration_profile='native-repo-r1',
        tracking_attempt='native-repo-repair-r1',manual_native_authority=member(ROOT/AUTHORITY),
        previous_final_generation_config=member(PRIOR_CONFIG),generation_policy=member(ROOT/POLICY))
    for key in ('manual_recall_authority','manual_schedule_authority'):
        value.pop(key,None)
    gen = value['generation']
    for key in ('repair','qualification_owner','W0_cache','W0_owner','W0_state_identity','W0_reuse','cache_scope'):
        gen.pop(key,None)
    gen.update(profile=PROFILE,source_sha=SHARED_SOURCE,package_tree=PACKAGE_TREE,
        generator_route=ROUTE,generation_route=ROUTE,evaluation_schedule=SCHEDULE,
        generation_schedule=TRACKING_SCHEDULE,generation_repair_instruction=REPAIR_INSTRUCTION,
        W0_generation_enabled=False,intermediate_generation_enabled=False,final_generation_requests=2000,
        no_GPU_file_poll=True,old_partial_generation_reuse=False,
        shared_source_members=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_generation_eval').glob('*.py'))])
    require(subprocess.check_output(['git','rev-parse','HEAD:project/run_scripts/experiment_generation_eval'],
        cwd=ROOT,text=True).strip() == PACKAGE_TREE, 'NATIVE_EXACT_SHARED_TREE')
    require(science_identity(value) == science_identity(old), 'NATIVE_SCIENCE_ASSET_PRECISION_UNCHANGED')
    ready(value)
    write(PREPARATION/'config.json',value)
    write(PREPARATION/'binding.json',dict(status='SOURCE_INPUT_READY_NOT_ACTUAL_GPU_PASS',
        previous_config=member(PRIOR_CONFIG),config=member(PREPARATION/'config.json'),
        shared_source=SHARED_SOURCE,package_tree=PACKAGE_TREE,
        scientific_identity_unchanged=science_identity(value),noCP=True,
        model_loads=0,generation_forwards=0,new_GPU_qualification=0,
        old_raw_relabelled=False,old_source_archive_and_cost_preserved=True))
    return value

if __name__ == '__main__': bind()
