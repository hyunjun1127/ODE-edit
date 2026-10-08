"""Explicit SH2 USER native generation profile; historic science stays frozen."""
from .generation_common import (ARMS, LOCAL, ROOT, SESSION, SOURCE_ENV, ENVELOPE,
    ENVELOPE_SHA, CONTRACT, CONTRACT_SHA, POLICY, digest, member, read, require,
    sha, stat_seal, verify, write, expected_counts, writer_identity)
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE

TASK = 'gptj-baselines-native-generation-repair'
NONCE = 'USER-DIRECT-SH2-GPTJ-NATIVE-FLUCON-REREGISTER-20261008-R1'
AUTHORITY = 'plans/updates/server2/gptj-baselines-fluency-consistency-2k/native-flucon-reregister-user-authority.json'
NATIVE_LOCAL = LOCAL / 'native-repo-repair-r1'
PREPARATION = NATIVE_LOCAL / 'preparation-r1'
ATTEMPT = NATIVE_LOCAL / 'attempt-r1'
SCHEDULE = 'FINAL_W20_ONLY'
TRACKING_SCHEDULE = 'W20_ONLY_FIRST2000'
SHARED_SOURCE = 'adb244e6f9c86b54f73bd6d8fb833b338f470ded'
PACKAGE_TREE = '91349ee439b1573e473d280ff86f88d71dd6d807'
REPAIR_INSTRUCTION = 'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'
POLICY_SHA = 'bb147bfff33d86f9a7a31d9b3193fd00a6075b3ab2df7496cad727cd9dbcf412'

def identity(config):
    require(config['task_id'] == TASK and config['instruction_id'] == NONCE
        and config['parent_task_id'] == 'gptj-baselines-fluency-consistency-2k',
        'NATIVE_REPO_OWNER_TASK_IDENTITY')
    return TASK, NONCE

def authority():
    for path, expected in ((ENVELOPE, ENVELOPE_SHA), (CONTRACT, CONTRACT_SHA), (POLICY, POLICY_SHA)):
        require(sha(ROOT / path) == expected, 'NATIVE_PARENT_POLICY_BYTES:' + path)
    value = read(ROOT / AUTHORITY)
    require(value['instruction_id'] == NONCE and value['task_id'] == TASK
        and value['owner'] == dict(server='server2', session=SESSION)
        and value['arms'] == list(ARMS) and value['project_GPU_cap'] == 2
        and value['requests'] == 2000 and value['batch_size'] == 100 and value['batches'] == 20
        and value['generation_schedule'] == TRACKING_SCHEDULE and value['profile'] == PROFILE
        and value['shared_source'] == SHARED_SOURCE and value['shared_package_tree'] == PACKAGE_TREE
        and value['noCP'] is True and value['protected_job'] == '61428'
        and value['automatic_retry'] is False, 'EXACT_NATIVE_USER_SCOPE')
    contract = read(ROOT / CONTRACT)
    require(contract['baseline_methods'] == ['MEMIT', 'PRUNE', 'RECT', 'AlphaEdit', 'AlphaEdit-BLUE', 'CAKE']
        and contract['science']['BS'] == 100 and contract['science']['batches'] == 20
        and contract['science']['noCP'] is True, 'NATIVE_PARENT_SCIENCE_FIXED')
    return value

def counts():
    return dict(edit_applications=12000, unique_request_occurrences=2000,
        native_per_arm={arm:{k:v*20 for k,v in expected_counts(arm).items()} for arm in ARMS},
        generation_endpoint_calls_per_arm=1, generation_edit_state_case_observations_per_arm=2000,
        planned_generation_case_observations=12000, cold_W0_generation_case_observations=0,
        intermediate_generation_case_observations=0, qualification_prompt_evaluations=0,
        additional_generation_per_metric=0, checkpoint_saves=0,
        cost_status='COUNT_PLAN_NOT_MEASURED_TIME_OR_MEMORY')

def ready(config):
    authority(); identity(config)
    require(config['registration_profile'] == 'native-repo-r1'
        and config['tracking_attempt'] == 'native-repo-repair-r1'
        and config['attempt'] == str(ATTEMPT), 'NATIVE_REPO_IMMUTABLE_ATTEMPT')
    verify(config['manual_native_authority'])
    require(config['manual_native_authority']['sha256'] == sha(ROOT / AUTHORITY), 'NATIVE_REPO_AUTHORITY_MEMBER')
    verify(config['generation_policy'])
    require(config['generation_policy']['sha256'] == POLICY_SHA, 'NATIVE_REPO_SCHEDULE_POLICY')
    gen = config['generation']
    require(gen['profile'] == PROFILE and gen['generation_route'] == ROUTE
        and gen['generator_route'] == ROUTE and gen['eval_seed'] == 20261007
        and gen['evaluation_schedule'] == SCHEDULE and gen['generation_schedule'] == TRACKING_SCHEDULE
        and gen['generation_repair_instruction'] == REPAIR_INSTRUCTION
        and gen['source_sha'] == SHARED_SOURCE and gen['package_tree'] == PACKAGE_TREE
        and gen['common_source_status'] == 'READY_BOUND' and gen['reference_status'] == 'READY_VERIFIED'
        and gen['W0_generation_enabled'] is False and gen['intermediate_generation_enabled'] is False
        and gen['final_generation_requests'] == 2000 and gen['no_GPU_file_poll'] is True,
        'NATIVE_W20_ONLY_EXACT_SOURCE_REFERENCE_PROFILE')
    forbidden = {'repair','qualification_owner','qualification_plan','qualification_plan_member',
        'qualification_receipt_member','generation_microbatch','old_w0_reuse','W0_cache'}
    require(not forbidden & gen.keys(), 'NATIVE_NO_OLD_QUALIFICATION_OR_W0_GENERATION_GATE')
    for row in gen['shared_source_members']: verify(row)
    verify(gen['reference_manifest']); verify(gen['reference_READY'])
    require(config['noCP'] is True and config['z_disk_cache'] is False
        and config['exact_resume'] == 'NOT_AVAILABLE' and set(config['arm_configs']) == set(ARMS),
        'NATIVE_NOCP_SIX_ARMS')
    r = config['resources']
    require(r['gpu'] == 1 and r['cpu'] == 6 and r['host_mib'] == 59392
        and r['project_cap'] == r['task_cap'] == 2 and r['collector_cpu'] == 6
        and r['collector_host_mib'] == 24576 and r['wall'] == '2-00:00:00'
        and r['collector_wall'] == '04:00:00', 'NATIVE_EXACT_RESOURCE_PROFILE')
    return True
