"""Explicit USER final-only observation schedule; science and shared math fixed."""
from pathlib import Path
from .generation_common import ARMS, LOCAL, ROOT, SESSION, digest, read, require, sha, verify
from .generation_cache_common import TASK, NONCE, identity, ENVELOPE as REPAIR_ENVELOPE, ENVELOPE_SHA as REPAIR_SHA
from .generation_common import ENVELOPE, ENVELOPE_SHA, CONTRACT, CONTRACT_SHA, POLICY
from .generation_cache_qualification import validate_plan

AUTHORITY = 'plans/updates/server2/gptj-baselines-fluency-consistency-2k/final-generation-user-override.json'
SCHEDULE = 'FINAL_W20_ONLY'
POLICY_SHA = 'bb147bfff33d86f9a7a31d9b3193fd00a6075b3ab2df7496cad727cd9dbcf412'
FINAL_LOCAL = LOCAL / 'final-generation-v1'
ATTEMPT = FINAL_LOCAL / 'attempt-r1'
PREPARATION = FINAL_LOCAL / 'preparation-r2'


def enabled(config):
    return config.get('generation', {}).get('evaluation_schedule') == SCHEDULE


def authority():
    # Historical envelopes stay byte-bound; the latest canonical policy now
    # supersedes their old forty-endpoint generation schedule explicitly.
    for path, expected in ((ENVELOPE,ENVELOPE_SHA),(CONTRACT,CONTRACT_SHA),
                           (REPAIR_ENVELOPE,REPAIR_SHA),(POLICY,POLICY_SHA)):
        require(sha(ROOT/path)==expected,'FINAL_POLICY_AUTHORITY_BYTES:'+path)
    parent, contract, repair, policy=(read(ROOT/path) for path in
        (ENVELOPE,CONTRACT,REPAIR_ENVELOPE,POLICY))
    require(parent['owner']['server']=='server2' and parent['owner']['session']==SESSION
        and parent['baseline_methods']==contract['baseline_methods']
        ==['MEMIT','PRUNE','RECT','AlphaEdit','AlphaEdit-BLUE','CAKE']
        and contract['science']['BS']==100 and contract['science']['batches']==20
        and contract['science']['noCP'] is True
        and repair['instruction_id']==NONCE
        and repair['owners']['server2']['session']==SESSION
        and repair['owners']['server2']['task_id']==TASK
        and repair['resources']['project_task_cap_server2']==2,
        'FINAL_PARENT_NATIVE_AND_REPAIR_SCOPE')
    sampling=policy['generation']['sampling']
    require(policy['generation_schedule']=='W20_ONLY_FIRST2000'
        and policy['evaluation']['prefixes']==['all_seen/post']
        and sampling==dict(samples_per_prompt=1,top_k=5,temperature=1,top_p=1,
            max_total_tokens=100,length_unit='unpadded model input tokens + generated tokens')
        and policy['generation']['randomness']['eval_seed']==20261007,
        'FINAL_CANONICAL_SCHEDULE_SAMPLER')
    value = read(ROOT / AUTHORITY)
    require(value['authority_kind'] == 'DIRECT_USER_SCHEDULE_OVERRIDE'
        and value['user_followup_exact'] == '지금 돌아가는 실험이 있으면 전부 다 수정해'
        and value['server'] == 'server2' and value['session'] == SESSION
        and value['task_id'] == TASK and value['arms'] == list(ARMS)
        and value['evaluation_schedule'] == SCHEDULE
        and value['final_generation_calls_per_arm'] == 1
        and value['final_generation_requests_per_arm'] == 2000
        and value['W0_generation_calls'] == value['intermediate_generation_calls'] == 0
        and value['RPN_intermediate_schedule_unchanged'] is True
        and value['project_gpu_cap'] == 2 and value['noCP'] is True,
        'FINAL_ONLY_EXACT_USER_AUTHORITY')
    return value


def counts():
    from .generation_plan import counts as prior_counts
    value = prior_counts()
    value.update(evaluation_schedule=SCHEDULE, generation_endpoint_calls_per_arm=1,
        generation_current_subset_reductions_per_arm=0,
        generation_edit_state_case_observations_per_arm=2000,
        cold_W0_generation_case_observations=0,
        planned_generation_case_observations=6 * 2000,
        prompt_count_if_fixed10_per_case=6 * 2000 * 10,
        observation_counts_include_cache_reuse=False,
        B1_pre_cases_reusable_from_exact_cold_W0=0,
        additional_unique_generation_case_upper_bound=6 * 2000,
        qualification_max_prompt_route_evaluations=8 * 3,
        qualification_scientific_generation_endpoint=False)
    return value


def ready(config):
    authority()
    require(enabled(config) and identity(config) == (TASK, NONCE)
        and config['registration_profile'] == 'final-v1'
        and config['tracking_attempt'] == 'final-generation-v1'
        and config['attempt'] == str(ATTEMPT), 'FINAL_ONLY_PROFILE_IDENTITY')
    manual = config['manual_schedule_authority']
    verify(manual)
    require(manual['sha256'] == sha(ROOT / AUTHORITY), 'FINAL_ONLY_USER_AUTHORITY_MEMBER')
    verify(config['generation_policy'])
    require(config['generation_policy']['sha256']==POLICY_SHA,
        'FINAL_ONLY_CURRENT_CANONICAL_POLICY_MEMBER')
    gen, repair = config['generation'], config['generation']['repair']
    require(gen['W0_generation_enabled'] is False
        and gen['intermediate_generation_enabled'] is False
        and gen['final_generation_requests'] == 2000
        and gen['qualification_owner'] == 'BASE_MEMIT'
        and gen['common_source_status'] == 'READY_BOUND'
        and gen['reference_status'] == 'READY_VERIFIED'
        and gen['no_GPU_file_poll'] is True, 'FINAL_ONLY_NO_W0_OR_MIDPOINT_GENERATION')
    plan = read(verify(repair['qualification_plan']))
    cohort = read(verify(repair['qualification_cohort']))
    validate_plan(plan, cohort)
    shared = read(verify(repair['shared_qualification_plan']))
    require(plan['shared_source_sha'] == gen['source_sha']
        and digest(plan) == repair['qualification_plan_sha256']
        and shared == cohort['shared_plan']
        and digest(shared) == repair['shared_qualification_plan_sha256']
        == plan['shared_plan_sha256'], 'FINAL_ONLY_FIXED_QUALIFICATION_PLAN')
    require(repair['status'] == 'PLAN_BOUND_NOT_ACTUAL_PASS'
        and repair['actual_qualification_status'] == 'NOT_RUN'
        and repair['qualification_in_first_replacement_job'] is True,
        'FINAL_ONLY_ACTUAL_RECEIPT_CREATED_AT_RUNTIME')
    require(set(config['arm_configs']) == set(ARMS)
        and config['noCP'] is True and config['z_disk_cache'] is False
        and config['exact_resume'] == 'NOT_AVAILABLE'
        and config['resources']['gpu'] == 1 and config['resources']['cpu'] == 6
        and config['resources']['host_mib'] == 59392,
        'FINAL_ONLY_NATIVE_SCOPE_RESOURCE_NOCP')
    return True
