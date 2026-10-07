"""Explicit repair authority/profile; the parent six-arm scientific scope stays fixed.

The original profile constants and historical archives remain unchanged. A
qualification PLAN may be frozen before submission; only the GPU runner can
create an actual qualification receipt. No guessed actual PASS is accepted.
"""
from pathlib import Path

from .generation_common import (ARMS, LOCAL, NONCE as PARENT_NONCE,
    ROOT, SESSION, TASK as PARENT_TASK, authority as parent_authority,
    digest, read, require, sha, verify)

TASK = 'gptj-baselines-generation-cache-repair'
NONCE = 'USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1'
ENVELOPE = 'messages/head/2026-10-08-baseline-generation-kv-batch-repair.json'
# Original dispatch SHA is preserved as historical identity; the GH amendment
# at89734ff5 explicitly separates a pre-submit PLAN from runtime actual proof.
ORIGINAL_ENVELOPE_SHA = '6902af1cb866fde3e6e01a0cd21cac7642db9c20de1bbdb86d18d72646371546'
ENVELOPE_SHA = '32f52ab0b884ef3b4bb83141d5a20fe3e6883db25f6235399bda202ddaac75ce'
REPAIR_LOCAL = LOCAL / 'cache-repair-r1'
ATTEMPT = REPAIR_LOCAL / 'attempt-r1'


def authority():
    parent_authority()
    require(sha(ROOT / ENVELOPE) == ENVELOPE_SHA, 'CACHE_REPAIR_AUTHORITY_BYTES')
    envelope = read(ROOT / ENVELOPE)
    owner = envelope['owners']['server2']
    require(envelope['instruction_id'] == NONCE
        and owner['session'] == SESSION and owner['task_id'] == TASK
        and owner['parent_task'] == PARENT_TASK and owner['model'] == 'gptj',
        'CACHE_REPAIR_OWNER_SCOPE')
    require(envelope['resources']['project_task_cap_server2'] == 2
        and envelope['implementation_order'][0] == 'cached singleton→equal-token-length batch 구현 및 CPU 검산'
        and 'plan_vs_actual_receipt' in envelope['qualification'],
        'CACHE_REPAIR_ORDER_RESOURCE_SCOPE')
    return envelope


def enabled(config):
    return isinstance(config.get('generation'), dict) \
        and isinstance(config['generation'].get('repair'), dict)


def identity(config):
    if enabled(config):
        require(config['task_id'] == TASK and config['instruction_id'] == NONCE
            and config['parent_task_id'] == PARENT_TASK, 'CACHE_REPAIR_PROFILE_IDENTITY')
        return TASK, NONCE
    require(config['task_id'] == PARENT_TASK and config['instruction_id'] == PARENT_NONCE,
        'PARENT_GENERATION_PROFILE_IDENTITY')
    return PARENT_TASK, PARENT_NONCE


def ready(config):
    """Submission readiness, deliberately not an actual GPU certification gate."""
    from .generation_plan import ready as parent_ready
    parent_ready(config)
    identity(config)
    authority()
    repair = config['generation']['repair']
    verify(repair['old_complete_case_inventory'])
    require(repair['status'] == 'PLAN_BOUND_NOT_ACTUAL_PASS'
        and repair['actual_qualification_status'] == 'NOT_RUN'
        and repair['qualification_in_first_replacement_job'] is True,
        'CACHE_REPAIR_PLAN_VS_ACTUAL')
    plan = read(verify(repair['qualification_plan']))
    cohort = read(verify(repair['qualification_cohort']))
    from .generation_cache_qualification import validate_plan
    validate_plan(plan, cohort)
    shared_plan = read(verify(repair['shared_qualification_plan']))
    require(shared_plan == cohort['shared_plan']
        and digest(shared_plan) == repair['shared_qualification_plan_sha256']
        == plan['shared_plan_sha256'], 'CACHE_REPAIR_EXACT_SHARED_PLAN')
    require(digest(plan) == repair['qualification_plan_sha256']
        and plan['shared_source_sha'] == config['generation']['source_sha'],
        'CACHE_REPAIR_PLAN_SOURCE_HASH')
    require(set(config['arm_configs']) == set(ARMS)
        and config['noCP'] is True and config['z_disk_cache'] is False
        and config['exact_resume'] == 'NOT_AVAILABLE', 'CACHE_REPAIR_UNCHANGED_SCIENCE')
    require(config['resources']['gpu'] == 1 and config['resources']['cpu'] == 6
        and config['resources']['host_mib'] == 59392, 'CACHE_REPAIR_RESOURCE_PROFILE')
    return True
