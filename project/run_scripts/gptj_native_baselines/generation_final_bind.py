"""Cheap final-only configuration port from exact r2: no new model/text/input."""
import copy
from .generation_common import ROOT, LOCAL, POLICY, member, read, require, stat_seal, write
from .generation_cache_reuse import semantic_identity
from .generation_final_common import AUTHORITY, SCHEDULE, FINAL_LOCAL, PREPARATION, ATTEMPT, ready, authority


def bind():
    authority()
    require(not PREPARATION.exists() and not ATTEMPT.exists(), 'FINAL_BIND_CREATE_ONCE')
    old_path = LOCAL / 'cache-repair-r2/preparation-r1/config.json'
    original = read(old_path)
    require(member(old_path)['sha256'] == '1924773f0bbb60addc0a9d1ace8281691a625da96a2099053a4c44e8a920411c',
        'FINAL_BIND_EXACT_PRIOR_CONFIG')
    for value in original['assets'] + original['runtime']['members'] + [original['observer_identity']]:
        stat_seal(value)
    current = copy.deepcopy(original)
    current.update(attempt=str(ATTEMPT), registration_profile='final-v1',
        tracking_attempt='final-generation-v1', manual_schedule_authority=member(ROOT / AUTHORITY),
        previous_generation_heavy_registration=str(LOCAL / 'cache-repair-r2/attempt-r1'))
    current['previous_generation_policy']=copy.deepcopy(original['generation_policy'])
    current['generation_policy']=member(ROOT/POLICY)
    gen = current['generation']
    gen.update(evaluation_schedule=SCHEDULE, W0_generation_enabled=False,
        intermediate_generation_enabled=False, final_generation_requests=2000,
        qualification_owner='BASE_MEMIT', W0_cache=str(ATTEMPT / 'UNUSED-W0-generation-cache'))
    repair = gen['repair']
    repair.update(qualification_receipt_path=str(ATTEMPT / 'BASE_MEMIT/qualification.json'))
    for key in ('compatibility_manifest_path','old_complete_case_inventory',
                'old_cold_observation_guard','old_w0_reuse_binding','old_w0_reuse','old_attempt'):
        repair.pop(key, None)
    repair.update(old_W0_generation_reuse_enabled=False,
        final_scientific_generation_calls_per_arm=1,
        full_W0_scientific_generation_calls=0)
    require(semantic_identity(current) == semantic_identity(original),
        'FINAL_BIND_NATIVE_MODEL_INPUT_PRECISION_UNCHANGED')
    ready(current)
    write(PREPARATION / 'config.json', current)
    write(PREPARATION / 'binding.json', dict(status='FINAL_ONLY_PLAN_BOUND_NOT_ACTUAL_PASS',
        previous_config=member(old_path), config=member(PREPARATION / 'config.json'),
        schedule=SCHEDULE, target_request_fits_unchanged=True,
        model_loads=0, generation_forwards=0, raw_relabelled=False, noCP=True,
        qualification_PLAN_reused_readonly=True, actual_GPU_qualification=False))
    return current


if __name__ == '__main__':
    bind()
