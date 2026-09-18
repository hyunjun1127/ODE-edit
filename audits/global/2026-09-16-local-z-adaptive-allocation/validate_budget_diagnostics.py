"""누적 비용 해석의 반례와 계측 계약을 CPU에서 검산한다."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = '2026-09-16-local-z-adaptive-allocation'
contract_path = ROOT / 'plans/global' / f'{BASE}-contract-v1.json'
spec_path = ROOT / 'plans/global' / f'{BASE}-budget-diagnostics-v1.md'
contract = json.loads(contract_path.read_text())
budget = contract['budget_diagnostics']
checks = {}


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks[name] = 'PASS'


def kl_bernoulli(p, q):
    return p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q))


check('no_hard_budget_invented', budget['hard_budget_enabled'] is False
      and budget['layer_norm_caps'] is None and budget['KL_ceiling'] is None)
check('W0_campaign_unchanged', contract['experiment']['new_batches'] == 70
      and contract['experiment']['entry'] == 'common pre-edit W0'
      and budget['new_editing_chains'] == 0)
check('observer_not_controller', budget['candidate_observer']['influences_runtime_selection'] is False)
observer = budget['candidate_observer']
check('observer_count_current', (6-1+7-1)*3 == observer['max_additional_current_candidate_states'] == 33)
check('observer_count_all', 33*1200 == observer['max_additional_current_prompt_states']
      and 22*1200 == observer['max_additional_fixed_cohort_prompt_states']
      and (33+22)*1200 == observer['max_additional_prompt_states_total'] == 66000)

# 같은 edit response라도 두 layer를 쓰면 locality 변화가 작을 수 있다.
single = (1.0, 0.0)
split = (.5, .5)
check('more_layers_can_cancel_locality_response', sum(single) == sum(split)
      and (split[0]-split[1])**2 < (single[0]-single[1])**2)

# Opposite writes have positive path cost but tiny net drift.
steps = [1., -.9]
check('sum_norm_squared_is_not_net_drift', sum(d*d for d in steps) > sum(steps)**2)
A, d, C0 = 1., -.75, 2.
actual_increment = (A+d)**2*C0 - A*A*C0
cross_plus_self = 2*d*C0*A + d*d*C0
check('signed_cumulative_mapping_identity', math.isclose(actual_increment, cross_plus_self)
      and actual_increment < 0)

W0, dW, k0, kt = 2., .3, 1., 1.4
direct, upstream, interaction = dW*k0, W0*(kt-k0), dW*(kt-k0)
check('L8_key_drift_decomposition', math.isclose((W0+dW)*kt-W0*k0, direct+upstream+interaction))

endpoint_kl = kl_bernoulli(.5, .7)
step_kl_sum = kl_bernoulli(.5, .6) + kl_bernoulli(.6, .7)
check('step_KL_not_additive_cumulative_drift', not math.isclose(endpoint_kl, step_kl_sum)
      and step_kl_sum < endpoint_kl)
drifts = [0., .1, .3, .2]
increments = [b-a for a, b in zip(drifts, drifts[1:])]
increase = sum(max(x, 0) for x in increments)
decrease = sum(max(-x, 0) for x in increments)
check('signed_output_ledger_telescopes', math.isclose(increase-decrease, drifts[-1]-drifts[0]))

# M=diag(n,0) has rank1 for every n>0 but current-key fitting attenuation varies.
response_n1 = 1/(1+1+1)
response_n10 = 1/(1+10+1)
check('same_history_rank_different_realization', response_n1 > response_n10)
c4 = [.2, .3]
c48 = c4 + [.18, .25]
check('nested_same_state_frontier_gain_nonnegative', min(c4)-min(c48) >= 0)

result = {
    'status':'PASS_CPU_DIAGNOSTIC_DEFINITIONS_ONLY',
    'new_model_or_gpu_experiments':False,
    'intrinsic_capacity_estimated':False,
    'check_count':len(checks), 'checks':checks,
    'illustrative_numbers_not_model_measurements':{
        'endpoint_bernoulli_KL':endpoint_kl,'stepwise_KL_sum':step_kl_sum,
        'signed_mapping_increment':actual_increment,
        'L8_module_difference':direct+upstream+interaction,
    },
    'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [contract_path, spec_path, Path(__file__)]},
}
out = Path(__file__).with_name('budget-checks.json')
out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'output':str(out)}, ensure_ascii=False))
