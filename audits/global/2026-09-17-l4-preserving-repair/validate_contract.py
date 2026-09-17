"""설계 계약·수치 scale의 CPU 검사. 실제 모델 실행 검증이 아니다."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT/'plans/global/2026-09-17-l4-preserving-batch-repair-contract-v1.json'
CELLS = ROOT/'plans/global/2026-09-17-l4-preserving-batch-repair-cells-v1.csv'


def run():
    d = json.loads(PLAN.read_text())
    cells = list(csv.DictReader(CELLS.open()))
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append({'name': name, 'pass': True})
    check('design_only', not d['model_runner_implemented'] and d['GPU_jobs_submitted']==0)
    check('write_at_local4_full', d['writer']['layer']==d['writer']['target_location']==4 and d['writer']['native_gate']==1)
    check('no_target_z8', d['writer']['repair_target_new_compute_z_calls']==0)
    check('no_native8_projection_or_target_ball', not d['writer']['repair_native_P8'] and not d['writer']['repair_edit_Z_ball'])
    check('history_once_no_inner', d['writer']['M4_append_per_committed_batch']==1 and d['writer']['inner_history_append']==0)
    check('W0_cold', d['capsule']['entry']=='common pretrained W0' and d['capsule']['M4_initial']=='zeros')
    check('all_five_new_cold', len(cells)==5 and all(c['start']=='W0' and c['execution_status']=='PLANNED_NOT_SUBMITTED' for c in cells))
    check('first1000_fixed_order', d['capsule']['ordinal_half_open']==[0,1000] and all(int(c['requests'])==1000 for c in cells))
    check('50_batches_planned', sum(int(c['batches']) for c in cells)==50)
    check('arms_match', [c['arm'] for c in cells]==d['plans']['main_arms'])
    check('no_P_N_feedback', not d['data']['official_P_or_N_in_controller'])
    check('no_paraphrase_training_set_generation_or_guard', all(d['data'][k] is False for k in ('paraphrase_training_set','paraphrase_generation','paraphrase_online_guard')))
    check('only_rewrite_guard_panels', d['data']['online_guard_panels']==d['quality']['panels']==['Current-R','Past-R'])
    check('official_PS_observer_and_evaluation_retained', d['data']['official_paraphrase_role']=='observer_only_after_selection_seal' and d['quality']['official_PS_evaluation_required'])
    check('Past_postnative_for_zero_feasibility', d['data']['Past_reference']=='post-native WN')
    check('no_plateau', not d['quality']['mean_plateau'])
    check('success_ID_not_count', d['quality']['WN_strict_success_ID_subset'])
    check('heldout_PS_not_guaranteed', not d['quality']['official_PS_online_guarantee'])
    check('no_quality_relaxation', not d['quality']['posthoc_tolerance_relaxation'])
    check('dimension_bounded', d['direction_space']['max_dimensions']==3 and d['direction_space']['first_batch_max_dimensions']==2)
    check('gradient_JVP_costs_match_rewrite_only_basis', d['cost_upper_structure']['gradient_panel_sweeps_max']==d['cost_upper_structure']['directional_JVP_sweeps_max']==3 and d['cost_upper_structure']['paraphrase_generation_or_guard_calls']==0)
    check('no_first_gradient_KL_equality_or_large_PCG', not d['response_model']['KL_first_gradient_equality'] and not d['response_model']['large_PCG'])
    check('correct_GN_probability', d['response_model']['base_gradient_teacher']=='p0' and d['response_model']['base_GN_probability']=='p_N at anchor')
    check('zero_is_part_of_controller', d['controller']['zero_candidate_always_available'])
    check('six_probes_one_accept', d['controller']['max_finite_probes']==6 and d['controller']['max_accepted_repairs_per_batch']==1)
    check('no_relinearization_claim', d['controller']['no_relinearization_v1'])
    check('zero_pp_scientific_goal', d['plans']['scientific_RS_PS_loss_allowance_pp']==0)
    # 명시한 최소 diagonal shift와 whitening trust bound를 대각 SPD fixture로 확인.
    rng = random.Random(20260917)
    max_condition = d['response_model']['max_condition_number']
    largest_ratio = 0.0
    for _ in range(100):
        hmax = 10**rng.uniform(-8, 2)
        hmin = hmax*10**rng.uniform(-12, 0)
        shift = max(0.0, (hmax-max_condition*hmin)/(max_condition-1))
        assert (hmax+shift)/(hmin+shift) <= max_condition*(1+1e-12)
        dims = rng.randrange(1,d['direction_space']['max_dimensions']+1)
        drift = 10**rng.uniform(-6, -1)
        radius = math.sqrt(2*drift/dims)
        u = [rng.uniform(-radius,radius) for _ in range(dims)]
        energy = 0.5*sum(x*x for x in u)
        assert energy <= drift*(1+1e-12)
        largest_ratio = max(largest_ratio,energy/drift)
    check('100_damping_and_trust_box_fixtures', True)
    result = {'status':'PASS_CONTRACT_AND_CPU_SCALE_ONLY','checks':checks,'count':len(checks),
              'model_calls':0,'GPU_calls':0,'max_sampled_predicted_energy_to_drift':largest_ratio,
              'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (PLAN,CELLS)],
              'limits':['실제 Llama gradient/JVP/Fisher·production QP·RS/PS/NS·walltime은 검증하지 않았다.',
                        '실제 model numerical lock은 아직 만들어지지 않았다. 별도 paraphrase corpus는 사용하지 않는다.']}
    (Path(__file__).parent/'contract-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'status':result['status']}))


if __name__=='__main__': run()
