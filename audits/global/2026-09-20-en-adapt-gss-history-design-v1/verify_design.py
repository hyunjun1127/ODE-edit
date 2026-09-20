"""CPU checks of gradient identities, subset objective, ledger and experiment cost."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from history_reference import (active_versions, factor_sketch, gss_prune,
                               history_weights, shared_cells)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
contract = ROOT/'plans/global/2026-09-20-en-adapt-gss-history-contract-v1.json'
cfg = json.loads(contract.read_text())
checks = {}


def check(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError((name,detail))
    checks[name] = dict(passed=True,detail=detail)


rng = np.random.default_rng(20260920)
a,k = rng.normal(size=(5,4)),rng.normal(size=(7,4))
basis = np.linalg.qr(rng.normal(size=(7,5)))[0]
p = basis@basis.T
maps = [(rng.normal(size=(5,3))/np.sqrt(3),rng.normal(size=(7,3))/np.sqrt(3)) for _ in range(2)]
sketched = factor_sketch(a,k,p,maps)
dense = np.concatenate([(o.T@(a@k.T@p)@v).ravel() for o,v in maps])/np.sqrt(2)
check('factor_sketch_equals_dense_gradient_projection',np.allclose(sketched,dense,rtol=1e-12,atol=1e-12))
other_a,other_k = rng.normal(size=(5,3)),rng.normal(size=(7,3))
inner = float(((a.T@other_a)*(k.T@p@other_k)).sum())
direct = float(((a@k.T@p)*(other_a@other_k.T@p)).sum())
check('exact_pair_inner_product_from_factors',abs(inner-direct)<1e-10)
changed_a = a+1
check('fixed_keys_do_not_make_gradient_state_invariant',
      not np.allclose(factor_sketch(changed_a,k,p,maps),sketched))
check('all_input_positions_matter',not np.allclose(factor_sketch(a[:,1:],k[:,1:],p,maps),sketched))

# At teacher equality, full-vocab KL has zero logit derivative, target NLL does not.
logits = np.array([1.,-.3,.2])
prob = np.exp(logits-logits.max());prob/=prob.sum()
kl_deriv = prob-prob.copy()
nll_deriv = prob-np.array([1.,0.,0.])
check('at_write_teacher_KL_zero_but_NLL_nonzero',np.linalg.norm(kl_deriv)==0 and np.linalg.norm(nll_deriv)>0)

features = rng.normal(size=(12,6))
ids = [f'fact-{i}' for i in range(12)]
result = gss_prune(features,ids,cap=6)
unit = features/np.linalg.norm(features,axis=1,keepdims=True)
for step,row in enumerate(result['trace']):
    ids_left = row['active']
    objectives = {j:float(np.linalg.norm(unit[[i for i in ids_left if i!=j]].sum(axis=0))**2) for j in ids_left}
    check(f'greedy_removal_minimizes_exact_one_step_surrogate_{step}',
          np.isclose(objectives[row['removed']],min(objectives.values()),rtol=1e-12,atol=1e-12))
    check(f'predicted_surrogate_matches_finite_removal_{step}',
          np.isclose(row['predicted_objective_after'],objectives[row['removed']],atol=1e-12))
check('bank_capacity_enforced',len(result['selected'])==6)
check('below_capacity_keeps_all_no_GSS',not gss_prune(features[:3],ids[:3],cap=6)['selection_exercised'])
opposite = np.array([[1.,0.],[1.,0.],[-1.,0.]])
kept = gss_prune(opposite,['a','b','c'],cap=2)['selected']
check('opposite_gradient_not_treated_as_duplicate',2 in kept and len(set(kept)&{0,1})==1)
scaled = features*np.exp(rng.normal(size=(12,1)))
check('positive_weights_cancel_in_cosine_selection',gss_prune(scaled,ids,cap=6)['selected']==result['selected'])
perm = rng.permutation(12)
permuted = gss_prune(features[perm],[ids[i] for i in perm],cap=6)
check('ID_ties_and_order_reproducible',{ids[i] for i in result['selected']}=={ids[perm[i]] for i in permuted['selected']})
zeros = gss_prune(np.zeros((7,3)),[str(i) for i in range(7)],cap=4)
check('zero_gradients_have_explicit_finite_fallback',len(zeros['selected'])==4 and zeros['zero_count']==7)
try:
    gss_prune(np.full((4,3),np.nan),list('abcd'),cap=2)
except ValueError:
    check('nonfinite_feature_rejected',True)
else:
    check('nonfinite_feature_rejected',False)

weights = history_weights([9,5,1],10)
check('history_weights_normalized_recent_larger',np.isclose(weights.sum(),1) and weights[0]>weights[1]>weights[2])
check('history_weight_ratio_bounded',weights.max()/weights.min()<=2)
check('uniform_ablation_and_equal_age_alias',np.array_equal(history_weights([1,1,1],2),history_weights([1,1,1],2,recency=False)))
check('empty_history_finite',history_weights([],1).size==0)
try:
    history_weights([3],3)
except ValueError:
    check('current_version_not_replayed_as_past',True)
else:
    check('current_version_not_replayed_as_past',False)

events = [dict(fact=('s','r'),target='old',batch=1,teacher='q1'),
          dict(fact=('s','r'),target='old',batch=3,teacher='q3')]
v = active_versions(events)[('s','r')]
check('same_target_repeat_keeps_teacher_and_birth',v['teacher']=='q1' and v['batch']==1)
events.append(dict(fact=('s','r'),target='new',batch=4,teacher='q4'))
v = active_versions(events)[('s','r')]
check('overwrite_leaves_only_new_target_version',v['target']=='new' and v['teacher']=='q4')

# Separately summing reference/history energy misses the cross term.
g_r = np.array([[1.,2.],[3.,4.]])
g_h = -0.5*g_r
check('combined_gradient_energy_requires_cross_term',
      np.isclose(np.linalg.norm(g_r+g_h)**2,np.linalg.norm(g_r)**2+np.linalg.norm(g_h)**2+2*(g_r*g_h).sum())
      and not np.isclose(np.linalg.norm(g_r+g_h)**2,np.linalg.norm(g_r)**2+np.linalg.norm(g_h)**2))

cells = shared_cells(final_batch=cfg['evaluation']['final_batch'])
path = ROOT/cfg['artifacts']['cells']
with path.open('w',newline='') as f:
    out = csv.DictWriter(f,fieldnames=list(cells[0]));out.writeheader();out.writerows(cells)
check('three_cold_chains_need_292_distinct_prefix_states',len(cells)==292)
check('reference_candidate_sweep_cap_584',sum(c['reference_candidate_sweeps_max'] for c in cells)==584)
check('GSS_first_exercised_B7',min(c['batch'] for c in cells if c['selection_exercised'])==7)
check('GSS_selection_sweep_cap_188',sum(c['selection_exercised'] for c in cells)==188)
check('GSS_NLL_fact_VJP_cap_115032',sum(c['history_NLL_fact_VJPs'] for c in cells)==115032)
check('only_B1_B2_are_preflight',[c['batch'] for c in cells if c['stage']=='B2_CHECK']==[1,2])
check('lifelong_immediately_continues_B3_through_B100',
      {c['batch'] for c in cells if c['stage']=='LIFELONG'}==set(range(3,101)))
check('no_intermediate_pilot_or_quality_gate',not cfg['evaluation']['intermediate_B300_B1000_pilots']
      and not cfg['evaluation']['performance_improvement_required_to_continue']
      and not cfg['evaluation']['first_overflow_is_gate'])
check('cold_prefix_continued_without_reexecution',cfg['evaluation']['continue_from_preflight_state']
      and not cfg['evaluation']['repeat_B1_B2_for_lifelong'] and cfg['model']['edit_requests']==10000)
check('all_512_reference_G256_and_no_PS_training',cfg['reference']['documents']==512
      and cfg['reference']['base_max_new_tokens']==256 and not cfg['history']['paraphrase_training'])
check('no_extra_z_or_margin_guard',all(c['extra_z']==0 for c in cells) and not cfg['controller']['output_margin_guard'])
check('pool_limit_and_bank_limit_distinct',max(c['history_forward_pool'] for c in cells)==612
      and cfg['history']['bank_capacity']==512)

design = ROOT/cfg['artifacts']['design']
links = [s for s in re.findall(r'\]\(([^)]+)\)',design.read_text()) if s.startswith('/')]
check('design_local_links_exist',all(Path(s).exists() for s in links))
payload = dict(status='PASS_CPU_DESIGN_NOT_NEURAL_VALIDATION',checks_count=len(checks),checks=checks,
               gpu_jobs_submitted=0,model_forward_backward_calls=0,
               dense_gradient_GB=612*4096*14336*4/1e9,sketch_MB=612*2048*4/1e6,
               hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [contract,design,path,HERE/'history_reference.py']})
(HERE/'design-checks.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'status':payload['status'],'checks_count':payload['checks_count'],
                  'shared_states':len(cells),'NLL_selection_fact_VJPs':sum(c['history_NLL_fact_VJPs'] for c in cells),
                  'dense_gradient_GB':payload['dense_gradient_GB'],'sketch_MB':payload['sketch_MB']},indent=2))
