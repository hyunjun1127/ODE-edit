"""Meaningful CPU checks for the spectral selector and execution budgets."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from selector_reference import frontier, choose, quadratic_second_scale

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = ROOT/'plans/global/2026-09-20-en-adaptive-nullspace-contract-v1.json'
config = json.loads(CONTRACT.read_text())
checks = {}

def check(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError((name,detail))
    checks[name] = dict(passed=True, detail=detail)


# Compare cumulative scalar statistics against explicit matrix projectors.
rng = np.random.default_rng(20260920)
V = np.linalg.qr(rng.normal(size=(9,6)))[0]
U = np.linalg.qr(rng.normal(size=(6,6)))[0]
Z = np.linalg.qr(rng.normal(size=(5,5)))[0]
sigma = np.array([.001,.01,.1,.8,3.])
Kbar = V@U[:,:5]@np.diag(sigma)@Z.T
G = rng.normal(size=(4,9))
native = rng.normal(size=(4,9))@V@V.T
energies = ((G@V@U[:,:5])**2).sum(axis=0)
exact = float(np.linalg.norm(G@V@U[:,5:])**2)
rho = np.linalg.norm(native)
AN = np.linalg.norm(native@Kbar)
rows = frontier(sigma**2, energies, exact_energy=exact, loss=.7,
                native_norm=rho, native_action=AN, epsilon=.05)
errors = []
for r in rows:
    k = r['released_modes']
    S = np.concatenate((U[:,5:],U[:,:k]),axis=1)
    Q = V@S@S.T@V.T
    H = G@Q
    D = -r['eta']*H
    errors.extend([abs(np.linalg.norm(H)**2-r['gradient_energy']),
                   abs(np.linalg.norm(H@Kbar)**2-r['unit_coefficient_action_squared']),
                   abs(-np.sum(G*D)-r['predicted_decrease'])])
    check(f'budget_at_boundary_{k}', np.linalg.norm(D@Kbar)<=.05*AN+1e-12
          and np.linalg.norm(D)<=rho+1e-12)
check('spectrum_matches_explicit_projectors',max(errors)<1e-10,max(errors))
check('selector_matches_direct_lexicographic_choice',
      choose(rows)['released_modes']==min(rows,key=lambda r:(
          -round(r['predicted_decrease'],12),round(r['correction_norm'],12),
          round(r['response_norm'],12),r['released_modes']))['released_modes'])

# A deterministic example selects exact / weak / all as the response budget opens.
example = []
for eps in [.001,.05,1.]:
    f = frontier([.01,100.],[4.,81.],exact_energy=1.,loss=.1,
                 native_norm=1.,native_action=1.,epsilon=eps)
    example.append(choose(f))
check('budget_changes_selected_space',[r['released_modes'] for r in example]==[0,1,2],
      [{k:r[k] for k in ['epsilon','released_modes','predicted_decrease','correction_norm','response_norm']}
       for r in example])

# Unit changes of all keys must preserve eta and rank when native action scales too.
scaled = frontier(49*sigma**2,energies,exact_energy=exact,loss=.7,
                  native_norm=rho,native_action=7*AN,epsilon=.05)
check('key_units_do_not_change_selection',choose(rows)['released_modes']==choose(scaled)['released_modes']
      and np.allclose([r['eta'] for r in rows],[r['eta'] for r in scaled],rtol=1e-12,atol=1e-14))

# Duplicating each key with half the weight is the same covariance.
duplicated = np.concatenate([Kbar/np.sqrt(2),Kbar/np.sqrt(2)],axis=1)
check('weighted_duplicate_invariance',np.allclose(duplicated@duplicated.T,Kbar@Kbar.T,atol=1e-12))

zero_response = frontier([1.],[1.],exact_energy=1.,loss=.1,
                         native_norm=1.,native_action=0.,epsilon=.05)
check('zero_native_action_keeps_only_zero_response_steps',
      zero_response[0]['eta']>0 and zero_response[1]['eta']==0)
zeros = frontier([1.],[0.],exact_energy=0.,loss=.1,
                 native_norm=1.,native_action=1.,epsilon=.05)
check('zero_gradient_has_no_nan',all(r['eta']==0 and r['predicted_decrease']==0 for r in zeros))
repeated = frontier([1.,1.,2.],[1.,2.,3.],exact_energy=1.,loss=1.,
                    native_norm=1.,native_action=1.,epsilon=.05)
check('equal_eigenvalue_group_not_split',[r['released_modes'] for r in repeated]==[0,2,3])
try:
    frontier([1.,1.],[1.,1.],exact_energy=1.,loss=1.,native_norm=1.,
             native_action=1.,epsilon=.05,group_ends=[1,2])
except ValueError:
    check('split_eigenvalue_group_rejected',True)
else:
    check('split_eigenvalue_group_rejected',False)

# The second point solves a genuine one-dimensional quadratic, not a fitted test implementation.
q = quadratic_second_scale(2.,-2.,6.)
check('second_point_matches_known_quadratic_minimum',abs(q['scale']-1/6)<1e-14,q)
check('second_point_stays_inside_first_budget',0<q['scale']<1)
check('nonpositive_curvature_does_not_expand',
      quadratic_second_scale(2.,-1.,.5)['scale'] is None)
historical = quadratic_second_scale(.00112039035013,-.00112039035013,.00304722341764)
check('historical_first_trial_gives_new_unmeasured_point',0<historical['scale']<.25,
      dict(**historical,note='Scalar interpolation only; KL/RS/PS/NS at this point are unmeasured'))

cells = list(csv.DictReader((ROOT/config['artifacts']['cells']).open()))
caps = config['cost_caps_excluding_T0_history_observers']
for stage,prefix in [('B1','B1'),('ALL','through_B3')]:
    selected = [r for r in cells if r['stage']=='B1'] if stage=='B1' else cells
    for field,name in [('native_batch_passes','native_batch_passes'),
                       ('reference_gradient_sweeps','reference_gradient_sweeps'),
                       ('reference_candidate_sweeps_max','reference_candidate_forward_sweeps')]:
        total = sum(int(r[field]) for r in selected)
        check(f'cost_{prefix}_{field}',total==caps[f'{prefix}_{name}'],total)
check('B1_four_endpoints',[r['arm'] for r in cells if r['stage']=='B1' and r['arm']!='SHARED']
      ==['N4','EN_EXACT','EN_NUM','EN_ADAPT'])
for batch in [2,3]:
    check(f'S3_fixed_chains_batch_{batch}',
          {r['arm'] for r in cells if int(r['batch'])==batch}=={'N4','EN_EXACT','EN_ADAPT'})
check('reference_512_generation_256_no_PS_training',
      config['data']['reference_documents']==512 and config['data']['base_max_new_tokens']==256
      and not config['data']['paraphrase_training_set'])
check('single_layer_cold_no_extra_z',config['model']['parameter']=='model.layers.4.mlp.down_proj.weight'
      and not config['model']['warm_start'] and config['model']['extra_z_passes_per_correction']==0)
check('no_margin_controller',not config['selector']['output_margin_constraint']
      and not config['selector']['native_NLL_1e_minus_4_guard'])
check('history_all_fits_capacity',config['history']['B300_max_past']<=config['history']['buffer_capacity'])
design = ROOT/config['artifacts']['design']
links = [x for x in re.findall(r'\]\(([^)]+)\)',design.read_text()) if x.startswith('/')]
check('local_design_links_exist',all(Path(x).exists() for x in links))
payload = dict(status='PASS_CPU_DESIGN_NOT_NEURAL_VALIDATION',checks=checks,
               checks_count=len(checks),model_forward_backward=0,gpu_jobs_submitted=0,
               files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [CONTRACT,design,ROOT/config['artifacts']['cells'],HERE/'selector_reference.py']})
(HERE/'design-checks.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'status':payload['status'],'checks_count':len(checks),
                  'example_selected_modes':[r['released_modes'] for r in example],
                  'historical_interpolation':historical},indent=2))
