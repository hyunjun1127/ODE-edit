"""Validate linked design budgets, arm cells and CPU reference; no model imports."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
PREFIX = ROOT / 'plans/global/2026-09-17-sequential-local-z-allocation'
C = json.loads(Path(str(PREFIX) + '-contract-v2.json').read_text())
checks = []


def require(label, condition):
    checks.append({'check': label, 'pass': bool(condition)})
    if not condition:
        raise AssertionError(label)


def main():
    arms = C['arms']
    by_id = {a['id']: a for a in arms}
    require('six unique arms', len(arms) == len(by_id) == 6)
    require('model runner explicitly pending', not C['model_runner_implemented'] and C['GPU_jobs_submitted'] == 0)
    require('all start W0', C['capsule']['initial_weights'] == 'pretrained_W0' and not C['capsule']['warm_checkpoint_allowed'] and all(a['cold_start'] for a in arms))
    require('cold stream coordinates', C['capsule']['ordinal_half_open'] == [0, 1000] and all(a['batches'] * a['batch_size'] == 1000 for a in arms))
    require('no paraphrase controller data', all(C['data'][k] is False for k in ['paraphrase_target_set', 'paraphrase_generation', 'paraphrase_online_guard', 'future_request_access']))
    require('local target uses actual prefix', C['path']['target_location'] == 'write_layer' and C['path']['target_state'] == 'actual current prefix' and C['path']['earlier_gate_change_invalidates_later_targets'])
    require('no entry bank or terminal redistribution', not C['path']['entry_state_frozen_target_bank'] and not C['path']['terminal_z_distribution'])
    require('prefix may be quality infeasible', not C['path']['prefix_quality_pruning'])
    require('independent gate bounds', C['path']['gate_bounds'] == [0, 1] and not C['path']['sum_to_one'])
    require('fixed native 25 evaluations 24 updates', C['native']['v_num_grad_steps'] == C['native']['max_loss_evaluations_per_request'] == 25 and C['native']['max_Adam_updates_per_request'] == 24 and not C['native']['adaptive_target_iterations'])
    require('no quality plateau', not C['quality']['mean_plateau'])
    require('strict and pair quality guards', all(C['quality'][k] for k in ['current_strict_subset', 'current_pair_subset', 'past_strict_subset', 'past_pair_subset']))
    require('quality limits and limits of guarantees', C['quality']['epsilon_E'] == C['quality']['epsilon_H'] == 1e-4 and C['quality']['epsilon_B'] == 1e-6 and not C['quality']['PS_online_guarantee'])
    require('raw static arm is unguarded', by_id['F48']['policy'] == 'raw_fixed' and not by_id['F48']['quality_commit_guard'] and by_id['F48']['fixed_gates'] == [.75, .5])
    require('raw static exception is explicit', C['quality']['raw_fixed_baseline_exception'] == ['F48'])
    require('continuous arms protected', all(by_id[k]['quality_commit_guard'] for k in ['C4', 'C48', 'C45678']))
    require('five-layer candidate set', by_id['C45678']['layers'] == [4, 5, 6, 7, 8])
    require('solver pinned and inward coordinates', C['optimizer']['version'] == '1.15.3' and C['optimizer']['coordinates'] == 'u=1-a')
    require('bounds before transform', C['optimizer']['raw_coordinate_bounds_checked_before_transform'])
    require('margin unit scales', C['optimizer']['pair_margin_scale'] == .05 and C['optimizer']['strict_logit_margin_scale'] == 1)
    require('observed incumbent required', C['selector']['scipy_return_not_auto_committed'])
    require('exact deletion and no gate snapping', C['pruning']['order'] == [8, 7, 6, 5] and C['pruning']['never_snap_small_gates'])
    require('same history protocol', all(a['history_layers'] == C['history']['layers_for_all_arms'] == [4, 5, 6, 7, 8] for a in arms) and C['path']['inner_history_appends'] == 0)
    b = C['budget']
    require('fit reserve sum', b['search_suffix_fits'] + b['pruning_reserved_suffix_fits'] == b['total_suffix_fits'])
    require('endpoint reserve sum', b['search_scored_endpoints'] + b['pruning_reserved_endpoints'] == b['total_extra_scored_endpoints'])
    require('whole native fit Adam reservation', b['pre_fit_Adam_reservation'] == C['capsule']['batch_size'] * C['native']['max_Adam_updates_per_request'])
    require('no mid-target truncation', not b['mid_target_truncation'] and b['charge_actual_Adam_and_release_unused'])
    target_cap = (1 + b['total_suffix_fits']) * C['capsule']['batch_size']
    adam_cap = b['pre_fit_Adam_reservation'] + b['extra_Adam_updates']
    require('target cap arithmetic', target_cap == b['max_total_targets_per_cont_multilayer_batch'])
    require('Adam cap arithmetic', adam_cap == b['max_total_Adam_per_cont_multilayer_batch'])
    require('loss cap arithmetic', target_cap + adam_cap == b['max_total_loss_evaluations_per_cont_multilayer_batch'])
    require('score cap arithmetic', 1 + b['total_extra_scored_endpoints'] == b['max_total_scores_per_cont_batch'])
    require('C4 has no suffix target cost', by_id['C4']['max_target_calls'] == 100 and by_id['C4']['max_solves'] == 1)
    require('multilayer continuous caps shared', all((by_id[k]['max_target_calls'], by_id[k]['max_Adam'], by_id[k]['max_scores']) == (target_cap, adam_cap, 1+b['total_extra_scored_endpoints']) for k in ['C48', 'C45678']))
    with Path(str(PREFIX) + '-cells-v2.csv').open() as f:
        rows = list(csv.DictReader(f))
    require('cells exactly cover arms', {r['cell_id'].split('-')[0] for r in rows} == set(by_id) and len(rows) == 6)
    for r in rows:
        a = by_id[r['cell_id'].split('-')[0]]
        require(a['id'] + ' cell parity', r['start'] == 'W0' and r['status'] == 'PLANNED_NOT_SUBMITTED' and int(r['batches']) == a['batches'] and list(map(int, r['allowed_layers'].split('|'))) == a['layers'] and int(r['max_Adam_per_batch']) == a['max_Adam'] and (r['quality_commit_guard'] == 'true') == a['quality_commit_guard'])
    u = C['science_upper_bounds']
    for dst, src in [('target_calls','max_target_calls'), ('Adam_updates','max_Adam'), ('solves','max_solves'), ('scored_endpoints','max_scores')]:
        require('science total ' + dst, u[dst] == sum(a[src] * a['batches'] for a in arms))
    require('science loss total', u['loss_evaluations'] == u['target_calls'] + u['Adam_updates'])
    require('science history total', u['history_appends'] == sum(len(a['history_layers']) * a['batches'] for a in arms))
    # Import the owned CPU control module only. scipy/torch/model code is not imported.
    spec = importlib.util.spec_from_file_location('allocation_design_reference_check', AUDIT/'controller_reference.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    limits = module.Limits()
    for field, key in [('max_suffix_fits','total_suffix_fits'), ('max_endpoints','total_extra_scored_endpoints'), ('reserve_suffix_fits_for_pruning','pruning_reserved_suffix_fits'), ('reserve_endpoints_for_pruning','pruning_reserved_endpoints'), ('max_extra_adam','extra_Adam_updates'), ('max_adam_per_suffix_fit','pre_fit_Adam_reservation')]:
        require('reference config ' + field, getattr(limits, field) == b[key])
    require('reference solver call cap', limits.max_search_proposals == C['optimizer']['maxiter'])
    require('reference canonical guard disabled', not limits.guard_canonical_mean)
    evidence = json.loads(Path(C['artifacts']['evidence']).read_text())
    require('structured convergence evidence exists', all(k in evidence for k in ['sources','step_groups','convergence_groups','residual_B1','precision','limits']))
    for path in [Path(C['artifacts']['design']), AUDIT/'convergence-evidence-ko.md']:
        for target in re.findall(r'\]\((/[^)]+)\)', path.read_text()):
            require('local link ' + target, Path(re.sub(r':\d+$', '', target)).exists())
    files = [Path(str(PREFIX)+s) for s in ['-design-v2.md','-contract-v2.json','-cells-v2.csv']]
    files += [AUDIT/n for n in ['convergence-evidence.json','convergence-evidence-ko.md','controller_reference.py','test_controller_reference.py','cpu-validation.json','validate_design.py']]
    result = {'status':'PASS','check_count':len(checks),'checks':checks,'scope':'design consistency and CPU reference configuration; no real model validation','GPU_calls':0,'science_upper_bounds':u,'artifacts':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (AUDIT/'design-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'GPU_calls':0}))


if __name__ == '__main__':
    main()
