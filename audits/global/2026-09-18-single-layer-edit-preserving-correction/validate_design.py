"""Validate the design package only; never load a model or dispatch an experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path('/mnt/raid5/janghj/ODE-edit')
STEM = '2026-09-18-single-layer-edit-preserving-correction'
HERE = Path(__file__).resolve().parent
PLANS = ROOT / 'plans/global'
CONTRACT = PLANS / f'{STEM}-contract-v1.json'
CELLS = PLANS / f'{STEM}-cells-v1.csv'
DESIGN = PLANS / f'{STEM}-design-v1.md'


def main():
    c = json.loads(CONTRACT.read_text())
    rows = list(csv.DictReader(CELLS.open()))
    checks = []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    check('not_model_runner', c['model_runner_implemented'] is False)
    check('no_gpu_dispatch', c['gpu_jobs_submitted'] == 0)
    check('warm_checkpoint_forbidden', c['capsule']['warm_checkpoint_allowed'] is False)
    check('single_L4', c['capsule']['physical_layer_zero_based'] == 4)
    check('native_z_no_extra', c['native']['additional_z_for_correction'] == 0)
    check('no_paraphrase_set', c['data']['official_paraphrase_target_set'] is False)
    check('no_paraphrase_generation', c['data']['paraphrase_generation'] is False)
    check('no_online_official_eval', c['data']['official_P_N_online'] is False)
    check('no_eval_candidate_selection', c['data']['official_P_N_candidate_selection'] is False)
    check('no_guard_P_N', c['quality']['guard_accesses_P_or_N'] is False)
    check('no_future_requests', c['data']['future_request_access_by_optimizer'] is False)
    check('report_final_only', c['evaluation']['Report_schedule'].startswith('L final'))
    check('development_exposure_declared', c['data']['previously_exposed_requests'] is True)
    check('no_unseen_guarantee', c['data']['unseen_request_confirmatory_claim'] is False)
    check('W0_teacher', c['reference']['teacher'] == 'W0 fixed')
    check('full_vocab_teacher', c['reference']['topk'] is None)
    check('old_new_all_tokens', c['lock']['include_all_valid_input_tokens'] is True)
    check('not_all_history_exact', c['lock']['all_history_exact_lock'] is False)
    check('no_native_quadratic_guard', c['geometry']['same_native_quadratic_cost_guard'] is False)
    check('no_output_span_restriction', c['geometry']['output_span_restriction'] is False)
    check('main_one_gradient_eight_trials', c['optimizer']['main_gradient_rounds'] == 1 and c['optimizer']['main_trials'] == 8)
    check('actual_movement_armijo', 'W_trial-W_current' in c['optimizer']['actual_armijo'])
    check('ideal_actual_state_separate', 'ideal D_acc64' in c['optimizer']['EN_F4_state'])
    check('finite_polyak_status', 'STEP_SCALE_UNRESOLVED' in c['optimizer']['polyak_overflow'])
    check('R_quality_same_four_metrics', all(x in c['advancement']['R_to_L'] for x in ['RS', 'PS', 'joint', 'Pstrict']))
    check('fallback_history_commit', 'native fallback' in c['history']['native_append'])
    check('zero_inner_history_appends', c['history']['candidate_appends'] == 0)
    check('no_posthoc_PS_allowance', c['advancement']['no_posthoc_PS_allowance'] is True)
    check('no_sequential_iid_claim', c['statistics']['S_sequential_batch_independence_claim'] is False)
    check('unique_cell_ids', len({r['cell_id'] for r in rows}) == len(rows))

    keys = list(c['science_upper_bounds_excluding_T_observers_teacher_and_key_preparation'])
    total = {k: 0 for k in keys}
    for r in rows:
        tag = r['cell_id']
        n = int(r['batches'])
        b = int(r['batch_size'])
        fit = int(r['native_fit_batches'])
        check(tag + ':W0', r['start_state'] in ['W0', 'W0_shared_native'])
        check(tag + ':z', int(r['extra_z_calls']) == 0)
        check(tag + ':target_count', int(r['native_target_calls_cap']) == fit * b)
        check(tag + ':adam_count', int(r['native_adam_cap']) == fit * b * 24)
        check(tag + ':ordinal_count', int(r['ordinal_end']) - int(r['ordinal_start']) == n * b)
        expected_append = n if r['phase'] in ['S', 'R', 'L'] else 0
        check(tag + ':history', int(r['history_appends']) == expected_append)
        if r['arm'] in c['arms']:
            a = c['arms'][r['arm']]
            controllers = n if a['grad'] else 0
            check(tag + ':controllers', int(r['correction_controllers']) == controllers)
            check(tag + ':grad', int(r['gradient_sweeps_cap']) == controllers * a['grad'])
            check(tag + ':neural_grad', int(r['neural_gradient_sweeps_cap']) == (controllers * a['grad'] if a['neural'] else 0))
            check(tag + ':trials', int(r['trial_sweeps_cap']) == controllers * a['trials'])
        if r['phase'] == 'M':
            check(tag + ':independent_cold', r['reset_each_batch'] == 'true')
            check(tag + ':one_shared_fit', fit == (1 if r['arm'] == 'N4' else 0))
        if r['phase'] in ['S', 'R', 'L']:
            check(tag + ':own_trajectory', fit == n and r['reset_each_batch'] == 'false')
        if r['phase'] != 'T':
            for k in keys:
                total[k] += int(r[k])

    for k, expected in c['science_upper_bounds_excluding_T_observers_teacher_and_key_preparation'].items():
        check('sum:' + k, total[k] == expected, {'computed': total[k], 'contract': expected})
    stage_counts = {p: sum(r['phase'] == p for r in rows) for p in ['T', 'M', 'S', 'R', 'L']}
    check('stage_cells', stage_counts == {'T': 1, 'M': 110, 'S': 5, 'R': 4, 'L': 6}, stage_counts)
    check('EN_F4_M_only', all(r['phase'] == 'M' for r in rows if r['arm'] == 'EN-F4'))
    check('random_not_chain', all(r['phase'] == 'M' and int(r['gradient_sweeps_cap']) == 0 for r in rows if r['arm'].startswith('RAND')))

    text = DESIGN.read_text()
    check('math_fences_balanced', text.count('\\[') == text.count('\\]'))
    for raw in re.findall(r'\]\(([^)]+)\)', text):
        if raw.startswith(('http://', 'https://')):
            continue
        path = Path(raw) if raw.startswith('/') else DESIGN.parent / raw
        check('local_link:' + raw, path.exists())
    for value in [70000, 1680000, 700, 430, 460, 450, 3600]:
        check('design_mentions_total:' + str(value), f'{value:,}' in text or str(value) in text)

    geometry = json.loads((HERE / 'geometry_checks.json').read_text())
    check('geometry_tests_pass', geometry['status'] == 'PASS' and geometry['tests_run'] == 11 and geometry['failures'] == 0 and geometry['errors'] == 0)
    check('geometry_no_model_claim', geometry['model_forward_calls'] == 0 and geometry['GPU_calls'] == 0 and geometry['production_runner'] is False)
    for name, provenance in geometry['sources'].items():
        check('geometry_source_hash:' + name, hashlib.sha256((HERE / name).read_bytes()).hexdigest() == provenance['sha256'])
    check('CPU_scope_declared', 'CPU' in text and '미구현' in text)
    failed = [v for v in checks if not v['passed']]
    receipt = {
        'status': 'PASS' if not failed else 'FAIL',
        'scope': 'design consistency, data-role constraints, arithmetic and file links only',
        'model_runner_validated': False,
        'gpu_model_tests': 0,
        'checks_total': len(checks),
        'checks_passed': len(checks) - len(failed),
        'failed': failed,
        'stage_cells': stage_counts,
        'science_upper_bounds': total,
        'checks': checks,
    }
    (HERE / 'validation_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    artifacts = [DESIGN, CONTRACT, CELLS, HERE / 'geometry_reference.py', HERE / 'test_geometry_reference.py', HERE / 'geometry_checks.json', Path(__file__), HERE / 'validation_receipt.json']
    manifest = {'scope': 'new design package only', 'model_experiments': 0, 'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]}
    (HERE / 'artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['status', 'checks_total', 'checks_passed', 'failed', 'stage_cells', 'science_upper_bounds']}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
