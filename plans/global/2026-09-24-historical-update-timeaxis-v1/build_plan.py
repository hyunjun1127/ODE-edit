"""Materialize an observation-only experiment design; never loads a model or submits work.

Run from any directory with python3. Inputs are existing local metadata and the
frozen 10k dataset. The default output directory is this file's directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / 'local/datasets/counterfact-fixed-10k-v1'
AUDIT = ROOT / 'audits/global/2026-09-22-alphaedit-native-criticality-audit'
PKG = ROOT / 'local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package'
CP = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ARMS = ['BASE_ALPHAEDIT', 'BASE_MEMIT']
JOB_IDS = dict(zip(ARMS, ['42657', '42658']))
PILOT = {1: [1, 5, 10, 20, 50, 100], 20: [20, 30, 50, 70, 100], 50: [50, 60, 70, 90, 100]}
NAMESPACE = 'historical-update-timeaxis-v1'


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest(obj):
    return digest_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def rank(label, value):
    return digest_bytes(f'{NAMESPACE}|{label}|{value}'.encode())


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((AUDIT / 'target-native-execution.lock.json').read_text())
    dataset_receipts = []
    for member in lock['fixed_dataset_binding']['members']:
        p = DATA / Path(member['path']).name
        b = p.read_bytes()
        assert len(b) == member['bytes'] and digest_bytes(b) == member['sha256'], p
        dataset_receipts.append(dict(path=str(p), bytes=len(b), sha256=digest_bytes(b), status='LOCALLY_REHASHED'))
    records = json.loads((DATA / 'counterfact.json').read_text())
    sample = json.loads((DATA / 'source-sample.lock.json').read_text())
    assert len(records) == len(sample['records']) == 10000
    assert digest(sample['records']) == lock['sample_root'] == sample['ordered_root']
    groups = defaultdict(list)
    by_case = {}
    for ordinal, (r, s) in enumerate(zip(records, sample['records'])):
        assert r['case_id'] == s['case_id']
        assert digest(r) == s['raw_record_sha256']
        assert digest(r['requested_rewrite']) == s['request_sha256']
        assert (s['ordinal'], s['batch_index'], s['batch_ordinal']) == (ordinal, ordinal // 100 + 1, ordinal % 100)
        assert len(r['paraphrase_prompts']) == 2
        groups[s['subject_relation_group']].append((r, s))
        by_case[r['case_id']] = r
    conflicting = {g for g, rr in groups.items() if len({r['requested_rewrite']['target_new']['id'] for r, _ in rr}) > 1}
    same_batch_conflicts = set()
    for g, rr in groups.items():
        for b in {s['batch_index'] for _, s in rr}:
            if len({r['requested_rewrite']['target_new']['id'] for r, s in rr if s['batch_index'] == b}) > 1:
                same_batch_conflicts.add((g, b))

    facts = []
    for r, s in zip(records, sample['records']):
        b = s['batch_index']
        ci = next(i for i, end in enumerate(CP[1:]) if b <= end)
        start, end = CP[ci], CP[ci + 1]
        req = r['requested_rewrite']
        g = s['subject_relation_group']
        future_diff = [s2['batch_index'] for r2, s2 in groups[g]
                       if s2['batch_index'] > b and r2['requested_rewrite']['target_new']['id'] != req['target_new']['id']]
        censor = min(future_diff) if future_diff else None
        conflict_at_birth = (g, b) in same_batch_conflicts
        facts.append(dict(case_id=r['case_id'], ordinal=s['ordinal'], birth_batch=b,
            cohort_id=f'U{start:03d}_{end:03d}', cohort_index=ci, update_start=start, anchor_batch=end,
            age_at_anchor=end - b, subject_relation_group=g,
            subject_sha256=digest(req['subject']), relation_id=req['relation_id'],
            target_new_id=req['target_new']['id'], target_true_id=req['target_true']['id'],
            target_new_sha256=s['target_new_sha256'], target_true_sha256=s['target_true_sha256'],
            request_sha256=s['request_sha256'], raw_record_sha256=s['raw_record_sha256'],
            repeated_group=len(groups[g]) > 1, same_batch_conflict=conflict_at_birth,
            first_later_conflict_batch=censor if censor is not None else '',
            active_at_anchor=not conflict_at_birth and (censor is None or censor > end),
            active_at_100=not conflict_at_birth and censor is None,
            analysis_group_fold=int(rank('group-fold', g), 16) % 5))
    cohort_facts = {i: [r for r in facts if r['cohort_index'] == i] for i in range(12)}
    pilot_ids = set()
    for end in PILOT:
        rr = [r for r in facts if r['anchor_batch'] == end]
        for b in sorted({r['birth_batch'] for r in rr}):
            take = 100 if end == 1 else 10
            bb = sorted((r for r in rr if r['birth_batch'] == b), key=lambda r: (rank('pilot', r['case_id']), r['case_id']))
            pilot_ids.update(r['case_id'] for r in bb[:take])
    assert len(pilot_ids) == 300
    for r in facts:
        r['pilot_selected'] = r['case_id'] in pilot_ids
        r['pilot_rank_sha256'] = rank('pilot', r['case_id']) if r['pilot_selected'] else ''
    write_csv('fact-ledger.csv', facts)
    write_csv('pilot-panel.csv', [r for r in facts if r['pilot_selected']])
    cohorts = []
    for i, rr in cohort_facts.items():
        a, b = CP[i], CP[i + 1]
        cohorts.append(dict(cohort_id=f'U{a:03d}_{b:03d}', cohort_index=i, start=a, anchor=b,
            request_count=len(rr), prompt_count=3 * len(rr), batch_count=b - a,
            later_endpoint_count=11 - i, active_at_anchor=sum(r['active_at_anchor'] for r in rr),
            active_at_100=sum(r['active_at_100'] for r in rr),
            same_batch_conflict_requests=sum(r['same_batch_conflict'] for r in rr),
            equal_size_temporal_primary=20 <= b <= 90,
            case_order_sha256=digest([r['case_id'] for r in rr])))
    write_csv('cohorts.csv', cohorts)

    # Every model is keyed by the set of interval updates retained. A prefix is
    # an existing endpoint. Physical construction still uses endpoint subtraction,
    # not summation of a long delta chain; aliases require numerical fidelity.
    state_bank = {}
    prefix_to_t = {(1 << i) - 1: CP[i] for i in range(13)}

    def state(arm, t, removed=()):
        end_i = CP.index(t)
        mask = (1 << end_i) - 1
        for ci in removed:
            assert 0 <= ci < end_i
            mask &= ~(1 << ci)
        sid = 'W0' if mask == 0 else f'{arm}__{mask:03x}'
        if sid not in state_bank:
            actual_t = prefix_to_t.get(mask)
            state_bank[sid] = dict(state_id=sid, family='SHARED' if mask == 0 else arm,
                retained_interval_mask=f'{mask:03x}', kind='ACTUAL' if actual_t is not None else 'COUNTERFACTUAL',
                actual_checkpoint=actual_t if actual_t is not None else '',
                construction_endpoint=t, removed_cohort_indices=';'.join(map(str, removed)),
                used_in_main=False, used_in_pilot=False, used_in_pairs=False)
        return sid

    cells = []
    for arm in ARMS:
        for i, c in enumerate(cohorts):
            for t in CP[i + 1:]:
                ids = dict(M_t=state(arm, t), B_t=state(arm, t, (i,)),
                           M_b=state(arm, c['anchor']), B_b=state(arm, c['start']))
                is_pilot = c['anchor'] in PILOT and t in PILOT[c['anchor']]
                for sid in ids.values():
                    state_bank[sid]['used_in_main'] = True
                    state_bank[sid]['used_in_pilot'] |= is_pilot
                cells.append(dict(cell_id=f'{arm}__{c["cohort_id"]}__T{t:03d}', family=arm,
                    original_job=JOB_IDS[arm], cohort_id=c['cohort_id'], cohort_index=i,
                    update_start=c['start'], anchor=c['anchor'], eval_t=t,
                    age_from_anchor=t-c['anchor'], post_anchor_requests=100*(t-c['anchor']),
                    request_count=c['request_count'], prompts_per_request=3,
                    is_diagonal=t == c['anchor'], pilot=is_pilot,
                    pilot_request_count=100 if is_pilot else 0,
                    **{f'{k}_state': v for k, v in ids.items()}))
    write_csv('main-cells.csv', cells)
    write_csv('pilot-cells.csv', [c for c in cells if c['pilot']])

    # Metadata-only, outcome-independent contrast: among four equal-size future
    # blocks, choose largest/smallest same-relation competitor-target exposure.
    # The final (90,100] block is excluded: its endpoint removal is W90 and its
    # interaction simply repeats the already-measured chronological C100-C90.
    pair_rows = []
    exposures = []
    for b in [20, 30, 40, 50]:
        ui = CP.index(b) - 1
        panel = cohort_facts[ui]
        values = []
        for vend in [60, 70, 80, 90]:
            vi = CP.index(vend) - 1
            future = cohort_facts[vi]
            counts = Counter((r['relation_id'], r['target_new_id']) for r in future)
            raw_c = sum(counts[(r['relation_id'], r['target_true_id'])] for r in panel)
            raw_y = sum(counts[(r['relation_id'], r['target_new_id'])] for r in panel)
            item = dict(past_anchor=b, past_cohort=cohorts[ui]['cohort_id'], future_end=vend,
                future_cohort=cohorts[vi]['cohort_id'], competitor_exposure_sum=raw_c,
                target_exposure_sum=raw_y, cohort_requests=len(panel), future_requests=len(future),
                tie_rank=rank(f'pair-{b}', vend), past_cohort_index=ui, future_cohort_index=vi)
            values.append(item)
            exposures.append(item)
        ordered = sorted(values, key=lambda r: (r['competitor_exposure_sum'], r['tie_rank']))
        chosen = [('low_competitor_exposure', ordered[0]), ('high_competitor_exposure', ordered[-1])]
        assert chosen[0][1]['future_end'] != chosen[1][1]['future_end']
        for arm in ARMS:
            for label, v in chosen:
                vi = v['future_cohort_index']
                ids = dict(M=state(arm, 100), minus_U=state(arm, 100, (ui,)),
                           minus_V=state(arm, 100, (vi,)), minus_UV=state(arm, 100, (ui, vi)))
                for sid in ids.values():
                    state_bank[sid]['used_in_pairs'] = True
                pair_rows.append(dict(pair_id=f'{arm}__U{b:03d}__V{v["future_end"]:03d}__T100',
                    family=arm, selection=label, eval_t=100, request_count=len(panel),
                    prompts_per_request=3, **v, **{f'{k}_state': sid for k, sid in ids.items()}))
    write_csv('pair-candidates.csv', exposures)
    write_csv('pair-cells.csv', pair_rows)
    write_csv('state-bank.csv', list(state_bank.values()))

    # An existing state does not imply that its scores on another U's panel exist.
    # Include anchor-minus-U evaluations on not-yet-written cohort facts.
    tasks = {}

    def add_task(sid, ci, phase, pilot=False):
        rr = cohort_facts[ci]
        if pilot:
            rr = [r for r in rr if r['pilot_selected']]
        ids = [r['case_id'] for r in rr]
        case_hash = digest(ids)
        key = (sid, case_hash)
        if key not in tasks:
            tasks[key] = dict(task_id='SCORE_' + digest([sid, case_hash])[:20], state_id=sid,
                cohort_id=cohorts[ci]['cohort_id'], cohort_index=ci,
                case_selector='pilot_panel' if pilot and len(rr) != len(cohort_facts[ci]) else 'whole_cohort',
                request_count=len(rr), prompt_count=3*len(rr), target_sequences=6*len(rr),
                case_order_sha256=case_hash, pilot=False, main=False, pairs=False)
        tasks[key][phase] = True

    for c in cells:
        for slot in ['M_t_state', 'B_t_state', 'M_b_state', 'B_b_state']:
            add_task(c[slot], c['cohort_index'], 'main')
            if c['pilot']:
                add_task(c[slot], c['cohort_index'], 'pilot', pilot=True)
    for c in pair_rows:
        for slot in ['M_state', 'minus_U_state', 'minus_V_state', 'minus_UV_state']:
            add_task(c[slot], c['past_cohort_index'], 'pairs')
    write_csv('score-tasks.csv', list(tasks.values()))
    sentinels = []
    for ci, rr in cohort_facts.items():
        for r in sorted(rr, key=lambda r: (rank('fidelity', r['case_id']), r['case_id']))[:16]:
            sentinels.append(dict(case_id=r['case_id'], cohort_id=r['cohort_id'], cohort_index=ci,
                birth_batch=r['birth_batch'], rank_sha256=rank('fidelity', r['case_id'])))
    write_csv('fidelity-panel.csv', sentinels)

    # Freeze the exact evaluator bytes from the helper commit used by both runs.
    evidence = OUT / 'source-evidence'
    evidence.mkdir(exist_ok=True)
    code = ['project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py',
            'project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py',
            'project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py',
            'project/run_scripts/blue_alphaedit_sequential_comparison/integrity.py']
    sources = []
    for rel in code:
        b = subprocess.check_output(['git', 'show', lock['source_head'] + ':' + rel], cwd=ROOT)
        expected = next(m for m in lock['members'] if m['path'].endswith('/' + rel))
        assert digest_bytes(b) == expected['sha256'], rel
        dest = evidence / (rel.split('/')[-2] + '__' + Path(rel).name)
        dest.write_bytes(b)
        sources.append(dict(repo_path=rel, source_commit=lock['source_head'], sha256=digest_bytes(b),
                            snapshot_path=str(dest.relative_to(OUT)), match_runtime_lock=True))
    indexed = list(csv.DictReader((ROOT / 'experiment-reports/global/checkpoint-location-index-2026-09-18-v1/checkpoint-index.csv').open()))
    cp_rows = [r for r in indexed if r['experiment_family'] == 'fixed10k-native-baselines' and r['recorded_arm'] in ARMS]
    assert len(cp_rows) == 24
    bindings = []
    for r in cp_rows:
        bindings.append(dict(family=r['recorded_arm'], batch=int(r['recorded_batch']),
            reported_host=r['server'], reported_archive_path=r['path'],
            recorded_file_sha256=r['recorded_payload_sha256'], file_bytes=int(r['bytes']),
            local_path_present_at_design=Path(r['path']).is_file(),
            tensor_payload_rehashed_this_design=False,
            historical_validation=r['restore_validation'], provenance=r['recorded_hash_provenance']))
    write_csv('checkpoint-bindings.csv', bindings)
    weight_receipts = [r for r in csv.DictReader((PKG / 'new-checkpoint-tensors.csv').open()) if r['arm'] in ARMS]
    assert len(weight_receipts) == 120
    write_csv('checkpoint-tensor-hashes.csv', weight_receipts)
    compatibility = [r for r in csv.DictReader((PKG / 'new-source-config-compatibility.csv').open()) if r['arm'] in ARMS]
    w0files = json.loads((AUDIT / 'target-native-bindings.json').read_text())['files']
    runtime = json.loads(next(r['raw'] for r in w0files if r['path'].endswith('main-cell-1/runtime.json')))
    w0 = {k: {z: v[z] for z in ['sha256', 'shape', 'dtype']} for k, v in runtime['W0']['weights'].items()}
    dump('asset-bindings.json', dict(status='DESIGN_BINDINGS_NOT_GPU_VERIFIED',
        dataset=dataset_receipts, ordered_root=sample['ordered_root'], model_revision=lock['revision'],
        model_local_snapshot=str(Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots') / lock['revision']),
        w0_selected_weights=w0, baseline_runtime=compatibility, evaluator_sources=sources,
        checkpoints_present_locally=sum(r['local_path_present_at_design'] for r in bindings),
        checkpoint_bytes=sum(r['file_bytes'] for r in bindings),
        remaining_bindings=['runtime full-model shard identity', 'checkpoint payload rehash at execution',
                            'prompt/target token identities', 'GPU reconstructed endpoint fidelity']))
    main_states = [r for r in state_bank.values() if r['used_in_main']]
    pair_extra = [r for r in state_bank.values() if r['used_in_pairs'] and not r['used_in_main']]
    fact_cells = sum(c['request_count'] for c in cells)
    summary = dict(status='DESIGN_MATERIALIZED_NO_MODEL_FORWARD', family_count=2,
        cohort_count_per_family=12, main_logical_cells=len(cells), pilot_logical_cells=sum(c['pilot'] for c in cells),
        main_actual_states=sum(r['kind'] == 'ACTUAL' for r in main_states),
        main_counterfactual_states=sum(r['kind'] == 'COUNTERFACTUAL' for r in main_states),
        main_fact_time_rows=fact_cells, main_prompt_time_rows=3*fact_cells,
        main_target_sequences_upper_bound_without_score_reuse=3*fact_cells*2*2,
        main_unique_score_tasks=sum(r['main'] for r in tasks.values()),
        main_target_sequences_after_state_panel_dedup=sum(r['target_sequences'] for r in tasks.values() if r['main']),
        pilot_unique_cases=len(pilot_ids), pilot_fact_time_rows=100*sum(c['pilot'] for c in cells),
        pilot_target_sequences_without_reuse=100*sum(c['pilot'] for c in cells)*3*2*2,
        pair_cells=len(pair_rows), pair_additional_parameter_states=len(pair_extra),
        pair_max_additional_target_sequences_without_reuse=len(pair_rows)*1000*3*2*2,
        pair_additional_score_tasks=sum(r['pairs'] and not r['main'] for r in tasks.values()),
        pair_additional_target_sequences=sum(r['target_sequences'] for r in tasks.values() if r['pairs'] and not r['main']),
        equal_size_temporal_primary_cohorts=[r['cohort_id'] for r in cohorts if r['equal_size_temporal_primary']],
        subject_relation_groups=len(groups), repeated_subject_relation_groups=sum(len(rr)>1 for rr in groups.values()),
        conflicting_target_groups=len(conflicting), same_batch_conflicting_group_batches=len(same_batch_conflicts),
        same_batch_conflict_requests=sum(r['same_batch_conflict'] for r in facts),
        active_at_anchor_requests=sum(r['active_at_anchor'] for r in facts),
        active_at_100_requests=sum(r['active_at_100'] for r in facts),
        model_forward_calls=0, jobs_submitted=0, external_messages_sent=0,
        effect_sizes_available=False, metadata_outcome_used_for_selection=False)
    assert len(cells) == 156 and sum(c['pilot'] for c in cells) == 32
    assert summary['main_actual_states'] == 25 and summary['main_counterfactual_states'] == 132
    assert len(pair_rows) == 16
    assert all(c['eval_t'] >= c['anchor'] for c in cells)
    dump('plan-summary.json', summary)
    dump('source-receipts.json', dict(dataset=dataset_receipts, evaluator=sources,
        metadata=[dict(path=str(p), sha256=digest_bytes(p.read_bytes())) for p in [
            AUDIT/'target-native-execution.lock.json', AUDIT/'target-native-bindings.json',
            PKG/'new-checkpoint-tensors.csv', PKG/'new-source-config-compatibility.csv',
            ROOT/'experiment-reports/global/checkpoint-location-index-2026-09-18-v1/checkpoint-index.csv']]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
