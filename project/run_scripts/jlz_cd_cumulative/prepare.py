"""CPU-only exact prior input/runtime binding; no model/GPU or scheduler calls."""
import argparse
import csv
import json
import os
import platform
import shutil
import unicodedata
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding, asset_binding
from project.run_scripts.jlz_realization.observe import active_flags
from .common import *

MANIFEST_SHA = 'a67ab3941e21e32627aa06e3d502005bc7239111b2c6906fbd614f991a8205f0'
ORDERED_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'


def bind_prior_inputs(prior, oldlock):
    require(oldlock['source_commit'] == FROZEN and oldlock['config_sha256'] == sha(PRIOR / 'config.json'), 'PRIOR_EXECUTION_SOURCE')
    for row in oldlock['dependency_sources'] + oldlock['native_reference']:
        verify(row)
    runtime = runtime_binding()
    require((runtime['torch'], runtime['transformers']) == (prior['runtime']['torch'], prior['runtime']['transformers']), 'RUNTIME_VERSION_REUSE')
    old_runtime = {r['module']: r['sha256'] for r in prior['runtime']['source_members']}
    require(all(old_runtime.get(r['module']) == r['sha256'] for r in runtime['source_members']), 'RUNTIME_SOURCE_REUSE')
    old_assets = {r['path']: r for r in prior['assets']}
    assets = [asset_binding(r['path'], old_assets, member(PRIOR / 'config.json')) for r in prior['assets']]
    reuse = prior['qualification_reuse']
    science = []
    for item in reuse['science_members']:
        relative = item['relative']
        current = ROOT / relative
        require(current.is_file() and sha(current) == item['current']['sha256'], 'FROZEN_READONLY_SOURCE:' + relative)
        science.append(dict(relative=relative, prior=item['current'], current=member(current)))
    for relative in ('project/run_scripts/jlz_realization/observe.py', 'project/run_scripts/jlz_realization/inputs.py',
                     'project/run_scripts/jlz_pilot/prompts.py', 'scripts/fixed_counterfact.py'):
        old = PRIOR / 'source' / relative
        require(old.is_file() and sha(old) == sha(ROOT / relative), 'EXACT_INPUT_EVALUATOR_SOURCE:' + relative)
        science.append(dict(relative=relative, prior=member(old), current=member(ROOT / relative)))
    alignment_path = verify(prior['native_input_alignment'])
    observer_path = verify(prior['observer_identity'])
    alignment = json.loads(alignment_path.read_text())
    identities = json.loads(observer_path.read_text())
    require(len(prior['packs']) == len(alignment['packs']) == 20 and alignment['requests'] == 2000, 'PRIOR_ALL20_PACKS')
    require(prior['packs'] == alignment['packs'] and len(alignment['rows']) == 2000, 'PRIOR_NATIVE_ALIGNMENT')
    require(len(identities['rows']) == 26000 and len({r['identity'] for r in identities['rows']}) == 26000, 'PRIOR_OBSERVER_IDENTITY')
    require(identities['ordered_ids_sha256'] == prior['ordered_ids_sha256'] == ORDERED_SHA, 'PRIOR_ORDERED_INPUT')
    return runtime, assets, science, alignment_path, observer_path


def w0_reuse(prior, identities, records):
    """Bind raw endpoint, never infer identity from equal aggregate scores."""
    folder = PRIOR / 'main-CD/W0'
    if not (folder / 'summary.json').is_file():
        return dict(status='NOT_AVAILABLE', reason='PRIOR_W0_RAW_MISSING')
    saved = json.loads((folder / 'summary.json').read_text())
    cold = prior['qualification_reuse']['cold_W0_H0']
    require(saved['state'] == cold and saved['endpoint'] == 'W0' and saved['requests'] == 2000
            and saved['no_mutation'] and saved['optimizer_feedback'] is False, 'PRIOR_W0_STATE')
    rows = rows_from(folder, cold)
    require(validate_rows(rows, identities, [r['case_id'] for r in records], 'W0') == saved['summary'], 'PRIOR_W0_IDENTITY_RAW_REDUCER')
    runtime = json.loads((PRIOR / 'main-CD/runtime.json').read_text())
    require(runtime['source'] == FROZEN and runtime['cold_W0_H0'] == cold
            and runtime['config'] == digest(prior), 'PRIOR_W0_EXECUTION_IDENTITY')
    files = [member(p) for p in sorted(folder.glob('chunk-*.json'))]
    return dict(status='CPU_IDENTITY_BOUND_CONDITIONAL_COLD_SETUP_CHECK', path=str(folder), state=cold,
        summary=member(folder / 'summary.json'), chunks=files, ordered_rows=digest([r['identity'] for r in rows]),
        runtime=member(PRIOR / 'main-CD/runtime.json'), prior_config=member(PRIOR / 'config.json'),
        exact_model_input_evaluator_runtime_sources=True, same_score_is_not_identity=True,
        actual_setup_cold_W_H_identity_check_required=True, new_forwards=0,
        reuse_scope='same-runtime first2000 W0 scalar rows only; no old fitted W/H or planner evidence transfer')


def bound_pack_rows(prior, alignment_path):
    alignment = json.loads(Path(alignment_path).read_text())
    packs = []
    for original in prior['packs']:
        pack = dict(original); n = pack['batch']; rows = []; canonical = set(pack['canonical_rows'])
        requests = [r for r in alignment['rows'] if r['batch'] == n]
        require([r['case_id'] for r in requests] == pack['ids'], 'ROW_MAP_REQUEST_ORDER')
        for owner, request in enumerate(requests):
            positions = request['lookup']; require(len(positions) >= 2, 'ROW_MAP_NATIVE_ROLES')
            for site, lookup in enumerate(positions):
                j = len(rows)
                rows.append(dict(global_row=j, owner=owner, kind='kl' if site == len(positions) - 1 else 'rewrite',
                    lookup=lookup, canonical=j in canonical))
        require(len(rows) == pack['native_rows'] and all(sum(r['canonical'] for r in rows if r['owner'] == owner) == 1
                for owner in range(len(pack['ids']))), 'ROW_MAP_CARDINALITY_CANONICAL')
        require(digest([[r['owner'] for r in rows], [r['kind'] for r in rows], [r['lookup'] for r in rows]])
                == pack['row_roles_sha256'], 'ROW_MAP_BOUND_ROLES_LOOKUPS')
        pack['row_map'] = rows; pack['row_map_sha256'] = digest(rows); packs.append(pack)
    return packs


def prepare(cpu_preflight=None):
    out = LOCAL / 'preparation-r1'
    require(not (out / 'configuration.json').exists(), 'CREATE_ONCE_PREPARATION')
    envelope = json.loads((ROOT / ENVELOPE).read_text())
    require(envelope['nonce'] == NONCE and envelope['task_id'] == TASK, 'CURRENT_AUTHORITY')
    require([r['id'] for r in envelope['scope']['arms']] == list(ARMS), 'CURRENT_ARMS')
    manifest = ROOT / DESIGN / 'artifact-manifest.json'
    require(sha(manifest) == MANIFEST_SHA == envelope['canonical']['manifest_sha256'], 'METHOD_MANIFEST_SHA')
    members = [member(ROOT / p) for p in (ENVELOPE, EXCEPTION, SCHEDULE, DESIGN + '/artifact-manifest.json')]
    for row in json.loads(manifest.read_text())['files']:
        path = ROOT / row['path']
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'CANONICAL_MEMBER')
        members.append(member(path))
    validation = json.loads((ROOT / DESIGN / 'validation.json').read_text())
    require(validation['CPU_reference']['status'] == 'PASS' and validation['CPU_reference']['tests'] == 11, 'DESIGN_CPU11_EVIDENCE')
    prior = json.loads((PRIOR / 'config.json').read_text()); oldlock = json.loads((PRIOR / 'execution.lock.json').read_text())
    runtime, assets, science, alignment_path, observer_path = bind_prior_inputs(prior, oldlock)
    packs = bound_pack_rows(prior, alignment_path)
    records = load_prefix(Path(prior['stream']).parent, 2000)
    require(digest([r['case_id'] for r in records]) == ORDERED_SHA, 'ORDERED_2000')
    require(sha(ROOT / SCHEDULE) == envelope['canonical']['schedule_sha256'], 'SCHEDULE_SHA')
    with (ROOT / SCHEDULE).open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; schedule = list(reader)
    require(fields == ['stream_index0', 'batch1', 'slot0', 'case_id', 'claim_sha256', 'target_new_sha256', 'active_at_W20'] and len(schedule) == 2000, 'ALL_SCHEDULE_FIELDS_ROWS')
    flags = active_flags(records)
    for i, (row, record) in enumerate(zip(schedule, records)):
        rw = record['requested_rewrite']
        expected = [i, i // 100 + 1, i % 100, record['case_id'],
            digest([unicodedata.normalize('NFC', ' '.join(rw['subject'].split())), rw['relation_id']]), digest(rw['target_new']), int(flags[record['case_id']])]
        require([row[k] for k in fields] == list(map(str, expected)), 'ALL_SCHEDULE_VALUES:' + str(i))
        require(len(record['paraphrase_prompts']) == 2 and len(record['neighborhood_prompts']) == 10, 'EVALUATION_DENOMINATORS')
    require([p['ids'] for p in prior['packs']] == [[r['case_id'] for r in records[i:i + 100]] for i in range(0, 2000, 100)], 'ALL20_PACK_ORDER')
    cpu = Path(cpu_preflight) if cpu_preflight is not None else out / 'cpu-tests.json'
    require(cpu.is_file(), 'NEW_PRODUCTION_CPU_PREFLIGHT_REQUIRED')
    cpudata = json.loads(cpu.read_text())
    require(cpudata.get('passed') is True or cpudata.get('status') == 'PASS', 'NEW_CPU_PREFLIGHT_FAILED')
    free = shutil.disk_usage(LOCAL).free; inodes = os.statvfs(LOCAL).f_favail
    reserve = 12 * 1024**3
    require(free >= reserve and inodes >= 10000, 'RESOURCE_BLOCKED_STORAGE')
    identities = json.loads(observer_path.read_text())['rows']
    reuse = w0_reuse(prior, identities, records)
    write(out / 'W0-reuse.json', reuse)
    reuse['identity_receipt'] = member(out / 'W0-reuse.json')
    c = {k: prior[k] for k in ('model', 'stream', 'contexts', 'stats', 'profile', 'native_root', 'stats_root')}
    c.update(instruction_id=NONCE, task_id=TASK, seed=20261002, runtime=runtime, assets=assets,
        native_reference=oldlock['native_reference'], native_hparams=prior['native_hparams'],
        authority_members=members, observer_identity=member(observer_path), native_input_alignment=member(alignment_path),
        packs=packs, ordered_ids_sha256=ORDERED_SHA, cpu_preflight=member(cpu),
        prior_execution=member(PRIOR / 'execution.lock.json'), readonly_science_members=science,
        W0_reuse=reuse, w0_reuse=reuse, cold_W0_H0=prior['qualification_reuse']['cold_W0_H0'],
        settings=dict(arms=list(ARMS), alpha=ALPHA, B=100, batches=20, requests=2000,
            candidate_cap=25, update_cap=24, fit_requests_per_group=1, observer_microbatch=2,
            milestones=list(MILESTONES), save_checkpoints=False, no_B21=True, extra_fullB_fits=0, new_baseline=0),
        qualification_reuse=dict(actual_B1_reuse_verified=False,
            cold_W0_H0=prior['qualification_reuse']['cold_W0_H0'],
            unchanged_scope='asset/runtime/native token/observer and unchanged CD writer evidence only',
            new_projected_fit_actual='NOT_RUN_REQUIRED_BEFORE_MAIN', design_CPU11='SYNTHETIC_NOT_GPU',
            receipts=[member(PRIOR / 'config.json'), member(PRIOR / 'execution.lock.json'), member(ROOT / DESIGN / 'validation.json')]),
        calibration=dict(path=str(LOCAL / 'attempt-r1/calibration.json'), rule='FIRST_NONZERO_B1_ONCE_SHARED_IMMUTABLE'),
        resources=dict(task_cap=2, project_cap=3, cpu=8, gpu_per_arm=1, host_mib=59392,
            hard_host_mib=60416, collector_host_mib=24576, wall='2-00:00:00', collector_wall='04:00:00',
            reserve_bytes=reserve, startup_free_bytes_min=6 * 1024**3, free_bytes=free, free_inodes=inodes,
            concurrent_output_plan_GiB=dict(two_arm_raw_metrics_events=4, reports_source_atomic=2, margin=6),
            host_peak_plan_GiB=dict(selected_RAM_W0=1.094, H=3.828, rollback_H=3.828,
                rollback_selected_W=1.094, model_load_bound=40, execution_estimate=46,
                native_C0_FP32_one_layer=.766, requested_host_limit=58),
            GPU_peak_plan_GiB=dict(actual_device_capacity=96, model=30,
                per_layer_rawA_C0_A_L_delta_Bmap_scratch_upper=16, planned_peak_upper=72,
                cached_weights_activation_and_qualification='bounded; one-layer geometry scratch streamed'),
            GPU_device_expected='NVIDIA RTX PRO 6000', geometry_compute_device='cuda',
            memory_plan_status='ESTIMATE_NOT_ACTUAL_GPU_PASS', original_runtime_memory_reference_only=True,
            state_policy='RAM own W/H per arm; frozen geometry/cache per batch; no resume bundle',
            ETA='NOT_MEASURED_NEW_PROJECTED_FIT; finite48h ceiling not ETA'))
    write(out / 'configuration.json', c)
    write(out / 'input-runtime-reuse.json', dict(nonce=NONCE, prior=member(PRIOR / 'config.json'),
        CSV_rows=2000, all_fields_checked=True, all20_native_packs=packs,
        observer_rows=26000, native_input_alignment=member(alignment_path), observer_identity=member(observer_path),
        current_source_reuse=science, prior_SHA_plus_fresh_asset_stat=assets, runtime=runtime,
        design_CPU11=validation['CPU_reference'], projected_actual_GPU='NOT_RUN', source_frozen=False,
        host=platform.node(), worktree=str(ROOT), new_model_or_tokenizer_load=False,
        description_ko='기존 입력/자산/관측 정합만 재사용. 새 projected fit의 실제모델 검증 PASS 아님.'))
    return dict(configuration=str(out / 'configuration.json'), packs=20, requests=2000, projected_GPU='NOT_RUN')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--cpu-preflight', type=Path)
    args = parser.parse_args(); print(json.dumps(prepare(args.cpu_preflight)))
