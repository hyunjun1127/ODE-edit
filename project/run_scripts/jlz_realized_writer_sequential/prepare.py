"""CPU-only all-2000 token/native/source/runtime binding; no model execution."""
import csv, importlib, json, platform, shutil, unicodedata
from pathlib import Path
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding, asset_binding
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import active_flags
from project.run_scripts.jlz_two_arm.baseline_pilot import _import_native, _source_closure
from .common import *

def prepare():
    out = LOCAL / 'preparation-r1'
    require(not (out / 'configuration.json').exists(), 'CREATE_ONCE_PREPARATION')
    require(sha(ROOT / ENVELOPE) == 'df956c6b22ed075615ff83d13e76acd06d0284a9d07abc10d8789e85e9f43051', 'ENVELOPE_SHA')
    require(sha(ROOT / CONTRACT) == 'c6aa2e52a1e71fccbf20b14486ba7f26b64fd84b8cdbda256125d69b9fe80d62', 'CONTRACT_SHA')
    envelope = json.loads((ROOT / ENVELOPE).read_text()); contract = json.loads((ROOT / CONTRACT).read_text())
    require(envelope['nonce'] == contract['instruction_id'] == NONCE, 'CURRENT_NONCE')
    require(contract['scope']['arms'] == list(ARMS) and contract['scope']['batches_each'] == 20, 'CURRENT_SCOPE')
    prior = json.loads((PRIOR / 'config.json').read_text())
    oldlock = json.loads((PRIOR / 'execution.lock.json').read_text())
    require(oldlock['source_commit'] == FROZEN and sha(PRIOR / 'config.json') == oldlock['config_sha256'], 'ORIGINAL_B1_EXECUTION')
    prior_members = {r['path']: r for r in prior['assets']}
    assets = [asset_binding(r['path'], prior_members, member(PRIOR / 'config.json')) for r in prior['assets']]
    runtime = runtime_binding()
    oldruntime = {r['module']: r['sha256'] for r in prior['runtime']['source_members']}
    require(runtime['torch'] == prior['runtime']['torch'] and runtime['transformers'] == prior['runtime']['transformers'], 'RUNTIME_VERSION_REUSE')
    require(all(oldruntime[r['module']] == r['sha256'] for r in runtime['source_members']), 'RUNTIME_SOURCE_REUSE')
    for row in oldlock['dependency_sources'] + oldlock['native_reference']:
        verify(row)
    actual_imports = json.loads((PRIOR / 'B1/actual-imports.json').read_text())['files']
    science = []
    for row in actual_imports:
        if not row['module'].startswith('project.run_scripts.'):
            continue
        oldpath = verify(row); relative = oldpath.relative_to(PRIOR / 'source')
        current = ROOT / relative
        require(sha(current) == row['sha256'], 'FROZEN_IMPORT_CLOSURE:' + str(relative))
        science.append(dict(relative=str(relative), original=row, current=member(current)))
    manifest = ROOT / DESIGN / 'artifact-manifest.json'
    require(sha(manifest) == '6e2794359186f437180afa28909f4f775b0f7f713c22a9b8fd4917fb67645d4c', 'METHOD_MANIFEST')
    authority_members = [member(ROOT / p) for p in (ENVELOPE, CONTRACT, EXCEPTION, SCHEDULE,
        DESIGN + '/artifact-manifest.json', 'plans/global/2026-10-04-jlz-v12-marginal-allocation/contract.json',
        'plans/global/2026-10-04-jlz-portable-execution-v1/runtime-contract.json')]
    for row in json.loads(manifest.read_text())['artifacts']:
        path = ROOT / DESIGN / row['file']
        contents = path.read_bytes()
        require(len(contents) == row['bytes'] and sha(path) == row['sha256'], 'METHOD_MEMBER')
        authority_members.append(member(path))
    old_read = json.loads((ROOT / 'audits/servers/server4/jlz-v13-realized-writer-b1/full-read.json').read_text())
    reused_read = {Path(r['path']).relative_to(Path(old_read['worktree'])): r['sha256'] for r in old_read['members']}
    for row in authority_members:
        relative = Path(row['path']).relative_to(ROOT)
        if relative in reused_read:
            require(reused_read[relative] == row['sha256'], 'PRIOR_FULL_READ_EXACT_SHA')
    receipts = [member(PRIOR / 'execution.lock.json'), member(PRIOR / 'config.json'), member(PRIOR / 'B1/actual-imports.json'),
        member(PRIOR / 'B1/terminal.json'), member(PRIOR / 'B1/W0/summary.json')]
    receipts += [member(p) for p in sorted((PRIOR / 'B1/qualification').glob('*.json'))]
    require(json.loads((PRIOR / 'B1/qualification/ready.json').read_text())['status'] == 'QUALIFIED_BOUNDED_ACTUAL_MODEL', 'ACTUAL_PLANNER_QUALIFICATION')
    require(json.loads((PRIOR / 'B1/terminal.json').read_text())['status'] == 'B1_FIVE_WRITERS_COMPLETE', 'ACTUAL_WRITER_QUALIFICATION')
    for arm in ARMS:
        for basename in ('complete.json', 'restore.json', 'writer.json', 'local-additivity.json'):
            receipts.append(member(PRIOR / 'B1' / arm / basename))
        wr = json.loads((PRIOR / 'B1' / arm / 'writer.json').read_text())
        require(wr['history_appends'] == len(prior['profile']['eligible_layers']) and all(
            r['solver']['numerical_projection_verified'] and r['ideal_effective_parity']['pass_'] for r in wr['layers'].values()), 'ACTUAL_WRITER_PARITY')
        require(json.loads((PRIOR / 'B1' / arm / 'restore.json').read_text())['verified'], 'ACTUAL_B1_RESTORE')
    records = load_prefix(Path(prior['stream']).parent, 2000)
    require(sha(ROOT / SCHEDULE) == 'dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2', 'SCHEDULE_SHA')
    with (ROOT / SCHEDULE).open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; schedule = list(reader)
    require(fields == ['stream_index0', 'batch1', 'slot0', 'case_id', 'claim_sha256', 'target_new_sha256', 'active_at_W20'], 'ALL_CSV_FIELDS')
    flags = active_flags(records)
    require(len(schedule) == len(records) == 2000, 'ALL_CSV_ROWS')
    for i, (row, record) in enumerate(zip(schedule, records)):
        rw = record['requested_rewrite']
        expected = [i, i // 100 + 1, i % 100, record['case_id'],
            digest([unicodedata.normalize('NFC', ' '.join(rw['subject'].split())), rw['relation_id']]),
            digest(rw['target_new']), int(flags[record['case_id']])]
        require([row[k] for k in fields] == list(map(str, expected)), 'CSV_FULL_FIELD_CHECK:' + str(i))
        require(len(record['paraphrase_prompts']) == 2 and len(record['neighborhood_prompts']) == 10, 'EVAL_PANEL_PROFILE')
    ordered_ids = digest([r['case_id'] for r in records])
    require(ordered_ids == '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4', 'ORDERED_2000_SHA')
    tok = AutoTokenizer.from_pretrained(prior['model'], local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'right'
    bench = CounterFactAdapter(tok, json.loads(Path(prior['contexts']).read_text()))
    packs = []; alignment = []
    with _import_native(prior['native_root']):
        native = importlib.import_module('memit.memit_seq_main'); closure = _source_closure(prior['native_root'])
        require(sha(native.__file__) == 'f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a', 'BLUE_NATIVE_SOURCE')
        for number, current, seen in batches(records, 100):
            pack = bench.prepare(current)
            if number == 1:
                require(pack['identity'] == prior['packing']['identity'] and pack['record_ids'] == prior['packing']['ids'], 'ORIGINAL_B1_INPUT')
            for request, spec in zip(pack['requests'], pack['specs']):
                prompts = [context.format(request['prompt']) + tok.decode(spec['target'][:-1]) for group in bench.contexts for context in group] + ['{} is a']
                positions = [native.find_fact_lookup_idx(p, request['subject'], tok, 'subject_last', verbose=False) for p in prompts]
                require(positions == spec['lookup'], 'NATIVE_LOOKUP_ALL_2000')
                ref = tok([p.format(request['subject']) for p in prompts], return_tensors='pt', padding=True)
                for j in range(len(prompts)):
                    n = int(ref['attention_mask'][j].sum()); idx = spec['offset'] + j
                    require(ref['input_ids'][j, :n].tolist() == pack['tokens']['input_ids'][idx, :n].tolist(), 'NATIVE_TOKENS_ALL_2000')
                alignment.append(dict(batch=number, case_id=request['case_id'], lookup=positions,
                    token_identity=digest([ref['input_ids'].tolist(), ref['attention_mask'].tolist()])))
            packs.append(dict(batch=number, identity=pack['identity'], ids=pack['record_ids'], B=pack['n_requests'],
                native_rows=len(pack['row_request']), canonical_rows=pack['canonical_rows'],
                row_roles_sha256=digest([pack['row_request'], pack['row_kind'], pack['lookup']]), seen_requests=len(seen)))
    write(out / 'native-input-alignment.json', dict(rows=alignment, packs=packs, requests=2000, all_rows_fields=True, GPU=False, fit_calls=0))
    observers = []
    for record in records:
        rw = record['requested_rewrite']
        for kind, prompts in bench.panels(record).items():
            for i, prompt in enumerate(prompts):
                row = dict(identity=digest([record['case_id'], kind, i, prompt, rw['target_new']['str'], rw['target_true']['str']]),
                    case_id=record['case_id'], kind=kind, prompt_index=i)
                for label in ('new', 'true'):
                    ids, target = bench.evaluation_ids(prompt, rw['target_' + label]['str'])
                    row[label + '_token_identity'] = digest([ids, target])
                observers.append(row)
    require(len(observers) == len({r['identity'] for r in observers}) == 26000, 'ALL_2000_OBSERVER_IDENTITIES')
    write(out / 'observer-identity.json', dict(rows=observers, ordered_ids_sha256=ordered_ids, raw_prompt_saved=False))
    cpu = out / 'cpu-tests.json'
    require(json.loads(cpu.read_text())['passed'], 'CPU_PREFLIGHT_REQUIRED')
    free = shutil.disk_usage(LOCAL).free
    require(free >= 12 * 1024**3, 'RESOURCE_BLOCKED_STORAGE')
    original_peak = json.loads((PRIOR / 'B1/terminal.json').read_text())
    c = {k: prior[k] for k in ('model', 'stream', 'contexts', 'stats', 'profile', 'native_root', 'stats_root')}
    c.update(instruction_id=NONCE, task_id=TASK, authority=AUTHORITY, seed=20261002, runtime=runtime, assets=assets,
        native_reference=closure, native_hparams=str(ROOT / 'project/run_scripts/memit_history_lifelong/hparams.json'),
        native_input_alignment=member(out / 'native-input-alignment.json'), observer_identity=member(out / 'observer-identity.json'),
        packs=packs, ordered_ids_sha256=ordered_ids, authority_members=authority_members, cpu_preflight=member(cpu),
        settings=dict(arms=list(ARMS), B=100, batches=20, requests=2000, fits_each=20, commits_each=20,
            candidate_cap=25, update_cap=24, fit_requests_per_group=1, observer_microbatch=2, milestones=list(MILESTONES),
            save_checkpoints=False, no_B21=True, extra_fullB_fits=0, new_baseline=0, sequential_pilot=False),
        qualification_reuse=dict(actual_B1_reuse_verified=True, source=FROZEN, actual_scope='same S4 B1 planner/native/writer; not universal20batch parity',
            cold_W0_H0=json.loads((PRIOR / 'B1/W0/summary.json').read_text())['state'], science_members=science,
            receipts=receipts, CPU_new_scope='two-batch transaction/cache/history/observer/horizon/collector fixtures', extra_GPU_qualification=0),
        resources=dict(task_cap=2, project_cap=3, gpu_per_arm=1, cpu=8, host_mib=59392, collector_host_mib=24576,
            wall='1-00:00:00', collector_wall='04:00:00', reserve_bytes=12 * 1024**3, startup_free_bytes_min=6 * 1024**3,
            free_bytes=free, concurrent_output_plan_GiB=dict(two_arm_raw_events_metrics=3, CPU_summaries_inventory=1,
                source_logs_atomic=1, margin=7), requested_aggregate_host_mib=118784,
            original_B1_peak_RSS_KiB=original_peak['peak_RSS_KiB'], original_B1_peak_VRAM_bytes=original_peak['peak_VRAM_bytes'],
            host_peak_plan_GiB_each=dict(load=36, execution=40, limit=58), GPU_peak_plan_GiB_each=dict(expected=54, margin=12),
            state_policy='one batch cache/metric resident; persistent H, new entry/cache and factor perbatch; no saved weights/teacher/plan',
            ETA='NOT_MEASURED_2K; originalB1 cost descriptive only; 24h requested wall is ceiling, not ETA'))
    write(out / 'configuration.json', c)
    write(out / 'full-read.json', dict(nonce=NONCE, authority=AUTHORITY, members=authority_members,
        current_full_read='new envelope/derived sequential contract plus method/experiment/contracts/telemetry/validation/portability',
        exact_prior_FULL_READ_reuse=member(ROOT / 'audits/servers/server4/jlz-v13-realized-writer-b1/full-read.json'),
        prior_reference_reuse_scope='unchanged reference/writer.py,test_ref.py,audit.json and their exact SHA; no duplicate GPU proof',
        CSV_rows=2000, CSV_all_fields_checked=True, all20_native_packs=packs, observer_rows=26000,
        science_members=science, runtime_reuse=True, actual_new_GPU='NOT_RUN', submission='NOT_SUBMITTED',
        host=platform.node(), session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd', worktree=str(ROOT)))
    print(json.dumps(dict(configuration=str(out / 'configuration.json'), packs=len(packs), requests=len(records), observer_rows=len(observers), actual_GPU='NOT_RUN')))

if __name__ == '__main__':
    prepare()
