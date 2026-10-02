"""CPU-only immutable v5 authority/input/runtime/W0 binding; no scheduler I/O."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import unicodedata

from .common import ROOT, LOCAL, INSTRUCTION, TASK, digest, member, require, sha, write


AUTHORITY = 'ba2deac2474087f8bb8b7b2b0d8848dd4097b16c'
ENVELOPE = 'messages/head/2026-10-02-jlz-v5-cancel-v4-2k-sh4.json'
EXCEPTION = 'control/experiment-exceptions/jlz-writer-coupled-v5-2k-20261002.json'
DESIGN = 'plans/global/2026-10-02-jlz-writer-coupled-v5'
CASE = 'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv'
EVALUATION = 'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/evaluation-schedule.json'
OLD_LOCAL = Path('/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1')
STREAM_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
CONTEXT_SHA = '33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e'


def verify_member(row):
    path = Path(row['path'])
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'MEMBER_CHANGED:' + str(path))
    return path


def bind_authority(root=ROOT):
    """Read all 14 bound members in full and bind the published authority."""
    for relative in (ENVELOPE, EXCEPTION):
        published = subprocess.check_output(['git', 'show', AUTHORITY + ':' + relative], cwd=root)
        require((root / relative).read_bytes() == published, 'PUBLISHED_AUTHORITY_CHANGED:' + relative)
    envelope = json.loads((root / ENVELOPE).read_text())
    exception = json.loads((root / EXCEPTION).read_text())
    require(envelope['instruction_id'] == INSTRUCTION and envelope['task_id'] == TASK, 'AUTHORITY_TASK')
    require(exception['authorized'] and exception['instruction_id'] == INSTRUCTION
            and exception['task_id'] == TASK and exception['project_gpu_cap'] == 2
            and not exception['old_v4_restart_authorized'], 'EXCEPTION_SCOPE')
    bindings = envelope['binding']['members']
    require(len(bindings) == 14 and len({row['path'] for row in bindings}) == 14, 'AUTHORITY_14_MEMBERS')
    rows = []
    for row in bindings:
        path = root / row['path']
        content = path.read_bytes()  # FULL read, including the complete CSV.
        require(len(content) == row['bytes'] and sha(path) == row['sha256'], 'AUTHORITY_MEMBER:' + row['path'])
        content.decode('utf-8')
        rows.append(dict(row, verification='FULL_BYTES_SHA256_SIZE', read_bytes=len(content)))
    return envelope, dict(authority_commit=AUTHORITY, envelope=member(root / ENVELOPE),
                          exception=member(root / EXCEPTION), members=rows,
                          full_read_scope='all bound bytes; CSV all rows/fields and evaluation all endpoints parsed',
                          v4_cancellation=member(root / envelope['cancellation']['receipt']))


def validate_schedule(records, case_path=ROOT / CASE, evaluation_path=ROOT / EVALUATION):
    from .observe import active_flags
    require(len(records) >= 2004, 'MAIN_AND_DEVELOPMENT_INPUT_COUNT')
    cases = records[:2000]
    require(len({row['case_id'] for row in cases}) == 2000, 'CASE_IDS_UNIQUE')
    flags = active_flags(cases)
    fields = ['stream_index0', 'batch1', 'slot0', 'case_id', 'claim_sha256', 'target_new_sha256', 'active_at_W20']
    with Path(case_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, 'CASE_CSV_FIELDS')
        rows = list(reader)
    require(len(rows) == 2000, 'CASE_CSV_ALL_2000_ROWS')
    for i, (row, record) in enumerate(zip(rows, cases)):
        rewrite = record['requested_rewrite']
        claim = [unicodedata.normalize('NFC', ' '.join(rewrite['subject'].split())), rewrite['relation_id']]
        expected = [i, i // 100 + 1, i % 100, record['case_id'], digest(claim),
                    digest(rewrite['target_new']), int(flags[record['case_id']])]
        require([row[field] for field in fields] == [str(value) for value in expected], 'CASE_CSV_ROW:' + str(i))
        require(len(record['paraphrase_prompts']) == 2 and len(record['neighborhood_prompts']) == 10, 'NATIVE_EVAL_PANELS')
    schedule = json.loads(Path(evaluation_path).read_text())
    require(schedule['case_schedule_sha256'] == sha(case_path), 'SCHEDULE_CASE_SHA')
    require(schedule['final_denominators'] == dict(R=2000, P=4000, N=20000), 'FINAL_DENOMINATORS')
    require(len(schedule['endpoints']) == 21, 'ALL_EVALUATION_ENDPOINTS')
    for batch, row in enumerate(schedule['endpoints']):
        if batch == 0:
            require(row['batch'] == 0 and row['scope'] == 'W0_full2k'
                    and row['ordinal_slice'] == [0, 2000] and row['requests'] == 2000
                    and [row[k] for k in ('R', 'P', 'N')] == [2000, 4000, 20000], 'W0_SCHEDULE')
            continue
        milestone = batch in (5, 10, 20)
        low, high = (batch - 1) * 100, batch * 100
        start = 0 if milestone else low
        expected = dict(batch=batch, scope='all_seen' if milestone else 'current',
                        ordinal_slice=[start, high], current_slice=[low, high], requests=high - start,
                        R=high - start, P=2 * (high - start), N=10 * (high - start),
                        current_rows_reused_from_same_endpoint=milestone)
        require(row == expected, 'EVALUATION_ENDPOINT:' + str(batch))
    require(schedule['preference'] == dict(R_P='new_nll < true_nll', N='true_nll < new_nll', ties='failure')
            and schedule['evaluation_feedback_to_solver'] is False, 'EVALUATION_SEMANTICS')
    derived = dict(schedule)
    derived.update(schema='JLZ_WRITER_COUPLED_V5_EVALUATION_SCHEDULE_V1',
                   source_schedule=member(evaluation_path), W0_mode='W0_REUSE', new_W0_forward=0)
    derived['endpoints'] = [dict(row) for row in schedule['endpoints']]
    derived['endpoints'][0].update(scope='W0_REUSE', historical_scope='W0_full2k', new_forward=0)
    return derived, dict(case_schedule=member(case_path), rows=2000, fields=fields,
                         all_fields_checked=True, ordered_case_ids_sha256=digest([r['case_id'] for r in cases]),
                         active_occurrences=sum(flags.values()), superseded_occurrences=sum(not x for x in flags.values()),
                         all_evaluation_endpoints_checked=21)


def asset_binding(path, previous_members, previous_receipt):
    path = Path(path)
    resolved = str(path.resolve())
    current = path.stat()
    prior = previous_members.get(resolved)
    if prior and (prior['bytes'], prior.get('inode'), prior.get('mtime_ns')) == (
            current.st_size, current.st_ino, current.st_mtime_ns):
        return dict(path=resolved, requested_path=str(path), bytes=current.st_size, inode=current.st_ino,
                    mtime_ns=current.st_mtime_ns, sha256=prior['sha256'],
                    verification='PRIOR_FULL_SHA_CURRENT_SIZE_INODE_MTIME', prior_receipt=previous_receipt)
    return dict(member(path), requested_path=str(path), verification='NEW_FULL_SHA')


def runtime_binding():
    import torch
    import transformers
    names = ['torch', 'torch.nn.functional', 'torch.autograd.function', 'numpy',
             'transformers', 'transformers.models.llama.modeling_llama',
             'transformers.masking_utils', 'transformers.cache_utils',
             'transformers.modeling_utils', 'transformers.tokenization_utils_base',
             'transformers.tokenization_utils_fast']
    rows = [dict(module=name, **member(importlib.import_module(name).__file__)) for name in names]
    require(not torch.cuda.is_initialized(), 'CPU_PREPARATION_MUST_NOT_INITIALIZE_CUDA')
    return dict(python=str(Path(sys.executable).absolute()), executable=member(sys.executable),
                python_version=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                torch_cuda=torch.version.cuda, source_members=rows, source_root_sha256=digest(rows),
                attention='eager', model_dtype='float32', geometry_dtype='float64', matmul_TF32=False,
                cudnn_TF32=False, autocast=False, cuda_initialized=False,
                GPU_runtime_qualification='PENDING_FIXED_CANDIDATE_RECEIPT')


def bind_w0(source_bridge, assets, records, bench, runtime):
    """Recheck the historical bridge's complete raw/token identity on CPU."""
    from .observe import active_flags, reduce_rows
    bridge = json.loads(Path(source_bridge).read_text())
    require(bridge['status'] == 'REUSED_HISTORICAL_VERIFIED' and bridge['new_W0_forward'] == 0, 'W0_BRIDGE_STATUS')
    for key in ('observations', 'prior_config', 'prior_lock', 'current_initial_state'):
        verify_member(bridge[key])
    for row in bridge['evaluator_sources']:
        verify_member(row)
    prior = json.loads(Path(bridge['prior_config']['path']).read_text())
    prior_assets = {row['path']: row for row in prior['inputs']}
    compared = []
    for row in assets:
        old = prior_assets.get(row['path'])
        require(old and (old['bytes'], old['sha256']) == (row['bytes'], row['sha256']), 'W0_INPUT_ASSET_CHANGED:' + row['path'])
        compared.append(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256']))
    observation = json.loads(Path(bridge['observations']['path']).read_text())
    require(observation['endpoint'] == 0 and observation['requests'] == 2000
            and observation['no_mutation'] and not observation['optimizer_feedback'], 'W0_OBSERVER_SCOPE')
    require(observation['state'] == bridge['state'], 'W0_BRIDGE_STATE')
    index = {row['identity']: row for row in observation['rows']}
    require(len(index) == len(observation['rows']) == 26000, 'W0_ROWS')
    flags = active_flags(records[:2000])
    token_identities, seen = [], set()
    for record in records[:2000]:
        rewrite = record['requested_rewrite']
        for kind, prompts in bench.panels(record).items():
            for number, prompt in enumerate(prompts):
                identity = digest([record['case_id'], kind, number, prompt,
                                   rewrite['target_new']['str'], rewrite['target_true']['str']])
                require(identity in index, 'W0_MISSING_ROW')
                row = index[identity]
                require((row['case_id'], row['kind'], row['prompt_index'], row['endpoint']) ==
                        (record['case_id'], kind, number, 0), 'W0_ROW_IDENTITY')
                require(row['active_at_endpoint'] == flags[record['case_id']], 'W0_ACTIVE_FLAGS')
                for label in ('true', 'new'):
                    prompt_ids, target_ids = bench.evaluation_ids(prompt, rewrite['target_' + label]['str'])
                    require(row[label + '_token_count'] == len(target_ids), 'W0_TOKEN_DENOMINATOR')
                    token_identities.append((identity, label, digest([prompt_ids, target_ids])))
                seen.add(identity)
    require(seen == set(index) and digest(token_identities) == bridge['validation']['token_identity'], 'W0_ALL_TOKEN_IDENTITY')
    summary = reduce_rows(observation['rows'])
    require({k: v['denominator'] for k, v in summary.items()} == dict(R=2000, P=4000, N=20000), 'W0_DENOMINATORS')
    for kind in summary:
        for key in ('numerator', 'denominator', 'true_nll_mean', 'new_nll_mean', 'desired_token_correct', 'desired_token_count'):
            require(summary[kind][key] == observation['summary'][kind][key], 'W0_SUMMARY:' + kind + '/' + key)
    # Exact historical runtime values, not a claim about numerical equivalence
    # of the v5 selected-position evaluator's new batching implementation.
    old_lock = json.loads(Path(bridge['prior_lock']['path']).read_text())
    producer = Path(bridge['prior_lock']['path']).parent / 'source/project/run_scripts/jlz_two_arm/run.py'
    producer_member = next(row for row in old_lock['source_members'] if Path(row['path']) == producer)
    verify_member(producer_member)
    producer_text = producer.read_text()
    require("str(torch.__version__)=='2.9.1+cu128'" in producer_text
            and "transformers.__version__=='4.57.1'" in producer_text, 'HISTORICAL_RUNTIME_PINS')
    require(runtime['torch'] == '2.9.1+cu128' and runtime['transformers'] == '4.57.1', 'W0_PINNED_RUNTIME_CHANGED')
    return dict(status='REUSED_HISTORICAL_VERIFIED', instruction_id=INSTRUCTION, source_bridge=member(source_bridge),
                observations=bridge['observations'], state=bridge['state'], assets=compared, new_W0_forward=0,
                validation=dict(all_tokens_checked=True, row_count=26000, token_identity=digest(token_identities),
                                case_order=digest([r['case_id'] for r in records[:2000]]), summary=summary),
                historical_runtime_lock=bridge['prior_lock'], historical_producer=producer_member,
                historical_runtime_version_pins=dict(torch='2.9.1+cu128', transformers='4.57.1'),
                historical_actual_package_source_SHA='NOT_RECORDED_IN_PRIOR_BRIDGE; historical pins match current versions',
                current_runtime_sources=runtime['source_members'],
                state_equality_at_actual_fresh_model_load='REQUIRED_BEFORE_SCIENCE',
                numerical_bitwise_equivalence='NOT_ESTABLISHED', layout_difference=bridge['layout_difference'],
                no_checkpoint=True, old_task_restarted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=LOCAL / 'preparation-v1')
    parser.add_argument('--previous-config', type=Path, default=OLD_LOCAL / 'preparation-v1/configuration.json')
    parser.add_argument('--w0-reuse', type=Path, default=OLD_LOCAL / 'user-remove-w0-r1/W0-reuse.json')
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(LOCAL) and not out.exists(), 'CREATE_ONCE_TASK_LOCAL_PREPARATION')
    envelope, authority = bind_authority()
    previous = json.loads(args.previous_config.read_text())
    previous_receipt = member(args.previous_config)
    prior_members = {row['path']: row for row in previous.get('assets', previous.get('inputs', []))}
    model, stream, contexts = map(Path, (previous['model'], previous['stream'], previous['contexts']))
    require(sha(stream) == STREAM_SHA and sha(contexts) == CONTEXT_SHA, 'EXACT_STREAM_CONTEXT_SHA')
    data = json.loads(stream.read_text())
    schedule, schedule_binding = validate_schedule(data)
    layers = [4, 5, 6, 7, 8]
    stats = previous['stats']
    require(isinstance(stats, dict) and set(stats) == {str(layer) for layer in layers}, 'ALL_ELIGIBLE_STATS')
    model_index = json.loads((model / 'model.safetensors.index.json').read_text())
    paths = [stream, contexts] + [Path(stats[str(layer)]) for layer in layers]
    paths += [model / name for name in sorted(set(model_index['weight_map'].values()) | {p.name for p in model.glob('*.json')})]
    assets = [asset_binding(path, prior_members, previous_receipt) for path in paths]
    # The prior schema was computed from the exact same bound npz bytes.
    schemas = previous['stats_schema']
    require({row['layer'] for row in schemas} == set(layers), 'PRIOR_SCHEMA_LAYER_SET')
    for row in schemas:
        require(row['count'] > 0 and row['shape'] == [14336, 14336], 'PRIOR_STATS_SCHEMA')
        current = next(asset for asset in assets if asset['requested_path'] == stats[str(row['layer'])])
        prior = prior_members.get(current['path'])
        require(prior and (prior['bytes'], prior['sha256']) == (current['bytes'], current['sha256']),
                'C0_CHANGED_REQUIRES_NEW_SCHEMA_QUALIFICATION')
    runtime = runtime_binding()
    from transformers import AutoTokenizer
    from .inputs import CounterFactAdapter
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    bench = CounterFactAdapter(tokenizer, json.loads(contexts.read_text()))
    packing = []
    for phase, start, B, count in [('pilot', 2000, 2, 2), ('timing', 0, 100, 1), ('main', 0, 100, 20)]:
        for number in range(count):
            records = data[start + number * B:start + (number + 1) * B]
            pack = bench.prepare(records)
            packing.append(dict(phase=phase, batch=number + 1, actual_B=len(records), ids=pack['record_ids'],
                                identity=pack['identity'], rows=len(pack['row_request']),
                                valid_tokens=int(pack['tokens']['attention_mask'].sum()),
                                width=int(pack['tokens']['input_ids'].shape[1]), key_prefix_exact=pack['entry_key_prefix_exact']))
    w0 = bind_w0(args.w0_reuse, assets, data, bench, runtime)
    free = shutil.disk_usage(LOCAL).free
    require(free >= 30 * 1024**3, 'OUTPUT_RESERVE_30GIB')
    snapshot = dict(hostname=platform.node(), cpu_count=os.cpu_count(), disk_free_bytes=free,
                    memory=Path('/proc/meminfo').read_text(), loadavg=Path('/proc/loadavg').read_text().strip(),
                    scheduler_queried=False, gpu_queried=False, scope='CPU_SAFE_LOCAL_RESOURCE_ONLY')
    resources = dict(cap=2, cpu=8, gpu=1, host_mib=60416, mem_mib=60416, wall='7-00:00:00',
                     qualification_wall='24:00:00', collector_wall='04:00:00', collector_host_mib=24576,
                     node='server4', partition='gpu', qos='lab_gpu_s4', reserve_bytes=30 * 1024**3,
                     wall_not_ETA=True, peak_measured=False, local_snapshot=snapshot)
    out.mkdir(parents=True)
    write(out / 'W0-reuse.json', w0)
    config = dict(instruction_id=INSTRUCTION, task_id=TASK, authority=AUTHORITY, authority_binding=authority,
                  contract=json.loads((ROOT / DESIGN / 'contract.json').read_text()),
                  model=str(model), stream=str(stream), dataset=str(stream), contexts=str(contexts), stats=stats,
                  assets=assets, previous_asset_receipt=previous_receipt, stats_schema=schemas,
                  stats_schema_verification='prior schema plus exact unchanged input bytes', packing=packing,
                  profile=dict(adapter='llama_causal_down_proj', benchmark='counterfact_native', eligible_layers=layers,
                               nll_layer=31, kl_factor=.0625, preservation_K=.0625, preservation_E=1.,
                               norm_factor=.5, clamp_factor=.75, lambda_C=15000),
                  experiment=dict(envelope['execution']), evaluation_schedule=schedule,
                  schedule_binding=schedule_binding, runtime=runtime,
                  settings=dict(seed=20261002, fit_microbatch=2, observer_microbatch=2, memory_capacity=128,
                                reference_cap=16, candidate_cap=25, backward_cap=24, save_checkpoints=False,
                                epsilon_num='PENDING_FIXED_CANDIDATE_QUALIFICATION', route='PENDING_QUALIFICATION'),
                  resources=resources, w0_reuse=dict(mode='REUSE_ONLY_USER_DIRECTED', receipt=member(out / 'W0-reuse.json'),
                                                   new_W0_forward=0),
                  native_reference=previous['native_reference'], checkpoint_saved=False,
                  exact_resume='NOT_AVAILABLE', new_baseline_runs=0, broadcast='NO_BROADCAST_NOT_REQUIRED')
    config['settings']['numerical_qualification'] = dict(
        dense_full_MB1_MB4_loss_absolute_max=5e-5, dense_full_MB1_MB4_gradient_relative_max=2e-3,
        direct_qualified_loss_absolute_max=5e-5, direct_qualified_gradient_relative_max=2e-3,
        direct_dense_fallback_discrepancy_loss_absolute_max=5e-4,
        direct_dense_fallback_discrepancy_gradient_relative_max=2e-2,
        epsilon_rule='max(1e-7,2*qualified_dense_samecandidate_MB1_MB4_SUM_difference/actual_B)',
        actual_commit_NLL_absolute_max=5e-5, actual_commit_key_absolute_max=1e-4,
        geometry_scaled_residual_max=1e-8, geometry_same_M_reference_checks_max=1,
        coefficients_or_quality_feedback=False, thresholds_fixed_before_outcome=True)
    write(out / 'configuration.json', config)
    write(out / 'full-read.json', dict(instruction_id=INSTRUCTION, **authority, schedule=schedule_binding,
                                      status='CPU_BOUND_GPU_NOT_RUN', job_ids=[], scheduler_queried=False))
    print(json.dumps(dict(configuration=str(out / 'configuration.json'), assets=len(assets), packing=len(packing),
                          status='CPU_INPUT_BINDING_PASS_GPU_NOT_RUN')))


if __name__ == '__main__':
    main()
