"""CPU-only independent review of six native chains and saved generation rows.

This module never imports a model, tokenizer, fitter or generation implementation.
It checks the published SH1 receipt protocol, independently reduces saved scalar
metrics, and preserves valid prefixes when later data are missing or inconsistent.
Stored token relations are checked; tokenization and scientific scoring are not
reexecuted. Optional scheduler accounting is one exact six-parent query, never
a monitor; its absence cannot erase valid scientific rows.
"""
import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path

from .generation_common import (
    ARMS, ARM_LAYERS, MILESTONES, NONCE, TASK, batches, digest,
    expected_counts, member, require, write, writer_identity,
)
from .generation_plan import counts as planned_counts
from .collect import _atomic_text, _csv, _metric_rows, _paired_rows
from project.run_scripts.gptj_cake_blue_prune_rect.collect import endpoint as rpn_endpoint
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, compare_summary, reduce_rows,
)

COUNT_KEYS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
HISTORY_ARMS = ('BASE_ALPHAEDIT', 'CAKE', 'ALPHAEDIT_BLUE')
SCHEMA = 'counterfact-cake-generation-metrics-v1'
PROFILE = 'cf-cake-prompt-inclusive-total100-eos-corrected-v1'
EVAL_SEED = 20261007
REPAIR_TASK = 'gptj-baselines-generation-cache-repair'
REPAIR_NONCE = 'USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1'
# Closed vocabulary from SH1 metrics.py at 83535c6a; not a new scoring rule.
REASONS = ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
           'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation',
           'asset_not_available', 'tokenizer_not_available')
TABLE_KEYS = ('metrics', 'paired', 'generation', 'counts', 'compute')
ROUTES = ('UNPADDED_FULL_PREFIX_NO_CACHE', 'UNPADDED_SINGLETON_KV_CACHE', 'EQUAL_TOKEN_LENGTH_KV_BATCH')
SHARED_ROUTES = ('UNPADDED_FULL_PREFIX_NO_CACHE', 'UNPADDED_KV_SINGLETON', 'EQUAL_LENGTH_KV_BATCH')
SHARED_ROUTE = dict(zip(ROUTES, SHARED_ROUTES))
SHARED_ROUTE.update({route: route for route in SHARED_ROUTES})
EXECUTION_ADAPTER = 'TASK_PRIVATE_SHARED_GENERATE_ROWS_PROOF_CONVERSION_NOT_SHARED_RUN_QUALIFICATION'
QUALIFICATION_GATES = ('tokens', 'EOS', 'row_mapping', 'seed_stream', 'logits', 'topk_ids',
                      'topk_probabilities', 'metrics', 'positions', 'coverage')
QUALIFICATION_COVERAGE = ('forced_prefix', 'full_vocab_logits', 'topk_probabilities', 'tokens',
                          'EOS', 'row_mapping', 'seed_stream', 'metrics')
QUALIFICATION_WORK = ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens',
                      'logical_row_token_decisions')
QUALIFICATION_COST = ('elapsed_seconds', 'synchronized_GPU_seconds', 'peak_gpu_allocated_bytes',
                      'peak_gpu_reserved_bytes', 'peak_host_RSS_bytes')
QUALIFICATION_TOLERANCES = dict(logits=dict(atol=2e-4, rtol=2e-4),
    topk_probabilities=dict(atol=2e-5, rtol=2e-4), metrics=dict(atol=1e-6, rtol=0),
    tokens='exact', topk_ids='exact', EOS='exact', row_mapping='exact', seed_stream='exact',
    validity_counts_and_missing_reasons='exact')


class RepairEvidencePending(RuntimeError):
    """Typed missing technical evidence, not an invented scientific success."""


def shared_compatibility_api():
    """Load exact pure SH1 files, avoiding its model-importing package __init__."""
    package = __name__ + '._shared_readonly'
    source = Path(__file__).resolve().parent.parent / 'experiment_generation_eval'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(source)]
        sys.modules[package] = module
    for name in ('common', 'compatibility'):
        key = package + '.' + name
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, source / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules[key]
                raise
    return sys.modules[package + '.compatibility']


def repair_enabled(config):
    return config.get('task_id') == REPAIR_TASK or 'repair' in config.get('generation', {})


def task_authority(config, lock):
    """The repair has its own authority; original r1 remains independently valid."""
    if repair_enabled(config):
        require(config['task_id'] == REPAIR_TASK
                and config['instruction_id'] == lock['instruction_id'] == REPAIR_NONCE
                and config['parent_task_id'] == TASK, 'COLLECT_REPAIR_TASK_AUTHORITY')
    else:
        require(config['task_id'] == TASK
                and config['instruction_id'] == lock['instruction_id'] == NONCE,
                'COLLECT_TASK_AUTHORITY')
    require(lock.get('task_id', config['task_id']) == config['task_id'], 'COLLECT_LOCK_TASK_IDENTITY')


def qualification_evidence(reader, config, lock, attempt, *, records=None):
    """PLAN is admissible before tests, but is never actual route qualification."""
    repair = config['generation']['repair']
    plan_member = repair['qualification_plan']
    plan = reader.bound(plan_member)
    plan_sha = digest(plan)
    require(plan['schema'] == 'gptj-generation-cache-qualification-plan-v1'
            and plan['status'] == 'CPU_PLAN_FROZEN_NOT_GPU_PASS'
            and repair['qualification_plan_sha256'] == plan_sha, 'QUALIFICATION_PLAN_BINDING')
    require(plan['model'] == 'gptj' and plan['profile'] == PROFILE and plan['eval_seed'] == EVAL_SEED
            and plan['max_prompts'] == 8 and type(plan['cohort_count']) is int
            and 1 <= plan['cohort_count'] <= 8 and plan['tolerances'] == QUALIFICATION_TOLERANCES
            and plan['routes'] == list(ROUTES) and plan['batch_microbatch'] in (4, 8)
            and plan['admission_reason'] == ('DEFAULT_MB8' if plan['batch_microbatch'] == 8
                                             else 'PREDECLARED_MEMORY_MB4')
            and plan['automatic_OOM_retry'] is False and plan['measure_each_route_once'] is True
            and plan['no_fit'] is True and plan['checkpoint_saved'] is False,
            'QUALIFICATION_FROZEN_PROFILE_TOLERANCES')
    cohort = reader.bound(repair['qualification_cohort'])
    require(cohort['schema'] == plan['schema'] and cohort['raw_local_only'] is True
            and digest(cohort) == plan['cohort_sha256']
            and len(cohort['prompts']) == plan['cohort_count'], 'QUALIFICATION_FROZEN_COHORT_BYTES')
    keys = [(value['occurrence'], value['prompt_index']) for value in cohort['prompts']]
    require(len(keys) == len(set(keys)), 'QUALIFICATION_COHORT_UNIQUE_PROMPTS')
    if 'shared_plan' in cohort:
        shared_plan = cohort['shared_plan']
        require(keys == [(value['occurrence'], value['prompt_index']) for value in shared_plan['requests']]
                and plan['shared_plan_sha256'] == digest(shared_plan), 'QUALIFICATION_COHORT_SHARED_PLAN_ORDER')
        shared_bound = repair['shared_qualification_plan']
        require(reader.bound(shared_bound) == shared_plan
                and repair['shared_qualification_plan_sha256'] == digest(shared_plan),
                'QUALIFICATION_COHORT_FROZEN_SHARED_PLAN_MEMBER')
    else:
        require(keys == sorted(keys), 'QUALIFICATION_COHORT_LEGACY_ORDER')
    for value in cohort['prompts']:
        require(type(value['occurrence']) is int and 1 <= value['occurrence'] <= 2000
                and type(value['prompt_index']) is int and value['prompt_index'] >= 0
                and type(value['input_token_count']) is int and value['input_token_count'] > 0
                and type(value['prompt']) is str, 'QUALIFICATION_COHORT_LOCAL_SCHEMA')
        if records is not None:
            record = records[value['occurrence'] - 1]
            require(value['case_id'] == record['case_id']
                    and value['prompt_index'] < len(record.get('generation_prompts', []))
                    and value['prompt'] == record['generation_prompts'][value['prompt_index']],
                    'QUALIFICATION_COHORT_FROZEN_INPUT')
    result = dict(status='BLOCKED_QUALIFICATION_PENDING', plan_status='PLAN_BOUND_NOT_ACTUAL_PASS',
        plan_sha256=plan_sha, plan_member_sha256=plan_member['sha256'], actual_qualification=False)
    actual_path = repair.get('qualification_receipt_path')
    if not actual_path or not Path(actual_path).is_file():
        return result
    actual_path = local_path(actual_path, attempt)
    actual = reader.json(actual_path)
    result.update(actual_member_sha256=reader.files[str(actual_path)]['sha256'])
    require(actual['schema'] == 'gptj-generation-cache-qualification-v1'
            and actual['plan_sha256'] == plan_sha
            and actual['cohort_sha256'] == plan['cohort_sha256']
            and actual['model_identity'] == plan['model_identity'] == config['generation']['model_identity']
            and actual['shared_source_sha'] == plan['shared_source_sha'] == config['generation']['source_sha']
            and actual['native_source_binding'] == plan['native_source_binding'],
            'QUALIFICATION_ACTUAL_PLAN_SOURCE')
    if 'shared_plan' in cohort:
        route_map = {key: SHARED_ROUTE[key] for key in ROUTES}
        require(actual['shared_plan_sha256'] == plan['shared_plan_sha256']
                and actual['shared_route_map'] == plan['shared_route_map'] == route_map,
                'QUALIFICATION_ACTUAL_SHARED_PLAN_ROUTE_MAP')
    require(actual['checkpoint_saved'] is False and actual['no_fit'] is True,
            'QUALIFICATION_NO_FIT_NOCP')
    if actual['status'] == 'CPU_FIXTURE_NOT_ACTUAL_QUALIFICATION' or actual.get('actual_GPU') is not True:
        result['status'] = 'BLOCKED_ACTUAL_QUALIFICATION_NOT_GPU'
        return result
    if actual['status'] != 'QUALIFIED_ACTUAL_GPU_ROUTE':
        result['status'] = 'BLOCKED_ACTUAL_QUALIFICATION_FAILED'
        return result
    require(plan.get('fixture_only') is False, 'QUALIFICATION_FIXTURE_PLAN_NOT_ACTUAL_PROOF')
    native = plan['native_source_binding']
    require(native['versions'] == dict(torch='2.9.1+cu128', transformers='4.57.1')
            and [value['relative'] for value in native['files']] == [
                'transformers/models/gptj/modeling_gptj.py', 'transformers/cache_utils.py'],
            'QUALIFICATION_NATIVE_SOURCE_VERSIONS')
    for source in native['files']:
        bound_bytes(reader, source, 'QUALIFICATION_NATIVE_SOURCE_BYTES')
    route, microbatch = actual['selected_route'], actual['fixed_microbatch']
    require(route in ROUTES
            and type(microbatch) is int
            and microbatch == (plan['batch_microbatch'] if route == ROUTES[2] else 1)
            and actual['selected_route_passed'] is True and actual['state_unchanged'] is True
            and actual['RNG_restored'] is True and actual['tolerances'] == plan['tolerances']
            and actual['error'] is None, 'QUALIFICATION_SELECTED_ROUTE_STATE_RNG')
    require(set(actual['route_results']) == set(actual['work']) == set(actual['cost']) == set(ROUTES),
            'QUALIFICATION_ALL_THREE_MEASURED_ROUTES')
    for measured_route in ROUTES:
        measured = actual['route_results'][measured_route]
        require(set(measured['gates']) == set(QUALIFICATION_GATES)
                and all(type(value) is bool for value in measured['gates'].values())
                and measured['status'] == ('PASS' if all(measured['gates'].values()) else 'FAILED_GATE'),
                'QUALIFICATION_ROUTE_GATE_EVIDENCE')
        for key in ('max_logit_abs_error', 'max_topk_probability_abs_error'):
            require(finite(measured[key], 'QUALIFICATION_NUMERIC_EVIDENCE') >= 0,
                    'QUALIFICATION_ERROR_NONNEGATIVE')
        work = actual['work'][measured_route]
        require(set(work) == set(QUALIFICATION_WORK), 'QUALIFICATION_WORK_SCALAR_PRIVACY')
        for key in QUALIFICATION_WORK:
            integer(work[key], 'QUALIFICATION_MEASURED_WORK_INTEGER')
        require(work['physical_forward_calls'] > 0 and work['prefill_query_tokens'] > 0,
                'QUALIFICATION_ACTUAL_PHYSICAL_WORK_REQUIRED')
        cost = actual['cost'][measured_route]
        require(set(cost) == set(QUALIFICATION_COST), 'QUALIFICATION_COST_SCALAR_PRIVACY')
        for key in ('elapsed_seconds', 'synchronized_GPU_seconds'):
            require(finite(cost[key], 'QUALIFICATION_MEASURED_TIME') >= 0, 'QUALIFICATION_TIME_NONNEGATIVE')
        for key in ('peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes', 'peak_host_RSS_bytes'):
            integer(cost[key], 'QUALIFICATION_MEASURED_MEMORY_INTEGER')
        require(cost['peak_gpu_allocated_bytes'] > 0 and cost['peak_host_RSS_bytes'] > 0
                and cost['peak_gpu_reserved_bytes'] >= cost['peak_gpu_allocated_bytes'],
                'QUALIFICATION_ACTUAL_GPU_MEMORY_REQUIRED')
    selected = actual['route_results'][route]
    require(selected['status'] == 'PASS' and all(selected['coverage'].get(key) is True
                for key in QUALIFICATION_COVERAGE), 'QUALIFICATION_SELECTED_COVERAGE')
    if route == ROUTES[2]:
        require(selected['coverage']['actual_max_microbatch'] == microbatch
                and selected['coverage']['active_row_removal'] is True
                and plan['coverage']['same_length_width'] >= microbatch,
                'QUALIFICATION_BATCH_MAPPING_FIXED_MB')
    chosen = next((key for key in reversed(ROUTES) if actual['route_results'][key]['status'] == 'PASS'), None)
    require(chosen == route, 'QUALIFICATION_PREDECLARED_FALLBACK_ORDER')
    result.update(status='QUALIFICATION_ACTUAL_VERIFIED', actual_qualification=True,
                  selected_route=route, fixed_microbatch=microbatch,
                  measured_route_work=actual['work'], measured_route_cost=actual['cost'],
                  qualification_timers_not_production_ETA=True)
    return result


def qualification_link(reader, config, holder, qualification, attempt):
    require(qualification.get('actual_qualification') is True, 'QUALIFICATION_ACTUAL_REQUIRED')
    bound = holder['qualification']
    require(bound['path'] == config['generation']['repair']['qualification_receipt_path']
            and bound['sha256'] == qualification['actual_member_sha256'], 'QUALIFICATION_MEMBER_LINK')
    reader.bound(bound)
    require(holder['qualification_plan_sha256'] == qualification['plan_sha256']
            and holder['selected_route'] == qualification['selected_route']
            and holder['fixed_microbatch'] == qualification['fixed_microbatch'],
            'QUALIFICATION_RUNTIME_SELECTED_ROUTE_LINK')


def repair_runtime_holder(reader, out):
    """Startup runtime is immutable; actual route evidence is an append-only file."""
    auxiliary = Path(out) / 'generation-repair-runtime.json'
    original = Path(out) / 'runtime.json'
    if auxiliary.exists():
        return reader.json(auxiliary)
    return reader.json(original) if original.exists() else None


def progress_member(reader, holder, records, attempt, *, ready=False, consumer=False):
    """Only the producer needs a final progress ledger; reuse is explicit."""
    if consumer:
        reused = holder.get('reused_complete_READY')
        progress = holder.get('generation_progress')
        if type(reused) is dict:
            require(reused.get('status') == 'REUSED_COMPLETE_READY' and reused.get('completed_cases') == 2000,
                    'GEN_CONSUMER_EXPLICIT_READY_REUSE')
            bound = reused['READY']
        else:
            require(type(progress) is dict and progress['status'] == 'REUSED_COMPLETE_READY_NO_NEW_GENERATION',
                    'GEN_CONSUMER_EXPLICIT_READY_REUSE')
            bound = progress['ready_member']
        local_path(bound['path'], attempt)
        ready_value = reader.bound(bound)
        require(ready_value['status'] == 'READY'
                and ready_value['identity']['ordered_occurrences'] == list(range(1, 2001)),
                'GEN_CONSUMER_COMPLETE_READY_MEMBER')
        if type(progress) is dict and progress.get('status') == 'REUSED_COMPLETE_READY_NO_NEW_GENERATION':
            require(progress['producer_generation_progress_member'] == ready_value['generation_progress'],
                    'GEN_CONSUMER_ORIGINAL_PRODUCER_PROGRESS_MEMBER')
            producer = progress_member(reader, ready_value, records, attempt, ready=True)
            require(producer['W0_mean_ready'] is True, 'GEN_CONSUMER_PRODUCER_COMPLETE_PROGRESS')
        if type(reused) is dict and 'producer_generation_progress_member' in reused:
            require(reused['new_generation'] == 0
                    and reused['producer_generation_progress_member'] == ready_value['generation_progress'],
                    'GEN_CONSUMER_ZERO_NEW_ORIGINAL_PROGRESS')
            producer = progress_member(reader, ready_value, records, attempt, ready=True)
            require(producer['W0_mean_ready'] is True, 'GEN_CONSUMER_PRODUCER_COMPLETE_PROGRESS')
        return dict(status='REUSED_COMPLETE_READY', completed_cases=2000, W0_mean_ready=True,
                    READY_sha256=bound['sha256'])
    bound = holder.get('generation_progress')
    if not bound:
        return dict(status='BLOCKED_GENERATION_PROGRESS_PENDING' if ready else 'PROGRESS_NOT_RECORDED',
                    W0_mean_ready=False)
    local_path(bound['path'], attempt)
    data = reader.bytes(bound['path'])
    require(len(data) <= 16 * 1024**2 and len(data) == bound['bytes']
            and reader.files[str(Path(bound['path']))]['sha256'] == bound['sha256'],
            'GEN_PROGRESS_BOUND_BYTES')
    values = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    return validate_generation_progress(values, total_cases=2000,
        total_prompts=sum(len(record.get('generation_prompts', [])) for record in records), ready=ready)


def semantic_inputs(config):
    """Source/asset identities only, never tensor loading or tokenizer replay."""
    generation = config['generation']
    model = digest(dict(model=config['model'], revision=config['model_revision'],
        model_assets=config['model_assets'], runtime=config['runtime'], scorer=config['observer_identity'],
        seed=config['seed'], precision='FP32/eager/TF32off/autocastoff'))
    require(model == generation['model_identity'], 'COMPATIBILITY_MODEL_TOKENIZER_IDENTITY')
    return dict(model_identity=model, model_assets_sha256=digest(config['model_assets']),
        model_revision=config['model_revision'], tokenizer_binding_sha256=digest(config['model_assets']),
        scientific_runtime_sha256=digest(config['runtime']), scoring_versions=generation['scoring_versions'],
        scorer_identity_sha256=digest(config['observer_identity']),
        reference_assets_sha256=generation['reference_assets_sha256'],
        native_inputs_sha256=digest({arm: value['native'] for arm, value in config['arm_configs'].items()}),
        cold_state_identity=generation['W0_state_identity'], profile=PROFILE, eval_seed=EVAL_SEED, schema=SCHEMA)


def bound_bytes(reader, bound, label):
    data = reader.bytes(bound['path'])
    require(len(data) == bound['bytes']
            and reader.files[str(Path(bound['path']))]['sha256'] == bound['sha256'], label)


def compatibility_evidence(reader, config, qualification, records, *, compatibility_member=None):
    """Independently retain original completed W0 row provenance; no relabeling."""
    require(qualification.get('actual_qualification') is True, 'COMPATIBILITY_ACTUAL_QUALIFICATION_REQUIRED')
    bound = (config['generation']['repair'].get('compatibility_manifest')
             if compatibility_member is None else compatibility_member)
    if bound is None:
        return dict(status='BLOCKED_COMPATIBILITY_MANIFEST_PENDING', full_W0_READY=False)
    manifest = reader.bound(bound)
    require(manifest['schema'] == 'gptj-generation-w0-compatibility-v1'
            and manifest['status'] == 'ACTUAL_QUALIFIED_REUSE_BINDING'
            and manifest['identity_sha256'] == digest(manifest['identity'])
            and manifest['raw_local_only'] is True and manifest['new_source_relabel'] is False
            and manifest['original_runtime_checks_disabled'] is False and manifest['full_W0_READY'] is False,
            'COMPATIBILITY_MANIFEST_ACTUAL_IDENTITY')
    identity = manifest['identity']
    require(identity['qualification']['sha256'] == qualification['actual_member_sha256']
            and identity['qualification']['path'] == config['generation']['repair']['qualification_receipt_path'],
            'COMPATIBILITY_QUALIFICATION_MEMBER')
    actual = reader.bound(identity['qualification'])
    expected = identity['qualification_expected']
    require({'plan_sha256', 'cohort_sha256', 'shared_source_sha', 'native_source_binding',
             'selected_route', 'fixed_microbatch'} <= set(expected)
            and all(actual.get(key) == value for key, value in expected.items()),
            'COMPATIBILITY_EXACT_ACTUAL_FIELDS')
    inventory = reader.bound(identity['inventory'])
    old = inventory['identity']
    require(inventory['schema'] == old['schema'] == 'gptj-generation-old-w0-inventory-v1'
            and inventory['identity_sha256'] == identity['inventory_identity_sha256'] == digest(old)
            and inventory['payload_sha256'] == digest({key: value for key, value in inventory.items()
                                                       if key != 'payload_sha256'})
            and inventory['raw_local_only'] is True and inventory['new_source_relabel'] is False,
            'COMPATIBILITY_INVENTORY_SOURCE_IDENTITY')
    semantics = semantic_inputs(config)
    require(old['semantic_inputs'] == identity['semantic_inputs'] == semantics
            and identity['old_runtime_identity'] == old['old_runtime_identity']
            and identity['old_runtime_sha256'] == old['old_runtime_sha256'] == digest(old['old_runtime_identity']),
            'COMPATIBILITY_SEMANTIC_INPUTS_UNCHANGED')
    old_runtime = old['old_runtime_identity']
    require(old_runtime == dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
                model_identity=semantics['model_identity'],
                generation_source_sha='83535c6a47c552cc4e5c6385f3a587d752820150',
                reference_assets_sha256=semantics['reference_assets_sha256'], route=ROUTES[0]),
            'COMPATIBILITY_ORIGINAL_RUNTIME_ROUTE_SOURCE')
    new_runtime = identity['new_runtime_identity']
    require(identity['new_runtime_sha256'] == digest(new_runtime)
            and new_runtime['schema'] == SCHEMA and new_runtime['profile'] == PROFILE
            and new_runtime['eval_seed'] == EVAL_SEED and new_runtime['model_identity'] == semantics['model_identity']
            and new_runtime['reference_assets_sha256'] == semantics['reference_assets_sha256']
            and new_runtime['generation_source_sha'] == config['generation']['source_sha']
            and new_runtime['route'] in (qualification['selected_route'],
                SHARED_ROUTE[qualification['selected_route']]), 'COMPATIBILITY_NEW_RUNTIME_BINDING')
    private_plan = reader.bound(config['generation']['repair']['qualification_plan'])
    if 'shared_plan_sha256' in private_plan:
        require(new_runtime['route'] == SHARED_ROUTE[qualification['selected_route']]
                and identity['qualification_route_map'] == {key: SHARED_ROUTE[key] for key in ROUTES}
                and new_runtime['generation_microbatch'] == qualification['fixed_microbatch']
                and type(new_runtime['qualification_receipt_sha256']) is str
                and len(new_runtime['qualification_receipt_sha256']) == 64,
                'COMPATIBILITY_REAL_SHARED_RUNTIME_ROUTE_MAP')
    old_config, old_lock = reader.bound(old['old_config']), reader.bound(old['old_lock'])
    old_observer = reader.bound(old['old_observer'])
    require(old_lock['config_sha256'] == old['old_config']['sha256']
            and old_lock['source_commit'] == '503081fa9bc6efc4dbdd461324fba522b8f36e8b'
            and old_lock['shared_generation_source'] == old_runtime['generation_source_sha']
            and semantic_inputs(old_config) == semantics
            and old_observer['identity'] == old_runtime
            and old_observer['identity_sha256'] == digest(old_runtime), 'COMPATIBILITY_OLD_SEALED_SOURCE')
    require(old['source_members_sha256'] == digest(old_lock['source_members'])
            and old['runtime_source_members_sha256'] == digest(old_lock['runtime_sources']),
            'COMPATIBILITY_OLD_SOURCE_CLOSURE')
    for source in old_lock['source_members'] + old_lock['runtime_sources']:
        bound_bytes(reader, source, 'COMPATIBILITY_OLD_SOURCE_BYTES')
    # Reuse previously sealed large assets via stat only, never model/tensor IO.
    for asset in old_config['model_assets']:
        path = Path(asset['path'])
        require(path.is_file() and not path.is_symlink(), 'COMPATIBILITY_ASSET_FILE')
        value = path.stat()
        require((value.st_size, value.st_ino, value.st_mtime_ns)
                == (asset['bytes'], asset['inode'], asset['mtime_ns'])
                and type(asset['sha256']) is str and len(asset['sha256']) == 64,
                'COMPATIBILITY_PRIOR_ASSET_STAT_SEAL')
    expected_records = []
    for ordinal, record in enumerate(records, 1):
        rewrite = record['requested_rewrite']
        expected_records.append(dict(ordered_occurrence=ordinal, case_id=record['case_id'],
            generation_prompts=record.get('generation_prompts', []), relation_id=rewrite.get('relation_id'),
            target_new_id=rewrite.get('target_new', {}).get('id')))
    require(len(records) == inventory['planned_cases'] == manifest['planned_cases'] == 2000
            and old['ordered_records_sha256'] == identity['ordered_records_sha256'] == digest(expected_records)
            and [entry['occurrence'] for entry in inventory['entries']] == list(range(1, 2001)),
            'COMPATIBILITY_ORDERED_FIRST2000')
    reused, provenance = 0, []
    old_directory = Path(old['old_config']['path']).parent / 'BASE_MEMIT' / 'generation-raw' / 'observations'
    for entry, record in zip(inventory['entries'], expected_records):
        if entry['status'] != 'REUSABLE_COMPLETE_CASE':
            require(entry['status'] == 'NOT_REUSABLE', 'COMPATIBILITY_CLOSED_ROW_STATUS')
            continue
        proof, row = entry['provenance'], entry['row']
        require(proof['raw']['path'] == str(old_directory / (entry['expected_identity_sha256'] + '.json')),
                'COMPATIBILITY_EXACT_ORIGINAL_RAW_PATH')
        raw = reader.bound(proof['raw'])
        original_identity = dict(runtime=digest(old_runtime), state_identity=semantics['cold_state_identity'],
                                 record_identity=record)
        require(raw['identity'] == original_identity
                and raw['identity_sha256'] == row['identity_sha256'] == proof['original_identity_sha256']
                    == entry['expected_identity_sha256'] == digest(original_identity)
                and raw['payload_sha256'] == row['payload_sha256'] == proof['original_payload_sha256']
                    == digest({key: value for key, value in raw.items() if key != 'payload_sha256'})
                and raw['occurrence'] == row['occurrence'] == entry['occurrence']
                and raw['case_id'] == row['case_id'] == record['case_id']
                and raw['metrics'] == row['metrics'] and raw['raw_local_only'] is True
                and raw['checkpoint_saved'] is False and row['observation_path'] == proof['raw']['path'],
                'COMPATIBILITY_ORIGINAL_ROW_PAYLOAD')
        require(proof['original_runtime_sha256'] == digest(old_runtime)
                and proof['original_source_sha'] == old_runtime['generation_source_sha']
                and proof['original_route'] == ROUTES[0]
                and proof['record_identity_sha256'] == digest(record)
                and proof['prompt_seed_stream_sha256'] == digest([value['seed'] for value in raw['observations']])
                and proof['input_token_bindings_sha256'] == digest([value['input_token_ids']
                                                                   for value in raw['observations']]),
                'COMPATIBILITY_UNCHANGED_ORIGINAL_PROVENANCE')
        require(len(raw['observations']) == proof['prompt_count'] == len(record['generation_prompts']),
                'COMPATIBILITY_COMPLETED_PROMPT_COVERAGE')
        tokens = capped = 0
        for index, (observed, prompt) in enumerate(zip(raw['observations'], record['generation_prompts'])):
            n, _, cap = validate_prompt(observed, prompt, entry['occurrence'], index, semantics['model_identity'])
            tokens += n
            capped += cap
        require(row['metrics']['generation_prompt_count'] == proof['prompt_count']
                and row['metrics']['generated_token_count'] == tokens
                and row['metrics']['length_cap_no_continuation_count'] == capped,
                'COMPATIBILITY_LOGICAL_CASE_COUNTS')
        reduce_generation([row])  # Independent finite/validity/missingness checks.
        reused += 1
        provenance.append(proof)
    require(reused == inventory['reusable_cases'] == manifest['reusable_cases']
            and inventory['unknown_cases'] == 2000 - reused
            and identity['provenance_bindings'] == provenance, 'COMPATIBILITY_ALL_OLD_PROVENANCE_BINDINGS')
    return dict(status='COMPATIBILITY_ACTUAL_VERIFIED', manifest_sha256=bound['sha256'],
        inventory_sha256=identity['inventory']['sha256'], eligible_reused_cases=reused,
        missing_or_unknown_cases=2000 - reused, planned_cases=2000, full_W0_READY=False,
        original_runtime_sha256=digest(old_runtime), new_runtime_sha256=digest(new_runtime),
        newly_generated_physical_work=0, source_relabel=False)


def validate_generation_progress(rows, *, total_cases, total_prompts, ready=False):
    """Check scalar W0 progress independently of edit/candidate axes and scores."""
    counters = ('completed_cases', 'total_cases', 'completed_prompts', 'total_prompts',
        'generated_tokens', 'new_cases', 'reused_cases', 'physical_forward_calls',
        'prefill_query_tokens', 'decode_query_tokens', 'step')
    rates = ('elapsed_sec', 'cases_per_sec', 'prompts_per_sec', 'tokens_per_sec')
    previous = None
    for row in rows:
        allowed = {'phase', 'route', 'model', 'job_id', 'event', 'status', 'edits'}
        allowed.update('generation_progress/' + key for key in counters + rates)
        require(type(row) is dict and set(row) <= allowed, 'GEN_PROGRESS_SCALAR_PRIVACY_KEYS')
        require(row['phase'] == 'W0_generation', 'GEN_PROGRESS_PUBLIC_PHASE')
        require('edits' not in row or type(row['edits']) is int and row['edits'] == 0,
                'GEN_PROGRESS_NOT_EDIT_AXIS')
        values = {key: row['generation_progress/' + key] for key in counters + rates}
        for key in counters:
            integer(values[key], 'GEN_PROGRESS_INTEGER')
        for key in rates:
            require(finite(values[key], 'GEN_PROGRESS_FINITE') >= 0, 'GEN_PROGRESS_NONNEGATIVE')
        require(values['total_cases'] == total_cases and values['total_prompts'] == total_prompts
                and values['completed_cases'] <= total_cases and values['completed_prompts'] <= total_prompts
                and values['new_cases'] + values['reused_cases'] == values['completed_cases'],
                'GEN_PROGRESS_DENOMINATORS')
        if previous is not None:
            require(values['step'] > previous['step'], 'GEN_PROGRESS_STEP_MONOTONIC')
            require(all(values[key] >= previous[key] for key in (
                'completed_cases', 'completed_prompts', 'generated_tokens', 'new_cases', 'reused_cases',
                'physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens', 'elapsed_sec')),
                'GEN_PROGRESS_COUNTER_MONOTONIC')
        previous = values
    complete = bool(previous and previous['completed_cases'] == total_cases
                    and previous['completed_prompts'] == total_prompts)
    if ready:
        require(complete and total_cases == 2000, 'GEN_PROGRESS_READY_REQUIRES_FULL2000')
    return dict(status='PROGRESS_COMPLETE' if complete else 'PROGRESS_PARTIAL',
        completed_cases=None if previous is None else previous['completed_cases'],
        total_cases=total_cases, completed_prompts=None if previous is None else previous['completed_prompts'],
        total_prompts=total_prompts, final_step=None if previous is None else previous['step'],
        final_counters=previous, W0_mean_ready=bool(ready and complete))


def integer(value, label):
    require(type(value) is int and value >= 0, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def compare_value(actual, saved, label):
    """No loose key matching, missing mean substitution, or scientific gate."""
    if isinstance(actual, dict):
        require(type(saved) is dict and set(actual) == set(saved), label + '_KEYS')
        for key in actual:
            compare_value(actual[key], saved[key], label)
    elif type(actual) is float:
        finite(saved, label)
        require(math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12), label)
    else:
        require(type(actual) is type(saved) and actual == saved, label)


def reduce_generation(rows):
    """Independent stdlib arithmetic over per-occurrence stored SH1 metrics."""
    entropy, cosine = [], []
    prompt_count = token_count = capped_count = 0
    reasons = {key: 0 for key in REASONS}
    for row in rows:
        metric = row['metrics']
        require(type(metric['fluency_valid']) is bool
                and type(metric['consistency_valid']) is bool, 'GEN_VALIDITY_BOOL')
        for flag, scalar, values in (('fluency_valid', 'ngram_entropy', entropy),
                                     ('consistency_valid', 'reference_score', cosine)):
            if metric[flag]:
                values.append(finite(metric[scalar], 'GEN_VALID_FINITE_SCALAR'))
            else:
                require(metric[scalar] is None, 'GEN_MISSING_NOT_ZERO')
        why = metric['reasons']
        require(type(why) is list and all(type(item) is str for item in why)
                and why == sorted(set(why)) and set(why) <= set(REASONS), 'GEN_REASON_VOCABULARY')
        for reason in why:
            reasons[reason] += 1
        prompts = integer(metric['generation_prompt_count'], 'GEN_PROMPT_COUNT')
        tokens = integer(metric['generated_token_count'], 'GEN_TOKEN_COUNT')
        capped = integer(metric['length_cap_no_continuation_count'], 'GEN_CAPPED_COUNT')
        require(capped <= prompts, 'GEN_CAPPED_PROMPT_BOUND')
        if not prompts:
            require(not metric['fluency_valid'] and not metric['consistency_valid']
                    and why == ['missing_generation_prompts'] and tokens == capped == 0,
                    'GEN_EMPTY_PROMPTS_TYPED_MISSING')
        prompt_count += prompts
        token_count += tokens
        capped_count += capped
    result = dict(planned_count=len(rows), fluency_count=len(entropy),
        consistency_count=len(cosine), fluency_sum=math.fsum(entropy),
        consistency_sum=math.fsum(cosine), missing_reason_counts=reasons,
        generation_prompt_count=prompt_count, generated_token_count=token_count,
        length_cap_no_continuation_prompt_count=capped_count,
        reason_count_unit='request_occurrences_nonexclusive', fluency_unit='bits',
        consistency_unit='cosine_0_to_1')
    if entropy:
        result['ngram_entropy'] = result['fluency_sum'] / len(entropy)
    if cosine:
        result['reference_score'] = result['consistency_sum'] / len(cosine)
    return result


def local_path(path, attempt):
    """Only explicitly referenced files within this attempt, never a scan."""
    path, attempt = Path(path), Path(attempt).resolve()
    require(path.is_absolute() and not path.is_symlink(), 'GEN_RAW_LOCAL_PATH')
    resolved = path.resolve()
    require(resolved.is_relative_to(attempt), 'GEN_RAW_OUTSIDE_ATTEMPT')
    return path


def runtime_identity(config):
    generation = config['generation']
    return dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
        model_identity=generation['model_identity'], generation_source_sha=generation['source_sha'],
        reference_assets_sha256=generation['reference_assets_sha256'],
        route='UNPADDED_FULL_PREFIX_NO_CACHE')


def validate_prompt(observation, prompt, occurrence, index, model_identity, *,
                    route='UNPADDED_FULL_PREFIX_NO_CACHE', microbatch=1):
    """Check recorded exact route/EOS/token/work relations without a tokenizer."""
    require(observation['profile'] == PROFILE and observation['prompt'] == prompt
            and observation['occurrence'] == occurrence and observation['prompt_index'] == index
            and observation['route'] == route
            and observation['RNG_restored'] is True, 'GEN_PROMPT_IDENTITY_ROUTE')
    seed = int(digest(dict(model_identity=model_identity, ordered_occurrence=occurrence,
                          prompt_index=index, eval_seed=EVAL_SEED))[:16], 16) % (2**63 - 1)
    require(type(observation['seed']) is int and observation['seed'] == seed, 'GEN_CASE_SEED')
    require(observation['sampling'] == dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100),
            'GEN_NATIVE_SAMPLING_PROFILE')
    original, generated, full = (observation[key] for key in
        ('input_token_ids', 'continuation_token_ids', 'full_token_ids'))
    require(all(type(values) is list and all(type(token) is int and token >= 0 for token in values)
                for values in (original, generated, full)) and original and full == original + generated,
            'GEN_TOKEN_SEQUENCE')
    n, m = len(original), len(generated)
    require(type(observation['input_token_count']) is int and type(observation['continuation_token_count']) is int
            and observation['input_token_count'] == n and observation['continuation_token_count'] == m
            and type(observation['text']) is str, 'GEN_RECORDED_TOKEN_LENGTH')
    binding = observation['eos_binding']
    require(type(binding) is dict and set(binding) <= {'generation_config', 'model_config', 'tokenizer'}
            and all(type(values) is list and all(type(token) is int and token >= 0 for token in values)
                    for values in binding.values()), 'GEN_EOS_BINDING')
    eos = sorted({token for values in binding.values() for token in values})
    require(observation['eos_ids'] == eos, 'GEN_EOS_UNION')
    stop = observation['stop_reason']
    if n >= 100:
        require(m == 0 and stop == 'length_cap_no_continuation', 'GEN_LONG_PROMPT_NO_CONTINUATION')
    else:
        require(0 < m <= 100 - n and not any(token in eos for token in generated[:-1]),
                'GEN_EOS_FIRST_TERMINATION')
        if stop == 'eos':
            require(generated[-1] in eos, 'GEN_EOS_LAST_TOKEN')
        else:
            require(stop == 'length_cap' and n + m == 100 and generated[-1] not in eos,
                    'GEN_TOTAL100_CAP')
    forwards = m
    work = n * m + m * (m - 1) // 2 if route == SHARED_ROUTES[0] else 0
    require(type(observation['model_forwards']) is int and observation['model_forwards'] == forwards
            and type(observation['full_prefix_token_work']) is int
            and observation['full_prefix_token_work'] == work, 'GEN_PROMPT_WORK_RELATION')
    if route != SHARED_ROUTES[0]:
        require(route in SHARED_ROUTES and 'qualification_forced_prefix_only' not in observation,
                'GEN_PRODUCTION_NOT_FORCED_QUALIFICATION')
        for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens'):
            integer(observation[key], 'GEN_PHYSICAL_WORK_INTEGER')
        require(observation['physical_forward_calls'] <= m
                and observation['prefill_query_tokens'] <= microbatch * n
                and observation['decode_query_tokens'] <= microbatch * m
                and (m > 0 or observation['physical_forward_calls'] == observation['prefill_query_tokens']
                     == observation['decode_query_tokens'] == 0), 'GEN_KV_PHYSICAL_WORK_BOUND')
    return forwards, work, stop == 'length_cap_no_continuation'


def validate_work(work, count, total_forwards, total_tokens, *, cached_only=False):
    require(type(work) is dict, 'GEN_WORK_REQUIRED')
    for key in ('new_case_observations', 'cached_case_observations', 'generation_forwards',
                'full_prefix_token_work'):
        integer(work[key], 'GEN_WORK_INTEGER')
    require(finite(work['seconds'], 'GEN_WORK_SECONDS') >= 0, 'GEN_WORK_SECONDS_NONNEGATIVE')
    fresh, cached = work['new_case_observations'], work['cached_case_observations']
    require(fresh + cached == count, 'GEN_WORK_CASE_COVERAGE')
    if cached_only:
        require(fresh == 0, 'GEN_SUBSET_NO_NEW_GENERATION')
    if not fresh:
        require(work['generation_forwards'] == work['full_prefix_token_work'] == 0,
                'GEN_CACHE_NO_RECHARGED_WORK')
    elif not cached:
        require(work['generation_forwards'] == total_forwards
                and work['full_prefix_token_work'] == total_tokens, 'GEN_FRESH_WORK_EXACT')
    else:
        # SH1 does not persist a per-row new/cache flag. Do not invent attribution.
        require(work['generation_forwards'] <= total_forwards
                and work['full_prefix_token_work'] <= total_tokens, 'GEN_MIXED_WORK_BOUND')
    return dict(work, work_attribution='EXACT' if not fresh or not cached else 'RECORDED_MIXED_BOUND')


def shared_qualification_evidence(reader, config, qualification, bound):
    """Verify SH1's real receipt and its explicitly projected GPT-J caller proof."""
    require(qualification.get('actual_qualification') is True, 'SHARED_QUALIFICATION_PRIVATE_PROOF_REQUIRED')
    actual = reader.bound(bound)
    require(actual['schema'] == 'gpt2-gptj-kv-fixed8-v1'
            and actual['identity_sha256'] == digest({key: value for key, value in actual.items()
                                                    if key != 'identity_sha256'})
            and actual['actual_qualification'] is True and actual['qualification_pass'] is True
            and actual['pretrained_GPU_PASS'] is True and actual['profile'] == PROFILE
            and actual['eval_seed'] == EVAL_SEED and actual['model_identity'] == config['generation']['model_identity']
            and actual['source_identity'] == config['generation']['source_sha']
            and all(actual[key] is True for key in ('model_no_mutation', 'RNG_restored', 'native_state_no_mutation'))
            and all(type(actual[key]) is int and actual[key] == 0
                    for key in ('fit_calls', 'edit_commits', 'retry_count')),
            'SHARED_ACTUAL_QUALIFICATION_SOURCE_STATE')
    require(actual['caller_proof_member']['sha256'] == qualification['actual_member_sha256']
            and actual['caller_proof_member']['path'] == config['generation']['repair']['qualification_receipt_path']
            and actual['execution_adapter'] == EXECUTION_ADAPTER,
            'SHARED_QUALIFICATION_EXPLICIT_CALLER_PROOF')
    reader.bound(actual['caller_proof_member'])
    proof = reader.bound(actual['caller_proof_member'])
    cohort = reader.bound(config['generation']['repair']['qualification_cohort'])
    plan = cohort['shared_plan']
    require(plan['schema'] == actual['schema'] and actual['plan_sha256'] == digest(plan)
            and plan['profile'] == PROFILE and plan['eval_seed'] == EVAL_SEED
            and plan['model_identity'] == config['generation']['model_identity']
            and 0 < len(plan['requests']) <= 8 and plan['max_prompts'] == 8
            and plan['actual_qualification'] is False and plan['fit_calls'] == plan['edit_commits'] == 0
            and plan['allowed_routes'] == list(SHARED_ROUTES) and plan['no_oom_retry'] is True
            and actual['tolerances'] == plan['tolerances']
                == {key: QUALIFICATION_TOLERANCES[key] for key in ('logits', 'topk_probabilities', 'metrics')},
            'SHARED_QUALIFICATION_FROZEN_PLAN')
    private_plan = reader.bound(config['generation']['repair']['qualification_plan'])
    require(private_plan['shared_plan_sha256'] == actual['plan_sha256']
            and actual['private_plan_sha256'] == qualification['plan_sha256'], 'SHARED_PRIVATE_PLAN_LINK')
    require(actual['raw_local_only'] is True and actual['native_models_scope'] == ['gptj']
            and actual['selection_reason'] == 'FIXED_GATE_ORDER_NOT_SCIENTIFIC_QUALITY'
            and actual['coverage'] == private_plan['coverage']
            and plan['microbatch'] == private_plan['batch_microbatch'], 'SHARED_GPTJ_CALLER_SCOPE')
    expected_results, expected_costs = {}, {}
    for private_route in ROUTES:
        result = proof['route_results'][private_route]
        gates, work, cost = result['gates'], proof['work'][private_route], proof['cost'][private_route]
        mapped = SHARED_ROUTE[private_route]
        expected_results[mapped] = dict(passed=result['status'] == 'PASS', executed=True,
            token_sequence_exact=gates['tokens'] and gates['EOS'] and gates['row_mapping'] and gates['seed_stream'],
            topk_mask_position_exact=gates['topk_ids'] and gates['positions'], logits_close=gates['logits'],
            topk_probabilities_close=gates['topk_probabilities'],
            metric_values_close_and_validity_reasons_exact=gates['metrics'],
            max_abs_logit_error=result['max_logit_abs_error'],
            max_abs_topk_probability_error=result['max_topk_probability_abs_error'], coverage=result['coverage'])
        expected_costs[mapped] = dict(elapsed_sec=cost['elapsed_seconds'], GPU_seconds=cost['synchronized_GPU_seconds'],
            peak_allocated_bytes=cost['peak_gpu_allocated_bytes'], peak_reserved_bytes=cost['peak_gpu_reserved_bytes'],
            host_max_RSS_bytes=cost['peak_host_RSS_bytes'],
            logical_row_forward_decisions=work['logical_row_token_decisions'],
            **{key: work[key] for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens')})
    compare_value(expected_results, actual['route_results'], 'SHARED_PRIVATE_ROUTE_PROJECTION')
    compare_value(expected_costs, actual['cost'], 'SHARED_PRIVATE_WORK_COST_PROJECTION')
    route = actual['selected_route']
    require(route == SHARED_ROUTE[qualification['selected_route']]
            and actual['microbatch'] == actual['fixed_microbatch'] == qualification['fixed_microbatch']
            and actual['route_results'][route]['passed'] is True, 'SHARED_QUALIFIED_ROUTE_MB')
    if route != SHARED_ROUTES[0]:
        selected = actual['route_results'][route]
        require(all(selected[key] is True for key in ('token_sequence_exact', 'topk_mask_position_exact',
            'logits_close', 'topk_probabilities_close', 'metric_values_close_and_validity_reasons_exact')),
            'SHARED_QUALIFICATION_NUMERIC_TOKEN_GATES')
        for key in ('max_abs_logit_error', 'max_abs_topk_probability_error'):
            require(finite(selected[key], 'SHARED_QUALIFICATION_FINITE_ERROR') >= 0,
                    'SHARED_QUALIFICATION_ERROR_NONNEGATIVE')
        if route == SHARED_ROUTES[2]:
            require(selected['coverage']['actual_max_microbatch'] == actual['fixed_microbatch']
                    and selected['coverage']['active_row_removal'] is True,
                    'SHARED_QUALIFICATION_REAL_BATCH_WIDTH')
    return dict(member_sha256=bound['sha256'], plan_sha256=actual['plan_sha256'],
                selected_route=route, fixed_microbatch=actual['fixed_microbatch'])


def repair_generation_endpoint(reader, receipt, selected, occurrences, expected_state, endpoint,
                               cohort, config, attempt, *, cached_only=False):
    qualification = qualification_evidence(reader, config, {}, attempt)
    if not qualification.get('actual_qualification'):
        raise RepairEvidencePending(qualification['status'])
    api = shared_compatibility_api()
    shared_runtime_identity, verified_endpoint_row = api.runtime_identity, api.verified_endpoint_row
    qualification_link(reader, config, receipt, qualification, attempt)
    path = local_path(receipt['rows_path'], attempt)
    require(receipt['raw_endpoint_member']['path'] == str(path), 'GEN_ENDPOINT_MEMBER_PATH')
    value = reader.bound(receipt['raw_endpoint_member'])
    shared = shared_qualification_evidence(reader, config, qualification, value['qualification_receipt_member'])
    require(receipt['shared_qualification_plan_sha256'] == shared['plan_sha256'],
            'GEN_SHARED_PLAN_COMPACT_LINK')
    runtime = shared_runtime_identity(dict(model_identity=config['generation']['model_identity'],
        generation_source_sha=config['generation']['source_sha'], generation_route=shared['selected_route'],
        generation_microbatch=shared['fixed_microbatch'],
        qualification_receipt_member=value['qualification_receipt_member']),
        config['generation']['reference_assets_sha256'])
    observer_path = path.parent.parent / 'observer-identity.json'
    if path == Path(config['generation'].get('W0_cache', '')) / 'W0-endpoint.json':
        observer_path = Path(attempt) / 'BASE_MEMIT' / 'generation-raw' / 'observer-identity.json'
    if 'observer_identity_member' in value:
        require(value['observer_identity_member']['path'] == str(observer_path), 'GEN_SHARED_OBSERVER_MEMBER_PATH')
        observer = reader.bound(value['observer_identity_member'])
    else:
        observer = reader.json(observer_path)
    identity = value['identity']
    require(observer['identity'] == runtime and observer['identity_sha256'] == digest(runtime)
            and observer['raw_local_only'] is True and observer['checkpoint_saved'] is False
            and identity['runtime'] == digest(runtime) and identity['state_sha256'] == digest(expected_state)
            and identity['endpoint'] == endpoint and identity['cohort'] == cohort
            and value['identity_sha256'] == digest(identity)
            and identity['qualification_receipt_sha256'] == shared['member_sha256']
            and receipt['identity'] == identity and receipt['identity_sha256'] == value['identity_sha256']
            and receipt['shared_runtime_identity'] == runtime and receipt['shared_state_identity'] == expected_state,
            'GEN_REPAIR_RUNTIME_QUALIFICATION_STATE')
    require(receipt['requests'] == len(selected)
            and receipt['cohort_identity'] == digest([r['case_id'] for r in selected])
            and all(holder[key] is True for holder in (value, receipt)
                    for key in ('RNG_restored', 'observer_no_mutation')) and value['raw_local_only'] is True,
            'GEN_REPAIR_COHORT_NONMUTATION')
    rows = value['rows']
    require(identity['ordered_occurrences'] == occurrences
            and [row['occurrence'] for row in rows] == occurrences and len(set(occurrences)) == len(occurrences)
            and all(type(row['occurrence']) is int and type(row['case_id']) is int for row in rows)
            and [row['case_id'] for row in rows] == [record['case_id'] for record in selected]
            and identity['observation_identities'] == [row['identity_sha256'] for row in rows]
            and identity['provenance_sha256'] == digest([row['provenance'] for row in rows]),
            'GEN_REPAIR_FULLROW_ORDER_PROVENANCE')
    original_entries = {}
    if 'compatibility_member' in value:
        manifest = reader.bound(value['compatibility_member'])
        compatible = manifest['identity']
        require(manifest['identity_sha256'] == digest(compatible) == identity['compatibility_sha256']
                and compatible['schema'] == 'generation-cold-W0-compatibility-v1'
                and compatible['new_runtime_sha256'] == identity['runtime']
                and compatible['qualification_receipt_member']['sha256'] == shared['member_sha256']
                and compatible['physical_state'] == expected_state == config['generation']['W0_state_identity']
                and compatible['model_identity'] == runtime['model_identity']
                and compatible['reference_assets_sha256'] == runtime['reference_assets_sha256']
                and compatible['raw_local_only'] is True and compatible['edited_trajectory_resume'] is False,
                'GEN_SHARED_COMPATIBILITY_EXPLICIT_COLD_ONLY')
        require(compatible['actual_qualification_receipt_member']['sha256'] == qualification['actual_member_sha256']
                and compatible['qualification_plan_member']['sha256']
                    == config['generation']['repair']['qualification_plan']['sha256'],
                'GEN_SHARED_COMPATIBILITY_DUAL_PROOF')
        reader.bound(compatible['actual_qualification_receipt_member'])
        reader.bound(compatible['qualification_plan_member'])
        inventory = reader.bound(compatible['inventory_member'])
        old = inventory['identity']
        require(inventory['identity_sha256'] == digest(old)
                and inventory['payload_sha256'] == digest({key: value for key, value in inventory.items()
                                                           if key != 'payload_sha256'})
                and old['semantic_inputs'] == semantic_inputs(config)
                and compatible['old_config_member'] == old['old_config']
                and compatible['old_observer_member'] == old['old_observer']
                and compatible['ordered_record_identity_sha256'] == old['ordered_records_sha256']
                and compatible['planned_cases'] == inventory['planned_cases'] == 2000,
                'GEN_SHARED_ORIGINAL_INVENTORY_BINDING')
        frozen_inventory = config['generation']['repair'].get('old_complete_case_inventory')
        if frozen_inventory is not None:
            require(compatible['inventory_member']['sha256'] == frozen_inventory['sha256'],
                    'GEN_SHARED_FROZEN_INVENTORY')
        for key in ('old_observer_member', 'old_config_member', 'old_reference_member', 'old_cold_guard_member'):
            reader.bound(compatible[key])
        old_observer = reader.bound(compatible['old_observer_member'])
        require(old_observer['identity_sha256'] == digest(old_observer['identity'])
                and old_observer['identity']['model_identity'] == runtime['model_identity']
                and old_observer['identity']['reference_assets_sha256'] == runtime['reference_assets_sha256']
                and old_observer['identity']['profile'] == PROFILE
                and old_observer['identity']['eval_seed'] == EVAL_SEED
                and old_observer['identity']['route'] == SHARED_ROUTES[0]
                and old_observer['identity'] == old['old_runtime_identity'], 'GEN_ORIGINAL_RUNTIME_UNCHANGED')
        old_config = reader.bound(compatible['old_config_member'])
        old_reference = reader.bound(compatible['old_reference_member'])
        require(old_config['generation']['source_sha'] == old_observer['identity']['generation_source_sha']
                and compatible['old_reference_member'] == old_config['generation']['generation_assets']
                and old_reference['identity_sha256'] == runtime['reference_assets_sha256']
                and semantic_inputs(old_config) == semantic_inputs(config), 'GEN_ORIGINAL_CONFIG_REFERENCE')
        guard = reader.bound(compatible['old_cold_guard_member'])
        require(guard['source_commit'] == '503081fa9bc6efc4dbdd461324fba522b8f36e8b'
                and guard['phase'] == 'W0_generation' and guard['commits'] == guard['history_appends'] == 0
                and guard['old_generation_runtime'] == old_observer['identity_sha256']
                and guard['original_state_identity'] == expected_state
                and guard['model_W'] == expected_state.get('selected_physical_W', expected_state.get('W'))
                and guard['inventory_member'] == compatible['inventory_member']
                and guard['whole_endpoint_guard_recorded'] is False
                and guard['proof_basis'] == 'FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN'
                and guard['partial_rows_authorized'] is True
                and guard['measured_final_history_zero'] is False
                and guard['final_RAM_history'] == 'NOT_RECORDED', 'GEN_OLD_PARTIAL_GUARD_NOT_FABRICATED')
        for guarded_member in guard['evidence_members'].values():
            bound_bytes(reader, guarded_member, 'GEN_OLD_SOURCE_PHASE_PROOF')
        original_entries = {entry['occurrence']: entry for entry in compatible['original_entries']}
        require(len(original_entries) == len(compatible['original_entries']) == compatible['eligible_cases'],
                'GEN_ORIGINAL_ROW_DUPLICATE')
        inventory_entries = {entry['occurrence']: entry for entry in inventory['entries']
                             if entry['status'] == 'REUSABLE_COMPLETE_CASE'}
        require(set(original_entries) == set(inventory_entries), 'GEN_SHARED_COMPLETE_ORIGINAL_ENTRY_SET')
        for ordinal, entry in original_entries.items():
            proof = inventory_entries[ordinal]['provenance']
            require(entry == dict(occurrence=ordinal, original_raw_member=proof['raw'],
                original_identity_sha256=proof['original_identity_sha256'],
                original_payload_sha256=proof['original_payload_sha256'],
                original_runtime_sha256=proof['original_runtime_sha256'],
                original_generation_source_sha=proof['original_source_sha'],
                original_route=proof['original_route']), 'GEN_SHARED_ORIGINAL_ENTRY_NOT_RELABELLED')
    physical = {key: 0 for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens')}
    logical = prefix_work = 0
    all_prompts = all_tokens = 0
    fresh = reused = 0
    for row, record, ordinal in zip(rows, selected, occurrences):
        rewrite = record['requested_rewrite']
        ri = dict(ordered_occurrence=ordinal, case_id=record['case_id'], generation_prompts=record.get('generation_prompts', []),
                  relation_id=rewrite.get('relation_id'), target_new_id=rewrite.get('target_new', {}).get('id'))
        proof = row['provenance']
        raw_path = Path(row['observation_path'])
        if proof['origin'] == 'COMPATIBLE_ORIGINAL_W0':
            entry = original_entries.get(ordinal)
            require(entry is not None and entry['original_raw_member'] == proof['raw_member']
                    and proof['compatibility_sha256'] == identity['compatibility_sha256']
                    and proof['runtime_sha256'] == old_observer['identity_sha256']
                    and proof['generation_source_sha'] == old_observer['identity']['generation_source_sha']
                    and proof['route'] == SHARED_ROUTES[0], 'GEN_OLD_ORIGINAL_PROVENANCE')
        else:
            require(proof['origin'] in ('NEW_CURRENT_RUNTIME', 'CURRENT_RUNTIME_CACHE'), 'GEN_CLOSED_ROW_ORIGIN')
            local_path(raw_path, attempt)
            require(proof['runtime_sha256'] == identity['runtime']
                    and proof['generation_source_sha'] == config['generation']['source_sha']
                    and proof['route'] == shared['selected_route'], 'GEN_NEW_SOURCE_ROUTE_PROVENANCE')
        require(proof['raw_member']['path'] == str(raw_path), 'GEN_FULLROW_ORIGINAL_PATH')
        raw = reader.bound(proof['raw_member'])
        require(raw == verified_endpoint_row(row, value, expected_record_identity=ri, expected_state=expected_state),
                'GEN_SHARED_VERIFIED_ORIGINAL_DOCUMENT')
        require(len(raw['observations']) == len(ri['generation_prompts']), 'GEN_FULLPROMPT_COVERAGE')
        tokens = capped = 0
        is_fresh = proof['origin'] == 'NEW_CURRENT_RUNTIME' and not cached_only
        fresh += int(is_fresh)
        reused += int(not is_fresh)
        for index, (observed, prompt) in enumerate(zip(raw['observations'], ri['generation_prompts'])):
            n, work, cap = validate_prompt(observed, prompt, ordinal, index, runtime['model_identity'],
                route=proof['route'], microbatch=shared['fixed_microbatch'])
            tokens += n
            capped += cap
            if is_fresh:
                logical += n
                prefix_work += work
                for key in physical:
                    physical[key] += observed.get(key, n if key == 'physical_forward_calls'
                        else work if key == 'prefill_query_tokens' else 0)
        require(row['metrics']['generation_prompt_count'] == len(raw['observations'])
                and row['metrics']['generated_token_count'] == tokens
                and row['metrics']['length_cap_no_continuation_count'] == capped, 'GEN_FULLROW_LOGICAL_COUNTS')
        all_prompts += len(raw['observations'])
        all_tokens += tokens
    reduced = reduce_generation(rows)
    for holder in (value, receipt):
        compare_value(reduced, holder['summary'], 'GEN_REPAIR_INDEPENDENT_SUMMARY')
    compare_value(reduced, receipt['shared_summary'], 'GEN_REPAIR_SHARED_SUMMARY')
    work = receipt['work']
    require(work['new_case_observations'] == fresh and work['cached_case_observations'] == reused
            and work['generation_forwards'] == logical and work['full_prefix_token_work'] == prefix_work,
            'GEN_REPAIR_NEW_WORK_NOT_OLD_LOGICAL_COST')
    for key in physical:
        require(integer(work.get(key, 0 if cached_only else None), 'GEN_PHYSICAL_WORK_REQUIRED') == physical[key],
                'GEN_REPAIR_PHYSICAL_WORK_EXACT')
    for key, expected in (('completed_prompts', all_prompts), ('generated_tokens', all_tokens)):
        if key in work:
            require(integer(work[key], 'GEN_REPAIR_LOGICAL_WORK_INTEGER') == expected,
                    'GEN_REPAIR_LOGICAL_WORK_COUNTS')
    require(finite(work['seconds'], 'GEN_REPAIR_SECONDS') >= 0, 'GEN_REPAIR_SECONDS_NONNEGATIVE')
    return dict(rows=rows, summary=reduced, work=dict(work, work_attribution='EXACT_ORIGINAL_PROVENANCE'),
                identity=identity, identity_sha256=value['identity_sha256'], shared_runtime_identity=runtime)


def generation_endpoint(reader, receipt, selected, occurrences, expected_state, endpoint,
                        cohort, config, attempt, *, cached_only=False):
    """Independent read_observed protocol verification, including original bytes."""
    if repair_enabled(config):
        return repair_generation_endpoint(reader, receipt, selected, occurrences, expected_state, endpoint,
            cohort, config, attempt, cached_only=cached_only)
    require(type(receipt) is dict and receipt['requests'] == len(selected)
            and receipt['cohort_identity'] == digest([r['case_id'] for r in selected]),
            'GEN_COMPACT_COHORT')
    require(receipt['shared_state_identity'] == expected_state, 'GEN_SHARED_PHYSICAL_STATE')
    path = local_path(receipt['rows_path'], attempt)
    require(receipt['raw_endpoint_member']['path'] == str(path), 'GEN_ENDPOINT_MEMBER_PATH')
    value = reader.bound(receipt['raw_endpoint_member'])
    identity = value['identity']
    runtime = runtime_identity(config)
    runtime_sha = digest(runtime)
    observer = reader.json(path.parent.parent / 'observer-identity.json')
    require(observer['identity'] == runtime and observer['identity_sha256'] == runtime_sha
            and observer['raw_local_only'] is True and observer['checkpoint_saved'] is False,
            'GEN_RUNTIME_SOURCE_REFERENCE')
    require(identity['runtime'] == runtime_sha and identity['state_sha256'] == digest(expected_state)
            and identity['endpoint'] == endpoint and identity['cohort'] == cohort
            and value['identity_sha256'] == digest(identity)
            and receipt['identity'] == identity and receipt['identity_sha256'] == value['identity_sha256'],
            'GEN_ENDPOINT_IDENTITY')
    require(value['RNG_restored'] is True and value['observer_no_mutation'] is True
            and value['raw_local_only'] is True and receipt['RNG_restored'] is True
            and receipt['observer_no_mutation'] is True, 'GEN_ENDPOINT_NONMUTATION')
    rows = value['rows']
    require(identity['ordered_occurrences'] == occurrences
            and [r['occurrence'] for r in rows] == occurrences
            and all(type(r['occurrence']) is int and type(r['case_id']) is int for r in rows)
            and len(set(occurrences)) == len(occurrences)
            and [r['case_id'] for r in rows] == [r['case_id'] for r in selected]
            and identity['observation_identities'] == [r['identity_sha256'] for r in rows],
            'GEN_ORDERED_OCCURRENCE_ROWS')
    forwards = token_work = 0
    for row, record, occurrence in zip(rows, selected, occurrences):
        raw = reader.json(local_path(row['observation_path'], attempt))
        rewrite = record['requested_rewrite']
        record_identity = dict(ordered_occurrence=occurrence, case_id=record['case_id'],
            generation_prompts=record.get('generation_prompts', []), relation_id=rewrite.get('relation_id'),
            target_new_id=rewrite.get('target_new', {}).get('id'))
        require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
                and raw['identity'] == dict(runtime=runtime_sha, state_identity=expected_state,
                                           record_identity=record_identity)
                and raw['payload_sha256'] == row['payload_sha256']
                and raw['payload_sha256'] == digest({k: v for k, v in raw.items() if k != 'payload_sha256'})
                and raw['occurrence'] == occurrence and raw['case_id'] == record['case_id']
                and raw['metrics'] == row['metrics'] and raw['raw_local_only'] is True
                and raw['checkpoint_saved'] is False, 'GEN_RAW_IDENTITY_PAYLOAD')
        prompts = record_identity['generation_prompts']
        require(type(prompts) is list and all(type(prompt) is str for prompt in prompts)
                and len(raw['observations']) == len(prompts), 'GEN_ORIGINAL_PROMPT_COVERAGE')
        tokens = capped = 0
        for index, (observed, prompt) in enumerate(zip(raw['observations'], prompts)):
            n, work, cap = validate_prompt(observed, prompt, occurrence, index, runtime['model_identity'])
            forwards += n
            token_work += work
            tokens += n
            capped += cap
        require(row['metrics']['generation_prompt_count'] == len(prompts)
                and row['metrics']['generated_token_count'] == tokens
                and row['metrics']['length_cap_no_continuation_count'] == capped,
                'GEN_CASE_TOKEN_COUNTS')
    reduced = reduce_generation(rows)
    compare_value(reduced, value['summary'], 'GEN_RAW_SUMMARY')
    compare_value(reduced, receipt['summary'], 'GEN_COMPACT_SUMMARY')
    compare_value(reduced, receipt['shared_summary'], 'GEN_SHARED_SUMMARY')
    work = validate_work(receipt['work'], len(rows), forwards, token_work, cached_only=cached_only)
    return dict(rows=rows, summary=reduced, work=work, identity=identity,
                identity_sha256=value['identity_sha256'])


def generation_table(arm, label, edits, observed):
    summary = observed['summary']
    result = dict(arm=arm, endpoint=label, edits=edits,
        **{key: value for key, value in summary.items() if key != 'missing_reason_counts'})
    result.update({'missing_' + key + '_count': value
                   for key, value in summary['missing_reason_counts'].items()})
    return result


def prune_terminal(native, number):
    applied = 'terminal_prune' in native
    require(applied == (number == 20), 'PRUNE_EXACTLY_ONCE_TERMINAL')
    if applied:
        terminal = native['terminal_prune']
        require(native['prune_applied'] is True and native['explicit_repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                and terminal['prune_applied'] is True
                and terminal['explicit_repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                and terminal['native_svd_calls'] == 12, 'PRUNE_TERMINAL_SAVED_W0_BASE_FIX')
        require(terminal['status'] == 'PRUNE_TERMINAL_APPLIED' and terminal['arm'] == 'PRUNE'
                and terminal['batch'] == 20 and terminal['state_edits'] == 2000
                and terminal['native_spectral_formula_unchanged'] is True
                and terminal['final_weight_base'] == 'saved_cold_W0'
                and terminal['W0_RAM_only'] is True and terminal['checkpoint_saved'] is False
                and terminal['exact_resume'] == 'NOT_AVAILABLE'
                and [row['layer'] for row in terminal['layers']] == list(ARM_LAYERS['PRUNE']),
                'PRUNE_TERMINAL_PHYSICAL_PROVENANCE')
        for row in terminal['layers']:
            require(row['native_svd_calls'] == 2
                    and integer(row['singular_values_compressed'], 'PRUNE_COMPRESSED_COUNT')
                    <= integer(row['singular_values'], 'PRUNE_SINGULAR_COUNT'), 'PRUNE_LAYER_SVD_COUNTS')
            for key in ('max_sigma_cold', 'max_sigma_update', 'max_sigma_compressed',
                        'dense_delta_norm', 'compressed_delta_norm', 'terminal_change_norm',
                        'cold_weight_norm', 'final_weight_norm'):
                require(finite(row[key], 'PRUNE_FINITE_SPECTRAL_SCALAR') >= 0, 'PRUNE_NORM_NONNEGATIVE')
        require(terminal['compressed_singular_values'] == sum(row['singular_values_compressed']
                    for row in terminal['layers'])
                and finite(terminal['seconds'], 'PRUNE_TERMINAL_SECONDS') >= 0, 'PRUNE_SPECTRAL_TOTALS')


def native_receipt(native, arm, number, measured, prior):
    require(native['status'] == 'NATIVE_APPLY_RETURNED' and native['arm'] == arm
            and native['writer'] == writer_identity(arm) and native['batch'] == number
            and native['requests'] == 100 and native['same_model_returned'] is True
            and native['native_has_history'] == (arm in HISTORY_ARMS)
            and native['caller_history_appends'] == 0 and native['native_z_disk_cache'] is False
            and native['cache_template'] is None and native['checkpoint_saved'] is False
            and native['exact_resume'] == 'NOT_AVAILABLE', 'NATIVE_DIRECT_APPLY_RECEIPT')
    require(all(native['counts'][key] == native['delta'][key] == measured[key]
                and native['cumulative'][key] == prior[key] + measured[key] for key in COUNT_KEYS),
            'NATIVE_CUMULATIVE_COUNTS')
    trace = native['counts']['fit_trace']
    require(len(trace) == measured['native_z']
            and [row['request_index'] for row in trace] == list(range(1, len(trace) + 1)),
            'NATIVE_MEASURED_FIT_TRACE_COUNT')
    for row in trace:
        require(type(row['evaluations']) is int and 1 <= row['evaluations'] <= 25
                and type(row['Adam_updates']) is int and row['Adam_updates'] == row['evaluations'] - 1
                and row['stop'] in ('TOTAL_LOSS_BELOW_005', 'BUDGET_EXHAUSTED'), 'NATIVE_FIT_BUDGET_TRACE')
        for key in ('loss', 'nll_loss', 'kl_loss', 'weight_decay'):
            finite(row[key], 'NATIVE_FIT_TRACE_FINITE')
    evaluations = sum(row['evaluations'] for row in trace)
    updates = sum(row['Adam_updates'] for row in trace)
    if arm not in ('BASE_MEMIT', 'BASE_ALPHAEDIT'):
        require(native['counts']['fit_forwards'] == evaluations
                and native['counts']['fit_updates'] == updates, 'NATIVE_FIT_TRACE_COUNTER_MATCH')
    return dict(measured_fit_evaluations=evaluations, measured_Adam_updates=updates)


def compact_terminal(value):
    allowed = ('status', 'stage', 'error_type', 'completed_batches', 'commits', 'edits',
        'native_counts', 'rollback_verified', 'checkpoint_saved', 'exact_resume', 'source',
        'config', 'program_seconds', 'peak_host_RSS_bytes', 'peak_gpu_allocated_bytes',
        'peak_gpu_reserved_bytes', 'generation_endpoints')
    return {key: value[key] for key in allowed if key in value}


def terminal_status(terminal, commits, w0_ready, issues):
    status = terminal.get('status')
    if status in ('FAILED', 'BLOCKED'):
        return status
    if status == 'COMPLETED' and commits == 20 and w0_ready and not issues:
        return 'COMPLETED_VALIDATED_ROWS_COUNTS'
    if status == 'COMPLETED':
        return 'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE'
    return 'PARTIAL' if commits else 'NOT_STARTED_OR_STARTUP_FAILED'


def allocation_once(reader, attempt, lock, *, runner=None):
    """One bounded read of only this task's six submitted GPU parent IDs."""
    submission_path = Path(attempt) / 'submission.json'
    if not submission_path.exists():
        return dict(status='NOT_RECORDED', reason='OWN_SUBMISSION_NOT_FOUND', queries=0)
    queries = 0
    try:
        submission = reader.json(submission_path)
        require(submission['task_id'] == lock.get('task_id', TASK)
                and submission['instruction_id'] == lock.get('instruction_id', NONCE)
                and submission['source_commit'] == lock['source_commit']
                and submission['lock']['path'] == str(Path(attempt) / 'execution.lock.json'),
                'ALLOCATION_OWN_SUBMISSION_SOURCE')
        reader.bound(submission['lock'])
        jobs = {arm: submission['jobs'][arm] for arm in ARMS}
        require(len(set(jobs.values())) == 6 and all(type(job) is str
                and re.fullmatch(r'[1-9][0-9]*', job) for job in jobs.values()), 'ALLOCATION_EXACT_SIX_IDS')
        argv = ['sacct', '-n', '-P', '-X', '-j', ','.join(jobs.values()),
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES']
        queries = 1
        value = (subprocess.run if runner is None else runner)(argv, text=True,
            capture_output=True, timeout=20, check=False)
        require(value.returncode == 0 and len(value.stdout) <= 1024**2, 'ALLOCATION_BOUNDED_RESPONSE')
        records = {}
        for line in value.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split('|')
            if len(fields) == 5 and not fields[-1]:
                fields.pop()
            require(len(fields) == 4, 'ALLOCATION_COLUMN_SCHEMA')
            job, state, elapsed, tres = fields
            if job not in jobs.values():
                require(any(job.startswith(parent + '.') for parent in jobs.values()),
                        'ALLOCATION_UNREQUESTED_JOB')
                continue  # Never double count child steps, even a malformed -X response.
            require(job not in records and elapsed.isdigit(), 'ALLOCATION_EXACT_PARENT_ROW')
            resources = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
            gpu = resources.get('gres/gpu')
            if gpu is None:
                typed = [count for key, count in resources.items() if key.startswith('gres/gpu:')]
                require(len(typed) <= 1, 'ALLOCATION_TYPED_GPU_SCHEMA')
                gpu = typed[0] if typed else '0'
            require(gpu in ('0', '1'), 'ALLOCATION_ONE_GPU_PER_PARENT')
            arm = next(arm for arm, item in jobs.items() if item == job)
            records[job] = dict(arm=arm, job_id=job, scheduler_state=state,
                elapsed_seconds=int(elapsed), allocated_GPUs=int(gpu),
                allocated_GPU_seconds=int(elapsed) * int(gpu), AllocTRES=tres,
                child_steps_excluded=True, allocation_not_program_timer_sum=True)
        require(set(records) == set(jobs.values()), 'ALLOCATION_ALL_SIX_PARENTS_REQUIRED')
        ordered = [records[jobs[arm]] for arm in ARMS]
        return dict(status='RECORDED', queries=queries, scope='OWN_SIX_GPU_PARENTS_ONLY',
            records=ordered, allocated_GPU_seconds=sum(row['allocated_GPU_seconds'] for row in ordered),
            scheduler_success_not_scientific_success=True)
    except Exception as error:
        return dict(status='NOT_RECORDED', queries=queries, error_type=type(error).__name__,
            reason='OWN_ALLOCATION_UNAVAILABLE_OR_IDENTITY_MISMATCH', no_retry=True,
            scheduler_success_not_scientific_success=True)


def review_arm(reader, attempt, config, lock, arm, identities, records, progress=None,
               qualification=None):
    out, result = Path(attempt) / arm, progress if progress is not None else {}
    totals = {key: 0 for key in COUNT_KEYS}
    result.update(arm=arm, writer=writer_identity(arm), commits=0, requests=0, state_links=0,
        measured_counts=totals, W0_RPN_available=False, W0_generation_available=False,
        generation_endpoint_calls=0, generation_current_subset_reductions=0,
        generation_edit_state_case_observations=0, issues=[], **{key: [] for key in TABLE_KEYS})
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else {}
    result['terminal'] = compact_terminal(terminal)
    if not (out / 'runtime.json').exists():
        result['scientific_status'] = terminal_status(terminal, 0, False, result['issues'])
        return result
    runtime = reader.json(out / 'runtime.json')
    if repair_enabled(config):
        qualification = (qualification_evidence(reader, config, lock, attempt)
                         if qualification is None else qualification)
        if not qualification.get('actual_qualification'):
            raise RepairEvidencePending(qualification['status'])
        route_holder = repair_runtime_holder(reader, out)
        qualification_link(reader, config, route_holder, qualification, attempt)
        shared = shared_qualification_evidence(reader, config, qualification,
            route_holder['shared_qualification_receipt_member'])
        require(route_holder['shared_qualification_plan_sha256'] == shared['plan_sha256'],
                'NATIVE_RUNTIME_SHARED_PLAN_LINK')
    require(runtime['source'] == lock['source_commit'] and runtime['config'] == digest(config)
            and runtime['arm'] == arm and runtime['initial_history_zero'] is True
            and runtime['checkpoint_saved'] is False, 'NATIVE_RUNTIME_SOURCE_COLD')
    layers = {str(layer) for layer in ARM_LAYERS[arm]}
    initial = runtime['initial_state']
    require(initial['W'] == {layer: config['cold_W'][layer] for layer in layers}
            and set(initial['H']) == (layers if arm in ('CAKE', 'ALPHAEDIT_BLUE') else set()),
            'NATIVE_INITIAL_COLD_LAYOUT')
    all_ids = [record['case_id'] for record in records]
    occurrence = {record['case_id']: index for index, record in enumerate(records, 1)}
    w0 = rpn_endpoint(reader, out / 'W0', identities, all_ids, 'W0', initial, records)
    if w0 is not None:
        require({key: w0['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=2000, P=4000, N=20000), 'W0_RPN_DENOMINATORS')
        result['W0_RPN_available'] = True
        result['metrics'].extend(_metric_rows(arm, 'W0_FIRST2000', 0, w0['summary']))
        result['compute'].append(dict(arm=arm, phase='W0_RPN', seconds=w0['seconds'],
                                      reference_only=w0['reference_only']))
    w0gen = None
    if (out / 'generation-W0-reference.json').exists():
        compact = reader.json(out / 'generation-W0-reference.json')
        require(compact['model_state'] == initial, 'W0_GEN_ARM_PHYSICAL_STATE')
        w0gen = generation_endpoint(reader, compact, records, list(range(1, 2001)),
            config['generation']['W0_state_identity'], 'W0', 'FIRST2000', config, attempt,
            cached_only=arm != 'BASE_MEMIT')
        if repair_enabled(config):
            w0_progress = progress_member(reader, compact, records, attempt, ready=True,
                                         consumer=arm != 'BASE_MEMIT')
            if not w0_progress['W0_mean_ready']:
                raise RepairEvidencePending(w0_progress['status'])
            result['generation_progress'] = w0_progress
        result['W0_generation_available'] = True
        result['W0_generation_identity'] = w0gen['identity_sha256']
        result['generation'].append(generation_table(arm, 'W0_FIRST2000', 0, w0gen))
        result['compute'].append(dict(arm=arm, phase='W0_generation', **w0gen['work']))
    previous, at_write = initial, []
    for number, current, seen in batches(records):
        folder = out / f'batch-{number:02d}'
        if not (folder / 'commit.json').exists():
            require(not any((out / f'batch-{later:02d}' / 'commit.json').exists()
                            for later in range(number + 1, 22)), 'NONCONSECUTIVE_OR_BATCH21_COMMIT')
            break
        receipt = reader.json(folder / 'commit.json')
        ids = [record['case_id'] for record in current]
        seen_ids = [record['case_id'] for record in seen]
        require(receipt['task'] == config['task_id'] and receipt['arm'] == arm and receipt['batch'] == number
                and receipt['writer'] == writer_identity(arm) and receipt['source'] == lock['source_commit']
                and receipt['config'] == digest(config) and receipt['case_ids'] == ids,
                'NATIVE_COMMIT_SOURCE_REQUEST_ORDER')
        require(receipt['before'] == previous and set(receipt['after']['W']) == layers
                and set(receipt['after']['H']) == (layers if arm in HISTORY_ARMS else set()),
                'NATIVE_PHYSICAL_STATE_LINK')
        require(receipt['ledger'] == digest(seen_ids) and receipt['seen_requests'] == number * 100
                and receipt['post_scope'] == ('ALL_SEEN' if number in MILESTONES else 'CURRENT'),
                'NATIVE_CURSOR_PREFIX')
        measured = receipt['native_counts']
        require(measured == expected_counts(arm)
                and all(type(measured[key]) is int and receipt['native']['delta'][key] == measured[key]
                        for key in COUNT_KEYS), 'NATIVE_MEASURED_COUNTS')
        fitting = native_receipt(receipt['native'], arm, number, measured, totals)
        seconds = finite(receipt['seconds'], 'NATIVE_BATCH_SECONDS')
        require(seconds >= 0, 'NATIVE_BATCH_SECONDS_NONNEGATIVE')
        require(receipt['observer_no_mutation'] is True
                and receipt['generation_observer_no_mutation'] is True
                and receipt['checkpoint_saved'] is False and receipt['exact_resume'] == 'NOT_AVAILABLE',
                'NATIVE_NOCP_OBSERVER')
        if arm == 'PRUNE':
            prune_terminal(receipt['native'], number)
        pre = rpn_endpoint(reader, folder / 'pre', identities, ids, f'B{number}_PRE', previous, seen)
        selected, post_ids = (seen, seen_ids) if number in MILESTONES else (current, ids)
        post = rpn_endpoint(reader, folder / 'post', identities, post_ids, f'W{number}', receipt['after'], seen)
        require(pre is not None and post is not None, 'COMMITTED_RPN_OBSERVATIONS_MISSING')
        compare_summary(pre['summary'], receipt['pre'])
        compare_summary(post['summary'], receipt['post'])
        require({key: pre['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=100, P=200, N=1000), 'PRE_CURRENT100_DENOMINATORS')
        require({key: post['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=len(selected), P=2 * len(selected), N=10 * len(selected)),
                'POST_SELECTED_DENOMINATORS')
        current_rows = [row for row in post['rows'] if row['case_id'] in set(ids)]
        current_summary = reduce_rows(current_rows)
        compare_summary(current_summary, receipt['post_current'])
        require({key: current_summary[key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=100, P=200, N=1000), 'POST_CURRENT100_NOT_PREFIX')
        prestate = config['generation']['W0_state_identity'] if number == 1 else previous
        require(receipt['gen_before']['model_state'] == previous
                and receipt['gen_current']['model_state'] == receipt['after'], 'GEN_ARM_STATE_LINK')
        pregen = generation_endpoint(reader, receipt['gen_before'], current,
            [occurrence[case] for case in ids], prestate, f'B{number}_PRE', 'CURRENT',
            config, attempt, cached_only=number == 1)
        currentgen = generation_endpoint(reader, receipt['gen_current'], current,
            [occurrence[case] for case in ids], receipt['after'],
            f'W{number}_CURRENT' if number in MILESTONES else f'W{number}', 'CURRENT',
            config, attempt, cached_only=number in MILESTONES)
        postgen = currentgen
        if number in MILESTONES:
            require(receipt['gen_current']['derived_subset'] is True
                    and receipt['gen_prefix']['model_state'] == receipt['after'], 'GEN_MILESTONE_SUBSET')
            postgen = generation_endpoint(reader, receipt['gen_prefix'], seen,
                list(range(1, number * 100 + 1)), receipt['after'], f'W{number}', 'ALL_SEEN', config, attempt)
            subset = [row for row in postgen['rows'] if row['case_id'] in set(ids)]
            require(subset == currentgen['rows'], 'GEN_SUBSET_SAME_OBSERVATIONS_NO_REGENERATION')
        else:
            require(receipt['gen_prefix'] is None and receipt['gen_current']['derived_subset'] is False,
                    'GEN_NONMILESTONE_CURRENT_ONLY')
        require(w0gen is not None and receipt['gen_W0']['identity'] == w0gen['identity']
                and receipt['gen_W0']['reference'] == str(out / 'generation-W0-reference.json'),
                'GEN_W0_REFERENCE_LINK')
        # Append report data only after every observation of this commit verifies.
        result['metrics'].extend(_metric_rows(arm, f'B{number}_PRE', number * 100, pre['summary']))
        result['metrics'].extend(_metric_rows(arm, f'W{number}_CURRENT', number * 100, current_summary))
        result['paired'].extend(_paired_rows(arm, f'B{number}_PRE', f'W{number}_CURRENT', pre['rows'], current_rows))
        result['generation'].append(generation_table(arm, f'B{number}_PRE', number * 100, pregen))
        result['generation'].append(generation_table(arm, f'W{number}_CURRENT', number * 100, currentgen))
        at_write.extend(current_rows)
        if number in MILESTONES:
            result['metrics'].extend(_metric_rows(arm, f'W{number}_ALL_SEEN', number * 100, post['summary']))
            result['paired'].extend(_paired_rows(arm, 'AT_WRITE', f'W{number}_ALL_SEEN', at_write, post['rows']))
            if w0 is not None:
                result['paired'].extend(_paired_rows(arm, 'W0', f'W{number}_ALL_SEEN',
                    [row for row in w0['rows'] if row['case_id'] in set(seen_ids)], post['rows']))
            result['generation'].append(generation_table(arm, f'W{number}_ALL_SEEN', number * 100, postgen))
            result['compute'].append(dict(arm=arm, batch=number, phase='generation_current_subset', **currentgen['work']))
            result['generation_current_subset_reductions'] += 1
        for key in COUNT_KEYS:
            totals[key] += measured[key]
        result['counts'].append(dict(arm=arm, batch=number, **measured, **fitting,
            terminal_svd_calls=12 if arm == 'PRUNE' and number == 20 else 0))
        result['compute'].extend((dict(arm=arm, batch=number, phase='batch_inclusive', seconds=seconds,
            nested_timers_not_added_again=True),
            dict(arm=arm, batch=number, phase='pre_RPN', seconds=pre['seconds']),
            dict(arm=arm, batch=number, phase='post_RPN', seconds=post['seconds']),
            dict(arm=arm, batch=number, phase='pre_generation', **pregen['work']),
            dict(arm=arm, batch=number, phase='post_generation', **postgen['work'])))
        result.update(commits=number, requests=number * 100, state_links=max(0, number - 1))
        result['generation_endpoint_calls'] += 2
        result['generation_edit_state_case_observations'] += 100 + len(selected)
        previous = receipt['after']
    require(not (out / 'batch-21' / 'commit.json').exists(), 'NO_BATCH21')
    if terminal:
        require(terminal.get('source') == lock['source_commit'] and terminal.get('config') == digest(config)
                and terminal.get('commits') == terminal.get('completed_batches') == result['commits']
                and terminal.get('checkpoint_saved') is False
                and terminal.get('exact_resume') == 'NOT_AVAILABLE', 'NATIVE_TERMINAL_PROGRESS_SOURCE')
        claimed = terminal.get('native_counts')
        if claimed is not None:
            require(all(type(claimed.get(key)) is int and claimed[key] >= totals[key] for key in COUNT_KEYS),
                    'NATIVE_TERMINAL_COUNTER_LOWER_BOUND')
            result['failed_or_uncommitted_claimed_counts'] = {key: claimed[key] - totals[key] for key in COUNT_KEYS}
        if terminal.get('status') == 'COMPLETED':
            require(claimed is not None and all(claimed[key] == totals[key] for key in COUNT_KEYS)
                    and terminal.get('edits') == 2000 and terminal.get('generation_endpoints') == 40,
                    'NATIVE_COMPLETE_COUNTERS_ENDPOINTS')
    if not result['W0_RPN_available']:
        result['issues'].append('W0_RPN_NOT_AVAILABLE')
    if not result['W0_generation_available']:
        result['issues'].append('W0_GENERATION_NOT_AVAILABLE')
    result['scientific_status'] = terminal_status(terminal, result['commits'],
        result['W0_RPN_available'] and result['W0_generation_available'], result['issues'])
    return result


def repair_ready_evidence(reader, ready, config, qualification, compatibility, records, attempt):
    """Complete original/new W0 rows, qualification and progress are separate proofs."""
    if not qualification.get('actual_qualification'):
        raise RepairEvidencePending(qualification['status'])
    qualification_link(reader, config, ready, qualification, attempt)
    require(compatibility['status'] == 'COMPATIBILITY_ACTUAL_VERIFIED'
            and ready['compatibility_manifest']['sha256'] == compatibility['manifest_sha256'],
            'READY_ACTUAL_COMPATIBILITY_LINK')
    reader.bound(ready['compatibility_manifest'])
    require(ready['status'] == 'READY' and ready['producer_arm'] == 'BASE_MEMIT'
            and ready['identity_sha256'] == digest(ready['identity'])
            and ready['identity']['cold_state_sha256'] == digest(config['generation']['W0_state_identity'])
            and ready['identity']['ordered_occurrences'] == list(range(1, 2001))
            and ready['checkpoint_saved'] is False and ready['raw_local_only'] is True,
            'READY_REPAIR_COMPLETE_COLD_AUTHORITY')
    expected_identity = dict(model_identity=config['generation']['model_identity'],
        source_sha=config['generation']['source_sha'],
        reference_assets_sha256=config['generation']['reference_assets_sha256'],
        profile=PROFILE, eval_seed=EVAL_SEED,
        cohort_sha256=digest([dict(record, occurrence_index=ordinal) for ordinal, record in enumerate(records, 1)]),
        qualification_sha256=qualification['actual_member_sha256'],
        shared_qualification_sha256=ready['shared_qualification_receipt_member']['sha256'],
        shared_plan_sha256=ready['shared_qualification_plan_sha256'])
    require(all(ready['identity'][key] == value for key, value in expected_identity.items()),
            'READY_FULL_SOURCE_MODEL_QUALIFICATION_COHORT_IDENTITY')
    raw = reader.bound(ready['endpoint'])
    require(ready['shared_qualification_receipt_member'] == raw['qualification_receipt_member']
            and ready['compatibility_member'] == raw['compatibility_member'], 'READY_SHARED_EVIDENCE_MEMBERS')
    compact = dict(ready, requests=2000, cohort_identity=digest([record['case_id'] for record in records]),
        rows_path=ready['endpoint']['path'], raw_endpoint_member=ready['endpoint'],
        identity=raw['identity'], identity_sha256=raw['identity_sha256'],
        summary=raw['summary'], shared_summary=raw['summary'], work=ready['producer_work'],
        shared_state_identity=config['generation']['W0_state_identity'],
        RNG_restored=raw['RNG_restored'], observer_no_mutation=raw['observer_no_mutation'])
    observed = generation_endpoint(reader, compact, records, list(range(1, 2001)),
        config['generation']['W0_state_identity'], 'W0', 'FIRST2000', config, attempt)
    runtime_sha = digest(observed['shared_runtime_identity'])
    require(ready['identity']['runtime'] == runtime_sha == compatibility['new_runtime_sha256'],
            'READY_ACTUAL_SHARED_RUNTIME_PRIVATE_COMPATIBILITY')
    progress = progress_member(reader, ready, records, attempt, ready=True)
    if not progress['W0_mean_ready']:
        raise RepairEvidencePending(progress['status'])
    final, work = progress['final_counters'], observed['work']
    expected = dict(new_cases=work['new_case_observations'], reused_cases=work['cached_case_observations'],
        generated_tokens=observed['summary']['generated_token_count'])
    expected.update({key: work[key] for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens')})
    require(all(final[key] == value for key, value in expected.items()), 'READY_PROGRESS_EXACT_PRODUCTION_WORK')
    return dict(status='READY_VERIFIED', producer_arm='BASE_MEMIT',
        endpoint_identity_sha256=observed['identity_sha256'], producer_work=work,
        full_cases=2000, qualification_actual=True, original_provenance_verified=True,
        producer_progress=progress, W0_mean_ready=True)


def collect(attempt):
    attempt, reader = Path(attempt).resolve(), Reader()
    config, lock = reader.json(attempt / 'config.json'), reader.json(attempt / 'execution.lock.json')
    task_authority(config, lock)
    require(reader.files[str(attempt / 'config.json')]['sha256'] == lock['config_sha256']
            and type(lock['source_commit']) is str and len(lock['source_commit']) == 40,
            'COLLECT_TASK_CONFIG_SOURCE')
    generation = config['generation']
    require(generation['common_source_status'] == 'READY_BOUND'
            and generation['reference_status'] == 'READY_VERIFIED'
            and generation['schema'] == SCHEMA and generation['profile'] == PROFILE
            and generation['eval_seed'] == EVAL_SEED and generation['W0_owner'] == 'BASE_MEMIT'
            and config['noCP'] is True and config['z_disk_cache'] is False,
            'COLLECT_GENERATION_SOURCE_REFERENCE_PROFILE')
    require(lock.get('shared_generation_source') == generation['source_sha']
            and lock.get('shared_generation_tree') == generation['package_tree']
            and lock.get('reference_identity') == generation['reference_assets_sha256'],
            'COLLECT_LOCK_SHARED_SOURCE_REFERENCE')
    # Small sealed source/reference manifests, never model/stat/projector tensors.
    for row in generation.get('shared_source_members', []):
        data = reader.bytes(row['path'])
        require(len(data) == row['bytes'] and reader.files[str(Path(row['path']))]['sha256'] == row['sha256'],
                'COLLECT_SHARED_SOURCE_BYTES')
    identities = reader.bound(config['observer_identity'])
    stream = next(row for row in config['assets'] if row['path'] == config['stream'])
    records = reader.bound(stream)[:2000]
    require(len(records) == 2000 and len({r['case_id'] for r in records}) == 2000
            and [r['case_id'] for r in records] == [case for pack in config['packs'] for case in pack['ids']],
            'COLLECT_FIRST2000_ORDER')
    list(batches(records))
    out = attempt / 'collector'
    require(not out.exists(), 'COLLECT_CREATE_ONCE_OUTPUT')
    out.mkdir()
    qualification = None
    compatibility = None
    if repair_enabled(config):
        try:
            qualification = qualification_evidence(reader, config, lock, attempt, records=records)
        except Exception as error:
            qualification = dict(status='BLOCKED_QUALIFICATION_EVIDENCE', actual_qualification=False,
                                 error_type=type(error).__name__)
        if qualification.get('actual_qualification'):
            try:
                compatibility_bound = generation['repair'].get('compatibility_manifest')
                runtime_holder = repair_runtime_holder(reader, attempt / 'BASE_MEMIT')
                if compatibility_bound is None and runtime_holder is not None:
                    compatibility_bound = runtime_holder.get('compatibility_manifest')
                compatibility = compatibility_evidence(reader, config, qualification, records,
                                                       compatibility_member=compatibility_bound)
            except Exception as error:
                compatibility = dict(status='BLOCKED_COMPATIBILITY_EVIDENCE', full_W0_READY=False,
                                     error_type=type(error).__name__)
        else:
            compatibility = dict(status='BLOCKED_ACTUAL_QUALIFICATION_REQUIRED', full_W0_READY=False)
    reviews = []
    for arm in ARMS:
        progress = {}
        try:
            reviewed = review_arm(reader, attempt, config, lock, arm, identities, records, progress,
                                  qualification=qualification)
            if repair_enabled(config) and not qualification.get('actual_qualification'):
                reviewed['technical_readiness'] = qualification['status']
                if reviewed['scientific_status'] not in ('FAILED', 'BLOCKED'):
                    reviewed['scientific_status'] = qualification['status']
            if repair_enabled(config) and compatibility['status'] != 'COMPATIBILITY_ACTUAL_VERIFIED':
                reviewed['compatibility_readiness'] = compatibility['status']
                if reviewed['scientific_status'] == 'COMPLETED_VALIDATED_ROWS_COUNTS':
                    reviewed['scientific_status'] = compatibility['status']
            reviews.append(reviewed)
        except RepairEvidencePending as error:
            progress.update(arm=arm, scientific_status=str(error),
                valid_prefix_preserved=True, original_raw_preserved=True)
            progress.setdefault('issues', []).append(str(error))
            reviews.append(progress)
        except Exception as error:
            # Do not publish exception text containing a raw case ID or prompt.
            progress.update(arm=arm, scientific_status='TECHNICAL_BLOCKED_REDUCER',
                error_type=type(error).__name__, valid_prefix_preserved=True, original_raw_preserved=True)
            progress.setdefault('issues', []).append('STORED_DATA_INCONSISTENCY')
            reviews.append(progress)
    ready_path = local_path(generation['W0_cache'], attempt) / 'READY.json'
    w0_ready = None
    if ready_path.exists():
        try:
            ready = reader.json(ready_path)
            if repair_enabled(config):
                w0_ready = repair_ready_evidence(reader, ready, config, qualification, compatibility,
                                                records, attempt)
                endpoint = reader.bound(ready['endpoint'])
            else:
                require(ready['status'] == 'READY' and ready['producer_arm'] == 'BASE_MEMIT'
                    and ready['identity_sha256'] == digest(ready['identity'])
                    and ready['identity']['runtime'] == digest(runtime_identity(config))
                    and ready['identity']['cold_state_sha256'] == digest(generation['W0_state_identity'])
                    and ready['identity']['ordered_occurrences'] == list(range(1, 2001)),
                    'COLLECT_SINGLE_COLD_W0_READY')
                endpoint = reader.bound(ready['endpoint'])
                require(endpoint['identity_sha256'] == digest(endpoint['identity'])
                    and endpoint['identity']['runtime'] == digest(runtime_identity(config))
                    and endpoint['identity']['state_sha256'] == digest(generation['W0_state_identity'])
                    and endpoint['identity']['endpoint'] == 'W0'
                    and endpoint['identity']['cohort'] == 'FIRST2000'
                    and endpoint['identity']['ordered_occurrences'] == list(range(1, 2001)),
                    'COLLECT_READY_ENDPOINT_COLD_IDENTITY')
                producer_work = ready['producer_work']
                for key in ('new_case_observations', 'cached_case_observations',
                        'generation_forwards', 'full_prefix_token_work'):
                    integer(producer_work[key], 'READY_WORK_INTEGER')
                require(producer_work['new_case_observations'] + producer_work['cached_case_observations'] == 2000
                    and finite(producer_work['seconds'], 'READY_WORK_SECONDS') >= 0,
                    'READY_WORK_COLD_COHORT')
                w0_ready = dict(status='READY_VERIFIED', producer_arm='BASE_MEMIT',
                endpoint_identity_sha256=endpoint['identity_sha256'], producer_work=ready['producer_work'])
            for review in reviews:
                if review.get('W0_generation_available'):
                    require(review['W0_generation_identity'] == endpoint['identity_sha256'], 'COLLECT_SAME_W0_ALL_ARMS')
        except RepairEvidencePending as error:
            w0_ready = dict(status=str(error), W0_mean_ready=False)
        except Exception as error:
            w0_ready = dict(status='TECHNICAL_BLOCKED_READY', error_type=type(error).__name__)
    for key in TABLE_KEYS:
        _csv(out / (key + '.csv'), [row for review in reviews for row in review.get(key, [])])
    compact = [{key: value for key, value in review.items() if key not in TABLE_KEYS} for review in reviews]
    actual = {key: sum(review.get('measured_counts', {}).get(key, 0) for review in reviews) for key in COUNT_KEYS}
    plan = planned_counts()
    expected = {key: sum(values[key] for values in plan['native_per_arm'].values()) for key in COUNT_KEYS}
    complete = all(review['scientific_status'] == 'COMPLETED_VALIDATED_ROWS_COUNTS' for review in reviews)
    complete = complete and w0_ready is not None and w0_ready['status'] == 'READY_VERIFIED' and actual == expected
    if repair_enabled(config):
        complete = complete and qualification.get('actual_qualification') is True \
            and compatibility['status'] == 'COMPATIBILITY_ACTUAL_VERIFIED'
    accounting = allocation_once(reader, attempt, lock)
    write(out / 'allocation.json', accounting)
    write(out / 'counts-plan-actual.json', dict(plan=plan, expected_native=expected,
        verified_committed_native=actual, native_z_unit='measured_native_target_fit_calls_not_Adam_updates',
        verified_committed_edit_generation_case_observations=sum(row.get('new_case_observations', 0)
            for review in reviews for row in review.get('compute', [])
            if row['phase'] in ('pre_generation', 'post_generation')),
        recorded_cold_W0_producer_work=(w0_ready['producer_work']
            if w0_ready and w0_ready['status'] == 'READY_VERIFIED' else None),
        generation_qualification=qualification,
        generation_compatibility=compatibility,
        failed_or_uncommitted_generation_work='LOCAL_RECEIPTS_PRESERVED_NOT_IN_COMMITTED_TOTAL',
        plan_is_not_actual=not complete, scientific_complete=complete))
    lines = ['# GPT-J 6개 native arm 생성·R/P/N 저장 결과 CPU 검산', '',
        '저장된 raw occurrence·SHA·분모·상태 연결·계수를 검산했다. 모델/토크나이저/fit 호출은 없다.', '',
        '| Arm | 과학 상태 | 검산 batch | 요청 | native fit | solves | H append |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for review in compact:
        count = review.get('measured_counts', {})
        lines.append(f"| {review['arm']} | {review['scientific_status']} | {review.get('commits', 0)} | "
                     f"{review.get('requests', 0)} | {count.get('native_z', 'NA')} | "
                     f"{count.get('solves', 'NA')} | {count.get('history_appends', 'NA')} |")
    lines.extend(['', f"계획: native fit {expected['native_z']}, solves {expected['solves']}, H append {expected['history_appends']}.",
        f"검산 완료 commit 실제: native fit {actual['native_z']}, solves {actual['solves']}, H append {actual['history_appends']}.",
        'native fit은 Adam update가 아니다. 실패/미완료 비용 및 미측정값은 0으로 대체하지 않는다.',
        'Fluency는 bits, consistency는 native cosine이다. 유효 분모가 0이면 평균을 생략한다.',
        'W0 생성은 BASE_MEMIT 한 번만 새로 관측하고, B1 pre와 milestone current는 동일 raw의 CPU subset이다.',
        'PRUNE W5/10/15는 dense, W20만 saved W0 terminal basefix 후 관측이다.',
        'NoCP / exact_resume=NOT_AVAILABLE. collector 완료는 모든 arm 과학 완료와 별개다.',
        'Raw 생성문·토큰·case ID는 local 원본에만 남긴다. Nested timer와 allocation은 합산하지 않는다.', '',
        'Allocation은 자신의 six parent ID만 sacct 1회로 조회하며, 미확보 시 NOT_RECORDED로 남긴다.', '',
        '[R/P/N](metrics.csv) · [paired](paired.csv) · [생성 지표](generation.csv) · [native 계수](counts.csv) · [계산비용](compute.csv)', ''])
    _atomic_text(out / 'report-ko.md', '\n'.join(lines))
    write(out / 'review.json', dict(task=config['task_id'], instruction_id=config['instruction_id'], source=lock['source_commit'],
        generation_source=generation['source_sha'], reference_identity=generation['reference_assets_sha256'],
        reviews=compact, W0_generation_READY=w0_ready, generation_qualification=qualification,
        generation_compatibility=compatibility,
        scientific_complete=complete,
        new_model_forwards=0, raw_copied=False, raw_publication=False))
    # This potentially large input listing is local only; the compact manifest
    # binds its bytes rather than embedding paths/row identities in the report.
    write(out / 'raw-input-manifest.json', dict(local_only=True, inputs=list(reader.files.values())))
    write(out / 'manifest.json', dict(task=config['task_id'], source=lock['source_commit'],
        input_files=len(reader.files), input_manifest=member(out / 'raw-input-manifest.json'),
        outputs=[member(path) for path in sorted(out.iterdir())
                 if path.is_file() and path.name != 'raw-input-manifest.json'],
        raw_copied=False, raw_input_manifest_local_only=True))
    write(out / 'terminal.json', dict(status='COMPLETED', scientific_complete=complete,
        report=member(out / 'report-ko.md'), source=lock['source_commit'], new_model_forwards=0))
    return dict(status='CPU_REPORT_WRITTEN', scientific_complete=complete, output=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    print(json.dumps(collect(parser.parse_args().attempt)))
