"""Read-only, source-backed completed old W0 cases; never relabel their runtime.

Inventories/manifests contain per-occurrence metadata and belong in ignored
local storage. This module does not load a model/tokenizer/assets, run a forward,
disable shared runtime guards, or assert GPU qualification from CPU fixtures.
"""
import copy
import ast
import hashlib
import json
import math
from pathlib import Path

from project.run_scripts.experiment_generation_eval.common import (
    EVAL_SEED, PROFILE, SCHEMA, case_seed, digest, immutable_write,
)
from project.run_scripts.experiment_generation_eval.generator import normalize_decode
from project.run_scripts.experiment_generation_eval.metrics import MISSING_REASONS


INVENTORY_SCHEMA = 'gptj-generation-old-w0-inventory-v1'
COMPATIBILITY_SCHEMA = 'gptj-generation-w0-compatibility-v1'
OLD_SOURCE = '83535c6a47c552cc4e5c6385f3a587d752820150'
OLD_ROUTE = 'UNPADDED_FULL_PREFIX_NO_CACHE'
OLD_TASK_SOURCE = '503081fa9bc6efc4dbdd461324fba522b8f36e8b'
QUALIFICATION_SCHEMA = 'gptj-generation-cache-qualification-v1'
COLD_GUARD_SCHEMA = 'gptj-generation-old-cold-observation-guard-v1'
REUSE_BINDING_SCHEMA = 'gptj-generation-old-w0-reuse-binding-v1'


class ReuseError(RuntimeError):
    """Closed error code only; never echoes raw observations or SDK errors."""


def require(condition, code):
    if not condition:
        raise ReuseError(code)


def read_member(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'REUSE_MEMBER_REGULAR_FILE')
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_ino, before.st_mtime_ns) ==
            (after.st_size, after.st_ino, after.st_mtime_ns), 'REUSE_MEMBER_CHANGED')
    member = dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if expected is not None:
        require(member['bytes'] == expected['bytes'] and member['sha256'] == expected['sha256'],
                'REUSE_MEMBER_IDENTITY')
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ReuseError('REUSE_JSON_INCOMPLETE_OR_INVALID') from error
    return value, member


def _verify_bytes(row):
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink(), 'REUSE_SOURCE_MEMBER_REGULAR_FILE')
    data = path.read_bytes()
    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
            'REUSE_SOURCE_MEMBER_IDENTITY')


def _sealed_asset(row):
    """Preserve prior exact full SHA and verify its current sealed stat; no GB rehash."""
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink(), 'REUSE_ASSET_REGULAR_FILE')
    stat = path.stat()
    require((stat.st_size, stat.st_ino, stat.st_mtime_ns) ==
            (row['bytes'], row['inode'], row['mtime_ns'])
            and type(row['sha256']) is str and len(row['sha256']) == 64,
            'REUSE_SEALED_ASSET_IDENTITY')


def semantic_identity(config):
    gen = config['generation']
    model = digest(dict(model=config['model'], revision=config['model_revision'],
        model_assets=config['model_assets'], runtime=config['runtime'],
        scorer=config['observer_identity'], seed=config['seed'],
        precision='FP32/eager/TF32off/autocastoff'))
    require(model == gen['model_identity'], 'REUSE_MODEL_TOKENIZER_RUNTIME_BINDING')
    require(gen['schema'] == SCHEMA and gen['profile'] == PROFILE and gen['eval_seed'] == EVAL_SEED,
            'REUSE_SEMANTIC_PROFILE')
    return dict(model_identity=model, model_assets_sha256=digest(config['model_assets']),
        model_revision=config['model_revision'], tokenizer_binding_sha256=digest(config['model_assets']),
        scientific_runtime_sha256=digest(config['runtime']),
        scoring_versions=copy.deepcopy(gen['scoring_versions']),
        scorer_identity_sha256=digest(config['observer_identity']),
        reference_assets_sha256=gen['reference_assets_sha256'],
        native_inputs_sha256=digest({arm: value['native'] for arm, value in config['arm_configs'].items()}),
        cold_state_identity=copy.deepcopy(gen['W0_state_identity']),
        profile=PROFILE, eval_seed=EVAL_SEED, schema=SCHEMA)


def record_identity(record, occurrence):
    rewrite = record['requested_rewrite']
    prompts = record.get('generation_prompts', [])
    require(type(prompts) is list and all(type(prompt) is str for prompt in prompts), 'REUSE_PROMPTS_SCHEMA')
    for field in ('ordered_occurrence', 'occurrence_index', 'occurrence', 'ordinal'):
        if field in record:
            require(type(record[field]) is int and record[field] == occurrence, 'REUSE_OCCURRENCE_ORDER')
    return dict(ordered_occurrence=occurrence, case_id=record['case_id'],
        generation_prompts=prompts, relation_id=rewrite.get('relation_id'),
        target_new_id=rewrite['target_new'].get('id'))


def _tokens(values):
    return type(values) is list and all(type(token) is int and 0 <= token < 50400 for token in values)


def validate_case(raw, member, *, runtime_identity, state_identity, record, occurrence, tokenizer=None):
    """Verify an old complete case against its original exact identity and source."""
    runtime_sha = digest(runtime_identity)
    identity = dict(runtime=runtime_sha, state_identity=state_identity,
                    record_identity=record_identity(record, occurrence))
    require(raw['identity'] == identity and raw['identity_sha256'] == digest(identity)
            and Path(member['path']).name == raw['identity_sha256'] + '.json'
            and raw['payload_sha256'] == digest({key: value for key, value in raw.items() if key != 'payload_sha256'})
            and raw['occurrence'] == occurrence and raw['case_id'] == record['case_id']
            and raw['raw_local_only'] is True and raw['checkpoint_saved'] is False,
            'REUSE_CASE_OR_PAYLOAD_IDENTITY')
    observations = raw['observations']
    prompts = identity['record_identity']['generation_prompts']
    require(type(observations) is list and len(observations) == len(prompts), 'REUSE_CASE_INCOMPLETE')
    for index, (prompt, observation) in enumerate(zip(prompts, observations)):
        require(observation['profile'] == PROFILE and observation['route'] == OLD_ROUTE
                and observation['occurrence'] == occurrence and observation['prompt_index'] == index
                and observation['prompt'] == prompt and observation['RNG_restored'] is True
                and observation['seed'] == case_seed(runtime_identity['model_identity'], occurrence, index)
                and observation['sampling'] == dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100),
                'REUSE_PROMPT_SOURCE_SEED_PROFILE')
        inputs, continuation, full = (observation[key] for key in
            ('input_token_ids', 'continuation_token_ids', 'full_token_ids'))
        require(_tokens(inputs) and bool(inputs) and _tokens(continuation) and _tokens(full)
                and full == inputs + continuation
                and observation['input_token_count'] == len(inputs)
                and observation['continuation_token_count'] == len(continuation), 'REUSE_PROMPT_TOKEN_IDENTITY')
        eos = observation['eos_ids']
        bindings = observation['eos_binding']
        require(_tokens(eos) and type(bindings) is dict and all(_tokens(value) for value in bindings.values())
                and eos == sorted({token for value in bindings.values() for token in value}), 'REUSE_NATIVE_EOS_BINDING')
        stop = observation['stop_reason']
        complete = ((stop == 'length_cap_no_continuation' and len(inputs) >= 100 and not continuation)
            or (stop == 'length_cap' and len(inputs) < 100 and len(full) == 100 and not set(continuation) & set(eos))
            or (stop == 'eos' and bool(continuation) and len(full) <= 100 and continuation[-1] in eos
                and not set(continuation[:-1]) & set(eos)))
        require(complete and observation['model_forwards'] == len(continuation)
                and observation['full_prefix_token_work'] ==
                    len(continuation) * len(inputs) + len(continuation) * (len(continuation) - 1) // 2
                and type(observation['text']) is str, 'REUSE_PROMPT_INCOMPLETE_OR_WORK')
        if tokenizer is not None:
            encoded = tokenizer(prompt, return_tensors='pt', padding=False, truncation=False)
            require(encoded['input_ids'][0].tolist() == inputs
                    and normalize_decode(tokenizer.decode(full, skip_special_tokens=True)) == observation['text'],
                    'REUSE_TOKENIZER_INPUT_DECODE_IDENTITY')
    metrics = raw['metrics']
    require(type(metrics['fluency_valid']) is bool and type(metrics['consistency_valid']) is bool
            and type(metrics['reasons']) is list and set(metrics['reasons']) <= set(MISSING_REASONS)
            and metrics['generation_prompt_count'] == len(observations)
            and metrics['generated_token_count'] == sum(value['continuation_token_count'] for value in observations)
            and metrics['length_cap_no_continuation_count'] == sum(
                value['stop_reason'] == 'length_cap_no_continuation' for value in observations), 'REUSE_CASE_METRIC_COUNTS')
    for name, validity in (('ngram_entropy', 'fluency_valid'), ('reference_score', 'consistency_valid')):
        value = metrics[name]
        # Preserve native metric values, including finite cosine roundoff > 1.
        # Reuse is an identity/completion gate, not a new scientific-quality gate.
        require((type(value) in (int, float) and math.isfinite(value)) if metrics[validity] else value is None,
                'REUSE_CASE_METRIC_VALIDITY')
    row = dict(occurrence=occurrence, case_id=raw['case_id'], identity_sha256=raw['identity_sha256'],
        observation_path=member['path'], payload_sha256=raw['payload_sha256'], metrics=copy.deepcopy(metrics))
    provenance = dict(raw=copy.deepcopy(member), original_identity_sha256=raw['identity_sha256'],
        original_payload_sha256=raw['payload_sha256'], original_runtime_sha256=runtime_sha,
        original_source_sha=runtime_identity['generation_source_sha'], original_route=OLD_ROUTE,
        record_identity_sha256=digest(identity['record_identity']),
        prompt_seed_stream_sha256=digest([value['seed'] for value in observations]),
        input_token_bindings_sha256=digest([value['input_token_ids'] for value in observations]),
        prompt_count=len(observations), tokenizer_reencoded=tokenizer is not None)
    return dict(row=row, provenance=provenance)


def build_inventory(old_attempt, target_config, records, *, out=None, tokenizer=None):
    """Scan only exact planned keys. Absent/incomplete/unknown cases are not complete."""
    old_attempt = Path(old_attempt)
    old, config_member = read_member(old_attempt / 'config.json')
    lock, lock_member = read_member(old_attempt / 'execution.lock.json')
    require(lock['config_sha256'] == config_member['sha256'] and lock['source_commit'] == OLD_TASK_SOURCE
            and lock['shared_generation_source'] == old['generation']['source_sha'] == OLD_SOURCE,
            'REUSE_OLD_SOURCE_CONFIG_LOCK')
    for source in lock['source_members'] + lock['runtime_sources']:
        _verify_bytes(source)
    for asset in old['model_assets']:
        _sealed_asset(asset)
    semantic = semantic_identity(old)
    require(semantic_identity(target_config) == semantic, 'REUSE_TARGET_SEMANTIC_INPUT_MISMATCH')
    observer, observer_member = read_member(old_attempt / 'BASE_MEMIT' / 'generation-raw' / 'observer-identity.json')
    runtime = observer['identity']
    require(observer['identity_sha256'] == digest(runtime)
            and runtime == dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
                model_identity=semantic['model_identity'], generation_source_sha=OLD_SOURCE,
                reference_assets_sha256=semantic['reference_assets_sha256'], route=OLD_ROUTE),
            'REUSE_ORIGINAL_RUNTIME_IDENTITY')
    manifest, _ = read_member(old['generation']['generation_assets']['path'], old['generation']['generation_assets'])
    require(manifest['identity_sha256'] == semantic['reference_assets_sha256']
            and manifest['versions'] == semantic['scoring_versions'], 'REUSE_SCORING_REFERENCE_RUNTIME')
    records = list(records)
    require(len(records) == 2000 and len({record['case_id'] for record in records}) == 2000,
            'REUSE_EXACT_FIRST2000_OCCURRENCES')
    entries = []
    directory = old_attempt / 'BASE_MEMIT' / 'generation-raw' / 'observations'
    for ordinal, record in enumerate(records, 1):
        identity = dict(runtime=digest(runtime), state_identity=semantic['cold_state_identity'],
                        record_identity=record_identity(record, ordinal))
        path = directory / (digest(identity) + '.json')
        entry = dict(occurrence=ordinal, expected_identity_sha256=digest(identity))
        if not path.exists():
            entry.update(status='NOT_REUSABLE', reason='MISSING_COMPLETED_CASE')
        else:
            try:
                raw, raw_member = read_member(path)
                checked = validate_case(raw, raw_member, runtime_identity=runtime,
                    state_identity=semantic['cold_state_identity'], record=record, occurrence=ordinal, tokenizer=tokenizer)
                entry.update(status='REUSABLE_COMPLETE_CASE', **checked)
            except (ReuseError, KeyError, TypeError, ValueError) as error:
                reason = str(error) if isinstance(error, ReuseError) else 'INCOMPLETE_OR_UNKNOWN_CASE_SCHEMA'
                entry.update(status='NOT_REUSABLE', reason=reason)
        entries.append(entry)
    identity = dict(schema=INVENTORY_SCHEMA, old_config=config_member, old_lock=lock_member,
        old_observer=observer_member, old_runtime_identity=runtime, old_runtime_sha256=digest(runtime),
        semantic_inputs=semantic, source_members_sha256=digest(lock['source_members']),
        runtime_source_members_sha256=digest(lock['runtime_sources']),
        ordered_records_sha256=digest([record_identity(record, index) for index, record in enumerate(records, 1)]))
    result = dict(schema=INVENTORY_SCHEMA, identity=identity, identity_sha256=digest(identity), entries=entries,
        planned_cases=2000, reusable_cases=sum(entry['status'] == 'REUSABLE_COMPLETE_CASE' for entry in entries),
        unknown_cases=sum(entry['status'] != 'REUSABLE_COMPLETE_CASE' for entry in entries),
        raw_local_only=True, model_loads=0, generation_forwards=0, actual_GPU=False,
        new_source_relabel=False, scoring_recomputed=False)
    result['payload_sha256'] = digest(result)
    if out is not None:
        immutable_write(out, result)
    return result


def read_reusable_case(inventory, occurrence, record, *, tokenizer=None):
    validate_inventory(inventory)
    require(type(occurrence) is int and 1 <= occurrence <= 2000, 'REUSE_OCCURRENCE_RANGE')
    entry = inventory['entries'][occurrence - 1]
    require(entry['occurrence'] == occurrence and entry['status'] == 'REUSABLE_COMPLETE_CASE', 'REUSE_CASE_NOT_COMPLETE')
    raw, member = read_member(entry['provenance']['raw']['path'], entry['provenance']['raw'])
    checked = validate_case(raw, member, runtime_identity=inventory['identity']['old_runtime_identity'],
        state_identity=inventory['identity']['semantic_inputs']['cold_state_identity'],
        record=record, occurrence=occurrence, tokenizer=tokenizer)
    require(checked['row'] == entry['row'], 'REUSE_STORED_ROW_IDENTITY')
    return dict(raw=raw, row=checked['row'], provenance=copy.deepcopy(entry['provenance']))


def validate_inventory(inventory):
    """Pure consumer gate; validates the sealed complete/unknown count plan."""
    require(inventory['schema'] == INVENTORY_SCHEMA and inventory['identity_sha256'] == digest(inventory['identity'])
            and inventory['payload_sha256'] == digest({key: value for key, value in inventory.items() if key != 'payload_sha256'}),
            'REUSE_INVENTORY_IDENTITY')
    entries = inventory['entries']
    require(type(entries) is list and len(entries) == inventory['planned_cases'] == 2000
            and [entry['occurrence'] for entry in entries] == list(range(1, 2001))
            and all(entry['status'] in ('REUSABLE_COMPLETE_CASE', 'NOT_REUSABLE') for entry in entries)
            and inventory['reusable_cases'] == sum(entry['status'] == 'REUSABLE_COMPLETE_CASE' for entry in entries)
            and inventory['unknown_cases'] == 2000 - inventory['reusable_cases']
            and inventory['raw_local_only'] is True and inventory['actual_GPU'] is False
            and inventory['new_source_relabel'] is False, 'REUSE_INVENTORY_COHORT_COUNTS')
    return inventory


def _frozen_source(lock, old_attempt, filename):
    path = old_attempt / 'source' / 'project/run_scripts/gptj_native_baselines' / filename
    bound = next((row for row in lock['source_members'] if Path(row['path']) == path), None)
    require(bound is not None, 'REUSE_COLD_SOURCE_NOT_LOCKED')
    _verify_bytes(bound)
    data = path.read_bytes()
    return ast.parse(data), dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _call_name(call):
    if isinstance(call, ast.Call):
        return ast.unparse(call.func)
    return None


def _cold_control_flow(runner, bridge):
    """Check the archived entry path, not a measured terminal/RAM counter."""
    functions = {node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)}
    chain = functions['execute_chain'].body
    batch_index = next(index for index, node in enumerate(chain) if isinstance(node, ast.For)
                       and _call_name(node.iter) == 'ops.batches')
    pre = chain[:batch_index]
    rpn_index = next(index for index, node in enumerate(pre)
                     if any(_call_name(call) == 'ops.install_W0' for call in ast.walk(node)))
    generation_index = next(index for index, node in enumerate(pre)
        if any(isinstance(value, ast.Attribute) and ast.unparse(value) == 'generation.load_W0'
               for value in ast.walk(node)))
    require(rpn_index < generation_index < batch_index
            and not any(_call_name(call) in ('engine.apply', 'engine.terminal_prune', 'ops.transaction')
                        for node in pre for call in ast.walk(node))
            and any(_call_name(call) == 'engine.apply' for call in ast.walk(chain[batch_index])),
            'REUSE_COLD_GENERATION_PRECEDES_FIRST_NATIVE_APPLY')
    startup = next(node.body for node in functions['main'].body if isinstance(node, ast.Try))
    initial = next(index for index, node in enumerate(startup)
                   if any(_call_name(call) == 'initial_history' for call in ast.walk(node)))
    runtime = next(index for index, node in enumerate(startup)
        if any(isinstance(call, ast.Call) and _call_name(call) == 'write'
               and any(isinstance(value, ast.Constant) and value.value == 'runtime.json'
                       for value in ast.walk(call.args[0])) for call in ast.walk(node)))
    execute = next(index for index, node in enumerate(startup)
                   if any(_call_name(call) == 'execute_chain' for call in ast.walk(node)))
    require(initial < runtime < execute, 'REUSE_COLD_RUNTIME_BEFORE_EXECUTE_CHAIN')
    klass = next(node for node in bridge.body if isinstance(node, ast.ClassDef) and node.name == 'GenerationObserver')
    load = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'load_W0')
    require(any(_call_name(call) == 'self._cold_weights' for call in ast.walk(load.body[0]))
            and any(_call_name(call) == 'self.shared.observe' for call in ast.walk(load)),
            'REUSE_COLD_BRIDGE_WEIGHT_GUARD')
    return ['initial_history_startup_guard', 'original_runtime_initial_state',
            'cold_W0_RPN_receipt', 'cold_weight_guard_before_W0_observe',
            'W0_generation_before_first_native_apply']


def build_old_w0_reuse_binding(inventory_member, *, out_directory):
    """Create immutable source-backed partial-phase evidence, never a final guard."""
    inventory, inventory_binding = read_member(inventory_member['path'], inventory_member)
    validate_inventory(inventory)
    evidence = inventory['identity']
    old, config_member = read_member(evidence['old_config']['path'], evidence['old_config'])
    lock, lock_member = read_member(evidence['old_lock']['path'], evidence['old_lock'])
    old_attempt = Path(config_member['path']).parent
    require(lock['source_commit'] == OLD_TASK_SOURCE and lock['config_sha256'] == config_member['sha256']
            and semantic_identity(old) == evidence['semantic_inputs'], 'REUSE_COLD_ORIGINAL_CONFIG_LOCK')
    for source in lock['source_members'] + lock['runtime_sources']:
        _verify_bytes(source)
    runner, runner_member = _frozen_source(lock, old_attempt, 'generation_run.py')
    bridge, bridge_member = _frozen_source(lock, old_attempt, 'generation_bridge.py')
    try:
        flow = _cold_control_flow(runner, bridge)
    except (KeyError, StopIteration, TypeError, IndexError) as error:
        raise ReuseError('REUSE_COLD_SOURCE_CONTROL_FLOW_UNKNOWN') from error
    arm = old_attempt / 'BASE_MEMIT'
    runtime, runtime_member = read_member(arm / 'runtime.json')
    summary, summary_member = read_member(arm / 'W0/summary.json')
    rpn, rpn_member = read_member(arm / 'W0/reuse.json')
    state = copy.deepcopy(evidence['semantic_inputs']['cold_state_identity'])
    weights = state.get('selected_physical_W', state.get('W'))
    require(weights == old['cold_W'] and runtime['initial_state'] == summary['state'] == dict(W=weights, H={})
            and runtime['source'] == lock['source_commit'] and runtime['config'] == digest(old)
            and runtime['model'] == old['model'] and runtime['torch'] == old['runtime']['torch']
            and runtime['transformers'] == old['runtime']['transformers']
            and runtime['arm'] == 'BASE_MEMIT' and runtime['checkpoint_saved'] is False
            and runtime['initial_history_zero'] is True and runtime['FP32'] is True
            and runtime['eager'] is True and runtime['TF32'] is False and runtime['autocast'] is False,
            'REUSE_COLD_RUNTIME_RPN_STATE')
    require(summary['endpoint'] == 'W0' and summary['requests'] == 2000
            and summary['no_mutation'] is True and summary['optimizer_feedback'] is False
            and summary['scalar_bridge_only'] is True and rpn['scalar_bridge_only'] is True
            and rpn['history_or_editor_resume'] is False and rpn['actual_cold_weights'] == weights,
            'REUSE_COLD_RPN_OBSERVATION_ONLY')
    original_rpn, rpn_result_member = read_member(rpn['source_result']['path'], rpn['source_result'])
    require(original_rpn['new_actual_evaluation'] is True and original_rpn['no_mutation'] is True
            and all(original_rpn[field] == 0 for field in ('edit_calls', 'fits', 'solves', 'history_appends')),
            'REUSE_COLD_RPN_REFERENCE_NO_EDIT')
    observer, observer_member = read_member(evidence['old_observer']['path'], evidence['old_observer'])
    require(observer['identity'] == evidence['old_runtime_identity']
            and observer['identity_sha256'] == evidence['old_runtime_sha256'], 'REUSE_COLD_OBSERVER_RUNTIME')
    native_commits = list(arm.glob('batch-*/commit.json'))
    require(not native_commits and not (arm / 'generation-W0-reference.json').exists()
            and not (arm / 'generation-W0/receipt.json').exists()
            and not list((arm / 'generation-raw/endpoints').glob('*.json')),
            'REUSE_COLD_PARTIAL_PHASE_NOT_ESTABLISHED')
    guard = dict(schema=COLD_GUARD_SCHEMA, source_commit=OLD_TASK_SOURCE, phase='W0_generation',
        commits=0, history_appends=0, model_W=weights, old_generation_runtime=evidence['old_runtime_sha256'],
        whole_endpoint_guard_recorded=False, proof_basis='FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN',
        partial_rows_authorized=True, original_state_identity=state,
        phase_counts_basis='SOURCE_CONTROL_FLOW_BEFORE_FIRST_NATIVE_APPLY_NOT_FINAL_RAM_COUNTER',
        observed_native_commit_files=0, final_RAM_history='NOT_RECORDED', final_RAM_RNG_guard='NOT_RECORDED',
        measured_final_history_zero=False, old_whole_W0_PASS=False,
        source_control_flow=flow, inventory_member=inventory_binding,
        completed_case_count=inventory['reusable_cases'], planned_cases=2000,
        evidence_members=dict(original_config=config_member, original_lock=lock_member,
            original_runtime=runtime_member, original_observer=observer_member,
            cold_RPN_summary=summary_member, cold_RPN_reuse=rpn_member, cold_RPN_reference_result=rpn_result_member,
            frozen_generation_run=runner_member, frozen_generation_bridge=bridge_member),
        raw_local_only=True, edited_trajectory_resume=False, new_native_qualification_and_observer_guard_required=True)
    out = Path(out_directory)
    guard_path = out / 'old-cold-observation-guard.json'
    immutable_write(guard_path, guard)
    _, guard_member = read_member(guard_path)
    spec = dict(observations_root=str(arm / 'generation-raw/observations'),
        observer_identity_member=observer_member, config_member=config_member, runtime_member=runtime_member,
        source_commit=OLD_TASK_SOURCE, allowed_only_state_W=weights,
        cold_observation_guard_member=guard_member, original_state_identity=state,
        inventory_member=inventory_binding)
    binding = dict(schema=REUSE_BINDING_SCHEMA, old_w0_reuse=spec,
        original_config_scalar_aliases=dict(generation_source_sha=old['generation']['source_sha'],
            assets_manifest_member=copy.deepcopy(old['generation']['generation_assets'])),
        alias_basis='READ_ONLY_SCALARS_FROM_ORIGINAL_CONFIG_NOT_A_REWRITTEN_ORIGINAL_CONFIG',
        raw_local_only=True, original_config_rewritten=False, original_rows_relabelled=False,
        full_W0_READY=False, actual_qualification_created=False)
    binding_path = out / 'old-w0-reuse-binding.json'
    immutable_write(binding_path, binding)
    _, binding_member = read_member(binding_path)
    return dict(old_w0_reuse=spec, cold_observation_guard_member=guard_member, binding_member=binding_member)


def build_compatibility(inventory_member, qualification_member, new_runtime_identity, qualification_expected,
                        *, qualification_plan_member, out=None):
    """Actual route receipt + exact semantic identity authorize explicit mixed rows."""
    inventory, _ = read_member(inventory_member['path'], inventory_member)
    require(inventory['schema'] == INVENTORY_SCHEMA and inventory['identity_sha256'] == digest(inventory['identity'])
            and inventory['payload_sha256'] == digest({key: value for key, value in inventory.items() if key != 'payload_sha256'}),
            'REUSE_INVENTORY_IDENTITY')
    qualification, _ = read_member(qualification_member['path'], qualification_member)
    plan, _ = read_member(qualification_plan_member['path'], qualification_plan_member)
    required = {'plan_sha256', 'cohort_sha256', 'shared_source_sha', 'native_source_binding', 'selected_route', 'fixed_microbatch'}
    require(required <= qualification_expected.keys()
            and all(qualification.get(key) == value for key, value in qualification_expected.items()),
            'REUSE_QUALIFICATION_EXACT_BINDING')
    semantic = inventory['identity']['semantic_inputs']
    require(qualification['schema'] == QUALIFICATION_SCHEMA
            and qualification['status'] == 'QUALIFIED_ACTUAL_GPU_ROUTE' and qualification['actual_GPU'] is True
            and qualification['selected_route_passed'] is True and qualification['state_unchanged'] is True
            and qualification['RNG_restored'] is True and qualification['no_fit'] is True
            and qualification['checkpoint_saved'] is False and qualification['model_identity'] == semantic['model_identity'],
            'REUSE_ACTUAL_QUALIFICATION_REQUIRED')
    from .generation_cache_qualification import validate_actual_receipt, SHARED_ROUTES
    try:
        validate_actual_receipt(qualification, plan)
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ReuseError('REUSE_ACTUAL_MEASURED_QUALIFICATION_REQUIRED') from error
    require(new_runtime_identity['schema'] == SCHEMA and new_runtime_identity['profile'] == PROFILE
            and new_runtime_identity['eval_seed'] == EVAL_SEED
            and new_runtime_identity['model_identity'] == semantic['model_identity']
            and new_runtime_identity['reference_assets_sha256'] == semantic['reference_assets_sha256']
            and new_runtime_identity['generation_source_sha'] == qualification['shared_source_sha']
            and new_runtime_identity['route'] == SHARED_ROUTES[qualification['selected_route']],
            'REUSE_NEW_RUNTIME_SEMANTIC_BINDING')
    identity = dict(schema=COMPATIBILITY_SCHEMA, inventory=copy.deepcopy(inventory_member),
        inventory_identity_sha256=inventory['identity_sha256'], qualification=copy.deepcopy(qualification_member),
        qualification_plan=copy.deepcopy(qualification_plan_member),
        qualification_expected=copy.deepcopy(qualification_expected),
        qualification_route_map=copy.deepcopy(SHARED_ROUTES),
        old_runtime_identity=copy.deepcopy(inventory['identity']['old_runtime_identity']),
        old_runtime_sha256=inventory['identity']['old_runtime_sha256'],
        new_runtime_identity=copy.deepcopy(new_runtime_identity), new_runtime_sha256=digest(new_runtime_identity),
        semantic_inputs=copy.deepcopy(semantic), ordered_records_sha256=inventory['identity']['ordered_records_sha256'],
        provenance_bindings=[entry['provenance'] for entry in inventory['entries']
                             if entry['status'] == 'REUSABLE_COMPLETE_CASE'])
    result = dict(schema=COMPATIBILITY_SCHEMA, status='ACTUAL_QUALIFIED_REUSE_BINDING',
        identity=identity, identity_sha256=digest(identity), reusable_cases=inventory['reusable_cases'],
        planned_cases=2000, raw_local_only=True, new_source_relabel=False,
        original_runtime_checks_disabled=False, full_W0_READY=False)
    if out is not None:
        immutable_write(out, result)
    return result


def build_shared_compatibility(inventory_member, shared_qualification_member, old_guard_member,
                               new_runtime_identity, state_original, *, out,
                               qualification_plan_member, actual_qualification_member):
    """Explicit shared-schema adapter; old config/state/raw remain byte-original.

    Bypass W0Compatibility's alternative config/state shape assumptions. This
    binds only identity-equivalent original cold state, not an edited trajectory
    or a fabricated whole-endpoint observer/RAM guard.
    """
    inventory, inventory_binding = read_member(inventory_member['path'], inventory_member)
    validate_inventory(inventory)
    evidence, semantic = inventory['identity'], inventory['identity']['semantic_inputs']
    require(state_original == semantic['cold_state_identity'], 'REUSE_SHARED_ORIGINAL_COLD_STATE')
    old, config_member = read_member(evidence['old_config']['path'], evidence['old_config'])
    reference, reference_member = read_member(old['generation']['generation_assets']['path'],
                                              old['generation']['generation_assets'])
    guard, guard_member = read_member(old_guard_member['path'], old_guard_member)
    require(guard['schema'] == COLD_GUARD_SCHEMA and guard['source_commit'] == OLD_TASK_SOURCE
            and guard['phase'] == 'W0_generation' and guard['commits'] == guard['history_appends'] == 0
            and guard['original_state_identity'] == state_original
            and guard['model_W'] == state_original.get('selected_physical_W', state_original.get('W'))
            and guard['old_generation_runtime'] == evidence['old_runtime_sha256']
            and guard['whole_endpoint_guard_recorded'] is False and guard['partial_rows_authorized'] is True
            and guard['proof_basis'] == 'FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN'
            and guard['inventory_member'] == inventory_binding and guard['measured_final_history_zero'] is False,
            'REUSE_SHARED_SOURCE_BACKED_COLD_GUARD')
    for item in guard['evidence_members'].values():
        _verify_bytes(item)
    actual, actual_member = read_member(actual_qualification_member['path'], actual_qualification_member)
    plan, plan_member = read_member(qualification_plan_member['path'], qualification_plan_member)
    from .generation_cache_qualification import validate_actual_receipt, SHARED_ROUTES
    from project.run_scripts.experiment_generation_eval.kv_qualification import verify_actual_receipt
    from project.run_scripts.experiment_generation_eval.compatibility import SCHEMA as shared_schema, load_compatibility
    try:
        validate_actual_receipt(actual, plan)
        shared, shared_member = read_member(shared_qualification_member['path'], shared_qualification_member)
        verify_actual_receipt(shared_member, expected_plan_sha256=plan['shared_plan_sha256'],
                              expected_model_identity=semantic['model_identity'])
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ReuseError('REUSE_SHARED_ACTUAL_QUALIFICATION_REQUIRED') from error
    require(shared['caller_proof_member'] == actual_member
            and shared['private_plan_sha256'] == digest(plan)
            and shared['execution_adapter'] == 'TASK_PRIVATE_SHARED_GENERATE_ROWS_PROOF_CONVERSION_NOT_SHARED_RUN_QUALIFICATION'
            and actual['shared_plan_sha256'] == shared['plan_sha256'] == plan['shared_plan_sha256']
            and shared['selected_route'] == SHARED_ROUTES[actual['selected_route']] == new_runtime_identity['route']
            and shared['fixed_microbatch'] == actual['fixed_microbatch'] == new_runtime_identity['generation_microbatch']
            and shared['source_identity'] == actual['shared_source_sha'] == new_runtime_identity['generation_source_sha']
            and new_runtime_identity['qualification_receipt_sha256'] == shared_member['sha256']
            and all(new_runtime_identity[key] == semantic[key] for key in
                    ('model_identity', 'reference_assets_sha256', 'profile', 'eval_seed', 'schema'))
            and reference['identity_sha256'] == semantic['reference_assets_sha256'],
            'REUSE_SHARED_DUAL_PROOF_RUNTIME_BINDING')
    original_entries, exclusions = [], []
    for entry in inventory['entries']:
        if entry['status'] != 'REUSABLE_COMPLETE_CASE':
            exclusions.append(dict(occurrence=entry['occurrence'], reason=entry['reason']))
            continue
        raw, raw_member = read_member(entry['provenance']['raw']['path'], entry['provenance']['raw'])
        ri = raw['identity']['record_identity']
        record = dict(case_id=ri['case_id'], generation_prompts=ri['generation_prompts'],
            requested_rewrite=dict(relation_id=ri['relation_id'], target_new=dict(id=ri['target_new_id'])))
        checked = validate_case(raw, raw_member, runtime_identity=evidence['old_runtime_identity'],
            state_identity=state_original, record=record, occurrence=entry['occurrence'])
        require(checked['row'] == entry['row'] and checked['provenance'] == entry['provenance'],
                'REUSE_SHARED_ORIGINAL_ROW_CHANGED')
        original_entries.append(dict(occurrence=entry['occurrence'], original_raw_member=raw_member,
            original_identity_sha256=raw['identity_sha256'], original_payload_sha256=raw['payload_sha256'],
            original_runtime_sha256=evidence['old_runtime_sha256'],
            original_generation_source_sha=evidence['old_runtime_identity']['generation_source_sha'], original_route=OLD_ROUTE))
    identity = dict(schema=shared_schema, physical_state=copy.deepcopy(state_original),
        model_identity=semantic['model_identity'], old_observer_member=copy.deepcopy(evidence['old_observer']),
        old_config_member=config_member, old_reference_member=reference_member,
        old_cold_guard_member=guard_member, old_runtime_member=guard['evidence_members']['original_runtime'],
        new_runtime_sha256=digest(new_runtime_identity), reference_assets_sha256=semantic['reference_assets_sha256'],
        qualification_receipt_member=shared_member, actual_qualification_receipt_member=actual_member,
        qualification_plan_member=plan_member, inventory_member=inventory_binding,
        ordered_record_identity_sha256=evidence['ordered_records_sha256'], original_entries=original_entries,
        excluded=exclusions, eligible_cases=len(original_entries), planned_cases=2000,
        raw_local_only=True, edited_trajectory_resume=False, old_whole_endpoint_guard_recorded=False,
        old_partial_evidence='SOURCE_CONTROL_FLOW_COLD_RUNTIME_RPN_AND_ATOMIC_COMPLETE_ROWS',
        new_native_qualification_and_observer_guard_required=True,
        original_config_scalar_aliases=dict(generation_source_sha=old['generation']['source_sha'],
            assets_manifest_member=reference_member),
        scalar_alias_basis='ORIGINAL_CONFIG_SCALARS_ONLY_NOT_A_REWRITTEN_ORIGINAL_CONFIG')
    result = dict(identity=identity, identity_sha256=digest(identity))
    immutable_write(out, result)
    _, manifest_member = read_member(out)
    load_compatibility(manifest_member, expected_runtime=digest(new_runtime_identity),
                       expected_qualification=shared_member['sha256'])
    return dict(manifest=result, member=manifest_member)
