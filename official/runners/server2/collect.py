"""CPU-only independent reduction of one source-bound official registration.

No model/checkpoint tensor is loaded, no experiment is submitted/retried, and
only the exact IDs from this attempt are queried for accounting. Terminal is
published atomically *after* result/inventory/report have been persisted.
"""
import argparse
import csv
import getpass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from official.evaluation import reduce as factual_reduce
from official.experiments.prepare import METHODS, digest, file_sha, read, write_new

TASK = 'official-baselines-server2-20261008-r1'
HISTORY_COUNTS = dict(FT=0, MEMIT=0, ALPHAEDIT=120, ALPHAEDIT_BLUE=40, MEMIT_FE=0, SPHERE=120)
TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                   'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def require(value, code):
    if not value:
        raise ValueError(code)


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'COLLECT_REGULAR_MEMBER')
    before = path.stat()
    checksum = file_sha(path)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'COLLECT_MEMBER_CHANGED')
    return dict(path=str(path), bytes=after.st_size, sha256=checksum)


def verify_member(value, inventory=None):
    require(isinstance(value, dict) and {'path', 'bytes', 'sha256'} <= set(value), 'COLLECT_MEMBER_SCHEMA')
    require(Path(value['path']).is_absolute(), 'COLLECT_ABSOLUTE_MEMBER_REQUIRED')
    actual = member(value['path'])
    require(all(actual[key] == value[key] for key in actual), 'COLLECT_MEMBER_SHA_SIZE')
    if inventory is not None:
        inventory.append(actual)
    return Path(value['path'])


def receipt_identity(value, manifest):
    require(value.get('code_commit') == manifest['code_commit']
        and value.get('official_tree_sha256') == manifest['official_tree_sha256']
        and value.get('manifest_sha256', value.get('base_manifest_sha256')) == manifest['base_manifest_sha256'],
        'COLLECT_RUNTIME_SOURCE_MANIFEST_IDENTITY')


def expected_checkpoint_identity(manifest, dataset, method):
    return manifest['checkpoint_identities'][dataset][method]


def validate_qualification(value, manifest, method):
    receipt_identity(value, manifest)
    require(value.get('status') == 'PASS_ACTUAL_QUALIFICATION'
        and value.get('actual_GPU') is True and value.get('model') == 'gptj'
        and value.get('method') == method and value.get('dataset') == 'cf',
        'COLLECT_ACTUAL_NATIVE_QUALIFICATION_REQUIRED')
    require(value.get('continuous_batches') == 3 and value.get('resume_after_batch') == 2
        and value.get('resumed_batches') == [3]
        and all(value.get(key) is True for key in ('weights_equal', 'history_equal', 'rng_equal', 'metrics_equal')),
        'COLLECT_ACTUAL_B2_B3_RESUME_REQUIRED')
    require(value.get('contexts_equal') is True
        and value.get('checkpoint_identity') == expected_checkpoint_identity(manifest, 'cf', method),
        'COLLECT_QUALIFICATION_CHECKPOINT_CONTEXT_IDENTITY')
    require(isinstance(value.get('native_owner_formula_parity'), dict)
        and {'path', 'bytes', 'sha256'} <= set(value['native_owner_formula_parity']),
        'COLLECT_QUALIFICATION_OWNER_FORMULA_MEMBER_REQUIRED')
    return dict(value)


def aggregate_qualification(methods, manifest):
    require(set(methods) == set(METHODS), 'COLLECT_SIX_METHOD_QUALIFICATIONS')
    verified = {method: validate_qualification(methods[method], manifest, method) for method in METHODS}
    return dict(status='PASS_ACTUAL_QUALIFICATION', actual_GPU=True, model='gptj', dataset='cf',
        code_commit=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        manifest_sha256=manifest['base_manifest_sha256'], methods=verified,
        owner_CPU_reducer=True, actual_model_proofs_not_CPU_fixtures=True)


def validate_qualification_calls(value, manifest, method):
    require(value.get('actual_native_batch_calls') == 4
        and value.get('actual_native_request_applications') == 400
        and value.get('durable_B2', {}).get('batch') == 2
        and value.get('checkpoint', {}).get('batch') == 3
        and value['durable_B2'].get('identity_sha256') == digest(value['checkpoint_identity'])
        and value['checkpoint'].get('identity_sha256') == digest(value['checkpoint_identity']),
        'COLLECT_QUALIFICATION_ACTUAL_CALL_DURABLE_COUNTS')
    native = value.get('native_commits')
    require(isinstance(native, list) and len(native) == 4
        and [row.get('batch') for row in native] == [1, 2, 3, 3]
        and all(row.get('method') == method and row.get('requests') == 100
            and row.get('history_appends_expected') == HISTORY_COUNTS[method] // 20
            and row.get('native_source_sha256') in manifest['native_source_sha256'][method]
            for row in native)
        and native[1].get('checkpoint') == value['durable_B2'],
        'COLLECT_QUALIFICATION_FOUR_SOURCE_BOUND_NATIVE_CALLS')


def validate_owner_formula_parity(value, manifest, method, *, dataset='cf', inventory=None):
    """Same-call owner formula controls are not an independent evaluator oracle."""
    proof = read(verify_member(value, inventory))
    plan = manifest['native_parity_plans'][dataset][method]
    require(plan.get('plan_sha256') == digest({key:item for key,item in plan.items() if key != 'plan_sha256'})
        and plan.get('schema') == 'official-server2-first-trajectory-native-parity-v1'
        and plan.get('status') == 'PLAN_READY_OWNER_SOURCE_FORMULA_CONTROLS'
        and plan.get('actual_GPU') is False and plan.get('scientific_quality_gate') is False
        and plan.get('after_result_tolerance_relaxation') is False
        and plan.get('model_revision') == manifest['model_revision']
        and plan.get('tokenizer_sha256') == manifest['tokenizer_sha256']
        and plan.get('stream_sha256') == manifest['streams'][dataset]['lock']['stream_sha256']
        and plan.get('method') == method and plan.get('dataset') == dataset,
        'COLLECT_OWNER_FORMULA_PREMEASUREMENT_PLAN_IDENTITY')
    source = plan['source']
    require(all(key in source and path in manifest['source_members']
        and source[key] == manifest['source_members'][path] for key,path in (
        ('hparams_sha256', f'hparams/{method}/gptj.json'),
        ('factual_sha256', 'evaluation/factual.py'), ('parity_sha256', 'runners/server2/parity.py'),
        ('nethook_sha256', 'baselines/easyedit/util/nethook.py'),
        ('native_easy_repr_sha256', 'baselines/easyedit/models/rome/repr_tools.py'),
        ('native_BLUE_repr_sha256', 'baselines/blue/rome/repr_tools.py'))),
        'COLLECT_OWNER_FORMULA_SOURCE_NOT_FROZEN_MEMBER')
    require(proof.get('schema') == 'official-server2-first-trajectory-native-parity-v1'
        and proof.get('status') == 'PASS_ACTUAL_OWNER_FORMULA_PARITY' and proof.get('actual_GPU') is True
        and proof.get('method') == method and proof.get('dataset') == dataset
        and proof.get('plan_sha256') == plan['plan_sha256']
        and proof.get('code_commit') == source['code_commit'] == manifest['code_commit']
        and proof.get('official_tree_sha256') == source['official_tree_sha256'] == manifest['official_tree_sha256']
        and proof.get('factual_source_sha256') == source['factual_sha256']
        and proof.get('native_hparams_sha256') == source['hparams_sha256'],
        'COLLECT_ACTUAL_OWNER_FORMULA_SOURCE_IDENTITY')
    require(all(proof.get(key) is True for key in ('candidate_nll_close', 'candidate_token_prefix_exact',
        'token_predictions_exact', 'subject_lookup_exact', 'observer_no_mutation', 'RNG_restored'))
        and all(type(proof.get(key)) is int and proof[key] == 0 for key in
                ('extra_LM_forward_calls', 'target_fit_calls', 'optimizer_calls', 'writer_calls'))
        and type(proof.get('existing_factual_forward_calls')) is int and proof['existing_factual_forward_calls'] > 0
        and type(proof.get('candidate_nll_abs_error')) in (int, float)
        and math.isfinite(proof['candidate_nll_abs_error']) and proof['candidate_nll_abs_error'] >= 0
        and proof['candidate_nll_abs_error'] <= plan['tolerance']['candidate_nll']['atol']
        and plan['tolerance']['candidate_nll']['rtol'] == 0
        and re.fullmatch(r'[0-9a-f]{64}', proof.get('native_candidate_identity_sha256', '')),
        'COLLECT_OWNER_FORMULA_CANDIDATE_TOKEN_LOOKUP_STATE_COUNTS')
    layers = list(dict.fromkeys((source['native_hparams']['layers'][0], source['native_hparams']['layers'][-1])))
    fc_names = {f'transformer.h.{layer}.mlp.fc_out' for layer in layers}
    blocks = {f'transformer.h.{layer}' for layer in [*layers, 27]}
    require(set(proof.get('fc_out', {})) == fc_names
        and set(proof.get('block_output_schemas', {})) == blocks, 'COLLECT_OWNER_FORMULA_GPTJ_LAYER_MAPPING')
    affine = plan['tolerance']['affine_readout']
    for name, check in [*proof['fc_out'].items(), ('readout27', proof.get('readout27', {}))]:
        shape = check.get('shape')
        require(check.get('close') is True and check.get('dtype') == 'float32'
            and isinstance(shape, list) and len(shape) == 2 and type(shape[0]) is int and shape[0] > 0
            and shape[1] == (50400 if name == 'readout27' else 4096)
            and check.get('atol') == affine['atol'] and check.get('rtol') == affine['rtol']
            and type(check.get('max_abs_error')) in (int, float)
            and math.isfinite(check['max_abs_error']) and check['max_abs_error'] >= 0,
            'COLLECT_OWNER_FORMULA_NATIVE_AFFINE_READOUT')
    for block in proof['block_output_schemas'].values():
        require(block.get('container') in ('Tensor', 'tuple', 'list')
            and isinstance(block.get('hidden_shape'), list) and len(block['hidden_shape']) == 2
            and type(block['hidden_shape'][0]) is int and block['hidden_shape'][0] > 0
            and block['hidden_shape'][1] == 4096, 'COLLECT_OWNER_FORMULA_GPTJ_BLOCK_LAYOUT')
    require(proof.get('independent_public_native_evaluator') == plan['independent_public_native_evaluator']
        == 'NOT_AVAILABLE_IN_OFFICIAL_DISTRIBUTION' and proof.get('independent_oracle_PASS') is False
        and proof.get('bitwise_full_evaluator_claim') is False and proof.get('scientific_quality_gate') is False
        and proof.get('evidence_scope') == plan['evidence_scope'],
        'COLLECT_OWNER_FORMULA_SCOPE_NOT_INDEPENDENT_EVALUATOR_PARITY')
    return dict(member=value, status=proof['status'], actual_GPU=True, plan_sha256=plan['plan_sha256'],
        same_existing_forward=True, extra_LM_forward_calls=0, independent_oracle_PASS=False,
        CF_original_evaluator_parity='NOT_ESTABLISHED_BY_OWNER_FORMULA_CONTROL')


def _compare_summary(recorded, reduced):
    require(isinstance(recorded, dict), 'COLLECT_FACTUAL_SUMMARY')
    for key, expected in reduced.items():
        actual = recorded.get(key)
        if type(expected) is int:
            require(type(actual) is int and actual == expected, 'COLLECT_FACTUAL_REDUCER_COUNT:'+key)
        else:
            # No model-numeric parity test: this is deterministic scalar
            # arithmetic on already stored case rows and native denominators.
            require(type(actual) in (int, float) and math.isfinite(actual)
                    and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10),
                    'COLLECT_FACTUAL_RAW_REDUCTION:'+key)


def expected_factual_external_identity(manifest, dataset):
    return dict(model_revision=manifest['model_revision'], tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        runtime=manifest['runtime'], source=manifest['code_commit'])


def validate_factual(value, dataset, records, *, manifest, endpoint=None):
    from official.evaluation.factual import SCHEMA, TOKENIZATION
    cases = value.get('cases')
    require(isinstance(cases, list) and len(cases) == len(records), 'COLLECT_FACTUAL_REQUEST_COUNT')
    require([case.get('case_id') for case in cases] == [record['case_id'] for record in records],
            'COLLECT_FACTUAL_CASE_ORDER')
    require([case.get('occurrence_index') for case in cases] ==
        [record['occurrence_index'] for record in records], 'COLLECT_FACTUAL_OCCURRENCE_ORDER')
    identity = value.get('identity')
    require(isinstance(identity, dict) and identity, 'COLLECT_FACTUAL_RUNTIME_IDENTITY')
    require(value.get('identity_sha256') == digest(identity), 'COLLECT_FACTUAL_IDENTITY_SHA')
    require(identity.get('schema') == SCHEMA and identity.get('dataset') == dataset
        and identity.get('tokenization') == TOKENIZATION
        and identity.get('ordered_occurrences') == [record['occurrence_index'] for record in records]
        and identity.get('padding') == 'RIGHT_EXPLICIT_ATTENTION_MASK' and identity.get('use_cache') is False
        and re.fullmatch(r'[0-9a-f]{64}', identity.get('cohort_sha256', '')),
        'COLLECT_FACTUAL_QUERY_COHORT_TOKENIZATION_IDENTITY')
    require(identity.get('external_identity') == expected_factual_external_identity(manifest, dataset),
        'COLLECT_FACTUAL_EXTERNAL_MODEL_SOURCE_RUNTIME_IDENTITY')
    if endpoint is not None and 'endpoint' in value:
        require(value['endpoint'] == endpoint, 'COLLECT_FACTUAL_ENDPOINT')
    if 'requests' in value:
        require(value['requests'] == len(records), 'COLLECT_FACTUAL_DENOMINATOR')
    reduced = factual_reduce.counterfact(cases) if dataset == 'cf' else factual_reduce.zsre(cases)
    _compare_summary(value['summary'], reduced)
    return dict(reduced, identity_sha256=digest(identity), query_cohort_sha256=identity['cohort_sha256'],
                external_identity_sha256=digest(identity['external_identity']), endpoint=endpoint,
                reduction='official.evaluation.reduce request-macro stored case rows')


def validate_checkpoint_metadata(folder, identity, *, final_batch=20, inventory=None):
    folder = Path(folder).absolute()
    pointer = folder / 'latest.json'
    ref = read(pointer)
    require(ref.get('batch') == final_batch and ref.get('identity_sha256') == digest(identity)
        and ref.get('final_W20') is (final_batch == 20), 'COLLECT_CHECKPOINT_POINTER_BATCH_IDENTITY')
    filename = ref.get('file')
    require(type(filename) is str and Path(filename).name == filename
        and not Path(filename).is_absolute(), 'COLLECT_CHECKPOINT_UNSAFE_MEMBER')
    path = folder / filename
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        and re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', '')), 'COLLECT_CHECKPOINT_METADATA')
    pointer_member = member(pointer)
    if inventory is not None:
        inventory.append(pointer_member)
    return dict(pointer=pointer_member, checkpoint_path=str(path), checkpoint_bytes=path.stat().st_size,
        saved_payload_sha256=ref['sha256'], batch=final_batch, final_W20=final_batch == 20,
        verification='POINTER_AND_NATIVE_ATOMIC_SAVE_HASH; PAYLOAD_NOT_REHASHED_OR_DESERIALIZED',
        tensor_loads=0)


def validate_generation_ready(value, assets, records, *, endpoint, inventory=None):
    from official.runners.server2 import generation as bridge
    from official.evaluation.generation.native_observer import read_observed
    from official.evaluation.generation.native_profile import PROFILE, ROUTE, runtime_identity
    config = bridge.configuration(assets)
    runtime = runtime_identity(config, config['reference_assets_sha256'])
    identity = value.get('identity', {})
    require(value.get('status') == 'READY_COMPLETE_ENDPOINT' and value.get('identity_sha256') == digest(identity)
        and identity.get('schema') == bridge.SCHEMA and identity.get('study_instruction') == bridge.STUDY
        and identity.get('endpoint') == endpoint and identity.get('full_requests') == 2000
        and identity.get('runtime_identity') == runtime and identity.get('runtime_sha256') == digest(runtime)
        and identity.get('model_identity') == config['model_identity']
        and identity.get('source_identity') == bridge.source_identity()
        and identity.get('reference_assets_sha256') == config['reference_assets_sha256']
        and identity.get('stream_sha256') == assets['streams']['cf']['lock']['stream_sha256']
        and identity.get('generation_schedule') == 'CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN'
        and identity.get('old_W20_only_or_partial_reused') is False
        and value.get('RNG_restored') is True and value.get('observer_no_mutation') is True,
        'COLLECT_OFFICIAL_GENERATION_READY_IDENTITY')
    require(runtime['profile'] == PROFILE and runtime['route'] == ROUTE and runtime['eval_seed'] == 20261007,
            'COLLECT_NATIVE_GENERATION_PROFILE')
    observed_path = verify_member(value['endpoint_member'], inventory)
    # Shared independent native reader checks every raw digest, ordered row,
    # token/EOS/length/profile/physical-work and actual execution receipt. Assets
    # aren't reloaded and generation/scoring/model forwards are never repeated.
    observed = read_observed(observed_path, expected_runtime=digest(runtime))
    require(observed['identity']['endpoint'] == endpoint
        and observed['identity']['cohort'] == bridge.COHORT
        and observed['identity']['ordered_occurrences'] == list(range(1, 2001))
        and observed['identity']['state_sha256'] == identity['physical_state_sha256']
        and [row['occurrence'] for row in observed['rows']] == list(range(1, 2001))
        and [row['case_id'] for row in observed['rows']] == [record['case_id'] for record in records]
        and observed['summary']['planned_count'] == 2000 and observed['summary'] == value['summary']
        and 'native_execution_member' in observed and 'parent_endpoint_member' not in observed,
        'COLLECT_GENERATION_2000_COMPLETE_SOURCE_EXECUTION')
    execution_member = observed['native_execution_member']
    require(value['native_execution_member'] == execution_member, 'COLLECT_GENERATION_EXECUTION_MEMBER')
    verify_member(execution_member, inventory)
    for row in observed['rows']:
        raw_member = member(row['observation_path'])
        if inventory is not None:
            inventory.append(raw_member)
    return dict(endpoint=endpoint, requests=2000, profile=PROFILE, route=ROUTE,
        identity_sha256=observed['identity_sha256'], runtime_sha256=digest(runtime),
        sampling_stream_sha256=observed['identity']['sampling_stream_sha256'],
        source_identity_sha256=digest(identity['source_identity']), summary=observed['summary'],
        one_generated_text_set_shared_between_fluency_consistency=True,
        reduction='STORED_RAW_DIGEST_NATIVE_PROFILE_EXECUTION_AND_SCALAR_SUMS; NO_NEW_GENERATION',
        original_W20_only_not_relabelled=True, GPU_forwards=0)


def validate_chain(value, manifest, assets, records, method, dataset, *, inventory=None):
    receipt_identity(value, manifest)
    require(value.get('status') == 'SCIENTIFIC_COMPLETE' and value.get('actual_GPU') is True
        and value.get('method') == method and value.get('dataset') == dataset
        and value.get('model') == 'gptj' and value.get('batches') == 20
        and value.get('history_appends') == HISTORY_COUNTS[method] and value.get('ownstate_links') == 19,
        'COLLECT_TWENTY_BATCH_NATIVE_COMPLETION')
    identity = expected_checkpoint_identity(manifest, dataset, method)
    require(value.get('checkpoint_identity') == identity, 'COLLECT_CHAIN_CHECKPOINT_IDENTITY')
    commits = value.get('commits')
    require(isinstance(commits, list) and len(commits) == 20, 'COLLECT_TWENTY_COMMIT_RECEIPTS')
    per_batch_history = HISTORY_COUNTS[method] // 20
    for batch, item in enumerate(commits, 1):
        commit = read(verify_member(item, inventory))
        require(commit.get('batch') == batch and commit.get('previous_batch') == batch - 1
            and commit.get('requests') == 100 and commit.get('checkpoint_identity') == identity,
            'COLLECT_COMMIT_ORDER_STATE_LINK')
        native = commit.get('native', commit)
        require(native.get('batch') == batch and native.get('method') == method
            and native.get('history_appends_expected') == per_batch_history
            and native.get('native_source_sha256') in manifest.get('native_source_sha256', {}).get(method, []),
            'COLLECT_COMMIT_NATIVE_SOURCE_HISTORY')
    checkpoint = validate_checkpoint_metadata(value['checkpoint_folder'], identity, inventory=inventory)
    require(commit.get('checkpoint', {}).get('sha256') == checkpoint['saved_payload_sha256']
        and commit['checkpoint'].get('batch') == 20
        and commit['checkpoint'].get('identity_sha256') == digest(identity),
        'COLLECT_FINAL_COMMIT_CHECKPOINT_POINTER')
    endpoints = value.get('factual_endpoints')
    require(isinstance(endpoints, dict) and set(endpoints) == {'W0', 'W5', 'W10', 'W15', 'W20'},
            'COLLECT_FACTUAL_ENDPOINT_COVERAGE')
    factual = {}
    for endpoint, count in (('W0', 2000), ('W5', 500), ('W10', 1000), ('W15', 1500), ('W20', 2000)):
        observed = read(verify_member(endpoints[endpoint], inventory))
        factual[endpoint] = validate_factual(observed, dataset, records[:count], manifest=manifest, endpoint=endpoint)
    require(value.get('W0_READY'), 'COLLECT_CHAIN_SAME_MODEL_COLD_W0_REQUIRED')
    cold_ready = read(verify_member(value['W0_READY'], inventory))
    validate_cold_w0(cold_ready, manifest, assets, {dataset:records},
        role='W0_'+dataset.upper(), inventory=inventory)
    require(endpoints['W0'] == cold_ready['factual'], 'COLLECT_CHAIN_SAME_MODEL_COLD_W0_BINDING')
    generation = None
    if dataset == 'cf':
        require(value.get('generation_ready') and value.get('w0_ready'), 'COLLECT_CF_W0_W20_GENERATION_REQUIRED')
        require(value['w0_ready'] == cold_ready['generation_READY'], 'COLLECT_CHAIN_COLD_GENERATION_BINDING')
        generation = {}
        for endpoint, key in (('W0', 'w0_ready'), ('W20', 'generation_ready')):
            ready = read(verify_member(value[key], inventory))
            generation[endpoint] = validate_generation_ready(ready, assets, records,
                endpoint=endpoint, inventory=inventory)
        if 'generation_endpoint' in value:
            endpoint = verify_member(value['generation_endpoint'], inventory)
            ready = read(value['generation_ready']['path'])
            require(str(endpoint) == ready['endpoint_member']['path'], 'COLLECT_CF_GENERATION_ENDPOINT_MEMBER')
    else:
        require(not value.get('generation_ready') and not value.get('generation_endpoint')
                and not value.get('w0_ready'), 'COLLECT_ZSRE_NO_GENERATION')
    return dict(status='SCIENTIFIC_COMPLETE', method=method, dataset=dataset, batches=20,
        commits=20, ownstate_links=19, history_appends=HISTORY_COUNTS[method],
        native_history_count_basis='EXPECTED_PER_NATIVE_SOURCE_CALL_TIMES_TWENTY_VERIFIED_COMMITS',
        factual=factual, generation=generation, checkpoint=checkpoint,
        tensor_loads=0, GPU_forwards=0)


def validate_cold_w0(value, manifest, assets, datasets, *, role=None, inventory=None):
    receipt_identity(value, manifest)
    require(value.get('status') == 'READY_COLD_W0_COMPLETE' and value.get('actual_GPU') is True
        and value.get('model') == 'gptj' and value.get('dataset') in ('cf', 'zsre')
        and value.get('model_revision') == manifest['model_revision']
        and value.get('tokenizer_sha256') == manifest['tokenizer_sha256'], 'COLLECT_ACTUAL_COLD_W0')
    dataset = value['dataset']
    require(role in (None, 'W0_'+dataset.upper()), 'COLLECT_W0_ROLE_DATASET_IDENTITY')
    require(value.get('stream_sha256') == assets['streams'][dataset]['lock']['stream_sha256'],
            'COLLECT_W0_STREAM_IDENTITY')
    observed = read(verify_member(value['factual'], inventory))
    summary = validate_factual(observed, dataset, datasets[dataset], manifest=manifest, endpoint='W0')
    if dataset == 'cf':
        from official.runners.server2.run import verify_cf_native_oracle
        binding = value.get('original_native_reference')
        summary['original_native_reference'] = verify_cf_native_oracle(manifest, binding)
        for key in ('proof', 'canonical', 'state'):
            verify_member(binding[key], inventory)
        canonical = read(binding['canonical']['path'])
        validate_factual(canonical, 'cf', datasets['cf'][:4], manifest=manifest,
            endpoint='native-reference-canonical-first4')
        ready = read(verify_member(value['generation_READY'], inventory))
        summary['generation'] = validate_generation_ready(ready, assets, datasets['cf'],
            endpoint='W0', inventory=inventory)
        require(value['generation'] == ready['endpoint_member'], 'COLLECT_W0_GENERATION_ENDPOINT_BINDING')
    else:
        require(not value.get('generation') and not value.get('generation_READY'),
                'COLLECT_ZSRE_W0_NO_GENERATION')
        reference = read(verify_member(value['w0_reference'], inventory))
        require(reference.get('evaluation', {}).get('cases') == observed['cases']
            and reference['evaluation'].get('summary') == observed['summary'],
            'COLLECT_ZSRE_W0_REFERENCE_EVALUATION_BINDING')
    return dict(summary, cold_W0_actual_GPU=True, dataset=dataset,
                source_manifest_identity_verified=True)


def validate_smoke(value, manifest, assets, records, role_out, *, inventory=None):
    receipt_identity(value, manifest)
    method = manifest.get('smoke_method', 'MEMIT')
    identity = expected_checkpoint_identity(manifest, 'zsre', method)
    require(value.get('status') == 'PASS_ACTUAL_SMOKE' and value.get('actual_GPU') is True
        and value.get('model') == 'gptj' and value.get('dataset') == 'zsre'
        and value.get('method') == method and value.get('batches') == 1
        and value.get('checkpoint_identity') == identity, 'COLLECT_ACTUAL_ZSRE_SMOKE')
    require(isinstance(value.get('commits'), list) and len(value['commits']) == 1,
            'COLLECT_SINGLE_SMOKE_NATIVE_COMMIT')
    commit = read(verify_member(value['commits'][0], inventory))
    require(commit.get('batch') == 1 and commit.get('previous_batch') == 0
        and commit.get('method') == method and commit.get('requests') == 100
        and commit.get('checkpoint_identity') == identity
        and commit.get('history_appends_expected') == HISTORY_COUNTS[method] // 20
        and commit.get('native_source_sha256') in manifest['native_source_sha256'][method],
        'COLLECT_SMOKE_NATIVE_SOURCE_STATE_HISTORY')
    checkpoint = validate_checkpoint_metadata(Path(role_out)/'checkpoints', identity,
        final_batch=1, inventory=inventory)
    require(commit.get('checkpoint', {}).get('sha256') == checkpoint['saved_payload_sha256'],
            'COLLECT_SMOKE_NATIVE_CHECKPOINT_POINTER')
    require(set(value.get('factual_endpoints', {})) == {'W0', 'W1'}, 'COLLECT_SMOKE_FACTUAL_COVERAGE')
    ready_member = member(manifest['W0_zsre_ready_path'])
    if inventory is not None:
        inventory.append(ready_member)
    ready = read(ready_member['path'])
    validate_cold_w0(ready, manifest, assets, {'zsre':records}, role='W0_ZSRE', inventory=inventory)
    require(value['factual_endpoints']['W0'] == ready['factual'], 'COLLECT_SMOKE_SAME_COLD_W0_BINDING')
    for endpoint, count in (('W0', 2000), ('W1', 100)):
        observed = read(verify_member(value['factual_endpoints'][endpoint], inventory))
        validate_factual(observed, 'zsre', records[:count], manifest=manifest, endpoint=endpoint)
    return dict(value, W0member=ready_member, collector_checkpoint_metadata=checkpoint,
                native_batch_calls=1, native_request_applications=100)


def accounting(jobs, stage, *, owner=None, run=None):
    owner = getpass.getuser() if owner is None else owner
    require(jobs and len(set(jobs.values())) == len(jobs)
        and all(re.fullmatch(r'[1-9][0-9]*', job) for job in jobs.values()), 'COLLECT_EXACT_OWN_JOB_IDS')
    argv = ['sacct', '-n', '-P', '-j', ','.join(jobs.values()),
            '--format=JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES%200,MaxRSS,User,NodeList']
    if run is None:
        def run(command):
            answer = subprocess.run(command, text=True, capture_output=True, timeout=45)
            require(answer.returncode == 0, 'COLLECT_ACCOUNTING_COMMAND_FAILED')
            return answer.stdout
    raw = run(argv)
    roots = {}
    steps = []
    for line in raw.splitlines():
        cells = line.split('|')
        if cells[-1] == '':
            cells.pop()  # one sacct delimiter; preserve an empty NodeList field
        require(len(cells) == 8, 'COLLECT_ACCOUNTING_FORMAT')
        job, name, state, elapsed, tres, rss, user, node = cells
        if job in jobs.values():
            role = next(role for role, identifier in jobs.items() if identifier == job)
            require(user == owner and name == TASK+'-'+stage+'-'+role
                and node in ('server2', '', 'None'), 'COLLECT_ACCOUNTING_OWNER_TASK_NODE')
            require(elapsed.isdigit(), 'COLLECT_ALLOCATED_ELAPSED')
            state = state.split(' ', 1)[0].removesuffix('+')
            match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres)
            gpus = int(match.group(1)) if match else sum(int(value) for value in
                re.findall(r'(?:^|,)gres/gpu:[^,=]+=(\d+)(?:,|$)', tres))
            require(gpus in ((0,) if role == 'collector' else (0, 1)), 'COLLECT_ALLOCATION_GPU_COUNT')
            roots[role] = dict(job_id=job, state=state, elapsed_seconds=int(elapsed),
                allocated_gpus=gpus, allocated_GPU_seconds=gpus*int(elapsed), MaxRSS=rss,
                AllocTRES=tres, owner=user, node=node)
        elif any(job.startswith(identifier+'.') for identifier in jobs.values()):
            steps.append(dict(job_id=job, state=state, MaxRSS=rss))
        else:
            require(False, 'COLLECT_UNREQUESTED_ACCOUNTING_JOB')
    return dict(status='ACCOUNTING_OBSERVED' if set(roots) == set(jobs) else 'ACCOUNTING_INCOMPLETE',
        exact_requested_jobs=jobs, jobs=roots, step_MaxRSS=steps,
        allocated_GPU_seconds=sum(row['allocated_GPU_seconds'] for row in roots.values()),
        allocated_GPU_hours=sum(row['allocated_GPU_seconds'] for row in roots.values()) / 3600,
        unobserved_roles=sorted(set(jobs)-set(roots)), raw=raw,
        cost_coverage='FULL_REGISTERED_IDS' if set(roots) == set(jobs) else 'OBSERVED_SUBSET_ONLY',
        only_exact_own_accounting=True, new_queries=1, other_job_queries=0)


def _atomic_terminal(path, value):
    path = Path(path)
    require(not path.exists(), 'COLLECT_TERMINAL_ALREADY_EXISTS')
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.terminal-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partial(out):
    for filename in ('failure.json', 'terminal.json'):
        path = Path(out) / filename
        if path.is_file() and not path.is_symlink():
            row = read(path)
            return dict(local_receipt=member(path), preserved=True,
                **{key: row[key] for key in ('status', 'stage', 'batch', 'last_committed_batch', 'error_type') if key in row})
    return dict(status='NOT_RECORDED', preserved=True)


def _write_inventory(path, values):
    unique = {row['path']: row for row in values}
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('path', 'bytes', 'sha256'))
        writer.writeheader()
        writer.writerows(unique[path] for path in sorted(unique))
    return dict(files=len(unique), logical_bytes=sum(row['bytes'] for row in unique.values()),
                private_local_only=True, per_case_source_in_Git=False)


def collect(attempt, *, account=None):
    attempt = Path(attempt).absolute()
    require(attempt.is_dir() and not attempt.is_symlink(), 'COLLECT_ATTEMPT_DIRECTORY')
    terminal = attempt / 'collector' / 'terminal.json'
    if terminal.is_file():
        return dict(read(terminal), duplicate_collection_prevented=True)
    manifest, submission = read(attempt/'manifest.json'), read(attempt/'submission.json')
    stage = manifest['registration_stage']
    if stage == 'cf':
        from official.runners.server2.submit import gate
        bound = manifest.get('qualification_receipt')
        require(isinstance(bound, dict), 'COLLECT_CF_NATIVE_RESUME_INPUT_REQUIRED')
        verify_member(bound)
        require(gate(bound['path'], manifest=manifest, kind='qualification') == bound,
                'COLLECT_CF_NATIVE_RESUME_INPUT_BINDING_CHANGED')
    require(submission.get('status') == 'SUBMISSION_HANDOFF' and submission.get('stage') == stage
        and submission.get('source', {}).get('code_commit') == manifest['code_commit']
        and submission.get('source', {}).get('official_tree_sha256') == manifest['official_tree_sha256']
        and submission.get('base_manifest_sha256') == manifest['base_manifest_sha256'],
        'COLLECT_SUBMISSION_SOURCE_IDENTITY')
    jobs = submission['jobs']
    require(set(jobs) == set(manifest['registration_roles']) | {'collector'}, 'COLLECT_REGISTERED_ROLE_COVERAGE')
    inventory = [member(attempt/'manifest.json'), member(attempt/'submission.json')]
    asset_binding = manifest.get('asset_manifest', manifest.get('assets_manifest'))
    if isinstance(asset_binding, str):
        item = member(asset_binding)
        require(item['sha256'] == manifest['asset_manifest_sha256'], 'COLLECT_ASSETS_MANIFEST_SHA')
        inventory.append(item)
        assets = read(asset_binding)
    else:
        require(isinstance(asset_binding, dict), 'COLLECT_ASSETS_MANIFEST_BINDING')
        assets = read(verify_member(asset_binding, inventory))
    datasets = {dataset: read(verify_member(value['member'], inventory))
                for dataset, value in assets['streams'].items()}
    out = attempt / 'collector'
    out.mkdir(exist_ok=True)
    summaries, failures, methods = {}, {}, {}
    for role in manifest['registration_roles']:
        role_out = attempt / role
        try:
            if stage == 'qualification':
                path = role_out/'qualification.json'
                inventory.append(member(path))
                result = validate_qualification(read(path), manifest, role)
                validate_qualification_calls(result, manifest, role)
                validate_owner_formula_parity(result['native_owner_formula_parity'], manifest, role,
                    inventory=inventory)
                continuous = read(verify_member(result['continuous_metric'], inventory))
                resumed = read(verify_member(result['resumed_metric'], inventory))
                validate_factual(continuous, 'cf', datasets['cf'][200:300], manifest=manifest)
                validate_factual(resumed, 'cf', datasets['cf'][200:300], manifest=manifest)
                require(continuous['cases'] == resumed['cases'] and continuous['summary'] == resumed['summary'],
                        'COLLECT_QUALIFICATION_RECORDED_METRIC_PARITY')
                checkpoint_verified = validate_checkpoint_metadata(role_out/'checkpoints',
                    result['checkpoint_identity'], final_batch=3, inventory=inventory)
                require(checkpoint_verified['saved_payload_sha256'] == result['checkpoint']['sha256'],
                        'COLLECT_QUALIFICATION_FINAL_POINTER')
                result['collector_checkpoint_metadata'] = checkpoint_verified
                summaries[role] = result
                methods[role] = result
            else:
                is_w0 = role.startswith('W0_') or stage == 'w0'
                path = role_out/('smoke.json' if role == 'ZSRE_SMOKE' else
                                 'READY.json' if is_w0 else 'result.json')
                inventory.append(member(path))
                result = read(path)
                if is_w0:
                    summaries[role] = validate_cold_w0(result, manifest, assets, datasets,
                        role=role if role.startswith('W0_') else None, inventory=inventory)
                elif stage in ('cf', 'zsre') and role != 'ZSRE_SMOKE':
                    summaries[role] = validate_chain(result, manifest, assets, datasets[stage], role, stage,
                                                     inventory=inventory)
                elif role == 'ZSRE_SMOKE':
                    summaries[role] = validate_smoke(result, manifest, assets, datasets['zsre'],
                        role_out, inventory=inventory)
                else:
                    require(False, 'COLLECT_UNKNOWN_STAGE_ROLE')
        except Exception as error:
            failures[role] = dict(status='INCOMPLETE_OR_INVALID_EVIDENCE', error_type=type(error).__name__,
                                 code=str(error)[:300], partial=_partial(role_out))
    try:
        observed_accounting = accounting(jobs, stage, run=account)
        raw_accounting = observed_accounting.pop('raw')
        write_new(out/'accounting.json', observed_accounting)
        with (out/'accounting.psv').open('x') as stream:
            stream.write(raw_accounting)
        inventory += [member(out/'accounting.json'), member(out/'accounting.psv')]
    except Exception as error:
        observed_accounting = dict(status='ACCOUNTING_NOT_RECORDED', error_type=type(error).__name__,
                                   code=str(error)[:300], allocated_GPU_hours=None, exact_requested_jobs=jobs)
        write_new(out/'accounting.json', observed_accounting)
        inventory.append(member(out/'accounting.json'))
    scheduler_failures = {role: row['state'] for role, row in observed_accounting.get('jobs', {}).items()
                          if role != 'collector' and row['state'] != 'COMPLETED'}
    accounted = set(observed_accounting.get('jobs', {}))
    missing_accounting = sorted(set(manifest['registration_roles']) - accounted)
    complete = not failures and not scheduler_failures and not missing_accounting
    if complete and stage == 'qualification':
        aggregate = aggregate_qualification(methods, manifest)
        write_new(out/'qualification.json', aggregate)
        inventory.append(member(out/'qualification.json'))
    if complete and manifest['registration_roles'] == ['W0_ZSRE', 'ZSRE_SMOKE']:
        smoke = summaries['ZSRE_SMOKE']
        write_new(out/'smoke.json', dict(smoke, smokemember=member(attempt/'ZSRE_SMOKE/smoke.json')))
        inventory.append(member(out/'smoke.json'))
    result = dict(schema='official-server2-independent-collector-v1',
        status='SCIENTIFIC_COMPLETE' if complete else 'PARTIAL_OR_FAILED', stage=stage,
        code_commit=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        manifest_sha256=manifest['base_manifest_sha256'], summaries=summaries, failures=failures,
        scheduler_failures=scheduler_failures, unobserved_scientific_accounting=missing_accounting,
        accounting=observed_accounting,
        scientific_complete=complete, terminal_scheduler_success_is_not_science=True,
        CPU_collector_GPU=0, model_loads=0, checkpoint_tensor_loads=0,
        new_model_forwards=0, new_submissions=0, automatic_retry=False,
        monitoring_active=False, raw_originals_preserved=True)
    write_new(out/'result.json', result)
    inventory.append(member(out/'result.json'))
    inventory_summary = _write_inventory(out/'inventory.csv', inventory)
    report = ('# Server2 official baseline CPU 수집\n\n'
        f"상태: {result['status']}. 과학 증거와 scheduler 성공은 별도로 확인했다.\n\n"
        f"source `{manifest['code_commit']}`, official tree `{manifest['official_tree_sha256']}`.\n\n"
        f"검산된 경로 {len(summaries)}, 미완료/오류 {len(failures)}. "
        f"accounting `{observed_accounting['status']}`; 실제 관측 GPUh "
        f"{observed_accounting.get('allocated_GPU_hours')}.\n\n"
        '모델/GPU/새 평가/checkpoint tensor load/새 제출/재시도는 0이다. '
        '원 raw·실패·부분 prefix·checkpoint를 보존했다. 세부 case/raw inventory는 ignored local에만 있다.\n')
    with (out/'report-ko.md').open('x') as stream:
        stream.write(report)
    terminal_result = dict(status='COLLECTOR_COMPLETE' if complete else 'COLLECTOR_PARTIAL_OR_FAILED',
        scientific_complete=complete, result=member(out/'result.json'), inventory=member(out/'inventory.csv'),
        inventory_summary=inventory_summary, report=member(out/'report-ko.md'),
        source=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        final_atomic_terminal=True, monitoring_active=False, automatic_retry=False)
    _atomic_terminal(terminal, terminal_result)
    return terminal_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    value = collect(args.attempt)
    print(json.dumps(value, sort_keys=True))
    raise SystemExit(0 if value['scientific_complete'] else 2)


if __name__ == '__main__':
    main()
