"""Future-only GPT-J W0 first-four independent native CounterFact comparison.

All scoring/forwards belong to the unchanged shared original-source adapter.
This connector locks a pre-measurement plan, physical state and exact canonical
member. It never re-identifies old observations, supplies canonical logits to
the oracle, changes tolerances, fits, generates or edits a model. Receipts with
tokens/cases/state pointers are LOCAL ONLY; ``compact`` is the scalar report.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from official.evaluation import cf_native_reference as reference
from official.evaluation import factual
from official.experiments.prepare import digest, file_sha, read, ROOT


SCHEMA = 'official-server2-cold-w0-native-oracle-v1'
INSTRUCTION = 'GH-SH2-OFFICIAL-NATIVE-ORACLE-READY-20261009-R1'
PARENT = 'USER-OFFICIAL-BASELINES-20261008-R1'
SHARED_COMMIT = '34001ec0950f00b61e89be753494f40b9da6f70f'
SHARED_SHA256 = '0473673abf92e483b19d534743278ee33177ce22e66c059e9788e40ba7e12216'
LOCK_SHA256 = 'e8f540ee60db8f2343bca99307f1ebaca51c11cf31cfe39c05bf3d808c5199ae'
MODEL_REVISION = '47e169305d2e8376be1d31e765533382721b2cc1'
LOCK = ROOT / 'hparams/cf-native-reference.lock.json'
NATIVE_MODULES = (
    'official.baselines.easyedit.models.ft.ft_main',
    'official.baselines.easyedit.models.memit.memit_main',
    'official.baselines.easyedit.models.alphaedit.AlphaEdit_main',
    'official.baselines.blue.AlphaEdit.AlphaEdit_main',
    'official.baselines.easyedit.models.memit_FE.memit_FE_main',
    'official.baselines.easyedit.models.SPHERE.SPHERE_main',
)
NATIVE_FIELDS = ('CONTEXT_TEMPLATES_CACHE', 'COV_CACHE', 'cache_c', 'cache_c_new',
                 'P', 'P_loaded')
HOOK_FIELDS = ('_forward_pre_hooks', '_forward_hooks', '_backward_hooks',
               '_forward_hooks_with_kwargs', '_forward_pre_hooks_with_kwargs',
               '_forward_hooks_always_called')


def require(value, code):
    if not value:
        raise ValueError(code)


def _clone(value):
    """Reject non-JSON/nonfinite identity; never serialize tensor contents."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError('ORACLE_NON_JSON_IDENTITY') from error


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'ORACLE_REGULAR_MEMBER')
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def _read_member(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
            'ORACLE_MEMBER_SCHEMA')
    require(member(value['path']) == value, 'ORACLE_MEMBER_CHANGED')
    return read(value['path'])


def _source_member(path):
    """Logical official member is stable across frozen-archive extraction."""
    value = member(path)
    value['path'] = 'official/' + str(Path(path).resolve().relative_to(ROOT.resolve()))
    return value


def _sources():
    require(file_sha(reference.__file__) == SHARED_SHA256, 'ORACLE_SHARED_SOURCE_CHANGED')
    require(file_sha(LOCK) == LOCK_SHA256, 'ORACLE_SHARED_LOCK_CHANGED')
    require(member(reference.SOURCE_PATH)['bytes'] == reference.SOURCE_BYTES
            and file_sha(reference.SOURCE_PATH) == reference.SOURCE_SHA256,
            'ORACLE_ORIGINAL_SOURCE_CHANGED')
    return dict(shared_commit=SHARED_COMMIT, shared_adapter=_source_member(reference.__file__),
        lock=_source_member(LOCK), original=_source_member(reference.SOURCE_PATH),
        task_adapter=_source_member(__file__), canonical_factual=_source_member(factual.__file__),
        original_commit=reference.UPSTREAM_COMMIT)


def external_identity(manifest):
    """EXACT existing runner factual_identity; no endpoint/state additions."""
    return _clone(dict(model_revision=manifest['model_revision'],
        tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams']['cf']['lock']['stream_sha256'],
        runtime=manifest['runtime'], source=manifest['code_commit']))


def _cohort(records):
    require(type(records) is list and len(records) == 4, 'ORACLE_FIRST_FOUR_ONLY')
    require([row.get('occurrence_index') for row in records] == [1, 2, 3, 4],
            'ORACLE_FIRST_ORDERED_OCCURRENCES')
    require(all(type(row.get('case_id')) is int for row in records), 'ORACLE_CASE_IDENTITY')
    return [dict(case_id=row['case_id'], occurrence_index=row['occurrence_index'])
            for row in records]


def plan(manifest, records4):
    """Persist BEFORE fresh canonical first4 and native original observations.

    No case IDs/text/tokens appear in the plan; the exact local cohort and full
    records are hash-bound. Actual GPU PASS cannot be produced by this function.
    """
    require(manifest['model'] == 'gptj' and manifest['owner']['server'] == 'server2'
            and manifest['model_revision'] == MODEL_REVISION, 'ORACLE_GPTJ_SERVER2_BINDING')
    locked = _cohort(records4)
    value = dict(schema=SCHEMA, instruction_id=INSTRUCTION, parent=PARENT,
        status='PREREGISTERED_BEFORE_CANONICAL_AND_NATIVE_W0', actual_GPU=False,
        evidence_scope=reference.SMOKE_SCOPE,
        full_2k_scope='NOT_APPLICABLE_GPTJ; NOT_OBSERVED',
        model='gptj', model_revision=MODEL_REVISION,
        model_snapshot=manifest['model_snapshot'],
        assets_identity_sha256=manifest['assets_identity_sha256'],
        tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams']['cf']['lock']['stream_sha256'],
        execution=dict(code_commit=manifest['code_commit'],
            official_tree_sha256=manifest['official_tree_sha256']),
        source=_sources(), existing_external_identity=external_identity(manifest),
        cohort=dict(selection='FIXED_FIRST_FOUR_ORDERED_OCCURRENCES', requests=4,
            ordered_occurrences=[1, 2, 3, 4], locked_cohort_sha256=digest(locked),
            records_sha256=digest(_clone(records4))),
        logical_state=dict(endpoint='W0', actual_applied_edits=0,
            target_fit_calls=0, optimizer_calls=0, solve_calls=0, history_appends=0),
        physical_state_lock='CAPTURE_BEFORE_CANONICAL_FIRST4; REQUIRE_EXACT_BEFORE_AFTER',
        tolerances=dict(nll_abs_nats=reference.NLL_ABS_TOL_NATS,
            nll_relative_to_native=reference.NLL_REL_TOL, strict_booleans='EXACT',
            aggregate_abs=reference.AGGREGATE_ABS_TOL),
        independent_original_forward=True, native_forward_calls=4,
        canonical_observation='FRESH_FIRST4_SAME_COLD_MODEL; ORIGINAL_IDENTITY_UNCHANGED',
        extra_fit_edit_generation_calls=0, after_result_tolerance_relaxation=False,
        checkpoint_resume_proof_replaced=False, scientific_performance_promotion=False,
        raw_local_only=True)
    value['plan_sha256'] = digest(value)
    return value


def _verify_plan(value, manifest=None, records=None):
    checked = _clone(value)
    claimed = checked.pop('plan_sha256', None)
    require(claimed == digest(checked), 'ORACLE_PLAN_SHA256')
    require(checked['schema'] == SCHEMA and checked['instruction_id'] == INSTRUCTION
        and checked['parent'] == PARENT
        and checked['status'] == 'PREREGISTERED_BEFORE_CANONICAL_AND_NATIVE_W0'
        and checked['actual_GPU'] is False
        and checked['evidence_scope'] == reference.SMOKE_SCOPE
        and checked['logical_state'] == dict(endpoint='W0', actual_applied_edits=0,
            target_fit_calls=0, optimizer_calls=0, solve_calls=0, history_appends=0),
        'ORACLE_PLAN_SCOPE')
    require(checked['source'] == _sources(), 'ORACLE_PREREGISTERED_SOURCE_CHANGED')
    require(checked['tolerances'] == dict(nll_abs_nats=reference.NLL_ABS_TOL_NATS,
        nll_relative_to_native=reference.NLL_REL_TOL, strict_booleans='EXACT',
        aggregate_abs=reference.AGGREGATE_ABS_TOL), 'ORACLE_FIXED_TOLERANCES')
    require(checked['cohort']['requests'] == 4
        and checked['cohort']['ordered_occurrences'] == [1, 2, 3, 4], 'ORACLE_PLAN_COHORT')
    if manifest is not None and records is not None:
        require(plan(manifest, records) == value, 'ORACLE_EXECUTION_PLAN_CHANGED')


def _tensor_metadata(value):
    # Pointer/version metadata is sufficient for a no-mutation guard, NOT a
    # weight-content checksum or proof of W0 by itself. No copy/device sync.
    return dict(object_id=id(value), pointer=value.data_ptr(), version=value._version,
        shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
        requires_grad=value.requires_grad)


def _native_value(value):
    if hasattr(value, 'data_ptr') and hasattr(value, '_version'):
        return dict(kind='TENSOR_METADATA_ONLY', **_tensor_metadata(value))
    if type(value) is dict:
        return dict(kind='DICT', entries=[dict(key=repr(key), value=_native_value(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))])
    if type(value) in (tuple, list):
        return dict(kind=type(value).__name__, values=[_native_value(item) for item in value])
    return dict(kind='JSON_VALUE', value=_clone(value))


def _native_globals():
    """Inspect already-loaded official caches only; import/generation/load zero."""
    result = {}
    for name in NATIVE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            result[name] = dict(loaded=False)
        else:
            result[name] = dict(loaded=True, source_sha256=file_sha(module.__file__),
                state={key: _native_value(getattr(module, key)) if hasattr(module, key)
                       else dict(kind='ABSENT') for key in NATIVE_FIELDS})
    return result


def _tokenizer_metadata(tok):
    fields = ('name_or_path', 'padding_side', 'truncation_side', 'pad_token_id',
              'bos_token_id', 'eos_token_id', 'add_bos_token', 'add_eos_token',
              'model_max_length', 'init_kwargs', 'special_tokens_map')
    # Match the shared reference's complete tokenizer metadata without touching
    # settings or recording the potentially large vocabulary/backend in raw.
    value = {key: str(getattr(tok, key, None)) for key in fields}
    if callable(getattr(tok, 'get_vocab', None)):
        value['vocab_sha256'] = digest(tok.get_vocab())
    backend = getattr(tok, 'backend_tokenizer', None)
    if backend is not None and callable(getattr(backend, 'to_str', None)):
        value['backend_sha256'] = digest(backend.to_str())
    return value


def capture_state(model, tok, native_state_callback=None):
    """Take BEFORE canonical observation and compare it again afterward.

    The optional callback supplies own-engine/context state in addition to the
    physical tensor/hooks/config and already-loaded native global caches. Cold
    W0 without an engine is explicitly represented, not an endpoint-only guard.
    Metadata/pointers are local proof, not checkpoint or full weight hashes.
    """
    config = model.config.to_dict() if hasattr(model.config, 'to_dict') else vars(model.config)
    modules = {}
    for name, module in model.named_modules():
        modules[name] = dict(training=module.training, hooks={field:
            [[key, id(hook)] for key, hook in getattr(module, field, {}).items()]
            for field in HOOK_FIELDS})
    native = None if native_state_callback is None else _clone(native_state_callback())
    if native is not None:
        require(type(native) is dict and bool(native), 'ORACLE_NATIVE_CALLBACK_SCHEMA')
        for key in ('batch', 'completed_batch', 'actual_applied_edits'):
            if key in native:
                require(type(native[key]) is int and native[key] == 0,
                        'ORACLE_NATIVE_CALLBACK_NOT_COLD_W0')
    return _clone(dict(schema='official-server2-oracle-physical-state-v1',
        tensors={kind+':'+name: _tensor_metadata(value)
            for kind, iterator in (('parameter', model.named_parameters()),
                                   ('buffer', model.named_buffers()))
            for name, value in iterator}, modules=modules, config=deepcopy(config),
        tokenizer=_tokenizer_metadata(tok), native_globals=_native_globals(),
        native_engine_state=native,
        native_engine_binding='NO_ENGINE_AT_COLD_W0' if native is None else 'CALLER_STATE_AND_CONTEXT'))


def _canonical(value, bound_member, frozen_plan):
    stored = _read_member(bound_member)
    require(_clone(value) == stored, 'ORACLE_CANONICAL_VALUE_MEMBER_MISMATCH')
    identity = stored['identity']
    require(identity['external_identity'] == frozen_plan['existing_external_identity'],
            'ORACLE_CANONICAL_EXTERNAL_IDENTITY_CHANGED')
    require(identity['dataset'] == 'cf' and len(stored['cases']) == 4
        and [row['occurrence_index'] for row in stored['cases']] == [1, 2, 3, 4],
        'ORACLE_CANONICAL_FIRST_FOUR')
    require(stored['identity_sha256'] == digest(identity), 'ORACLE_CANONICAL_IDENTITY_SHA')
    return stored


def compare(model, tok, manifest, records4, canonical, *, frozen_plan,
            canonical_member, state_identity, state_callback=None, test_only_cpu=False):
    """Run only the exact shared original scorer and retain typed failures.

    ``state_identity`` MUST have been capture_state BEFORE fresh canonical
    first4. ``state_callback`` is the same optional native engine callback used
    by that capture. Production never selects a test route or relaxes gates.
    """
    _verify_plan(frozen_plan, manifest, records4)
    canonical = _canonical(canonical, canonical_member, frozen_plan)
    require(type(state_identity) is dict and state_identity.get('schema')
        == 'official-server2-oracle-physical-state-v1', 'ORACLE_PRE_CANONICAL_PHYSICAL_STATE_REQUIRED')
    current = lambda: capture_state(model, tok, state_callback)
    require(current() == state_identity, 'ORACLE_CANONICAL_MODEL_TOKENIZER_NATIVE_STATE_MUTATED')
    locked_cohort = _cohort(records4)
    model_binding = dict(model='gptj', revision=manifest['model_revision'],
        snapshot=manifest['model_snapshot'], payload_identity=manifest['assets_identity_sha256'],
        runtime=manifest['runtime'], config_sha256=digest(state_identity['config']),
        physical_state_sha256=digest(state_identity))
    tokenizer_binding = dict(source_sha256=manifest['tokenizer_sha256'],
        observed_signature_sha256=digest(state_identity['tokenizer']))
    scope = reference.CPU_SCOPE if test_only_cpu else reference.SMOKE_SCOPE
    result = reference.compare_native_counterfact(model, tok, records4, canonical,
        identity=frozen_plan['existing_external_identity'], evidence_scope=scope,
        locked_cohort=locked_cohort, state_callback=current,
        test_only_cpu=test_only_cpu, model_identity=model_binding,
        tokenizer_identity=tokenizer_binding, state_identity=state_identity)
    require(current() == state_identity, 'ORACLE_ORIGINAL_MODEL_TOKENIZER_NATIVE_STATE_MUTATED')
    require(member(canonical_member['path']) == canonical_member,
            'ORACLE_ORIGINAL_CANONICAL_MEMBER_MUTATED')
    require(digest(_clone(canonical)) == digest(_read_member(canonical_member)),
            'ORACLE_ORIGINAL_CANONICAL_PAYLOAD_MUTATED')
    status = 'PASS_ACTUAL_GPU_SMOKE' if result['status'] == 'PASS' else result['status']
    value = dict(schema=SCHEMA, instruction_id=INSTRUCTION, status=status,
        actual_GPU=not test_only_cpu and result['work']['forward_calls'] > 0,
        test_only_cpu_fixture=bool(test_only_cpu), plan_sha256=frozen_plan['plan_sha256'],
        binding=dict(canonical_member=_clone(canonical_member),
            canonical_payload_sha256=digest(canonical),
            canonical_identity_sha256=canonical['identity_sha256'],
            existing_external_identity_sha256=digest(frozen_plan['existing_external_identity']),
            model=model_binding, tokenizer=tokenizer_binding,
            physical_state_sha256=digest(state_identity),
            locked_cohort_sha256=digest(locked_cohort), source=frozen_plan['source']),
        evidence_scope=scope, evidence=_clone(result['evidence']),
        full_2k_scope='NOT_APPLICABLE_GPTJ; NOT_OBSERVED',
        shared_original_result=result, work=_clone(result['work']),
        no_fit_edit_generation=True, checkpoint_resume_proof_replaced=False,
        scientific_performance_promotion=False, raw_local_only=True)
    value['receipt_sha256'] = digest(value)
    verify(frozen_plan, value, canonical_member)
    return value


def verify(frozen_plan, receipt, canonical_member):
    """CPU-only source/member/scope verification; not a second observation."""
    _verify_plan(frozen_plan)
    value = _clone(receipt)
    claimed = value.pop('receipt_sha256', None)
    require(claimed == digest(value), 'ORACLE_RECEIPT_SHA256')
    require(value['schema'] == SCHEMA and value['instruction_id'] == INSTRUCTION
        and value['plan_sha256'] == frozen_plan['plan_sha256'], 'ORACLE_RECEIPT_PLAN_BINDING')
    binding, observed = value['binding'], value['shared_original_result']
    require(binding['source'] == frozen_plan['source']
        and binding['canonical_member'] == canonical_member
        and binding['locked_cohort_sha256'] == frozen_plan['cohort']['locked_cohort_sha256'],
        'ORACLE_RECEIPT_MEMBER_SOURCE_COHORT')
    canonical = _canonical(_read_member(canonical_member), canonical_member, frozen_plan)
    require(binding['canonical_payload_sha256'] == digest(canonical)
        and binding['canonical_identity_sha256'] == canonical['identity_sha256']
        and binding['existing_external_identity_sha256']
            == digest(frozen_plan['existing_external_identity']), 'ORACLE_RECEIPT_CANONICAL_SHA')
    expected_scope = reference.CPU_SCOPE if value['test_only_cpu_fixture'] else reference.SMOKE_SCOPE
    require(value['evidence_scope'] == observed['evidence_scope'] == expected_scope
        and value['evidence'] == observed['evidence'], 'ORACLE_EVIDENCE_SCOPE')
    expected_status = 'PASS_ACTUAL_GPU_SMOKE' if observed['status'] == 'PASS' else observed['status']
    require(value['status'] == expected_status and observed['status'] in
        ('PASS', 'CPU_FIXTURE_PASS', 'MISMATCH', 'NOT_QUALIFIED'), 'ORACLE_RECEIPT_STATUS')
    require(('native' in observed) == (observed['status'] != 'NOT_QUALIFIED'),
            'ORACLE_COMPARED_STATUS_REQUIRES_ORIGINAL_PROOF')
    expected_gpu = not value['test_only_cpu_fixture'] and observed['work']['forward_calls'] > 0
    require(value['actual_GPU'] is expected_gpu
        and (observed['status'] != 'CPU_FIXTURE_PASS' or value['test_only_cpu_fixture'])
        and (observed['status'] != 'PASS' or not value['test_only_cpu_fixture']),
        'ORACLE_CPU_NOT_ACTUAL_GPU_PASS')
    require(value['full_2k_scope'] == 'NOT_APPLICABLE_GPTJ; NOT_OBSERVED'
        and all(observed['evidence'][scope] == 'NOT_OBSERVED' for scope in
                (reference.CPU_SCOPE, reference.SMOKE_SCOPE, reference.FULL_SCOPE, reference.MATCHED_SCOPE)
                if scope != expected_scope)
        and observed['evidence'].get(expected_scope) ==
            ('PASS' if observed['status'] in ('PASS', 'CPU_FIXTURE_PASS') else
             'FAIL' if observed['status'] == 'MISMATCH' else 'NOT_OBSERVED'),
        'ORACLE_NO_SCOPE_PROMOTION')
    if 'native' in observed:
        native = observed['native']
        identity = native['identity']
        require(native['identity_sha256'] == digest(identity)
            and identity['external_identity'] == frozen_plan['existing_external_identity']
            and identity['source_commit'] == reference.UPSTREAM_COMMIT
            and identity['source_sha256'] == reference.SOURCE_SHA256
            and identity['source_bytes'] == reference.SOURCE_BYTES
            and identity['original_function'] == 'test_batch_prediction'
            and identity['locked_cohort_sha256'] == binding['locked_cohort_sha256']
            and identity['ordered_occurrences'] == [1, 2, 3, 4]
            and native['canonical_payload_sha256'] == binding['canonical_payload_sha256'],
            'ORACLE_ORIGINAL_IDENTITY')
        original_binding = identity['reference_binding']
        state = original_binding['state_identity']
        require(original_binding['model_identity'] == binding['model']
            and original_binding['tokenizer_identity'] == binding['tokenizer']
            and digest(state) == binding['physical_state_sha256']
            and binding['model']['physical_state_sha256'] == binding['physical_state_sha256']
            and binding['model']['model'] == frozen_plan['model']
            and binding['model']['revision'] == frozen_plan['model_revision']
            and binding['model']['snapshot'] == frozen_plan['model_snapshot']
            and binding['model']['payload_identity'] == frozen_plan['assets_identity_sha256']
            and binding['model']['runtime'] == frozen_plan['existing_external_identity']['runtime']
            and binding['tokenizer']['source_sha256'] == frozen_plan['tokenizer_sha256']
            and binding['model']['config_sha256'] == digest(state['config'])
            and binding['tokenizer']['observed_signature_sha256']
                == digest(state['tokenizer'])
            and state['schema'] == 'official-server2-oracle-physical-state-v1'
            and bool(state['tensors']) and bool(state['modules'])
            and set(state['native_globals']) == set(NATIVE_MODULES),
            'ORACLE_SEPARATE_PHYSICAL_BINDINGS')
        require(observed['canonical_identity_sha256'] == canonical['identity_sha256']
            and observed['tolerances'] == frozen_plan['tolerances']
            and native['model_no_mutation'] is True and native['RNG_restored'] is True
            and native['checkpoint_saved'] is False and native['raw_local_only'] is True
            and digest([dict(case_id=row['case_id'], occurrence_index=row['occurrence_index'])
                       for row in native['cases']]) == binding['locked_cohort_sha256'],
            'ORACLE_ORIGINAL_FIXED_GUARDS')
        require(observed['work']['forward_calls'] == 4
            and identity['native_device'] == ('TEST_ONLY_CPU_CUDA_TRANSPORT_BRIDGE'
                if value['test_only_cpu_fixture'] else 'cuda'),
            'ORACLE_ACTUAL_ORIGINAL_FORWARDS')
        if observed['status'] in ('PASS', 'CPU_FIXTURE_PASS'):
            require(observed['mismatches'] == [] and observed['display_mismatches'] == [],
                    'ORACLE_PASS_WITH_MISMATCHES')
    require(value['work'] == observed['work'] and value['raw_local_only'] is True
        and value['scientific_performance_promotion'] is False
        and value['checkpoint_resume_proof_replaced'] is False, 'ORACLE_RAW_AND_QUALIFICATION_BOUNDARY')
    return compact(receipt)


def compact(receipt):
    """Only public scalar/hash metadata; no raw IDs/text/tokens/state/pointers."""
    return dict(schema=SCHEMA, status=receipt['status'], actual_GPU=receipt['actual_GPU'],
        evidence_scope=receipt['evidence_scope'], evidence=deepcopy(receipt['evidence']),
        plan_sha256=receipt['plan_sha256'], receipt_sha256=receipt['receipt_sha256'],
        canonical_member_sha256=receipt['binding']['canonical_member']['sha256'],
        physical_state_sha256=receipt['binding']['physical_state_sha256'],
        original_forward_calls=receipt['work']['forward_calls'],
        candidate_sequences=receipt['work']['candidate_sequences'],
        seconds=receipt['work']['seconds'], full_2k_scope=receipt['full_2k_scope'],
        checkpoint_resume_proof_replaced=False, scientific_performance_promotion=False)
