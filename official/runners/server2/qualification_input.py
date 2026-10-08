"""Read-only, provenance-preserving qualification producer/consumer bridge.

An old actual B2->B3 proof is never rewritten as a new source's observation.
This preregistered source/physical-input projection permits its use only when
the native computational closure and fixed method/configuration remain exact.
The original independent CF oracle is separate evidence, NOT supplied by this
resume/formula proof. No old module is imported or executed by this bridge.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from official.evaluation import reduce as factual_reduce
from official.evaluation.factual import TOKENIZATION
from official.experiments.prepare import METHODS, ROOT, digest, file_sha, read
from official.runners.server2 import collect


SCHEMA = 'official-server2-provenance-preserving-qualification-input-v1'
PRODUCER_ATTEMPT = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/'
                        '20261008-r1/registration-qualification-r1')
PRODUCER_COMMIT = '18e7fbd99bc2692c9da9ebdc9f4598caba79a1f8'
PRODUCER_TREE = '7f550b1a02b29173da91372e25dd9279c334504c'
PRODUCER_MANIFEST_SHA = '8559616975d2aac5c19bd0dfc2f9f1c7bc13489aa0554838b7891aaaed87ce2f'
PRODUCER_LOCK_SHA = '84c6f1397a42de671261f8a2b52ce5acd8efa57d01102d673c6d46cb6786929e'
OLD_REDUCER_SHA = '9bfb19a42c0e3e0a0c62c3fadaf60645cb9096a9bd7595c64db003ebb08c6db0'
NEW_REDUCER_SHA = '1e18b024ac900665b2ed7f1164b1074d3d4720ad8bde626daa23aea70609731f'
SOURCE_PREFIXES = ('baselines/', 'hparams/', 'tracking/', 'evaluation/generation/')
SOURCE_FILES = ('__init__.py', 'experiments/__init__.py', 'experiments/checkpoint.py',
    'experiments/prepare.py', 'evaluation/factual.py',
    'runners/server2/native.py', 'runners/server2/parity.py',
    'runners/server2/telemetry.py', 'runners/server2/assets.py',
    'runners/server2/generation.py')
EXCLUDED_EXTRA = ('hparams/cf-native-reference.lock.json',)
RUN_FUNCTIONS = ('configuration', 'checkpoint_identity', 'load_model', 'native_apply',
                 'qualification', 'save', 'equal', 'signature', 'factual',
                 'tensor_hash', 'physical_state', 'factual_identity', 'require', 'member', 'log')
ASSET_KEYS = tuple('C0_L'+str(layer) for layer in range(3,9)) + ('projector',)
MEMBER_FIELDS = ('path', 'bytes', 'sha256')


def require(value, code):
    if not value:
        raise ValueError(code)


def _clone(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _member(path):
    return collect.member(path)


def _verified(value):
    return read(collect.verify_member(value))


def _content(value):
    require(type(value) is dict and all(key in value for key in MEMBER_FIELDS),
            'QUALIFICATION_PHYSICAL_MEMBER_REQUIRED')
    return {key: value[key] for key in ('bytes','sha256')}


def _selected(name):
    return name not in EXCLUDED_EXTRA and (name.startswith(SOURCE_PREFIXES) or name in SOURCE_FILES)


def _ast_functions(path):
    tree = ast.parse(Path(path).read_bytes())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef))]
    definitions = {node.name:node for node in nodes}
    require(len(nodes) == len(definitions), 'QUALIFICATION_DUPLICATE_RUN_FUNCTION_BINDING')
    require(all(name in definitions for name in RUN_FUNCTIONS), 'QUALIFICATION_CRITICAL_RUN_FUNCTION_MISSING')
    return {name:digest(ast.dump(definitions[name], include_attributes=False)) for name in RUN_FUNCTIONS}


def _run_bindings(path):
    tree = ast.parse(Path(path).read_bytes())
    imports, assignments = {}, {}
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            module = node.module if isinstance(node,ast.ImportFrom) else None
            level = node.level if isinstance(node,ast.ImportFrom) else 0
            for alias in node.names:
                name = alias.asname or (alias.name if module is not None else alias.name.split('.')[0])
                require(name not in imports,'QUALIFICATION_DUPLICATE_RUN_IMPORT_BINDING')
                imports[name] = dict(module=module,name=alias.name,asname=alias.asname,level=level)
        elif isinstance(node,ast.Assign):
            require(all(isinstance(target,ast.Name) for target in node.targets),
                    'QUALIFICATION_UNSUPPORTED_RUN_GLOBAL_ASSIGNMENT')
            for target in node.targets:
                assignments[target.id] = digest(ast.dump(node.value,include_attributes=False))
    definitions = {node.name for node in tree.body if isinstance(node,
        (ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    require(not (definitions & set(imports)), 'QUALIFICATION_RUN_DEFINITION_SHADOWS_IMPORT')
    require(not (definitions & set(assignments)), 'QUALIFICATION_RUN_ASSIGNMENT_SHADOWS_DEFINITION')
    return dict(imports=imports,assignments=assignments)


def _source_projection(old_attempt, producer, consumer, root):
    old = producer['source_members']
    new = consumer['source_members']
    selected_old, selected_new = {key for key in old if _selected(key)}, {key for key in new if _selected(key)}
    require(selected_old == selected_new and set(SOURCE_FILES) <= selected_old,
            'QUALIFICATION_COMPUTATIONAL_MEMBER_SET_CHANGED')
    require(all(f'hparams/{method}/gptj.json' in selected_old for method in METHODS),
            'QUALIFICATION_SIX_METHOD_CONFIG_SOURCE_REQUIRED')
    projected = {}
    old_root = old_attempt/'source/official'
    for name in sorted(selected_old):
        require(old[name] == new[name], 'QUALIFICATION_COMPUTATIONAL_SOURCE_CHANGED:'+name)
        before, after = _member(old_root/name), _member(root/name)
        require(before['sha256'] == old[name] and after['sha256'] == new[name]
            and before['bytes'] == after['bytes'], 'QUALIFICATION_COMPUTATIONAL_BYTES_CHANGED:'+name)
        projected[name] = dict(bytes=before['bytes'],sha256=before['sha256'])
    old_run, new_run = _member(old_root/'runners/server2/run.py'), _member(root/'runners/server2/run.py')
    require(old_run['sha256'] == old['runners/server2/run.py']
        and new_run['sha256'] == new['runners/server2/run.py'], 'QUALIFICATION_RUN_MEMBER_SHA')
    old_ast, new_ast = _ast_functions(old_run['path']), _ast_functions(new_run['path'])
    require(old_ast == new_ast, 'QUALIFICATION_CRITICAL_RUN_AST_CHANGED')
    old_bindings,new_bindings = _run_bindings(old_run['path']),_run_bindings(new_run['path'])
    require(all(new_bindings['imports'].get(key)==item for key,item in old_bindings['imports'].items())
        and new_bindings['assignments'] == old_bindings['assignments']
        and not (set(new_bindings['assignments']) & set(old_bindings['imports'])),
        'QUALIFICATION_CRITICAL_RUN_IMPORT_CONSTANT_BINDINGS_CHANGED')
    allowed_additional = {'oracle':dict(module='official.runners.server2',name='oracle',asname=None,level=0),
        'qualification_input':dict(module='official.runners.server2',name='qualification_input',asname=None,level=0)}
    require(all(allowed_additional.get(key)==new_bindings['imports'][key]
        for key in set(new_bindings['imports'])-set(old_bindings['imports'])),
        'QUALIFICATION_ONLY_DECLARED_FUTURE_RUN_IMPORTS')
    old_reduce, new_reduce = _member(old_root/'evaluation/reduce.py'), _member(root/'evaluation/reduce.py')
    require(old_reduce['sha256'] == old['evaluation/reduce.py'] == OLD_REDUCER_SHA
        and new_reduce['sha256'] == new['evaluation/reduce.py']
        and new_reduce['sha256'] in (OLD_REDUCER_SHA, NEW_REDUCER_SHA),
        'QUALIFICATION_ONLY_DECLARED_DISPLAY_REDUCER_CHANGE')
    return dict(members=projected, closure_sha256=digest(projected), critical_run_AST=old_ast,
        run_original_bindings=old_bindings,
        run_members=dict(producer=_content(old_run),consumer=_content(new_run)),
        display_exception=dict(producer=_content(old_reduce), consumer=_content(new_reduce),
            only_changed_field='Score_AlphaEdit_display',
            change='PYTHON_ROUND_FSUM_DISPLAY_TO_UPSTREAM_NUMPY_MEAN_AROUND_DISPLAY',
            old_summary_relabelled=False, old_summary_reduced_with_new_display=False))


def _physical_assets(manifest):
    """Closed physical projection; source provenance and path location separate."""
    require(manifest['model'] == 'gptj' and set(manifest['assets']) == set(ASSET_KEYS),
            'QUALIFICATION_GPTJ_PHYSICAL_ASSET_MAPPING')
    covariance = {}
    for index, key in enumerate(ASSET_KEYS[:-1]):
        value = manifest['assets'][key]
        fields = ('layer','count','sample_size','physical_slot','shape','dtype','stored','native_covariance')
        covariance[key] = dict(_content(value), **{field:value[field] for field in fields})
        require(value['layer'] == index+3 and value['physical_slot'] == index
            and value['shape'] == [16384,16384] and value['dtype'] == 'float32'
            and value['count'] == 54924275 and value['native_covariance'] == 'mom2_sum / count',
            'QUALIFICATION_C0_SLOT_COUNT_DTYPE')
    p = manifest['assets']['projector']
    p_fields = ('shape','dtype','slot_layers','slots','threshold','BLUE_physical_slots','BLUE_history_slots')
    projector = dict(_content(p), **{key:p[key] for key in p_fields})
    require(p['shape'] == [6,16384,16384] and p['slot_layers'] == [3,4,5,6,7,8]
        and p['threshold'] == .02 and p['BLUE_physical_slots'] == [0,5]
        and p['BLUE_history_slots'] == [0,1], 'QUALIFICATION_PROJECTOR_NATIVE_MAPPING')
    model = {}
    for item in manifest['model_assets']:
        name = Path(item['snapshot_path']).name
        require(Path(item['snapshot_path']).parent == Path(manifest['model_snapshot'])
            and Path(item['snapshot_path']).resolve() == Path(item['path']).resolve(),
            'QUALIFICATION_MODEL_LOAD_PATH_MEMBER_BINDING')
        require(name not in model, 'QUALIFICATION_DUPLICATE_MODEL_MEMBER')
        model[name] = _content(item)
    require(set(model) == {'added_tokens.json','config.json','merges.txt','pytorch_model.bin',
        'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'},
        'QUALIFICATION_MODEL_MEMBER_SET')
    streams = {}
    require(set(manifest['streams']) == {'cf','zsre'}, 'QUALIFICATION_DATASET_STREAMS')
    for dataset, stream in manifest['streams'].items():
        require(Path(stream['path']).absolute() == Path(stream['member']['path']).absolute(),
                'QUALIFICATION_STREAM_LOAD_PATH_MEMBER_BINDING')
        require(stream['member']['sha256'] == stream['lock']['stream_sha256'],
                'QUALIFICATION_STREAM_BYTES_ORDER_LOCK')
        streams[dataset] = dict(member=_content(stream['member']), lock=stream['lock'],
            lock_member=_content(stream['lock_member']), source=_content(stream['source']),
            tokenizer_audit=_content(stream['tokenizer_audit']), tokenizer_summary=stream['tokenizer_summary'])
    runtime = manifest['runtime']
    require(set(runtime) == {'members','python','python_version','torch','transformers'},
            'QUALIFICATION_RUNTIME_FIELD_SET')
    modules = {item['module']:_content(item) for item in runtime['members']}
    require(len(modules) == len(runtime['members']) and set(modules) == {'torch','transformers','tokenizers',
        'transformers.models.gptj.modeling_gptj', 'transformers.models.gpt2.tokenization_gpt2',
        'transformers.models.gpt2.tokenization_gpt2_fast','transformers.modeling_utils','transformers.masking_utils'},
        'QUALIFICATION_NATIVE_RUNTIME_MODULE_SET')
    g = manifest['generation']
    ref = _verified(g['generation_assets'])
    g_fields = ('profile','eval_seed','generation_route','schedule','scoring_versions',
                'reference_identity_sha256','reference_assets_sha256')
    generation = {field:g[field] for field in g_fields}
    generation.update(reference_manifest=_content(g['generation_assets']),
        reference_READY=_content(g['reference_READY']),
        reference_files={name:_content(item) for name,item in ref['files'].items()},
        NLTK=ref['tokenizer'], reference_versions=ref['versions'])
    require(set(ref['files']) == {'attribute_snippets.json','idf.npy','tfidf_vocab.json'}
        and g['eval_seed'] == 20261007
        and g['profile'] == 'cf-cake-native-casebatch-kv-total100-globalrng-v1'
        and g['generation_route'] == 'NATIVE_CASE_PADDED_KV_GLOBAL_RNG'
        and g['schedule'] == 'CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN',
        'QUALIFICATION_GENERATION_REFERENCE_PROFILE')
    return _clone(dict(model=manifest['model'], model_id=manifest['model_id'],
        model_identity=manifest['model_identity'],model_revision=manifest['model_revision'],
        model_members=model, tokenizer_sha256=manifest['tokenizer_sha256'],
        tokenizer_files_sha256=manifest['tokenizer_files_sha256'], covariance=covariance,
        projector=projector, streams=streams,
        runtime=dict(modules=modules,**{key:runtime[key] for key in runtime if key!='members'}),
        generation=generation))


def _config_projection(manifest):
    require(set(manifest['checkpoint_identities']) == {'cf','zsre'}, 'QUALIFICATION_CONFIG_DATASETS')
    return {dataset:{method:manifest['checkpoint_identities'][dataset][method]['config_sha256']
        for method in METHODS} for dataset in ('cf','zsre')}


def plan(old_attempt, newmanifest, *, test_only_cpu=False, source_root=None):
    """Preregister compatibility before waiting for any old actual receipt.

    Test-only plans may refer to temporary metadata fixtures, but verify can
    never promote them to an actual qualification consumer binding.
    """
    old_attempt = Path(old_attempt).absolute()
    require(not old_attempt.is_symlink() and old_attempt.is_dir(), 'QUALIFICATION_PRODUCER_DIRECTORY')
    require(test_only_cpu or old_attempt == PRODUCER_ATTEMPT, 'QUALIFICATION_ONLY_EXACT_PRODUCER_ATTEMPT')
    require(test_only_cpu or source_root is None, 'QUALIFICATION_SOURCE_ROOT_TEST_ONLY')
    root = ROOT if source_root is None else Path(source_root).absolute()
    producer = read(old_attempt/'manifest.json')
    lock = read(old_attempt/'execution.lock.json')
    members = {name:_member(old_attempt/name) for name in
               ('manifest.json','execution.lock.json','submission.json')}
    if not test_only_cpu:
        require(members['manifest.json']['sha256'] == PRODUCER_MANIFEST_SHA
            and members['execution.lock.json']['sha256'] == PRODUCER_LOCK_SHA,
            'QUALIFICATION_EXACT_ORIGINAL_MANIFEST_LOCK')
    old_assets, new_assets = _member(producer['asset_manifest']), _member(newmanifest['asset_manifest'])
    require(old_assets['sha256'] == producer['asset_manifest_sha256']
        and new_assets['sha256'] == newmanifest['asset_manifest_sha256'],
        'QUALIFICATION_PRODUCER_CONSUMER_ASSET_MANIFEST_MEMBERS')
    if not test_only_cpu:
        require(read(old_assets['path'])['assets_sha256'] == producer['assets_identity_sha256']
            and read(new_assets['path'])['assets_sha256'] == newmanifest['assets_identity_sha256'],
            'QUALIFICATION_SEPARATE_ASSET_PROVENANCE_IDENTITY')
    require(producer['code_commit'] == lock['code_commit'] == PRODUCER_COMMIT
        and producer['official_tree_sha256'] == lock['official_tree_sha256'] == PRODUCER_TREE
        and producer['owner'] == newmanifest['owner']
        and producer['owner']['server'] == 'server2'
        and producer['registration_stage'] == 'qualification'
        and producer['registration_roles'] == list(METHODS), 'QUALIFICATION_PRODUCER_SOURCE_OWNER_SCOPE')
    require(_member(lock['manifest']['path']) == lock['manifest']
        and lock['manifest']['sha256'] == members['manifest.json']['sha256']
        and _member(lock['archive']['path']) == lock['archive'], 'QUALIFICATION_PRODUCER_LOCK_MEMBERS')
    submission = read(members['submission.json']['path'])
    require(submission['status'] == 'SUBMISSION_HANDOFF' and submission['stage'] == 'qualification'
        and submission['source']['code_commit'] == PRODUCER_COMMIT
        and submission['source']['official_tree_sha256'] == PRODUCER_TREE
        and submission['base_manifest_sha256'] == producer['base_manifest_sha256']
        and submission['lock'] == members['execution.lock.json']
        and set(submission['jobs']) == set(METHODS)|{'collector'}, 'QUALIFICATION_PRODUCER_REGISTRATION_BINDING')
    projection = _source_projection(old_attempt, producer, newmanifest, root)
    physical_old, physical_new = _physical_assets(producer), _physical_assets(newmanifest)
    require(physical_old == physical_new, 'QUALIFICATION_PHYSICAL_INPUT_MODEL_ORDER_RUNTIME_CHANGED')
    configs_old, configs_new = _config_projection(producer), _config_projection(newmanifest)
    require(configs_old == configs_new, 'QUALIFICATION_NATIVE_METHOD_CONFIG_CHANGED')
    value = dict(schema=SCHEMA,status='PREREGISTERED_PROVENANCE_PRESERVING_NATIVE_COMPATIBILITY',
        test_only_cpu_fixture=bool(test_only_cpu), actual_GPU=False,
        producer=dict(attempt=str(old_attempt),code_commit=PRODUCER_COMMIT,official_tree_sha256=PRODUCER_TREE,
            base_manifest_sha256=producer['base_manifest_sha256'], members=members,
            archive=lock['archive'], assets_identity_sha256=producer['assets_identity_sha256'],
            asset_manifest=old_assets,
            checkpoint_identities=producer['checkpoint_identities'], jobs=submission['jobs']),
        consumer=dict(code_commit=newmanifest['code_commit'],official_tree_sha256=newmanifest['official_tree_sha256'],
            assets_identity_sha256=newmanifest['assets_identity_sha256'], asset_manifest=new_assets,
            checkpoint_configs=configs_new),
        computational_projection=projection, physical_inputs_sha256=digest(physical_old),
        physical_inputs=physical_old, expected_proof_path=str(old_attempt/'collector/qualification.json'),
        preserve_original_source_identity=True, old_receipts_relabelled=False,
        original_CF_oracle_evidence='NOT_ESTABLISHED_BY_RESUME_OR_OWNER_FORMULA_PROOF',
        consumer_original_CF_oracle='SEPARATE_NEW_W0_FIRST4_INPUT', no_extra_fit_or_model_load=True)
    value['plan_sha256'] = digest(value)
    return value


def _factual_original(value, producer, records):
    """Validate old request macros; retain old Python-round display unchanged."""
    identity = value['identity']
    require(value['identity_sha256'] == digest(identity)
        and identity['schema'] == 'official-factual-causal-v1' and identity['dataset'] == 'cf'
        and identity['tokenization'] == TOKENIZATION
        and re.fullmatch(r'[0-9a-f]{64}',identity.get('cohort_sha256',''))
        and identity['external_identity'] == collect.expected_factual_external_identity(producer,'cf')
        and identity['ordered_occurrences'] == [row['occurrence_index'] for row in records]
        and identity['padding'] == 'RIGHT_EXPLICIT_ATTENTION_MASK' and identity['use_cache'] is False,
        'QUALIFICATION_OLD_FACTUAL_SOURCE_IDENTITY')
    cases = value['cases']
    require(len(cases) == len(records) == 100
        and [(row['case_id'],row['occurrence_index']) for row in cases]
            == [(row['case_id'],row['occurrence_index']) for row in records],
        'QUALIFICATION_CONTINUOUS_RESUMED_CASE_ORDER')
    for row, record in zip(cases,records):
        require(len(row['rewrite_prompts_probs']) == 1
            and len(row['paraphrase_prompts_probs']) == len(record['paraphrase_prompts'])
            and len(row['neighborhood_prompts_probs']) == len(record['neighborhood_prompts']),
            'QUALIFICATION_FACTUAL_NATIVE_PROMPT_COUNTS')
    expected = factual_reduce.counterfact(cases)
    expected['Score_AlphaEdit_display'] = factual_reduce.harmonic([
        round(expected[label],2) for label in ('Efficacy','Generalization','Specificity')])
    require(set(value['summary']) == set(expected), 'QUALIFICATION_OLD_FACTUAL_SUMMARY_FIELDS')
    collect._compare_summary(value['summary'], expected)
    require(value.get('RNG_restored') is True and value.get('observer_no_mutation') is True,
            'QUALIFICATION_OLD_FACTUAL_NONMUTATION_RNG')


def verify(frozen_plan, newmanifest, path=None):
    """Pure local metadata/JSON validation, never a job/progress query."""
    value = _clone(frozen_plan)
    checksum = value.pop('plan_sha256')
    require(checksum == digest(value), 'QUALIFICATION_INPUT_PLAN_SHA')
    require(value['schema'] == SCHEMA and value['actual_GPU'] is False
        and value['preserve_original_source_identity'] is True
        and value['old_receipts_relabelled'] is False, 'QUALIFICATION_PROVENANCE_PLAN')
    test = value['test_only_cpu_fixture']
    old_attempt = Path(value['producer']['attempt'])
    reproduced = plan(old_attempt,newmanifest,test_only_cpu=test,
        source_root=Path(newmanifest['_test_source_root']) if test else None)
    require(reproduced == frozen_plan, 'QUALIFICATION_INPUT_PLAN_OR_SOURCE_CHANGED')
    expected = Path(value['expected_proof_path'])
    require(path is None or Path(path).absolute() == expected, 'QUALIFICATION_EXACT_EXPECTED_PROOF_PATH')
    binding = dict(schema=SCHEMA,plan_sha256=checksum,
        producer_code_commit=value['producer']['code_commit'],
        producer_official_tree_sha256=value['producer']['official_tree_sha256'],
        producer_assets_identity_sha256=value['producer']['assets_identity_sha256'],
        consumer_code_commit=newmanifest['code_commit'],consumer_official_tree_sha256=newmanifest['official_tree_sha256'],
        consumer_assets_identity_sha256=newmanifest['assets_identity_sha256'],
        consumer_manifest_sha256=newmanifest.get('base_manifest_sha256'),
        native_closure_sha256=value['computational_projection']['closure_sha256'],
        physical_inputs_sha256=value['physical_inputs_sha256'], original_receipts_relabelled=False,
        CF_original_evaluator_parity='NOT_ESTABLISHED_BY_THIS_INPUT',
        additional_native_fit_calls=0, checkpoint_tensor_loads=0, source_modules_dynamically_imported=0)
    if not expected.exists():
        return dict(status='INPUT_PENDING_NOT_ACTUAL_QUALIFIED', actual_GPU=False,
            expected_proof_path=str(expected), consumer_binding=binding,
            original_evidence_NOT_OBSERVED=True, recurring_monitor=False)
    aggregate_member = _member(expected)
    aggregate = read(expected)
    producer = _verified(value['producer']['members']['manifest.json'])
    require(aggregate['status'] == 'PASS_ACTUAL_QUALIFICATION' and aggregate['actual_GPU'] is True
        and aggregate['actual_model_proofs_not_CPU_fixtures'] is True
        and set(aggregate['methods']) == set(METHODS), 'QUALIFICATION_SIX_ACTUAL_PRODUCER_METHODS')
    collect.receipt_identity(aggregate,producer)
    if not test:
        require(not aggregate.get('fixture') and aggregate.get('test_only_cpu') is not True,
                'QUALIFICATION_CPU_FIXTURE_NOT_PRODUCTION_EVIDENCE')
    inventory = [aggregate_member]
    records = _verified(producer['streams']['cf']['member'])[200:300]
    methods = {}
    for method in METHODS:
        method_member = _member(old_attempt/method/'qualification.json')
        inventory.append(method_member)
        proof = read(method_member['path'])
        require(not proof.get('fixture') or test, 'QUALIFICATION_CPU_FIXTURE_NOT_PRODUCTION_EVIDENCE')
        collect.validate_qualification(proof,producer,method)
        collect.validate_qualification_calls(proof,producer,method)
        owner = collect.validate_owner_formula_parity(proof['native_owner_formula_parity'],producer,method,
            inventory=inventory)
        continuous = _verified(proof['continuous_metric'])
        resumed = _verified(proof['resumed_metric'])
        inventory += [proof['continuous_metric'],proof['resumed_metric']]
        _factual_original(continuous,producer,records)
        _factual_original(resumed,producer,records)
        require(continuous['cases'] == resumed['cases'] and continuous['summary'] == resumed['summary'],
                'QUALIFICATION_ORIGINAL_CONTINUOUS_RESUMED_RAW_EQUALITY')
        checkpoint = collect.validate_checkpoint_metadata(old_attempt/method/'checkpoints',
            proof['checkpoint_identity'],final_batch=3,inventory=inventory)
        require(checkpoint['saved_payload_sha256'] == proof['checkpoint']['sha256'],
                'QUALIFICATION_ORIGINAL_B3_CHECKPOINT_POINTER')
        aggregated = aggregate['methods'][method]
        require(all(aggregated.get(key) == item for key,item in proof.items()),
                'QUALIFICATION_AGGREGATE_ORIGINAL_PROOF_CHANGED')
        methods[method] = dict(member=method_member,source=proof['code_commit'],
            checkpoint_identity=proof['checkpoint_identity'],owner_formula=owner,
            continuous=proof['continuous_metric'],resumed=proof['resumed_metric'],
            durable_B2=proof['durable_B2'],checkpoint=proof['checkpoint'],
            actual_native_batch_calls=4,actual_native_request_applications=400,
            history_appends_per_batch=collect.HISTORY_COUNTS[method]//20)
    result_member = _member(old_attempt/'collector/result.json')
    result = read(result_member['path'])
    collect.receipt_identity(result,producer)
    require(result['status'] == 'SCIENTIFIC_COMPLETE' and result['scientific_complete'] is True
        and result['stage'] == 'qualification' and result['failures'] == {}
        and result['scheduler_failures'] == {} and result['unobserved_scientific_accounting'] == []
        and result['accounting']['exact_requested_jobs'] == value['producer']['jobs']
        and all(result['accounting']['jobs'][method]['state'] == 'COMPLETED'
            for method in METHODS), 'QUALIFICATION_PRODUCER_COLLECTOR_COMPLETION')
    terminal_member = _member(old_attempt/'collector/terminal.json')
    terminal = read(terminal_member['path'])
    require(terminal['status'] == 'COLLECTOR_COMPLETE' and terminal['scientific_complete'] is True
        and terminal['source'] == PRODUCER_COMMIT and terminal['official_tree_sha256'] == PRODUCER_TREE
        and terminal['result'] == result_member and terminal['final_atomic_terminal'] is True,
        'QUALIFICATION_PRODUCER_ATOMIC_TERMINAL')
    inventory += [result_member,terminal_member]
    unique_members = {item['path']:{key:item[key] for key in MEMBER_FIELDS} for item in inventory}
    ordered_members = [unique_members[path] for path in sorted(unique_members)]
    binding.update(producer_aggregate=aggregate_member,producer_result=result_member,
        producer_terminal=terminal_member,methods=methods,original_source_proof_preserved=True)
    return dict(status='VERIFIED_CPU_FIXTURE_INPUT_NOT_ACTUAL' if test else
                'VERIFIED_PRODUCER_QUALIFICATION_INPUT',actual_GPU=not test,
        producer_aggregate=aggregate_member,producer_proof=aggregate,consumer_binding=binding,
        producer_status=aggregate['status'],producer_original_evaluator_parity='NOT_ESTABLISHED_BY_RESUME',
        proof_member_sha256=digest(ordered_members),proof_member_count=len(ordered_members),
        no_extra_fit=True, recurring_monitor=False)
