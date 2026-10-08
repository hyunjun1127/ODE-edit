"""Stored-JSON/CPU collector fixtures: no actual model, Slurm, or GPU proof."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from official.evaluation import reduce as factual_reduce
from official.experiments.prepare import METHODS, digest, write_new
from official.runners.server2 import collect as collector


def manifest():
    return dict(code_commit='a'*40, official_tree_sha256='b'*40, base_manifest_sha256='c'*64,
        model_revision='r'*40, tokenizer_sha256='t'*64,
        runtime={'fixture':'CPU_JSON_COLLECTOR_NO_MODEL_OR_GPU'},
        streams={dataset:{'lock':{'stream_sha256':'s'*64}} for dataset in ('cf','zsre')},
        checkpoint_identities={dataset: {method: {'code_commit':'a'*40, 'method':method, 'dataset':dataset}
                              for method in METHODS} for dataset in ('cf', 'zsre')},
        native_source_sha256={method:['d'*64] for method in METHODS})


def qualification(method, value=None):
    value = manifest() if value is None else value
    return dict(status='PASS_ACTUAL_QUALIFICATION', actual_GPU=True, model='gptj', dataset='cf',
        method=method, code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
        manifest_sha256=value['base_manifest_sha256'], continuous_batches=3, resume_after_batch=2,
        resumed_batches=[3], weights_equal=True, history_equal=True, contexts_equal=True,
        rng_equal=True, metrics_equal=True, checkpoint_identity=value['checkpoint_identities']['cf'][method],
        native_owner_formula_parity=dict(path='/fixture/owner-formula-'+method+'.json', bytes=1, sha256='e'*64))


def records(count=2000):
    return [dict(case_id=90000-index*3, occurrence_index=index+1) for index in range(count)]


def factual(dataset, rows, endpoint='W20', value=None):
    from official.evaluation.factual import SCHEMA, TOKENIZATION
    value = manifest() if value is None else value
    cases = []
    for record in rows:
        case = dict(record)
        if dataset == 'cf':
            case.update(rewrite_prompts_probs=[{'target_new':1., 'target_true':2.}],
                paraphrase_prompts_probs=[{'target_new':1., 'target_true':2.}]*2,
                neighborhood_prompts_probs=[{'target_new':2., 'target_true':1.}]*3)
        else:
            case.update(rewrite_prompts_correct=[True, False], paraphrase_prompts_correct=[True],
                neighborhood_W0_agreement=[True, True], neighborhood_prompts_correct=[False, False])
        cases.append(case)
    reduced = factual_reduce.counterfact(cases) if dataset == 'cf' else factual_reduce.zsre(cases)
    identity = dict(schema=SCHEMA, dataset=dataset, tokenization=TOKENIZATION,
        cohort_sha256=digest({'CPU_query_cohort_fixture_only':rows}),
        ordered_occurrences=[row['occurrence_index'] for row in rows],
        external_identity=collector.expected_factual_external_identity(value, dataset),
        padding='RIGHT_EXPLICIT_ATTENTION_MASK', use_cache=False)
    return dict(cases=cases, summary=reduced, identity=identity, identity_sha256=digest(identity),
                requests=len(rows), endpoint=endpoint)


def owner_formula_fixture(value, method='MEMIT', dataset='cf'):
    layers = [8] if method == 'FT' else [3, 8] if method == 'ALPHAEDIT_BLUE' else [3, 4, 5, 6, 7, 8]
    plan = dict(schema='official-server2-first-trajectory-native-parity-v1',
        status='PLAN_READY_OWNER_SOURCE_FORMULA_CONTROLS', actual_GPU=False,
        scientific_quality_gate=False, after_result_tolerance_relaxation=False,
        method=method, dataset=dataset, model_revision=value['model_revision'],
        tokenizer_sha256=value['tokenizer_sha256'], stream_sha256=value['streams'][dataset]['lock']['stream_sha256'],
        source=dict(code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
            factual_sha256='f'*64, hparams_sha256='h'*64, parity_sha256='p'*64,nethook_sha256='n'*64,
            native_easy_repr_sha256='e'*64,native_BLUE_repr_sha256='b'*64,native_hparams={'layers':layers}),
        tolerance=dict(candidate_nll={'atol':1e-4, 'rtol':0.0}, affine_readout={'atol':1e-4,'rtol':1e-5}),
        independent_public_native_evaluator='NOT_AVAILABLE_IN_OFFICIAL_DISTRIBUTION',
        evidence_scope='OWNER_REVIEWED_NATIVE_FORMULA_PACKING_AND_SAME_CALL_MODEL_OPERATOR_CONTROLS')
    plan['plan_sha256'] = digest(plan)
    endpoints = list(dict.fromkeys([layers[0],layers[-1]]))
    def affine(width):
        return dict(shape=[2,width], dtype='float32', max_abs_error=0.0, atol=1e-4,rtol=1e-5,close=True)
    proof = dict(schema='official-server2-first-trajectory-native-parity-v1',
        status='PASS_ACTUAL_OWNER_FORMULA_PARITY', actual_GPU=True, method=method, dataset=dataset,
        plan_sha256=plan['plan_sha256'], code_commit=value['code_commit'],
        official_tree_sha256=value['official_tree_sha256'], factual_source_sha256='f'*64,
        native_hparams_sha256='h'*64, candidate_nll_close=True, candidate_nll_abs_error=0.0,
        candidate_token_prefix_exact=True, token_predictions_exact=True, subject_lookup_exact=True,
        observer_no_mutation=True, RNG_restored=True, extra_LM_forward_calls=0,
        target_fit_calls=0,optimizer_calls=0,writer_calls=0,existing_factual_forward_calls=10,
        native_candidate_identity_sha256='a'*64,
        fc_out={f'transformer.h.{layer}.mlp.fc_out':affine(4096) for layer in endpoints},
        readout27=affine(50400),
        block_output_schemas={f'transformer.h.{layer}':dict(container='Tensor',hidden_shape=[2,4096])
            for layer in [*endpoints,27]},
        independent_public_native_evaluator=plan['independent_public_native_evaluator'],
        independent_oracle_PASS=False,bitwise_full_evaluator_claim=False,scientific_quality_gate=False,
        evidence_scope=plan['evidence_scope'], fixture='CPU_JSON_ONLY_NO_REAL_MODEL_OR_GPU')
    value.setdefault('native_parity_plans',{}).setdefault(dataset,{})[method] = plan
    value.setdefault('source_members',{}).update({f'hparams/{method}/gptj.json':'h'*64,
        'evaluation/factual.py':'f'*64,'runners/server2/parity.py':'p'*64,
        'baselines/easyedit/util/nethook.py':'n'*64,
        'baselines/easyedit/models/rome/repr_tools.py':'e'*64,
        'baselines/blue/rome/repr_tools.py':'b'*64})
    return proof


class OfficialCollectorTests(unittest.TestCase):
    def test_qualification_six_methods_identity_and_actual_flags_required(self):
        value = manifest()
        entries = {method:qualification(method, value) for method in METHODS}
        aggregate = collector.aggregate_qualification(entries, value)
        self.assertEqual(set(aggregate['methods']), set(METHODS))
        self.assertEqual(aggregate['status'], 'PASS_ACTUAL_QUALIFICATION')
        with self.assertRaisesRegex(ValueError, 'SIX_METHOD'):
            collector.aggregate_qualification({method:entries[method] for method in METHODS[:-1]}, value)
        for field, invalid in [('actual_GPU', False), ('rng_equal', False), ('contexts_equal', False),
                               ('resumed_batches', [2, 3]), ('code_commit', 'e'*40)]:
            wrong = deepcopy(entries)
            wrong['MEMIT'][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                collector.aggregate_qualification(wrong, value)

    def test_factual_counterfact_strict_ties_request_macro_not_pair_macro(self):
        cohort = records(2)
        observed = factual('cf', cohort)
        observed['cases'][1]['paraphrase_prompts_probs'] = [{'target_new':1., 'target_true':1.}]*7
        observed['summary'] = factual_reduce.counterfact(observed['cases'])
        reduced = collector.validate_factual(observed, 'cf', cohort, manifest=manifest())
        self.assertEqual(reduced['Generalization'], 50)
        self.assertEqual(reduced['Score'], 75)
        wrong = deepcopy(observed)
        wrong['summary']['Generalization'] = 100
        with self.assertRaisesRegex(ValueError, 'RAW_REDUCTION'):
            collector.validate_factual(wrong, 'cf', cohort, manifest=manifest())
        wrong = deepcopy(observed)
        wrong['cases'].reverse()
        with self.assertRaisesRegex(ValueError, 'CASE_ORDER'):
            collector.validate_factual(wrong, 'cf', cohort, manifest=manifest())

    def test_qualification_four_actual_source_calls_not_fake_twenty_commit_count(self):
        value = manifest()
        proof = qualification('ALPHAEDIT_BLUE', value)
        identity = proof['checkpoint_identity']
        proof.update(actual_native_batch_calls=4, actual_native_request_applications=400,
            durable_B2=dict(batch=2, identity_sha256=digest(identity), sha256='e'*64),
            checkpoint=dict(batch=3, identity_sha256=digest(identity), sha256='f'*64))
        proof['native_commits'] = [dict(batch=batch, method='ALPHAEDIT_BLUE', requests=100,
            history_appends_expected=2, native_source_sha256='d'*64) for batch in (1,2,3,3)]
        proof['native_commits'][1]['checkpoint'] = proof['durable_B2']
        collector.validate_qualification_calls(proof, value, 'ALPHAEDIT_BLUE')
        for field in ('actual_native_request_applications', 'actual_native_batch_calls'):
            wrong = deepcopy(proof); wrong[field] = 20
            with self.assertRaisesRegex(ValueError, 'ACTUAL_CALL'):
                collector.validate_qualification_calls(wrong, value, 'ALPHAEDIT_BLUE')
        wrong = deepcopy(proof); wrong['native_commits'][-1]['batch'] = 4
        with self.assertRaisesRegex(ValueError, 'FOUR_SOURCE'):
            collector.validate_qualification_calls(wrong, value, 'ALPHAEDIT_BLUE')

    def test_zsre_W0_agreement_not_loc_ans_and_token_denominators(self):
        cohort = records(3)
        observed = factual('zsre', cohort)
        reduced = collector.validate_factual(observed, 'zsre', cohort, manifest=manifest())
        self.assertEqual(reduced['Specificity'], 100)
        self.assertEqual(reduced['Specificity_loc_ans'], 0)
        self.assertEqual(reduced['Efficacy'], 50)

    def test_factual_exact_external_model_source_runtime_and_native_cohort_identity(self):
        value = manifest()
        cohort = records(3)
        observed = factual('cf',cohort,value=value)
        reduced = collector.validate_factual(observed,'cf',cohort,manifest=value)
        self.assertEqual(reduced['query_cohort_sha256'],observed['identity']['cohort_sha256'])
        mutations = [('model_revision','different-model'),('tokenizer_sha256','other-tokenizer'),
                     ('stream_sha256','another-ordered-stream'),('runtime',{'different_runtime':True}),
                     ('source','e'*40)]
        for key, replacement in mutations:
            wrong = deepcopy(observed)
            wrong['identity']['external_identity'][key] = replacement
            wrong['identity_sha256'] = digest(wrong['identity'])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'EXTERNAL_MODEL_SOURCE_RUNTIME'):
                collector.validate_factual(wrong,'cf',cohort,manifest=value)
        for key, replacement in [('schema','OTHER_SCHEMA'),('dataset','zsre'),
                                 ('tokenization','different-token-definition'),('ordered_occurrences',[3,2,1]),
                                 ('use_cache',True),('cohort_sha256','MISSING_QUERY_SHA')]:
            wrong = deepcopy(observed)
            wrong['identity'][key] = replacement
            wrong['identity_sha256'] = digest(wrong['identity'])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'QUERY_COHORT_TOKENIZATION'):
                collector.validate_factual(wrong,'cf',cohort,manifest=value)
        wrong = deepcopy(observed)
        wrong['cases'][0].pop('occurrence_index')
        with self.assertRaisesRegex(ValueError,'OCCURRENCE_ORDER'):
            collector.validate_factual(wrong,'cf',cohort,manifest=value)

    def test_owner_formula_parity_real_member_plan_counts_source_and_scope_required(self):
        value = manifest()
        proof = owner_formula_fixture(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'formula.json'
            path.write_text(json.dumps(proof))
            reduced = collector.validate_owner_formula_parity(collector.member(path),value,'MEMIT')
            self.assertFalse(reduced['independent_oracle_PASS'])
            self.assertEqual(reduced['CF_original_evaluator_parity'],'NOT_ESTABLISHED_BY_OWNER_FORMULA_CONTROL')
            for key, replacement in [('actual_GPU',False),('plan_sha256','b'*64),
                    ('factual_source_sha256','c'*64),('native_hparams_sha256','d'*64),
                    ('observer_no_mutation','PENDING_EXIT_GUARD'),('token_predictions_exact',False),
                    ('extra_LM_forward_calls',1),('candidate_nll_abs_error',0.01),('independent_oracle_PASS',True),
                    ('bitwise_full_evaluator_claim',True)]:
                wrong = deepcopy(proof)
                wrong[key] = replacement
                path.write_text(json.dumps(wrong))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    collector.validate_owner_formula_parity(collector.member(path),value,'MEMIT')
            wrong = deepcopy(proof)
            wrong['readout27']['shape'] = [2,4096]
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'AFFINE_READOUT'):
                collector.validate_owner_formula_parity(collector.member(path),value,'MEMIT')
            missing = qualification('MEMIT',value)
            missing.pop('native_owner_formula_parity')
            with self.assertRaisesRegex(ValueError,'FORMULA_MEMBER_REQUIRED'):
                collector.validate_qualification(missing,value,'MEMIT')

    def test_raw_member_sha_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'raw.json'
            write_new(path, {'v':1})
            expected = collector.member(path)
            self.assertEqual(collector.verify_member(expected), path)
            path.write_text('{"v":2}')
            with self.assertRaisesRegex(ValueError, 'SHA_SIZE'):
                collector.verify_member(expected)

    def test_checkpoint_pointer_metadata_no_payload_load_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            # This is deliberately NOT a serialized tensor/checkpoint: reader
            # is constrained to metadata and must not deserialize the payload.
            (folder/'batch-20-fixture.pt').write_bytes(b'CPU_METADATA_FIXTURE')
            identity = manifest()['checkpoint_identities']['cf']['MEMIT']
            write_new(folder/'latest.json', dict(batch=20, identity_sha256=digest(identity),
                final_W20=True, file='batch-20-fixture.pt', sha256='f'*64))
            with patch.object(collector, 'file_sha', side_effect=lambda path:
                'pointer-hash' if Path(path).name == 'latest.json' else self.fail('PAYLOAD HASH FORBIDDEN')):
                result = collector.validate_checkpoint_metadata(folder, identity)
            self.assertEqual(result['tensor_loads'], 0)
            self.assertEqual(result['saved_payload_sha256'], 'f'*64)
            wrong = deepcopy(identity)
            wrong['method'] = 'ALPHAEDIT'
            with self.assertRaisesRegex(ValueError, 'IDENTITY'):
                collector.validate_checkpoint_metadata(folder, wrong)

    def test_accounting_exact_job_task_owner_only_and_step_no_double_GPU(self):
        jobs = {'MEMIT':'62001', 'collector':'62002'}
        raw = (f'62001|{collector.TASK}-qualification-MEMIT|COMPLETED|120|cpu=6,gres/gpu=1|32G|fixture|server2|\n'
               '62001.batch|batch|COMPLETED|120|cpu=6,gres/gpu=1|35G||server2|\n'
               f'62002|{collector.TASK}-qualification-collector|RUNNING|2|cpu=6|2G|fixture|server2|\n')
        argv_seen = []
        result = collector.accounting(jobs, 'qualification', owner='fixture',
            run=lambda argv: argv_seen.append(argv) or raw)
        self.assertEqual(result['allocated_GPU_seconds'], 120)
        self.assertEqual(result['step_MaxRSS'][0]['MaxRSS'], '35G')
        self.assertEqual(argv_seen[0][4], '62001,62002')
        self.assertEqual(result['other_job_queries'], 0)
        with self.assertRaisesRegex(ValueError, 'OWNER_TASK_NODE'):
            collector.accounting(jobs, 'qualification', owner='other', run=lambda _:raw)
        with self.assertRaisesRegex(ValueError, 'UNREQUESTED'):
            collector.accounting(jobs, 'qualification', owner='fixture',
                run=lambda _:raw+'62099|other|RUNNING|1|gres/gpu=1|1G|fixture|server2|\n')

    def test_accounting_cancelled_zero_allocation_and_unknown_missing_not_full_cost(self):
        jobs = {'MEMIT':'62001', 'collector':'62002'}
        raw = f'62001|{collector.TASK}-cf-MEMIT|CANCELLED by 1025|0|||fixture||\n'
        result = collector.accounting(jobs, 'cf', owner='fixture', run=lambda _:raw)
        self.assertEqual(result['jobs']['MEMIT']['state'], 'CANCELLED')
        self.assertEqual(result['jobs']['MEMIT']['allocated_gpus'], 0)
        self.assertEqual(result['status'], 'ACCOUNTING_INCOMPLETE')
        self.assertEqual(result['cost_coverage'], 'OBSERVED_SUBSET_ONLY')
        self.assertEqual(result['unobserved_roles'], ['collector'])

    def test_twenty_native_commits_nineteen_links_and_zsre_no_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = manifest()
            method, dataset = 'ALPHAEDIT_BLUE', 'zsre'
            identity = value['checkpoint_identities'][dataset][method]
            commits = []
            for batch in range(1, 21):
                path = root/f'commit-{batch}.json'
                write_new(path, dict(method=method, batch=batch, previous_batch=batch-1, requests=100,
                    history_appends_expected=2, native_source_sha256='d'*64, checkpoint_identity=identity,
                    checkpoint=dict(batch=batch, sha256='e'*64, identity_sha256=digest(identity))))
                commits.append(collector.member(path))
            endpoints = {}
            cohort = records()
            for name, count in [('W0',2000), ('W5',500), ('W10',1000), ('W15',1500), ('W20',2000)]:
                path = root/(name+'.json')
                write_new(path, factual(dataset, cohort[:count], name))
                endpoints[name] = collector.member(path)
            assets = {'streams':{'zsre':{'lock':{'stream_sha256':'s'*64}}}}
            reference_path = root/'W0-reference.json'
            write_new(reference_path, dict(evaluation=factual(dataset, cohort, 'W0')))
            cold_path = root/'READY.json'
            write_new(cold_path, dict(status='READY_COLD_W0_COMPLETE', actual_GPU=True, model='gptj',
                dataset=dataset, code_commit=value['code_commit'],
                official_tree_sha256=value['official_tree_sha256'],
                manifest_sha256=value['base_manifest_sha256'], model_revision=value['model_revision'],
                tokenizer_sha256=value['tokenizer_sha256'], stream_sha256='s'*64,
                factual=endpoints['W0'], w0_reference=collector.member(reference_path)))
            checkpoint_folder = root/'checkpoints'
            checkpoint_folder.mkdir()
            (checkpoint_folder/'final.pt').write_bytes(b'NO_TENSOR_LOAD_CPU_METADATA_FIXTURE')
            write_new(checkpoint_folder/'latest.json', dict(batch=20, final_W20=True,
                identity_sha256=digest(identity), file='final.pt', sha256='e'*64))
            row = dict(status='SCIENTIFIC_COMPLETE', actual_GPU=True, method=method, dataset=dataset,
                model='gptj', batches=20, ownstate_links=19, history_appends=40, checkpoint_identity=identity,
                commits=commits, factual_endpoints=endpoints, checkpoint_folder=str(checkpoint_folder),
                W0_READY=collector.member(cold_path),
                code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
                manifest_sha256=value['base_manifest_sha256'])
            result = collector.validate_chain(row, value, assets, cohort, method, dataset)
            self.assertEqual(result['commits'], 20)
            self.assertIsNone(result['generation'])
            wrong = deepcopy(row)
            wrong['generation_ready'] = {'path':'forbidden'}
            with self.assertRaisesRegex(ValueError, 'ZSRE_NO_GENERATION'):
                collector.validate_chain(wrong, value, assets, cohort, method, dataset)
            wrong = deepcopy(row)
            wrong['commits'] = commits[:19]
            with self.assertRaisesRegex(ValueError, 'TWENTY_COMMIT'):
                collector.validate_chain(wrong, value, assets, cohort, method, dataset)
            wrong = deepcopy(row)
            wrong['W0_READY'] = None
            with self.assertRaisesRegex(ValueError, 'COLD_W0_REQUIRED'):
                collector.validate_chain(wrong, value, assets, cohort, method, dataset)

    def test_zsre_smoke_collector_binds_W0_READY_and_one_native_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            value = manifest()
            value.update(registration_stage='zsre', registration_roles=['W0_ZSRE', 'ZSRE_SMOKE'])
            cohort = records()
            stream_path = attempt/'zsre.json'
            write_new(stream_path, cohort)
            stream = collector.member(stream_path)
            assets = {'streams':{'zsre':{'member':stream, 'lock':{'stream_sha256':stream['sha256']}}}}
            value['streams']['zsre']['lock']['stream_sha256'] = stream['sha256']
            asset_path = attempt/'assets.json'
            write_new(asset_path, assets)
            value['asset_manifest'] = collector.member(asset_path)
            w0, smoke = attempt/'W0_ZSRE', attempt/'ZSRE_SMOKE'
            w0.mkdir(); smoke.mkdir()
            w0_factual, w1_factual = w0/'W0.json', smoke/'W1.json'
            write_new(w0_factual, factual('zsre', cohort, 'W0', value))
            write_new(w1_factual, factual('zsre', cohort[:100], 'W1', value))
            reference = w0/'W0-reference.json'
            write_new(reference, dict(evaluation=factual('zsre', cohort, 'W0', value)))
            ready = dict(status='READY_COLD_W0_COMPLETE', actual_GPU=True, model='gptj', dataset='zsre',
                code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
                manifest_sha256=value['base_manifest_sha256'], model_revision=value['model_revision'],
                tokenizer_sha256=value['tokenizer_sha256'], stream_sha256=stream['sha256'],
                factual=collector.member(w0_factual), w0_reference=collector.member(reference))
            write_new(w0/'READY.json', ready)
            value['W0_zsre_ready_path'] = str(w0/'READY.json')
            identity = value['checkpoint_identities']['zsre']['MEMIT']
            checkpoints = smoke/'checkpoints'; checkpoints.mkdir()
            (checkpoints/'batch1.pt').write_bytes(b'CPU_METADATA_FIXTURE_NO_TENSOR')
            pointer = dict(batch=1, final_W20=False, sha256='e'*64,
                identity_sha256=digest(identity), file='batch1.pt')
            write_new(checkpoints/'latest.json', pointer)
            commit = smoke/'commit1.json'
            write_new(commit, dict(batch=1, previous_batch=0, method='MEMIT', requests=100,
                history_appends_expected=0, native_source_sha256='d'*64,
                checkpoint_identity=identity, checkpoint=pointer))
            smoke_receipt = dict(status='PASS_ACTUAL_SMOKE', actual_GPU=True, model='gptj', dataset='zsre',
                method='MEMIT', batches=1, code_commit=value['code_commit'],
                official_tree_sha256=value['official_tree_sha256'], manifest_sha256=value['base_manifest_sha256'],
                commits=[collector.member(commit)], checkpoint_identity=identity,
                factual_endpoints={'W0':collector.member(w0_factual), 'W1':collector.member(w1_factual)})
            write_new(smoke/'smoke.json', smoke_receipt)
            write_new(attempt/'manifest.json', value)
            jobs = {'W0_ZSRE':'63101', 'ZSRE_SMOKE':'63102', 'collector':'63103'}
            write_new(attempt/'submission.json', dict(status='SUBMISSION_HANDOFF', stage='zsre', jobs=jobs,
                source={'code_commit':value['code_commit'], 'official_tree_sha256':value['official_tree_sha256']},
                base_manifest_sha256=value['base_manifest_sha256']))
            raw = '\n'.join(f'{job}|{collector.TASK}-zsre-{role}|COMPLETED|10|'
                f'{"cpu=6" if role=="collector" else "cpu=6,gres/gpu=1"}||{__import__("getpass").getuser()}|server2|'
                for role,job in jobs.items())+'\n'
            result = collector.collect(attempt, account=lambda _:raw)
            self.assertTrue(result['scientific_complete'])
            aggregate = collector.read(attempt/'collector/smoke.json')
            self.assertEqual(aggregate['W0member'], collector.member(w0/'READY.json'))
            self.assertEqual(aggregate['smokemember'], collector.member(smoke/'smoke.json'))
            self.assertEqual(aggregate['native_request_applications'], 100)

    def test_failed_or_partial_qualifications_never_emit_aggregate_success(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            value = manifest()
            value.update(registration_stage='qualification', registration_roles=list(METHODS),
                         asset_manifest={})
            asset_path = attempt/'assets.json'
            write_new(asset_path, {'streams':{}})
            value['asset_manifest'] = collector.member(asset_path)
            write_new(attempt/'manifest.json', value)
            jobs = {method:str(63000+index) for index,method in enumerate(METHODS)}
            jobs['collector'] = '63009'
            write_new(attempt/'submission.json', dict(status='SUBMISSION_HANDOFF', stage='qualification',
                jobs=jobs, source={'code_commit':value['code_commit'],
                'official_tree_sha256':value['official_tree_sha256']},
                base_manifest_sha256=value['base_manifest_sha256']))
            for method in METHODS:
                (attempt/method).mkdir()
                write_new(attempt/method/'failure.json', dict(status='TECHNICAL_FAILURE', batch=0,
                                                            preserved_source_raw_checkpoint=True))
            raw = '\n'.join(f'{job}|{collector.TASK}-qualification-{role}|FAILED|0|||{__import__("getpass").getuser()}||'
                            for role, job in jobs.items())+'\n'
            result = collector.collect(attempt, account=lambda _:raw)
            self.assertFalse(result['scientific_complete'])
            self.assertFalse((attempt/'collector/qualification.json').exists())
            self.assertTrue((attempt/'collector/result.json').is_file())
            self.assertTrue((attempt/'collector/inventory.csv').is_file())
            self.assertTrue((attempt/'collector/report-ko.md').is_file())
            self.assertTrue((attempt/'collector/terminal.json').is_file())
            self.assertTrue(result['final_atomic_terminal'])
            self.assertTrue(collector.collect(attempt)['duplicate_collection_prevented'])


if __name__ == '__main__':
    unittest.main()
