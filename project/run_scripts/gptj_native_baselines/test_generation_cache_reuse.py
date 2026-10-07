"""Bounded CPU provenance fixtures, not actual GPU qualification or real generation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.experiment_generation_eval.common import case_seed, digest, immutable_write
from . import generation_cache_reuse as reuse
from . import generation_cache_qualification as qualification_contract


def member(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generation-reuse-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [dict(case_id=9000 - ordinal, generation_prompts=['fixture'],
            requested_rewrite={'relation_id': 'r', 'target_new': {'id': 't', 'str': 'new'}})
            for ordinal in range(2000)]
        asset = self.root / 'fixture-model-asset.json'
        immutable_write(asset, {'fixture_only': True})
        self.config = dict(model='fixture-gptj', model_revision='fixture', model_assets=[member(asset)],
            runtime={'torch': 'fixture', 'transformers': 'fixture', 'members': []},
            observer_identity={'sha256': 'b' * 64}, seed=20261002,
            arm_configs={'BASE_MEMIT': {'native': {'fit_evaluations': 25}}}, generation=dict(
                schema=reuse.SCHEMA, profile=reuse.PROFILE, eval_seed=reuse.EVAL_SEED,
                source_sha=reuse.OLD_SOURCE, reference_assets_sha256='c' * 64,
                scoring_versions={'nltk': 'fixture'}, W0_state_identity={'full_cold': 'fixture'}))
        self.config['generation']['model_identity'] = digest(dict(model=self.config['model'],
            revision=self.config['model_revision'], model_assets=self.config['model_assets'],
            runtime=self.config['runtime'], scorer=self.config['observer_identity'],
            seed=self.config['seed'], precision='FP32/eager/TF32off/autocastoff'))
        manifest = self.root / 'fixture-reference.json'
        immutable_write(manifest, {'identity_sha256': 'c' * 64, 'versions': {'nltk': 'fixture'}})
        self.config['generation']['generation_assets'] = member(manifest)
        immutable_write(self.root / 'config.json', self.config)
        source = self.root / 'fixture-source.json'
        immutable_write(source, {'source_fixture': True})
        immutable_write(self.root / 'execution.lock.json', dict(config_sha256=member(self.root / 'config.json')['sha256'],
            source_commit=reuse.OLD_TASK_SOURCE, shared_generation_source=reuse.OLD_SOURCE,
            source_members=[member(source)], runtime_sources=[]))
        self.runtime = dict(schema=reuse.SCHEMA, profile=reuse.PROFILE, eval_seed=reuse.EVAL_SEED,
            model_identity=self.config['generation']['model_identity'], generation_source_sha=reuse.OLD_SOURCE,
            reference_assets_sha256='c' * 64, route=reuse.OLD_ROUTE)
        self.raw_root = self.root / 'BASE_MEMIT' / 'generation-raw'
        immutable_write(self.raw_root / 'observer-identity.json',
                        {'identity': self.runtime, 'identity_sha256': digest(self.runtime)})

    def raw(self, occurrence=1):
        record = self.records[occurrence - 1]
        identity = dict(runtime=digest(self.runtime), state_identity={'full_cold': 'fixture'},
                        record_identity=reuse.record_identity(record, occurrence))
        identity['state_identity'] = copy.deepcopy(self.config['generation']['W0_state_identity'])
        observation = dict(profile=reuse.PROFILE, route=reuse.OLD_ROUTE, occurrence=occurrence,
            prompt_index=0, prompt='fixture', seed=case_seed(self.runtime['model_identity'], occurrence, 0),
            RNG_restored=True, sampling=dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100),
            input_token_ids=[1, 2], continuation_token_ids=[50256], full_token_ids=[1, 2, 50256],
            input_token_count=2, continuation_token_count=1, text='fixture decoded', stop_reason='eos',
            eos_ids=[50256], eos_binding={'model_config': [50256], 'tokenizer': [50256]},
            model_forwards=1, full_prefix_token_work=2)
        metrics = dict(fluency_valid=True, consistency_valid=False, ngram_entropy=0., reference_score=None,
            reasons=['missing_reference'], generation_prompt_count=1, generated_token_count=1,
            length_cap_no_continuation_count=0)
        result = dict(identity=identity, identity_sha256=digest(identity), occurrence=occurrence,
            case_id=record['case_id'], observations=[observation], metrics=metrics,
            raw_local_only=True, checkpoint_saved=False)
        result['payload_sha256'] = digest(result)
        return result

    def write_raw(self, raw):
        path = self.raw_root / 'observations' / (raw['identity_sha256'] + '.json')
        immutable_write(path, raw)
        return path

    def inventory(self, out=None):
        return reuse.build_inventory(self.root, self.config, self.records, out=out)

    def test_complete_only_inventory_and_unchanged_original_reader(self):
        raw = self.raw()
        path = self.write_raw(raw)
        before = path.read_bytes()
        inventory = self.inventory()
        self.assertEqual((inventory['reusable_cases'], inventory['unknown_cases']), (1, 1999))
        checked = reuse.read_reusable_case(inventory, 1, self.records[0])
        self.assertEqual(checked['raw'], raw)
        self.assertEqual(checked['row']['metrics']['ngram_entropy'], 0.)
        self.assertEqual(checked['provenance']['original_runtime_sha256'], digest(self.runtime))
        self.assertEqual(checked['provenance']['original_source_sha'], reuse.OLD_SOURCE)
        self.assertEqual(checked['provenance']['original_route'], reuse.OLD_ROUTE)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(inventory['actual_GPU'])
        self.assertFalse(inventory['new_source_relabel'])
        with self.assertRaisesRegex(reuse.ReuseError, 'CASE_NOT_COMPLETE'):
            reuse.read_reusable_case(inventory, 2, self.records[1])

    def test_incomplete_prompt_unknown_schema_or_seed_mismatch_are_not_complete(self):
        for occurrence, mutate in ((1, lambda raw: raw['observations'].clear()),
            (2, lambda raw: raw['observations'][0].update(seed=0)),
            (3, lambda raw: raw['observations'][0].update(stop_reason='unknown')),
            (4, lambda raw: raw['metrics'].update(fluency_valid=True, ngram_entropy=True))):
            raw = self.raw(occurrence)
            mutate(raw)
            raw['payload_sha256'] = digest({key: value for key, value in raw.items() if key != 'payload_sha256'})
            self.write_raw(raw)
        inventory = self.inventory()
        self.assertEqual(inventory['reusable_cases'], 0)
        self.assertEqual([entry['reason'] for entry in inventory['entries'][:4]],
            ['REUSE_CASE_INCOMPLETE', 'REUSE_PROMPT_SOURCE_SEED_PROFILE',
             'REUSE_PROMPT_INCOMPLETE_OR_WORK', 'REUSE_CASE_METRIC_VALIDITY'])

    def test_token_prefix_eos_and_work_integrity_are_exact(self):
        for field, value in (('full_token_ids', [1, 9, 50256]), ('model_forwards', 0),
                             ('full_prefix_token_work', 99), ('eos_ids', [0]), ('route', 'NEW_ROUTE')):
            raw = self.raw()
            raw['observations'][0][field] = value
            raw['payload_sha256'] = digest({key: value for key, value in raw.items() if key != 'payload_sha256'})
            path = self.raw_root / 'observations' / (raw['identity_sha256'] + '.json')
            with self.subTest(field=field), self.assertRaises(reuse.ReuseError):
                reuse.validate_case(raw, {'path': str(path)}, runtime_identity=self.runtime,
                    state_identity={'full_cold': 'fixture'}, record=self.records[0], occurrence=1)

    def test_reuse_rechecks_raw_sha_size_and_record_identity(self):
        raw = self.raw()
        path = self.write_raw(raw)
        inventory = self.inventory()
        changed = copy.deepcopy(self.records[0])
        changed['generation_prompts'] = ['different fixture']
        with self.assertRaisesRegex(reuse.ReuseError, 'CASE_OR_PAYLOAD_IDENTITY'):
            reuse.read_reusable_case(inventory, 1, changed)
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(reuse.ReuseError, 'MEMBER_IDENTITY'):
            reuse.read_reusable_case(inventory, 1, self.records[0])

    def test_model_native_scoring_cold_semantics_and_original_source_are_bound(self):
        for field in ('cold', 'scoring', 'native'):
            target = copy.deepcopy(self.config)
            if field == 'cold':
                target['generation']['W0_state_identity'] = {'full_cold': 'different'}
            elif field == 'scoring':
                target['generation']['scoring_versions']['nltk'] = 'different'
            else:
                target['arm_configs']['BASE_MEMIT']['native']['fit_evaluations'] = 24
            with self.subTest(field=field), self.assertRaisesRegex(reuse.ReuseError, 'SEMANTIC_INPUT_MISMATCH'):
                reuse.build_inventory(self.root, target, self.records)
        bad = copy.deepcopy(self.config)
        bad['generation']['model_identity'] = 'wrong-model'
        with self.assertRaisesRegex(reuse.ReuseError, 'MODEL_TOKENIZER_RUNTIME_BINDING'):
            reuse.semantic_identity(bad)

    def test_optional_existing_tokenizer_checks_input_and_decode_without_a_model(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {'input_ids': [Tokens([1, 2])]}
            def decode(self, tokens, **kwargs):
                return 'fixture decoded'
        class Tokens(list):
            def tolist(self):
                return list(self)
        raw = self.raw()
        self.write_raw(raw)
        inventory = reuse.build_inventory(self.root, self.config, self.records, tokenizer=Tokenizer())
        self.assertEqual(inventory['reusable_cases'], 1)
        self.assertTrue(inventory['entries'][0]['provenance']['tokenizer_reencoded'])

    def cold_evidence(self):
        weights = {'3': 'fixture-cold-weight'}
        self.config['cold_W'] = weights
        self.config['generation']['W0_state_identity'] = dict(model_state='cold_W0',
            selected_physical_W=weights, actual_model_edits=0, no_history_identity_shared=True)
        (self.root / 'config.json').write_text(json.dumps(self.config))
        source = self.root / 'source/project/run_scripts/gptj_native_baselines'
        source.mkdir(parents=True)
        runner = source / 'generation_run.py'
        runner.write_text('''def execute_chain():
    w0 = ops.install_W0()
    w0gen = guarded_generation(generation.load_W0)
    for current in ops.batches(records):
        engine.apply(current)
def main():
    try:
        initial_history(engine)
        write(out / 'runtime.json', value)
        execute_chain()
    finally:
        pass
''')
        bridge = source / 'generation_bridge.py'
        bridge.write_text('''class GenerationObserver:
    def load_W0(self):
        self._cold_weights()
        return self.shared.observe()
''')
        lock, _ = reuse.read_member(self.root / 'execution.lock.json')
        lock['config_sha256'] = member(self.root / 'config.json')['sha256']
        lock['source_members'] += [member(runner), member(bridge)]
        (self.root / 'execution.lock.json').write_text(json.dumps(lock))
        initial = dict(W=weights, H={})
        immutable_write(self.root / 'BASE_MEMIT/runtime.json', dict(initial_state=initial,
            source=reuse.OLD_TASK_SOURCE, config=digest(self.config), arm='BASE_MEMIT',
            model=self.config['model'], torch=self.config['runtime']['torch'], transformers=self.config['runtime']['transformers'],
            checkpoint_saved=False, initial_history_zero=True, FP32=True, eager=True, TF32=False, autocast=False))
        immutable_write(self.root / 'BASE_MEMIT/W0/summary.json', dict(state=initial, endpoint='W0',
            requests=2000, no_mutation=True, optimizer_feedback=False, scalar_bridge_only=True))
        result = self.root / 'fixture-original-RPN-result.json'
        immutable_write(result, dict(new_actual_evaluation=True, no_mutation=True,
            edit_calls=0, fits=0, solves=0, history_appends=0))
        immutable_write(self.root / 'BASE_MEMIT/W0/reuse.json', dict(scalar_bridge_only=True,
            history_or_editor_resume=False, actual_cold_weights=weights, source_result=member(result)))
        self.write_raw(self.raw())
        inventory_path = self.root / 'inventory-local.json'
        self.inventory(inventory_path)
        return member(inventory_path), reuse.build_old_w0_reuse_binding(member(inventory_path),
            out_directory=self.root / 'cold-binding')

    def test_cold_phase_guard_is_source_backed_not_measured_final_history(self):
        inventory, binding = self.cold_evidence()
        guard, _ = reuse.read_member(binding['cold_observation_guard_member']['path'])
        self.assertEqual(guard['commits'], 0)
        self.assertEqual(guard['history_appends'], 0)
        self.assertEqual(guard['final_RAM_history'], 'NOT_RECORDED')
        self.assertFalse(guard['measured_final_history_zero'])
        self.assertFalse(guard['whole_endpoint_guard_recorded'])
        self.assertEqual(binding['old_w0_reuse']['original_state_identity'], self.config['generation']['W0_state_identity'])
        commit = self.root / 'BASE_MEMIT/batch-01/commit.json'
        immutable_write(commit, {'fixture_only': True})
        with self.assertRaisesRegex(reuse.ReuseError, 'PARTIAL_PHASE_NOT_ESTABLISHED'):
            reuse.build_old_w0_reuse_binding(inventory, out_directory=self.root / 'different-binding')

    def test_native_finite_metric_roundoff_is_preserved_not_a_new_quality_gate(self):
        raw = self.raw()
        raw['metrics'].update(consistency_valid=True, reference_score=1.0000000000000002, reasons=[])
        raw['payload_sha256'] = digest({key:value for key,value in raw.items() if key != 'payload_sha256'})
        self.write_raw(raw)
        self.assertEqual(self.inventory()['reusable_cases'], 1)

    def qualification(self, *, actual=True):
        # Synthetic consumer protocol only: these flags are never GPU evidence.
        plan = dict(fixture_only=False, cohort_sha256='e' * 64,
            model_identity=self.runtime['model_identity'], shared_source_sha='f' * 40,
            native_source_binding={'native': 'same'}, tolerances=qualification_contract.TOLERANCES,
            batch_microbatch=8, shared_plan_sha256='8' * 64)
        plan_path = self.root / 'qualification-plan.json'
        immutable_write(plan_path, plan)
        expected = dict(plan_sha256=digest(plan), cohort_sha256='e' * 64,
            shared_source_sha='f' * 40, native_source_binding={'native': 'same'},
            selected_route=qualification_contract.SINGLETON, fixed_microbatch=1)
        selected = dict(status='PASS', max_logit_abs_error=0., max_topk_probability_abs_error=0.,
            gates={key: True for key in ('tokens', 'EOS', 'row_mapping', 'seed_stream', 'logits',
                'topk_ids', 'topk_probabilities', 'metrics', 'positions', 'coverage')},
            coverage={key: True for key in qualification_contract.REQUIRED_COVERAGE})
        receipt = dict(schema=reuse.QUALIFICATION_SCHEMA,
            status='QUALIFIED_ACTUAL_GPU_ROUTE' if actual else 'CPU_FIXTURE_NOT_ACTUAL_QUALIFICATION',
            actual_GPU=actual, selected_route_passed=True, state_unchanged=True, RNG_restored=True,
            no_fit=True, checkpoint_saved=False, model_identity=self.runtime['model_identity'], **expected,
            shared_plan_sha256=plan['shared_plan_sha256'],
            tolerances=qualification_contract.TOLERANCES,
            route_results={qualification_contract.REFERENCE: {'status': 'PASS'},
                qualification_contract.SINGLETON: selected, qualification_contract.BATCH: {'status': 'NOT_QUALIFIED'}},
            work={route: {key: 1 for key in qualification_contract.WORK_FIELDS}
                  for route in qualification_contract.ROUTES},
            cost={route: {key: 1 for key in qualification_contract.COST_FIELDS}
                  for route in qualification_contract.ROUTES})
        path = self.root / ('qualification-actual.json' if actual else 'qualification-fixture.json')
        immutable_write(path, receipt)
        runtime = dict(self.runtime, generation_source_sha='f' * 40,
                       route=qualification_contract.SHARED_ROUTES[expected['selected_route']])
        return member(path), runtime, expected, member(plan_path)

    def test_fixture_qualification_cannot_promote_reuse_and_actual_binds_original_provenance(self):
        self.write_raw(self.raw())
        path = self.root / 'inventory-local.json'
        inventory = self.inventory(path)
        qualification, runtime, expected, plan = self.qualification(actual=False)
        with self.assertRaisesRegex(reuse.ReuseError, 'ACTUAL_QUALIFICATION_REQUIRED'):
            reuse.build_compatibility(member(path), qualification, runtime, expected, qualification_plan_member=plan)
        qualification, runtime, expected, plan = self.qualification()
        manifest = reuse.build_compatibility(member(path), qualification, runtime, expected, qualification_plan_member=plan)
        self.assertEqual(manifest['identity']['qualification']['sha256'], qualification['sha256'])
        self.assertEqual(manifest['identity']['qualification_plan']['sha256'], plan['sha256'])
        self.assertEqual(manifest['identity']['old_runtime_sha256'], digest(self.runtime))
        self.assertEqual(manifest['identity']['new_runtime_sha256'], digest(runtime))
        self.assertEqual(manifest['identity']['provenance_bindings'][0], inventory['entries'][0]['provenance'])
        self.assertFalse(manifest['original_runtime_checks_disabled'])
        self.assertFalse(manifest['full_W0_READY'])

    def test_qualification_plan_source_route_and_model_mismatch_are_rejected(self):
        path = self.root / 'inventory-local.json'
        self.inventory(path)
        qualification, runtime, expected, plan = self.qualification()
        for field, value in (('plan_sha256', 'wrong'), ('selected_route', 'wrong')):
            with self.subTest(field=field), self.assertRaisesRegex(reuse.ReuseError, 'QUALIFICATION_EXACT_BINDING'):
                reuse.build_compatibility(member(path), qualification, runtime, dict(expected, **{field: value}),
                                          qualification_plan_member=plan)
        with self.assertRaisesRegex(reuse.ReuseError, 'NEW_RUNTIME_SEMANTIC_BINDING'):
            reuse.build_compatibility(member(path), qualification, dict(runtime, model_identity='other-model'), expected,
                                      qualification_plan_member=plan)
        receipt, _ = reuse.read_member(qualification['path'])
        receipt['route_results'][qualification_contract.SINGLETON]['gates'] = {}
        bad_path = self.root / 'qualification-unmeasured.json'
        immutable_write(bad_path, receipt)
        with self.assertRaisesRegex(reuse.ReuseError, 'ACTUAL_MEASURED_QUALIFICATION_REQUIRED'):
            reuse.build_compatibility(member(path), member(bad_path), runtime, expected,
                                      qualification_plan_member=plan)

    def test_private_proof_uses_explicit_shared_route_map_without_runtime_relabel(self):
        path = self.root / 'inventory-local.json'
        self.inventory(path)
        qualification, runtime, expected, plan = self.qualification()
        before = copy.deepcopy(runtime)
        result = reuse.build_compatibility(member(path), qualification, runtime, expected,
                                          qualification_plan_member=plan)
        self.assertEqual(runtime, before)
        self.assertEqual(result['identity']['new_runtime_identity'], before)
        self.assertEqual(result['identity']['qualification_route_map'], qualification_contract.SHARED_ROUTES)
        self.assertNotEqual(runtime['route'], expected['selected_route'])
        with self.assertRaisesRegex(reuse.ReuseError, 'NEW_RUNTIME_SEMANTIC_BINDING'):
            reuse.build_compatibility(member(path), qualification, dict(runtime, route=expected['selected_route']),
                                      expected, qualification_plan_member=plan)

    def test_shared_manifest_cpu_protocol_preserves_original_raw_and_dual_proof(self):
        from project.run_scripts.experiment_generation_eval import kv_qualification as shared_q
        from project.run_scripts.experiment_generation_eval.compatibility import verified_endpoint_row
        inventory_member, binding = self.cold_evidence()
        private_member, _, _, plan_member = self.qualification()
        plan, _ = reuse.read_member(plan_member['path'])
        plan['shared_plan_sha256'] = '8' * 64
        linked_plan_path = self.root / 'linked-plan.json'
        immutable_write(linked_plan_path, plan)
        actual, _ = reuse.read_member(private_member['path'])
        actual.update(plan_sha256=digest(plan), shared_plan_sha256=plan['shared_plan_sha256'],
                      selected_route=qualification_contract.REFERENCE)
        actual['route_results'][qualification_contract.REFERENCE] = actual['route_results'][qualification_contract.SINGLETON]
        actual['route_results'][qualification_contract.SINGLETON] = {'status': 'NOT_QUALIFIED'}
        actual_path = self.root / 'linked-private-actual.json'
        immutable_write(actual_path, actual)
        actual_member = {key:member(actual_path)[key] for key in ('path', 'bytes', 'sha256')}
        # Synthetic receipt-shaped protocol fixture, never actual GPU evidence.
        shared = dict(schema=shared_q.QUALIFICATION_SCHEMA, actual_qualification=True,
            qualification_pass=True, pretrained_GPU_PASS=True, selected_route=reuse.OLD_ROUTE,
            microbatch=1, fixed_microbatch=1, plan_sha256=plan['shared_plan_sha256'],
            model_identity=self.runtime['model_identity'], source_identity='f' * 40,
            tolerances=shared_q.TOLERANCES, model_no_mutation=True, RNG_restored=True,
            native_state_no_mutation=True, route_results={reuse.OLD_ROUTE: {'passed': True}},
            caller_proof_member=actual_member, private_plan_sha256=digest(plan),
            execution_adapter='TASK_PRIVATE_SHARED_GENERATE_ROWS_PROOF_CONVERSION_NOT_SHARED_RUN_QUALIFICATION')
        shared['identity_sha256'] = digest(shared)
        shared_path = self.root / 'linked-shared-actual.json'
        immutable_write(shared_path, shared)
        shared_member = {key:member(shared_path)[key] for key in ('path', 'bytes', 'sha256')}
        runtime = dict(self.runtime, generation_source_sha='f' * 40, generation_microbatch=1,
            qualification_receipt_sha256=shared_member['sha256'])
        result = reuse.build_shared_compatibility(inventory_member, shared_member,
            binding['cold_observation_guard_member'], runtime, self.config['generation']['W0_state_identity'],
            out=self.root / 'shared-compatibility.json', qualification_plan_member=member(linked_plan_path),
            actual_qualification_member=actual_member)
        manifest = result['manifest']
        inventory, _ = reuse.read_member(inventory_member['path'])
        row = copy.deepcopy(inventory['entries'][0]['row'])
        original_raw = inventory['entries'][0]['provenance']['raw']
        original_before = Path(original_raw['path']).read_bytes()
        row['provenance'] = dict(origin='COMPATIBLE_ORIGINAL_W0', raw_member=original_raw,
            runtime_sha256=digest(self.runtime), generation_source_sha=reuse.OLD_SOURCE, route=reuse.OLD_ROUTE)
        identity = dict(runtime=digest(runtime), state_sha256=digest(self.config['generation']['W0_state_identity']),
            qualification_receipt_sha256=shared_member['sha256'], compatibility_sha256=manifest['identity_sha256'])
        endpoint = dict(identity=identity, identity_sha256=digest(identity), compatibility_member=result['member'])
        checked = verified_endpoint_row(row, endpoint, expected_state=self.config['generation']['W0_state_identity'])
        self.assertEqual(checked['identity']['runtime'], digest(self.runtime))
        self.assertEqual(Path(original_raw['path']).read_bytes(), original_before)
        self.assertEqual(manifest['identity']['eligible_cases'], 1)
        self.assertEqual(manifest['identity']['planned_cases'], 2000)
        self.assertFalse(manifest['identity']['old_whole_endpoint_guard_recorded'])
        broken = copy.deepcopy(runtime)
        broken['generation_microbatch'] = 8
        with self.assertRaisesRegex(reuse.ReuseError, 'DUAL_PROOF_RUNTIME_BINDING'):
            reuse.build_shared_compatibility(inventory_member, shared_member,
                binding['cold_observation_guard_member'], broken, self.config['generation']['W0_state_identity'],
                out=self.root / 'bad-manifest.json', qualification_plan_member=member(linked_plan_path),
                actual_qualification_member=actual_member)


if __name__ == '__main__':
    unittest.main()
