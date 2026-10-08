"""Model-free bridge fixtures; published CPU evidence is not GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from official.evaluation.generation.common import digest, immutable_write
from official.evaluation.generation.metrics import MISSING_REASONS
from official.evaluation.generation.native_profile import PROFILE, ROUTE, runtime_identity
from official.evaluation.generation.progress import FIELDS, GenerationProgress
from official.runners.server2 import generation as bridge
from official.tracking import schema as tracking_schema


def fixture_summary():
    return dict(planned_count=2000, fluency_count=2000, consistency_count=2000,
        ngram_entropy=2.5, reference_score=0.25, fluency_sum=5000.0, consistency_sum=500.0,
        generation_prompt_count=2000, generated_token_count=12000,
        missing_reason_counts={name: 0 for name in MISSING_REASONS})


class Assets:
    sha = 'fixed-reference-sha'


class ObserverFixture:
    """No model forward; count actual wrapper invocations and callback payloads."""
    calls = []
    latest = None
    def __init__(self, model, tok, assets, config, out, state_callback=None, progress_callback=None):
        self.config, self.out = config, Path(out)
        self.runtime_sha = digest(runtime_identity(config, assets.sha))
        self.progress_callback = progress_callback
        self.state_callback = state_callback
    def observe(self, records, endpoint, cohort, state):
        type(self).calls.append((endpoint, copy.deepcopy(state)))
        if self.progress_callback:
            event = {'generation_progress/' + name: 0 for name in FIELDS}
            event.update(phase='W0_generation' if endpoint == 'W0' else 'generation_evaluation')
            self.progress_callback(event)
        identity = dict(runtime=self.runtime_sha, endpoint=endpoint, cohort=cohort,
            state_sha256=digest(state), ordered_occurrences=list(range(1, 2001)))
        execution = self.out / 'native-execution-fixture.json'
        immutable_write(execution, {'CPU_FIXTURE_ONLY': True})
        path = self.out / 'endpoint-fixture.json'
        value = dict(identity=identity, identity_sha256=digest(identity), summary=fixture_summary(),
            rows=[dict(occurrence=row['occurrence_index'], case_id=row['case_id']) for row in records],
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True,
            native_execution_member=bridge.member(execution))
        immutable_write(path, value)
        value = dict(value, rows_path=str(path), work=dict(new_case_observations=2000,
            cached_case_observations=0, physical_forward_calls=99, seconds=0.5))
        type(self).latest = value
        return value
    def read_observed(self, path):
        return type(self).latest


class GenerationBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [dict(case_id=i + 10, occurrence_index=i + 1,
            generation_prompts=['synthetic'], requested_rewrite={'target_new': {'str': 'fixture'}})
            for i in range(2000)]
        stream = self.root / 'cf-stream.json'
        stream.write_text(json.dumps(self.records))
        self.manifest = dict(model='gptj', model_id='EleutherAI/gpt-j-6b',
            model_revision='fixed-revision', tokenizer_sha256='fixed-tokenizer',
            model_assets=[{'path': '/fixture/pytorch_model.bin', 'sha256': 'fixed-weight-sha'}],
            runtime={'torch': 'fixed', 'transformers': 'fixed'},
            generation=dict(profile=PROFILE, eval_seed=20261007, generation_route=ROUTE,
                schedule='CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN',
                reference_assets_sha256=Assets.sha),
            streams={'cf': {'path': str(stream), 'lock': {'stream_sha256': bridge.file_sha(stream)}}})
        ObserverFixture.calls, ObserverFixture.latest = [], None
        self.patches = [patch.object(bridge, 'NativeGenerationObserver', ObserverFixture),
                        patch.object(bridge, 'load_assets', return_value=Assets()),
                        patch.object(bridge, 'read_observed', side_effect=lambda *a, **k: ObserverFixture.latest)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def observe(self, endpoint, batch=None, log=None, out=None):
        batch = (0 if endpoint == 'W0' else 20) if batch is None else batch
        return bridge.observe(None, None, self.manifest, self.records,
            out or self.root / endpoint, endpoint, {'completed_batch': batch, 'physical_fixture': True},
            lambda: {'actual_state': 'fixed'}, log)

    def test_W20_one_observation_two_metrics_same_full_receipt(self):
        events = []
        value = self.observe('W20', log=events.append)
        self.assertEqual(len(ObserverFixture.calls), 1)
        self.assertEqual(value, ObserverFixture.latest)
        self.assertNotIn('scalar_payload', value)
        payload = bridge.payload(value, 'W20')
        self.assertEqual(payload['all_seen/post/fluency/ngram_entropy'], 2.5)
        self.assertEqual(payload['all_seen/post/consistency/reference_score'], .25)
        self.assertEqual(payload['edits'], 2000)
        self.assertEqual(payload['pre_state_edits'], 1900)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['phase'], 'W20_generation')
        ready = bridge.read(self.root / 'W20/READY.json')
        self.assertEqual(ready['work']['physical_forward_calls'], 99)
        self.assertTrue(ready['new_generation_observation_performed'])

    def test_midpoint_and_incomplete_commit_generation_forbidden(self):
        with self.assertRaisesRegex(Exception, 'W0_OR_W20_ONLY'):
            self.observe('W5', batch=5)
        with self.assertRaisesRegex(Exception, 'TWENTY_COMMITS'):
            self.observe('W20', batch=19)
        self.assertEqual(ObserverFixture.calls, [])

    def test_W0_is_canonical_cold_model_not_method_pointer(self):
        value = self.observe('W0')
        state = ObserverFixture.calls[0][1]
        self.assertEqual(state['completed_batch'], 0)
        self.assertNotIn('physical_fixture', state)
        self.assertEqual(state['physical_model'], bridge.model_identity(self.manifest))
        payload = bridge.payload(value, 'W0')
        self.assertEqual(payload['edits'], 0)
        self.assertEqual(payload['post_state_edits'], 0)
        self.assertIn('W0_first2000/generation/planned_count', payload)

    def test_full_first2000_order_not_case_sort_or_dedup(self):
        with self.assertRaisesRegex(Exception, 'FULL_FIRST2000'):
            bridge._records(self.manifest, self.records[:100])
        wrong = copy.deepcopy(self.records)
        wrong[0]['requested_rewrite']['target_new']['str'] = 'changed'
        with self.assertRaisesRegex(Exception, 'TOKEN_TARGET_ORDER_IDENTITY'):
            bridge._records(self.manifest, wrong)

    def test_complete_W0_CPU_reuse_no_second_generation(self):
        first = self.observe('W0')
        second = bridge.reuse_w0(self.root / 'W0', self.manifest, self.records)
        self.assertEqual(second['identity_sha256'], first['identity_sha256'])
        self.assertEqual(len(ObserverFixture.calls), 1)
        repeated = self.observe('W0')
        self.assertEqual(repeated['identity_sha256'], first['identity_sha256'])
        self.assertEqual(len(ObserverFixture.calls), 1)

    def test_old_W20_only_or_partial_READY_rejected(self):
        self.observe('W0')
        path = self.root / 'W0/READY.json'
        value = bridge.read(path)
        value['identity']['generation_schedule'] = 'W20_ONLY_FIRST2000'
        value['identity_sha256'] = digest(value['identity'])
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(Exception, 'STUDY_SOURCE_IDENTITY'):
            bridge.reuse_w0(path, self.manifest, self.records)

    def test_wrong_model_and_partial_receipt_rejected(self):
        self.observe('W0')
        with self.assertRaisesRegex(Exception, 'MODEL_IDENTITY'):
            bridge.reuse_w0(self.root / 'W0', self.manifest, self.records,
                           model_identity={'model_id': 'wrong'})
        changed = copy.deepcopy(ObserverFixture.latest)
        changed['summary']['planned_count'] = 100
        with self.assertRaisesRegex(Exception, 'COMPLETE_ENDPOINT'):
            bridge._complete(changed, self.records, 'W0', changed['identity']['runtime'])

    def test_global_native_profile_and_scientific_source_bytes(self):
        config = bridge.configuration(self.manifest)
        runtime = runtime_identity(config, Assets.sha)
        self.assertEqual(runtime['profile'], PROFILE)
        self.assertEqual(runtime['route'], ROUTE)
        self.assertEqual(runtime['sampling_scope'], 'ENDPOINT_GLOBAL_BATCH_STREAM')
        self.assertFalse(runtime['EOS_stop'])
        self.assertEqual(config['generation_source_sha'], digest(bridge.source_identity()))
        self.assertFalse(any('job' in key or 'arm' in key for key in config['model_identity']))

    def test_progress_whitelist_rejects_text_and_secret(self):
        callback = bridge._progress(None, endpoint='W0')
        event = {'generation_progress/' + name: 0 for name in FIELDS}
        event['phase'] = 'W0_generation'
        callback(event)
        for key in ('text', 'API_KEY', 'case_id'):
            with self.assertRaisesRegex(Exception, 'PROGRESS_SCHEMA'):
                callback(dict(event, **{key: 'forbidden'}))

    def tracking_config(self):
        # Validate the real shared official schema, not a duplicated logger or
        # fake online receipt. Fixture identities are never uploaded.
        config = dict(server='server2', task_id='official-baselines-20261008',
            arm='MEMIT', attempt='cpu-generation-fixture', source_sha='a' * 40,
            config_sha='b' * 64, model='gptj', model_family='gptj', writer='MEMIT',
            baseline='MEMIT', role='scientific', metric_schema=tracking_schema.OFFICIAL_SCHEMA,
            instruction_id=tracking_schema.OFFICIAL_INSTRUCTION, dataset='cf',
            generation_metric_schema='counterfact-cake-generation-metrics-v1',
            generation_profile=PROFILE, generation_eval_seed=20261007,
            reference_assets_sha256='c' * 64,
            generation_source_sha=digest(bridge.source_identity()),
            generation_repair_instruction=tracking_schema.OFFICIAL_INSTRUCTION,
            generation_schedule=tracking_schema.OFFICIAL_GENERATION_SCHEDULE)
        return tracking_schema.config(config)

    def test_real_native_progress_shared_adapter_preserves_counts_and_axis(self):
        config = self.tracking_config()
        for endpoint, raw_phase, published_phase in (
                ('W0', 'W0_generation', 'W0_generation'),
                ('W20', 'generation_evaluation', 'W20_generation')):
            with self.subTest(endpoint=endpoint):
                captured = []
                progress = GenerationProgress(2000, 2000,
                    bridge._progress(captured.append, endpoint=endpoint),
                    phase=raw_phase, clock=lambda: 0.0, first_step=7)
                raw = progress.emit('start', force=True)
                expected = dict(raw, phase=published_phase)
                self.assertEqual(captured, [expected])
                self.assertEqual(raw['phase'], raw_phase)
                self.assertEqual(captured[0]['generation_progress/step'], 7)
                self.assertEqual(set(captured[0]),
                    {'generation_progress/' + key for key in FIELDS} | {'phase'})
                self.assertNotIn('edits', captured[0])
                self.assertEqual(tracking_schema.metrics(captured[0], scientific=True,
                    config_values=config), captured[0])

    def test_shared_adapter_rejects_wrong_endpoint_and_noninteger_counts(self):
        event = {'generation_progress/' + name: 0 for name in FIELDS}
        event['phase'] = 'W0_generation'
        with self.assertRaisesRegex(ValueError, 'CALLBACK_PHASE'):
            bridge._progress(None, endpoint='W20')(event)
        with self.assertRaisesRegex(ValueError, 'CALLBACK_ENDPOINT'):
            bridge._progress(None, endpoint='W5')(event)
        with self.assertRaisesRegex(ValueError, 'PROGRESS_INTEGER'):
            bridge._progress(None, endpoint='W0')(
                dict(event, **{'generation_progress/completed_cases': 0.5}))

    def test_actual_bridge_endpoints_satisfy_shared_generation_schema(self):
        config = self.tracking_config()
        for endpoint in ('W0', 'W20'):
            with self.subTest(endpoint=endpoint):
                events = []
                self.observe(endpoint, log=events.append)
                for event in events:
                    self.assertEqual(tracking_schema.metrics(event, scientific=True,
                        config_values=config), event)

    def test_nonmutation_callback_required_before_loading_assets(self):
        with self.assertRaisesRegex(Exception, 'NONMUTATION_CALLBACK_REQUIRED'):
            bridge.observe(None, None, self.manifest, self.records, self.root,
                'W0', {'completed_batch': 0}, None)
        bridge.load_assets.assert_not_called()


if __name__ == '__main__':
    unittest.main()
