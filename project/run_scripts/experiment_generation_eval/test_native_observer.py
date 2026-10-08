"""Native adapter CPU software evidence; NOT pretrained/GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .common import GenerationError, digest
from .generator import rng_snapshot, rng_equal
from .metrics import score_case, generation_payload
from .native_observer import NativeGenerationObserver, verify_native_raw
from .native_profile import PROFILE, ROUTE, runtime_identity, source_identity
from .test_native_generator import Model, Tokenizer, prompt
from .test_observer import FakeAssets, record


class NativeObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.tok, self.assets = Model(stochastic=True), Tokenizer(), FakeAssets()
        self.external = dict(H='empty', context='fixed')
        self.config = dict(model_identity=dict(model='fake', runtime='CPUfake'),
            generation_source_sha='new-exact-source', profile=PROFILE, eval_seed=20261007)
        self.observer = NativeGenerationObserver(self.model,self.tok,self.assets,self.config,
            self.root/'local',lambda:dict(self.external))
        self.state = dict(W='actual-same-weights',H={})
        self.records = [record(1,prompts=[prompt(98),prompt(99)]),
                        record(2,501,prompts=[prompt(99)])]

    def test_original_case_batches_one_text_set_metrics_guard_and_exact_cache(self):
        saved = rng_snapshot()
        value = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        self.assertTrue(rng_equal(saved))
        self.assertEqual([row['query'].shape[0] for row in self.model.calls],[2,2,1])
        self.assertEqual(value['work']['physical_forward_calls'],3)
        self.assertEqual(value['work']['generation_forwards'],5)
        self.assertTrue(value['RNG_restored'] and value['observer_no_mutation'])
        for row in value['rows']:
            raw = json.loads(Path(row['observation_path']).read_text())
            self.assertEqual(raw['metrics'],score_case(raw['observations'],['a a a'],
                self.assets.vectorizer,self.assets.word_tokenize))
            verify_native_raw(raw,expected_runtime=self.observer.runtime_sha,assets=self.assets)
        repeated = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        self.assertEqual(repeated['rows_path'],value['rows_path'])
        self.assertEqual(len(self.model.calls),3)
        self.assertEqual(repeated['work']['cached_case_observations'],2)
        self.assertEqual(repeated['work']['physical_forward_calls'],0)
        execution = json.loads(Path(value['native_execution_member']['path']).read_text())
        self.assertFalse(execution['qualification_performed'])
        self.assertIsNone(self.observer.qualification_member)

    def test_cpu_subsets_immutable_paths_readback_and_scalar_privacy(self):
        events = []
        self.observer.progress_callback = events.append
        whole = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        first = self.observer.subset(whole,self.records[:1],'CURRENT','CURRENT')
        second = self.observer.subset(whole,self.records[1:],'SECOND','CURRENT')
        repeated = self.observer.subset(whole,self.records[:1],'CURRENT','CURRENT')
        self.assertEqual(first['rows_path'],repeated['rows_path'])
        self.assertNotEqual(first['rows_path'],second['rows_path'])
        self.assertEqual(self.observer.read_observed(first['rows_path'])['summary'],first['summary'])
        self.assertEqual(first['summary']['planned_count'],1)
        self.assertEqual(first['summary']['generation_prompt_count'],2)
        self.assertEqual(len(self.model.calls),3)
        for result in (first,second,repeated):
            self.assertEqual(Path(result['rows_path']).stem,result['identity_sha256'])
            payload = generation_payload('all_seen/post',result['summary'])
            self.assertTrue(all(type(value) in (int,float) for value in payload.values()))
        self.assertTrue(events)
        self.assertTrue(all(type(value) in (int,float) for event in events
                            for key,value in event.items() if key != 'phase'))
        self.assertFalse(any('text' in key or 'prompt_id' in key for event in events for key in event))

    def test_different_cohort_never_skips_case_rng_stream_by_raw_cache(self):
        whole = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        smaller = self.observer.observe(self.records[1:],'OTHER','CURRENT',self.state)
        self.assertEqual(len(self.model.calls),4)
        self.assertEqual(smaller['work']['new_case_observations'],1)
        self.assertNotEqual(whole['rows'][1]['identity_sha256'],smaller['rows'][0]['identity_sha256'])
        self.assertNotEqual(whole['identity']['sampling_stream_sha256'],
                            smaller['identity']['sampling_stream_sha256'])

    def test_endpoint_identity_and_source_collision_and_wrong_slice_fail(self):
        whole = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        with self.assertRaisesRegex(GenerationError,'IMMUTABLE_GENERATION_IDENTITY_CONFLICT'):
            NativeGenerationObserver(self.model,self.tok,self.assets,
                dict(self.config,generation_source_sha='other'),self.root/'local')
        wrong = copy.deepcopy(whole)
        wrong['summary']['planned_count'] = 99
        with self.assertRaisesRegex(GenerationError,'SUBSET_ENDPOINT_IDENTITY'):
            self.observer.subset(wrong,self.records[:1],'CURRENT')
        altered = record(1,prompts=[prompt(99)])
        with self.assertRaisesRegex(GenerationError,'SUBSET_PROMPT_IDENTITY'):
            self.observer.subset(whole,[altered],'CURRENT')

    def test_profile_is_distinct_no_old_qualification_or_no_cache_fallback(self):
        identity = runtime_identity(self.config,self.assets.sha)
        self.assertEqual(identity['route'],ROUTE)
        self.assertEqual(identity['original_generator'],source_identity())
        self.assertNotEqual(PROFILE,'cf-cake-prompt-inclusive-total100-eos-corrected-v1')
        for key,value in [('qualification_receipt_member',{}),('generation_microbatch',8),
                          ('old_w0_reuse',{}),('generation_route','UNPADDED_FULL_PREFIX_NO_CACHE')]:
            with self.assertRaises(GenerationError):
                runtime_identity(dict(self.config,**{key:value}),self.assets.sha)

    def test_error_restores_rng_no_retry_and_no_complete_endpoint(self):
        self.model.failure = True
        saved = rng_snapshot()
        with self.assertRaisesRegex(GenerationError,'FORWARD_COMPATIBILITY:ValueError'):
            self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        self.assertTrue(rng_equal(saved))
        self.assertEqual(self.external,dict(H='empty',context='fixed'))
        self.assertEqual(list((self.root/'local'/'endpoints').glob('*.json')),[])
        self.assertEqual(list((self.root/'local'/'native-executions').glob('*.json')),[])

    def test_model_parameter_persistent_cache_and_native_state_guards(self):
        self.model.mutate = True
        with self.assertRaisesRegex(GenerationError,'MODEL_HOOK_CACHE_MUTATION'):
            self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        self.model.mutate = False
        self.model._cache = {'tokens':[]}
        original = self.model.forward
        def mutate_cache(*args,**kwargs):
            self.model._cache['tokens'].append(1)
            return original(*args,**kwargs)
        self.model.forward = mutate_cache
        with self.assertRaisesRegex(GenerationError,'MODEL_HOOK_CACHE_MUTATION'):
            self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        def mutate_state(*args,**kwargs):
            self.external['H'] = 'changed'
            return original(*args,**kwargs)
        self.model.forward = mutate_state
        with self.assertRaisesRegex(GenerationError,'NATIVE_STATE_MUTATION'):
            self.observer.observe(self.records,'W20','ALL_SEEN',self.state)

    def test_rehashed_native_profile_corruption_is_rejected_on_cache_read(self):
        whole = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        raw_path = Path(whole['rows'][0]['observation_path'])
        raw = json.loads(raw_path.read_text())
        raw['observations'][0]['EOS_stop'] = True
        raw['payload_sha256'] = digest({k:v for k,v in raw.items() if k != 'payload_sha256'})
        # Corruption fixture: this deliberately bypasses immutable_write.
        raw_path.write_text(json.dumps(raw))
        with self.assertRaises(GenerationError):
            self.observer.read_observed(whole['rows_path'])

    def test_subset_cannot_drop_completed_parent_execution_binding(self):
        whole = self.observer.observe(self.records,'W20','ALL_SEEN',self.state)
        subset = self.observer.subset(whole,self.records[:1],'CURRENT','CURRENT')
        self.assertEqual(subset['identity']['sampling_stream_sha256'],
                         whole['identity']['sampling_stream_sha256'])
        self.assertEqual(subset['identity']['parent_endpoint_identity_sha256'],whole['identity_sha256'])
        self.assertEqual(subset['parent_endpoint_member']['path'],whole['rows_path'])
        saved = json.loads(Path(subset['rows_path']).read_text())
        saved.pop('parent_endpoint_member')
        Path(subset['rows_path']).write_text(json.dumps(saved))  # corruption fixture.
        with self.assertRaisesRegex(GenerationError,'EXECUTION_RECEIPT_REQUIRED'):
            self.observer.read_observed(subset['rows_path'])


if __name__ == '__main__':
    unittest.main()
