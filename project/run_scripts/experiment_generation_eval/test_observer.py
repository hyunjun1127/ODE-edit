import json
from pathlib import Path
import tempfile
import unittest

from .common import GenerationError,digest
from .generator import rng_snapshot,rng_equal
from .observer import GenerationObserver
from .test_generator import FakeModel,FakeTokenizer
from .test_metrics import FixedVectorizer


class FakeAssets:
    sha='fixed-asset-identity'
    vectorizer=FixedVectorizer()
    word_tokenize=staticmethod(str.split)
    def snippets_for(self,relation,target):return ['a a a'] if relation=='r' and target=='t' else []


def record(ordinal,case=500,prompts=None):
    return dict(ordered_occurrence=ordinal,case_id=case,
        generation_prompts=['a a'] if prompts is None else prompts,
        requested_rewrite=dict(relation_id='r',target_new=dict(id='t',str='new')))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.model,self.tok=FakeModel(),FakeTokenizer()
        self.external={'H':'empty','context':'fixed'}
        self.config=dict(model_identity=dict(model='fake',payload='same',runtime='CPUfake'),
            generation_source_sha='source-exact')
        self.observer=GenerationObserver(self.model,self.tok,FakeAssets(),self.config,
            Path(self.temp.name)/'local',lambda:dict(self.external))
        self.state={'W':'samephysicalcold','H':{}}

    def test_one_generation_shared_by_both_metrics_and_endpoint_overlap_cache(self):
        observed=self.observer.observe([record(1),record(2,501)],'W0','first2',self.state)
        self.assertEqual(len(self.model.calls),2)
        self.assertEqual(observed['summary']['fluency_count'],2)
        self.assertEqual(observed['summary']['consistency_count'],2)
        repeated=self.observer.observe([record(1)],'B1PRE','current',self.state)
        self.assertEqual(len(self.model.calls),2)
        self.assertEqual(repeated['work']['cached_case_observations'],1)
        self.assertEqual(repeated['work']['generation_forwards'],0)
        self.assertEqual(len(list((Path(self.temp.name)/'local'/'observations').glob('*.json'))),2)

    def test_readback_subset_only_cpu_no_full_summary_copied_to_current(self):
        full=self.observer.observe([record(1),record(2,501,prompts=[])],'W0','all',self.state)
        loaded=self.observer.read_observed(full['rows_path'])
        subset=self.observer.subset(loaded,[record(2,501,prompts=[])],'current','incoming')
        self.assertEqual(subset['summary']['planned_count'],1)
        self.assertEqual(subset['summary']['fluency_count'],0)
        self.assertNotIn('ngram_entropy',subset['summary'])
        self.assertEqual(len(self.model.calls),1)

    def test_occurrence_keeps_repeated_case_no_dedup_and_missing_occurrence_blocks(self):
        result=self.observer.observe([record(1),record(2)],'W0',None,self.state)
        self.assertEqual(result['summary']['planned_count'],2)
        self.assertNotEqual(result['rows'][0]['identity_sha256'],result['rows'][1]['identity_sha256'])
        row=record(3);row.pop('ordered_occurrence')
        with self.assertRaisesRegex(GenerationError,'OCCURRENCE_REQUIRED'):
            self.observer.observe([row],'W0',None,self.state)

    def test_source_runtime_wrong_subset_and_mutated_raw_hard_fail(self):
        result=self.observer.observe([record(1)],'W0',None,self.state)
        wrong=dict(self.config,generation_source_sha='other-source')
        second=GenerationObserver(self.model,self.tok,FakeAssets(),wrong,Path(self.temp.name)/'second',lambda:{})
        with self.assertRaisesRegex(GenerationError,'RUNTIME_IDENTITY'):
            second.read_observed(result['rows_path'])
        path=Path(result['rows'][0]['observation_path']);raw=json.loads(path.read_text())
        raw['observations'][0]['text']='changed'
        path.write_text(json.dumps(raw))  # corruption fixture only, not implementation write.
        with self.assertRaisesRegex(GenerationError,'OBSERVATION_BYTES_IDENTITY'):
            self.observer.read_observed(result['rows_path'])

    def test_internally_rehashed_mixed_state_row_cannot_be_blessed_as_W0(self):
        result=self.observer.observe([record(1)],'W0',None,self.state)
        path=Path(result['rows'][0]['observation_path']);raw=json.loads(path.read_text())
        raw['identity']['state_identity']={'W':'differentphysicalstate','H':{}}
        raw['identity_sha256']=digest(raw['identity'])
        raw['payload_sha256']=digest({k:v for k,v in raw.items() if k!='payload_sha256'})
        path.write_text(json.dumps(raw))
        # Rehash all row and endpoint metadata, retaining the falsely claimed W0 state.
        endpoint_path=Path(result['rows_path']);saved=json.loads(endpoint_path.read_text())
        for endpoint in (saved,result):
            endpoint['rows'][0]['identity_sha256']=raw['identity_sha256']
            endpoint['rows'][0]['payload_sha256']=raw['payload_sha256']
            endpoint['identity']['observation_identities']=[raw['identity_sha256']]
            endpoint['identity_sha256']=digest(endpoint['identity'])
        endpoint_path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(GenerationError,'OBSERVATION_BYTES_IDENTITY'):
            self.observer.read_observed(endpoint_path)
        with self.assertRaisesRegex(GenerationError,'SUBSET_ROW_IDENTITY'):
            self.observer.subset(result,[record(1)],'W0_ALIAS')

    def test_subset_rejects_tampered_endpoint_sums_order_and_identity(self):
        import copy
        result=self.observer.observe([record(1)],'W0',None,self.state)
        for field in ('summary','identity','identity_sha256'):
            wrong=copy.deepcopy(result)
            if field=='summary':wrong['summary']['fluency_count']=99
            elif field=='identity':wrong['identity']['ordered_occurrences']=[2]
            else:wrong['identity_sha256']='wrong'
            with self.assertRaisesRegex(GenerationError,'SUBSET_ENDPOINT_IDENTITY'):
                self.observer.subset(wrong,[record(1)],'W0_ALIAS')

    def test_subset_rejects_stale_row_payload_hash_even_when_raw_is_self_valid(self):
        result=self.observer.observe([record(1)],'W0',None,self.state)
        result['rows'][0]['payload_sha256']='stale-row-payload'
        with self.assertRaisesRegex(GenerationError,'SUBSET_ROW_IDENTITY'):
            self.observer.subset(result,[record(1)],'W0_ALIAS')

    def test_rng_and_state_guard_failure_forward_original_preserved(self):
        self.model.failure=ValueError('forward-original')
        saved=rng_snapshot()
        with self.assertRaisesRegex(ValueError,'forward-original'):
            self.observer.observe([record(1)],'W0',None,self.state)
        self.assertTrue(rng_equal(saved))
        self.assertEqual(self.external,{'H':'empty','context':'fixed'})

    def test_parameter_mutation_detected_without_checkpoint_or_rescue_copy(self):
        self.model.mutate=True
        with self.assertRaisesRegex(GenerationError,'MODEL_HOOK_CACHE_MUTATION'):
            self.observer.observe([record(1)],'W0',None,self.state)
        self.assertEqual(float(self.model.weight.item()),2)

    def test_persistent_cache_and_native_state_mutation_are_detected(self):
        original=self.model.forward
        self.model._cache={'tokens':[]}
        def mutate_cache(*args,**kwargs):
            self.model._cache['tokens'].append(1)
            return original(*args,**kwargs)
        self.model.forward=mutate_cache
        with self.assertRaisesRegex(GenerationError,'MODEL_HOOK_CACHE_MUTATION'):
            self.observer.observe([record(1)],'W0',None,self.state)
        self.model.forward=original
        def mutate_native(*args,**kwargs):
            self.external['H']='changed'
            return original(*args,**kwargs)
        self.model.forward=mutate_native
        with self.assertRaisesRegex(GenerationError,'NATIVE_STATE_MUTATION'):
            self.observer.observe([record(2,case=501)],'W0',None,self.state)

    def test_same_rng_seed_for_same_model_occurrence_across_arm_names(self):
        a=self.observer.observe([record(1)],'CAKE-W0',None,self.state)
        cfg=dict(self.config,arm='RECT',job_id='fake-not-uploaded')
        another=GenerationObserver(self.model,self.tok,FakeAssets(),cfg,Path(self.temp.name)/'other',lambda:{})
        b=another.observe([record(1)],'RECT-W0',None,self.state)
        ar=json.loads(Path(a['rows'][0]['observation_path']).read_text())
        br=json.loads(Path(b['rows'][0]['observation_path']).read_text())
        self.assertEqual(ar['observations'][0]['seed'],br['observations'][0]['seed'])
        self.assertEqual(ar['observations'][0]['full_token_ids'],br['observations'][0]['full_token_ids'])


if __name__=='__main__':unittest.main()
