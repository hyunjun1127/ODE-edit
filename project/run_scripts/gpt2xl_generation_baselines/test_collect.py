"""Independent stored-scalar reducer fixtures; no model/GPU/native fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import collect as review


def metrics(value=0.,reference=0.,reason=None):
    return dict(ngram_entropy=value,reference_score=reference,fluency_valid=value is not None,
        consistency_valid=reference is not None,reasons=[] if reason is None else [reason],
        generation_prompt_count=1 if value is not None else 0,generated_token_count=0,
        length_cap_no_continuation_count=1 if value is not None else 0)


class Tests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);g=dict(schema='counterfact-cake-generation-metrics-v1',profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',
            eval_seed=20261007,model_identity=dict(model='gpt2xl',revision='a'*40),generation_source_sha='b'*64,reference_assets_sha256='c'*64)
        c=dict(generation=g);state=dict(W={str(i):str(i)*64 for i in (13,14,15,16,17)},H={})
        records=[dict(case_id=901,generation_prompts=['long prompt'],requested_rewrite=dict(relation_id='r',target_new={'id':'t'})),
            dict(case_id=1,generation_prompts=[],requested_rewrite=dict(relation_id='r',target_new={'id':'t'}))]
        runtime=review.digest(review.runtime_identity(c));rows=[]
        for ordinal,record in enumerate(records,1):
            rewrite=record['requested_rewrite'];identity=dict(runtime=runtime,state_identity=state,
                record_identity=dict(ordered_occurrence=ordinal,case_id=record['case_id'],generation_prompts=record['generation_prompts'],relation_id='r',target_new_id='t'))
            observation=[]
            if record['generation_prompts']:
                seed=int(review.digest(dict(model_identity=g['model_identity'],ordered_occurrence=ordinal,prompt_index=0,eval_seed=g['eval_seed']))[:16],16)%(2**63-1)
                observation=[dict(prompt='long prompt',occurrence=ordinal,prompt_index=0,profile=g['profile'],seed=seed,
                    sampling=dict(top_k=5,temperature=1,top_p=1,max_total_tokens=100),RNG_restored=True,
                    route='UNPADDED_FULL_PREFIX_NO_CACHE',input_token_ids=[7]*100,continuation_token_ids=[],
                    full_token_ids=[7]*100,input_token_count=100,continuation_token_count=0,
                    stop_reason='length_cap_no_continuation',model_forwards=0,eos_ids=[50256],text='aa aa')]
            item=metrics() if observation else metrics(None,None,'missing_generation_prompts')
            if observation:item['reasons']=['length_cap_no_continuation']
            raw=dict(identity=identity,identity_sha256=review.digest(identity),occurrence=ordinal,case_id=record['case_id'],
                observations=observation,metrics=item,raw_local_only=True,checkpoint_saved=False)
            raw['payload_sha256']=review.digest(raw);path=root/f'case-{ordinal}.json';review.write(path,raw)
            rows.append(dict(occurrence=ordinal,case_id=record['case_id'],identity_sha256=raw['identity_sha256'],
                observation_path=str(path),payload_sha256=raw['payload_sha256'],metrics=item))
        identity=dict(runtime=runtime,endpoint='W0',cohort='FIRST2000',state_sha256=review.digest(state),
            ordered_occurrences=[1,2],observation_identities=[r['identity_sha256'] for r in rows])
        value=dict(identity=identity,identity_sha256=review.digest(identity),rows=rows,
            summary=review.reduce_generation(rows),RNG_restored=True,observer_no_mutation=True)
        path=root/'endpoint.json';review.write(path,value)
        receipt=dict(rows=review.member(path),identity=identity,identity_sha256=value['identity_sha256'],summary=value['summary'],
            RNG_restored=True,observer_no_mutation=True,work=dict(new_case_observations=2,cached_case_observations=0,
                generation_forwards=0,full_prefix_token_work=0,seconds=.1))
        return c,state,records,receipt,value

    def test_independent_macro_sums_missing_and_valid_zero(self):
        rows=[dict(metrics=metrics(2.,.1)),dict(metrics=metrics(0.,0.)),dict(metrics=metrics(None,None,'missing_reference'))]
        value=review.reduce_generation(rows)
        self.assertEqual(value['ngram_entropy'],1.);self.assertAlmostEqual(value['reference_score'],.05)
        self.assertEqual(value['planned_count'],3);self.assertEqual(value['fluency_count'],2)
        self.assertEqual(value['consistency_count'],2);self.assertEqual(value['missing_reason_counts']['missing_reference'],1)
        missing=review.reduce_generation([dict(metrics=metrics(None,None,'missing_reference'))])
        self.assertNotIn('reference_score',missing);self.assertNotIn('ngram_entropy',missing)
    def test_missing_zero_and_nonfinite_rejected(self):
        item=metrics();item['fluency_valid']=False
        with self.assertRaisesRegex(RuntimeError,'MISSING_NOT_ZERO'):review.reduce_generation([dict(metrics=item)])
        with self.assertRaisesRegex(RuntimeError,'FINITE'):review.reduce_generation([dict(metrics=metrics(float('nan'),0.))])
    def test_production_endpoint_binds_order_seed_raw_tokens_state_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,state,records,receipt,_=self.fixture(tmp)
            value=review.generation_endpoint(review.Reader(),receipt,c,records,'W0','FIRST2000',state)
            self.assertEqual(value['summary']['planned_count'],2);self.assertEqual(value['summary']['fluency_count'],1)
            self.assertEqual([r['case_id'] for r in value['rows']],[901,1])
            self.assertEqual(value['summary']['ngram_entropy'],0.)
    def test_swapped_occurrence_and_state_are_hard_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,state,records,receipt,_=self.fixture(tmp)
            with self.assertRaisesRegex(RuntimeError,'IDENTITY|ORDER'):
                review.generation_endpoint(review.Reader(),receipt,c,records[::-1],'W0','FIRST2000',state)
            wrong=copy.deepcopy(state);wrong['W']['13']='f'*64
            with self.assertRaisesRegex(RuntimeError,'STATE_RNG'):
                review.generation_endpoint(review.Reader(),receipt,c,records,'W0','FIRST2000',wrong)
    def test_summary_count_mismatch_and_false_RNG_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,state,records,receipt,_=self.fixture(tmp)
            wrong=copy.deepcopy(receipt);wrong['summary']['fluency_count']=2
            with self.assertRaisesRegex(RuntimeError,'RAW_SUM_COUNT'):
                review.generation_endpoint(review.Reader(),wrong,c,records,'W0','FIRST2000',state)
            wrong=copy.deepcopy(receipt);wrong['RNG_restored']=False
            with self.assertRaisesRegex(RuntimeError,'STATE_RNG'):
                review.generation_endpoint(review.Reader(),wrong,c,records,'W0','FIRST2000',state)
    def test_actual_seed_tamper_not_hidden_by_new_payloadhash(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,state,records,receipt,endpoint=self.fixture(tmp)
            raw=json.loads(Path(endpoint['rows'][0]['observation_path']).read_text())
            raw['observations'][0]['seed']+=1;raw['payload_sha256']=review.digest({k:v for k,v in raw.items() if k!='payload_sha256'})
            # Fixture mutation in RAM; immutable production files stay preserved.
            endpoint['rows'][0]['payload_sha256']=raw['payload_sha256']
            class FixtureReader(review.Reader):
                def bound(self,value):return endpoint if value['path']==receipt['rows']['path'] else super().bound(value)
                def json(self,path):return raw if str(path)==endpoint['rows'][0]['observation_path'] else super().json(path)
            with self.assertRaisesRegex(RuntimeError,'RNG_KEY'):
                review.generation_endpoint(FixtureReader(),receipt,c,records,'W0','FIRST2000',state)
    def test_six_expected_native_counts_preserve_all_algorithms(self):
        self.assertEqual(review.EXPECTED['BASE_MEMIT']['history_appends'],0)
        self.assertEqual(review.EXPECTED['BASE_ALPHAEDIT']['history_appends'],5)
        self.assertEqual(review.EXPECTED['CAKE']['native_z'],100)
        self.assertEqual(review.EXPECTED['ALPHAEDIT_BLUE']['native_z'],200)
        self.assertEqual(review.ARM_LAYERS['ALPHAEDIT_BLUE'],(13,17))
        self.assertEqual(review.EXPECTED['PRUNE'],review.EXPECTED['RECT'])
    def test_accounting_is_exact_six_gpu_parents_once_and_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);lock=dict(source_commit='a'*40,owner='testowner')
            jobs={a:str(700+i) for i,a in enumerate(review.ARMS)}
            review.write(path/'submission.json',dict(instruction_id=review.NONCE,task_id=review.TASK,source_commit='a'*40,jobs=jobs))
            calls=[]
            def runner(argv,**kw):
                calls.append(argv)
                return SimpleNamespace(returncode=0,stdout='\n'.join(f'{jobs[a]}|{review.TASK}-{a}|testowner|COMPLETED|0:0|10|cpu=8,gres/gpu=1|' for a in review.ARMS))
            value=review.allocation_once(review.Reader(),path,lock,runner=runner,owner='testowner')
            self.assertEqual(value['status'],'RECORDED');self.assertEqual(len(calls),1)
            self.assertEqual(sum(x['allocated_GPU_seconds'] for x in value['records']),60)
            jobs['RECT']='not-a-real-ID'
            self.assertEqual(review.allocation_once(review.Reader(),path,dict(lock,owner='wrong'),runner=runner,owner='testowner')['status'],'NOT_RECORDED')
    def test_partial_collector_report_manifest_before_terminal_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);records=[dict(case_id=i) for i in range(2000)]
            review.write(path/'records.json',records);review.write(path/'observer.json',dict(rows=[]))
            config=dict(task_id=review.TASK,instruction_id=review.NONCE,
                observer_identity=review.member(path/'observer.json'),stream=str(path/'records.json'),
                assets=[review.member(path/'records.json')],
                packs=[dict(ids=list(range(i*100,(i+1)*100))) for i in range(20)])
            review.write(path/'config.json',config)
            review.write(path/'execution.lock.json',dict(instruction_id=review.NONCE,source_commit='a'*40,
                config_sha256=review.sha(path/'config.json')))
            value=review.collect(path)
            self.assertEqual(value['collector_status'],'COMPLETED');self.assertFalse(value['scientific_complete'])
            terminal=json.loads((path/'collector/terminal.json').read_text())
            self.assertEqual(terminal['scientific_status'],'PARTIAL_OR_TECHNICAL_BLOCKED')
            self.assertTrue((path/'collector/report-ko.md').exists());self.assertTrue((path/'collector/manifest.json').exists())
            reviews=json.loads((path/'collector/review.json').read_text())['reviews']
            self.assertEqual(len(reviews),6);self.assertTrue(all(r['commits']==0 for r in reviews))
    def test_returned_W0_work_after_READY_failure_remains_uncommitted_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,state,records,receipt,_=self.fixture(tmp)
            review.write(Path(tmp)/'generation-work/W0.json',dict(phase='W0',batch=0,
                observation=receipt,scientific_commit_not_asserted=True,raw_local_only=True,checkpoint_saved=False))
            value=review.returned_work(review.Reader(),Path(tmp),c,commits=0,W0_recorded=False)
            self.assertEqual(len(value),1);self.assertFalse(value[0]['counted_already_in_commit_receipt'])
            self.assertEqual(value[0]['seconds'],.1)


if __name__=='__main__':unittest.main()
