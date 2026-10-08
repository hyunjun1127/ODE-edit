"""CPU API/caller checks. No pretrained/GPU/network or native edit qualification."""
import copy
import tempfile
import unittest
from pathlib import Path

from official.tests.test_factual import CharacterTokenizer, TransitionLM, cf, zr
from official.evaluation import factual, reduce
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server4 import observe, tracking
from official.tracking.schema import metrics


class CallerTests(unittest.TestCase):
    def setUp(self):
        self.tok,self.model=CharacterTokenizer(),TransitionLM()
        self.assets=dict(model_revision='a'*40,tokenizer_sha256='b'*64,
                         runtime_sha256='c'*64,generation_identity_sha256='d'*64)

    def test_published_signature_and_sha(self):
        api=observe.verify_api()
        self.assertEqual(api['source_sha256'],
            '2bc41883b9261d084a3b4b99e0920911789659401a6d9b2b0f0d801a446ab47b')

    def test_actual_published_API_used_and_numeric_receipt_excludes_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=observe.FactualObserver(self.model,self.tok,'cf',self.assets,
                                           Path(tmp),qualification=True)
            self.assertIsNone(adapter.verify_w0([cf()]))
            result=adapter.observe([cf()],endpoint='cpu-fixture')
            self.assertGreater(result['work']['forward_calls'],0)
            self.assertEqual(result['summary']['requests'],1)
            self.assertEqual(result['identity']['external_identity'],observe.external_identity(self.assets))
            self.assertNotIn('work',observe.numeric_receipt(result))

    def test_milestone_current_subset_not_allseen_macro(self):
        rows=[cf(1,paraphrases=['Bob was']*3),cf(2,paraphrases=['Ada was'])]
        actual=factual.evaluate_counterfact(self.model,self.tok,rows)
        # Reduction-only 500/100 fixture, not 500 actual model observations.
        cases=[]
        for i in range(1,501):
            row=copy.deepcopy(actual['cases'][int(i>400)])
            row.update(occurrence_index=i,case_id=i)
            cases.append(row)
        all_seen=dict(actual,cases=cases,summary=reduce.counterfact(cases))
        current=observe.subset(all_seen,[dict(occurrence_index=i,case_id=i)
                                       for i in range(401,501)],'cf')
        self.assertEqual(current['summary']['requests'],100)
        self.assertEqual(all_seen['summary']['requests'],500)
        self.assertEqual(current['summary']['Generalization'],100)
        self.assertEqual(all_seen['summary']['Generalization'],20)
        self.assertNotIn('cohort_sha256',current['identity'])
        for prefix,result in [('current/pre',current),('current/post',current),
                              ('all_seen/post',all_seen)]:
            payload=observe.scores(result,prefix,5)
            cfg=tracking.config(dict(run_id='llama3-cf-alphaedit',method='ALPHAEDIT',dataset='cf'),
                                self.assets,dict(code_commit='a'*40,config_sha256='b'*64),'cpu-fixture')
            metrics(payload,scientific=True,config_values=cfg)
            self.assertEqual(payload['edits'],500)
            self.assertFalse(any(key.startswith('current/post/R/') for key in payload))
        self.assertEqual(observe.scores(current,'current/pre',5)['pre_state_edits'],400)

    def test_subset_rejects_wrong_case_and_duplicate_occurrence(self):
        actual=factual.evaluate_counterfact(self.model,self.tok,[cf()])
        with self.assertRaisesRegex(ValueError,'SUBSET_IDENTITY'):
            observe.subset(actual,[dict(occurrence_index=1,case_id=999)],'cf')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_OCCURRENCE'):
            observe.subset(dict(actual,cases=actual['cases']*2),[cf()],'cf')

    def test_zsre_reference_evaluation_is_reused_without_forward(self):
        identity=observe.external_identity(self.assets)
        rows=[zr(i) for i in range(1,2001)]
        reference=factual.build_zsre_w0_reference(self.model,self.tok,rows,identity=identity)
        before=len(self.model.calls)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'w0.json';write_new(path,reference)
            assets=dict(self.assets,w0={'zsre':dict(path=str(path),sha256=file_sha(path))})
            adapter=observe.FactualObserver(self.model,self.tok,'zsre',assets,Path(tmp)/'raw')
            self.assertEqual(adapter.w0,reference['evaluation'])
            adapter.verify_w0(rows)
            self.assertEqual(len(self.model.calls),before)
            bad=copy.deepcopy(assets);bad['runtime_sha256']='e'*64
            with self.assertRaisesRegex(ValueError,'W0_RUNTIME_IDENTITY'):
                observe.FactualObserver(self.model,self.tok,'zsre',bad,Path(tmp)/'other')

    def test_missing_zsre_reference_blocks(self):
        with self.assertRaisesRegex(ValueError,'ZSRE_W0_PREDICTIONS_REQUIRED'):
            observe.FactualObserver(self.model,self.tok,'zsre',self.assets,Path('/unused'),
                                    qualification=True)

    def test_actual_W0_cohort_guard_no_forward(self):
        actual=factual.evaluate_counterfact(self.model,self.tok,[cf()])
        adapter=observe.FactualObserver.__new__(observe.FactualObserver)
        adapter.w0=actual;adapter.dataset='cf';adapter.tokenizer=self.tok
        before=len(self.model.calls)
        self.assertIs(adapter.verify_w0([cf()]),actual)
        with self.assertRaisesRegex(ValueError,'W0_COHORT_ROW_ORDER'):
            adapter.verify_w0([cf(2)])
        with self.assertRaisesRegex(ValueError,'W0_COHORT_TOKEN_IDENTITY'):
            adapter.verify_w0([cf(subject='Other')])
        self.assertEqual(len(self.model.calls),before)

    def test_cf_and_zsre_logging_config(self):
        ready=dict(code_commit='a'*40,config_sha256='b'*64)
        for dataset in ('cf','zsre'):
            config=dict(run_id='llama3-'+dataset+'-alphaedit',method='ALPHAEDIT',dataset=dataset)
            value=tracking.config(config,self.assets,ready,'cpu-config-fixture')
            self.assertEqual(value['metric_schema'],'official-baselines-scalar-v1')
            self.assertEqual('generation_schedule' in value,dataset=='cf')
            self.assertNotIn('job_id',value)  # actual parent Slurm identity belongs to common transport

    def test_progress_adapter_only_renames_W20_phase(self):
        raw={'phase':'generation_evaluation','generation_progress/step':1,
             'generation_progress/completed_cases':2}
        points=[]
        fake=type('Tracker',(),{'log':lambda _,values:points.append(values)})()
        tracking.progress(fake,'W20')(raw)
        self.assertEqual(points[0]['phase'],'W20_generation')
        self.assertEqual(raw['phase'],'generation_evaluation')
        self.assertEqual(points[0]['generation_progress/completed_cases'],2)


if __name__=='__main__':unittest.main()
