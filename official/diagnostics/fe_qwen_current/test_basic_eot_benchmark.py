import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import basic_eot_benchmark as b
from official.diagnostics.fe_qwen_current.benchmark import native_records
from official.experiments.prepare import write_new
from official.runners.server1.common import member
from official.tests.test_factual import TransitionLM,zr,cf,CharacterTokenizer
from official.tests.test_zsre_paper import Tokenizer
from official.tracking.schema import config as validate_config,metrics


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_gate_missing_partial_failed_and_thresholds(self):
        b.pass_gate(dict(Efficacy=95.,Generalization=85.,Specificity=70.))
        for key in ('Efficacy','Generalization','Specificity'):
            for value in (0.,float('nan'),float('inf')):
                v=dict(Efficacy=95.,Generalization=85.,Specificity=70.);v[key]=value
                with self.assertRaises(ValueError):b.pass_gate(v)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';write_new(p,dict(status='QUALITY_EARLY_STOP'))
            with self.assertRaises(ValueError):b.audit_heldout(dict(heldout_result=member(p)),None)
        with self.assertRaises((KeyError,ValueError)):b.audit_heldout({},None)

    def test_policy_rejects_clamp_tuned_sink_and_mask(self):
        good=dict(input_policy=b.POLICY,L2=10,clamp_norm_factor=1.,no_mask=True);b.settings(good)
        for key,value in (('L2',100),('clamp_norm_factor',.5),('input_policy','SINK_LIST'),('no_mask',False)):
            bad=dict(good);bad[key]=value
            with self.assertRaises(ValueError):b.settings(bad)

    def test_endpoint_slices_full_w0_and_current_not_cumulative(self):
        rows=list(range(2000))
        for i in range(1,21):self.assertEqual(b.endpoint_rows(rows,dict(batch=i,endpoint='current/post',requests=100)),rows[(i-1)*100:i*100])
        for i in (5,10,15,20):self.assertEqual(len(b.endpoint_rows(rows,dict(batch=i,endpoint='all_seen/post',requests=i*100))),i*100)
        self.assertEqual(len(b.endpoint_rows(rows,dict(batch=0,endpoint='W0_first2000',requests=2000))),2000)

    def test_tracking_dataset_schema(self):
        bindings=dict(execution=dict(server='rent',attempt='r1',source_commit='a'*40,job_id='121259',source_run_id='parent'),main_stream=dict(sha256='b'*64))
        for dataset in ('cf','zsre'):
            bindings['dataset']=dataset
            with patch.dict('os.environ',{'ODEEDIT_RENT_JOB_ID':'121259'}):values=b.tracking(bindings,{})
            validate_config(values)
            self.assertEqual(values['dataset'],dataset);self.assertEqual(values['task_id'],'fe-qwen-basic-eot-final-main-20261012')
            self.assertEqual('generation_schedule' in values,dataset=='cf')

    def test_actual_zsre_public_queries_macro_and_state_preservation(self):
        tok=Tokenizer();model=TransitionLM(consume_rng=True);model.config.model_type='qwen2'
        row=zr();row['neighborhood_prompts'][0]['prompt']='nq question: Bob is?'
        rows=[dict(copy.deepcopy(row),case_id=i,occurrence_index=i+1) for i in range(100)]
        original=copy.deepcopy(rows);native=native_records(rows)
        self.assertEqual(rows,original);self.assertEqual(len(native),100);self.assertEqual(native[0]['target_new'],row['requested_rewrite']['target_new']['str'])
        bindings=dict(dataset='zsre',execution=dict(server='rent',attempt='r1',source_commit='a'*40,job_id='121259',source_run_id='parent'),main_stream=dict(sha256='b'*64))
        with patch.dict('os.environ',{'ODEEDIT_RENT_JOB_ID':'121259'}):values=b.tracking(bindings,{})
        logged=[]
        def log(payload):metrics(payload,scientific=True,config_values=values);logged.append(payload);return True
        with tempfile.TemporaryDirectory() as d:
            obs=b.FinalObserver.__new__(b.FinalObserver);obs.out=Path(d);obs.b=bindings;obs.values=values;obs.stream=rows;obs.identity=dict(dataset='zsre');obs.raw=[]
            obs.full_queries=b.zsre_paper.build_queries(tok,rows,model_family='qwen25');obs.tracker=SimpleNamespace(log=log)
            before=torch.get_rng_state().clone();H=torch.zeros(1,2,2)
            obs.evaluate(model,tok,H,rows,'current/post',1,'B01-current')
            self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.equal(H,torch.zeros_like(H)))
            self.assertEqual(logged[0]['zsre/current/post/requests'],100)
            ev=obs.raw[0];audit=b.audit_endpoint(ev,rows,'zsre',tok,obs.identity)
            self.assertEqual(audit['Specificity'],ev['summary']['Specificity_loc_ans'])
            raw=b.read(b.verify(ev['raw']));raw['identity']['dataset']='cf'
            bad=Path(d)/'wrong.json';write_new(bad,raw)
            with self.assertRaises(ValueError):b.audit_endpoint(dict(ev,raw=member(bad)),rows,'zsre',tok,obs.identity)

    def test_zsre_low_loc_does_not_use_cf_quality_gate(self):
        obs=b.FinalObserver.__new__(b.FinalObserver);obs.b=dict(dataset='zsre');obs.batch=1;obs.names=set();obs.nonedited={};obs.raw=[]
        model=torch.nn.Module();obs.stream=[{}]*2000
        def evaluate(*args):obs.raw.append(dict(summary=dict(Efficacy=0.,Generalization=0.,Specificity=0.,requests=100)));return {'sha256':'x'}
        obs.evaluate=evaluate
        with tempfile.TemporaryDirectory() as d,patch.object(b,'quality_stop',side_effect=AssertionError('CF_GATE_ON_ZSRE')):
            obs.out=Path(d);obs.identity={};obs.snapshot(None,model,None,torch.zeros(1),[],1,[])
            self.assertTrue((obs.out/'commits/B01.json').is_file())

    def test_actual_cf_current_and_independent_nll_reduction(self):
        tok=CharacterTokenizer();model=TransitionLM(consume_rng=True);rows=[cf(i+1) for i in range(100)]
        bindings=dict(dataset='cf',execution=dict(server='rent',attempt='r1',source_commit='a'*40,job_id='121259',source_run_id='parent'),main_stream=dict(sha256='b'*64))
        with patch.dict('os.environ',{'ODEEDIT_RENT_JOB_ID':'121259'}):values=b.tracking(bindings,{})
        def log(payload):metrics(payload,scientific=True,config_values=values);return True
        with tempfile.TemporaryDirectory() as d:
            obs=b.FinalObserver.__new__(b.FinalObserver);obs.out=Path(d);obs.b=bindings;obs.values=values;obs.stream=rows;obs.identity=dict(dataset='cf');obs.raw=[];obs.tracker=SimpleNamespace(log=log)
            before=torch.get_rng_state().clone();obs.evaluate(model,tok,torch.zeros(1),rows,'current/post',1,'B01-current')
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            audit=b.audit_endpoint(obs.raw[0],rows,'cf',tok,obs.identity);self.assertEqual(audit['requests'],100)

    def test_uncompleted_collector_rejected_before_model_or_queries(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'partial.json';write_new(p,dict(status='QUALITY_EARLY_STOP'))
            with self.assertRaises(ValueError):b.collect(p,p,Path(d)/'report.json')
            self.assertFalse((Path(d)/'report.json').exists())

    def test_twenty_commit_and_twentyfive_endpoint_coverage(self):
        events=[dict(batch=0,endpoint='W0_first2000',requests=2000)]
        events += [dict(batch=i,endpoint='current/post',requests=100) for i in range(1,21)]
        events += [dict(batch=i,endpoint='all_seen/post',requests=i*100) for i in (5,10,15,20)]
        with self.assertRaises(ValueError):b.audit_chain(dict(evaluations=events[:-1]),[],[],'cf',None,list(range(1,21)),('W0_first2000',2000))
        with self.assertRaises(ValueError):b.audit_chain(dict(evaluations=events),[{}]*19,[],'cf',None,list(range(1,21)),('W0_first2000',2000))


if __name__=='__main__':unittest.main()
