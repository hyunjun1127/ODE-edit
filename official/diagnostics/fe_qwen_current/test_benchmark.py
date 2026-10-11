import copy
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import torch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member
from official.diagnostics.fe_qwen_current.benchmark import native_records,reduce_zsre,validated_selection,verify_query_subset
from official.diagnostics.fe_qwen_current.contract import SPEC,tracking_config
from official.evaluation.zsre_paper import build_queries,evaluate
from official.tests.test_zsre_paper import Tokenizer
from official.tests.test_factual import TransitionLM,zr
from official.tracking import official_zsre_metrics
from official.tracking.schema import config as validate_config,bind_job_identity,metrics

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tok=Tokenizer();self.model=TransitionLM(consume_rng=True)
        self.model.config.model_type='qwen2';self.record=zr()
        self.record['neighborhood_prompts'][0]['prompt']='nq question: Bob is?'
    def observation(self):
        plan=build_queries(self.tok,[self.record],model_family='qwen25')
        raw=evaluate(self.model,self.tok,[self.record],model_family='qwen25',batch_size=4)
        return raw,plan
    def test_unexpanded_native_edit_retains_author_prompt_target(self):
        before=copy.deepcopy(self.record);row=native_records([self.record])[0]
        self.assertEqual(self.record,before)
        self.assertEqual(row['prompt'],before['requested_rewrite']['prompt'])
        self.assertEqual(row['target_new'],before['requested_rewrite']['target_new']['str'])
        self.assertEqual(len(row['neighborhood_prompts']),1)
    def test_pinned_native_loader_reads_unexpanded_zsre_exactly_once(self):
        author=os.environ.get('FE_NATIVE_AUTHOR_ROOT')
        if not author:self.skipTest('author source not provided')
        tree=ast.parse((Path(author)/'load.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_data')
        namespace={'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned-author-load.py','exec'),namespace)
        rows=native_records([self.record])
        with tempfile.TemporaryDirectory() as d:
            write_new(Path(d)/'zsre-native.json',rows)
            cfg=SimpleNamespace(data='zsre-native.json',data_dir=d,negetive_prompt_test=False,
                target_true_test=False,algs=SimpleNamespace(name='memit'),num_edits=2000)
            self.assertEqual(namespace['load_data'](cfg),rows)
    def test_actual_current100_observer_public_query_payload(self):
        from official.diagnostics.fe_qwen_current import observer
        from official.runners.fe_original import versions
        rows=[dict(copy.deepcopy(self.record),case_id=i,occurrence_index=i+1) for i in range(100)]
        c=dict(stage='final_benchmark',dataset='zsre',server='rent',L2=.1,attempt='r1',
            config_sha256='a'*64,selection_contract_sha256=digest(SPEC),stream=dict(sha256='b'*64),selected_result=dict(sha256='d'*64))
        values=bind_job_identity(tracking_config(c,'c'*40),{'ODEEDIT_RENT_JOB_ID':'120201'})
        logged=[]
        def log(payload):
            metrics(payload,scientific=True,config_values=values);logged.append(payload);return True
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);contexts=[['{}']]
            write_new(out/'contexts.json',dict(contexts=contexts,sha256=digest(contexts)))
            state=dict(c=c,rows=rows,out=out,tracker=SimpleNamespace(log=log),identity={},values=values,
                model=self.model,tok=self.tok,excluded=set(),nonedited=versions(self.model),last_batch=0)
            with patch.object(observer,'_state',return_value=state):
                observer.snapshot(SimpleNamespace(),self.model,self.tok,torch.ones(1,2,2),contexts,1,[])
            self.assertEqual(state['last_batch'],1)
            self.assertEqual(logged[0]['zsre/current/post/requests'],100)
            self.assertTrue((out/'raw-audits/B01-current.json').is_file())
            self.assertFalse((out/'checkpoints').exists())
    def test_actual_tiny_query_forward_independent_reduction(self):
        raw,plan=self.observation();audit=reduce_zsre(raw,plan)
        self.assertEqual(audit['requests'],1)
        self.assertEqual(audit['metrics']['Specificity'],raw['summary']['Specificity_loc_ans'])
        for call in self.model.calls:self.assertFalse(call['use_cache'])
    def test_current_query_subset_matches_full_public_plan(self):
        rows=[dict(copy.deepcopy(self.record),case_id=i,occurrence_index=i+1) for i in range(3)]
        full=build_queries(self.tok,rows,model_family='qwen25')
        subset=build_queries(self.tok,rows[1:2],model_family='qwen25')
        verify_query_subset(full,subset,rows[1:2])
        subset['queries'][0]['input_ids'][0]+=1
        subset['query_sha256']=digest(subset['queries'])
        with self.assertRaisesRegex(ValueError,'SUBSET_PARITY'):verify_query_subset(full,subset,rows[1:2])
    def test_target_mapping_bits_count_and_summary_tampering_rejected(self):
        raw,plan=self.observation()
        for change in ('target','bits','summary','count'):
            bad=copy.deepcopy(raw)
            if change=='target':bad['cases'][0]['rewrite_observations'][0]['target_token_id']+=1
            if change=='bits':bad['cases'][0]['rewrite_prompts_correct'][0]=not bad['cases'][0]['rewrite_prompts_correct'][0]
            if change=='summary':bad['summary']['Specificity']+=1
            if change=='count':bad['token_denominators']['rewrite']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):reduce_zsre(bad,plan)
    def test_selected_receipt_reaudit_and_nonpromotion_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection-result.json'
            receipt=dict(selection=dict(status='SELECTED',selected_L2=.1),results={str(l):{} for l in SPEC['L2_arms']})
            write_new(p,receipt)
            with patch('official.diagnostics.fe_qwen_current.collect.collect',return_value=receipt) as audit:
                self.assertEqual(validated_selection(member(p)),receipt)
                self.assertEqual(audit.call_args.kwargs,{'write_result':False})
            for bad in (dict(receipt,selection=dict(status='NO_PROMOTION',selected_L2=None)),dict(receipt,results={'0.1':{}})):
                with patch('official.diagnostics.fe_qwen_current.collect.collect',return_value=bad),self.assertRaises(ValueError):
                    validated_selection(member(p))
    def test_dataset_groups_schema_and_current_allseen_axes(self):
        raw,_=self.observation()
        for dataset in ('cf','zsre'):
            c=dict(stage='final_benchmark',dataset=dataset,server='rent',L2=.1,attempt='r1',
                config_sha256='a'*64,selection_contract_sha256=digest(SPEC),stream=dict(sha256='b'*64),selected_result=dict(sha256='d'*64))
            values=bind_job_identity(tracking_config(c,'c'*40),{'ODEEDIT_RENT_JOB_ID':'120201'})
            validate_config(values)
            self.assertIn('final-benchmark',values['task_id'])
            self.assertEqual('generation_schedule' in values,dataset=='cf')
            if dataset=='zsre':
                for endpoint,edits,requests in [('W0_first2000',0,2000),('current/post',500,100),('all_seen/post',500,500),('all_seen/post',2000,2000)]:
                    summary=dict(raw['summary'],requests=requests)
                    payload=official_zsre_metrics(summary,config_values=values,endpoint=endpoint,edits=edits,post_state_edits=edits)
                    metrics(payload,config_values=values)

if __name__=='__main__':unittest.main()
