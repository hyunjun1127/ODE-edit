"""Actual production loop CPU mocks; not Llama or GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from . import run, record_only as policy, collect
from .budget import PanelBudget


class RecordOnly(unittest.TestCase):
    def comparison(self):
        return dict(status='UNQUALIFIED',same_materialization=True,
            errors=dict(gradient_block=1.3612662244489588e-5,gradient_component=1.5020370483398438e-5,prox_abs=7.819800487141038e-6),
            limits=dict(gradient_block=1e-5,gradient_component=1e-5,prox_abs=5e-6))

    def test_original_failure_warning_and_retained_hard_checks(self):
        c=self.comparison();d=policy.point_decision(c,dict(feasible=True))
        self.assertEqual(c['status'],'UNQUALIFIED')
        self.assertEqual(d['original_verdict'],'FAIL')
        self.assertEqual(d['effective_action'],'CONTINUE_WITH_WARNING')
        for point,comparison in [(dict(feasible=False),c),(dict(feasible=True),c|{'same_materialization':False})]:
            with self.assertRaises(RuntimeError):policy.point_decision(comparison,point)
        with self.assertRaises(FloatingPointError):policy.point_decision(c,dict(feasible=True,smooth=float('nan')))
        with self.assertRaises(RuntimeError):policy.hard(False,'RESTORE')

    def rows(self):
        return [dict(identity=f'{kind}-{i}',case_id=i//n,kind=kind,prompt_index=i%n,
            true_nll=2.,new_nll=1.,true_token_count=1,new_token_count=1,
            true_token_correct=0,new_token_correct=1,true_strict=False,new_strict=True,
            true_predictions=[0],new_predictions=[1]) for kind,n in [('R',1),('P',2),('N',10)] for i in range(100*n)]

    def test_observer_identity_remains_fatal(self):
        a=dict(rows=self.rows());b=copy.deepcopy(a);b['rows'][0]['identity']='wrong'
        with self.assertRaisesRegex(RuntimeError,'IDENTITY'):policy.observer_identity(a,b)
        b=copy.deepcopy(a);b['rows'][0]['true_nll']=float('inf')
        with self.assertRaises(FloatingPointError):policy.observer_identity(a,b)

    def test_production_eight_calls_observer_collector_despite_numerical_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output';out.mkdir();prior=root/'prior';(prior/'output').mkdir(parents=True)
            (prior/'output/B100-entry.json').write_text(json.dumps(dict(case_ids=list(range(100)),input_identity='fixture')))
            entrycmp=dict(status='UNQUALIFIED',key_bitwise=True,active_exact=True,anchor_maxabs=.002,teacher_maxabs=.002,nll_maxabs=.002)
            panel=dict(c=torch.ones(500),rho=torch.ones(500),mask=torch.ones(500,dtype=torch.bool),
                adj={l:torch.zeros(2,100,dtype=torch.float64) for l in run.LAYERS},
                entry_receipt=dict(case_ids=list(range(100)),input_identity='fixture',comparison=entrycmp))
            W={l:torch.zeros(2,2) for l in run.LAYERS};hashes={str(l):'w' for l in run.LAYERS}
            point=dict(feasible=True,weight_sha=hashes,point_sha='x',smooth=1.,nll=[1.]*100,kl=[0.]*100)
            class Oracle:
                def __init__(self):self.records=[]
                def __call__(self,x):
                    self.records.append(dict(seconds=1.));return 1.,torch.zeros_like(x),dict(weights=W)
            class Txn:
                restored=True
                def __init__(self,*args):pass
                def __enter__(self):return self
                def __exit__(self,*args):return False
            obs=dict(rows=self.rows(),seconds=1.,work={},fallback_full_groups=0)
            obscmp=dict(status='UNQUALIFIED',discrete_exact=False,nll_maxabs=.002,logit_maxabs=.002,logit_rms=.002)
            budget=PanelBudget()
            with patch.object(run,'entry',return_value=panel),patch.object(run,'direction',return_value=torch.zeros(500,2)),\
                 patch.object(run,'make_oracle',side_effect=lambda *args:Oracle()),patch.object(run,'point_record',side_effect=lambda *args:(point,torch.zeros(500,2))),\
                 patch.object(run,'compare_points',side_effect=lambda *args:self.comparison()),patch.object(run,'Transaction',Txn),\
                 patch.object(run,'weights',return_value=W),patch.object(run,'state_hash',return_value={'W':hashes}),\
                 patch.object(run.evaluation,'evaluate',return_value=obs),patch.object(run.evaluation,'compare',return_value=obscmp):
                result=run.large(SimpleNamespace(config=SimpleNamespace(hidden_size=2)),None,[{}]*100,[],{},None,budget,'E123_MB4',out,record_only={'root':str(prior)})
            self.assertEqual(budget.counts['B100'],8)
            self.assertEqual(len(result['observer_rows']),8)
            self.assertEqual(result['numerical_certification'],'NOT_ESTABLISHED')
            self.assertTrue((out/'CONTINUATION_INITIAL_VALID.json').is_file())
            self.assertTrue(all(r['comparison']['original_verdict']=='FAIL' for r in result['oracle_rows'][1:]))
            (root/'execution.lock.json').write_text('{}')
            (out/'terminal.json').write_text(json.dumps(dict(status='COMPLETED_RECORD_ONLY_BENCHMARK',budget=budget.report(),numerical_policy='RECORD_ONLY_USER_DIRECTED')))
            with patch('sys.argv',['collect','--run',str(root),'--job','MOCK']),patch.object(collect.subprocess,'run',return_value=SimpleNamespace(stdout='CPU_MOCK',stderr='')):
                collect.main()
            receipt=json.loads((root/'collected/receipt.json').read_text())
            self.assertEqual(receipt['independent_reducer_errors'],[])
            self.assertEqual(receipt['budget']['B100'],8)
            self.assertGreater(receipt['warning_count'],0)
            self.assertEqual(receipt['numerical_certification'],'NOT_ESTABLISHED')


if __name__=='__main__':unittest.main()
