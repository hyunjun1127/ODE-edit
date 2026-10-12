import contextlib
import copy
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import rescue as r,rescue_observer as o,observer as base,contract,prepare,benchmark
from official.diagnostics.fe_qwen_current.patch import function_text
from official.experiments.prepare import write_new,digest
from official.runners.server1.common import read
from official.tracking.schema import config as validate,bind_job_identity
from official.tests.test_fe_native import config,tokenize


class RescueTests(unittest.TestCase):
    def test_actual_prepare_lock_expression_uses_active_spec(self):
        import ast
        tree=ast.parse(Path(prepare.__file__).read_text())
        expressions=[n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='promotion_candidates']
        self.assertEqual(len(expressions),1)
        code=compile(ast.Expression(expressions[0]),'<actual-prepare-lock-expression>','eval')
        self.assertEqual(eval(code,{'SPEC':contract.SPEC}),[.1,1,10])
        self.assertEqual(eval(code,{'SPEC':r.SPEC}),[100])

    def test_prospective_single_candidate_and_old_contract_unchanged(self):
        old=copy.deepcopy(contract.SPEC)
        with r.bindings():
            self.assertEqual(prepare.SPEC['L2_arms'],[100])
            self.assertIs(benchmark.validated_selection,r.validated_selection)
        self.assertEqual(contract.SPEC,old);self.assertEqual(prepare.SPEC['L2_arms'],[0,.1,1,10])
        self.assertEqual(r.SPEC['clamp'],1)
    def test_absolute_gate_no_partial_no_best_failed_promotion(self):
        v=dict(Efficacy=96,Generalization=90,Specificity=80,requests=2000,committed_batches=20)
        v['Score']=3/sum(1/v[k] for k in ('Efficacy','Generalization','Specificity'))
        self.assertEqual(r.select_complete(v)['selected_L2'],100)
        bad=dict(v,Specificity=69);bad['Score']=3/sum(1/bad[k] for k in ('Efficacy','Generalization','Specificity'))
        self.assertEqual(r.select_complete(bad)['status'],'NO_PROMOTION')
        for bad in (dict(v,requests=1900),dict(v,committed_batches=19),dict(v,Score=float('nan'))):
            with self.assertRaises(ValueError):r.select_complete(bad)
    def test_exact_quality_thresholds(self):
        v=dict(Efficacy=80,Generalization=70,Specificity=65,requests=100)
        self.assertEqual(r.quality_stop(v),{})
        for key in r.SPEC['early_stop']:
            self.assertIn(key,r.quality_stop(dict(v,**{key:v[key]-.001})))
        with self.assertRaises(ValueError):r.quality_stop(dict(v,Specificity=float('nan')))
    def test_stop_after_commit_preserves_checkpoint_and_not_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cp=out/'checkpoints/latest.pt';cp.parent.mkdir();cp.write_bytes(b'prior-B5')
            state=dict(c={},out=out,identity={'source':'fixture'},last_batch=5,finished=False,
                tracker=SimpleNamespace(finish=lambda **kw:None))
            def committed(*args):
                state['last_batch']=6
                write_new(out/'factual/B06-current.json',dict(summary=dict(requests=100,Efficacy=79,Generalization=90,Specificity=80)))
                write_new(out/'commits/B06.json',dict(batch=6))
            import sys
            previous=sys.excepthook
            try:
                with patch.object(base,'_STATE',state),patch.object(base,'snapshot',side_effect=committed),self.assertRaises(o.QualityEarlyStop):
                    o.snapshot(None,None,None,None,None,6,None)
            finally:sys.excepthook=previous
            self.assertEqual(read(out/'QUALITY_EARLY_STOP.json')['status'],'QUALITY_EARLY_STOP')
            self.assertTrue((out/'commits/B06.json').exists());self.assertFalse((out/'FAILURE.json').exists())
            self.assertEqual(cp.read_bytes(),b'prior-B5')
    def test_solve_diagnostic_exact_and_no_rng_mutation(self):
        A=torch.tensor([[4.,1.],[1.,3.]],dtype=torch.float64);B=torch.arange(12,dtype=torch.float64).reshape(2,6)
        X=torch.linalg.solve(A,B);before=(A.clone(),B.clone(),X.clone(),torch.get_rng_state().clone())
        result=o.solve_diagnostics(A,B,X)
        self.assertLess(result['solve_relative_residual_sampled'],1e-14)
        self.assertAlmostEqual(result['rhs_frobenius'],torch.linalg.vector_norm(B).item())
        self.assertFalse(result['full_solve_residual_measured'])
        for a,b in zip((A,B,X,torch.get_rng_state()),before):self.assertTrue(torch.equal(a,b))
        self.assertAlmostEqual(o.norm(B.T),torch.linalg.vector_norm(B.T).item())
    def test_tracking_new_group_both_datasets(self):
        c=dict(server='rent',L2=100,attempt='r1',config_sha256='a'*64,
            selection_contract_sha256=digest(r.SPEC),stream=dict(sha256='b'*64))
        with patch.dict(os.environ,{},clear=True):
            v=bind_job_identity(r.tracking_config(c,'c'*40),{'ODEEDIT_RENT_JOB_ID':'120832'})
            validate(v);self.assertIn('rescue',v['task_id']);self.assertEqual(v['dataset'],'cf')
            for dataset in ('cf','zsre'):
                f=dict(c,stage='final_benchmark',dataset=dataset,selected_result=dict(sha256='d'*64))
                v=bind_job_identity(r.tracking_config(f,'c'*40),{'ODEEDIT_RENT_JOB_ID':'120832'})
                validate(v);self.assertIn('rescue-selected-final',v['task_id'])
                if dataset=='zsre':self.assertNotIn('generation_schedule',v)
    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'),'native pin required')
    def test_real_native_writer_observation_exact_W_H_RNG(self):
        root=Path(os.environ['FE_NATIVE_AUTHOR_ROOT']);text=(root/'algs/memit/FE-memit_main.py').read_text()
        changed=r.native_source(text)
        self.assertEqual(function_text(text,'batch_edit'),function_text(changed,'batch_edit'))
        self.assertIn('rescue_observer import',changed)
        K=torch.arange(300,dtype=torch.float32).reshape(100,3)/300
        ns=dict(torch=torch,nethook=SimpleNamespace(get_parameter=lambda m,n:m.get_parameter(n)),
            get_context_templates=lambda *a:[['{}']],compute_ks=lambda *a:K.clone(),
            time=__import__('time'),covs=[torch.eye(3) for _ in range(5)],
            get_module_input_output_at_words=lambda m,t,l,**kw:(None,torch.zeros(100,2)))
        exec(function_text(text,'upd_matrix_match_shape')+'\n'+function_text(text,'batch_edit'),ns)
        def model():
            m=torch.nn.Module();m.layers=torch.nn.ModuleList([torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16) for _ in range(9)])
            with torch.no_grad():
                for p in m.parameters():p.fill_(.25)
            return m
        a=model();b=model();cfg=config();cfg.bs=100;cfg.llms.layers=r.SPEC['layers'];cfg.llms.mom2_update_weight=[2.]*5;cfg.algs.L2=100
        requests=[dict(case_id=i,subject='x',prompt='{}',sample_idx=i) for i in range(100)]
        z={l:torch.ones(2,100)*(l+1) for l in r.SPEC['layers']};ha=torch.zeros(5,3,3);hb=ha.clone()
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            ns['batch_edit'](cfg,a,tokenize,requests,'cpu',ha,z);rng=torch.get_rng_state().clone()
            with patch.object(base,'_STATE',dict(out=Path(d),last_batch=0)):
                o.observed_write(ns['batch_edit'],cfg,b,tokenize,requests,'cpu',hb,z,1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(ha,hb))
            for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
            for row in read(Path(d)/'writer/B01.json')['layers']:
                self.assertLess(row['solve_relative_residual_sampled'],1e-12)
                self.assertAlmostEqual(row['post_weight_norm'],o.norm(dict(b.named_parameters())[row['weight_name']]))


if __name__=='__main__':unittest.main()
