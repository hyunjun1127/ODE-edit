"""Small CPU fixtures only; never model/GPU qualification."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from official.baselines import registry
from official.baselines.memit_fe_history import history_solve,apply_memit_fe_history_to_model,native
from official.experiments import checkpoint

class HistoryTests(unittest.TestCase):
    def test_zero_exact_native(self):
        cov=torch.tensor([[2.,.5],[.5,3.]])
        k=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        self.assertTrue(torch.equal(history_solve(cov,k,15000,torch.zeros(2,2)),
                                   torch.linalg.solve(15000*cov.double()+k@k.T,k)))

    def test_nonzero_additive(self):
        cov=torch.eye(2);k=torch.ones(2,1,dtype=torch.float64);h=torch.eye(2)*3
        self.assertTrue(torch.equal(history_solve(cov,k,5,h),torch.linalg.solve(5*cov.double()+h.double()+k@k.T,k)))

    def test_invalid_history(self):
        for h in (torch.zeros(3,3),torch.zeros(2,2,dtype=torch.float64),torch.full((2,2),float('nan'))):
            with self.assertRaises(ValueError):history_solve(torch.eye(2),torch.ones(2,1,dtype=torch.float64),5,h)

    def fixture(self):
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False)])
        hp=SimpleNamespace(layers=[0,1],rewrite_module_tmp='layers.{}')
        h={'0':torch.zeros(2,2),'1':torch.zeros(2,2)}
        return model,hp,h,[dict(prompt='{} is',subject='X',target_new='Y')]

    def test_append_own_final_once(self):
        m,hp,h,r=self.fixture()
        def apply(model,*a,**kw):
            with torch.no_grad():
                for p in model.parameters():p.add_(1)
            return model,{}
        with patch.object(native,'apply_memit_FE_to_model',side_effect=apply),patch.object(native,'get_context_templates',return_value=[['{}']]),patch.object(native,'compute_ks',return_value=torch.tensor([[2.,3.]])) as ks:
            apply_memit_fe_history_to_model(m,None,r,hp,history=h)
        self.assertEqual(ks.call_count,2)
        self.assertTrue(torch.equal(h['0'],torch.tensor([[4.,6.],[6.,9.]])))

    def test_failure_no_history_advance_restore_weights(self):
        m,hp,h,r=self.fixture();before={n:p.clone() for n,p in m.named_parameters()}
        def bad(model,*a,**kw):
            with torch.no_grad():next(model.parameters()).add_(2)
            raise ValueError('fixture failure')
        with patch.object(native,'apply_memit_FE_to_model',side_effect=bad):
            with self.assertRaises(ValueError):apply_memit_fe_history_to_model(m,None,r,hp,history=h)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in m.named_parameters()))
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in h.values()))

    def test_key_failure_no_partial_history(self):
        m,hp,h,r=self.fixture()
        with patch.object(native,'apply_memit_FE_to_model',return_value=(m,{})),patch.object(native,'get_context_templates',return_value=[['{}']]),patch.object(native,'compute_ks',side_effect=[torch.ones(1,2),ValueError('last layer failed')]):
            with self.assertRaises(ValueError):apply_memit_fe_history_to_model(m,None,r,hp,history=h)
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in h.values()))

    def test_native_profiles_unchanged(self):
        for model in ('llama3','gptj','qwen25'):
            self.assertEqual(vars(registry.hparams('MEMIT_FE',model)),vars(registry.hparams('MEMIT_FE_HISTORY',model)))
        self.assertNotEqual(registry.implementation('MEMIT_FE','llama3')[0],registry.implementation('MEMIT_FE_HISTORY','llama3')[0])

    def test_checkpoint_history_cursor_restore(self):
        identity={k:'fixture-'+k for k in checkpoint.IDENTITY_FIELDS}
        with tempfile.TemporaryDirectory() as p:
            checkpoint.save(p,batch=0,weights={'w':torch.ones(2,2)},cache_c={'0':torch.zeros(2,2)},contexts={'successful_calls':0},
                evaluation_cursor={'completed_batch':0},identity=identity,method='MEMIT_FE_HISTORY',evaluation_complete=True)
            checkpoint.save(p,batch=1,weights={'w':torch.ones(2,2)*2},cache_c={'0':torch.eye(2)},contexts={'successful_calls':1},
                evaluation_cursor={'completed_batch':1},identity=identity,method='MEMIT_FE_HISTORY',evaluation_complete=True)
            value=checkpoint.load(p,identity)
            self.assertTrue(torch.equal(value['cache_c']['0'],torch.eye(2)))
            self.assertEqual(value['batch'],value['evaluation_cursor']['completed_batch'])

    def test_stock_checkpoint_rejects_history(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):checkpoint.save(p,batch=0,weights={'w':torch.ones(1)},cache_c={'0':torch.ones(1)},contexts={},
                evaluation_cursor={},identity={k:'x' for k in checkpoint.IDENTITY_FIELDS},method='MEMIT_FE',evaluation_complete=True)

    def test_cap_uses_free_lanes(self):
        from .fe_history_submit import scheduling
        graph,width=scheduling({'jobs':[dict(key='old',gpus=1,parents=[])]},['gptj','llama3','qwen25'],4)
        self.assertEqual(width,4);self.assertTrue(all(not p for p in graph.values()))

    def test_cap_serializes_when_full(self):
        from .fe_history_submit import scheduling
        graph,width=scheduling({'jobs':[dict(key=str(i),gpus=1,parents=[]) for i in range(4)]},['gptj','llama3','qwen25'],4)
        self.assertEqual(width,4);self.assertTrue(all(p for p in graph.values()))

    def test_new_method_tracking_identity(self):
        from official.tracking.schema import config
        for model in ('llama3','gptj','qwen25'):
            value=dict(server='server1',task_id='memit-fe-history-three-model-2k',model=model,model_family=model,
                writer='memit_fe_history',baseline='MEMIT_FE_HISTORY',role='scientific',arm='cf-MEMIT_FE_HISTORY',attempt='r1',
                source_sha='a'*40,config_sha='b'*64,dataset='cf',metric_schema='official-baselines-scalar-v1',
                instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',generation_schedule='DEFERRED_CHECKPOINT_EVALUATION')
            self.assertEqual(config(value)['baseline'],'MEMIT_FE_HISTORY')

if __name__=='__main__':unittest.main()
