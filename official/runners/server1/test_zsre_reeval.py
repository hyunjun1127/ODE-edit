import copy
import unittest
import torch
from .zsre_reeval_restore import restore_payload

class RestoreTests(unittest.TestCase):
    def fixture(self):
        m=torch.nn.Module();m.layers=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=True)])
        original=dict(identity={'source':'fixture'},method='MEMIT',hparams={'layers':[0],'rewrite_module_tmp':'layers.{}'},expected_history=False)
        p=dict(schema='official-baseline-checkpoint-v1',batch=20,identity=original['identity'],method='MEMIT',
            evaluation_cursor={'completed_batch':20},contexts={'successful_calls':20},cache_c={},weights={'layers.0.weight':torch.ones(2,2)})
        return m,p,original
    def test_restore_only_selected(self):
        m,p,o=self.fixture();bias=m.layers[0].bias.clone();r=restore_payload(m,p,o)
        self.assertTrue(torch.equal(m.layers[0].weight,torch.ones(2,2)));self.assertTrue(torch.equal(m.layers[0].bias,bias));self.assertTrue(r['no_native_apply'])
    def test_partial_rejected(self):
        m,p,o=self.fixture();p['batch']=19
        with self.assertRaises(ValueError):restore_payload(m,p,o)
    def test_identity_rejected(self):
        m,p,o=self.fixture();p['identity']={}
        with self.assertRaises(ValueError):restore_payload(m,p,o)
    def test_invalid_tensor_before_mutation(self):
        m,p,o=self.fixture();before=m.layers[0].weight.clone();p['weights']['layers.0.weight'][0,0]=float('nan')
        with self.assertRaises(ValueError):restore_payload(m,p,o)
        self.assertTrue(torch.equal(m.layers[0].weight,before))
    def test_history_contract(self):
        m,p,o=self.fixture();p['cache_c']={'0':torch.eye(2)}
        with self.assertRaises(ValueError):restore_payload(m,p,o)

if __name__=='__main__':unittest.main()
