"""CPU-only regressions for the actual controller-to-runtime boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from project.run_scripts.single_layer_edit_preserving_correction.technical import require,TechnicalHold
from project.run_scripts.single_layer_edit_preserving_correction.binding import quality_ok
from project.run_scripts.single_layer_edit_preserving_correction.runner import ideal_check,selection_seal,reuse_observation

class IntegrationBoundaryTests(unittest.TestCase):
    def test_projector_status_receipt_collision_regression(self):
        with tempfile.TemporaryDirectory() as d:
            ref=require(Path(d),'projector',True,{'status':'PASS','relative_idempotence':1e-15})
            self.assertEqual(json.loads(Path(ref['path']).read_text())['check_status'],'PASS')
    def test_failure_receipt_precedes_raise(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TechnicalHold):require(Path(d),'projector',False,{'status':'FAIL'})
            self.assertEqual(json.loads((Path(d)/'projector.json').read_text())['check_status'],'FAIL')
    def test_per_sequence_not_mean_or_count(self):
        a={'1:canonical:new':dict(nll=.1,strict=True,kind='canonical',branch='new'),
           '1:canonical:old':dict(nll=.2,strict=False,kind='canonical',branch='old'),
           '2:canonical:new':dict(nll=2.,strict=False,kind='canonical',branch='new')}
        c={k:dict(v) for k,v in a.items()};c['1:canonical:new']['nll']=.101;c['2:canonical:new']['nll']=.1
        self.assertFalse(quality_ok(c,a)[0])
        c={k:dict(v) for k,v in a.items()};c['1:canonical:old']['nll']=.05
        self.assertIn(('1:canonical:new','PAIR_ID_LOST'),quality_ok(c,a)[1])
    def test_null_proposal_checker_uses_ideal(self):
        K=torch.tensor([[1.],[0.]],dtype=torch.float32)
        self.assertTrue(ideal_check(torch.tensor([[0.,2.]],dtype=torch.float64),K).passed)
        self.assertFalse(ideal_check(torch.tensor([[.1,2.]],dtype=torch.float64),K).passed)
    def test_seal_endpoint_and_order(self):
        seal=selection_seal('b001','EN-F',torch.tensor([[1.]]),[2,1],'f'*64)
        self.assertEqual(seal['status'],'SELECTION_SEALED');self.assertEqual(seal['endpoint_id'],'EN-F')
    def test_exact_postseal_observer_alias_preserves_rows_not_cost(self):
        compatibility={'endpoint_weight_sha256':'a'*64,'request_order_sha256':'b'*64}
        prior={'compatibility':compatibility,'metrics':{'RS':{'numerator':3,'denominator':100}},
               'work':{'model_forward_calls':18,'observer_seconds':3.2},'raw_token_payload':'NOT_RETAINED_PRIOR'}
        seal={'status':'SELECTION_SEALED','endpoint_weight_sha256':'a'*64,'endpoint_id':'N4'}
        result=reuse_observation(prior,{'path':'prior.json','sha256':'c'*64},seal,compatibility)
        self.assertEqual(result['metrics'],prior['metrics']);self.assertEqual(result['new_observer_forwards'],0)
        self.assertEqual(result['raw_token_payload'],'NOT_RETAINED_PRIOR')
        self.assertEqual(result['work']['model_forward_calls'],0);self.assertEqual(prior['work']['model_forward_calls'],18)
        with self.assertRaises(ValueError):reuse_observation(prior,{},seal,{**compatibility,'endpoint_weight_sha256':'d'*64})
        with self.assertRaises(ValueError):reuse_observation(prior,{},dict(seal,status='UNSEALED'),compatibility)

if __name__=='__main__':unittest.main()
