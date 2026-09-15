import copy
import json
from pathlib import Path
import unittest
import torch
from project.run_scripts.single_layer_zflow.config import validate_config,validate_main
from project.run_scripts.single_layer_zflow.runtime import state_fingerprint,tokenizer_metadata


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[4]/'plans/global/2026-09-16-single-layer-zflow-contract-v1.json'
        self.c=json.loads(p.read_text())
    def test_main_exact(self):self.assertEqual(validate_main(self.c),self.c)
    def test_invalid_api_cannot_silently_disable_barrier(self):
        for mode,budget in [('off',1),('fixed',None),('exponential',None),('fixed',0),('fixed',-1)]:
            c=copy.deepcopy(self.c);c['integrator'].update(barrier_mode=mode,budget=budget)
            with self.subTest(mode=mode,budget=budget),self.assertRaises(ValueError):validate_config(c)
    def test_toy_default_or_native_endpoint_rejected(self):
        c=copy.deepcopy(self.c);c['integrator']['max_oracle_calls']=128
        with self.assertRaises(ValueError):validate_main(c)
        c=copy.deepcopy(self.c);c['runtime']['native_z_warm_start']=True
        with self.assertRaises(ValueError):validate_main(c)

    def test_exact_nested_rng_fingerprint(self):
        state={'python':(3,(1,2),None),'numpy':('MT19937',[1,2],1,0,0.0),
               'torch':torch.tensor([1,2],dtype=torch.uint8),'cuda':[torch.tensor([3],dtype=torch.uint8)]}
        clone=copy.deepcopy(state)
        self.assertEqual(state_fingerprint(state),state_fingerprint(clone))
        clone['cuda'][0][0]=4
        self.assertNotEqual(state_fingerprint(state),state_fingerprint(clone))
        clone=copy.deepcopy(state);clone['python']=list(clone['python'])
        self.assertNotEqual(state_fingerprint(state),state_fingerprint(clone))

    def test_fast_tokenizer_without_optional_bos_attribute(self):
        class Fast:
            padding_side='right';bos_token_id=1;pad_token_id=2
            def encode(self,text,add_special_tokens=True):return [1,3] if add_special_tokens else [3]
        token=Fast();before=dict(token.__dict__)
        result=tokenizer_metadata(token)
        self.assertFalse(result['add_bos_attribute_present'])
        self.assertIsNone(result['add_bos_attribute'])
        self.assertEqual(result['fixed_probe_default_ids'],[1,3])
        self.assertEqual(before,token.__dict__)

if __name__=='__main__':unittest.main()
