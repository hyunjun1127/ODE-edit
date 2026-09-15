import copy
import json
from pathlib import Path
import unittest
import torch
from project.run_scripts.single_layer_zflow.config import validate_config,validate_main
from project.run_scripts.single_layer_zflow.runtime import state_fingerprint


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

if __name__=='__main__':unittest.main()
