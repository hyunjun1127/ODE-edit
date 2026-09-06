import unittest
from types import SimpleNamespace
import torch
from project.run_scripts.ordered_response_barrier_ode.runtime import _tensor_content_state_identity
from .state_identity import content_identity,verify_warm_state


class StateIdentityTests(unittest.TestCase):
    def test_pointer_is_not_a_portable_content_identity(self):
        a=torch.arange(6,dtype=torch.float32).reshape(2,3);b=a.clone()
        old=lambda value:_tensor_content_state_identity('same',{'cache':value},include_version=False)
        self.assertNotEqual(old(a),old(b))
        self.assertEqual(content_identity({'cache':a}),content_identity({'cache':b}))
        b[0,0]+=1
        self.assertNotEqual(content_identity({'cache':a}),content_identity({'cache':b}))

    def test_exact_warm_bytes_and_static_cov_guards(self):
        cache=torch.arange(6,dtype=torch.float32).reshape(2,3)
        warm={'alpha_cache':cache.clone(),'commit':{'committed_method_state_sha256':'old-address-dependent'}}
        f=SimpleNamespace(family='AlphaEdit',module=SimpleNamespace(cache_c=cache.clone()),method_state_identity=lambda:'new-address-dependent')
        self.assertEqual(verify_warm_state(f,warm,None)['status'],'PASS')
        f.module.cache_c[0,0]+=1
        with self.assertRaises(RuntimeError):verify_warm_state(f,warm,None)
        f.family='MEMIT';f.module.COV_CACHE={'layer4':cache.clone()}
        identity=content_identity(f.module.COV_CACHE)
        self.assertEqual(verify_warm_state(f,warm,identity)['status'],'PASS')
        f.module.COV_CACHE['layer4'][0,0]+=1
        with self.assertRaises(RuntimeError):verify_warm_state(f,warm,identity)
