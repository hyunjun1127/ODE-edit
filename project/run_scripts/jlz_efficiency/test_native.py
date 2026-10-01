"""Production cached-adapter fixtures; no actual-pretrained claim."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from . import native

class NativeAdapter(unittest.TestCase):
    def test_exact_first_adam_and_independent_request_steps(self):
        torch.set_num_threads(1)
        hp=SimpleNamespace(layers=[4,5,6,7,8],v_lr=.1,clamp_norm_factor=.75)
        states=[]
        def loss(d):return (d-1).square().sum()+.2
        for i in range(4):
            d=torch.zeros(3,requires_grad=True);opt=torch.optim.Adam([d],lr=.1)
            s=dict(initial=torch.ones(3)*10,teacher=torch.zeros(1,5),near_clamp=False,
                request={'case_id':i},prefix={},trace=[],updates=0,clamp=[],gradients={})
            for n in range(25):
                opt.zero_grad();value=loss(d);s['trace'].append(dict(iteration=n,total=float(value.detach())))
                if n==24:break
                value.backward();s['updates']+=1
                if n in (0,1):s['gradients'][n]=d.grad.detach().clone()
                opt.step();s['clamp'].append(False)
                if n==0:s['first_delta']=d.detach().clone();s['first_adam']=copy.deepcopy(opt.state_dict())
            s['final_delta']=d.detach().clone();states.append(s)
        def losses(model,hp,b,lh,fh,delta,initial,teacher):
            values=(delta-1).square().sum(1)+.2
            return values,values,values*0,values*0,None
        model=torch.nn.Linear(2,2)
        with patch.object(native,'prepare_batch',return_value={'tokens':{}}),patch.object(native,'capture_prefix',return_value={}),patch.object(native,'suffix_hidden',return_value=(None,None)),patch.object(native,'native_losses',side_effect=losses):
            for batched in (False,True):
                result=native.cached(model,None,states,hp,None,[],batched)
                self.assertEqual(result['status'],'PASS')
                self.assertEqual([r['candidates'] for r in result['requests']],[25]*4)
                self.assertEqual([r['updates'] for r in result['requests']],[24]*4)
                self.assertEqual(result['extra_reference_request_backwards'],0)
            changed=copy.deepcopy(states);changed[0]['near_clamp']=True
            self.assertEqual(native.cached(model,None,changed,hp,None,[],False)['status'],'UNQUALIFIED')
    def test_actual_ast_and_tensor_tuple_interface_cleanup(self):
        compute,lookup,hp=native.load_native(lambda *a:None,lambda *a:None)
        self.assertEqual(hp.v_num_grad_steps,25);self.assertTrue(callable(compute))
        m=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False));x=torch.ones(1,2);before=m(x)
        with native.TraceDict(m,['0'],edit_output=lambda out,name:(out[0]+1,)) as trace:
            self.assertTrue(torch.equal(m(x),before+1));self.assertTrue(torch.equal(trace['0'].output[0],before+1))
        self.assertTrue(torch.equal(m(x),before));self.assertFalse(m[0]._forward_hooks)

if __name__=='__main__':unittest.main()
