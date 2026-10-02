"""CPU fixtures only, not actual Llama-8B qualification."""
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from .physical import PhysicalLinear,materialize,LlamaAdapter

class PhysicalTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(14);torch.set_num_threads(1)
    def test_direct_dense_input_and_D(self):
        for B in (1,3):
            x=torch.randn(2,5,7,requires_grad=True);d=torch.randn(4,B,requires_grad=True)
            p=torch.randn(7,B,dtype=torch.float64);w=torch.randn(4,7)
            we=materialize(w,d,p);g=torch.randn(2,5,4)
            yd=torch.nn.functional.linear(x,we);gd=torch.autograd.grad((yd*g).sum(),(x,d))
            yc=PhysicalLinear.apply(x,d,p,we.detach());gc=torch.autograd.grad((yc*g).sum(),(x,d))
            self.assertTrue(torch.equal(yd,yc))
            for a,b in zip(gd,gc):torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-6)

    def test_alltoken_crosslayer_cache_zero_nonzero(self):
        cfg=LlamaConfig(vocab_size=43,hidden_size=16,intermediate_size=24,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
        cfg._attn_implementation='eager';m=LlamaForCausalLM(cfg).float().eval()
        a=LlamaAdapter(m,dict(eligible_layers=[1,2],nll_layer=3))
        tok=dict(input_ids=torch.tensor([[1,2,3,4],[5,6,0,0]]),attention_mask=torch.tensor([[1,1,1,1],[1,1,0,0]]))
        cache=a.prefix(tok);guard=a.guard();hooks=a.hook_signature()
        for scale in (0.,.03):
            d={l:(scale*torch.randn(16,2)).requires_grad_() for l in a.sites}
            p={l:torch.randn(24,2,dtype=torch.float64) for l in a.sites}
            w={l:materialize(a.weights[l],d[l],p[l]) for l in a.sites}
            with a.install(d,p,w,'dense'):full,_=a.full(tok)
            gf=torch.autograd.grad(full[:,-1].square().sum(),tuple(d.values()))
            with a.install(d,p,{l:x.detach() for l,x in w.items()},'direct'):cached,_=a.cached(cache)
            gc=torch.autograd.grad(cached[:,-1].square().sum(),tuple(d.values()))
            torch.testing.assert_close(full,cached,atol=1e-6,rtol=2e-5)
            for g,h in zip(gf,gc):
                torch.testing.assert_close(g,h,atol=1e-5,rtol=2e-4)
                self.assertGreater(float(g.norm()),0)
        self.assertEqual(a.guard(),guard);self.assertEqual(a.hook_signature(),hooks)

if __name__=='__main__':unittest.main()
