"""Tiny random Llama CPU adapter regression, explicitly NOT actual model evidence."""
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from .profile import LlamaAdapter
from .causal_builder import build
from .qualification import full_native_reference,singleton_permuted

class Adapter(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        cfg=LlamaConfig(hidden_size=8,intermediate_size=12,num_hidden_layers=4,num_attention_heads=2,
            num_key_value_heads=2,vocab_size=32,max_position_embeddings=64,attention_dropout=0.)
        cfg._attn_implementation='eager';self.a=LlamaAdapter(LlamaForCausalLM(cfg),dict(eligible_layers=[0,1,2],nll_layer=3))
        self.tokens=dict(input_ids=torch.tensor([[1,2,3,4],[1,5,6,7]]),attention_mask=torch.ones(2,4,dtype=torch.long))
        self.rows=[dict(kind='rewrite',request=j,lookup=1,global_row=j) for j in range(2)]
        self.group=dict(rows=self.rows,tokens=self.tokens,cache=self.a.prefix(self.tokens))
        self.D={l:(torch.randn(8,2)*.01).requires_grad_(True) for l in self.a.sites}
        self.e=dict(groups=[self.group],pack=dict(n_requests=2,key_context_weights=[1.,1.],key_request=[0,1]),
            factors={l:torch.eye(12,dtype=torch.float64) for l in self.a.sites},
            entry_weights={l:w.detach().clone() for l,w in self.a.weights.items()},first_geometry=None)
    def test_native_cached_full(self):
        left=self.a.native(self.group,self.D,True)
        with full_native_reference(self.a):right=self.a.native(self.group,self.D,True)
        torch.testing.assert_close(left[0],right[0],atol=1e-6,rtol=1e-5)
        for l in self.a.sites:torch.testing.assert_close(left[2][l],right[2][l],atol=1e-6,rtol=1e-5)
        gl=torch.autograd.grad(left[0].square().sum(),tuple(self.D.values()),retain_graph=True)
        gr=torch.autograd.grad(right[0].square().sum(),tuple(self.D.values()))
        for x,y in zip(gl,gr):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    def test_builder_final_and_backward(self):
        built=build(self.a,self.e,self.D,1)
        n,f,h,keys=self.a.actual(self.group,self.D,built['P'],built['weights'],capture=True)
        for l in self.a.sites:torch.testing.assert_close(keys[l].T.double(),built['geometry'][l]['K'],atol=1e-6,rtol=1e-5)
        with self.a.install(self.D,built['P'],built['weights'],route='dense'):
            ref,_=self.a.full(self.tokens)
        torch.testing.assert_close(n,ref,atol=1e-6,rtol=1e-5)
        g=torch.autograd.grad(n.square().sum()+sum(x['G'].sum() for x in built['geometry'].values()),tuple(self.D.values()))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in g))
        self.assertTrue(built['P'][1].requires_grad)
    def test_wholeB_context_permutation_and_microbatch(self):
        a=build(self.a,self.e,self.D,1)
        b=build(self.a,singleton_permuted(self.e),self.D,1)
        for l in self.a.sites:
            torch.testing.assert_close(a['P'][l],b['P'][l],atol=1e-8,rtol=1e-5)
        ga=torch.autograd.grad(sum(g['P'].square().sum() for g in a['geometry'].values()),tuple(self.D.values()),allow_unused=True)
        gb=torch.autograd.grad(sum(g['P'].square().sum() for g in b['geometry'].values()),tuple(self.D.values()),allow_unused=True)
        for x,y in zip(ga,gb):
            if x is not None:torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-4)

if __name__=='__main__':unittest.main()
