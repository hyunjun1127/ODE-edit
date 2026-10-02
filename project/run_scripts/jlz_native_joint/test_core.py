"""Bounded CPU synthetic tests, not actual Llama/experiment PASS."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from .adapter import LlamaAdapter
from .allocation import Geometry,policy
from .inputs import pool
from .oracle import Oracle
from .writer import Transaction,commit,state,capture_keys
from .observe import reduce_rows
from .optimize import fit

def fixture(B=2):
    torch.manual_seed(70)
    c=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=24,num_hidden_layers=4,
                  num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0)
    c._attn_implementation='eager'
    m=LlamaForCausalLM(c).float().eval();m.requires_grad_(False)
    a=LlamaAdapter(m,dict(eligible_layers=[0,1,2],nll_layer=3,kl_factor=.0625,norm_factor=.5,clamp_factor=.75,learning_rate=.1))
    n=3*B;ids=torch.randint(1,40,(n,7));mask=torch.ones_like(ids);lookup=[];targets=torch.full_like(ids,-100)
    for i in range(n):
        lookup.append(1+i%3)
        if i%3!=2:targets[i,-2:]=torch.tensor([5,8])
    rw=[i for i in range(n) if i%3!=2];kl=[i for i in range(n) if i%3==2]
    spec=dict(tokens=dict(input_ids=ids,attention_mask=mask),targets=targets,n_requests=B,n_rw=2,
        specs=[dict(n_rw=2) for _ in range(B)],row_request=[i//3 for i in range(n)],
        row_kind=['kl' if i%3==2 else 'rewrite' for i in range(n)],lookup=lookup,rw_rows=rw,
        canonical_rows=[3*i for i in range(B)],key_lookup=[lookup[i] for i in rw],
        key_tokens=dict(input_ids=ids[rw].clone(),attention_mask=mask[rw].clone()),
        context_group_slices=[(0,1),(1,2)],entry_key_prefix_exact=True,identity='fixture')
    H={l:torch.zeros(d[1],d[1]) for l,d in a.dims.items()}
    return a,spec,H

class Core(unittest.TestCase):
    def test_full_25_candidate_budget_and_native_clamp(self):
        for eta in (0,1):
            a,s,H=fixture()
            with tempfile.TemporaryDirectory() as tmp:
                paths={}
                for l in a.sites:
                    path=Path(tmp)/f'{l}.npz';np.savez(path,**{'mom2.mom2':np.eye(24,dtype=np.float32),'mom2.count':1});paths[str(l)]=str(path)
                geo=Geometry(a,H,paths,2)
                D,o,r=fit(a,s,geo,eta,Path(tmp)/'fit',qualification=True)
                self.assertEqual(r['candidates'],25);self.assertEqual(r['Adam_updates'],24)
                for l in D:self.assertTrue((D[l].norm(dim=0)<=.75*o.anchors[l].norm(dim=0)+1e-6).all())
                self.assertTrue(all(torch.count_nonzero(h)==0 for h in H.values()))

    def test_joint_zero_and_nonzero_prefix(self):
        a,s,H=fixture();o=Oracle(a,s,4,'strict_prefix')
        D={l:torch.zeros(d[0],2,requires_grad=True) for l,d in a.dims.items()}
        init=o.evaluate(D,initialize=True)
        self.assertTrue(all(d.grad is not None and d.grad.norm()>0 for d in D.values()))
        for d in D.values():
            with torch.no_grad():d.add_(torch.randn_like(d)*.01)
            d.grad=None
        cached=o.evaluate(D);g={l:d.grad.clone() for l,d in D.items()}
        for d in D.values():d.grad=None
        ref=o.evaluate(D,route='full_reference',full_head=True)
        self.assertLess(float((cached['nll']-ref['nll']).abs().max()),1e-5)
        self.assertLess(float((cached['kl']-ref['kl']).abs().max()),1e-5)
        for l,d in D.items():self.assertTrue(torch.allclose(g[l],d.grad,atol=1e-5,rtol=1e-4))

    def test_partial_microbatch(self):
        a,s,H=fixture(3);o=Oracle(a,s,4,'full_reference')
        D={l:torch.zeros(d[0],3,requires_grad=True) for l,d in a.dims.items()}
        x=o.evaluate(D,initialize=True);g={l:d.grad.clone() for l,d in D.items()}
        for d in D.values():d.grad=None
        single=Oracle(a,s,1,'full_reference');y=single.evaluate(D,initialize=True)
        self.assertTrue(torch.allclose(x['nll'],y['nll'],atol=1e-5))
        for l in D:self.assertTrue(torch.allclose(g[l],D[l].grad,atol=1e-5,rtol=1e-4))

    def test_geometry_all_B(self):
        for B in (1,3,24,31):
            a,s,H=fixture(1)
            with tempfile.TemporaryDirectory() as tmp:
                paths={}
                for l in a.sites:
                    path=Path(tmp)/f'{l}.npz';x=torch.randn(24,24);cov=x@x.T+torch.eye(24)
                    np.savez(path,**{'mom2.mom2':cov.numpy()*10,'mom2.count':10});paths[str(l)]=str(path)
                geo=Geometry(a,H,paths,2)
                K=torch.randn(24,B);g=geo.solve(0,K)
                A=geo.system(0);direct=torch.linalg.solve(A+K.double()@K.double().T,K.double())
                self.assertTrue(torch.allclose(g['P'],direct,atol=1e-10,rtol=1e-8))
                D=torch.randn(16,B,dtype=torch.float64,requires_grad=True)
                v,analytic,_=policy(D,g,torch.tensor(3.))
                self.assertTrue(torch.allclose(torch.autograd.grad(v,D)[0],analytic,atol=1e-10))

    def test_stream_writer_history_and_rollback(self):
        a,s,H=fixture();before=state(a,H)
        with Transaction(a,H) as tx:
            with torch.no_grad():a.weights[0].add_(.5);H[0].add_(1)
        self.assertTrue(tx.rollback_verified);self.assertEqual(before,state(a,H))
        with tempfile.TemporaryDirectory() as tmp:
            paths={}
            for l in a.sites:
                path=Path(tmp)/f'c{l}.npz';np.savez(path,**{'mom2.mom2':np.eye(24,dtype=np.float32),'mom2.count':1});paths[str(l)]=str(path)
            geo=Geometry(a,H,paths,2);D={l:torch.randn(d[0],2)*.01 for l,d in a.dims.items()}
            with Transaction(a,H) as tx:
                commit(a,s,D,geo,tx,Path(tmp)/'out',microbatch=4,qualification=True);tx.finish()
            keys=capture_keys(a,s,4)
            for l in a.sites:self.assertTrue(torch.allclose(H[l],keys[l]@keys[l].T,atol=1e-6,rtol=1e-4))

    def test_ties_fail_and_finite(self):
        row=dict(identity='x',kind='N',true_nll=1.,new_nll=1.,true_token_count=2,true_token_correct=1,
                 true_strict=False,new_strict=False)
        self.assertEqual(reduce_rows([row])['N']['numerator'],0)
        row['new_nll']=float('nan')
        with self.assertRaises(RuntimeError):reduce_rows([row])

if __name__=='__main__':unittest.main()
