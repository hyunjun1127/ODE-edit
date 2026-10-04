"""Production CPU algebra/state/reducer regressions, not GPU qualification."""
import ast,json,tempfile,unittest
from pathlib import Path
import torch
from .geometry import prepare_metric,solve_realized,targets,owner_weights,action_parity
from .telemetry import shares
from .collect import paired
from .common import ROOT,BRANCHES
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(13);torch.set_num_threads(2)
    def solve(self,n,b,d=3):
        raw=torch.randn(n,n,dtype=torch.float64);A,L,_=prepare_metric(raw@raw.T+torch.eye(n,dtype=torch.float64))
        K=torch.randn(n,b,dtype=torch.float64);T=torch.randn(d,b,dtype=torch.float64)
        U,r=solve_realized(A,L,K,T,torch.full((b,),1/b,dtype=torch.float64),'equality')
        return A,L,K,T,U,r
    def test_full_rank_dense_minimum_energy(self):
        for b in (1,3,7):
            A,L,K,T,U,r=self.solve(b+3,b)
            Y=torch.linalg.solve(A,K);ref=torch.linalg.solve((K.T@Y).T,T.T).T@Y.T
            torch.testing.assert_close(U,ref,atol=1e-10,rtol=1e-9)
            torch.testing.assert_close(U@K,T,atol=1e-10,rtol=1e-9)
            self.assertTrue(r['numerical_projection_verified'])
            self.assertAlmostEqual(r['ideal_Q'],float(((U@A)*U).sum()),places=8)
    def test_duplicate_zero_range(self):
        A,L,_=prepare_metric(torch.eye(3,dtype=torch.float64))
        K=torch.tensor([[1,1,0],[0,0,1],[0,0,0]],dtype=torch.float64)
        T=torch.tensor([[1,3,0]],dtype=torch.float64)
        U,r=solve_realized(A,L,K,T,torch.tensor([.25,.25,.5],dtype=torch.float64),'equality')
        self.assertEqual(r['status'],'RANGE_PROJECTED_WEIGHTED_LEAST_SQUARES')
        torch.testing.assert_close(U@K,torch.tensor([[2,2,0]],dtype=torch.float64))
        self.assertEqual(r['shape']['constraints'],3)
    def test_variable_CD_KL_owner_target(self):
        owners=torch.tensor([0,0,0,1,1]);D=torch.tensor([[2.,0.],[1.,0.]])
        w=owner_weights(owners,2);self.assertEqual(float(w[:3].sum()),.5)
        T=targets('CD',D,None,None,owners);torch.testing.assert_close(T[:,[2,4]],D)
        self.assertEqual(T.shape[1],5)
    def test_zero_tracking_does_not_skip(self):
        D=torch.zeros(2,3);z=torch.ones(2,3);h=torch.zeros(2,3)
        for b in ('RT','MT'):self.assertTrue(bool(targets(b,D,z,h,None).count_nonzero()))
        for b in ('RD','MD'):self.assertFalse(bool(targets(b,D,z,h,None).count_nonzero()))
    def test_zero_rank_is_valid_range(self):
        A,L,_=prepare_metric(torch.eye(3,dtype=torch.float64));K=torch.zeros(3,2,dtype=torch.float64)
        U,r=solve_realized(A,L,K,torch.ones(1,2,dtype=torch.float64),torch.full((2,),.5,dtype=torch.float64),'equality')
        self.assertEqual(r['rank'],0);self.assertTrue(r['numerical_projection_verified']);self.assertEqual(float(U.norm()),0)
    def test_symmetric_metric_no_jitter(self):
        raw=torch.tensor([[3.,.3],[.1,2.]],dtype=torch.float64);A,L,r=prepare_metric(raw)
        torch.testing.assert_close(A,(raw+raw.T)/2);self.assertGreater(r['skew_frobenius'],0)
        with self.assertRaises(torch.linalg.LinAlgError):prepare_metric(torch.diag(torch.tensor([1.,-1.],dtype=torch.float64)))
    def test_elementwise_parity_not_ratio(self):
        r=action_parity(torch.tensor([1.,-1.]),torch.tensor([1.,1.]));self.assertFalse(r['pass_'])
        r=action_parity(torch.zeros(2),torch.zeros(2));self.assertTrue(r['pass_'])
    def test_ill_conditioned_numeric_not_rank_gate(self):
        A,L,_=prepare_metric(torch.eye(2,dtype=torch.float64));rot=torch.tensor([[.6,-.8],[.8,.6]],dtype=torch.float64)
        K=rot@torch.diag(torch.tensor([1.,1e-12],dtype=torch.float64))@rot.T
        _,r=solve_realized(A,L,K,torch.tensor([[1.,3.]],dtype=torch.float64),torch.full((2,),.5,dtype=torch.float64),'equality')
        self.assertEqual(r['rank'],2);self.assertFalse(r['numerical_projection_verified'])
    def test_ridge_unweighted_native_operator(self):
        A,L,K,T,_,_=self.solve(7,3)
        U,r=solve_realized(A,L,K,T,torch.full((3,),1/3,dtype=torch.float64),'ridge')
        torch.testing.assert_close(U,T@torch.linalg.solve(A+K@K.T,K).T)
        self.assertTrue(r['numerical_projection_verified'])
    def test_rollback_W_H_RNG(self):
        class Fake:
            def __init__(self):self.weights={4:torch.zeros(2,3)}
            def guard(self):return 'unchanged'
        a=Fake();H={4:torch.zeros(3,3)};rng=rng_snapshot();t=Transaction(a,H)
        with t:a.weights[4].add_(1);H[4].add_(1);torch.randn(3)
        self.assertTrue(t.rollback_verified and rng_equal(rng));self.assertEqual(float(H[4].sum()),0)
    def test_zero_plan_share_null(self):
        p={4:torch.zeros(2,1),5:torch.zeros(2,1)};x={4:torch.ones(2,1),5:torch.ones(2,1)};a={4:torch.ones(1),5:torch.ones(1)}
        r=shares(p,x,a,[4,5])[0];self.assertIsNone(r['planned_share']);self.assertIsNone(r['direct_share'])
        self.assertEqual(r['status'],'NO_PLANNED_EDIT')
    def test_paired_sign_and_tokens(self):
        before=[];after=[]
        for k in ('R','P','N'):
            r=dict(identity=k,kind=k,new_nll=2.,true_nll=1.,new_strict=False,true_strict=True,new_token_identity='a',true_token_identity='b')
            before.append(r);after.append(r|dict(new_nll=.5,new_strict=True,true_strict=False))
        r=paired(before,after);self.assertEqual(r['R']['preference']['gained'],1);self.assertEqual(r['N']['preference']['lost'],1)
        after[0]['new_token_identity']='changed'
        with self.assertRaises(Exception):paired(before,after)
    def test_source_scope_no_B2_no_disk_tensor(self):
        folder=ROOT/'project/run_scripts/jlz_realized_writer'
        for p in folder.glob('*.py'):
            if p.name.startswith('test_'):continue
            tree=ast.parse(p.read_text())
            calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
            self.assertNotIn('torch.save',calls);self.assertNotIn('np.save',calls)
        tree=ast.parse((folder/'run.py').read_text())
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fit' for n in ast.walk(tree)),1)
        self.assertEqual(BRANCHES,('RT','RD','MT','MD','CD'))

    def test_tiny_same_plan_five_branches_history_restore(self):
        import numpy as np
        from types import SimpleNamespace
        from transformers import LlamaConfig,LlamaForCausalLM
        from .capture import Adapter,capture_native_sites,virtual_terminal,cache_identity
        from .writer import apply_sequential
        from .common import state
        from project.run_scripts.jlz_shared_budget.entry import prepare_entry
        from project.run_scripts.jlz_shared_budget.optimize import fit
        from project.run_scripts.jlz_shared_budget.telemetry import Events
        model=LlamaForCausalLM(LlamaConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=2,vocab_size=31,max_position_embeddings=64,_attn_implementation='eager'))
        a=Adapter(model,dict(eligible_layers=[0,1],anchor_layer=1,nll_layer=2,kl_factor=.0625))
        bench=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        pack=dict(identity='CPU_TINY',n_requests=3,n_rw=2,record_ids=[11,12,13],canonical_rows=[0,3,6],
            context_group_slices=[(0,1),(1,2)],lookup=[1]*9,row_kind=['rewrite','rewrite','kl']*3,
            row_request=[i//3 for i in range(9)],tokens=dict(input_ids=torch.arange(45).reshape(9,5)%29+1,
            attention_mask=torch.ones(9,5,dtype=torch.long)),targets=torch.full((9,5),-100))
        for i in range(9):
            if pack['row_kind'][i]=='rewrite':pack['targets'][i,3:]=torch.tensor([2,3])
        H={l:torch.zeros(16,16) for l in a.sites};before=state(a,H)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);stats={}
            for l in a.sites:
                path=root/f'stats{l}.npz';np.savez(path,**{'mom2.mom2':np.eye(16,dtype=np.float32),'mom2.count':np.array(1)})
                stats[str(l)]=str(path)
            entry=prepare_entry(a,bench,pack,H,stats,1);initial=capture_native_sites(a,entry,a.sites)
            a.capture_virtual=True;plan,receipt=fit(a,entry,Events(root/'fit-events.jsonl','CPU',1),root/'fit')
            a.capture_virtual=False;virtual=virtual_terminal(a,entry,plan);cached=cache_identity(entry)
            for branch in BRANCHES:
                with Transaction(a,H) as tx:
                    wr=apply_sequential(a,entry,plan,virtual,initial,H,branch,root/branch)
                    self.assertEqual(wr['history_appends'],2)
                    final=capture_native_sites(a,entry,a.sites)
                    for l in a.sites:
                        k=final['mean'][l];native=k.T.contiguous().T
                        torch.testing.assert_close(H[l],native@native.T,atol=0,rtol=0)
                self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,H),before);self.assertEqual(cache_identity(entry),cached)
            self.assertLessEqual(receipt['request_evaluations'],75)

if __name__=='__main__':unittest.main()
