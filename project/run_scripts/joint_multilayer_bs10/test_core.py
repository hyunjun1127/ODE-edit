"""Bounded CPU regressions; tiny random models are not actual pinned Llama evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from .common import *
from .solver import basis,projected,al_value,next_dual,JointHook,Objective,energy,bounds_for,warm_dual,working_set,solve
from .reduce import paired,transitions
from .observations import Scorer

torch.set_num_threads(2)

class Core(unittest.TestCase):
    def test_raw_P_nonsymmetric(self):
        torch.manual_seed(2);p=torch.randn(7,7);m=torch.randn(7,7);m=m@m.T;k=torch.randn(7,3)
        q,r=basis(p,m,k);a=torch.linalg.solve(p.double()@(k.double()@k.double().T+m.double())+10*torch.eye(7),p.double()@k.double())
        self.assertEqual(q.shape,(7,3));self.assertLess(float((a-q.double()@(q.double().T@a)).norm()),1e-5)
        self.assertLess(r['raw_solve_relative_residual'],1e-10)
    def test_rank_truncation(self):
        k=torch.ones(5,3);q,r=basis(torch.eye(5),torch.eye(5),k);self.assertEqual(r['rank'],1);self.assertEqual(r['discarded'],2)
    def test_zero_rank(self):
        q,r=basis(torch.eye(5),torch.eye(5),torch.zeros(5,3));self.assertEqual(q.shape[1],0)
    def test_projection_is_joint(self):
        v=[torch.ones(3,2)*.02 for _ in range(5)];g=[torch.zeros_like(x) for x in v]
        ans=projected(v,g,.001);self.assertLessEqual(2*energy(ans),1.000001e-4)
        self.assertGreater(sum(float(x.norm()) for x in ans),.01)
    def test_actual_displacement_armijo(self):
        v=[torch.tensor([[.01]])];g=[torch.tensor([[-1.]])];ans=projected(v,g,.001)
        self.assertAlmostEqual(float(ans[0]),.01,places=8)
    def test_AL_inactive(self):self.assertEqual(al_value({'a':-1},{'a':0},2),0)
    def test_AL_derivative(self):
        c=.3;l=.7;b=4.;e=1e-6
        fd=(al_value({'x':c+e},{'x':l},b)-al_value({'x':c-e},{'x':l},b))/(2*e)
        self.assertAlmostEqual(fd,max(0,l+b*c),places=8)
    def test_dual_projection_and_preservation(self):
        a=next_dual({'base':2,'history:old':4},{'base':-.5},2);self.assertEqual(a,{'base':1.,'history:old':4})
    def test_step_cum_only_bounds_differ(self):
        a=bounds_for('JOINT_CUM',.2,{'old:1':.3,'new:2':.4},.25,{'old:1':.35,'new:2':.5})
        b=bounds_for('JOINT_STEP',.2,{'old:1':.3,'new:2':.4},.25,{'old:1':.35,'new:2':.5})
        self.assertAlmostEqual(a['base'],.25);self.assertAlmostEqual(b['base'],.30)
        self.assertAlmostEqual(a['history']['new:2'],.5);self.assertAlmostEqual(b['history']['new:2'],.6)
    def test_new_edit_duals_reset_old_protection_retained(self):
        self.assertEqual(warm_dual({'base':2.,'history:old:1':3.,'edit:7':4.,'preference:7':5.}),{'base':2.,'history:old:1':3.})
    def test_least_slack_working_set_and_omitted_violator(self):
        history={**{f'old:{i}':{} for i in range(16)},**{f'new:{i}':{} for i in range(40)}}
        residual={f'history:{h}':float(h.split(':')[-1]) for h in history}
        w=working_set(history,residual);self.assertEqual(len(w),48);self.assertIn('new:39',w);self.assertNotIn('new:0',w)
        self.assertTrue(all(f'old:{i}' in w for i in range(16)))
    def test_atomic_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';save(p,{'x':1})
            with self.assertRaises(FileExistsError):save(p,{'x':2})
            self.assertEqual(read(p),{'x':1})
    def test_finite_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save(Path(d)/'a',{'x':float('nan')})
    def test_five_weight_IO(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'w.pt';weights={l:torch.randn(3,4) for l in LAYERS};tensor_save(p,weights)
            r=torch.load(p,weights_only=True)
            self.assertEqual(set(r),set(LAYERS));self.assertTrue(all(torch.equal(r[l],weights[l]) for l in LAYERS))
    @staticmethod
    def row(label,value):return dict(pair_id='a',row_id=label,case_id=1,role='continuation',kind='R',prompt_index=0,label=label,nll=value,token_nll=[value],target_count=1,token_correct=0,strict=False)
    def test_tie_fails(self):self.assertFalse(paired([self.row('true',1),self.row('new',1)])[0]['preferred'])
    def test_N_true_direction(self):
        rows=[self.row('true',1),self.row('new',2)]
        for r in rows:r['kind']='N'
        self.assertTrue(paired(rows)[0]['preferred'])
    def test_missing_pair(self):
        with self.assertRaises(RuntimeError):paired([self.row('true',1)])
    def test_duplicate_pair(self):
        with self.assertRaises(RuntimeError):paired([self.row('new',1),self.row('new',2)])
    def test_paired_transitions(self):
        a=[self.row('new',1),self.row('true',2)];b=[self.row('new',3),self.row('true',2)]
        self.assertEqual(transitions(a,b)['lost'],['a'])
    @staticmethod
    def tiny():
        sys.path.insert(0,str(DEPS));from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(42);cfg=LlamaConfig(vocab_size=32,hidden_size=8,intermediate_size=12,num_hidden_layers=9,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
        cfg._attn_implementation='eager';m=LlamaForCausalLM(cfg).float().eval();m.requires_grad_(False)
        rt=SimpleNamespace(model=m);q=[torch.linalg.qr(torch.randn(12,2)).Q for _ in LAYERS]
        v=[(torch.randn(8,2)*.002).requires_grad_(True) for _ in LAYERS];return rt,q,v
    def test_tiny_joint_all_five_gradients_and_materialization(self):
        rt,q,v=self.tiny();ids=torch.tensor([[1,2,3,4]])
        with JointHook(rt,q,v,[1.]*5):
            hooked=rt.model(ids,use_cache=False).logits;loss=hooked[0,-1].square().sum();loss.backward()
        self.assertTrue(all(x.grad is not None and float(x.grad.norm())>0 for x in v))
        with torch.no_grad():
            for i,l in enumerate(LAYERS):rt.model.model.layers[l].mlp.down_proj.weight.add_(v[i]@q[i].T)
            physical=rt.model(ids,use_cache=False).logits
        torch.testing.assert_close(hooked,physical,atol=1e-6,rtol=1e-5)
    def test_live_downstream_input_not_entry_constant(self):
        rt,q,v=self.tiny();ids=torch.tensor([[1,2,3,4]]);seen=[]
        h=rt.model.model.layers[8].mlp.down_proj.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
        with torch.no_grad():rt.model(ids,use_cache=False)
        with JointHook(rt,q,v,[1.]*5):rt.model(ids,use_cache=False)
        h.remove();self.assertGreater(float((seen[1]-seen[0]).abs().max()),0)
        self.assertEqual(seen[0].shape[1],4)
    def test_hook_exception_cleans_up(self):
        rt,q,v=self.tiny()
        try:
            with JointHook(rt,q,v,[1.]*5):raise ValueError('fixture')
        except ValueError:pass
        self.assertTrue(all(not rt.model.model.layers[l].mlp.down_proj._forward_hooks for l in LAYERS))
    def test_full_objective_streamed_gradient(self):
        # Compare one full graph with exact component-weighted streaming on the same tiny shared model.
        rt,q,v=self.tiny();sc=Scorer(rt)
        def row(i):return dict(row_id=str(i),input_ids=[1,2,3],positions=[1,2],target_ids=[3,4])
        rows=[row(i) for i in range(6)];new=dict(row(7),kind='R',label='new');true=dict(row(8),kind='R',label='true',target_ids=[5,6])
        base=row(9)
        with torch.no_grad():teacher=sc.forward(base)[0].detach()+.001
        obj=Objective(sc,{1:[new,true]},{1:rows},[base],{base['row_id']:teacher},{'old:1':row(10)},dict(base=.05,history={'old:1':1.}))
        with JointHook(rt,q,v,[1.]*5):
            c,_=obj.evaluate(['old:1'],'cpu');g,_=obj.gradient(['old:1'],c,{},2,v)
            for x in v:x.grad=None
            loss=sum(x.square().sum() for x in v)/2
            for k,(bound,terms) in obj.components(['old:1']).items():
                val=sum(w*obj.scalar(r,kind,'cpu_direct') for r,w,kind in terms)-bound
                loss=loss+torch.relu(2*val).square()/4
            loss.backward()
        for x,y in zip(v,g):torch.testing.assert_close(x.grad,y,atol=1e-5,rtol=1e-4)

    def test_reject_budget_and_rollback_actual_solver_control(self):
        from unittest.mock import patch
        rt,_,_=self.tiny();rt.weights={l:rt.model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
        rt.w0={l:w.detach().clone() for l,w in rt.weights.items()};rt.M=torch.stack([torch.eye(12) for _ in LAYERS]);rt.P=rt.M.clone()
        rt.snapshot=lambda:{l:w.detach().clone() for l,w in rt.weights.items()}
        rt.hashes=lambda:{str(l):tensor_sha(w) for l,w in rt.weights.items()}
        def restore(w):
            with torch.no_grad():
                for l in LAYERS:rt.weights[l].copy_(w[l])
        rt.restore=restore;rt.keys=lambda req,l:torch.ones(12,1);rt.rng_get=torch.get_rng_state;rt.rng_set=torch.set_rng_state
        rt.check_fixed=lambda:None;rt.config={'settings':{'materialization_parity_nll_abs':2.5e-4,'materialization_parity_margin_abs':5e-4}}
        rt.append=lambda req:self.fail('reject cannot append')
        counters={'full':0,'gradient':0}
        class Stub:
            def __init__(self,*a):pass
            def evaluate(self,history,category):
                if category=='full_guard_physical':counters['full']+=1
                return dict(base=-.05,**{'edit:1':2.,'history:old:1':-.1}),[dict(row_id='fixture',kind='nll',value=3.)]
            def gradient(self,h,c,d,b,v):
                counters['gradient']+=1;return [torch.zeros_like(x) for x in v],[]
        before=rt.hashes();mh=tensor_sha(rt.M);anchors={'old:1':1.}
        with tempfile.TemporaryDirectory() as d,patch('project.run_scripts.joint_multilayer_bs10.solver.Objective',Stub):
            r,_=solve(rt,None,[{}],{}, {},[],{}, {'old:1':{}},anchors,.1,{},'JOINT_CUM',d,1)
        self.assertFalse(r['accepted']);self.assertEqual(counters,{'full':5,'gradient':40});self.assertEqual(r['trials'],40)
        self.assertEqual(rt.hashes(),before);self.assertEqual(tensor_sha(rt.M),mh);self.assertEqual(anchors,{'old:1':1.})
        self.assertFalse(any(k.startswith('edit:') for k in r['dual']))

    def test_bs1_keeps_five_layers_and_four_saves(self):
        self.assertEqual((BATCH_SIZE,STEPS,LAYERS,SAVE_STEPS),(1,100,(4,5,6,7,8),(25,50,75,100)))
        self.assertEqual(9*len(SAVE_STEPS)*5*4096*14336*4,42278584320)
        self.assertEqual(9*STEPS*BATCH_SIZE,900)

    def test_bs1_basis_rank_bound(self):
        q,r=basis(torch.eye(7),torch.eye(7),torch.randn(7,1));self.assertEqual(q.shape,(7,1))

    def test_bs10_cannot_enter_bs1_solver(self):
        with self.assertRaisesRegex(RuntimeError,'JOINT_USER_BS1'):
            solve(None,None,[{}]*10,None,None,None,None,None,None,None,None,None,None,None)

    @staticmethod
    def config_fixture():
        return dict(user_override=dict(nonce=OVERRIDE_NONCE,edit_layers=list(LAYERS)),
            contract=dict(batch_size=1,batches_per_trajectory=100,edits_per_trajectory=100,
                weight_snapshots=dict(offered_batches=list(SAVE_STEPS),snapshots=36,tensors_per_snapshot=5,total_tensor_bytes=42278584320),
                budgets=dict(scientific_batch_attempts=900)),execution_ids=list(range(100)),
            execution_batches=[dict(step=i+1,case_ids=[i]) for i in range(100)],
            trajectories=[dict(name=f'{cp}-{arm}') for cp in ('B010','B050','B090') for arm in ('JOINT_STEP','NATIVE','JOINT_CUM')])

    def test_execution_override_binding_and_reject_old_save_policy(self):
        c=self.config_fixture();validate_execution(c)
        c['contract']['weight_snapshots']['offered_batches']=[25,50]
        with self.assertRaisesRegex(RuntimeError,'USER_OVERRIDE_SAVE25'):validate_execution(c)

    def test_collector_never_calls_missing_branches_complete(self):
        from .reduce import collect
        with tempfile.TemporaryDirectory() as d:
            cfg=save(Path(d)/'cfg.json',self.config_fixture())
            lock=save(Path(d)/'lock.json',dict(configuration=cfg,attempt=d,source='CPU_FIXTURE'))
            collect(lock['path']);r=read(Path(d)/'collector/terminal.json')
            self.assertEqual(r['status'],'INCOMPLETE_OR_TECHNICAL_FAILED');self.assertEqual(len(r['trajectories']),9)
            self.assertTrue(all(x['terminal']=='MISSING_TERMINAL' for x in r['trajectories']))

if __name__=='__main__':unittest.main()
