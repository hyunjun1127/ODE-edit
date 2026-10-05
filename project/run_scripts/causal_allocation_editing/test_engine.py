"""Narrow production replay/geometry regressions; not actual-model evidence."""
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from .engine import CausalObjective
from .geometry import compact_cost, cost_adjoint, solve_vjp
from project.run_scripts.jlz_native_writer_aware.physical import linear


class TinyAdapter:
    def __init__(self):
        self.device=torch.device('cpu');self.sites=(0,2);self.first=0;self.nll_layer=4
        self.dims={0:(3,4),2:(3,5)};self.profile={'lambda_KL':.0625}
        self.bridge=torch.randn(5,3)*.2;self.output=torch.randn(7,3)*.3
    def head(self,x):return F.linear(x,self.output)
    def stage_full(self,l,nxt,key,residual,R,P,W,kw,route='direct'):
        x=torch.tanh(residual+linear(key,R,P,W,route))
        return F.linear(x,self.bridge), x*.7, x.new_empty(0)
    def suffix(self,l,key,residual,R,P,W,kw,route='direct'):
        x=torch.tanh(residual+linear(key,R,P,W,route))
        return x,x


def fixture(B=3):
    torch.manual_seed(103);a=TinyAdapter();groups=[];teachers={}
    for owner in range(B):
        rows=[];T=3+owner
        for c in range(3):
            target=torch.full((T,),-100,dtype=torch.long)
            if c<2:target[-1]=owner%7
            rows.append(dict(global_row=owner*3+c,request=owner,kind='rewrite' if c<2 else 'kl',
                             lookup=1,target=target))
        groups.append(dict(rows=rows,tokens=dict(input_ids=torch.zeros(3,T,dtype=torch.long),
                                               attention_mask=torch.ones(3,T,dtype=torch.long)),
                           cache=dict(key=torch.randn(3,T,4)*.2,residual=torch.randn(3,T,3)*.2,kwargs={})))
        teachers[owner]=torch.randn(7).log_softmax(-1)
    factors={};weights={}
    for l,(_,n) in a.dims.items():
        z=torch.randn(n,n,dtype=torch.float64);A=z@z.T+torch.eye(n,dtype=torch.float64)*2
        A[0,1]+=.03 # deliberately raw nonsymmetric stored metric
        lu,piv=torch.linalg.lu_factor(A)
        factors[l]=dict(A=A,LU=lu,pivots=piv,L=None,device='cpu')
        weights[l]=torch.randn(3,n)*.1
    pack=dict(n_requests=B,n_rw=2,context_group_slices=[(0,1),(1,2)])
    entry=dict(groups=groups,teachers=teachers,factors=factors,entry_weights=weights,
               anchors={l:torch.ones(B)*(1+l*.1) for l in a.sites},pack=pack,first_geometry={})
    u={l:torch.randn(3,B)*.08 for l in a.sites}
    return a,entry,u


class EngineTests(unittest.TestCase):
    def test_nonfinite_actual_or_reference_never_passes_comparison(self):
        from .qualification import comparison
        for value in (float('nan'),float('inf'),float('-inf')):
            for left,right in ((torch.tensor([value]),torch.tensor([0.])),
                               (torch.tensor([0.]),torch.tensor([value]))):
                with self.assertRaisesRegex(RuntimeError,'QUALIFICATION_NONFINITE_COMPARISON'):
                    comparison(left,right,2e-5,2e-4)

    def test_raw_compact_Q_VJP(self):
        torch.manual_seed(4)
        R=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
        P=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
        A=torch.randn(4,4,dtype=torch.float64)
        q=compact_cost(R,P,A);U=R@P.T;dense=((U@A)*U).sum()
        self.assertTrue(torch.allclose(q,dense,atol=1e-12,rtol=1e-12))
        dr,dp=torch.autograd.grad(q,(R,P))
        self.assertTrue(torch.allclose(dr,R@((P.T@A@P)+(P.T@A@P).T),atol=1e-12))
        self.assertTrue(torch.allclose(dp,(A+A.T)@P@(R.T@R),atol=1e-12))

    def test_cached_nonsymmetric_solve_VJP(self):
        a,e,u=fixture(2);l=0;K=torch.randn(4,2,dtype=torch.float64)
        P=torch.linalg.solve(e['factors'][l]['A']+K@K.T,K);gp=torch.randn_like(P)
        ref,_=solve_vjp(K,P,gp,e['factors'][l],False)
        fast,_=solve_vjp(K,P,gp,e['factors'][l],True)
        self.assertTrue(torch.allclose(ref,fast,atol=1e-10,rtol=1e-9))

    def test_streamed_full_gradient_matches_same_group_dense(self):
        for B in (1,3):
            a,e,u=fixture(B);engine=CausalObjective(a,e,route='dense')
            actual=engine.evaluate(u,gradient=True,lambda_Q=.3)
            reference=engine.dense_evaluate(u,.3,native_full=True)
            self.assertAlmostEqual(actual['smooth_sum'],reference['smooth_sum'],places=5)
            for l in a.sites:
                err=(actual['gradient'][l]-reference['gradient'][l]).abs()
                self.assertTrue(bool((err<=1e-6+2e-4*reference['gradient'][l].abs()).all()),str((l,err.max())))
            self.assertEqual(engine.calls['logical_candidates'],1)
            self.assertEqual(engine.calls['gradient_channels'],1)

    def test_direct_route_and_channel_cost_once(self):
        a,e,u=fixture(3);engine=CausalObjective(a,e)
        result=engine.evaluate(u,gradient=True,lambda_Q=.4,channels=True)
        reference=engine.dense_evaluate(u,.4)
        for l in a.sites:
            err=(result['gradient'][l]-reference['gradient'][l]).abs()
            self.assertTrue(bool((err<=1e-6+2e-4*reference['gradient'][l].abs()).all()))
            self.assertTrue(torch.equal(result['gradient'][l],result['task_gradient'][l]+.4*result['Q_gradient'][l]))

    def test_downstream_Q_negative_control(self):
        a,e,u=fixture(3);engine=CausalObjective(a,e,route='dense')
        payload=engine.evaluate(u)['payload']
        full=engine.backward(payload,channels=True)['Q_gradient'][0]
        stopped=engine.backward(payload,channels=True,stop_solve=True)['Q_gradient'][0]
        self.assertGreater(float((full-stopped).norm()),1e-6)

    def test_zero_blocks_are_eligible_and_no_entry_mutation(self):
        a,e,u=fixture(2);engine=CausalObjective(a,e)
        old={l:w.clone() for l,w in e['entry_weights'].items()}
        zeros=engine.zeros();result=engine.evaluate(zeros,gradient=True,lambda_Q=.4)
        self.assertEqual(result['Q'],0.)
        self.assertGreater(sum(float(g.norm()) for g in result['gradient'].values()),0.)
        self.assertTrue(all(torch.equal(old[l],e['entry_weights'][l]) for l in a.sites))
        calls=engine.calls['logical_candidates']
        engine.backward(result['payload'],.4)
        self.assertEqual(calls,engine.calls['logical_candidates'])

    def test_native_reference_R_override_keeps_returned_delta(self):
        a,e,u=fixture(2);engine=CausalObjective(a,e)
        r={l:torch.zeros_like(u[l]) for l in a.sites}
        r[2]=torch.tensor([[.12345679,.22223334],[.37654123,.42132324],[.09283746,.46381928]])
        coords={l:r[l]/e['anchors'][l][None,:] for l in a.sites}
        result=engine.evaluate(coords,gradient=True,channels=True,reference_R=r)
        self.assertTrue(all(torch.equal(result['payload']['R'][l],r[l]) for l in a.sites))
        self.assertTrue(result['payload']['native_reference_exact_R'])

    def test_telemetry_no_forward_and_effective_update_definition(self):
        a,e,u=fixture(2);engine=CausalObjective(a,e)
        e['history_entry']={l:torch.zeros_like(f['A'],dtype=torch.float32) for l,f in e['factors'].items()}
        payload=engine.evaluate(u)['payload'];before=dict(engine.calls)
        result=engine.terminal_telemetry(payload)
        self.assertEqual(before,engine.calls)
        self.assertTrue(result['no_extra_model_forward'])
        for l in a.sites:
            row=result['layers'][str(l)]
            self.assertEqual(row['effective_Q_H'],0.)
            self.assertAlmostEqual(row['ideal_Q_H'],0.)
            self.assertIsNotNone(row['mean']['directional_ratio'])

    def test_physical_probe_finally_exact_restore_and_input_name_separation(self):
        """CPU mock only: restore must survive token loading and native errors."""
        from types import SimpleNamespace
        from .qualification import _physical_probe
        projection=torch.nn.Linear(3,2,bias=False)
        projection.weight.requires_grad_(False)
        original=projection.weight.detach().clone()
        rows=[dict(global_row=i,request=0,kind='rewrite' if i<2 else 'kl',lookup=1,
                   reduction_index=i if i<2 else 0,target=torch.tensor([-100,0])) for i in range(3)]
        tokens=dict(input_ids=torch.zeros(3,2,dtype=torch.long),attention_mask=torch.ones(3,2,dtype=torch.long))
        class PhysicalMock:
            device=torch.device('cpu');sites=(0,)
            weights={0:projection.weight};blocks=[SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))]
            def guard(self):return 'CPU-mock-nonselected-state'
            def head(self,x):return x
            def full(self,inputs):
                x=projection(torch.ones(3,2,3));return x,x
        entry=dict(groups=[dict(rows=rows,tokens=tokens)],
                   pack=dict(n_requests=1,n_rw=2,context_group_slices=[(0,1),(1,2)],tokens=tokens))
        payload=dict(weights={0:torch.ones_like(original)},K={0:torch.ones(3,1,dtype=torch.float64)},
                     nll=torch.zeros(1,2,dtype=torch.float64),kl=torch.zeros(1,dtype=torch.float64),
                     nll_hidden={0:torch.full((3,2,2),3.)},final_hidden=[torch.full((3,2,2),3.)])
        values=[('rewrite',0,0,0.),('rewrite',0,1,0.),('kl',0,0,0.)]
        def captured_terms(a,e,g,nh,fh,capture=None):
            from .engine import selected_logits
            if capture is not None:capture['selected_logits']=selected_logits(a,g['rows'],nh,fh)[0]
            return torch.tensor(0.),values
        with patch('project.run_scripts.causal_allocation_editing.qualification.native_terms',
                   side_effect=captured_terms):
            result=_physical_probe(PhysicalMock(),entry,payload)
        self.assertTrue(result['RAM_restore_exact'])
        self.assertTrue(torch.equal(projection.weight,original))
        with patch('project.run_scripts.causal_allocation_editing.qualification.native_terms',
                   side_effect=RuntimeError('confirmed mock native technical error')):
            with self.assertRaisesRegex(RuntimeError,'confirmed mock native'):
                _physical_probe(PhysicalMock(),entry,payload)
        self.assertTrue(torch.equal(projection.weight,original))
        self.assertEqual(len(projection._forward_pre_hooks),0)


if __name__=='__main__':unittest.main()
