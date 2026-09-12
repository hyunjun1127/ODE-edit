"""Small CPU dense constrained oracles for A-OS; no model loading."""
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.multilayer_joint_compensation.functional import PanelLinearization, OutputBatch, FunctionalPanel
from project.run_scripts.multilayer_joint_compensation.track_a.protocol import AProblem, run_os
from project.run_scripts.multilayer_joint_compensation.linear_solve import dot


def flatten(values):
    return torch.cat([v.reshape(-1) for v in values])


def tree(values):
    return tuple(v.reshape(1, 2) for v in values.split(2))


class QuadraticPanel:
    """Exact linear-output quadratic, with known full cross-layer curvature."""
    def __init__(self, role, anchor, gradient, matrix, nll_gradient):
        self.role, self.tau = role, .1
        self.anchor = flatten(anchor).clone()
        self.gradient = gradient.float().clone()
        self.matrix = matrix.float().clone()
        self.nll_gradient = nll_gradient.float().clone()
        self.counts = dict(linearizations=0, observations=0, ggn_matvecs=0)
        self.seen_ggn_weights = []

    def linearize(self, weights, *, need_nll_gradient=False):
        self.counts['linearizations'] += 1
        x = flatten(weights)-self.anchor
        gradient = self.gradient + self.matrix@x
        ng = self.nll_gradient if need_nll_gradient else torch.zeros_like(gradient)
        observation = self.observe(weights)
        return PanelLinearization(observation['value'], observation['mean_nll'],
                                   tree(gradient), tree(ng), [], {})

    def ggn(self, weights, direction):
        self.counts['ggn_matvecs'] += 1
        self.seen_ggn_weights.append(tuple(w.clone() for w in weights))
        return tree(self.matrix@flatten(direction))

    def observe(self, weights):
        self.counts['observations'] += 1
        x = flatten(weights)-self.anchor
        return dict(value=float(1+self.gradient@x+.5*x@(self.matrix@x)),
                    mean_nll=float(2+self.nll_gradient@x), context_rows=[])


def fixture(seed=1, native_scale=1., d0_zero=False, zero_a=False):
    generator = torch.Generator().manual_seed(seed)
    we = tree(torch.randn(4, generator=generator)*.2)
    nominal = tree(torch.zeros(4) if d0_zero else torch.randn(4, generator=generator)*.1)
    wa = tuple(w+d for w,d in zip(we, nominal))
    d0 = tuple(w-e for w,e in zip(wa,we))
    matrices = []
    for _ in range(3):
        factor = torch.randn(4,4,generator=generator)*.15
        matrices.append(factor.T@factor+.2*torch.eye(4))
    gradients = [torch.randn(4,generator=generator)*.05 for _ in range(2)] + [torch.zeros(4)]
    a = torch.zeros(4) if zero_a else torch.randn(4,generator=generator)
    panels = [QuadraticPanel(role,wa,g,m,a) for role,g,m in zip(('base','past','current'),gradients,matrices)]
    s = torch.diag(torch.tensor([2.,9.,1.,14.])*native_scale)
    h = s.clone()
    h[:2,:2] /= s[:2,:2].trace()/2
    h[2:,2:] /= s[2:,2:].trace()/2
    def op(matrix):
        return lambda values:tree(matrix@flatten(values))
    guard=lambda values:dict(selected_fp32_shape_valid=all(w.dtype==torch.float32 and w.shape==(1,2) for w in values),
                            nonselected_unchanged=True, history_append_count=0, compute_z_count=0)
    problem=AProblem(we,d0,(4,8),*panels,op(h),op(torch.linalg.inv(h)),op(s),lambda x:x,
                     (.07,.035),(.2,.9),1.3,guard,dict(seed=seed),wa=wa)
    balance=torch.diag(torch.tensor([.1*.2/1.3**2]*2+[.1*.9/1.3**2]*2))@s
    k=.01*h+2*matrices[0]/.07+matrices[1]/.035+matrices[2]+balance
    u=2*gradients[0]/.07+gradients[1]/.035+balance@flatten(d0)
    inverse_u=torch.linalg.solve(k.double(),u.double())
    correction=-inverse_u
    if not zero_a:
        ia=torch.linalg.solve(k.double(),a.double())
        correction=correction-ia*(a.double()@correction)/(a.double()@ia)
    return problem, correction.float(), k, u, a, s, balance


class AProtocolTests(unittest.TestCase):
    def test_77_dense_joint_equality_oracles(self):
        # 7 seeds × 11 raw native physical scales; normalized H stays fixed,
        # while the *raw S* balance must change. No outcome-based case filter.
        for seed in range(7):
            for exponent in range(-5,6):
                scale=2.**exponent
                with self.subTest(seed=seed, native_scale=scale):
                    p,oracle,_,_,a,_,_=fixture(seed,scale)
                    endpoint,row=run_os(p)
                    actual=flatten(tuple(w-b for w,b in zip(endpoint,p.wa)))
                    torch.testing.assert_close(actual,oracle,rtol=3e-4,atol=3e-6)
                    self.assertLess(abs(float(a@actual)),3e-6)
                    self.assertEqual(row['solver']['h'],1.)
                    self.assertEqual(row['solver']['independent_rhs_count'],2)
                    self.assertEqual(row['joint_equality_count'],1)
                    self.assertEqual(row['layer_equality_count'],0)

    def test_cumulative_balance_and_raw_metric_are_not_optional(self):
        p,oracle,k,u,a,s,balance=fixture(4,16.)
        endpoint,row=run_os(p)
        actual=flatten(tuple(w-b for w,b in zip(endpoint,p.wa)))
        def constrained(kk,uu):
            out=-torch.linalg.solve(kk.double(),uu.double())
            ia=torch.linalg.solve(kk.double(),a.double())
            return out-ia*(a.double()@out)/(a.double()@ia)
        missing_anchor=constrained(k,u-balance@flatten(p.d0))
        self.assertGreater(float((actual.double()-missing_anchor).norm()),1e-3)
        # Wrong H normalization in the balance changes this oracle clearly.
        normalized_s=s.clone()
        normalized_s[:2,:2]/=s[:2,:2].trace()/2
        normalized_s[2:,2:]/=s[2:,2:].trace()/2
        wrong_balance=torch.diag(torch.tensor([.1*.2/1.3**2]*2+[.1*.9/1.3**2]*2))@normalized_s
        wrong=constrained(k-balance+wrong_balance,u-balance@flatten(p.d0)+wrong_balance@flatten(p.d0))
        self.assertGreater(float((actual.double()-wrong).norm()),1e-3)
        self.assertEqual(row['balance_metric'],'RAW_PROJECTED_S_NOT_NORMALIZED_H')
        self.assertGreater(row['balance_objective_before'],0)

    def test_cross_block_joint_condition_allows_signed_layer_exchange(self):
        p,oracle,k,u,a,_,_=fixture(9)
        endpoint,row=run_os(p)
        without_cross=k.clone();without_cross[:2,2:]=0;without_cross[2:,:2]=0
        bad=-torch.linalg.solve(without_cross.double(),u.double())
        ia=torch.linalg.solve(without_cross.double(),a.double())
        bad-=ia*(a.double()@bad)/(a.double()@ia)
        self.assertGreater(float((oracle.double()-bad).norm()),1e-4)
        changes=[x['signed_current_linear_change'] for x in row['per_layer']]
        self.assertGreater(abs(changes[0]),1e-4)
        self.assertAlmostEqual(sum(changes),0.,places=6)
        self.assertEqual(row['functional_cross_block_dropped'],0)

    def test_stage_callbacks_precede_pcg_completion_and_pin_anchor(self):
        p,_,_,_,_,_,_=fixture(8)
        original_wa=tuple(w.clone() for w in p.wa)
        events=[];solutions=[]
        def stage(name,receipt):
            events.append((name,receipt))
            if name=='LINEARIZATION_COMPLETE':
                # Caller-owned mutable input cannot rewrite pinned derivatives.
                for weight in p.wa:weight.add_(3.)
        endpoint,row=run_os(p,on_stage=stage,on_solution=lambda *values:solutions.append(values))
        self.assertEqual([name for name,_ in events],['LINEARIZATION_COMPLETE','FIRST_FULL_OPERATOR_MATVEC_COMPLETE'])
        self.assertEqual(events[1][1]['operator_counts']['full_operator_matvecs'],1)
        self.assertGreater(events[1][1]['quadratic_form'],0.)
        self.assertFalse(events[1][1]['pcg_completion_required'])
        self.assertEqual(len(solutions),1)
        for panel in (p.base,p.past,p.current):
            for observed in panel.seen_ggn_weights:
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(observed,original_wa)))
        self.assertEqual(row['extra_h_application'],0)
        self.assertEqual(row['history_append'],0)
        self.assertEqual(row['compute_z'],0)

    def test_zero_current_equality_retains_full_curvature_and_noop_balance_origin(self):
        p,oracle,_,_,_,_,_=fixture(3,d0_zero=True,zero_a=True)
        endpoint,row=run_os(p)
        actual=flatten(tuple(w-b for w,b in zip(endpoint,p.wa)))
        torch.testing.assert_close(actual,oracle,rtol=3e-4,atol=3e-6)
        self.assertTrue(row['solver']['zero_current_gradient'])
        self.assertEqual(row['solver']['independent_rhs_count'],1)
        self.assertEqual(row['balance_objective_before'],0)
        self.assertGreater(row['counts']['current']['ggn_matvecs'],0)

    def test_invalid_state_and_exact_saved_anchor_binding(self):
        p,*_=fixture()
        p.d0=tuple(d+.1 for d in p.d0)
        with self.assertRaisesRegex(ValueError,'WA_D0_IDENTITY'):run_os(p)
        p,*_=fixture();p.support=(8,)
        with self.assertRaisesRegex(ValueError,'SUPPORT'):run_os(p)
        p,*_=fixture();p.validate_state=lambda w:dict(selected_fp32_shape_valid=True,nonselected_unchanged=False)
        with self.assertRaisesRegex(RuntimeError,'GUARD'):run_os(p)

    def test_finite_pcg_maxiter_is_honestly_retained(self):
        generator=torch.Generator().manual_seed(77)
        n=64
        split=lambda value:tuple(v.reshape(1,n//2) for v in value.split(n//2))
        we=split(torch.zeros(n));d0=split(torch.ones(n)*.01);wa=tuple(w+d for w,d in zip(we,d0))

        class LargePanel:
            tau=.1
            def __init__(self,role):
                self.role=role;self.counts=dict(linearizations=0,observations=0,ggn_matvecs=0)
                self.gradient=torch.randn(n,generator=generator)*.01 if role!='current' else torch.zeros(n)
                self.a=torch.randn(n,generator=generator)
                self.diagonal=torch.logspace(-2,4,n) if role=='current' else torch.ones(n)*.001
            def linearize(self,weights,*,need_nll_gradient=False):
                self.counts['linearizations']+=1
                return PanelLinearization(1.,2.,split(self.gradient),
                                           split(self.a if need_nll_gradient else torch.zeros(n)),[],{})
            def ggn(self,weights,direction):
                self.counts['ggn_matvecs']+=1
                return split(self.diagonal*flatten(direction))
            def observe(self,weights):
                self.counts['observations']+=1
                return dict(value=1.,mean_nll=2.,context_rows=[])

        panels=[LargePanel(role) for role in ('base','past','current')]
        identity=lambda x:x
        guard=lambda w:dict(selected_fp32_shape_valid=True,nonselected_unchanged=True)
        p=AProblem(we,d0,(4,8),*panels,identity,identity,identity,identity,(1.,1.),(.2,.8),1.,guard,{},wa=wa)
        endpoint,row=run_os(p)
        self.assertEqual(row['status'],'APPROXIMATE_PCG_NONCONVERGENCE')
        self.assertTrue(all(torch.isfinite(w).all() for w in endpoint))
        self.assertTrue(any(r['iterations']==20 and r['relative_residual']>1e-4 for r in row['solver']['pcg'].values()))
        self.assertFalse(row['endpoint_selected_by_performance'])

    def test_actual_shared_functional_jvp_vjp_integration(self):
        we=(torch.tensor([[.1,.2],[-.1,.3]]),torch.tensor([[.4,-.1],[.2,.1]]))
        wa=tuple(w+.01 for w in we);d0=tuple(w-e for w,e in zip(wa,we))
        x=torch.tensor([[.3,.7],[.2,-.1]])
        logits=lambda ws:torch.tanh(x@ws[0])@ws[1]
        def make(role):
            ref=wa if role=='current' else we
            lp=logits(ref).detach().log_softmax(-1)
            batch=OutputBatch(logits,torch.tensor([0,1]),torch.tensor([0,1]),torch.ones(2),
                              torch.full((2,),.5),lp,-lp.diag(),identity=role,input_tokens=4)
            return FunctionalPanel([batch],role)
        base,past,current=[make(role) for role in ('base','past','current')]
        identity=lambda x:x
        guard=lambda ws:dict(selected_fp32_shape_valid=all(w.dtype==torch.float32 for w in ws),nonselected_unchanged=True)
        p=AProblem(we,d0,(4,8),base,past,current,identity,identity,identity,identity,
                   (.01,.01),(.3,.6),1.,guard,dict(actual_shared_kernel=True),wa=wa)
        endpoint,row=run_os(p)
        self.assertTrue(all(torch.isfinite(w).all() for w in endpoint))
        self.assertLess(row['current_profile_gradient_observed_norm'],1e-6)
        for role in ('base','past','current'):
            self.assertGreater(row['counts'][role]['jvp_calls'],0)
            self.assertGreater(row['counts'][role]['vjp_calls'],0)
        self.assertFalse(row['current_profile_gradient_in_u'])

    def test_initial_marker_never_precedes_nonpositive_curvature_failure(self):
        for curvature in (-1., 0.):
            with self.subTest(curvature=curvature):
                p,*_=fixture()
                p.base.matrix=torch.eye(4)*curvature
                p.past.matrix.zero_();p.current.matrix.zero_()
                p.native_metric=lambda x:tuple(torch.zeros_like(v) for v in x)
                p.raw_native_metric=lambda x:tuple(torch.zeros_like(v) for v in x)
                events=[]
                with self.assertRaisesRegex(FloatingPointError,'^PCG_NONPOSITIVE_CURVATURE$'):
                    run_os(p,on_stage=lambda name,receipt:events.append(name))
                self.assertEqual(events,['LINEARIZATION_COMPLETE'])

    def test_all_zero_action_is_valid_without_positive_operator_marker(self):
        p,*_=fixture(d0_zero=True,zero_a=True)
        for panel in (p.base,p.past,p.current):panel.gradient.zero_()
        events=[]
        endpoint,row=run_os(p,on_stage=lambda name,receipt:events.append(name))
        self.assertTrue(all(torch.equal(w,b) for w,b in zip(endpoint,p.wa)))
        self.assertEqual(row['actual_delta_norm'],0.)
        self.assertEqual(events,['LINEARIZATION_COMPLETE'])
        self.assertFalse(row['initial_positive_full_operator_observed'])

    def test_zero_first_call_does_not_consume_first_positive_marker(self):
        from project.run_scripts.multilayer_joint_compensation.track_a import protocol
        original=protocol.solve_constrained
        p,*_=fixture()
        events=[]
        def with_zero_probe(operator,u,a,t_e,**kwargs):
            operator(tuple(torch.zeros_like(v) for v in u))
            return original(operator,u,a,t_e,**kwargs)
        with patch.object(protocol,'solve_constrained',with_zero_probe):
            _,row=run_os(p,on_stage=lambda name,receipt:events.append((name,receipt)))
        self.assertEqual([name for name,_ in events],['LINEARIZATION_COMPLETE','FIRST_FULL_OPERATOR_MATVEC_COMPLETE'])
        self.assertEqual(events[1][1]['operator_counts']['full_operator_matvecs'],2)
        self.assertGreater(events[1][1]['quadratic_form'],0.)
        self.assertTrue(row['initial_positive_full_operator_observed'])


if __name__=='__main__':unittest.main()
