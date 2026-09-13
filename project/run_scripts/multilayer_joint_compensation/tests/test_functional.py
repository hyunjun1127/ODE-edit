import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.functional import (
    OutputBatch, FunctionalPanel, values, context_nll, logit_ggn_action)
from project.run_scripts.multilayer_joint_compensation.linear_solve import dot


DT=torch.float64


def fixture(chunk_size=4):
    # Two stacked editable blocks: a change to block4 changes block8 inputs.
    gen=torch.Generator().manual_seed(931)
    inputs=torch.randn(8,3,generator=gen,dtype=DT)
    weights=(torch.randn(3,3,generator=gen,dtype=DT)*.1,
             torch.randn(3,5,generator=gen,dtype=DT)*.1)
    def full(ws):return torch.tanh(inputs@ws[0])@ws[1]
    teachers=full(tuple(w+.04 for w in weights)).detach().log_softmax(-1)
    targets=torch.tensor([0,1,2,3,4,0,1,2])
    reference=-teachers.gather(1,targets[:,None]).squeeze(1).reshape(4,2).mean(-1)
    batches=[]
    for start in range(0,4,chunk_size):
        end=min(start+chunk_size,4);i0=start*2;i1=end*2;c=end-start
        batches.append(OutputBatch(lambda ws,i0=i0,i1=i1:full(ws)[i0:i1],targets[i0:i1],
            torch.arange(c).repeat_interleave(2),torch.full((2*c,),.5,dtype=DT),
            torch.full((c,),.25,dtype=DT),teachers[i0:i1],reference[start:end],
            identity=f'contexts:{start}:{end}',input_tokens=8))
    return weights,batches


class TestFunctional(unittest.TestCase):
    def test_global_mean_microbatch_1_2_4(self):
        for role in ('base','past','current'):
            linear=[];actions=[]
            for chunk in (1,2,4):
                w,batches=fixture(chunk);p=FunctionalPanel(batches,role)
                linear.append(p.linearize(w,need_nll_gradient=True))
                actions.append(p.ggn(w,tuple(torch.ones_like(x)*.03 for x in w)))
            for i in (0,1):
                self.assertAlmostEqual(linear[i].value,linear[2].value,places=13)
                self.assertAlmostEqual(linear[i].mean_nll,linear[2].mean_nll,places=13)
                for a,b in zip(linear[i].gradient,linear[2].gradient):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)
                for a,b in zip(actions[i],actions[2]):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

    def test_true_gradient_direction_fd(self):
        w,batches=fixture();d=tuple(torch.ones_like(x)*.02 for x in w)
        for role in ('base','past','current'):
            p=FunctionalPanel(batches,role);r=p.linearize(w)
            eps=1e-5
            plus=p.value(tuple(x+eps*y for x,y in zip(w,d)))
            minus=p.value(tuple(x-eps*y for x,y in zip(w,d)))
            self.assertAlmostEqual((plus-minus)/(2*eps),dot(r.gradient,d),places=8)

    def test_full_cross_block_psd_symmetry(self):
        w,batches=fixture();gen=torch.Generator().manual_seed(937)
        x=tuple(torch.randn(t.shape,generator=gen,dtype=DT) for t in w)
        y=tuple(torch.randn(t.shape,generator=gen,dtype=DT) for t in w)
        for role in ('base','past','current'):
            p=FunctionalPanel(batches,role);fx=p.ggn(w,x);fy=p.ggn(w,y)
            self.assertGreaterEqual(dot(x,fx),-1e-12)
            self.assertAlmostEqual(dot(x,fy),dot(fx,y),places=11)
        cross=FunctionalPanel(batches,'base').ggn(w,(x[0],torch.zeros_like(x[1])))[1]
        self.assertGreater(float(cross.norm()),1e-5)

    def test_past_both_curvature_terms(self):
        w,batches=fixture();b=batches[0]
        logits=b.logits_fn(w).detach().requires_grad_(True)
        nll=context_nll(logits.log_softmax(-1),b).detach()
        # Interior of inactive, quadratic and linear psi regions.
        b.reference_nll=nll-torch.tensor([-.03,.02,.07,.2],dtype=DT)
        direction=torch.arange(logits.numel(),dtype=DT).reshape_as(logits)/logits.numel()
        loss=values(logits,b,'past',.1)[0]
        grad=torch.autograd.grad(loss,logits,create_graph=True)[0]
        exact=torch.autograd.grad((grad*direction).sum(),logits)[0]
        actual=logit_ggn_action(logits.detach(),direction,b,'past',.1)
        torch.testing.assert_close(actual,exact,atol=1e-12,rtol=1e-10)
        self.assertEqual(float(actual[:2].norm()),0.)
        self.assertGreater(float(actual[6:].norm()),0.) # psi' CE Fisher remains.

    def test_current_profile_outer_not_residual_hessian(self):
        w,batches=fixture();b=batches[0]
        logits=b.logits_fn(w).detach().requires_grad_(True)
        direction=torch.randn_like(logits)
        # At teacher/reference, residual is zero: exact logit Hessian = GGN.
        b.teacher_logp=logits.detach().log_softmax(-1)
        b.reference_nll=context_nll(b.teacher_logp,b)
        grad=torch.autograd.grad(values(logits,b,'current',.1)[0],logits,create_graph=True)[0]
        exact=torch.autograd.grad((grad*direction).sum(),logits)[0]
        torch.testing.assert_close(logit_ggn_action(logits.detach(),direction,b,'current',.1),exact,atol=1e-11,rtol=1e-10)

    def test_empty_and_invalid_weights(self):
        w,b=fixture()
        panel=FunctionalPanel([],'base');self.assertEqual(panel.value(w),0.)
        self.assertEqual(dot(panel.linearize(w).gradient,w),0.)
        b[0].context_weights*=2
        with self.assertRaisesRegex(ValueError,'MEAN_NOT_ONE'):FunctionalPanel(b,'base')


if __name__=='__main__':unittest.main()
