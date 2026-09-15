"""Deterministic CPU fixtures; explicitly not actual Llama/GPU parity."""
from types import SimpleNamespace
import sys
import unittest

import torch

from .model_adapter import (EpisodeAdapter, ModelBoundary, anchored_weight,
                           capture_native_fit, tensor_sha)
from .technical import TechnicalNumerics, directional_fd, functional_materialized


class TinyTokenizer:
    bos_token_id = 1
    unk_token_id = 0
    eos_token_id = 2
    pad_token_id = 2
    padding_side = 'right'

    def encode(self, text, add_special_tokens=False):
        ids = [3+ord(c)%6 for c in str(text).strip()]
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def __call__(self, text, add_special_tokens=True):
        return dict(input_ids=self.encode(text, add_special_tokens))


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(9, 5)
        self.down_proj = torch.nn.Linear(5, 3, bias=False)
        self.head = torch.nn.Linear(3, 9, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        values = self.embed(input_ids) * attention_mask[...,None]
        denominator = attention_mask.cumsum(1).clamp_min(1)[...,None]
        # Prefix activation propagation makes the down-projection affect more
        # than only scored positions while preserving causal left-pad masking.
        hidden = values.cumsum(1) / denominator
        output = self.down_proj(hidden)
        logits = self.head(torch.tanh(output))
        return SimpleNamespace(logits=logits)


class TinyTeacher:
    def __init__(self, model):
        self.ids = torch.tensor([[1]+[3+i%6 for i in range(256)]], dtype=torch.long)
        with torch.no_grad():
            logits = model(self.ids, torch.ones_like(self.ids)).logits
            self.logp0 = torch.log_softmax(logits[:,128:256,:], -1).clone()

    def indices(self, role):
        if role == 'S64':
            return [0,1,2]
        if role == 'Dev128':
            return [3,4]
        raise ModelBoundary('UNAPPROVED_GENERIC_ROLE')

    def document(self, index, device):
        return self.ids.to(device), self.logp0.to(device), 'fixture-'+str(index)


def records(count=19):
    return [dict(case_id=i, requested_rewrite=dict(prompt='{} writes', subject='A'+str(i),
        target_new=dict(str='target' if i%2 else 'x'), target_true=dict(str='old')),
        paraphrase_prompts=[], neighborhood_prompts=[]) for i in range(count)]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(345)
        self.model=TinyCausal().float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.teacher=TinyTeacher(self.model)
        self.adapter=EpisodeAdapter(self.model,TinyTokenizer(),'down_proj.weight',self.teacher)
        self.records=records()
        with torch.no_grad():
            self.model.down_proj.weight.add_(.07)
        self.raw=self.model.down_proj.weight.detach().clone()
        self.a=torch.randn(len(self.records),5)*.15
        self.adapter.set_episode(self.raw,self.a)

    def test_c0_raw_bytes_and_vjp(self):
        raw=torch.tensor([[-0.,1.,2.,3.,4.],[1.,2.,3.,4.,5.],[0.,0.,0.,0.,0.]])
        a=torch.randn(2,5)
        c=torch.zeros(3,2,requires_grad=True)
        value=anchored_weight(c,raw,a)
        self.assertEqual(tensor_sha(raw),tensor_sha(value))
        cot=torch.randn_like(value)
        actual=torch.autograd.grad((value*cot).sum(),c)[0]
        self.assertTrue(torch.equal(actual,cot@a.T))

    def test_nonzero_all_token_functional_materialized(self):
        correction=torch.randn(3,len(self.records))*.01
        before=tensor_sha(self.model.down_proj.weight)
        receipt=functional_materialized(self.adapter,correction)
        self.assertEqual(receipt['status'],'PASS')
        self.assertTrue(receipt['all_logits'])
        self.assertEqual(before,tensor_sha(self.model.down_proj.weight))

    def test_current_source_layout_and_strict(self):
        observed=self.adapter.current(self.records)
        self.assertEqual(observed['denominator'],len(self.records))
        self.assertEqual(observed['counts']['forwards'],2)
        weighted=[]
        with torch.no_grad():
            for group,ids,attention,positions in self.adapter._current_groups(self.records):
                logits=self.model(ids,attention).logits.float()
                lp=torch.log_softmax(logits,-1)
                for j,(_,_,_,target) in enumerate(group):
                    weighted.append(float(-lp[j,positions[j],:].gather(1,torch.tensor(target)[:,None]).mean()))
        self.assertEqual([r['nll'] for r in observed['rows']],weighted)
        self.assertEqual(observed['strict_ids'],[r['case_id'] for r in observed['rows'] if r['all_tokens_correct']])

    def test_separate_gradient_sweeps_and_forbid_repeat(self):
        result=self.adapter.gradient_sweeps(self.records)
        self.assertEqual(result['receipt']['current_sweeps'],1)
        self.assertEqual(result['receipt']['generic_sweeps'],1)
        self.assertEqual(result['current']['counts']['backwards'],2)
        self.assertEqual(result['generic']['counts']['backwards'],3)
        self.assertTrue(torch.isfinite(result['gE']).all())
        self.assertFalse(torch.equal(result['gE'],result['gD']))
        self.assertEqual(result['current']['rows'],self.adapter.current(self.records)['rows'])
        self.assertEqual(result['generic']['rows'],self.adapter.generic()['rows'])
        with self.assertRaisesRegex(ModelBoundary,'ALREADY_CONSUMED'):
            self.adapter.gradient_sweeps(self.records)

    def test_gradient_request_mean_mass(self):
        result=self.adapter.gradient_sweeps(self.records)
        c=torch.zeros(3,len(self.records),requires_grad=True)
        expected=torch.zeros_like(c)
        # Independent one-request manual forwards. Different target lengths do
        # not reweight request mass; each request receives exactly 1/N.
        for record in self.records:
            for group,ids,attention,pos in self.adapter._current_groups([record]):
                logits=self.adapter._forward(ids,attention,c)
                nll,_=self.adapter._current_values(logits,group,pos)
                expected+=torch.autograd.grad(nll.sum()/len(self.records),c)[0]
        self.assertTrue(torch.allclose(result['gE'],expected,atol=1e-7,rtol=2e-5))

    def test_generic_scoring_shift_and_direction(self):
        result=self.adapter.generic()
        inputs,teacher,_=self.teacher.document(0,torch.device('cpu'))
        with torch.no_grad():
            logits=self.model(inputs,torch.ones_like(inputs)).logits
            lp=torch.log_softmax(logits[:,128:256,:],-1)
            manual=(teacher.exp()*(teacher-lp)).sum(-1).mean()
            nll=-lp.gather(-1,inputs[:,129:257,None]).mean()
        self.assertAlmostEqual(result['D'],float(manual),places=12)
        self.assertEqual(result['rows'][0]['natural_nll'],float(nll))
        self.assertEqual(result['counts']['scored_tokens'],3*128)

    def test_directional_fd_current_and_generic(self):
        result=self.adapter.gradient_sweeps(self.records)
        for objective,key in [('E','gE'),('D','gD')]:
            receipt=directional_fd(self.adapter,self.records,result[key],objective=objective)
            self.assertEqual(receipt['status'],'PASS')
            self.assertEqual(receipt['backward_calls'],0)

    def test_no_model_grad_or_rng_side_effect(self):
        state=torch.random.get_rng_state().clone()
        self.adapter.gradient_sweeps(self.records)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_reject_wrong_microbatch_and_mutable_map(self):
        with self.assertRaisesRegex(ModelBoundary,'MICROBATCH16'):
            EpisodeAdapter(self.model,TinyTokenizer(),'down_proj.weight',self.teacher,current_microbatch=1)
        self.adapter.fixed_a.add_(1)
        with self.assertRaisesRegex(ModelBoundary,'MUTATED'):
            self.adapter.gradient_sweeps(self.records)

    def test_no_dev_backward_or_report_access(self):
        c=torch.zeros(3,len(self.records),requires_grad=True)
        with self.assertRaisesRegex(ModelBoundary,'ONLY_S64'):
            self.adapter._generic('Dev128',c,backward=True)
        with self.assertRaisesRegex(ModelBoundary,'UNAPPROVED'):
            self.adapter.generic('Report256')

    def test_numerical_contract_not_quality_tolerance(self):
        numerical=TechnicalNumerics().to_dict()
        self.assertEqual(numerical['current_positive_quality_tolerance'],0.)
        self.assertEqual(numerical['fd_step_factors'],(1.,.5))


def fake_native_compute_z(model,tok,request,hparams,layer,contexts):
    target_init=torch.ones(3)
    delta=torch.zeros(3,requires_grad=True)
    opt=SimpleNamespace(state={delta:{'step':torch.tensor(0.)}})
    it=0
    lookup_idxs=[1,1]
    kl_distr_init=torch.log_softmax(torch.ones(1,9),-1)
    loss=torch.tensor(.01)
    return target_init+delta


class CaptureTests(unittest.TestCase):
    def test_native_return_frame_anchor_and_counter(self):
        class Fitter:
            module=SimpleNamespace(compute_z=fake_native_compute_z)
            def fit(self,model,tok,hp,history,p,requests,**kwargs):
                targets=[self.module.compute_z(model,tok,r,hp,4,[['{}']]) for r in requests]
                return dict(captures={'compute_z':[x.detach().clone() for x in targets]},receipt={})
        reqs=[dict(case_id=3),dict(case_id=4)]
        result=capture_native_fit(Fitter(),None,None,SimpleNamespace(clamp_norm_factor=.75),None,None,reqs)
        self.assertEqual(result['anchors'].shape,(3,2))
        self.assertEqual(result['target'].shape,(3,2))
        self.assertEqual(result['receipt']['anchor_capture']['adam_updates'],0)
        self.assertEqual(result['receipt']['anchor_capture']['loss_evaluations'],2)
        self.assertIsNone(sys.getprofile())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
