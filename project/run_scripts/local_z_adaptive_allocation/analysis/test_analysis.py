"""Focused CPU negative/denominator/selector/transition/publication fixtures."""
import copy
import math
import unittest
import torch
from .metrics import digest,summary,pair
from .reduce import selector,past,statuses,scheduler_tables
from .audit import strict_pair
from .tensors import materialize,tensor_hash,describe

def row(i,new,true,ns=False):
    return dict(case_id=i,prompt_index=0,identity=digest([i,0]),new_nll=new,true_nll=true,
        success=true<new if ns else new<true,desired_margin=new-true if ns else true-new,
        new_strict=new<.5,true_strict=true<.5,new_token_count=1,new_token_correct=int(new<.5),true_token_count=1,true_token_correct=int(true<.5))

def candidate(name='N4',**kw):
    return dict(dict(candidate_id=name,E=.01,H=None,D=.1,S_cur=[1],S_past=[],L8_zero=True,action_norm=1.),**kw)

def record(i,subject='x',target='y'):
    return dict(case_id=i,requested_rewrite=dict(subject=subject,relation_id='r',target_new={'str':target}))

class AnalysisTests(unittest.TestCase):
    def test_rewrite_ties_failure(self):self.assertEqual(summary([row(1,1,1),row(2,1,2)],'RS')['numerator'],1)
    def test_neighborhood_reverse(self):self.assertEqual(summary([row(1,2,1,True),row(2,1,2,True)],'NS')['numerator'],1)
    def test_same_count_not_same_identity(self):
        r=pair([row(1,1,2),row(2,2,1)],[row(1,2,1),row(2,1,2)],'RS');self.assertEqual((r['lost'],r['gained'],r['delta_pp']),(1,1,0))
    def test_pair_identity_mismatch(self):
        with self.assertRaises(AssertionError):pair([row(1,1,2)],[row(2,1,2)],'RS')
    def test_pair_duplicate(self):
        with self.assertRaises(AssertionError):pair([row(1,1,2)]*2,[row(1,1,2)],'RS')
    def test_cluster_ci_zero(self):
        r=pair([row(1,1,2),row(2,1,2)],[row(1,1,2),row(2,1,2)],'RS',ci=True);self.assertEqual((r['CI95_low'],r['CI95_high']),(0,0))
    def test_strict_not_nll_success(self):
        r=strict_pair([row(1,.1,3)],[row(1,1,3)],'RS');self.assertEqual(r['lost'],1)
    def test_two_p_request_cluster(self):
        a=[row(1,.1,3),row(1,.1,3)];a[1]['prompt_index']=1;a[1]['identity']='p1';b=copy.deepcopy(a);b[1]['new_strict']=False
        self.assertEqual(strict_pair(a,b,'PS')['twoP_lost'],1)
    def test_plateau(self):self.assertEqual(selector([candidate(),candidate('L',E=.04,D=.05)])[0],'L')
    def test_no_plateau_shadow(self):self.assertEqual(selector([candidate(),candidate('L',E=.04,D=.05)],'no_plateau')[0],'N4')
    def test_same_count_strict_loss(self):self.assertEqual(selector([candidate(),candidate('L',S_cur=[2],D=.01)])[0],'N4')
    def test_past_constraint(self):self.assertEqual(selector([candidate(H=.2,S_past=[1]),candidate('L',H=.3,S_past=[1],D=.01)])[0],'N4')
    def test_past_strict_constraint(self):self.assertEqual(selector([candidate(H=.2,S_past=[1]),candidate('L',H=.1,S_past=[2],D=.01)])[0],'N4')
    def test_D_tie_N4_priority(self):self.assertEqual(selector([candidate(),candidate('L',D=.1-5e-7,action_norm=.1)])[0],'N4')
    def test_D_not_tied(self):self.assertEqual(selector([candidate(),candidate('L',D=.1-2e-6)])[0],'L')
    def test_nonfinite_not_fallback(self):
        with self.assertRaises(ValueError):selector([candidate(),candidate('L',D=math.nan)])
    def test_past_empty(self):self.assertEqual(past([record(1)],0,[record(1)]),[])
    def test_past_latest_excludes_current(self):
        rr=[record(1),record(2,target='z'),record(3,'b')];self.assertEqual(past(rr,3,[record(4)]),[2])
    def test_past_latest_same_target(self):self.assertEqual(past([record(1),record(2)],2,[record(3,'b')]),[1])
    def test_status_target_reissue(self):self.assertEqual(statuses([record(1),record(2)]),{1:'ACTIVE',2:'ACTIVE'})
    def test_status_overwrite(self):self.assertEqual(statuses([record(1),record(2,target='z')]),{1:'SUPERSEDED',2:'ACTIVE'})
    def test_FP32_gate_endpoints(self):
        a=torch.tensor([1,3.],dtype=torch.float32);b=torch.tensor([2,5.]);self.assertTrue(torch.equal(materialize(a,b,0),a));self.assertTrue(torch.equal(materialize(a,b,1),b))
    def test_FP32_gate_half(self):self.assertTrue(torch.equal(materialize(torch.tensor([1.]),torch.tensor([3.]),.5),torch.tensor([2.])))
    def test_header_hash_distinct(self):
        t=torch.tensor([1.,2.]);self.assertNotEqual(describe(t)['header_bytes_sha256'],tensor_hash(t))
    def test_exact_job_overlap_receipt_only(self):
        jobs,segments=scheduler_tables();self.assertEqual(len(jobs),8);self.assertEqual(max(r['GPUs'] for r in segments),2)
        self.assertEqual(sum(r['seconds'] for r in segments if r['above_cap1']),19428)
    def test_no_CUDA_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
