"""Review-only CPU regression; no runtime model, scheduler, or original writes."""
import copy
import unittest
from .review_completed_20260919 import reused,check_capsule,replay_guard


class ReviewTests(unittest.TestCase):
    def test_null_proof_is_not_reuse(self):
        self.assertFalse(reused(dict(same_episode_exact_endpoint_reuse=None,reduced_metrics_reuse_proof=None)))
        self.assertTrue(reused(dict(same_episode_exact_endpoint_reuse={'sha':'evidence'})))

    def fixture(self,n=256,eos=False):
        prompt=[1]*129;y=[2]*n
        if eos:y[-1]=9
        row=dict(input_ids=prompt,source_row_id='r0',role='R512')
        cap=dict(**row,actual_length=n,y0=y,tf_input_ids=prompt+y[:-1],score_positions=list(range(128,128+n)),
            attention_mask=[1]*(128+n),position_ids=list(range(128+n)),length_censored=not eos)
        return cap,row

    def test_EOS_one_intermediate_max256(self):
        for n,eos in ((1,True),(55,True),(256,True),(256,False)):
            cap,row=self.fixture(n,eos);self.assertEqual(check_capsule(cap,row,{9}),eos)

    def test_shift_early_EOS_fake_short_and_cross_input_rejected(self):
        for fault in ('shift','earlyEOS','short','input'):
            cap,row=self.fixture()
            if fault=='shift':cap['score_positions'][0]=127
            if fault=='earlyEOS':cap['y0'][0]=9
            if fault=='short':cap,row=self.fixture(2,False)
            if fault=='input':row=copy.deepcopy(row);row['input_ids'][0]=3
            with self.subTest(fault=fault),self.assertRaises(AssertionError):check_capsule(cap,row,{9})

    def anchor(self):
        base=dict(case_id=0,labels=[1],positions=[2],kind='canonical',context=0)
        return {'0:canonical:new':dict(**base,sequence_id='0:canonical:new',branch='new',nll=.1,strict=True,predictions=[1]),
                '0:canonical:old':dict(**base,sequence_id='0:canonical:old',branch='old',nll=.2,strict=False,predictions=[2])}

    def test_per_sequence_strict_and_pair_guard(self):
        a=self.anchor();self.assertEqual(replay_guard(a,a)['new_sequences'],1)
        for fault in ('NLL','ID','pair','nan','identity'):
            b=copy.deepcopy(a)
            if fault=='NLL':b['0:canonical:new']['nll']=.101
            if fault=='ID':b['0:canonical:new'].update(strict=False,predictions=[2])
            if fault=='pair':b['0:canonical:old']['nll']=.1
            if fault=='nan':b['0:canonical:new']['nll']=float('nan')
            if fault=='identity':b['0:canonical:new']['positions']=[3]
            with self.subTest(fault=fault),self.assertRaises(AssertionError):replay_guard(a,b)


if __name__=='__main__':unittest.main()
