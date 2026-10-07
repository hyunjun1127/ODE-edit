"""Narrow CPU implementation checks, not model experiments or qualification."""
import copy
import unittest
import torch
from .anchor import guarded_anchor
from .profile import modified_profile, forbid_w0, NATIVE
from .calibration import calibrate


class Corrections(unittest.TestCase):
    def test_m1_mean_of_five_norms_and_exclusions(self):
        rows=[dict(request=0,kind='rewrite',global_row=i,lookup=0) for i in range(6)]
        rows.append(dict(request=0,kind='kl',global_row=6,lookup=0))
        h=torch.tensor([[[1000.,0.]],[[3.,4.]],[[-3.,-4.]],[[0.,5.]],[[5.,0.]],[[0.,-5.]],[[9000.,0.]]])
        self.assertEqual(guarded_anchor(h,rows,[0],0,True).item(),5.)
        self.assertEqual(guarded_anchor(h,rows,[0],0,False).item(),1000.)
        rows[0]['lookup']=1;h=torch.cat([h,h],dim=1)
        self.assertTrue(torch.equal(guarded_anchor(h,rows,[0],0,True),h[0,1].norm()))
    def test_prefix_count_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'FIVE_PREFIX'):
            guarded_anchor(torch.ones(1,1,2),[dict(request=0,kind='rewrite',global_row=0,lookup=0)],[0],0,True)
    def test_profiles_and_grace_ablation(self):
        for model,(layers,k,lr,lam) in NATIVE.items():
            p=dict(writer='memit',eligible_layers=layers,K_eval=k,max_updates=k-1,lr=lr,
                   lambda_C=lam,arm='CAP075',beta_base=.75,cap_mode='native',K_grace=12)
            evidence=dict(model=model,layer=layers[0],median_realization=.6)
            if model=='llama3':self.assertEqual(modified_profile(p,'LLAMA_REPRO',evidence),p)
            elif model=='gptj':self.assertEqual(modified_profile(p,'GPTJ_M1',evidence)['K_grace'],12)
            else:
                self.assertEqual(modified_profile(p,'GPT2XL_M1_M2',evidence)['K_grace'],9)
                self.assertEqual(modified_profile(p,'GPT2XL_M1_M3',evidence)['K_grace'],12)
                evidence['median_realization']=.34
                with self.assertRaisesRegex(ValueError,'NOT_RECORDED'):
                    modified_profile(p,'GPT2XL_M1_M3',evidence)
                q=modified_profile(p,'GPT2XL_M1_M2',evidence)
                self.assertEqual(q['lambda_C'],20000.)
    def test_no_w0_default(self):
        with self.assertRaisesRegex(ValueError,'NEW_W0'):
            forbid_w0({})
        forbid_w0(dict(w0_policy='REUSE_ONLY_NO_FORWARD'))
    def test_m3_cpu_native_and_root_logic(self):
        # Function fixtures only; not a calibration receipt for any model.
        K=torch.eye(100,dtype=torch.float64);C=torch.eye(100,dtype=torch.float64)
        self.assertEqual(calibrate(K,C,.5,key_identity_verified=True,stat_identity_verified=True)['lambda_C'],.5)
        r=calibrate(K,C,4.,key_identity_verified=True,stat_identity_verified=True)
        self.assertEqual(r['lambda_C'],1.)
        self.assertEqual(r['median'],.5)
        with self.assertRaisesRegex(ValueError,'IDENTITY'):
            calibrate(K,C,4.,key_identity_verified=False,stat_identity_verified=True)


if __name__=='__main__':unittest.main()
