import copy
import hashlib
import unittest
import numpy as np
from .review_supplement import stats,cluster,choice,choice_pair
from .review_local_solver import audit


class SupplementTests(unittest.TestCase):
    def fixture(self):
        return dict(rows=[dict(source_row_id='id',capsule_sha256='a'*64,scored_positions=2,
            positions=[128,129],labels=[1,2],predictions=[1,3],correct=[True,False],
            mismatches=1,margins=[0.,-2.],mu=-2.)],phi_reference=4.,mismatches=1)

    def test_choice_tie_is_ID_not_strict_margin(self):
        r=choice(self.fixture());self.assertEqual(r['ties'],1);self.assertEqual(r['mismatches'],1)
        self.assertEqual(r['Phi'],4.)

    def test_choice_corrupt_count_and_position_fail(self):
        for field,value in [('mismatches',0),('positions',[127,128])]:
            d=self.fixture();d['rows'][0][field]=value
            with self.assertRaises(ValueError):choice(d)

    def test_same_count_different_choice_set(self):
        d=self.fixture();other=copy.deepcopy(d);other['rows'][0]['correct']=[False,True]
        r=choice_pair(d,other);self.assertEqual((r['native_safe_new_flips'],r['old_flips_recovered']),(1,1))

    def test_cluster_groups_related_prompts(self):
        a=[dict(case_id=i,pair_id=str((i,j)),success=False,true_nll=1.,new_nll=1.,desired_margin=0.) for i in range(2) for j in range(2)]
        b=[dict(r,success=r['case_id']==0) for r in a]
        r=cluster(a,b,np.array([[0,0],[1,1]]))['success_pp']
        self.assertEqual(r['mean'],50.)
        self.assertAlmostEqual(r['low'],2.5);self.assertAlmostEqual(r['high'],97.5)

    def test_hash_protocols_are_not_interchangeable(self):
        payload=np.ones((2,3),dtype=np.float32).tobytes()
        self.assertNotEqual(hashlib.sha256(payload).hexdigest(),hashlib.sha256(b'(2, 3)|torch.float32|'+payload).hexdigest())

    def test_nonfinite_stats_rejected(self):
        with self.assertRaises(ValueError):stats([float('inf')])

    def test_saved_solver_arithmetic_and_corruption(self):
        phase=dict(coefficients=[0.],risk=.5,risk_limit=.5,certification_gap_tolerance=1e-12,
            dual_audit=dict(labels=['linear:1'],multipliers=[1.],stationarity_vector=[0.]))
        receipt=dict(radius=1.,row_scales=[1.,1.],initial_reference_risk=.5,
            phase1=phase,phase2=dict(phase,dual_audit=dict(labels=[],multipliers=[],stationarity_vector=[0.])))
        r=audit(np.array([-1.,0.]),np.array([[1.],[-1.]]),receipt)
        self.assertEqual(r[0]['stationarity_l2'],0.)
        receipt['phase1']['dual_audit']['multipliers']=[-1.]
        with self.assertRaises(ValueError):audit(np.array([-1.,0.]),np.array([[1.],[-1.]]),receipt)


if __name__=='__main__':unittest.main()
