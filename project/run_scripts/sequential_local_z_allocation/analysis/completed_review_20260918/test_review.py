import copy,json,math,unittest
from reducer import digest,expected,success,validate
from audit import choose,reasons,score_equal,past_expected
from performance import transition

def candidate(name='n4',B=.1,E=.1,strict=(1,),gates=(1.,0.),norm=1.,is_n4=False):
    return dict(state_token=name,gates=list(gates),action_norm=norm,active_layers=[4] if gates[1]==0 else [4,8],is_n4=is_n4,scores=dict(base_kl=B,training_e=E,canonical_e=E,current_strict=list(strict),current_pair=list(strict),past_h=None,past_strict=[],past_pair=None),feasible=True,reasons=[])
class ReducerTest(unittest.TestCase):
    def test_tie_failure_all(self):
        for tag in ['RS','PS','NS']:self.assertFalse(success(dict(new_nll=1.,true_nll=1.),tag))
    def test_canonical_direction(self):
        r=dict(new_nll=1.,true_nll=2.)
        self.assertTrue(success(r,'RS'));self.assertTrue(success(r,'PS'));self.assertFalse(success(r,'NS'))
    def test_hash_unicode_ascii(self):
        import hashlib
        self.assertEqual(digest(['서울']),hashlib.sha256(b'["\\uc11c\\uc6b8"]').hexdigest())
    def test_exact_id_not_count(self):
        n=candidate(is_n4=True);c=candidate('x',B=.01,strict=(2,))
        self.assertIn('CURRENT_STRICT_IDS',reasons(c,n));self.assertEqual(choose([n,c],'C48'),n)
    def test_mean_allowance_boundary(self):
        n=candidate(is_n4=True);c=candidate('x',E=.1001,B=.01)
        self.assertFalse(reasons(c,n));c['scores']['training_e']=.100100001;self.assertIn('CURRENT_TRAINING_MEAN',reasons(c,n))
    def test_no_plateau(self):
        n=candidate(E=.001,is_n4=True);c=candidate('x',E=.01,B=.001)
        self.assertEqual(choose([n,c],'C48'),n)
    def test_global_epsilon_no_accumulation(self):
        n=candidate(is_n4=True,B=.1000015);a=candidate('a',B=.1000008,norm=.5);b=candidate('b',B=.1,norm=2.)
        self.assertIs(choose([n,a,b],'C48'),a)
    def test_raw_tie(self):
        n=candidate(is_n4=True,B=.1000005);c=candidate('x',B=.1,norm=.1)
        self.assertIs(choose([c,n],'C48'),n)
    def test_support_tie_before_norm(self):
        n=candidate(is_n4=True,B=.2);a=candidate('a',B=.1,norm=2.);b=candidate('b',B=.1,norm=.1,gates=(.75,.5))
        self.assertIs(choose([n,b,a],'C48'),a)
    def test_F48_fixed_exception(self):
        n=candidate(is_n4=True);c=candidate('x',E=9.,B=9.,gates=(.75,.5));self.assertIs(choose([n,c],'F48'),c)
    def test_set_serial_order_not_scalar_relaxation(self):
        a={'current_strict':[10,2],'past_pair':None,'training_e':1.};b={'current_strict':[2,10],'past_pair':None,'training_e':1.}
        self.assertTrue(score_equal(a,b));b['training_e']=1.0000000001;self.assertFalse(score_equal(a,b))
    def test_transition_identity_reject(self):
        r=dict(identity='a',case_id=1,prompt_index=0,new_nll=1.,true_nll=2.,new_strict=True,true_strict=False)
        with self.assertRaises(AssertionError):transition([r],[dict(r,identity='b')],'RS')
    def test_transition_lost_gained(self):
        r=dict(identity='a',case_id=1,prompt_index=0,new_nll=1.,true_nll=2.,new_strict=True,true_strict=False)
        x,_=transition([r],[dict(r,new_nll=3.,new_strict=False)],'RS');self.assertEqual((x['lost'],x['gained'],x['delta_pp']),(1,0,-100))
    def test_past_current_overwrite(self):
        def r(i,f):return dict(case_id=i,requested_rewrite=dict(subject=f,relation_id='p',target_new={'str':'x'}))
        rs=[r(0,'A'),r(1,'B'),r(2,'A'),r(3,'B')];p,_=past_expected(rs,3,4);self.assertEqual(p,[2])
if __name__=='__main__':unittest.main()
