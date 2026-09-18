import copy
import unittest
from .enf_sequential_review import pairs, paired, quality
from .enf_tensor_audit import past_ids

def observation():
    raw={}; metrics={}
    for metric,prefix,n in [('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)]:
        for branch in ('new','true'):
            raw[prefix+'_target_'+branch]=[dict(case_id=1,prompt_index=i,prompt='x',target=branch,
                target_token_ids=[2 if branch=='new' else 3],nll=1.,token_correct=[True]) for i in range(n)]
        metrics[metric]=dict(numerator=0)
    return dict(requests=1,raw=raw,metrics=metrics)

class ReviewTests(unittest.TestCase):
    def test_ties_failure(self): self.assertFalse(any(r['success'] for r in pairs(observation())))
    def test_metric_inequality(self):
        o=observation()
        for k,v in o['raw'].items():
            if k.endswith('_new'):
                for r in v:r['nll']=.5
        o['metrics']['RS']['numerator']=1;o['metrics']['PS']['numerator']=2
        r=pairs(o);self.assertEqual(sum(x['success'] for x in r),3)
    def test_nonfinite(self):
        o=observation();o['raw']['rewrite_target_new'][0]['nll']=float('nan')
        with self.assertRaises(AssertionError):pairs(o)
    def test_duplicate(self):
        o=observation();o['raw']['rephrase_target_new'][1]=copy.deepcopy(o['raw']['rephrase_target_new'][0]);o['raw']['rephrase_target_true'][1]=copy.deepcopy(o['raw']['rephrase_target_true'][0])
        with self.assertRaises(AssertionError):pairs(o)
    def test_prompt_mismatch(self):
        o=observation();o['raw']['rewrite_target_true'][0]['prompt']='different'
        with self.assertRaises(AssertionError):pairs(o)
    def test_pair_target_identity(self):
        a=pairs(observation());b=copy.deepcopy(a);b[0]['identity']='wrong'
        with self.assertRaises(AssertionError):paired(a,b,'test',[])
    def test_id_not_count(self):
        a={'1:new':dict(branch='new',kind='native',nll=1.,strict=True),'2:new':dict(branch='new',kind='native',nll=1.,strict=False)}
        b=copy.deepcopy(a);b['1:new']['strict']=False;b['2:new']['strict']=True
        reasons,_=quality(b,a);self.assertIn(['1:new','STRICT_ID_LOST'],reasons)
    def test_individual_nll(self):
        a={'1:new':dict(branch='new',kind='native',nll=1.,strict=False),'2:new':dict(branch='new',kind='native',nll=1.,strict=False)}
        b=copy.deepcopy(a);b['1:new']['nll']=1.001;b['2:new']['nll']=.9
        reasons,_=quality(b,a);self.assertIn(['1:new','PER_SEQUENCE_NLL'],reasons)
    def test_past_latest_overwrite(self):
        def r(i,s):return dict(case_id=i,requested_rewrite=dict(subject=s,relation_id='r'))
        self.assertEqual(past_ids([r(1,'a'),r(2,'a'),r(3,'b')],[r(4,'b')]),[2])
    def test_empty_past(self):self.assertEqual(past_ids([],[]),[])

if __name__=='__main__':unittest.main()
