import copy
import unittest
from .review import reduce_raw,pair,digest

def fixture():
    raw={};metrics={}
    for tag,prefix,m in [('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)]:
        new=[];true=[];rows=[]
        for c in range(100):
            for j in range(m):
                a=float(c%3);b=1.;prompt=f'synthetic {c} {j}'
                n=dict(case_id=c,prompt_index=j,prompt=prompt,target='new',nll=a,all_tokens_correct=c%2==0)
                t=dict(case_id=c,prompt_index=j,prompt=prompt,target='true',nll=b,all_tokens_correct=c%2!=0)
                new.append(n);true.append(t)
                rows.append(dict(case_id=c,prompt_index=j,identity=digest([c,j,prompt,'new','true']),
                    new_nll=a,true_nll=b,success=b<a if tag=='NS' else a<b,
                    new_strict=n['all_tokens_correct'],true_strict=t['all_tokens_correct']))
        raw[prefix+'_target_new']=new;raw[prefix+'_target_true']=true
        metrics[tag]=dict(rows=rows,numerator=sum(r['success'] for r in rows),denominator=len(rows))
    return dict(raw=raw,metrics=metrics)

class ReviewTests(unittest.TestCase):
    def test_strict_inequality_tie_and_denominators(self):
        r=reduce_raw(fixture(),list(range(100)))
        self.assertEqual([len(r[k]) for k in ['RS','PS','NS']],[100,200,1000])
        self.assertEqual(sum(x['success'] for x in r['RS']),34)
        self.assertEqual(sum(x['success'] for x in r['NS']),330)
        self.assertEqual(sum(x['tie'] for x in r['RS']),33)
    def test_identity_order_corruption(self):
        x=fixture();x['raw']['rewrite_target_true'][0]['prompt']='wrong'
        with self.assertRaisesRegex(ValueError,'PAIR_IDENTITY'):reduce_raw(x,list(range(100)))
        with self.assertRaisesRegex(ValueError,'PAIR_ORDER'):reduce_raw(fixture(),list(reversed(range(100))))
    def test_nonfinite_and_aggregate(self):
        x=fixture();x['raw']['rewrite_target_new'][0]['nll']=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE'):reduce_raw(x,list(range(100)))
        x=fixture();x['metrics']['RS']['numerator']=0
        with self.assertRaisesRegex(ValueError,'METRIC'):reduce_raw(x,list(range(100)))
    def test_paired_lost_gained_identity(self):
        r=reduce_raw(fixture(),list(range(100)))['RS'];x=copy.deepcopy(r)
        x[0]['success']=False;x[1]['success']=True
        p=pair(r,x);self.assertEqual(sum(v['lost'] for v in p),1);self.assertEqual(sum(v['gained'] for v in p),1)
        x[0]['identity']='wrong'
        with self.assertRaisesRegex(ValueError,'CROSS_ENDPOINT'):pair(r,x)

if __name__=='__main__':unittest.main()
