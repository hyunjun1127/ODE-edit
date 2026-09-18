import copy
import unittest
from .review import reduce_raw,pair,digest,audit_scan,bind_endpoint,audit_qp_numbers
import numpy as np

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
    def test_reference_arithmetic_and_tie_identity(self):
        p=dict(position=128,target=2,choice=1,margin=0.,kappa=0.,mu=0.,logp=-1.,d=.2,preserved=False)
        x=dict(documents=[dict(source_row_id='synthetic',positions=[p],all_choices=False)],
            positions=1,token_flips=1,sequence_retained=0,all_choices=False)
        self.assertEqual(audit_scan(x,1)['token_flips'],1)
        y=copy.deepcopy(x);y['documents'][0]['positions'][0]['mu']=.1
        with self.assertRaisesRegex(ValueError,'REFERENCE_MU'):audit_scan(y,1)
        y=copy.deepcopy(x);y['documents'][0]['positions'][0]['preserved']=True
        with self.assertRaisesRegex(ValueError,'REFERENCE_CHOICE_ID'):audit_scan(y,1)
        y=copy.deepcopy(x);y['token_flips']=0
        with self.assertRaisesRegex(ValueError,'REFERENCE_AGGREGATE'):audit_scan(y,1)
    def test_endpoint_binding_rejects_stale_observer(self):
        x=dict(selection_seal=dict(endpoint_weight_sha256='abc'),compatibility=dict(endpoint_weight_sha256='abc',request_order_sha256=digest([1])),request_order=digest([1]),requests=1)
        bind_endpoint(x,'abc',[1])
        with self.assertRaisesRegex(ValueError,'ENDPOINT_SHA'):bind_endpoint(x,'other',[1])
        with self.assertRaisesRegex(ValueError,'ORDER_BINDING'):bind_endpoint(x,'abc',[2])
    def test_missing_protected_position_rejected(self):
        p=dict(position=128,target=2,choice=2,margin=1.,kappa=.1,mu=.9,logp=-1.,d=.2,preserved=True)
        x=dict(documents=[dict(source_row_id='synthetic',positions=[p],all_choices=True,eos=False,censored=True)],positions=1,token_flips=0,sequence_retained=1,all_choices=True)
        capsule=dict(source_row_id='synthetic',positions=[128,129],y0=[2,3],eos=False,censored=True)
        with self.assertRaisesRegex(ValueError,'CAPSULE_BINDING'):audit_scan(x,1,[capsule])
    def test_independent_KKT_rejects_false_source_pass(self):
        policy=dict(feasibility_absolute=1e-9,feasibility_relative=1e-10,objective_relative=1e-8)
        sol=dict(alpha=[1.,1.],policy=policy,diagnostics=dict(KKT_pass=True))
        self.assertTrue(audit_qp_numbers(np.eye(2),np.ones(2),sol,policy)['independent_locked_row_scaled_KKT'])
        sol['alpha']=[0.,0.]
        with self.assertRaisesRegex(ValueError,'KKT_FAIL'):audit_qp_numbers(np.eye(2),np.ones(2),sol,policy)

if __name__=='__main__':unittest.main()
