import copy
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.multilayer_joint_compensation.track_a.analyze_partial import (
    AnalysisBoundary, attribution, canonical, digest, endpoint_rows, functional_summary,
    member, stats, summarize, verify_package, symmetric_effects)


class PartialAnalysisTests(unittest.TestCase):
    def endpoint(self, new=1., true=2.):
        record=dict(case_id=7,requested_rewrite=dict(prompt='{} is',subject='S',target_new={'str':' N'},target_true={'str':' T'}),
                    paraphrase_prompts=['p1','p2'],neighborhood_prompts=['n'+str(i) for i in range(10)])
        panel={'panels':{'Current100':[0]}}
        rows=[]
        for metric,prompts in [('RS',['S is']),('PS',record['paraphrase_prompts']),('NS',record['neighborhood_prompts'])]:
            for i,p in enumerate(prompts):
                rows.append(dict(panel='Current100',metric=metric,case_id=7,prompt_index=i,
                  identity=digest([7,i,p,' N',' T']),new_nll=new,true_nll=true,margin=true-new,
                  success=true<new if metric=='NS' else new<true,new_strict=True,true_strict=False,
                  new_token_count=1,true_token_count=2,new_token_correct=1,true_token_correct=1))
        return dict(rows=rows,pairs=13,panel_identity=digest(panel),controller_influence=0),panel,[record]

    def test_endpoint_exact_inventory_direction_and_tie(self):
        doc,panel,records=self.endpoint()
        rr=endpoint_rows(doc,panel,records)
        self.assertEqual(sum(r['success'] for r in rr),3)
        doc,panel,records=self.endpoint(2.,2.)
        self.assertFalse(any(r['success'] for r in endpoint_rows(doc,panel,records)))

    def test_endpoint_wrong_success_duplicate_nonfinite_fail(self):
        doc,panel,records=self.endpoint();doc['rows'][0]['success']=False
        with self.assertRaisesRegex(AnalysisBoundary,'SUCCESS'):endpoint_rows(doc,panel,records)
        doc,panel,records=self.endpoint();doc['rows'][1]=doc['rows'][0]
        with self.assertRaisesRegex(AnalysisBoundary,'SAMPLE_ORDER'):endpoint_rows(doc,panel,records)
        doc,panel,records=self.endpoint();doc['rows'][0]['new_nll']=float('nan')
        with self.assertRaisesRegex(AnalysisBoundary,'NONFINITE'):endpoint_rows(doc,panel,records)

    def test_no_missing_observation_zero_fill(self):
        doc,_,_=self.endpoint();rr=[r for r in doc['rows'] if r['metric']=='RS']
        s=summarize(rr,'We');self.assertEqual(len(s),1);self.assertEqual(s[0]['metric'],'RS')

    def test_signed_attribution_negative_interaction(self):
        doc,_,_=self.endpoint();states={}
        for name,nll in [('We',10.),('We_plus_D4',5.),('We_plus_D8',7.),('A0',3.)]:
            states[name]=copy.deepcopy(doc['rows'])
            for row in states[name]:row['new_nll']=nll
        rows,agg=attribution(states)
        self.assertEqual(rows[0]['standalone_E4'],5);self.assertEqual(rows[0]['standalone_E8'],3)
        self.assertEqual(rows[0]['standalone_E48'],7);self.assertEqual(rows[0]['interaction'],-1)
        self.assertEqual(rows[0]['signed_L4'],4.5);self.assertEqual(rows[0]['signed_L8'],2.5)
        self.assertEqual(rows[0]['signed_L4']+rows[0]['signed_L8'],7.)
        self.assertEqual(len(agg),16)

    def test_symmetric_identity_negative_zero_and_scaled(self):
        for a,b,c in [(5.,3.,7.),(-4.,2.,-2.),(3.,-3.,0.),(1e6,-2e6,3e6)]:
            x=symmetric_effects(a,b,c)
            self.assertEqual(x['signed_L4']+x['signed_L8'],c)
            self.assertNotIn('share',x)
        with self.assertRaisesRegex(AnalysisBoundary,'NONFINITE'):
            symmetric_effects(float('nan'),1.,2.)

    def test_functional_uneven_weighted_reduction_not_mean_of_chunks(self):
        d={'Base':dict(context_rows=[dict(identity='a|b',values=[1.,2.],nll=[3.,4.],context_weights=[.2,.3]),
                                    dict(identity='c',values=[4.],nll=[5.],context_weights=[.5])],value=2.8,mean_nll=4.3)}
        b={'context_counts':{'Base':3},'inventory':{'bank':{'Base':[1,2,3]}}}
        summary,_,checks=functional_summary(d,b,{'raw_native_risk':{'Base':.7}})
        self.assertAlmostEqual(summary[0]['weighted_value'],2.8)
        self.assertAlmostEqual(summary[0]['A0_to_native_ratio'],4.)
        self.assertEqual(len(checks),2)
        canonical(checks)  # numpy scalar leakage must not break final receipt serialization.
        d['Base']['value']=2.25
        with self.assertRaisesRegex(AnalysisBoundary,'WEIGHTED_REDUCTION'):functional_summary(d,b,{'raw_native_risk':{}})

    def test_package_hash_bytes_mode_and_root(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'x.json').write_bytes(b'{}\n')
            members=[member(p/'x.json')]
            (p/'terminal.json').write_bytes(canonical(dict(members=members,members_root=digest(members))))
            _,r=verify_package(p/'terminal.json');self.assertEqual(r['member_count'],1)
            (p/'x.json').write_bytes(b'{ }\n')
            with self.assertRaisesRegex(AnalysisBoundary,'RAW_MEMBER_MISMATCH'):verify_package(p/'terminal.json')

    def test_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'x').write_bytes(b'a');(p/'link').symlink_to(p/'x')
            with self.assertRaisesRegex(AnalysisBoundary,'SYMLINK'):member(p/'link')

    def test_quantile_and_empty_boundary(self):
        self.assertEqual(stats([0,1,2])['median'],1)
        self.assertAlmostEqual(stats([0,1,2])['p90'],1.8)
        with self.assertRaises(AnalysisBoundary):stats([])


if __name__=='__main__':unittest.main()
