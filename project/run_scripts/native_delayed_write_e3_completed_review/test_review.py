"""Narrow CPU reducer regressions, not model or runtime validation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from .common import digest, save
from .reducer import load_scores, compare, ci


def fixture():
    panel=[]; raw=[]
    for kind,label in [('R','true'),('R','new'),('N','true'),('N','new')]+[('GENERAL','natural')]*128:
        i=len(panel);cid=i-4 if kind=='GENERAL' else kind
        p=dict(row_id=str(i),pair_id=str(cid),panel=kind,kind=kind,label=label,case_id=cid,subject=str(cid),
            prompt_cluster=str(cid),input_ids=[1,2],positions=[0],target_ids=[2],prompt_index=0)
        r={k:p[k] for k in ('row_id','pair_id','panel','kind','label','case_id','subject','prompt_cluster')}
        r.update(input_sha=digest([[1,2],[0],[2]]),target_count=1,token_nll=[1.],token_predictions=[2],
            token_correct=1,strict=True,nll=1.)
        panel.append(p);raw.append(r)
    return panel,raw


class ReviewTests(unittest.TestCase):
    def parse(self,panel,raw):
        with tempfile.TemporaryDirectory() as d:
            save(Path(d)/'0000.json',raw)
            return load_scores(d,panel)

    def test_tie_is_failure_and_tf_separate(self):
        p,r=fixture();pairs,g,v=self.parse(p,r)
        self.assertEqual(len(g),128)
        self.assertTrue(all(x['tie'] and not x['success'] and x['strict'] for x in pairs))

    def test_direction_and_margin(self):
        p,r=fixture();r[1]['nll']=r[1]['token_nll'][0]=.5;r[3]['nll']=r[3]['token_nll'][0]=2.
        pairs,_,_=self.parse(p,r)
        self.assertEqual([x['success'] for x in pairs],[True,True])
        self.assertEqual([x['desired_margin'] for x in pairs],[.5,1.])

    def test_bad_order(self):
        p,r=fixture();r[0],r[1]=r[1],r[0]
        with self.assertRaises(AssertionError):self.parse(p,r)

    def test_input_hash(self):
        p,r=fixture();r[0]['input_sha']='wrong'
        with self.assertRaises(AssertionError):self.parse(p,r)

    def test_nonfinite(self):
        p,r=fixture();r[0]['nll']=float('nan')
        # JSON strict writer also rejects non-finite values before parsing.
        with self.assertRaises(ValueError):self.parse(p,r)

    def test_false_strict(self):
        p,r=fixture();r[0]['strict']=False
        with self.assertRaises(AssertionError):self.parse(p,r)

    def test_token_count(self):
        p,r=fixture();r[0]['target_count']=2
        with self.assertRaises(AssertionError):self.parse(p,r)

    def test_transition_not_just_net_count(self):
        p,r=fixture();x,_,_=self.parse(p,r);a=[x[0],dict(x[0],pair_id='other')]
        a[0]['success']=True;b=copy.deepcopy(a);b[0]['success']=False;b[1]['success']=True
        rows=compare('test',a,b,[])
        self.assertEqual((rows[0]['gained'],rows[0]['lost']),(1,1))
        self.assertEqual(rows[0]['before_success'],rows[0]['after_success'])

    def test_connected_cluster_and_seed(self):
        rows=[dict(case_id=1,subject='a',prompt_cluster='p'),dict(case_id=2,subject='b',prompt_cluster='p'),
              dict(case_id=3,subject='b',prompt_cluster='q'),dict(case_id=4,subject='c',prompt_cluster='r')]
        a=ci([1.,2.,3.,4.],rows)
        self.assertEqual(a['cluster_count'],2);self.assertEqual(a,ci([1.,2.,3.,4.],rows))

    def test_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';save(p,dict(ok=True))
            with self.assertRaises(FileExistsError):save(p,{})

if __name__=='__main__':unittest.main()
