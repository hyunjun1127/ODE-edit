import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .policy import choose,strict_shadow,materialized,past_indices,event_id
from .common import ARMS,EXPECTED

class PolicyTests(unittest.TestCase):
    def row(self,name='N4',**kw):
        return dict(dict(candidate_id=name,E=.01,H=.02,D=.1,S_cur=[1],S_past=[2],L8_zero=True,action_norm=1.),**kw)
    def test_plateau(self):
        a=self.row();b=self.row('L',E=.05009,D=.01)
        self.assertEqual(choose([a,b])['selected'],'L');self.assertEqual(choose([a,b],plateau=False)['selected'],'N4')
    def test_id_not_count(self):
        self.assertEqual(choose([self.row(),self.row('L',S_cur=[3],D=0)])['selected'],'N4')
    def test_past_constraint(self):
        self.assertEqual(choose([self.row(),self.row('L',H=.0202,D=0)])['selected'],'N4')
    def test_nonfinite_fatal(self):
        for key in ('E','H','D','action_norm'):
            with self.assertRaises(ValueError):choose([self.row(),self.row('L',**{key:float('nan')})])
    def test_tie_n4_first(self):
        self.assertEqual(choose([self.row(),self.row('L',D=.0999995,action_norm=0)])['selected'],'N4')
    def test_tie_l8_zero(self):
        rows=[self.row(D=.2),self.row('a',D=0,L8_zero=False,action_norm=0),self.row('b',D=0)]
        self.assertEqual(choose(rows)['selected'],'b')
    def test_zero_one_and_fp32(self):
        u=torch.randn(7,11);v=torch.randn(7,11)
        self.assertTrue(torch.equal(materialized(u,v,0),u));self.assertTrue(torch.equal(materialized(u,v,1),v))
        for g in (.5,.75):self.assertTrue(torch.equal(materialized(u,v,g),u+g*(v-u)))
        with self.assertRaises(ValueError):materialized(u,v,.3)
    def records(self,n):
        return [dict(case_id=i,requested_rewrite=dict(subject=str(i),relation_id='r',target_new={'str':'new'})) for i in range(n)]
    def test_past_received_only(self):
        r=self.records(200);a=past_indices(list(range(100)),r[100:],r)
        self.assertEqual(len(a),64);self.assertTrue(set(a)<=set(range(100)))
        self.assertEqual(a,past_indices(list(range(100)),r[100:],r));self.assertEqual(past_indices([],r[:100],r),[])
    def test_past_latest_overwrite(self):
        r=self.records(4);r[1]['requested_rewrite']['subject']='0';r[3]['requested_rewrite']['subject']='2'
        self.assertEqual(past_indices([0,1,2],[r[3]],r),[1]);self.assertNotEqual(event_id(0,r[0]),event_id(1,r[1]))
    def test_shadow_has_no_mutation(self):
        rows=[self.row(),self.row('L',D=0)];before=copy.deepcopy(rows);strict_shadow(rows);self.assertEqual(rows,before)
    def test_plan_counts(self):
        self.assertEqual(sum(EXPECTED[a][0]*10 for a in ARMS),12000)
        self.assertEqual(sum(EXPECTED[a][1]*10 for a in ARMS),150)
        self.assertEqual(sum(EXPECTED[a][2]*10 for a in ARMS),190)

class EngineTests(unittest.TestCase):
    def test_all_policy_inventory_and_restore(self):
        from .engine import generate
        class Toy:
            lock={'editor_sha256':'fixture'};context=[['{}']];timing={}
            hp={4:type('HP',(),{})(),8:type('HP',(),{})()}
            def __init__(self):self.w={4:torch.zeros(2,3),8:torch.zeros(2,3)};self.targets=0;self.solves=0
            def snapshot(self):return dict(W=copy.deepcopy(self.w),M={},rng={},contexts=self.context)
            def state(self):return dict(W={str(l):w.tolist() for l,w in self.w.items()},M={},P={})
            def restore(self,s):self.w=copy.deepcopy(s['W'])
            def apply(self,w,rng):self.w=copy.deepcopy(w)
            def fit(self,r,l):
                self.targets+=100;self.solves+=1;self.w[l]+=1
                return dict(weight=self.w[l].clone(),receipt=dict(layer=l,compute_z=100,solve=1))
            def targets8(self,r):self.targets+=100;return torch.zeros(2,100),dict(compute_z=100),[]
            def terminal_fit(self,r,l,z):
                self.solves+=1;self.w[l]+=2
                return dict(weight=self.w[l].clone(),receipt=dict(layer=l,compute_z=0,solve=1,readout_layer=8))
        for arm in ARMS:
            with tempfile.TemporaryDirectory() as temp:
                rt=Toy();start=rt.state();c,m,e,f=generate(rt,[dict(case_id=i) for i in range(100)],arm,Path(temp)/arm)
                self.assertEqual((rt.targets,rt.solves,len(c)),EXPECTED[arm]);self.assertEqual(rt.state(),start)
                if arm=='TD':self.assertIn('N4',c)

if __name__=='__main__':unittest.main()
