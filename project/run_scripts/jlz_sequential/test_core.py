import copy
import json
import random
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from project.run_scripts.jlz_pilot.test_integration import DriverIntegrationTest
from project.run_scripts.jlz_pilot.test_prompts import fixture
from project.run_scripts.jlz_pilot.prompts import prepare, native_loss
from project.run_scripts.jlz_pilot.solver import prox_blocks
from .policy import solve
from .oracle import Oracle
from .state import Transaction,state_hash,weights
from .observation import reduce_rows,active_flags

class FixedPolicyTest(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)
    def args(self):
        return torch.zeros(2,3),torch.zeros(2),torch.ones(2)*3,torch.tensor([True,False])
    def test_budget_commits_fresh_initial_but_not_converged(self):
        seen=[]
        def f(x):
            seen.append(x.clone());return float((x-2).square().sum()),2*(x-2),{'point':x.clone()}
        r=solve(f,*self.args(),cap=2,tol=1e-30)
        self.assertEqual(r['status'],'BUDGET_STOP');self.assertTrue(r['commit_eligible'])
        self.assertEqual(r['calls'],2);torch.testing.assert_close(r['x'],seen[-1])
    def test_nonfinite_trial_finite_final_fatal(self):
        seen=[]
        def f(x):
            seen.append(x.clone())
            return (float('nan') if len(seen)==2 else float((x-2).square().sum())),2*(x-2),{}
        with self.assertRaisesRegex(FloatingPointError,'NONFINITE'): solve(f,*self.args())
        self.assertEqual(len(seen),3)
    def test_rejected_trial_not_returned(self):
        def f(x): return (0. if bool((x==0).all()) else 100.),torch.ones_like(x),{'point':x.clone()}
        r=solve(f,*self.args(),max_trials=2)
        self.assertEqual(r['status'],'LINESEARCH_FAILED');self.assertEqual(int(r['x'].count_nonzero()),0)
        self.assertTrue(r['commit_eligible'])
    def test_all_inactive_evaluates_twice(self):
        x,c,r,m=self.args();m[:]=False
        out=solve(lambda x:(0.,torch.zeros_like(x),{'nll':[1.,2.],'kl':[0.,0.]}),x,c,r,m)
        self.assertEqual(out['status'],'POLICY_ZERO_STEP');self.assertEqual(out['calls'],2)
    def test_cap120_and_inactive_preserved(self):
        x,c,r,m=self.args()
        out=solve(lambda x:(float(((x-2)*torch.tensor([1.,4.,8.])).square().sum()),
                            2*(x-2)*torch.tensor([1.,16.,64.]),{}),x,c,r,m,tol=1e-30)
        self.assertLessEqual(out['calls'],120);self.assertFalse(bool(out['x'][1].any()))

class Integration(DriverIntegrationTest):
    def test_shared_selected_matches_reference(self):
        legacy=self.oracle(2); new=Oracle(self.model,self.spec,self.teacher,self.keys,self.adj,self.active,2)
        ref=legacy(self.point);actual=new(self.point)
        self.assertAlmostEqual(ref[0],actual[0],places=5)
        torch.testing.assert_close(ref[1],actual[1],atol=2e-7,rtol=2e-5)
        for l in weights(self.model): torch.testing.assert_close(ref[2]['weights'][l],actual[2]['weights'][l],rtol=0,atol=0)
        self.assertEqual(new.materializations,1)
    def test_transaction_observer_and_partial_append_failure(self):
        for stage in ('commit','append','observer','publish'):
            H={l:torch.zeros(24,24) for l in weights(self.model)};ledger=[];contexts=copy.deepcopy(self.contexts)
            original=state_hash(self.model,H); rng=torch.get_rng_state().clone()
            tx=Transaction(self.model,H,contexts,ledger)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with tx:
                    with torch.no_grad():
                        for w in weights(self.model).values():w.add_(.001)
                    torch.rand(9);random.random();np.random.rand()
                    if stage=='commit':raise RuntimeError('injected')
                    for i,l in enumerate(H):
                        tx.append(l,self.keys[l])
                        if stage=='append' and i==2:raise RuntimeError('injected')
                    if stage=='observer':raise RuntimeError('injected')
                    tx.finish({'batch':1},lambda x: (_ for _ in ()).throw(RuntimeError('injected')))
            self.assertTrue(tx.restored);self.assertEqual(state_hash(self.model,H),original)
            self.assertEqual(ledger,[]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_duplicate_append_fatal_and_10commits_9links(self):
        H={l:torch.zeros(24,24) for l in weights(self.model)};ledger=[]; contexts=copy.deepcopy(self.contexts)
        with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):
            with Transaction(self.model,H,contexts,ledger) as tx:
                tx.append(4,self.keys[4]);tx.append(4,self.keys[4])
        for b in range(10):
            with Transaction(self.model,H,contexts,ledger) as tx:
                if ledger:self.assertEqual(tx.before,ledger[-1]['post'])
                for l in H:tx.append(l,self.keys[l])
                tx.finish({'entry':tx.before,'post':state_hash(self.model,H),'batch':b+1},lambda x:json.dumps(x))
        self.assertEqual(len(ledger),10)
        self.assertEqual(sum(ledger[i]['entry']==ledger[i-1]['post'] for i in range(1,10)),9)

    def test_publication_late_failure_reconciles_canonical_prefix(self):
        from .run import write_json,reconcile_ledger
        H={l:torch.zeros(24,24) for l in weights(self.model)};ledger=[];contexts=copy.deepcopy(self.contexts)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def publish(rows):
                write_json(root/'ledger.json',rows)
                raise OSError('injected after rename/directory sync')
            with self.assertRaises(OSError):
                with Transaction(self.model,H,contexts,ledger) as tx:
                    for l in H:tx.append(l,self.keys[l])
                    tx.finish({'batch':1,'commit':True},publish)
            self.assertTrue(tx.restored);self.assertEqual(ledger,[])
            self.assertEqual(len(json.loads((root/'ledger.json').read_text())),1)
            self.assertTrue(reconcile_ledger(root,ledger))
            self.assertEqual(json.loads((root/'ledger.json').read_text()),[])

class PackingTest(unittest.TestCase):
    def test_BS100_multitoken_inactive_global_sum(self):
        tok,rs,contexts,_=fixture(); records=[copy.deepcopy(rs[i%2]) for i in range(100)]
        for i,r in enumerate(records):r['case_id']=i
        s=prepare(tok,records,contexts,'cpu');self.assertEqual(len(s['row_request']),700)
        self.assertEqual(len(s['key_lookup']),600)
        self.assertEqual(s['key_context_weights'][:6],[.5,.1,.1,.1,.1,.1])
        logits=torch.randn(700,s['targets'].shape[1],128)
        teacher=torch.randn(100,128).log_softmax(-1);active=torch.arange(100)%3!=0
        full=native_loss(logits,s,teacher,active)
        parts=[native_loss(logits[i:i+2],s,teacher,active,rows=list(range(i,min(i+2,700)))) for i in range(0,700,2)]
        for k in range(3):torch.testing.assert_close(sum(p[k] for p in parts),full[k],atol=1e-3,rtol=1e-5)
    def test_canonical_ties_and_desired_tokens(self):
        rows=[dict(identity=k,kind=k,new_nll=1.,true_nll=1.,new_token_correct=1,new_token_count=2,
                   true_token_correct=2,true_token_count=3,new_strict=False,true_strict=False) for k in ('R','P','N')]
        out=reduce_rows(rows);self.assertTrue(all(v['numerator']==0 for v in out.values()))
        self.assertAlmostEqual(out['N']['desired_token_micro'],2/3)
    def test_overwrite_original_denominator_preserved(self):
        rs=[{'case_id':i,'requested_rewrite':{'subject':' A ','relation_id':'P','target_new':{'id':t,'str':t}}} for i,t in enumerate(('x','x','y'))]
        self.assertEqual(active_flags(rs),{0:False,1:False,2:True})

if __name__=='__main__':unittest.main()
