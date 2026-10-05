"""Narrow new-route regressions; synthetic CPU model is not actual-Llama PASS."""
import json,tempfile,unittest
from pathlib import Path
import torch
from .optimizer import project,EfficiencyAdam
from .routes import annotate,cached_vjp
from .test_cpu import fixture
from .builder import build,reverse
from .subject import evaluate
from .qualification_minimal import check
from .optimize import fit,stop
from .telemetry import commit_measure
from .collect import validate_fit,harmonic
from .common import state
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_realized_subject.geometry import ridge
from project.run_scripts.jlz_realized_subject.qualification import fixed,compare

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_projection_KKT_cast_both_constraints_and_dimensions(self):
        for norms in ([0,0,0],[.1,.2,.3],[2,0,0],[.6,.6,.6],[3,2,1,0,0],[1.],[.75,.75],[.9,.4,.3,.2]):
            blocks=[torch.ones(i+1,dtype=torch.float32)*(n/(i+1)**.5) for i,n in enumerate(norms)]
            for radius in (.75,1.5):
                values,r=project(blocks,radius)
                expected=[min(.75,max(n-r['tau'],0.)) for n in r['proposal_norm']]
                self.assertLess(max(abs(a-b) for a,b in zip(expected,r['projected64_norm'])),1e-10)
                self.assertLessEqual(r['stored_budget'],radius+1e-6);self.assertLessEqual(max(r['stored_norm']),.750001)
                if r['tau']>0:self.assertAlmostEqual(sum(expected),radius,places=12)
        # Radial scaling is not this Euclidean capped projection.
        _,r=project([torch.tensor([3.]),torch.tensor([1.]),torch.tensor([.2])])
        self.assertAlmostEqual(r['projected64_norm'][0],.75)
    def test_optimizer_moments_and_zero_reentry(self):
        u=[torch.zeros(3),torch.zeros(5)];opt=EfficiencyAdam(u,torch.tensor(2.))
        u,_=opt.step(u,[torch.ones(3),torch.zeros(5)]);self.assertEqual(float(u[1].norm()),0)
        u,r=opt.step(u,[torch.zeros(3),torch.ones(5)]);self.assertGreater(float(u[1].norm()),0)
        self.assertEqual(opt.t,2);self.assertGreater(float(opt.m[0].norm()),0)
    def test_nonsymmetric_cached_VJP(self):
        torch.manual_seed(7)
        A=torch.randn(7,7,dtype=torch.float64)+torch.eye(7,dtype=torch.float64)*9
        LU,piv=torch.linalg.lu_factor(A);prior=dict(A=A,LU=LU,pivots=piv,L=None)
        for B in (1,3,9):
            K=torch.randn(7,B,dtype=torch.float64);K[:,0]=0;GP=torch.randn_like(K)
            leaf=K.clone().requires_grad_(True);P=torch.linalg.solve(A+leaf@leaf.T,leaf)
            ref=torch.autograd.grad(P,leaf,GP)[0]
            g,r=cached_vjp(K,P.detach(),GP,prior)
            torch.testing.assert_close(g,ref,atol=1e-11,rtol=1e-10)
            self.assertLess(r['relative_residual'],1e-8)
    def test_fullgradient_zero_firstsite_R_and_complete_owner(self):
        with tempfile.TemporaryDirectory() as t:
            a,e,H=fixture(Path(t),2);annotate(e);R={l:r.detach() for l,r in fixed(a,e).items()};R[a.first].zero_()
            b=build(a,e,R,0);obs=evaluate(a,e,b,True)
            ref,_=reverse(a,e,R,b,obs['adjoint'],cached=False,prune_first=False)
            got,ledger=reverse(a,e,R,b,obs['adjoint']);self.assertTrue(all(r['passed'] for r in compare(got,ref).values()))
            self.assertGreater(float(got[a.first].norm()),0);self.assertEqual(ledger['layers'][-1]['adjoint_status'],'NOT_NEEDED')
            partial=evaluate(a,e,b,group_indices={0});self.assertFalse(partial['complete_owner'][0])
            allgroups={i for i,g in enumerate(e['groups']) if any(r['request']==0 for r in g['rows'])}
            complete=evaluate(a,e,b,group_indices=allgroups);self.assertTrue(complete['complete_owner'][0]);self.assertEqual(float(complete['F'][0]),float(obs['F'][0]))
            self.assertFalse(stop(torch.tensor([.05,.001]),0));self.assertTrue(stop(torch.tensor([.001,.002]),0));self.assertTrue(stop(torch.tensor([10.,10.]),24))
            b['P'][a.first].add_(1.)
            with self.assertRaisesRegex(Exception,'CACHE_MUTATION'):reverse(a,e,R,b,obs['adjoint'])
    def test_fixed_candidate_and_two_batch_transaction(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,e,H=fixture(root,2);cold=state(a,H);q=check(a,e,root/'qualification')
            self.assertEqual(state(a,H),cold);self.assertTrue(q['routes']['R2'])
            tx=Transaction(a,H)
            with tx:
                plan=fit(a,e,root/'fit1',q['routes']);wr=commit_measure(a,e,H,plan,root/'writer1');tx.finish()
            self.assertNotEqual(state(a,H),cold);validate_fit(root/'fit1');first=state(a,H)
            # New own-entry teachers/keys/factors from committed W/H, never W0 reset.
            from project.run_scripts.jlz_realized_subject.entry import prepare_entry
            from types import SimpleNamespace
            stats={str(l):str(root/f'stats{l}.npz') for l in a.sites}
            nextentry=annotate(prepare_entry(a,SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),e['pack'],H,stats,2))
            self.assertIsNot(nextentry['factors'],e['factors'])
            before=rng_snapshot();tx=Transaction(a,H)
            with tx:
                plan=fit(a,nextentry,root/'fit2',q['routes']);commit_measure(a,nextentry,H,plan,root/'writer2')
                # Simulated fault after native H append: no finish restores own B2 entry.
            self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,H),first);self.assertTrue(rng_equal(before))
            self.assertEqual(wr['history_appends'],len(a.sites));self.assertFalse(list(root.rglob('*.pt')))
    def test_harmonic_and_horizon_source(self):
        self.assertAlmostEqual(harmonic({k:{'rate':.5} for k in ('R','P','N')}),.5)
        s=(Path(__file__).parent/'run.py').read_text();self.assertIn('range(1,21)',s);self.assertIn('OWN_CHAIN_ENTRY_JOIN',s)
        s=(Path(__file__).parent/'collect.py').read_text();self.assertLess(s.index("(out/'report-ko.md').write_text"),s.index("write(out/'terminal.json'"))

if __name__=='__main__':unittest.main()
