import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.tests.test_observations import Toy,rows
from project.run_scripts.multilayer_joint_compensation.observations import JointView,teacher
from project.run_scripts.multilayer_joint_compensation.evaluation import materialized,panel
from project.run_scripts.multilayer_joint_compensation.contracts import Ledger

class Binding(unittest.TestCase):
    def test_success_and_exception_exact_storage_restore(self):
        torch.manual_seed(17);m=Toy().eval().requires_grad_(False);ledger=Ledger()
        v=JointView(m,('first.weight','second.weight'));ww=tuple(x+.1 for x in v.entry)
        for fail in (False,True):
            try:
                with materialized(m,v.names,ww,ledger):
                    self.assertTrue(torch.equal(m.first.weight,ww[0]))
                    if fail:raise RuntimeError('forced')
            except RuntimeError:
                self.assertTrue(fail)
            v.assert_live(bytes_check=True)
        self.assertEqual(ledger.counts['observation_restores'],2)

    def test_shared_panel_uneven_chunks_and_cross_action(self):
        torch.manual_seed(7);m=Toy().eval().requires_grad_(False)
        v=JointView(m,('first.weight','second.weight'));saved=teacher(v,v.entry,rows(),0,2)
        w=tuple(x+.05 for x in v.entry);direction=(torch.zeros_like(w[0]),torch.randn_like(w[1]))
        reference=None
        for mb in (1,2,4):
            p=panel(v,rows(),saved,0,'current',mb)
            q=p.linearize(w,need_nll_gradient=True);gg=p.ggn(w,direction)
            if reference is None:reference=q,gg
            else:
                self.assertAlmostEqual(q.value,reference[0].value,places=6)
                for a,b in zip(gg,reference[1]):self.assertTrue(torch.allclose(a,b,atol=5e-5,rtol=5e-5))
            self.assertGreater(float(gg[0].norm()),0.)
        v.assert_live(bytes_check=True)

if __name__=='__main__':unittest.main()
