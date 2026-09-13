import unittest
from unittest.mock import patch
import torch
from project.run_scripts.baseline_mechanism_first.initial_probe import probe


class InitialProbeTests(unittest.TestCase):
    def test_same_fixed_probe_and_restore(self):
        m=torch.nn.Sequential(torch.nn.Linear(3,2,bias=False));w=m[0].weight
        entry=w.detach().clone();end=entry+.02;pointer=w.data_ptr()
        def scalar(model,tok,record):return model(torch.ones(1,3)).square().sum()
        with patch('project.run_scripts.baseline_mechanism_first.initial_probe.diagnostic_margin',scalar):
            r=probe(m,None,w,'0',entry,end,{},dict(fd_alpha=2**-8,fd_relative_tolerance=.05))
        self.assertEqual(r['diagnostic_forwards'],5);self.assertEqual(w.data_ptr(),pointer)
        self.assertTrue(torch.equal(w,entry));self.assertTrue(w.requires_grad)
        self.assertEqual(r['finite_difference']['status'],'PASS')

    def test_exception_restores(self):
        m=torch.nn.Sequential(torch.nn.Linear(3,2,bias=False));w=m[0].weight;e=w.detach().clone()
        with patch('project.run_scripts.baseline_mechanism_first.initial_probe.diagnostic_margin',side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                probe(m,None,w,'0',e,e+.1,{},dict(fd_alpha=2**-8,fd_relative_tolerance=.05))
        self.assertTrue(torch.equal(w,e));self.assertTrue(w.requires_grad)
