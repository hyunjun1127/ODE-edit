import unittest
import torch
from project.run_scripts.baseline_mechanism_first.terminal_performance import observe_guarded,paired
from project.run_scripts.baseline_mechanism_first.fixtures import tensor_sha
from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary

class TerminalPerformanceTests(unittest.TestCase):
    def test_guard_observation_and_wrong_entry(self):
        m=torch.nn.Linear(2,2);h=torch.eye(2);w=tensor_sha(m.weight);hs=tensor_sha(h)
        value,r=observe_guarded(m,m.weight,h,w,hs,lambda:3)
        self.assertEqual(value,3);self.assertTrue(r['RNG_exact'])
        with self.assertRaises(ContractBoundary):observe_guarded(m,m.weight,h,'wrong',hs,lambda:3)
    def test_mutation_rejected(self):
        m=torch.nn.Linear(2,2);h=torch.eye(2)
        def bad():
            with torch.no_grad():m.weight.add_(1)
        with self.assertRaises(ContractBoundary):observe_guarded(m,m.weight,h,tensor_sha(m.weight),tensor_sha(h),bad)
    def test_full_seen_denominator_not_current(self):
        a=dict(requests=6000,request_order='same',metrics={})
        b=dict(a,requests=100)
        with self.assertRaises(ContractBoundary):paired(a,b,'seen-full')
