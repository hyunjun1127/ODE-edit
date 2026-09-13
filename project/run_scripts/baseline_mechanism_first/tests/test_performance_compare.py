import unittest
from project.run_scripts.baseline_mechanism_first.performance_compare import compare_rows
from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary

class PerformanceTests(unittest.TestCase):
    def row(self,new,true,success):
        return dict(case_id=1,prompt_index=0,identity='paired',new_nll=new,true_nll=true,success=success)
    def test_sign_tie_and_transition(self):
        r=compare_rows([self.row(1,2,True)],[self.row(2,2,False)],'RS')
        self.assertEqual((r['delta_pp'],r['loss'],r['recovery']),(-100,1,0))
        r=compare_rows([self.row(1,2,False)],[self.row(3,2,True)],'NS')
        self.assertEqual((r['delta_pp'],r['recovery']),(100,1))
    def test_identity_mismatch(self):
        a=self.row(1,2,True);b=dict(a,identity='different')
        with self.assertRaises(ContractBoundary):compare_rows([a],[b],'RS')
