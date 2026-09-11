from types import SimpleNamespace
import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.tests.test_observations import Toy,rows
from project.run_scripts.multilayer_joint_compensation.observations import JointView,teacher
from project.run_scripts.multilayer_joint_compensation.contracts import Ledger
from project.run_scripts.multilayer_joint_compensation.track_b.runtime import SelectedView
from project.run_scripts.multilayer_joint_compensation.track_b.runtime_gate import initial_gate


class TestBRuntimeGate(unittest.TestCase):
    def test_full_callable_materialization_and_gate_nonmutation(self):
        torch.manual_seed(36);model=Toy().eval().requires_grad_(False);ledger=Ledger()
        full=JointView(model,('first.weight','second.weight'),ledger)
        wn=(full.entry[0]+.02,full.entry[1]);view=SelectedView(full,wn,(8,))
        rr=rows()
        for i,r in enumerate(rr):r.update(ordinal=1000+(i==2),case_id=71+(i==2))
        saved=teacher(view,view.entry,rr,0,2)
        problem=SimpleNamespace(wn=tuple(x.cpu() for x in view.entry),project=lambda x:x)
        result=initial_gate(problem,view,{'Current':rr},saved,SimpleNamespace(pad_token_id=0),model,ledger,None)
        self.assertEqual(result['status'],'ACTUAL_FIRST_REQUEST_KERNEL_AND_APPLICATION_VALID')
        self.assertEqual(result['physical_equality'],'BYTE_EXACT')
        self.assertTrue(all(r['passed'] for r in result['FD']))
        self.assertEqual(result['first_ordinal'],1000)
        full.assert_live(bytes_check=True)


if __name__=='__main__':unittest.main()
