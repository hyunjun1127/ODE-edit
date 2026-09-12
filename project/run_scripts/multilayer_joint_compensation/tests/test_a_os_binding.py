"""Focused CPU receipt/storage boundary checks; no model asset loading."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from project.run_scripts.multilayer_joint_compensation.contracts import member, save, digest, sha
from project.run_scripts.multilayer_joint_compensation.common_reference.session import verify_publication
from project.run_scripts.multilayer_joint_compensation.track_a.run_os import storage_probe
from project.run_scripts.multilayer_joint_compensation.observations import JointView
from project.run_scripts.multilayer_joint_compensation.tests.test_observations import Toy, rows


class Binding(unittest.TestCase):
    def test_publication_bytes_and_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);save(root/'entry.json',{'state':'immutable'})
            members=[member(root/'entry.json')]
            save(root/'terminal.json',dict(status='TERMINAL_VALID',members=members,members_root=digest(members)))
            got=verify_publication(root/'terminal.json',status='TERMINAL_VALID',expected_sha=sha(root/'terminal.json'))
            self.assertEqual(got['members'],members)
            with self.assertRaisesRegex(RuntimeError,'IDENTITY'):
                verify_publication(root/'terminal.json',status='TERMINAL_VALID',expected_sha='0'*64)
            with (root/'entry.json').open('ab') as f:f.write(b' ')
            with self.assertRaisesRegex(RuntimeError,'BYTES'):
                verify_publication(root/'terminal.json',status='TERMINAL_VALID')

    def test_storage_probe_cpu_and_no_mutation(self):
        torch.manual_seed(20260911)
        view=JointView(Toy().eval().requires_grad_(False),('first.weight','second.weight'))
        rr=[dict(r,ordinal=5000) for r in rows()]
        d=tuple(torch.randn_like(w)*.01 for w in view.entry)
        w=tuple(e+x for e,x in zip(view.entry,d))
        result=storage_probe(view,rr,SimpleNamespace(pad_token_id=0),w,d)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['primal']['byte_equal'])
        self.assertTrue(result['jvp']['byte_equal'])
        view.assert_live(bytes_check=True)

    def test_no_a0_optimizer_in_os_source(self):
        import ast
        import inspect
        from project.run_scripts.multilayer_joint_compensation.track_a import run_os
        tree=ast.parse(inspect.getsource(run_os))
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(calls & {'plan_joint_targets','compute_z','run_initial_checks'})


if __name__=='__main__':unittest.main()
