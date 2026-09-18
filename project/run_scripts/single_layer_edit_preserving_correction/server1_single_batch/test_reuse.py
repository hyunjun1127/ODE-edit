"""Focused CPU routing/identity tests, not T or actual-model validation."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ..common import write,member
from ..geometry import RightSpace
from .reuse import validation_binding,space_from,verify_endpoint,COMPLETED,MISSING

class ReuseTests(unittest.TestCase):
    def test_partition(self):
        self.assertEqual(len(set(COMPLETED+MISSING)),8)
        self.assertEqual(MISSING,('EN-F','EN-COV','EN-F4'))

    def test_episode_and_storage_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json'
            write(p,dict(status='SKIPPED_USER_DIRECTED',identity={'teacher':'original'},EN_COV_resolution=0.))
            lock=dict(instruction_id='ODEEDIT-S06-ENFC-SINGLE-BATCH-M-RESUME-SH1-V1',
                allowed_stages=['M_B001_ONLY'],native_fit_new_allowed=False,runtime_node='devbox',
                full_numerical_validation='NOT_ESTABLISHED',storage_waiver_inherited=False,
                technical_evidence=member(p),teacher_manifest={'path':'derived','sha256':'payload-bound'})
            before=p.read_bytes();v,provisional,skipped=validation_binding(lock,0)
            self.assertTrue(skipped);self.assertFalse(provisional)
            self.assertEqual(v['identity']['teacher'],lock['teacher_manifest']);self.assertEqual(before,p.read_bytes())
            for changed,episode in [({},1),({'storage_waiver_inherited':True},0),({'native_fit_new_allowed':True},0)]:
                with self.assertRaises(ValueError):validation_binding(dict(lock,**changed),episode)

    def test_geometry_factor_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'geometry';root.mkdir()
            s=RightSpace(np.eye(3),np.eye(3)[:,:1],'RESOLVED')
            torch.save(dict(basis=None,blocked=torch.from_numpy(s.blocked),status='RESOLVED'),root/'EN-F-factors.pt')
            write(root/'EN-F.json',s.receipt())
            reused=space_from(d,'EN-F',RightSpace(np.eye(3),np.empty((3,0)),'RESOLVED'))
            self.assertTrue(np.array_equal(s.project(np.ones((2,3))),reused.project(np.ones((2,3)))))
            with self.assertRaises(ValueError):space_from(d,'EN-F',RightSpace(-np.eye(3),np.empty((3,0)),'RESOLVED'))

    def test_bad_endpoint_cannot_become_complete(self):
        with patch('project.run_scripts.single_layer_edit_preserving_correction.server1_single_batch.reuse.read',return_value={}),patch('torch.load',return_value={'weight':torch.zeros((2,3))}):
            with self.assertRaisesRegex(ValueError,'SHAPE'):verify_endpoint('/absent','EN-F',ids=[],W0='',WN='',context_tokens='')

    def test_runner_frozen_controller_and_observer_barrier(self):
        text=Path(__file__).with_name('runner.py').read_text();tree=ast.parse(text)
        self.assertIn('native=rt.native(records,root/\'native\',reuse=True)',text)
        self.assertNotIn('sequential_',text)
        self.assertLess(text.index('ALL_SELECTIONS_SEALED.json'),text.index('obs=CanonicalObserver'))
        self.assertLess(text.index('if arm in COMPLETED:'),text.index('result=optimize('))
        optimize_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='optimize']
        self.assertEqual(len(optimize_calls),1)

if __name__=='__main__':unittest.main()
