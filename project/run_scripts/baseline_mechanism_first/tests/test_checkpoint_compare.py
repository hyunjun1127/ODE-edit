import json
from pathlib import Path
import tempfile
import unittest
import torch

from project.run_scripts.baseline_mechanism_first.checkpoint_compare import compare
from project.run_scripts.baseline_mechanism_first.contracts import member


class CheckpointCompareTests(unittest.TestCase):
    def test_exact_endpoint_does_not_certify_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cp=dict(weights={'w':torch.ones(2,3)},cache_c=torch.eye(3)[None],
                    metadata=dict(batch=20,seen_ids=[1,2],base_model_revision='x',contexts=[],rng={},covariance={}))
            torch.save(cp,root/'actual.pt');torch.save(cp,root/'reference.pt')
            compare(root/'actual.pt',root/'reference.pt',member(root/'reference.pt')['sha256'],root/'receipt.json')
            r=json.loads((root/'receipt.json').read_text())
            self.assertTrue(r['endpoint_state_exact'])
            self.assertFalse(r['original_trajectory_equivalence'])

    def test_nonexact_is_recorded_not_hardware_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=dict(weights={'w':torch.ones(2,3)},cache_c=torch.eye(3)[None],metadata={})
            torch.save(cp,root/'reference.pt');cp['weights']['w'][0,0]+=1;torch.save(cp,root/'actual.pt')
            compare(root/'actual.pt',root/'reference.pt',member(root/'reference.pt')['sha256'],root/'receipt.json')
            r=json.loads((root/'receipt.json').read_text())
            self.assertEqual(r['status'],'NONEXACT_CAUSE_UNRESOLVED')
            self.assertEqual(r['tensors'][0]['difference_norm'],1)
            self.assertFalse(r['hardware_cause_attribution'])
