import unittest
from unittest.mock import patch
import torch
from .capture import capture_entry
from .test_shared import fixture
from .fixture import RNGSnapshot
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256


class CaptureTests(unittest.TestCase):
    def test_raw_capture_is_observation_only_and_qref_frozen(self):
        with patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
            f,d,n=fixture();before=RNGSnapshot();qref=d.qref
            raw,receipt=capture_entry(f,d)
            self.assertTrue(before.matches())
            self.assertEqual(qref,d.qref)
            self.assertEqual(receipt['reference']['capture_count'],0)
            self.assertEqual(receipt['jvp_ledger']['jvp_call_count'],5)
            self.assertEqual(raw['raw_responses32'].shape,(5,2,2))
            self.assertEqual(raw['raw_responses32'].dtype,torch.float32)
            self.assertEqual(receipt['repeatability']['initialization_capture_index'],0)
            self.assertEqual(receipt['actual_write_count'],0)
            self.assertEqual(tensor_set_sha256(f.parameters),f.w0_sha256)

    def test_new_reference_captured_once(self):
        with patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
            f,d,n=fixture();d.qref=None;d.qfref=None
            _,receipt=capture_entry(f,d)
            self.assertEqual(receipt['reference']['capture_count'],1)
            _,second=capture_entry(f,d)
            self.assertEqual(second['reference']['capture_count'],0)


if __name__=='__main__':unittest.main()
