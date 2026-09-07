"""Real serial JVP/overlay on a tiny CPU family; no LLM/GPU/Slurm action."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .test_shared import fixture
from .gpu_fidelity import run_gpu_fidelity, FidelityBoundary, FD_EPSILONS


def toy_logits(family):
    return family.terminal_graph().detach().clone()


class GPUFidelityTests(unittest.TestCase):
    def test_two_states_real_jvp_and_parity_with_no_target_or_qref_recapture(self):
        f, dictionary, norm = fixture()
        qref = dictionary.qref
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(dictionary, 'capture_reference', side_effect=AssertionError('qref recapture')), \
             patch.object(f, 'compute_fixed_z', side_effect=AssertionError('z recompute')), \
             patch.object(f, 'run_official', side_effect=AssertionError('Official repeated')):
            result = run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits,
                                     logits_observer_identity='CPU_TOY_LOGITS')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['receipt_scope'], 'CPU_SYNTHETIC_FIXTURE')
            self.assertEqual(len(result['states']), 2)
            self.assertEqual(result['states'][0]['state_version'], 0)
            self.assertGreater(result['states'][1]['state_version'], 0)
            self.assertNotEqual(result['states'][0]['q_layers'], result['states'][1]['q_layers'])
            self.assertEqual(result['jvp_ledger']['jvp_call_count'], 10)
            self.assertEqual(result['jvp_ledger']['finite_difference_forward_count'], 60)
            self.assertEqual(result['counters']['joint_overlay_step_completed_count'], 1)
            self.assertEqual(result['counters']['temporary_materialization_completed_count'], 2)
            self.assertEqual(result['authoritative_write_count'], 0)
            self.assertEqual(result['history_append_count'], 0)
            self.assertEqual(result['qref_capture_count'], 0)
            self.assertEqual(dictionary.qref, qref)
            self.assertTrue(result['W0_bytes_restore'] and result['W0_pointer_restore'] and result['M_restore'])
            self.assertEqual(len(list(Path(folder).glob('fd-*-layer-*.json'))), 10)
            self.assertTrue((Path(folder)/'gpu_fidelity_checks.json').is_file())

    def test_fd_sign_boundary_persists_layer_and_after_restore(self):
        f, dictionary, norm = fixture()
        original = TerminalResponseObserver.observe
        def reversed_response(observer, *args, **kwargs):
            result = original(observer, *args, **kwargs)
            return replace(result, response=-result.response)
        with tempfile.TemporaryDirectory() as folder, patch.object(TerminalResponseObserver, 'observe', reversed_response):
            with self.assertRaises(FidelityBoundary) as caught:
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            self.assertEqual(caught.exception.status, 'FD_RESPONSE_IDENTITY_BOUNDARY')
            layer = json.loads((Path(folder)/'fd-entry-layer-4-boundary.json').read_text())
            after = json.loads((Path(folder)/'gpu-fidelity-boundary-after-restore.json').read_text())
            self.assertEqual([row['epsilon'] for row in layer['eps']], list(FD_EPSILONS))
            self.assertTrue(layer['adjacent_sufficient_signal'])
            self.assertFalse(layer['adjacent_pass'])
            self.assertEqual(after['stage'], 'entry/layer-4')
            self.assertTrue(after['W0_bytes_restore'] and after['M_restore'])
            self.assertEqual(after['counters']['joint_overlay_step_completed_count'], 0)

    def test_nonzero_state_fd_is_independently_checked(self):
        f, dictionary, norm = fixture()
        original = TerminalResponseObserver.observe
        def changed_only(observer, *args, **kwargs):
            result = original(observer, *args, **kwargs)
            return replace(result, response=-result.response) if observer.overlay.state_version else result
        with tempfile.TemporaryDirectory() as folder, patch.object(TerminalResponseObserver, 'observe', changed_only):
            with self.assertRaises(FidelityBoundary) as caught:
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            record = caught.exception.receipt
            self.assertEqual(record['status'], 'FD_RESPONSE_IDENTITY_BOUNDARY')
            self.assertEqual(record['stage'], 'joint-step-1/layer-4')
            self.assertEqual(record['counters']['joint_overlay_step_completed_count'], 1)
            self.assertTrue(record['W0_bytes_restore'] and record['M_restore'])

    def test_insufficient_fd_signal_is_unresolved_not_false_pass(self):
        f, dictionary, norm = fixture()
        original = TerminalResponseObserver._function
        def tiny_direction(observer, build):
            function = original(observer, build)
            return lambda coefficient: function(coefficient*1e-8)
        with tempfile.TemporaryDirectory() as folder, patch.object(TerminalResponseObserver, '_function', tiny_direction):
            with self.assertRaises(FidelityBoundary) as caught:
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            self.assertEqual(caught.exception.status, 'JVP_NUMERICALLY_UNRESOLVED')
            row = json.loads((Path(folder)/'fd-entry-layer-4.json').read_text())
            self.assertFalse(row['adjacent_sufficient_signal'])
            self.assertFalse(row['adjacent_pass'])
            self.assertTrue(caught.exception.receipt['W0_bytes_restore'])

    def test_injected_logits_exception_has_original_error_and_restore(self):
        f, dictionary, norm = fixture()
        calls = 0
        def broken(family):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('forced physical logits observation failure')
            return toy_logits(family)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FidelityBoundary) as caught:
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=broken)
            receipt = caught.exception.receipt
            self.assertEqual(receipt['original_exception']['type'], 'ValueError')
            self.assertIn('forced physical', receipt['original_exception']['message'])
            self.assertTrue(receipt['W0_bytes_restore'] and receipt['M_restore'])
            self.assertEqual(receipt['counters']['temporary_materialization_attempt_count'], 1)
            self.assertEqual(receipt['counters']['temporary_materialization_completed_count'], 0)
            self.assertEqual(tensor_set_sha256(f.parameters), f.w0_sha256)

    def test_wrong_normalization_is_persisted_binding_boundary(self):
        f, dictionary, norm = fixture()
        norm = type(norm).from_source(norm._source, 'NRMS_ENTRY')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FidelityBoundary) as caught:
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            self.assertEqual(caught.exception.status, 'FIXTURE_FIXED_Z_QREF_N0_BINDING')
            receipt = json.loads((Path(folder)/'gpu-fidelity-boundary-after-restore.json').read_text())
            self.assertEqual(receipt['stage'], 'BINDING')
            self.assertTrue(receipt['W0_bytes_restore'])
            self.assertEqual(receipt['jvp_ledger'], {})

    def test_create_once_receipts_cannot_be_overwritten(self):
        f, dictionary, norm = fixture()
        with tempfile.TemporaryDirectory() as folder:
            run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            before = (Path(folder)/'gpu_fidelity_checks.json').read_bytes()
            with self.assertRaises(FidelityBoundary):
                run_gpu_fidelity(f, dictionary, norm, folder, logits_observer=toy_logits)
            self.assertEqual(before, (Path(folder)/'gpu_fidelity_checks.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
