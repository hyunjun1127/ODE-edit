"""Real CPU receipt round trips and fail-closed B1 routing; no model evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .basis import build_functional_basis, append_covariance_direction
from .solver import solve_coefficients
from .program import B1_ONLY_AUTHORITY, require_program
from .science import batch
from .technical_decision import classify_fd
from .test_technical_decision import points
from .test_program_integration import ProgramHarness, lock_fixture
from project.run_scripts.single_layer_edit_preserving_correction.common import write


def b1_lock():
    return dict(lock_fixture(),maximum_batch=1,b1_only_authority=B1_ONLY_AUTHORITY,
        sequential_authorized=False,auto_continue=False,agent_monitoring_after_release=True)


class Tests(unittest.TestCase):
    def roundtrip(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            write(path,value)
            self.assertEqual(json.loads(path.read_text()),json.loads(json.dumps(value,allow_nan=False)))
            with self.assertRaises(FileExistsError):write(path,value)

    def test_actual_functional_and_covariance_receipts_roundtrip(self):
        rng=np.random.default_rng(20260919)
        for h in (np.zeros((5,7)),np.ones((5,7)),rng.normal(size=(5,7))):
            basis=build_functional_basis(h)
            self.roundtrip(dict(basis=basis.receipt,weight_snapshot_saved=False))
            for row in basis.receipt['components']:self.assertIs(type(row['added']),bool)
            cov=append_covariance_direction(basis,rng.normal(size=(5,7)),
                [rng.normal(size=(7,3))],expected_documents=1)
            self.roundtrip(cov.receipt)
            self.assertIs(type(cov.receipt['covariance']['added']),bool)

    def test_actual_solver_receipts_roundtrip(self):
        for mu,j,r in [([-1.],[[1.]],2.),([-2.,-1.],np.eye(2),1.),([-1.,.1],[[1.],[-1.]],2.)]:
            self.roundtrip(solve_coefficients(mu,j,r).receipt)

    def test_plain_json_rejects_original_numpy_bool_regression(self):
        with self.assertRaises(TypeError):json.dumps({'added':np.bool_(True)})

    def test_actual_numpy_noise_resolved_and_unresolved_FD_receipts(self):
        for derivative in (0.,2.):
            value=classify_fd(points(AD=derivative),AD=derivative,
                noise=np.float64(1e-8),direction_norm=1.)
            self.roundtrip(value)
            self.assertTrue(all(type(row['resolved']) is bool for row in value['full_grid']))

    def test_pass_and_fail_both_stop_at_B1(self):
        for passed in (True,False):
            h=ProgramHarness(b1=passed);h.rt.lock=b1_lock();h.run()
            self.assertEqual(len(h.batch_calls),1)
            self.assertEqual(h.batch_calls[0]['batch'],1)
            terminal=h.evidence.ending('terminal.json')[0]
            self.assertEqual(terminal['status'],'B1_COMPLETE_USER_LIMIT')
            self.assertEqual(terminal['maximum_batch'],1)
            self.assertFalse(terminal['sequential_authorized'])
            self.assertEqual(terminal['last_gate']['pass'],passed)
            self.assertFalse(h.evidence.ending('S3-to-S10.json'))
            self.assertFalse(h.evidence.ending('STEP-CUM-alias.json'))
            self.assertEqual(h.resource_calls,[('T0_B1',24*2**30)])

    def test_missing_or_expanded_authority_is_blocked(self):
        for key in ('b1_only_authority','sequential_authorized','auto_continue','agent_monitoring_after_release'):
            lock=b1_lock();lock.pop(key)
            with self.assertRaises(ValueError):require_program(lock)
        lock=b1_lock();lock['maximum_batch']=10
        with self.assertRaises(ValueError):require_program(lock)

    def test_science_B2_blocked_before_any_io(self):
        h=ProgramHarness();h.rt.lock=b1_lock()
        with patch.object(Path,'mkdir',side_effect=AssertionError('IO_BEFORE_AUTHORITY')):
            with self.assertRaisesRegex(ValueError,'USER_B1_ONLY'):
                batch(h.rt,object(),[],stage='S3',arm_names=('N4',),batch_number=2,
                      ledger=[],directory='/NOT_ALLOWED',gate={'pass':True})

    def test_T0_failure_still_blocks_B1(self):
        h=ProgramHarness(technical=False);h.rt.lock=b1_lock()
        with self.assertRaisesRegex(RuntimeError,'T0_FULL_CHECKS_NOT_ESTABLISHED'):h.run()
        self.assertEqual(h.batch_calls,[])


if __name__=='__main__':unittest.main()
