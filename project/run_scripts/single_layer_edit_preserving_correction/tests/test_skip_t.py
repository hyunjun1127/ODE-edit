"""Narrow CPU admission routing tests. No model, numerical replay or scheduler."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from project.run_scripts.single_layer_edit_preserving_correction.common import write
from project.run_scripts.single_layer_edit_preserving_correction.validation_route import (
    validation_binding, check_waiver, SKIP_NONCE, SKIPPED)

AUTHORITY=dict(nonce=SKIP_NONCE,gpu_cap=2,technical_phase=SKIPPED,
    full_numerical_validation='NOT_ESTABLISHED',new_T_submit=False,M_requires_T_READY=False,
    new_M_old_T_failcancel=False,M_independent_episodes=10,M_final_L4_endpoints=80,
    M_native_fit_B1='REUSE',M_new_native_fits_max=9,reuse_first=True,
    method_acceptance_guards_unchanged=True,S_R_L_submit=False)


class SkipTRouting(unittest.TestCase):
    def make(self,root):
        waiver=write(root/'waiver.json',AUTHORITY)
        evidence=write(root/'binding.json',dict(status=SKIPPED,identity={},EN_COV_resolution=0))
        return dict(stage='M',allowed_stages=['M'],skip_T_override=waiver,
            technical_evidence=evidence,T_job=None,T_failure_path=None,parallel_override=None,
            old_T_failcancel=False,full_numerical_validation='NOT_ESTABLISHED')

    def test_all10_no_T_failure_or_READY_stat(self):
        with tempfile.TemporaryDirectory() as d:
            lock=self.make(Path(d))
            with patch.object(Path,'exists',side_effect=AssertionError('No T marker read permitted')):
                for i in range(10):
                    evidence,provisional,skipped=validation_binding(lock,i)
                    self.assertEqual(evidence['status'],SKIPPED)
                    self.assertFalse(provisional);self.assertTrue(skipped)

    def test_no_scope_or_validation_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            lock=self.make(Path(d))
            for k,v in [('stage','T'),('allowed_stages',['T','M']),('T_job',49928),
                        ('T_failure_path','/old/failure.json'),('old_T_failcancel',True),
                        ('full_numerical_validation','PASS')]:
                with self.assertRaises(ValueError):validation_binding(dict(lock,**{k:v}),0)
            for i in (-1,10):
                with self.assertRaises(ValueError):validation_binding(lock,i)

    def test_waiver_exact_method_cap_and_storage(self):
        check_waiver(AUTHORITY)
        for k,v in [('nonce','old'),('gpu_cap',3),('method_acceptance_guards_unchanged',False),
                    ('M_final_L4_endpoints',0),('S_R_L_submit',True),('new_T_submit',True)]:
            with self.assertRaises(ValueError):check_waiver(dict(AUTHORITY,**{k:v}))

    def test_mutated_binding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            lock=self.make(Path(d));p=Path(lock['technical_evidence']['path'])
            p.write_text(json.dumps(dict(status='T_READY')))
            with self.assertRaises(ValueError):validation_binding(lock,0)


if __name__=='__main__':unittest.main()
