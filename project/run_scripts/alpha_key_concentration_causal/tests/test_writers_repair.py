"""CPU-only user-policy, reuse corruption, and scope regression fixtures."""
import copy
import json
import inspect
from pathlib import Path
import tempfile
import unittest
import torch

from project.run_scripts.alpha_key_concentration_causal.numerical_policy import OBSERVE,STRICT,annotate
from project.run_scripts.alpha_key_concentration_causal import writer_runner,writer_reuse,writers_repair,interventions
from project.run_scripts.alpha_key_concentration_causal.tests.test_integration_cpu import sham_fixture
from project.run_scripts.alpha_key_concentration_causal.common import file_sha


class RepairTests(unittest.TestCase):
    def test_observation_retains_failed_verdict(self):
        r=annotate({'status':'FAIL','relative_error':1e-5},OBSERVE)
        self.assertEqual(r['status'],OBSERVE);self.assertEqual(r['comparison_verdict'],'FAIL')
        self.assertFalse(r['blocks_execution']);self.assertEqual(r['relative_error'],1e-5)

    def test_default_unchanged_blocking(self):
        self.assertTrue(annotate({'status':'FAIL'})['blocks_execution'])
        self.assertEqual(annotate({'status':'PASS'},OBSERVE)['status'],'PASS')

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):annotate({'status':'FAIL'},'ignore_everything')

    def test_sham_relative_difference_nonblocking_only_by_explicit_policy(self):
        native,observed=sham_fixture();sham=copy.deepcopy(native)
        sham['factors'][4]['delta'][0,0]+=0.001
        with tempfile.TemporaryDirectory() as d:
            r=writer_runner.verify_sham(native,sham,observed,observed,Path(d)/'observation.json',policy=OBSERVE)
            self.assertEqual(r['status'],OBSERVE)
            self.assertEqual(r['comparison_verdict'],'NUMERICAL_CONTROL_NOT_ESTABLISHED')
            self.assertGreater(r['differences'][2]['max_abs'],0)
            with self.assertRaises(RuntimeError):
                writer_runner.verify_sham(native,sham,observed,observed,Path(d)/'strict.json')

    def test_nonfinite_still_blocks(self):
        native,observed=sham_fixture();sham=copy.deepcopy(native)
        sham['factors'][4]['delta'][0,0]=float('nan')
        with tempfile.TemporaryDirectory() as d,self.assertRaises(AssertionError):
            writer_runner.verify_sham(native,sham,observed,observed,Path(d)/'out.json',policy=OBSERVE)

    def test_identity_still_blocks(self):
        native,observed=sham_fixture();other=copy.deepcopy(observed)
        other['current']['metrics']['RS']['rows'][0]['identity']='wrong'
        with tempfile.TemporaryDirectory() as d,self.assertRaises(AssertionError):
            writer_runner.verify_sham(native,native,observed,other,Path(d)/'out.json',policy=OBSERVE)

    def test_nonfinite_scalar_nll_still_blocks(self):
        native,observed=sham_fixture();other=copy.deepcopy(observed)
        other['current']['metrics']['PS']['rows'].append(dict(identity='second',new_nll=float('nan'),true_nll=2.))
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(AssertionError,'NONFINITE_NLL'):
            writer_runner.verify_sham(native,native,observed,other,Path(d)/'out.json',policy=OBSERVE)

    def test_shape_still_blocks(self):
        native,observed=sham_fixture();sham=copy.deepcopy(native)
        sham['factors'][4]['K']=torch.zeros(3,4)
        with tempfile.TemporaryDirectory() as d,self.assertRaises(AssertionError):
            writer_runner.verify_sham(native,sham,observed,observed,Path(d)/'out.json',policy=OBSERVE)

    def test_source_formula_unchanged(self):
        self.assertIn('refresh = M_native - stamp_gram + replacement',inspect.getsource(interventions.history_operand))

    def test_reuse_exact_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';p.write_bytes(b'unchanged input')
            manifest={'members':[dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))]}
            writer_reuse.verify_members(manifest)
            p.write_bytes(b'corrupted input')
            with self.assertRaises(AssertionError):writer_reuse.verify_members(manifest)

    def test_reuse_view_does_not_copy_or_overwrite_old_control(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src';dst=Path(d)/'dst';src.mkdir();dst.mkdir()
            (src/'sham-control.json').write_text('{"status":"FAILED"}')
            writer_reuse.link_branch(src,dst)
            self.assertTrue((dst/'historical-sham-control.json').is_symlink())
            self.assertFalse((dst/'sham-control.json').exists())
            self.assertEqual((src/'sham-control.json').read_text(),'{"status":"FAILED"}')

    def test_two_phases_and_no_sibling_mutations(self):
        source=inspect.getsource(writers_repair.submit)
        self.assertIn("for phase in ('writers','reduce')",source)
        self.assertNotIn('scancel',source)
        self.assertNotIn("['scontrol','hold'",source)
        self.assertIn('afterany:52564:',source)

    def test_no_fresh_native_for_completed_W50(self):
        source=inspect.getsource(writer_runner.run_writers)
        self.assertIn("entry==50 and branch in ('NATIVE','SHAM')",source)
        self.assertIn('restore_completed_branch',source)
        self.assertIn('new_native_targets',source)


if __name__=='__main__':unittest.main()
