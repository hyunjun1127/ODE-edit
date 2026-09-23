import unittest,tempfile,json
from pathlib import Path
from project.run_scripts.alpha_key_concentration_causal.reduce import independent_pairs
from project.run_scripts.alpha_key_concentration_causal.reduce import reduce_package
from project.run_scripts.alpha_key_concentration_causal.common import save,file_sha

def row(nll):return dict(case_id=1,prompt_index=0,prompt='x',target='y',nll=nll)
class IndependentReducer(unittest.TestCase):
    def test_repair_attempt_reducer_keeps_old_report_and_uses_new_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'root';repo=Path(d)/'repo';out=root/'execution/attempt-r2';out.mkdir(parents=True)
            base=repo/'experiment-reports/servers/server4/alpha-key-causal-20260923-r1'
            old=base/'generated-r1';old.mkdir(parents=True);save(old/'sentinel.json',{'old':'KEEP'})
            lock_path=root/'controls/attempt-r2/execution.lock.json'
            lock={'attempt':'attempt-r2','report_subdir':'generated-r2','execution_lock_path':str(lock_path),'source_commit':'source2','source_tree':'tree2'}
            save(lock_path,lock)
            reduce_package(root,out,repo,lock)
            receipt=json.loads((base/'generated-r2/source-state-receipt.json').read_text())
            self.assertEqual(receipt['execution_lock_sha256'],file_sha(lock_path))
            self.assertEqual(receipt['status'],'TECHNICAL_FAILED_OR_PARTIAL')
            self.assertEqual(json.loads((old/'sentinel.json').read_text()),{'old':'KEEP'})

    def test_repair_report_cannot_target_old_package(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(AssertionError,'REDUCER_ATTEMPT_COLLISION'):
                reduce_package(d,d,d,{'attempt':'attempt-r2','report_subdir':'generated-r1'})

    def test_rewrite_tie_failure(self):
        self.assertEqual(independent_pairs({'rewrite_target_new':[row(1.)],'rewrite_target_true':[row(1.)]})['RS']['numerator'],0)
    def test_neighborhood_reverse(self):
        self.assertEqual(independent_pairs({'locality_target_new':[row(2.)],'locality_target_true':[row(1.)]})['NS']['numerator'],1)
    def test_nonfinite_is_error(self):
        with self.assertRaises(AssertionError):independent_pairs({'rewrite_target_new':[row(float('nan'))],'rewrite_target_true':[row(1.)]})
    def test_denominator_not_filled(self):
        with self.assertRaises(AssertionError):independent_pairs({'rewrite_target_new':[row(1.)],'rewrite_target_true':[]})
