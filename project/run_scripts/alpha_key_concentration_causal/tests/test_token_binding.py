"""Narrow real-tokenizer CPU regression, no model/GPU/forward or CP scan."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from project.run_scripts.alpha_key_concentration_causal.technical import token_fixtures,TechnicalFailure
from project.run_scripts.alpha_key_concentration_causal.token_binding import load_reference,compare_reference

ROOT=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1')

@unittest.skipUnless((ROOT/'failure-diagnosis-r1/bos-cpu-reproduction.json').exists(),'server4 tokenizer witness required')
class NativeTokenBindingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_TEST_REQUIRES_HIDDEN_GPU'
        import transformers
        assert transformers.__version__=='4.44.2'
        old=json.loads((ROOT/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
        tok=transformers.AutoTokenizer.from_pretrained(old['snapshot'],local_files_only=True)
        tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
        contexts=json.loads(Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B050/contexts.json').read_text())
        cls.records=json.loads(Path(old['dataset']).read_text())[5000:5002]
        cwd=os.getcwd();prior=list(sys.path)
        try:
            sys.path.insert(0,old['blue_root']);os.chdir(old['blue_root'])
            representation=importlib.import_module('rome.repr_tools')
        finally:
            os.chdir(cwd);sys.path[:]=prior
        cls.rt=SimpleNamespace(tok=tok,repr=representation,contexts=contexts,writer_tokenizer_reference=load_reference(ROOT))
        cls.actual=token_fixtures(cls.rt,cls.records)

    def test_actual_original_native_ids_masks_lookup_and_BOS(self):
        self.assertEqual(self.actual['native_token_binding']['status'],'PASS')
        self.assertEqual(self.actual['actual_bos_sequences'],12)
        self.assertFalse(self.actual['writer_add_bos_token'])
        self.assertFalse(self.actual['native_token_binding']['native_tokenization_modified'])

    def test_token_corruption_is_rejected(self):
        rows=copy.deepcopy(self.actual['rows']);rows[0]['input_token_ids'][1]+=1
        with self.assertRaisesRegex(AssertionError,'INPUT_IDS_DRIFT'):
            compare_reference(self.rt.tok,rows,self.rt.contexts,self.rt.writer_tokenizer_reference)

    def test_BOS_removal_is_rejected_not_silently_waived(self):
        rows=copy.deepcopy(self.actual['rows']);rows[0]['input_token_ids']=rows[0]['input_token_ids'][1:]
        with self.assertRaisesRegex(AssertionError,'INPUT_IDS_DRIFT'):
            compare_reference(self.rt.tok,rows,self.rt.contexts,self.rt.writer_tokenizer_reference)

    def test_context_and_lookup_drift_are_rejected(self):
        contexts=copy.deepcopy(self.rt.contexts);contexts[0][0]='other {}'
        with self.assertRaisesRegex(AssertionError,'CONTEXT_DRIFT'):
            compare_reference(self.rt.tok,self.actual['rows'],contexts,self.rt.writer_tokenizer_reference)
        rows=copy.deepcopy(self.actual['rows']);rows[0]['subject_last']+=1
        with self.assertRaisesRegex(AssertionError,'POSITION_DRIFT'):
            compare_reference(self.rt.tok,rows,self.rt.contexts,self.rt.writer_tokenizer_reference)

    def test_missing_reference_does_not_bypass_check(self):
        rt=copy.copy(self.rt);del rt.writer_tokenizer_reference
        with self.assertRaisesRegex(TechnicalFailure,'NATIVE_REFERENCE_REQUIRED'):
            token_fixtures(rt,self.records)

    def test_reference_backend_and_count_must_match(self):
        ref=copy.deepcopy(self.rt.writer_tokenizer_reference);ref['postprocessor_sha256']='invalid'
        with self.assertRaisesRegex(AssertionError,'BACKEND_POSTPROCESSOR_DRIFT'):
            compare_reference(self.rt.tok,self.actual['rows'],self.rt.contexts,ref)
        with self.assertRaisesRegex(AssertionError,'SEQUENCE_COUNT'):
            compare_reference(self.rt.tok,self.actual['rows'][:-1],self.rt.contexts,self.rt.writer_tokenizer_reference)

if __name__=='__main__':unittest.main()
