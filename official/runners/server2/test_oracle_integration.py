"""CPU-only mocked connector/order tests; no pretrained/GPU evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from official.evaluation import factual
from official.experiments.prepare import read, write_new
from official.runners.server2 import oracle, run
from official.tests.test_cf_native_reference import FixtureLM, NativeCharacterTokenizer, record


class OracleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.model, self.tok = FixtureLM().eval(), NativeCharacterTokenizer()
        self.records = [record(i) for i in range(1, 5)]
        self.manifest = dict(model='gptj', owner=dict(server='server2'),
            model_revision=oracle.MODEL_REVISION, tokenizer_sha256='a'*64,
            streams=dict(cf=dict(lock=dict(stream_sha256='b'*64))),
            runtime=dict(profile='CPU_CONNECTOR_FIXTURE_NO_PRETRAINED_OR_GPU'),
            code_commit='c'*40, official_tree_sha256='d'*64,
            model_snapshot='CPU_FIXTURE_NOT_MODEL_ASSET', assets_identity_sha256='e'*64)
        self.manifest['cf_native_reference_plan'] = oracle.plan(self.manifest, self.records)

    def canonical_fixture(self, model, tok, manifest, records, dataset, out, endpoint):
        self.assertFalse(model.config.use_cache)
        self.assertTrue((out/'native-oracle/plan.json').is_file())
        self.assertTrue((out/'native-oracle/state-before-canonical.json').is_file())
        observed = factual.evaluate_counterfact(model, tok, records,
            batch_size=3, identity=run.factual_identity(manifest, dataset))
        value = dict(observed, endpoint=endpoint, dataset=dataset, requests=len(records))
        path = out/'factual'/(endpoint+'.json')
        write_new(path, value)
        return value, run.member(path)

    def test_CPU_not_qualified_preserved_before_gate_and_call_config_restored(self):
        self.model.config.use_cache = True
        before = oracle.capture_state(self.model, self.tok)
        with patch.object(run, 'factual', side_effect=self.canonical_fixture):
            with self.assertRaisesRegex(ValueError, 'ACTUAL_INDEPENDENT_ORIGINAL_CF_SMOKE_NOT_PASSED'):
                run.cf_native_oracle(self.model, self.tok, self.manifest, self.records, self.out)
        self.assertEqual(oracle.capture_state(self.model, self.tok), before)
        stored = read(self.out/'native-oracle/comparison.json')
        self.assertEqual(stored['status'], 'NOT_QUALIFIED')
        self.assertFalse(stored['actual_GPU'])
        self.assertEqual(stored['work']['forward_calls'], 0)
        self.assertFalse((self.out/'READY.json').exists())

    def test_plan_missing_or_changed_blocks_before_observation(self):
        for manifest in ({}, dict(self.manifest, cf_native_reference_plan={} )):
            with patch.object(run, 'factual') as observed:
                with self.assertRaisesRegex(ValueError, 'PREMEASUREMENT_CF_NATIVE_ORACLE_PLAN_REQUIRED'):
                    run.cf_native_oracle(self.model, self.tok, manifest, self.records, self.out)
                observed.assert_not_called()

    def test_original_exception_not_replaced_and_native_config_restored(self):
        self.model.config.use_cache = True
        before = oracle.capture_state(self.model, self.tok)
        with patch.object(run, 'factual', side_effect=self.canonical_fixture), \
             patch.object(run.oracle, 'compare', side_effect=RuntimeError('original scorer failure')):
            with self.assertRaisesRegex(RuntimeError, 'original scorer failure'):
                run.cf_native_oracle(self.model, self.tok, self.manifest, self.records, self.out)
        self.assertEqual(oracle.capture_state(self.model, self.tok), before)
        self.assertFalse((self.out/'READY.json').exists())

    def test_cf_full_W0_or_generation_never_runs_after_oracle_rejection(self):
        with patch.object(run, 'cf_native_oracle', side_effect=ValueError('oracle blocked')), \
             patch.object(run, 'factual') as observed, patch.object(run.generation, 'observe') as generated:
            with self.assertRaisesRegex(ValueError, 'oracle blocked'):
                run.cold_w0(self.model, self.tok, self.manifest, self.records, 'cf', self.out, None)
            observed.assert_not_called()
            generated.assert_not_called()

    def test_stored_CPU_proof_cannot_issue_actual_W0_READY_binding(self):
        with patch.object(run, 'factual', side_effect=self.canonical_fixture):
            with self.assertRaises(ValueError):
                run.cf_native_oracle(self.model, self.tok, self.manifest, self.records, self.out)
        proof = read(self.out/'native-oracle/comparison.json')
        binding = dict(plan_sha256=self.manifest['cf_native_reference_plan']['plan_sha256'],
            proof=run.member(self.out/'native-oracle/comparison.json'),
            canonical=proof['binding']['canonical_member'],
            state=run.member(self.out/'native-oracle/state-before-canonical.json'),
            scope=proof['evidence_scope'], full_2k_parity='NOT_APPLICABLE_GPTJ; NOT_OBSERVED')
        with self.assertRaisesRegex(ValueError, 'W0_ORIGINAL_CF_ACTUAL_SOURCE_STATE_SCOPE_PROOF'):
            run.verify_cf_native_oracle(self.manifest, binding)


if __name__ == '__main__':
    unittest.main()
