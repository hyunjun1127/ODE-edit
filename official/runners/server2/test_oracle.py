"""CPU fixture/mock only; NEVER actual GPT-J pretrained/GPU certification."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.evaluation import cf_native_reference as reference
from official.evaluation import factual
from official.experiments.prepare import digest, write_new
from official.runners.server2 import oracle
from official.tests.test_cf_native_reference import FixtureLM, NativeCharacterTokenizer, record


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model, self.tok = FixtureLM().eval(), NativeCharacterTokenizer()
        self.rows = [record(index) for index in range(1, 5)]
        self.manifest = dict(model='gptj', owner=dict(server='server2'),
            model_revision=oracle.MODEL_REVISION, tokenizer_sha256='a'*64,
            streams=dict(cf=dict(lock=dict(stream_sha256='b'*64))),
            runtime=dict(profile='TEST_ONLY_CPU_FIXTURE_NOT_PRETRAINED'),
            code_commit='c'*40, official_tree_sha256='d'*64,
            model_snapshot='TEST_ONLY_CPU_MODEL_NO_ASSET', assets_identity_sha256='e'*64)
        self.frozen = oracle.plan(self.manifest, self.rows)
        self.state = oracle.capture_state(self.model, self.tok)
        self.canonical = factual.evaluate_counterfact(self.model, self.tok, self.rows,
            batch_size=3, identity=oracle.external_identity(self.manifest))
        self.path = Path(self.temp.name)/'canonical-first4.json'
        write_new(self.path, self.canonical)
        self.member = oracle.member(self.path)

    def compare(self, **changes):
        options = dict(frozen_plan=self.frozen, canonical_member=self.member,
            state_identity=self.state, test_only_cpu=True)
        options.update(changes)
        return oracle.compare(self.model, self.tok, self.manifest, self.rows,
                              self.canonical, **options)

    def test_preregistered_fixed_first4_source_tolerances_without_actual_claim(self):
        self.assertFalse(self.frozen['actual_GPU'])
        self.assertEqual(self.frozen['evidence_scope'], reference.SMOKE_SCOPE)
        self.assertEqual(self.frozen['source']['shared_adapter']['sha256'], oracle.SHARED_SHA256)
        self.assertEqual(self.frozen['source']['lock']['sha256'], oracle.LOCK_SHA256)
        for name, value in self.frozen['source'].items():
            if isinstance(value, dict):
                self.assertTrue(value['path'].startswith('official/'))
                self.assertFalse(Path(value['path']).is_absolute())
        self.assertEqual(self.frozen['tolerances']['nll_abs_nats'], 1e-4)
        self.assertEqual(self.frozen['tolerances']['nll_relative_to_native'], 1e-5)
        self.assertNotIn('case_id', json.dumps(self.frozen))
        self.assertEqual(self.frozen['existing_external_identity'],
                         self.canonical['identity']['external_identity'])
        for rows in (self.rows[:3], self.rows+[record(5)], list(reversed(self.rows))):
            with self.assertRaises(ValueError):
                oracle.plan(self.manifest, rows)

    def test_shared_unchanged_native_CPU_pipeline_original_forward4_and_external_identity(self):
        before = len(self.model.calls)
        observed = self.compare()
        self.assertEqual(observed['status'], 'CPU_FIXTURE_PASS')
        self.assertFalse(observed['actual_GPU'])
        self.assertEqual(len(self.model.calls)-before, 4)
        self.assertEqual(observed['work']['forward_calls'], 4)
        self.assertEqual(observed['evidence'][reference.CPU_SCOPE], 'PASS')
        self.assertEqual(observed['evidence'][reference.SMOKE_SCOPE], 'NOT_OBSERVED')
        self.assertEqual(observed['evidence'][reference.FULL_SCOPE], 'NOT_OBSERVED')
        native = observed['shared_original_result']['native']
        self.assertEqual(native['identity']['external_identity'],
                         self.canonical['identity']['external_identity'])
        self.assertEqual(native['canonical_payload_sha256'], digest(self.canonical))
        self.assertEqual(oracle.capture_state(self.model, self.tok), self.state)
        self.assertEqual(oracle.verify(self.frozen, observed, self.member), oracle.compact(observed))

    def test_same_original_GPU_scope_on_CPU_is_not_qualified_no_forward_no_CPU_as_GPU(self):
        before = len(self.model.calls)
        observed = self.compare(test_only_cpu=False)
        self.assertEqual(observed['status'], 'NOT_QUALIFIED')
        self.assertFalse(observed['actual_GPU'])
        self.assertEqual(len(self.model.calls), before)
        self.assertEqual(observed['evidence'][reference.SMOKE_SCOPE], 'NOT_OBSERVED')
        self.assertFalse(oracle.verify(self.frozen, observed, self.member)['actual_GPU'])

    def test_unknown_original_error_preserved_no_retry_and_rng_restored(self):
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        self.model.consume_rng, self.model.corrupt = True, 'error'
        before = len(self.model.calls)
        with self.assertRaisesRegex(RuntimeError, 'originating forward failure'):
            self.compare()
        self.assertEqual(len(self.model.calls)-before, 1)
        self.assertEqual(py, random.getstate())
        self.assertEqual(np_state[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_plan_horizon_tolerance_source_or_identity_tamper_rejected_before_forward(self):
        for update in ('tolerance', 'source', 'identity', 'horizon'):
            changed = deepcopy(self.frozen)
            if update == 'tolerance':
                changed['tolerances']['nll_abs_nats'] = .1
            elif update == 'source':
                changed['source']['shared_commit'] = '0'*40
            elif update == 'identity':
                changed['existing_external_identity']['state_identity'] = 'FAKE_W0'
            else:
                changed['cohort']['requests'] = 2000
            changed['plan_sha256'] = digest({key:value for key,value in changed.items() if key!='plan_sha256'})
            before = len(self.model.calls)
            with self.assertRaises(ValueError):
                self.compare(frozen_plan=changed)
            self.assertEqual(len(self.model.calls), before)

    def test_raw_canonical_member_mutation_or_new_identity_is_not_relabelled(self):
        changed = deepcopy(self.canonical)
        changed['identity']['external_identity']['state_identity'] = 'FAKE_W0'
        changed['identity_sha256'] = digest(changed['identity'])
        before = len(self.model.calls)
        with self.assertRaisesRegex(ValueError, 'VALUE_MEMBER_MISMATCH'):
            oracle.compare(self.model, self.tok, self.manifest, self.rows, changed,
                frozen_plan=self.frozen, canonical_member=self.member,
                state_identity=self.state, test_only_cpu=True)
        self.assertEqual(len(self.model.calls), before)
        self.path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'MEMBER_CHANGED'):
            self.compare()
        changed_member = oracle.member(self.path)
        with self.assertRaisesRegex(ValueError, 'EXTERNAL_IDENTITY_CHANGED'):
            oracle.compare(self.model, self.tok, self.manifest, self.rows, changed,
                frozen_plan=self.frozen, canonical_member=changed_member,
                state_identity=self.state, test_only_cpu=True)

    def test_precanonical_parameter_version_tokenizer_hook_and_config_mutation_guard(self):
        before = len(self.model.calls)
        with torch.no_grad():
            self.model.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
            self.compare()
        self.assertEqual(len(self.model.calls), before)
        self.state = oracle.capture_state(self.model, self.tok)
        self.tok.pad_token_id = 9
        with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
            self.compare()
        self.tok.pad_token_id = 0
        handle = self.model.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
            self.compare()
        handle.remove()
        self.model.config.use_cache = True
        with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
            self.compare()

    def test_native_context_history_and_loaded_global_state_are_guarded_without_tensor_copy(self):
        native = dict(completed_batch=0, contexts={'native_templates': [['{}']]},
                      history={}, cache_loaded=False)
        callback = lambda: deepcopy(native)
        state = oracle.capture_state(self.model, self.tok, callback)
        native['contexts']['native_templates'][0].append('MUTATED')
        with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
            self.compare(state_identity=state, state_callback=callback)
        native['completed_batch'] = 1
        with self.assertRaisesRegex(ValueError, 'NOT_COLD_W0'):
            oracle.capture_state(self.model, self.tok, callback)
        module_name = oracle.NATIVE_MODULES[1]
        module = SimpleNamespace(__file__=oracle.__file__, cache_c=torch.zeros((1,2,2)),
            CONTEXT_TEMPLATES_CACHE=[['{}']], COV_CACHE={})
        with patch.dict(sys.modules, {module_name:module}):
            state = oracle.capture_state(self.model, self.tok)
            self.assertEqual(state['native_globals'][module_name]['state']['cache_c']['kind'],
                             'TENSOR_METADATA_ONLY')
            module.CONTEXT_TEMPLATES_CACHE[0].append('MUTATED')
            with self.assertRaisesRegex(ValueError, 'STATE_MUTATED'):
                self.compare(state_identity=state)

    def test_local_mismatch_preserved_no_tolerance_relaxation_or_fallback(self):
        changed = deepcopy(self.canonical)
        row = changed['cases'][0]
        row['rewrite_prompts_probs'][0]['target_new'] += .01
        row['rewrite_observations'][0]['target_new']['mean_nll'] += .01
        path = Path(self.temp.name)/'canonical-mismatched.json'
        write_new(path, changed)
        before = len(self.model.calls)
        observed = oracle.compare(self.model, self.tok, self.manifest, self.rows, changed,
            frozen_plan=self.frozen, canonical_member=oracle.member(path),
            state_identity=self.state, test_only_cpu=True)
        self.assertEqual(observed['status'], 'MISMATCH')
        self.assertEqual(len(self.model.calls)-before, 4)
        self.assertEqual(observed['evidence'][reference.CPU_SCOPE], 'FAIL')
        self.assertFalse(observed['actual_GPU'])
        self.assertEqual(observed['shared_original_result']['tolerances'], self.frozen['tolerances'])

    def test_receipt_scope_actual_identity_and_proof_tampering_rejected(self):
        observed = self.compare()
        for field in ('actual_GPU', 'status', 'scope', 'state', 'forward', 'external', 'missing_native'):
            changed = deepcopy(observed)
            if field == 'actual_GPU':
                changed['actual_GPU'] = True
            elif field == 'status':
                changed['status'] = 'PASS_ACTUAL_GPU_SMOKE'
            elif field == 'scope':
                changed['evidence'][reference.FULL_SCOPE] = 'PASS'
                changed['shared_original_result']['evidence'][reference.FULL_SCOPE] = 'PASS'
            elif field == 'state':
                changed['binding']['physical_state_sha256'] = '0'*64
            elif field == 'forward':
                changed['shared_original_result']['work']['forward_calls'] = 0
                changed['work']['forward_calls'] = 0
            elif field == 'external':
                identity = changed['shared_original_result']['native']['identity']
                identity['external_identity']['state_identity'] = 'FAKE'
                changed['shared_original_result']['native']['identity_sha256'] = digest(identity)
            else:
                del changed['shared_original_result']['native']
            changed['receipt_sha256'] = digest({key:value for key,value in changed.items() if key!='receipt_sha256'})
            with self.assertRaises(ValueError):
                oracle.verify(self.frozen, changed, self.member)

    def test_missing_native_cannot_promote_fixture_to_actual_smoke(self):
        observed = self.compare()
        changed = deepcopy(observed)
        del changed['shared_original_result']['native']
        changed['status'], changed['shared_original_result']['status'] = 'PASS_ACTUAL_GPU_SMOKE', 'PASS'
        changed['actual_GPU'], changed['test_only_cpu_fixture'] = True, False
        changed['evidence_scope'] = changed['shared_original_result']['evidence_scope'] = reference.SMOKE_SCOPE
        evidence = {key:'PASS' if key == reference.SMOKE_SCOPE else 'NOT_OBSERVED'
                    for key in changed['evidence']}
        changed['evidence'] = deepcopy(evidence)
        changed['shared_original_result']['evidence'] = deepcopy(evidence)
        changed['receipt_sha256'] = digest({key:value for key,value in changed.items() if key!='receipt_sha256'})
        with self.assertRaisesRegex(ValueError, 'REQUIRES_ORIGINAL_PROOF'):
            oracle.verify(self.frozen, changed, self.member)

    def test_compact_no_prompt_token_case_pointer_raw_member_path_or_model_copy(self):
        compact = oracle.compact(self.compare())
        text = json.dumps(compact)
        for word in ('Ada', 'case_id', 'input_token', 'pointer', 'snapshot', 'path', 'target_new'):
            self.assertNotIn(word, text)
        self.assertEqual(compact['original_forward_calls'], 4)
        self.assertFalse(compact['actual_GPU'])


if __name__ == '__main__':
    unittest.main()
