"""CPU proof/schema tests against retained observations, never an 8B replay.

The optional local-fixture tests read exact W0/N4 JSON and a local tokenizer.
Endpoint SHAs come from sealed native/cold receipts, not a newly loaded model.
The nested runner.reuse_proof body is extracted unchanged by AST so tests cover
the actual closure without starting Runtime, touching a GPU or writing proofs.
Tiny CPU tests cover selected-weight version rebinding and reset separately.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng
from project.run_scripts.single_layer_edit_preserving_correction import observer as observer_module
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, model_guard
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, member, tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.observer import CanonicalObserver, ObserverBoundary
from project.run_scripts.single_layer_edit_preserving_correction.runtime import Runtime
from project.run_scripts.single_layer_edit_preserving_correction.tests.test_observer import Toy, Tokenizer, record


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE.parent
LOCAL = Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1')
BASE_BINDING = LOCAL/'inputs/base-binding.json'
REUSE_BINDING = LOCAL/'reuse/observer-binding-r1.json'


@dataclass(frozen=True)
class SealedEndpoint:
    """A prior SHA reference, explicitly NOT a tensor or fresh byte audit."""
    sha256: str


def endpoint_sha(value):
    return value.sha256 if isinstance(value, SealedEndpoint) else tensor_sha(value)


def make_seal(weight, records, endpoint='N4'):
    return dict(status='SELECTION_SEALED', episode_id='CPU_PROOF_FIXTURE', endpoint_id=endpoint,
                endpoint_weight_sha256=endpoint_sha(weight),
                request_order_sha256=digest([r['case_id'] for r in records]),
                selection_ledger_sha256='a'*64)


def extracted_reuse_proof(namespace):
    tree = ast.parse((PACKAGE/'runner.py').read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    functions = [node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == 'reuse_proof']
    if len(functions) != 1:
        raise AssertionError('Expected exactly one actual runner.reuse_proof closure')
    module = ast.Module(body=[copy.deepcopy(functions[0])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(PACKAGE/'runner.py'), 'exec'), namespace)
    return namespace['reuse_proof']


@unittest.skipUnless(BASE_BINDING.is_file() and REUSE_BINDING.is_file(),
                     'Exact S4 retained-observation fixtures are not available')
class RetainedRunnerObserverBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        torch.set_num_threads(1)
        cls.base = json.loads(BASE_BINDING.read_text())
        cls.binding = json.loads(REUSE_BINDING.read_text())
        cls.records = load_prefix(cls.base['dataset_root'], 1000)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.base['snapshot'], local_files_only=True)
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
        if cls.tokenizer.padding_side != 'right':
            raise AssertionError('Actual canonical tokenizer no longer has right-padding default')
        # The same byte-identical source files under the current worktree avoid
        # switching historical module namespaces when all CPU tests run together.
        bind_evaluation_sources(SCRIPTS/'blue_alphaedit_sequential_comparison', helper_root=SCRIPTS)
        obs = CanonicalObserver.__new__(CanonicalObserver)
        from project.run_scripts.baseline_mechanism_first.evaluation import _binding
        obs.bindings, obs.tok, obs.runtime_identity = _binding(), cls.tokenizer, digest(cls.base)
        obs.source_sha256s = sorted(v['sha256'] for v in obs.bindings['receipt'].values())
        cls.obs = obs
        cls.native_binding = json.loads((LOCAL/'reuse/b001-native-binding.json').read_text())
        cold = json.loads(Path(cls.base['cold_capsule']['path']).read_text())
        cls.W0 = SealedEndpoint(cold['W0']['4'])
        cls.WN = SealedEndpoint(cls.native_binding['b1_endpoint_verified'])
        cls.original_members = {name:member(cls.binding[name]['path']) for name in ('W0_first1000', 'N4_B1')}
        for name, actual in cls.original_members.items():
            if actual != cls.binding[name]:
                raise AssertionError('Retained observation source identity drift: '+name)
        cls.source_metrics = {name:json.loads(Path(cls.binding[name]['path']).read_bytes())
                              for name in cls.original_members}

    @classmethod
    def tearDownClass(cls):
        for name, before in cls.original_members.items():
            if member(before['path']) != before:
                raise AssertionError('Read-only fixture bytes changed: '+name)

    def closure_bundle(self, records, weight, role, ref_name):
        captured = []
        ids = [r['case_id'] for r in records]
        namespace = dict(Path=Path, member=member, json=json, digest=digest, tensor_sha=endpoint_sha,
                         obs=self.obs, records=records, ids=ids,
                         rt=types.SimpleNamespace(W0=self.W0, identity={'W0':self.W0.sha256}),
                         lock={'observer_reuse_binding':self.binding,
                               'reused_native_binding':self.native_binding},
                         root=Path('/CPU_FIXTURE_NO_DISK_WRITES'),
                         write=lambda path, value:captured.append((str(path), copy.deepcopy(value))))
        fn = extracted_reuse_proof(namespace)
        seal = make_seal(weight, records, 'W0-reference' if role.startswith('W0') else 'N4')
        with patch.object(observer_module, 'tensor_sha', endpoint_sha):
            supplied = fn(weight, seal, self.binding[ref_name], role)
            compatibility = self.obs.compatibility_for(records, weight, selection_seal=seal)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][1], supplied['proof'])
        self.assertEqual(supplied['proof']['lineage'], self.binding)
        return supplied, compatibility

    def reduce_bundle(self, supplied, compatibility, records, *, w0=False):
        _, expected, _ = self.obs._pairs_and_identity(records)
        return self.obs._reduced_reuse(supplied, compatibility, expected, w0=w0)

    @staticmethod
    def changed_proof(bundle, **fields):
        result = copy.deepcopy(bundle)
        result['proof'].update(fields)
        result['proof']['proof_sha256'] = digest({k:v for k,v in result['proof'].items() if k != 'proof_sha256'})
        return result

    @staticmethod
    def changed_source(bundle, change):
        result = copy.deepcopy(bundle)
        prior = json.loads(result['source_bytes'])
        change(prior)
        result['source_bytes'] = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()
        return RetainedRunnerObserverBoundaryTests.changed_proof(
            result, source_raw_sha256=hashlib.sha256(result['source_bytes']).hexdigest())

    def test_actual_N4_B1_closure_preserves_exact_reduced_values(self):
        records = self.records[:100]
        supplied, compatibility = self.closure_bundle(records, self.WN, 'ENDPOINT_CURRENT', 'N4_B1')
        metrics, proof, scope = self.reduce_bundle(supplied, compatibility, records)
        self.assertEqual(metrics, self.source_metrics['N4_B1']['metrics'])
        self.assertEqual([metrics[k]['denominator'] for k in ('RS', 'PS', 'NS')], [100, 200, 1000])
        self.assertEqual(proof['endpoint_weight_sha256'], self.native_binding['b1_endpoint_verified'])
        self.assertIsNone(scope)
        self.assertFalse(any('raw' in row or 'token_correct' in row for group in metrics.values() for row in group['rows']))

    def test_actual_W0_first_middle_last_cold_subsets_preserve_population_and_rows(self):
        for start in (0, 400, 900):
            with self.subTest(start=start):
                records = self.records[start:start+100]
                supplied, compatibility = self.closure_bundle(records, self.W0, 'W0_REFERENCE_SUBSET', 'W0_first1000')
                metrics, _, scope = self.reduce_bundle(supplied, compatibility, records, w0=True)
                ids = {r['case_id'] for r in records}
                for kind in ('RS', 'PS', 'NS'):
                    expected = [r for r in self.source_metrics['W0_first1000']['metrics'][kind]['rows'] if r['case_id'] in ids]
                    self.assertEqual(metrics[kind]['rows'], expected)
                self.assertEqual([metrics[k]['denominator'] for k in ('RS', 'PS', 'NS')], [100, 200, 1000])
                self.assertEqual(scope['source_population'], 1000)
                self.assertEqual(scope['selected_population'], 100)
                self.assertEqual(scope['current_B100_padding_bit_parity'], 'NOT_CLAIMED')
                self.assertTrue(scope['source_bytes_unchanged'])

    def test_current_endpoint_mismatch_rejected_but_distinct_W0_reference_allowed(self):
        records = self.records[:100]
        supplied, compatibility = self.closure_bundle(records, self.WN, 'ENDPOINT_CURRENT', 'N4_B1')
        wrong = self.changed_proof(supplied, endpoint_weight_sha256='f'*64)
        with self.assertRaisesRegex(ObserverBoundary, 'ENDPOINT_WEIGHT_SHA256_MISMATCH'):
            self.reduce_bundle(wrong, compatibility, records)
        w0, _ = self.closure_bundle(records, self.W0, 'W0_REFERENCE_SUBSET', 'W0_first1000')
        # Retention must compare W0 reference with a different selected endpoint.
        self.assertNotEqual(w0['proof']['endpoint_weight_sha256'], compatibility['endpoint_weight_sha256'])
        self.reduce_bundle(w0, compatibility, records, w0=True)

    def test_runner_cannot_seal_wrong_current_or_W0_weight_even_with_self_consistent_seal(self):
        records = self.records[:100]
        for role, source, error in [('ENDPOINT_CURRENT', 'N4_B1', 'NATIVE_ENDPOINT_MISMATCH'),
                                    ('W0_REFERENCE_SUBSET', 'W0_first1000', 'W0_ENDPOINT_MISMATCH')]:
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, error):
                self.closure_bundle(records, SealedEndpoint('f'*64), role, source)

    def test_exact_source_sha_runtime_and_layout_rejected(self):
        records = self.records[:100]
        supplied, compatibility = self.closure_bundle(records, self.WN, 'ENDPOINT_CURRENT', 'N4_B1')
        cases = [('SOURCE_BYTES_SHA', self.changed_proof(supplied, source_raw_sha256='f'*64)),
                 ('RUNTIME_IDENTITY', self.changed_proof(supplied, runtime_identity='f'*64)),
                 ('HISTORICAL_LAYOUT', self.changed_source(supplied, lambda p:p.update(evaluator_layout='CHANGED')))]
        for error, wrong in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ObserverBoundary, error):
                self.reduce_bundle(wrong, compatibility, records)

    def test_actual_pair_identity_denominator_token_count_rejected(self):
        records = self.records[:100]
        supplied, compatibility = self.closure_bundle(records, self.WN, 'ENDPOINT_CURRENT', 'N4_B1')
        changes = [('PAIR_IDENTITY', lambda p:p['metrics']['RS']['rows'][0].update(identity='f'*64)),
                   ('PAIR_CARDINALITY', lambda p:p['metrics']['PS'].update(denominator=199)),
                   ('TOKEN_COUNT', lambda p:p['metrics']['RS']['rows'][0].update(new_token_count=999))]
        for error, change in changes:
            with self.subTest(error=error), self.assertRaisesRegex(ObserverBoundary, error):
                self.reduce_bundle(self.changed_source(supplied, change), compatibility, records)

    def test_W0_subset_order_population_and_pretrained_endpoint_rejected(self):
        records = self.records[100:200]
        supplied, compatibility = self.closure_bundle(records, self.W0, 'W0_REFERENCE_SUBSET', 'W0_first1000')
        fields = [dict(selected_case_ids=list(reversed(supplied['proof']['selected_case_ids']))),
                  dict(source_population=100), dict(source_request_order_sha256='f'*64),
                  dict(model_pretrained_weight_sha256='f'*64)]
        for changed in fields:
            with self.subTest(changed=next(iter(changed))), self.assertRaises(ObserverBoundary):
                self.reduce_bundle(self.changed_proof(supplied, **changed), compatibility, records, w0=True)

    def test_proof_constructor_requires_selection_seal_before_prompt_access(self):
        with self.assertRaisesRegex(ObserverBoundary, 'SELECTION_SEAL'):
            self.obs.compatibility_for([{'case_id':1}], torch.ones(1), selection_seal={})


class PostSealResetBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_evaluation_sources(SCRIPTS/'blue_alphaedit_sequential_comparison', helper_root=SCRIPTS)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(984)
        self.model, self.tok = Toy(), Tokenizer()
        self.W = dict(self.model.named_parameters())[WEIGHT]
        self.WN = self.W.detach().clone()
        self.W0 = self.WN-.02
        self.obs = CanonicalObserver(self.model, self.tok, runtime_identity='b'*64)
        self.rt = Runtime.__new__(Runtime)
        self.rt.model, self.rt.W, self.rt.W0 = self.model, self.W, self.W0
        self.rt.M = torch.zeros(4, 4)
        self.rt.context = [['{}']]
        self.rt.module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[['{}']], COV_CACHE={})
        self.rt.rng, self.rt.base_guard = capture_rng(), model_guard(self.model)
        self.rt.oracles = []

    def test_postseal_observe_reset_restore_and_next_observe_preserve_identity(self):
        records = [record()]
        candidate = self.WN+.01
        version = self.W._version
        observed = self.obs.observe(records, candidate, selection_seal=make_seal(candidate, records), greedy=False)
        self.assertTrue(observed['entry_selected_weight_restored_exact'])
        self.assertTrue(torch.equal(self.W, self.WN))
        self.assertGreater(self.W._version, version)
        calls = []
        oracle = types.SimpleNamespace(acknowledge_selected_write=lambda expected:calls.append(tensor_sha(expected)))
        self.rt.oracles = [oracle]
        self.rt.sync_oracles()
        self.assertEqual(calls, [tensor_sha(self.WN)])
        # Same runner initial-gate sequence: W0 copy, then WN exact restore.
        self.rt.copy_weight(self.W0)
        self.rt.copy_weight(self.WN)
        self.assertEqual(calls[-2:], [tensor_sha(self.W0), tensor_sha(self.WN)])
        again = self.obs.observe(records, candidate, selection_seal=make_seal(candidate, records), reuse=observed, greedy=False)
        self.assertEqual(again['work']['model_forward_calls'], 0)
        reset = self.rt.reset()
        self.assertEqual(reset['W'], tensor_sha(self.W0))
        self.assertEqual(reset['M'], tensor_sha(torch.zeros(4, 4)))
        self.assertEqual(reset['rng'], digest(self.rt.rng))
        self.assertEqual(self.rt.oracles, [])

    def test_nonselected_mutation_and_selected_parameter_replacement_fail(self):
        records = [record()]
        with torch.no_grad():
            self.model.head.weight.add_(.01)
        with self.assertRaisesRegex(ObserverBoundary, 'BASE_NONSELECTED'):
            self.obs.observe(records, self.WN, selection_seal=make_seal(self.WN, records), greedy=False)
        # Restore the metadata baseline only in this tiny negative-test fixture.
        self.obs.base_guard = model_guard(self.model)
        self.model.model.layers[4].mlp.down_proj.weight = torch.nn.Parameter(self.WN.clone(), requires_grad=False)
        with self.assertRaisesRegex(ObserverBoundary, 'SELECTED_PARAMETER_OBJECT_REPLACED'):
            self.obs.observe(records, self.WN, selection_seal=make_seal(self.WN, records), greedy=False)


if __name__ == '__main__':
    unittest.main()
