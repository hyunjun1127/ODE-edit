"""CPU synthetic CapturedHookedFitter schema checks, no actual model or I/O."""
from io import BytesIO
import json
import unittest

import numpy as np
import torch

from .storage import StorageBoundary, native_evidence


def fixture():
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    update = weight.clone()+.5
    targets = [torch.arange(4, dtype=torch.float32)+i for i in range(2)]
    return dict(weight=weight, captures=dict(compute_z=targets,
        compute_ks=[torch.ones((2, 8), dtype=torch.float32)],
        get_module_input_output_at_words=[torch.ones((2, 4), dtype=torch.float32)],
        solve_update=[update]), target=torch.stack(targets, 1),
        anchors=torch.ones((4, 2)), radii=torch.ones(2),
        target_observations=[dict(case_id=i, anchor=torch.ones(4), delta=torch.ones(4)*.1,
                                  target=targets[i], radius=1., adam_updates=3)
                             for i in range(2)],
        receipt=dict(source={'source_sha256': 'source-sha', 'module_global_patch_count': 0},
            compute_z=2, compute_ks=1, solve=1, history_append=0,
            entry_weight_sha256='entry-sha', endpoint_weight_sha256='endpoint-sha',
            history_sha256='history-sha', projector_sha256='P-sha',
            actual_delta_norm=1., z_hook=dict(batches=[dict(
                gradients=[[torch.ones(4)] for _ in range(2)],
                deltas=torch.ones((2, 4)), loss_steps=[4, 4], adam_steps=[3, 3])])) )


def policy(payload):
    return payload['native_evidence_storage_policy']


def tensor_count(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(x) for x in value)
    return 0


class NativeStorageTests(unittest.TestCase):
    def test_actual_capture_retains_z_key_source_excludes_full_solve(self):
        value = fixture()
        payload = native_evidence(value)
        self.assertNotIn('weight', payload)
        self.assertNotIn('solve_update', payload['captures'])
        self.assertTrue(torch.equal(payload['target'], value['target']))
        self.assertTrue(torch.equal(payload['captures']['compute_ks'][0], value['captures']['compute_ks'][0]))
        self.assertEqual(payload['receipt']['source'], value['receipt']['source'])
        self.assertEqual(payload['receipt']['endpoint_weight_sha256'], 'endpoint-sha')
        self.assertEqual(set(policy(payload)['excluded_field_paths']), {'weight', 'captures/solve_update'})

    def test_small_activation_delta_and_gradient_are_not_weight_delta(self):
        payload = native_evidence(fixture())
        self.assertEqual(payload['target_observations'][0]['delta'].shape, (4,))
        self.assertEqual(payload['receipt']['z_hook']['batches'][0]['deltas'].shape, (2, 4))
        self.assertEqual(payload['receipt']['z_hook']['batches'][0]['gradients'][0][0].shape, (4,))

    def test_input_and_owned_result_have_no_mutation_alias(self):
        value = fixture()
        before = value['weight'].clone()
        payload = native_evidence(value)
        self.assertTrue(torch.equal(value['weight'], before))
        self.assertIn('solve_update', value['captures'])
        value['target'].add_(17)
        self.assertFalse(torch.equal(value['target'], payload['target']))
        payload['receipt']['source']['source_sha256'] = 'new'
        self.assertEqual(value['receipt']['source']['source_sha256'], 'source-sha')

    def test_nested_states_and_full_delta_equivalents_excluded(self):
        value = fixture()
        value['extra'] = dict(W={'4': value['weight'].clone()}, M4=torch.ones((8, 8)),
            rng={'python': (3, tuple(range(30)), None)},
            optimizer_state={'exp_avg': torch.ones((4, 8))},
            actual_delta=value['weight'].clone(), ideal_delta=value['weight'].clone(),
            D_acc64=value['weight'].double(), full_delta=value['weight'].reshape(-1).clone(),
            delta={'chunks': [value['weight'].clone()]}, kept={'seconds': 3.})
        payload = native_evidence(value)
        self.assertEqual(payload['extra'], {'kept': {'seconds': 3.}})
        paths = policy(payload)['excluded_field_paths']
        for key in value['extra']:
            if key != 'kept': self.assertIn('extra/'+key, paths)

    def test_state_scalar_hash_receipts_are_retained(self):
        value = fixture()
        value['entry'] = dict(W='sha-W', M='sha-M', rng='sha-rng')
        payload = native_evidence(value)
        self.assertEqual(payload['entry'], value['entry'])

    def test_renamed_endpoint_alias_removed_even_before_original(self):
        value = fixture()
        alias = value['weight'][:, :2]
        value = dict(analysis={'gradient': alias}, **value)
        payload = native_evidence(value)
        self.assertEqual(payload['analysis'], {})
        self.assertIn('analysis/gradient', policy(payload)['excluded_field_paths'])

    def test_renamed_solve_update_view_removed(self):
        value = fixture()
        value['analysis'] = {'other': value['captures']['solve_update'][0][0]}
        payload = native_evidence(value)
        self.assertEqual(payload['analysis'], {})

    def test_independent_gradient_analysis_is_not_checkpoint(self):
        value = fixture()
        value['analysis'] = dict(gradient=torch.ones((4, 8)), projected_gradient=torch.ones((4, 8))*.5)
        payload = native_evidence(value)
        self.assertEqual(tensor_count(payload['analysis']), 2)

    def test_unclassified_full_writer_sized_tensor_removed(self):
        value = fixture()
        value['analysis'] = dict(unknown_clone=value['weight'].clone(),
                                 transposed=value['weight'].T.clone(), flat=value['weight'].flatten().clone())
        payload = native_evidence(value)
        self.assertEqual(payload['analysis'], {})

    def test_exclusion_receipt_is_json_scalar_only(self):
        payload = native_evidence(fixture())
        receipt = policy(payload)
        self.assertEqual(tensor_count(receipt), 0)
        json.dumps(receipt, allow_nan=False)
        weight = next(x for x in receipt['excluded'] if x['path'] == 'weight')
        self.assertEqual(weight['identity']['shape'], [4, 8])
        self.assertEqual(weight['identity']['logical_bytes'], 128)
        self.assertEqual(len(weight['identity']['sha256_header_and_bytes']), 64)
        self.assertFalse(receipt['exact_resume_from_payload'])

    def test_excluded_rng_receipt_does_not_retain_rng_scalar_values(self):
        value = fixture()
        value['rng'] = {'random_state': [123456789, 987654321]}
        text = json.dumps(policy(native_evidence(value)))
        self.assertNotIn('123456789', text)
        self.assertNotIn('987654321', text)

    def test_numpy_arrays_and_aliases(self):
        value = fixture()
        value['extra'] = dict(gradient=np.ones((4, 8)), scalar=np.float64(.5))
        result = native_evidence(value)
        self.assertIsInstance(result['extra']['gradient'], np.ndarray)
        self.assertIsInstance(result['extra']['scalar'], float)
        value['extra']['gradient'][0, 0] = 50
        self.assertEqual(result['extra']['gradient'][0, 0], 1.)

    def test_cpu_only_rejects_non_cpu_without_transfer(self):
        value = fixture()
        value['target'] = torch.empty((4, 2), device='meta')
        with self.assertRaisesRegex(StorageBoundary, 'CPU_STRIDED'):
            native_evidence(value)

    def test_unknown_object_cannot_pickle_hidden_state(self):
        value = fixture()
        value['unknown'] = object()
        with self.assertRaisesRegex(StorageBoundary, 'UNSUPPORTED_NATIVE'):
            native_evidence(value)

    def test_complete_payload_weights_only_in_memory_roundtrip(self):
        payload = native_evidence(fixture())
        stream = BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True, map_location='cpu')
        self.assertNotIn('weight', loaded)
        self.assertNotIn('solve_update', loaded['captures'])
        self.assertTrue(torch.equal(loaded['target'], payload['target']))

    def test_repeated_filter_is_explicit_error_not_fake_fresh_receipt(self):
        filtered = native_evidence(fixture())
        # A filtered payload intentionally is not an executable native result.
        with self.assertRaisesRegex(StorageBoundary, 'NATIVE_RESULT_SCHEMA_REQUIRED'):
            native_evidence(filtered)


if __name__ == '__main__':
    unittest.main()
