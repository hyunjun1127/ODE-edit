"""CPU-only production state-path regressions; no model or CUDA initialization."""
import unittest
from unittest.mock import Mock

import torch

from .backend import Backend
from .common import KEYS, tensor_sha


class StateRouting(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.endpoint_weights = {
            t: {k: (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 17 + t * .03125 + layer)
                for layer, k in enumerate(KEYS)}
            for t in (0, 1, 5, 10)
        }
        self.backend = Backend.__new__(Backend)
        self.backend.torch = torch
        self.backend.weights = {k: torch.zeros_like(v) for k, v in self.endpoint_weights[0].items()}
        self.backend.endpoint = Mock(side_effect=lambda t: self.endpoint_weights[t])
        self.backend.unchanged = Mock()
        self.backend.upload_bytes = 0
        self.backend.materialize_seconds = 0.
        self.backend.restore_seconds = 0.
        self.immutable_hashes = {t: {k: tensor_sha(v) for k, v in ws.items()}
                                 for t, ws in self.endpoint_weights.items()}

    def assert_weights(self, expected):
        self.assertEqual(len(self.backend.weights), 5)
        for k in KEYS:
            self.assertEqual(self.backend.weights[k].dtype, torch.float32)
            self.assertTrue(torch.equal(self.backend.weights[k], expected[k]), k)

    def tearDown(self):
        self.assertEqual(self.immutable_hashes,
                         {t: {k: tensor_sha(v) for k, v in ws.items()}
                          for t, ws in self.endpoint_weights.items()})

    def test_actual_recipe_without_counterfactual_fields(self):
        with self.backend.state(dict(kind='ACTUAL', actual_checkpoint=5)) as receipt:
            self.assert_weights(self.endpoint_weights[5])
        self.assertEqual(receipt['restore'], 'EXACT_BYTES')
        self.backend.unchanged.assert_called_once()

    def test_t1_actual_recipe_forced_diagonal_all_five(self):
        for a, b, index in ((0, 1, 0), (1, 5, 1), (5, 10, 2)):
            with self.subTest(a=a, b=b):
                with self.backend.state(dict(kind='ACTUAL', actual_checkpoint=a),
                                        force_removal=(b, [index])) as receipt:
                    self.assert_weights(self.endpoint_weights[a])
                    self.assertEqual(receipt['removed'], [index])
                    self.assertEqual(receipt['endpoint'], b)
                self.assert_weights(self.endpoint_weights[b])
                self.assertEqual(receipt['restore'], 'EXACT_BYTES')

    def test_counterfactual_single_matches_fp64_one_cast(self):
        expected = {k: (self.endpoint_weights[10][k].double() -
                        (self.endpoint_weights[5][k].double() - self.endpoint_weights[1][k].double())).float()
                    for k in KEYS}
        with self.backend.state(dict(kind='COUNTERFACTUAL', construction_endpoint='10',
                                     removed_cohort_indices='1')):
            self.assert_weights(expected)
        self.assert_weights(self.endpoint_weights[10])

    def test_counterfactual_pair_matches_ordered_fp64_one_cast(self):
        expected = {k: (self.endpoint_weights[10][k].double() -
                        (self.endpoint_weights[1][k].double() - self.endpoint_weights[0][k].double()) -
                        (self.endpoint_weights[5][k].double() - self.endpoint_weights[1][k].double())).float()
                    for k in KEYS}
        with self.backend.state(dict(kind='COUNTERFACTUAL', construction_endpoint=10,
                                     removed_cohort_indices='0;1')):
            self.assert_weights(expected)
        self.assert_weights(self.endpoint_weights[10])

    def test_exception_restores_immutable_entry_not_inverse_update(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic observer failure'):
            with self.backend.state(dict(kind='ACTUAL', actual_checkpoint=0),
                                    force_removal=(1, [0])) as receipt:
                for value in self.backend.weights.values():
                    value.zero_()
                raise RuntimeError('synthetic observer failure')
        self.assert_weights(self.endpoint_weights[1])
        self.assertEqual(receipt['restore'], 'EXACT_BYTES')

    def test_malformed_counterfactual_is_not_silently_actual(self):
        with self.assertRaises(KeyError):
            with self.backend.state(dict(kind='COUNTERFACTUAL', actual_checkpoint=0)):
                self.fail('malformed recipe must fail')
        self.backend.endpoint.assert_not_called()


if __name__ == '__main__':
    unittest.main()
