"""Synthetic request-cluster statistics only; no raw/model/evaluator access."""
import unittest

from .refresh_uncertainty import request_cluster_bootstrap


def row(case, prompt, new, true):
    return dict(case_id=case, prompt_index=prompt,
                identity=f'case-{case}-prompt-{prompt}-target-pair',
                new_nll=float(new), true_nll=float(true))


class RequestClusterBootstrapTests(unittest.TestCase):
    def test_two_outcome_p2_clusters_stay_together(self):
        before = [row(c, p, 3 if c == 0 else 1, 2)
                  for c in range(2) for p in range(2)]
        after = [row(c, p, 1 if c == 0 else 3, 2)
                 for c in range(2) for p in range(2)]
        result = request_cluster_bootstrap(before, after, 'PS')
        self.assertEqual(result['request_denominator'], 2)
        self.assertEqual(result['prompt_denominator'], 4)
        self.assertEqual(result['preference_delta_pp'], 0)
        self.assertEqual(result['desired_target_nll_mean_delta'], 0)
        self.assertEqual(result['preference_delta_pp_ci95']['low'], -100)
        self.assertEqual(result['preference_delta_pp_ci95']['high'], 100)
        self.assertEqual(result['desired_target_nll_mean_delta_ci95']['low'], -2)
        self.assertEqual(result['desired_target_nll_mean_delta_ci95']['high'], 2)
        self.assertEqual(result['method']['observed_prompts_per_request'], [2])
        self.assertFalse(result['method']['CI_zero_is_gate'])

    def test_ns_uses_true_target_and_ties_fail(self):
        before = [row(c, p, 2, 2) for c in range(2) for p in range(10)]
        after = [row(c, p, 2, 1) for c in range(2) for p in range(10)]
        result = request_cluster_bootstrap(before, after, 'NS')
        self.assertEqual(result['request_denominator'], 2)
        self.assertEqual(result['prompt_denominator'], 20)
        self.assertEqual(result['preference_delta_pp'], 100)
        self.assertEqual(result['desired_target_nll_mean_delta'], -1)
        self.assertEqual(result['preference_delta_pp_ci95']['low'], 100)
        self.assertEqual(result['preference_delta_pp_ci95']['high'], 100)
        self.assertEqual(result['method']['desired_target_nll_field'], 'true_nll')

    def test_serialization_order_seed_and_block_reproducible(self):
        before = [row(c, p, 3 if c % 2 else 1, 2)
                  for c in range(5) for p in range(2)]
        after = [row(c, p, 1 if c % 2 else 3, 2)
                 for c in range(5) for p in range(2)]
        original = request_cluster_bootstrap(before, after, 'PS')
        repeated = request_cluster_bootstrap(before, after, 'PS')
        reordered = request_cluster_bootstrap(before[::-1], after[::-1], 'PS')
        self.assertEqual(original, repeated)
        self.assertEqual(original, reordered)
        blocked = request_cluster_bootstrap(before, after, 'PS', block_size=1)
        for key in ('preference_delta_pp_ci95', 'desired_target_nll_mean_delta_ci95'):
            self.assertEqual(original[key], blocked[key])
        self.assertEqual(original['method']['replicate_distribution_sha256'],
                         blocked['method']['replicate_distribution_sha256'])

    def test_partial_groups_use_actual_prompt_denominator(self):
        before = [row(0, 0, 3, 2), row(1, 0, 1, 2), row(1, 1, 1, 2)]
        after = [row(0, 0, 1, 2), row(1, 0, 3, 2), row(1, 1, 3, 2)]
        result = request_cluster_bootstrap(before, after, 'PS')
        self.assertAlmostEqual(result['preference_delta_pp'], -100 / 3)
        self.assertAlmostEqual(result['desired_target_nll_mean_delta'], 2 / 3)
        self.assertEqual(result['method']['observed_prompts_per_request'], [1, 2])

    def test_empty_is_na_not_zero(self):
        result = request_cluster_bootstrap([], [], 'RS')
        self.assertEqual(result['status'], 'NOT_AVAILABLE_EMPTY_GROUP')
        self.assertEqual(result['actual_resamples'], 0)
        self.assertIsNone(result['preference_delta_pp'])
        self.assertIsNone(result['desired_target_nll_mean_delta_ci95'])

    def test_identity_mismatch_duplicate_nonfinite_rejected(self):
        sample = [row(0, 0, 1, 2)]
        with self.assertRaisesRegex(ValueError, 'PAIRED_IDENTITY_SET'):
            request_cluster_bootstrap(sample, [dict(sample[0], identity='different')], 'RS')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_IDENTITY'):
            request_cluster_bootstrap(sample * 2, sample * 2, 'RS')
        with self.assertRaisesRegex(ValueError, 'NONFINITE_PAIR'):
            request_cluster_bootstrap(sample, [dict(sample[0], new_nll=float('nan'))], 'RS')
        for kwargs in ({'resamples': 0}, {'seed': -1}, {'block_size': 0}):
            with self.assertRaises(ValueError):
                request_cluster_bootstrap(sample, sample, 'RS', **kwargs)


if __name__ == '__main__':
    unittest.main()
