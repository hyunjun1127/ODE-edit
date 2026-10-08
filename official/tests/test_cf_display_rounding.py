"""Original AlphaEdit display rounding is distinct from the raw Score."""
import unittest

import numpy as np
from scipy.stats import hmean

from official.evaluation.reduce import counterfact, harmonic


class DisplayRoundingTests(unittest.TestCase):
    def test_numpy_not_python_half_round(self):
        self.assertNotEqual(round(16.055, 2), float(np.around(16.055, 2)))
        success = {"target_new": 1., "target_true": 2.}
        failure = {"target_new": 2., "target_true": 1.}
        base = {"paraphrase_prompts_probs": [success],
                "neighborhood_prompts_probs": [failure]}
        good, bad = dict(base, rewrite_prompts_probs=[success]), dict(base, rewrite_prompts_probs=[failure])
        value = counterfact([good] * 3211 + [bad] * (20000 - 3211))
        rates = [16.055, 100., 100.]
        self.assertAlmostEqual(value["Score"], harmonic(rates), places=12)
        expected = float(hmean(np.around(np.asarray(rates), 2)))
        self.assertAlmostEqual(value["Score_AlphaEdit_display"], expected, places=12)

    def test_actual_2000_request_native_aggregation_order(self):
        rng = np.random.default_rng(20261009)
        for _ in range(58):
            counts = rng.integers(0, 11, size=2000)
        new, true = {"target_new": 1., "target_true": 2.}, {"target_new": 2., "target_true": 1.}
        cases = [dict(rewrite_prompts_probs=[new], paraphrase_prompts_probs=[new],
                      neighborhood_prompts_probs=[true] * int(n) + [new] * (10 - int(n)))
                 for n in counts]
        value = counterfact(cases)
        native_specificity = np.mean([np.mean([True] * int(n) + [False] * (10-int(n)))
                                      for n in counts]) * 100
        self.assertEqual(value["Specificity"], 49.165)
        self.assertEqual(native_specificity, 49.165000000000006)
        self.assertEqual(np.around(native_specificity, 2), 49.17)
        self.assertEqual(np.around(value["Specificity"], 2), 49.16)
        expected = float(hmean([100., 100., np.around(native_specificity, 2)]))
        self.assertAlmostEqual(value["Score_AlphaEdit_display"], expected, places=12)
        self.assertAlmostEqual(value["Score"], harmonic([100., 100., 49.165]), places=12)

    def test_zero_rate_and_ties_remain_failures(self):
        case = {name + "_prompts_probs": [{"target_new": 1., "target_true": 1.}]
                for name in ("rewrite", "paraphrase", "neighborhood")}
        value = counterfact([case])
        self.assertEqual(value["Score"], 0.)
        self.assertEqual(value["Score_AlphaEdit_display"], 0.)
        self.assertEqual(value["requests"], 1)


if __name__ == "__main__":
    unittest.main()
