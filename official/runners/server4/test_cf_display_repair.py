"""CPU raw-shaped fixtures only; no model, network or GPU qualification."""
import copy
import unittest
from official.evaluation.reduce import counterfact
from official.runners.server4.qwen_run import _factual_scalars, _milestone_scalars
from official.runners.server1.test_tracking_binding import cf_endpoint
from official.tracking.method import validate, DISPLAY_COMPONENTS


class CFDisplayRepair(unittest.TestCase):
    def test_request_bits_boundary_and_legacy_failure(self):
        cases = []
        for i in range(2000):
            case = {}
            for kind, population, successes in (("rewrite", 1, 164),
                                                ("paraphrase", 2, 438),
                                                ("neighborhood", 10, 17711)):
                pairs = []
                for j in range(population):
                    good = i * population + j < successes
                    true, new = ((1., 2.) if good else (2., 1.)) if kind == "neighborhood" else ((2., 1.) if good else (1., 2.))
                    pairs.append(dict(target_true=true, target_new=new))
                case[kind + "_prompts_probs"] = pairs
            cases.append(case)
        summary = counterfact(cases)
        before = copy.deepcopy((cases, summary))
        payload = _factual_scalars("cf", cases, summary, "W0_first2000", 0)
        validate(payload, official=True)
        self.assertEqual(payload["official/W0_first2000/Specificity_AlphaEdit_display"], 88.56)
        legacy = {k: v for k, v in payload.items() if not any(k.endswith(x) for x in DISPLAY_COMPONENTS)}
        with self.assertRaisesRegex(ValueError, "OFFICIAL_DISPLAY_SCORE_MISMATCH"):
            validate(legacy, official=True)
        self.assertEqual((cases, summary), before)

    def test_endpoint_scopes_and_no_mutation(self):
        for n, prefix, edits in ((100, "current/post", 100), (2000, "W0_first2000", 0)):
            raw = cf_endpoint(n); before = copy.deepcopy(raw)
            payload = _factual_scalars("cf", raw["cases"], raw["summary"], prefix, edits)
            validate(payload, official=True)
            self.assertEqual(raw, before)
            for field in DISPLAY_COMPONENTS:
                self.assertIn(f"official/{prefix}/{field}", payload)
        raw = cf_endpoint(500)
        payload = _milestone_scalars("cf", raw["cases"], raw["summary"], 500)
        validate(payload, official=True)
        self.assertEqual(payload["official/current/post/requests"], 100)
        self.assertEqual(payload["official/all_seen/post/requests"], 500)

    def test_zsre_unchanged_and_empty_fail_closed(self):
        summary = dict(requests=100, Efficacy=50., Generalization=50., Specificity=50., Score=50.)
        payload = _factual_scalars("zsre", [{}] * 100, summary, "current/post", 100)
        self.assertFalse(any(k.endswith(x) for k in payload for x in DISPLAY_COMPONENTS))
        raw = cf_endpoint(100); raw["cases"][0]["rewrite_prompts_probs"] = []
        with self.assertRaisesRegex(ValueError, "DISPLAY_EMPTY_REQUEST"):
            _factual_scalars("cf", raw["cases"], raw["summary"], "current/post", 100)

if __name__ == "__main__": unittest.main()
