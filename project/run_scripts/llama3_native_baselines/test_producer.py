"""Small CPU scalar-mapping regressions, not a toy editing experiment."""
import unittest

from .producer import generation_values, scientific_config, transport_support
from .common import PROFILE


def aggregate(count=100, fluency_count=100, consistency_count=100):
    return dict(planned_count=count, fluency_count=fluency_count,
                consistency_count=consistency_count, fluency_sum=0., consistency_sum=0.,
                generation_prompt_count=count, generated_token_count=0, reason_counts={})


class ProducerTests(unittest.TestCase):
    def test_measured_zero_not_missing(self):
        values = generation_values('current/post', aggregate(), 100)
        self.assertEqual(values['current/post/fluency/ngram_entropy'], 0.)
        self.assertEqual(values['current/post/consistency/reference_score'], 0.)

    def test_missing_omitted_but_coverage_preserved(self):
        a = aggregate(fluency_count=0, consistency_count=0)
        a['reason_counts'] = {'missing_generation_prompts': 100}
        values = generation_values('current/pre', a, 100)
        self.assertNotIn('current/pre/fluency/ngram_entropy', values)
        self.assertNotIn('current/pre/consistency/reference_score', values)
        self.assertEqual(values['current/pre/generation/planned_count'], 100)

    def test_macro_and_units(self):
        a = aggregate(fluency_count=80, consistency_count=50)
        a.update(fluency_sum=240., consistency_sum=25.)
        values = generation_values('current/post', a, 100)
        self.assertEqual(values['current/post/fluency/ngram_entropy'], 3.)
        self.assertEqual(values['current/post/consistency/reference_score'], .5)

    def test_milestone_not_current(self):
        with self.assertRaisesRegex(RuntimeError, 'GENERATION_COHORT_DENOMINATOR'):
            generation_values('current/post', aggregate(500), 100)

    def test_nonfinite_and_unsupported_rejected(self):
        for change in ({'fluency_sum': float('nan')}, {'consistency_sum': 101.},
                       {'reason_counts': {'prompt_text': 1}}, {'generated_token_count': -1}):
            a = aggregate(); a.update(change)
            with self.assertRaises(RuntimeError): generation_values('current/post', a, 100)

    def test_native_identity_not_writer_alias(self):
        for method, writer in (('MEMIT', 'memit'), ('RECT', 'rect'), ('CAKE', 'cake'),
                               ('PRUNE', 'prune'), ('ALPHAEDIT_BLUE', 'alphaedit-blue')):
            cfg = scientific_config(method, 'attempt-x', 'a'*40, 'b'*64,
                                    dict(profile=PROFILE, assets_sha256='c'*64, source_sha='d'*40))
            self.assertEqual(cfg['writer'], writer)
            self.assertEqual(cfg['baseline'], method)
            self.assertNotIn('job_id', cfg)

    def test_support_probe_never_online(self):
        self.assertEqual(transport_support()['online_validation'], 'NOT_OBSERVED')


if __name__ == '__main__': unittest.main()
