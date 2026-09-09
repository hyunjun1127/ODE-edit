import unittest
from rte_scoring import canonical_prediction, prediction_record, summarize, score_source_rows


class RteScoringTests(unittest.TestCase):
    def test_true_false_against_both_gold_labels(self):
        self.assertTrue(prediction_record(1,0)['correct'])
        self.assertFalse(prediction_record(1,1)['correct'])
        self.assertTrue(prediction_record(0,1)['correct'])
        self.assertFalse(prediction_record(0,0)['correct'])

    def test_invalid_not_inverted(self):
        for value in (-1, -2, 99, None, 'unknown'):
            self.assertEqual(canonical_prediction(value), value)
            self.assertFalse(prediction_record(value,0)['valid'])
            self.assertFalse(prediction_record(value,0)['correct'])

    def test_perfect_and_inverted_metrics(self):
        perfect = summarize([prediction_record(1,0),prediction_record(0,1)])
        inverse = summarize([prediction_record(0,0),prediction_record(1,1)])
        self.assertEqual((perfect['accuracy'],perfect['weighted_f1'],perfect['mcc']), (1.,1.,1.))
        self.assertEqual((inverse['accuracy'],inverse['weighted_f1'],inverse['mcc']), (0.,0.,-1.))

    def test_invalid_denominator(self):
        result = summarize([prediction_record(-1,0),prediction_record(0,1)])
        self.assertEqual((result['total'],result['invalid'],result['correct']), (2,1,1))

    def test_both_branches_same_mapping_and_input_immutable(self):
        import copy
        gold=[dict(sentence1='p',sentence2='q',label=0),dict(sentence1='r',sentence2='s',label=1)]
        rows=[dict(sentence1='p',sentence2='q',answer=1,highest_probability_answer='True',prob_yes=.8,prob_no=.2),
              dict(sentence1='r',sentence2='s',answer=0,highest_probability_answer='False',prob_yes=.2,prob_no=.8)]
        before=copy.deepcopy((rows,gold))
        result=score_source_rows(rows,gold,{'f1':0})
        self.assertEqual(result['f1'],1.)
        self.assertEqual(result['f1_new'],1.)
        self.assertEqual(result['generation'],result['alternative'])
        self.assertEqual((rows,gold),before)
        self.assertEqual(result['original_bug_diagnostic'],{'f1':0})

    def test_unknown_alternative_remains_invalid(self):
        gold=[dict(sentence1='p',sentence2='q',label=0)]
        rows=[dict(sentence1='p',sentence2='q',answer=-1,
                   highest_probability_answer='unknown',prob_yes=.8,prob_no=.2)]
        result=score_source_rows(rows,gold,{})
        self.assertEqual(result['generation']['invalid'],1)
        self.assertEqual(result['alternative']['invalid'],1)
        self.assertEqual(result['records'][0]['alternative']['canonical_prediction'],-1)


if __name__ == '__main__':
    unittest.main()
