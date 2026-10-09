"""Paper locality is request-macro answer accuracy, not W0 agreement."""
import unittest
from official.evaluation.reduce import zsre


class ZsreLocDefinition(unittest.TestCase):
    def test_wrong_unchanged_base_is_not_correct(self):
        out = zsre([dict(rewrite_prompts_correct=[True], paraphrase_prompts_correct=[True],
                        neighborhood_prompts_correct=[False, False],
                        neighborhood_W0_agreement=[True, True])])
        self.assertEqual(out['Specificity'], 0)
        self.assertEqual(out['Specificity_loc_ans'], 0)
        self.assertEqual(out['W0_prediction_agreement'], 100)

    def test_request_macro_not_token_micro(self):
        rows = [dict(rewrite_prompts_correct=[True], paraphrase_prompts_correct=[False],
                     neighborhood_prompts_correct=bits, neighborhood_W0_agreement=[True]*len(bits))
                for bits in ([True], [False, False, False])]
        self.assertEqual(zsre(rows)['Specificity'], 50)
        self.assertNotEqual(zsre(rows)['Specificity'], 25)

    def test_loc_measured_without_W0(self):
        rows = [dict(rewrite_prompts_correct=[True], paraphrase_prompts_correct=[True],
                     neighborhood_prompts_correct=[True, False], neighborhood_W0_agreement=None)]
        out = zsre(rows)
        self.assertEqual(out['Specificity'], 50)
        self.assertNotIn('W0_prediction_agreement', out)

    def test_empty_or_missing_answer_is_not_zero(self):
        row = dict(rewrite_prompts_correct=[True], paraphrase_prompts_correct=[True],
                   neighborhood_W0_agreement=[True])
        with self.assertRaises(KeyError):
            zsre([row])
        with self.assertRaises(ValueError):
            zsre([dict(row, neighborhood_prompts_correct=[])])


if __name__ == '__main__':
    unittest.main()
