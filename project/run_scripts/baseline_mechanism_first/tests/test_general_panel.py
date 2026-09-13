import unittest

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary
from project.run_scripts.baseline_mechanism_first.general_panel import select_rows


class Tokenizer:
    def encode(self,text,add_special_tokens=True):
        return ([1] if add_special_tokens else [])+[ord(x) for x in text]


class GeneralPanelTests(unittest.TestCase):
    def test_exact_prefix_and_token_prefix_no_filter(self):
        rows=[dict(id=str(i),text='abcdef') for i in range(130)]
        selected=select_rows(rows,Tokenizer(),count=128,max_tokens=3)
        self.assertEqual([r['ordinal'] for r in selected],list(range(128)))
        self.assertTrue(all(r['input_ids']==[1,97,98] for r in selected))
        self.assertEqual(selected[0]['predicted_tokens'],2)

    def test_short_selected_row_not_replaced(self):
        with self.assertRaisesRegex(ContractBoundary,'TOO_SHORT'):
            select_rows([dict(text=''),dict(text='valid')],Tokenizer(),count=1)

    def test_missing_source_rows_and_nontext_boundary(self):
        with self.assertRaisesRegex(ContractBoundary,'CORPUS_TOO_SHORT'):
            select_rows([dict(text='valid')],Tokenizer(),count=2)
        with self.assertRaisesRegex(ContractBoundary,'TEXT_SCHEMA'):
            select_rows([dict(text=None)],Tokenizer(),count=1)


if __name__=='__main__': unittest.main()
