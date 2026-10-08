import json
from pathlib import Path
import tempfile
import unittest
from .monitor import numbers,expected_row,write

class OperationsTests(unittest.TestCase):
    def test_numeric_privacy_and_RPN_summary(self):
        x={'prompt':'private','key':'secret','case_id':44,'W':[1,2],
            'R':{'numerator':1,'denominator':2},'P':[[1]],'norm':[1.,2.],'status':'raw'}
        y=numbers(x)
        self.assertEqual(y,{'R':{'numerator':1,'denominator':2},'norm':[1.,2.]})
    def test_exact_journal_step_and_partial_line(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);rows=[{'op':'log','values':{'fit/loss':2},'step':None},
                {'op':'log','values':{'batch':2,'edits':200,'current/post/N/count':1000},'step':None}]
            (p/'accepted-scalars.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{')
            self.assertEqual(expected_row(p,2)['step'],1)
            self.assertIsNone(expected_row(p,1))
    def test_upload_size_guard(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):write(Path(t)/'large.json',{'x':'a'*600000})

if __name__=='__main__':unittest.main()
