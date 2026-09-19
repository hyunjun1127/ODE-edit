import tempfile
import unittest
from pathlib import Path
from .review_publication import tables,links,canonical_bytes,RenderCheck

class PublicationTests(unittest.TestCase):
    def test_table_unicode_and_escaped_pipe(self):
        t=tables('| 가 | 나 |\n| --- | --- |\n| x\\|y | z |\n')
        self.assertEqual(len(t),1);self.assertEqual(len(t[0][2]),2)

    def test_bad_columns_and_unclosed_fence(self):
        with self.assertRaises(ValueError):tables('| a | b |\n| --- | --- |\n| c |\n')
        with self.assertRaises(ValueError):tables('```python\nx=1\n')

    def test_missing_link(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):links('[x](missing.csv)',Path(d))
            self.assertEqual(links('[x](new.json)',Path(d),('new.json',)),['new.json'])

    def test_canonical_and_render_counts(self):
        self.assertEqual(canonical_bytes({'b':1,'a':2}),canonical_bytes({'a':2,'b':1}))
        r=RenderCheck();r.feed('<table><tr><th>가</th><td>1</td></tr></table><img src="x">')
        self.assertEqual((r.tables,r.cells,r.images),(1,2,1))

if __name__=='__main__':unittest.main()

