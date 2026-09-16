import tempfile
from pathlib import Path
import unittest
from .common import label,label_text,mdtable,with_toc
from .checks import render,check_scope,check_views

class Tests(unittest.TestCase):
    def test_native_distinction(self):
        self.assertEqual(label('BASE_ALPHAEDIT'),'BASE_ALPHAEDIT_NATIVE')
        self.assertEqual(label('AlphaEdit_ORIGINAL'),'AlphaEdit_BLUE(L4+L8)')
        self.assertNotEqual(label('BASE_MEMIT'),label('MEMIT_ORIGINAL'))
    def test_singleton_physical_layer(self):
        for f in ('AlphaEdit','MEMIT'):
            for l in range(4,9):self.assertEqual(label(f'{f}_L{l}_ONLY'),f'{f}_BLUE(L{l}-only)')
    def test_out_of_scope_rejected(self):
        for bad in ('CAP1','REFIT4','BG-1','NORM_ONLY'):
            with self.assertRaises(ValueError):label(bad)
    def test_text_mapping(self):
        self.assertEqual(label_text('AlphaEdit_L4_ONLY_TO_CAKE'),'AlphaEdit_BLUE(L4-only)_TO_CAKE')
    def test_table_escaped_pipe(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.md';p.write_text(mdtable(['a','b'],[['x|y','z']]))
            self.assertEqual(render(p).tables[0][1],['x|y','z'])
    def test_malformed_width_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.md';p.write_text('| a | b |\n|---|---|\n| 1 | 2 | 3 |\n')
            with self.assertRaises(AssertionError):render(p)
    def test_toc_targets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.md';p.write_text(with_toc('# X\n\n## A\n\na\n\n## B\n\nb'))
            self.assertEqual(render(p).ids,{'section-1','section-2'})
    def test_scope(self):self.assertEqual(check_scope()['alpha_arms'],4)
    def test_all_compact_views(self):self.assertGreater(check_views(),60)

if __name__=='__main__':unittest.main()
