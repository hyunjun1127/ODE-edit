import unittest
import matplotlib.pyplot as plt
from .revision_labels import LABELS,ORDER,display,labeled_rows
from .revision import regroup_sections,split_method_tables,rates
from .plots import update_figure,setup

class RevisionTests(unittest.TestCase):
    def test_exact_six_labels(self):
        self.assertEqual(len(LABELS),6)
        self.assertEqual(LABELS['MEMIT_ORIGINAL'],'MEMIT_BLUE (L4+L8)')
        self.assertTrue(all('BLUE' in v and 'ORIGINAL' not in v for v in LABELS.values()))
    def test_raw_columns_immutable(self):
        x={'arm':'AlphaEdit_ORIGINAL','RS_numerator':'9888','job':'39283_1'}
        y=labeled_rows([x])[0]
        self.assertEqual({k:y[k] for k in x},x)
        self.assertEqual(y['arm_display_label'],'AlphaEdit_BLUE (L4+L8)')
    def test_family_order(self):
        body='## X\n'+''.join('### '+a+'\nvalue '+a+'\n' for a in [ORDER[0],ORDER[3],ORDER[1],ORDER[4],ORDER[2],ORDER[5]])
        r=regroup_sections(body)
        self.assertLess(r.index('MEMIT_BLUE_L8_ONLY'),r.index('AlphaEdit_BLUE (L4+L8)'))
        self.assertEqual(r.count('value '),6)
    def test_split_family_tables(self):
        r=split_method_tables('|arm|n|\n|---|---|\n|MEMIT_ORIGINAL|1|\n|AlphaEdit_ORIGINAL|2|\n')
        self.assertIn('**MEMIT 계열**',r);self.assertIn('**AlphaEdit 계열**',r)
        self.assertEqual(r.count('|MEMIT_ORIGINAL|1|'),1)
    def test_reference_numerator_denominator(self):
        r={t+'_'+k:v for t in ['RS','PS','NS'] for k,v in [('num','71'),('den','1000'),('rate','.071')]}
        self.assertEqual(rates(r)['RS'],'71/1000 (7.100%)')
    def test_weights_no_reference_line_or_ambiguous_legend(self):
        setup();rows=[{'arm':a,'layer':'4' if 'L8_ONLY' not in a else '8','update_norm':'1'} for a in ORDER]
        f=update_figure(rows,LABELS)
        self.assertEqual(f._suptitle.get_text(),'Layer-wise Update Magnitude')
        self.assertTrue(all(len(ax.lines)==0 for ax in f.axes))
        self.assertTrue(all('BLUE' in t.get_text() for ax in f.axes for t in ax.get_legend().get_texts()))
        plt.close(f)

if __name__=='__main__':unittest.main()
