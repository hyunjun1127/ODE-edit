import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import review_package,review_plots
from .review_provenance import csvwrite,sha
class PackageTests(unittest.TestCase):
 def test_rehash_detects_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a.md').write_text('fixture');review_package.seal(p);review_package.verify(p)
   (p/'a.md').write_text('mutated')
   with self.assertRaises(AssertionError):review_package.verify(p)
 def test_extra_member_detected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a.md').write_text('fixture');review_package.seal(p);(p/'extra.csv').write_text('x')
   with self.assertRaises(AssertionError):review_package.verify(p)
 def test_deterministic_plot_order_and_weight_title(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);arms=review_plots.ARMS
   csvwrite(p/'first-core-table.csv',[dict(state=a,**{g+'_'+m+'_rate':.7 for g in ['current','historical'] for m in ['RS','PS','NS']}) for a in arms])
   csvwrite(p/'nll-distributions.csv',[dict(state=a,panel=g,metric=m,field='true_nll' if m=='NS' else 'new_nll',population='ALL',aggregation_unit='PROMPT',median=1,p90=2,p99=3) for a in arms for g in ['current','historical'] for m in ['RS','PS','NS']])
   csvwrite(p/'paired-transitions.csv',[dict(before='N4',after=a,panel=g,metric='NS',population='ALL',lost=2,gained=3) for a in arms[1:] for g in ['current','historical']])
   csvwrite(p/'layer-action.csv',[dict(arm=a,layer=l,norm=1) for a in arms for l in [4,8]])
   csvwrite(p/'compute-ledger.csv',[dict(arm=a,policy_instrumented_online_seconds=1,policy_M8_setup_seconds=2) for a in arms])
   captured=[];orig=review_plots.plt.close
   def check(fig=None):
    if hasattr(fig,'axes'):
     for ax in fig.axes:
      if ax.get_title()=='Layer-wise Update Magnitude':
       self.assertEqual(len(ax.lines),0);self.assertEqual([x.get_text() for x in ax.get_xticklabels()],arms);captured.append(True)
    orig(fig)
   with patch.object(review_plots.plt,'close',check):first=review_plots.run(p)
   second=review_plots.run(p)
   self.assertTrue(captured);self.assertEqual(len(first),5);self.assertEqual([r['sha256'] for r in first],[r['sha256'] for r in second])
   self.assertTrue(all((p/r['path']).exists() for r in first))
if __name__=='__main__':unittest.main()
