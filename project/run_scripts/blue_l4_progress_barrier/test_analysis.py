import unittest
from .analysis import metric_check,paired,summaries

class ReducerTests(unittest.TestCase):
    def data(self,flip=False):
        return [dict(panel='Current100',metric='PS',case_id=i,prompt_index=j,identity=f'{i}-{j}',
          new_nll=2. if flip else 1.,true_nll=1. if flip else 2.,margin=-1. if flip else 1.,success=not flip,
          new_strict=not flip,true_strict=flip,new_token_correct=0 if flip else 2,new_token_count=2,
          true_token_correct=2 if flip else 0,true_token_count=2) for i in range(100) for j in range(2)]
    def test_denominator(self):
        x=self.data();metric_check(x);r=summaries(x,'Middle','EP',8)[0]
        self.assertEqual((r['numerator'],r['denominator']),(200,200))
        self.assertEqual(r['new_token_denominator'],400)
    def test_paired_cluster(self):
        r=paired(self.data(),self.data(True),'Middle','EP−N')[0]
        self.assertEqual((r['requests'],r['prompts']),(100,200));self.assertEqual(r['delta'],100)
        self.assertEqual(r['ci_low'],100);self.assertEqual(r['failure_to_success'],200)
    def test_identity_mismatch(self):
        x=self.data();y=self.data();y[0]['identity']='wrong'
        with self.assertRaises(AssertionError):paired(x,y,'Middle','EP−N')
    def test_tie_is_failure(self):
        x=self.data();x[0]['new_nll']=x[0]['true_nll']
        with self.assertRaises(AssertionError):metric_check(x)
        x[0]['success']=False;x[0]['margin']=0.;metric_check(x)
    def test_terminal_curve_inventory(self):
        from project.run_scripts.single_layer_cumulative_risk.panels import curve_rows
        values=[];neighbors={}
        for pn,panel in enumerate(('Current100','Fixed100','Past100')):
            for i in range(100):
                case=pn*100+i;neighbors[str(case)]=[0,1]
                for metric,count in [('RS',1),('PS',2),('NS',10)]:
                    values.extend(dict(panel=panel,metric=metric,case_id=case,prompt_index=j) for j in range(count))
        self.assertEqual(len(values),3900)
        curve=curve_rows(values,dict(neighbors=neighbors));self.assertEqual(len(curve),1100)
        for panel in ('Current100','Fixed100','Past100'):
            self.assertEqual(sum(r['panel']==panel and r['metric']=='NS' for r in curve),200)

if __name__=='__main__':unittest.main()
