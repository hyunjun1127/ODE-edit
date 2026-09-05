"""Small CPU fixtures: scope, paired metrics, seals, deterministic plots."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from .. import round0_analysis as prior
from ..sequential_analysis import (
    ARMS,CELLS,KINDS,AnalysisBoundary,canonical_hash,canonical_v2_reducer,
    deep_identity,stats,summarize_requests,write_csv,
)
from ..sequential_plots import ORDER,weight_figure,STYLE
import matplotlib.pyplot as plt


class AnalysisTests(unittest.TestCase):
    def test_linear_quantiles(self):
        x=stats([1,2,3,4]);self.assertEqual(x['median'],2.5);self.assertAlmostEqual(x['p90'],3.7)
        self.assertEqual(x['q25'],1.75);self.assertEqual(x['q75'],3.25)

    def test_empty_is_not_zero(self):
        x=stats([]);self.assertEqual(x['n'],0);self.assertIsNone(x['mean'])

    def test_receipt_vs_external_reference(self):
        x={'schema':'toy','value':1};x['identity_sha256']=canonical_hash(x)
        self.assertEqual(deep_identity({'wrapper':x,'reference':{'identity_sha256':'not-own-hash'}}),1)
        x['value']=2
        with self.assertRaises(AnalysisBoundary):deep_identity(x)

    def test_nonfinite_rejected(self):
        with self.assertRaises(AnalysisBoundary):deep_identity({'x':float('nan')})

    def test_reducer_globals_restored(self):
        before=(prior.KIND_COUNTS,prior.KIND_ORDER,prior.EXPECTED_EVALUATION_ROWS)
        with canonical_v2_reducer():
            self.assertEqual(prior.EXPECTED_EVALUATION_ROWS,2600);self.assertEqual(prior.KIND_COUNTS,KINDS)
        self.assertEqual((prior.KIND_COUNTS,prior.KIND_ORDER,prior.EXPECTED_EVALUATION_ROWS),before)

    def test_strict_direction_tie(self):
        new=np.array([1.,2.,1.]);true=np.array([2.,1.,1.])
        self.assertEqual((new<true).tolist(),[True,False,False])
        self.assertEqual((true<new).tolist(),[False,True,False])

    def test_request_prompt_denominators(self):
        f=pd.DataFrame({'rewrite_success':[1,0],'rephrase_prompt_success_count':[2,1],'rephrase_strict_success':[1,0],
           'canonical_ns_numerator':[8,5],'rewrite_target_new_accuracy':[1,0],'rephrase_target_new_accuracy_count':[1,1],
           'locality_target_true_accuracy_count':[2,1],'locality_prediction_preservation_denominator':[np.nan,np.nan]})
        x=summarize_requests(f)
        self.assertEqual((x['RS_num'],x['RS_den']),(1,2));self.assertEqual((x['PS_num'],x['PS_den']),(3,4))
        self.assertEqual((x['NS_num'],x['NS_den']),(13,20));self.assertEqual(x['PS_strict_rate'],.5)

    def test_deterministic_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.csv.gz',Path(tmp)/'b.csv.gz';f=pd.DataFrame({'v':[1.,3.]})
            write_csv(a,f);write_csv(b,f);self.assertEqual(a.read_bytes(),b.read_bytes())
            with self.assertRaises(FileExistsError):write_csv(a,f)

    def test_duplicate_evaluation_rejected(self):
        e={'rows':[{'request_sha256':'x','case_id':1,'kind':'rewrite_target_new','prompt_index':0,
            'nll':1.,'all_tokens_correct':False,'correct_token_count':0,'target_token_count':1,
            'input_identity_sha256':'x','observation_identity_sha256':'x'}]*2600}
        with canonical_v2_reducer(),self.assertRaises(AnalysisBoundary):prior._validate_evaluation(e,['x']*100,endpoint=False)

    def test_weight_title_order_no_reference_line(self):
        plt.rcParams.update(STYLE)
        frame=pd.DataFrame([{'cell':c,'arm':a,'layer':l,'update_magnitude':float(l)} for c in CELLS for a in ARMS for l in range(4,9)])
        fig=weight_figure(frame)
        self.assertEqual(ORDER,('LM','LA','QM','QA'))
        self.assertEqual(fig._suptitle.get_text(),'Layer-wise Update Magnitude')
        for ax in fig.axes:
            self.assertEqual(len(ax.lines),0);self.assertEqual(len(ax.patches),25)
            self.assertFalse(any('bars:' in x.get_text().lower() for x in ax.texts))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'plot.png';fig.savefig(f);self.assertGreater(f.stat().st_size,1000)
        plt.close(fig)


if __name__=='__main__':unittest.main()
