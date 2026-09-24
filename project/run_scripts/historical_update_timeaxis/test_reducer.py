"""Small synthetic CPU join/plot fixture, never published as experimental data."""
import copy
import tempfile
import unittest
from pathlib import Path
from .common import ROOT,FAMILIES
from .reduce import contribution,pair,tables

def score(case,panel,idx,margin):
    return dict(schema_version=1,case_id=case,panel=panel,prompt_index=idx,target_version='Q_test',prompt_token_hash='a'*64,target_token_hash='b'*64,competitor_token_hash='c'*64,
        evaluator_signature='fixture-only',valid=True,missing_reason=None,state_weight_hash='d'*64,margin=margin,target_nll=2.-margin,competitor_nll=2.,target_strict_tf=False,
        target_token_correct=0,target_token_count=2,competitor_strict_tf=True,competitor_token_correct=2,competitor_token_count=2)

class Reducer(unittest.TestCase):
    def fixture(self):
        cc=[];pp=[]
        for family in FAMILIES:
            for case in (1,2):
                f=dict(case_id=str(case),birth_batch='11',same_batch_conflict='False',first_later_conflict_batch='',subject_relation_group=str(case),repeated_group='False')
                for panel,idx in [('rewrite',0),('paraphrase',0),('paraphrase',1)]:
                    for t in (20,30):
                        c=dict(cell_id=family+str(t),family=family,cohort_id='U010_020',update_start='10',anchor='20',eval_t=str(t))
                        scores={k:score(case,panel,idx,v) for k,v in dict(M_t=1. if t==20 else -.3,B_t=.2,M_b=1.,B_b=.2).items()}
                        cc.append(contribution(c,f,scores))
                    c=dict(pair_id=family+'__fixture',family=family,past_cohort='U010_020',future_cohort='U050_060',selection='low_competitor_exposure')
                    pp.append(pair(c,f,{k:score(case,panel,idx,v) for k,v in dict(M=.4,minus_U=.2,minus_V=.3,minus_UV=.15).items()}))
        return cc,pp

    def test_join_arithmetic_and_denominator(self):
        cc,pp=self.fixture();self.assertEqual(len(cc),24);self.assertEqual(len(pp),12)
        for r in cc:self.assertLessEqual(abs(r['delta_M']-r['delta_B']-r['delta_C']),1e-10)
        for r in pp:self.assertAlmostEqual(r['interaction'],.05)
    def test_token_mismatch_is_failure(self):
        c=dict(cell_id='x',family=FAMILIES[0],cohort_id='U010_020',update_start='10',anchor='20',eval_t='30')
        f=dict(birth_batch='11',same_batch_conflict='False',first_later_conflict_batch='',subject_relation_group='test',repeated_group='False')
        ss={k:score(1,'rewrite',0,1.) for k in ('M_t','B_t','M_b','B_b')};ss['B_t']['target_token_hash']='e'*64
        with self.assertRaises(AssertionError):contribution(c,f,ss)
    def test_small_plot_path(self):
        cc,pp=self.fixture()
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_CPU_FIXTURE_',dir=ROOT) as d:
            summary=tables(Path(d),cc,pp)
            self.assertEqual(summary['contribution_rows'],24);self.assertEqual(len(list(Path(d).glob('*.png'))),5)
    def test_completed_after_report_and_inventory(self):
        text=(Path(__file__).parent/'reduce.py').read_text()
        final=text.index("save(out/'terminal.json',dict(status=completion(nw)")
        self.assertLess(text.index("(out/'report-ko.md').write_text(text)"),final)
        self.assertLess(text.index("save(out/'artifact-index.json'"),final)

if __name__=='__main__':unittest.main()
