"""Bounded CPU policy/routing fixtures, not GPU numerical certification."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from .common import ROOT,FAMILIES,save,read
from .fidelity import *
from .worker import Run
from .reduce import worker_completion


def raw():
    result={}
    for kind in ('rewrite','rephrase'):
        for side in ('new','true'):
            label=11 if side=='new' else 12
            result[kind+'_target_'+side]=[dict(case_id=1,kind=kind+'_target_'+side,prompt_index=0,prompt='fixture',
                target=side,target_token_ids=[label],token_predictions=[label],token_correct=[True],all_tokens_correct=True,
                nll=1. if side=='new' else 2.)]
    return result


class RecordOnly(unittest.TestCase):
    def test_previous_failure_and_large_finite_are_warnings(self):
        for difference in (.0005242824554443359,50.):
            a=raw();b=copy.deepcopy(a);b['rewrite_target_new'][0]['nll']+=difference
            d=compare_raw(a,b)
            self.assertGreater(d['warning_count'],0);self.assertAlmostEqual(d['max_nll'],difference)
            self.assertEqual(d['numerical_certification'],'NOT_ESTABLISHED')
            self.assertEqual(d['nll_rows'][0]['limit'],.00025)

    def test_nonboundary_flip_records_location(self):
        a=raw();b=copy.deepcopy(a);b['rewrite_target_new'][0]['nll']=3.
        d=compare_raw(a,b);self.assertIn('NONBOUNDARY_FLIP',[w['code'] for w in d['warnings']])
        self.assertEqual(d['boundary_flips'][0]['case_id'],1)

    def test_historical_large_difference_and_flip(self):
        a=raw();n=a['rewrite_target_new'][0];t=a['rewrite_target_true'][0]
        old=dict(identity='x',new_nll=10.,true_nll=1.,margin=-9.,success=False)
        r=historical_row(n,t,old,dict(pair_identity='x'))
        self.assertEqual(r['warnings'],['HISTORICAL_FIDELITY','ORIGINAL_NONBOUNDARY_FLIP'])
        with self.assertRaisesRegex(AssertionError,'IDENTITY'):historical_row(n,t,old,dict(pair_identity='wrong'))

    def test_identity_nonfinite_duplicate_still_block(self):
        for field,value in (('case_id',2),('target_token_ids',[9]),('nll',float('nan')),('nll',float('inf'))):
            a=raw();b=copy.deepcopy(a);b['rewrite_target_new'][0][field]=value
            with self.assertRaises(AssertionError):compare_raw(a,b)
        a=raw();a['rewrite_target_new']*=2
        with self.assertRaises(AssertionError):compare_raw(a,a)

    def test_actual_raw_saved_before_nonfinite_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            r=Run.__new__(Run);r.identity={'fixture':True};r.policy={'numerical_certification':CERTIFICATION}
            r.pairs=lambda ids:{'fixture':[]}
            def evaluate(pairs,mb,evidence_sink):
                values=[{'nll':float('nan')}];evidence_sink(values);raise AssertionError('NONFINITE_NLL')
            r.b=SimpleNamespace(evaluate=evaluate)
            with self.assertRaises(AssertionError):r.raw([1],evidence=Path(d))
            self.assertEqual(read(Path(d)/'fixture.json')['rows'][0]['nll'],{'nonfinite_float':'nan'})

    def test_warning_peer_and_collector_join(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root=Path(d);identity={'waiver_sha256':'x'}
            common=dict(identity=identity,numerical_certification=CERTIFICATION,numerical_fidelity_policy=POLICY,waiver_sha256='x')
            for f in FAMILIES:
                for s in ('T1','T2P','T2F','T3B'):
                    save(root/f/s/'PASS.json',dict(**common,status='PASS',pass_semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',numerical_diagnostics={'warning_count':100}))
                save(root/f/'terminal.json',dict(**common,status='COMPLETED_WITH_NUMERICAL_WARNINGS',numerical_diagnostics={'warning_count':100}))
            r=Run.__new__(Run);r.root=root;r.identity=identity;r.family=FAMILIES[0];r.lock={'barrier_timeout_seconds':1}
            r.barrier('T1')
            ready,receipts=worker_completion(root,identity);self.assertTrue(ready)
            self.assertEqual(sum(x['numerical_diagnostics']['warning_count'] for x in receipts),200)
            save(root/FAMILIES[1]/'FAILURE.json',{'error':'fixture structural failure'})
            self.assertFalse(worker_completion(root,identity)[0])
            with self.assertRaises(RuntimeError):r.barrier('T1')

    def test_wrong_policy_join_blocks(self):
        with self.assertRaises(AssertionError):check_gate(dict(status='PASS',identity={'waiver_sha256':'wrong'}),{'waiver_sha256':'right'})

    def test_no_postrelease_scheduler_query(self):
        text=(Path(__file__).parent/'launch.py').read_text().split("for job in list(jobs.values())+[collector]:")[-1]
        self.assertNotIn("command(['squeue'",text);self.assertNotIn("command(['sacct'",text)
        self.assertIn('INITIAL_NOT_OBSERVED',text)

    def test_persist_before_finite_and_numeric(self):
        source=(Path(__file__).parent/'backend.py').read_text()
        self.assertLess(source.index('evidence_sink(result)'),source.index("assert all(__import__('math').isfinite"))
        production=(Path(__file__).parent/'worker.py').read_text()
        for name in ('NLL_FIDELITY','MARGIN_FIDELITY','NONBOUNDARY_FLIP','HISTORICAL_FIDELITY'):
            self.assertNotIn(name,production)

if __name__=='__main__':unittest.main()
