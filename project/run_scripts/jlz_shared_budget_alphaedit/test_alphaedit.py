"""Independent native-expression fixtures and scoped scheduler/restore failure checks."""
import json,tempfile,unittest,inspect,ast
from pathlib import Path
from unittest.mock import patch
import torch
from .writer import solve
from .native_reference import native_update
from .common import ROOT,NONCE,TASK
from . import submit
NATIVE='/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream/AlphaEdit/AlphaEdit_main.py'

class AlphaEdit(unittest.TestCase):
    def test_noncommuting_native_order_and_history(self):
        torch.manual_seed(611);Q,_=torch.linalg.qr(torch.randn(13,13));Pi=Q[:,:7]@Q[:,:7].T
        K=torch.randn(13,4);R=torch.randn(5,4);h=torch.randn(13,3);H=h@h.T
        self.assertGreater(float((Pi@(K@K.T+H)-(K@K.T+H)@Pi).norm()),1.)
        for history in (torch.zeros_like(H),H):
            U,res=solve(Pi,history,K,R,'cpu');ref=native_update(Pi,history,K,R,NATIVE)
            self.assertEqual(U.dtype,torch.float32);torch.testing.assert_close(U,ref,atol=0,rtol=0)
            self.assertTrue(torch.isfinite(torch.tensor(res)))
        self.assertGreater(float((solve(Pi,H,K,R,'cpu')[0]-solve(Pi,torch.zeros_like(H),K,R,'cpu')[0]).norm()),1e-3)
        zero,_=solve(Pi,H,K,torch.zeros_like(R),'cpu');self.assertEqual(float(zero.norm()),0.)
        with self.assertRaises(Exception):solve(Pi.double(),H,K,R,'cpu')

    def test_metadata_schema_no_false_ridge(self):
        from .telemetry import Events
        from .events import metadata
        class A:sites=(4,5,6,7,8);dims={i:(4096,14336) for i in sites}
        c=dict(model='model',stream='stream',packing=[],native_input_alignment={'sha256':'x'},profile={'anchor_layer':8})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'events';metadata(Events(p,'TEST'),A(),c,True,'source')
            r=json.loads(p.read_text())['payload'];self.assertEqual(r['native_profile']['lambda_covariance'],0.)
            self.assertIn('FP32_solve',r['native_profile']['cast_add_order'])
        self.assertIn('alphaedit',TASK);self.assertIn('SH3',NONCE)

    def test_independent_metrics_ties_and_strict(self):
        from .reducer import reduce_rows
        rows=[dict(identity=str(i),kind=kind,true_nll=t,new_nll=n,true_token_count=2,true_token_correct=tc,
            new_token_count=3,new_token_correct=nc,true_strict=tc==2,new_strict=nc==3)
            for i,(kind,t,n,tc,nc) in enumerate([('R',2.,1.,1,3),('R',1.,1.,2,0),('N',1.,2.,2,1)])]
        r=reduce_rows(rows);self.assertEqual(r['R']['numerator'],1);self.assertEqual(r['R']['token_micro'],.5)
        self.assertEqual(r['N']['strict_numerator'],1);self.assertEqual(r['N']['numerator'],1)
        with self.assertRaises(Exception):reduce_rows(rows+rows[:1])

    def test_namespace_and_launch_dag(self):
        resources=dict(host_mib=59392,collector_host_mib=24576,wall='7-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00')
        p=Path('/example/attempt/source');script=submit.launcher(p,'abc','project.run_scripts.jlz_shared_budget_alphaedit.run',['--attempt','/example/attempt'])
        self.assertIn(str(submit.OVERLAY),script);self.assertIn('alphaedit.run',script)
        for name in submit.NAMES:
            a=submit.arguments(name,'afterany:58179',p.parent,resources)
            self.assertIn('--export=NONE',a);self.assertIn('--no-requeue',a)
            self.assertIn('--dependency=afterany:58179',a)
            self.assertEqual('--gres=gpu:1' in a,name!='collector')
            self.assertIn('--mem='+('24576' if name=='collector' else '59392')+'M',a)
        self.assertEqual(submit.NAMES,('pilot','V12_ALPHAEDIT','collector'))
        from . import run
        source=inspect.getsource(run.main)
        self.assertLess(source.index('PILOT_NOT_QUALIFIED'),source.index('setup(config,out)'))

    def test_held_inspection_fails_before_release(self):
        with patch.object(submit,'command',return_value='JobId=999 JobName=wrong UserId=other(2) JobState=PENDING Reason=JobHeldUser'):
            with self.assertRaises(Exception):submit.inspect('999','pilot','',[],Path('/example'),{})

    def test_no_checkpoint_or_ridge_solve_in_main_writer(self):
        from . import writer
        source=inspect.getsource(writer);self.assertNotIn('15000',source);self.assertNotIn('build_prior_from_npz',source)
        for name in ['writer','run','probe','optimize']:
            s=(ROOT/'project/run_scripts/jlz_shared_budget_alphaedit'/f'{name}.py').read_text()
            self.assertNotIn('torch.save(',s);self.assertNotIn('np.save',s)
        self.assertNotIn('cholesky',inspect.getsource(solve).lower());self.assertNotIn('.double()',inspect.getsource(solve))

if __name__=='__main__':unittest.main()
