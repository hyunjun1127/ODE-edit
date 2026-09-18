"""CPU boundary/linear-algebra tests; not actual model validation."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from .config import NUMERIC,require_scope
from . import factors,controller
from .transaction import commit

def lock():
    return dict(max_batches=1,sequential_authorized=False,arms=['N4','BPCW512'],batch_size=100,
        numeric={**NUMERIC,'pair_rows':list(NUMERIC['pair_rows'])},native_fit_reuse=False,
        native_policy='FRESH_SAME_HOST_SHARED_ONCE',task_gpu_cap=1,project_gpu_cap=2)

class Boundaries(unittest.TestCase):
    def test_scope(self):
        require_scope(lock())
        for key,value in [('max_batches',10),('sequential_authorized',True),('arms',['N4']),
                          ('native_fit_reuse',True),('task_gpu_cap',2),('batch_size',1000)]:
            bad=lock();bad[key]=value
            with self.assertRaises(ValueError):require_scope(bad)
        with self.assertRaises(ValueError):require_scope(lock(),batch=2)

    def test_numeric_drift(self):
        bad=lock();bad['numeric']['gap_reserve_max']=.1
        with self.assertRaises(ValueError):require_scope(bad)

    def test_duplicate_before_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'checkpoint.pt').touch()
            with self.assertRaisesRegex(ValueError,'DUPLICATE'):
                commit(SimpleNamespace(lock=lock()),'N4',None,None,None,tmp)

    def test_all_retained_rows(self):
        def scan(c):return dict(documents=[dict(index=i,source_row_id=f'r{i}',
            worst=dict(position=10,target=1,competitor=c,kappa=.0001)) for i in range(512)])
        first=controller.expose(scan(2),[]);second=controller.expose(scan(3),first)
        self.assertEqual(len(first),512);self.assertEqual(len(second),1024)
        self.assertTrue({x['pair_id'] for x in first}<={x['pair_id'] for x in second})
        self.assertEqual(controller.expose(scan(2),first),first)

    def test_factor_gram_positive_reconstruction(self):
        torch.manual_seed(9)
        with tempfile.TemporaryDirectory() as tmp:
            files=[];dense=[]
            for i,length in enumerate([2,4,3,5,2]):
                a=torch.randn(7,length);u=torch.randn(11,length,dtype=torch.float64)
                path=Path(tmp,f'{i}.pt');torch.save(dict(A=a,U=u),path);files.append(path)
                dense.append(a.double()@u.T)
            expected=torch.stack(dense).flatten(1);expected=expected@expected.T
            actual=factors.gram(files,'cpu',tile=2)
            np.testing.assert_allclose(actual,expected.numpy(),rtol=1e-12,atol=1e-12)
            alpha=np.array([.1,0,.7,.3,0])
            actualD=factors.reconstruct(files,alpha,(7,11),'cpu')
            torch.testing.assert_close(actualD,sum(a*d for a,d in zip(alpha,dense)),rtol=1e-12,atol=1e-12)

    def test_raw_rhs_center_not_projected(self):
        class Space:
            basis=np.eye(3);blocked=np.array([[0.],[0.],[1.]])
            status='RESOLVED'
        A=torch.tensor([[2.,1.],[1.,3.]])
        K=torch.tensor([[1.,2.],[0.,1.],[4.,3.]])
        oracle=SimpleNamespace(device='cpu',pair_factor=lambda *a:(A,K,5.))
        WN=torch.zeros(2,3);center=torch.tensor([[0.,0.,.3],[0.,0.,.2]])
        pair=dict(index=0,pair_id='r|p1|c2',position=1,target=1,competitor=2,kappa=.1)
        scan=dict(documents=[dict(retained_pairs=[dict(pair_id=pair['pair_id'],mu=-.2)])])
        with tempfile.TemporaryDirectory() as tmp:
            files,b=factors.build(oracle,Space(),center,WN,[pair],scan,tmp)
            expected=.2+float(((A.double()@K.double().T)*center.double()).sum())
            self.assertAlmostEqual(b[0],expected)
            self.assertNotAlmostEqual(b[0],.2)

    def test_native_safe_no_qp(self):
        W=torch.zeros(2,3)
        ref=SimpleNamespace(scan=lambda *a:dict(all_choices=True,token_flips=0))
        with tempfile.TemporaryDirectory() as tmp,patch.object(factors,'build',side_effect=AssertionError('must skip')):
            selected,ideal,result,_=controller.run(None,W,ref,None,None,None,None,None,None,1e-4,tmp)
            self.assertEqual(result['status'],'NATIVE_ALREADY_FEASIBLE')
            self.assertEqual(result['pair_scalar_gradients'],0)
            self.assertEqual(result['current_guard_count'],0)
            self.assertTrue(torch.equal(selected,W));self.assertFalse(result['native_fallback'])

    def test_firewall_and_runner(self):
        root=Path(__file__).parent
        text=(root/'reference_inputs.py').read_text()
        self.assertIn("r['scope']=='Wiki128'",text)
        self.assertIn('native_current_or_future_access=False',text)
        run=(root/'runner.py').read_text()
        self.assertLess(run.index("create_json(out/'endpoint-seals.json'"),run.index('observer=CanonicalObserver'))
        self.assertNotIn('sbatch',run);self.assertNotIn('range(10)',run)
        self.assertIn('if a.max_batches!=1',run)
        for path in root.glob('*.py'):ast.parse(path.read_text())

if __name__=='__main__':unittest.main()
