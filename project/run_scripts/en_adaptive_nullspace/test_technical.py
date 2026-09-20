"""CPU evidence/transaction tests; do not assert real-model numerical PASS."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from . import technical as t


class TechnicalTests(unittest.TestCase):
    def test_reused_targets_satisfy_native_call_count_contract(self):
        fitter=t._TargetsOnlyFitter.__new__(t._TargetsOnlyFitter)
        fitter.targets=torch.arange(12,dtype=torch.float32).reshape(3,4)
        fitter.position=0
        counts={}
        def fake_fit(model):
            return compute_z(model)
        model=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]))
        with patch.object(t.NativeSingletonFitter,'_functions',return_value=(fake_fit,None)):
            fit,_=fitter._functions(counts,None)
        actual=torch.stack([fit(model) for _ in range(4)],dim=1)
        self.assertTrue(torch.equal(actual,fitter.targets))
        self.assertEqual(counts,{'compute_z':4,'reused_T0_z':4})
        with self.assertRaisesRegex(ValueError,'OVERCONSUMED'):fit(model)
        self.assertEqual(counts['compute_z'],4)

    def test_panel_hash_order_and_no_outcome_dependency(self):
        rows=[dict(case_id=i,score=1) for i in range(10)]
        first=t.fixed_indices(rows,lambda x:x['case_id'])
        for r in rows:r['score']=-1
        self.assertEqual(first,t.fixed_indices(rows,lambda x:x['case_id']))
        self.assertEqual(len(first),4)
        self.assertEqual(first,sorted(range(10),key=lambda i:t.digest(i))[:4])

    def test_duplicate_identity_refused(self):
        with self.assertRaises(ValueError):t.fixed_indices([1,1,2,3],lambda x:x)

    def test_nonfinite_or_shape_residual_refused(self):
        with self.assertRaises(ValueError):t.residual(torch.zeros(2),torch.ones(3))
        with self.assertRaises(ValueError):t.residual(torch.zeros(2),torch.tensor([1.,float('nan')]))

    def test_zero_reference_does_not_invent_relative_pass(self):
        r=t.residual(torch.zeros(2),torch.ones(2))
        self.assertTrue(r['reference_zero']);self.assertIsNone(r['relative_l2'])
        self.assertNotIn('pass',r)

    def test_tensor_persistence_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TypeError):t._write(Path(temp)/'a.json',{'weight':torch.zeros(2)})

    def test_failure_restores_actual_weight_and_rng(self):
        w=torch.nn.Parameter(torch.zeros(2,2),requires_grad=False)
        def install(value):
            with torch.no_grad():w.copy_(value)
        records=[dict(case_id=i,requested_rewrite={'prompt':'{} is','subject':str(i),'target_new':{'str':'new'},'target_true':{'str':'old'}}) for i in range(4)]
        rt=SimpleNamespace(W=w,W0=w.detach().clone(),records=records,install=install,
                           model=None,tok=None,module=None,hp=None,context=None)
        obj=SimpleNamespace(store=SimpleNamespace(indices=lambda role:range(4)),
                            ref=SimpleNamespace(_capsule_shas=list('abcd')),rebind=lambda weight:None)
        before=torch.get_rng_state().clone()
        def fail(*args,**kwargs):
            with torch.no_grad():w.fill_(7)
            torch.rand(3)
            raise RuntimeError('synthetic technical failure')
        with tempfile.TemporaryDirectory() as temp,patch.object(t,'compare_native_z_paths',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic'):t.run_t0(rt,obj,None,temp)
            self.assertTrue(torch.equal(w,rt.W0))
            self.assertTrue(torch.equal(torch.get_rng_state(),before))
            receipt=json.loads((Path(temp)/'t0-final-state.json').read_text())
            self.assertEqual(receipt['status'],'FAILED')
            self.assertTrue(receipt['entry_weight_exact_restored'])
            self.assertFalse(any(p.suffix=='.pt' for p in Path(temp).iterdir()))


if __name__=='__main__':unittest.main()
