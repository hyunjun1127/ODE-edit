import json
from pathlib import Path
import tempfile
import unittest
import torch
from .backend import groups, rotation
from .common import save, read, INSTRUCTION
from .stages import ORDER


class ContractTests(unittest.TestCase):
    def test_atomic_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';save(p,{'status':'PASS'})
            with self.assertRaises(FileExistsError): save(p,{'status':'FAIL'})
            self.assertEqual(read(p),{'status':'PASS'})

    def test_nonfinite_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save(Path(d)/'x',{'value':float('nan')})
            self.assertFalse((Path(d)/'x').exists())

    def test_chunks_keep_position_and_target_order(self):
        rows=[dict(panel='N',kind='N',label='new',i=i) for i in range(35)]
        out=groups(rows)
        self.assertEqual(list(map(len,out)),[16,16,3])
        self.assertEqual([r for g in out for r in g],rows)
        general=[dict(panel='G',kind='GENERAL',label='natural',i=i) for i in range(10)]
        self.assertEqual(list(map(len,groups(general))),[4,4,2])

    def test_factorial_cross_sign_and_receiver(self):
        torch.manual_seed(123)
        w0=torch.randn(7,11,dtype=torch.float64);ws=torch.randn_like(w0);wt=torch.randn_like(w0)
        k0=torch.randn(2,3,11,dtype=torch.float64);k1=torch.randn_like(k0)
        a=ws-w0;wr=wt-a
        v00=k0@wr.T;v10=k0@wt.T;v01=k1@wr.T;v11=k1@wt.T
        action=(k1-k0)@a.T
        torch.testing.assert_close(v11-v10-v01+v00,action,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(v11-action,v10+v01-v00,rtol=1e-12,atol=1e-12)

    def test_rotation_fixed_rms_pad_zero(self):
        x=torch.arange(2*3*17,dtype=torch.float32).reshape(2,3,17);x[0,0]=0
        a=rotation(x,2026092401);b=rotation(x,2026092401)
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,x))
        torch.testing.assert_close(a.double().square().sum(-1),x.double().square().sum(-1),rtol=1e-12,atol=1e-12)
        self.assertEqual(float(a[0,0].abs().sum()),0.)

    def test_no_scientific_outcome_gate(self):
        self.assertEqual(ORDER,['G00','G10','G20','G21','G30','G31','G40','G50','G51','G60','G70'])
        from . import stages
        text=Path(stages.__file__).read_text()
        self.assertNotIn('apply_memit_to_model(',text)
        self.assertNotIn('apply_AlphaEdit_to_model(',text)
        self.assertNotIn('torch.save(',text)
        self.assertIn('scientific_positive_effect_required=False',text)

    def test_native_tensor_hash_schema(self):
        import hashlib
        from .backend import tensor_sha
        x=torch.arange(12,dtype=torch.float32).reshape(3,4)
        h=hashlib.sha256(str(('torch.float32',[3,4])).encode());h.update(x.numpy().tobytes())
        self.assertEqual(tensor_sha(x),h.hexdigest())

    def test_pair_reducer_strict_ties(self):
        from .reduce import pair_rows
        from .common import digest
        idx={};raw=[]
        for label in ('true','new'):
            rowid='x-'+label
            p=dict(pair_id='x',case_id=1,kind='N',label=label,target_ids=[7],input_ids=[3],positions=[0],prompt_index=0,new_target='a')
            idx[rowid]=p
            raw.append(dict(row_id=rowid,pair_id='x',case_id=1,kind='N',label=label,target_count=1,
                input_sha=digest([[3],[0],[7]]),nll=1.,strict=False,token_correct=0,subject='one',prompt_cluster='p',panel='N_diag1000'))
        result=pair_rows(raw,idx)[0]
        self.assertTrue(result['tie']);self.assertFalse(result['success'])

    def test_dependency_identity_not_existence(self):
        from .stages import Run
        with tempfile.TemporaryDirectory() as d:
            run=object.__new__(Run);run.output=Path(d);run.binding={'source':'one'}
            save(Path(d)/'G00/gate-result.json',dict(status='FAIL',binding=run.binding))
            with self.assertRaises(AssertionError):run.require('G10')


if __name__=='__main__':unittest.main()
