"""Synthetic lane orchestration and dependency-local status tests; GPU0."""
import tempfile
import unittest
from pathlib import Path

import torch

from .common import CONTRACT,tensor_sha
from .operator_lane import (NATIVE_ENTRY,compute_history,make_rhs_bank,
                            residual_from_capture,save_analysis_tensor,
                            validate_bank,validate_dependencies)
from .operators import native_dense


class LaneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.g=torch.Generator().manual_seed(745)
        self.d,self.o,self.n=6,3,2
        q,_=torch.linalg.qr(self.rand(self.d,self.d).double())
        self.p=(q[:,:4]@q[:,:4].T).float()
        self.bank={"native":{},"w0_tensor_sha256":CONTRACT['identity']['w0_fp32_tensor_sha256']}
        self.residuals={}
        for j,b in enumerate(NATIVE_ENTRY):
            self.bank['native'][f'B{b:03d}']=self.capture([2*j,2*j+1])
            self.residuals[b]=self.rand(self.o,self.n)
        self.bank['geometry']=self.capture([100,101,102])

    def rand(self,*shape):return torch.randn(shape,generator=self.g,dtype=torch.float32)

    def capture(self,ids):
        n=len(ids)
        return dict(K=self.rand(self.d,n),bare_K=self.rand(self.d,n),h0=self.rand(self.o,n),
                    ids=ids,native_group_sizes=[1,5],key_weights=[.5,.1,.1,.1,.1,.1])

    def test_complete_bank_dimensions_order_and_native_blocks(self):
        receipt=validate_bank(self.bank,input_dim=self.d,output_dim=self.o,native_count=2,geometry_count=3)
        self.assertEqual(receipt['total_native_requests'],14)
        bank,slices=make_rhs_bank(self.bank,torch.device('cpu'))
        self.assertEqual(bank.shape,(self.d,17));self.assertEqual(bank.dtype,torch.float64)
        self.assertEqual(slices['geometry'],slice(0,3))
        for b in NATIVE_ENTRY:
            torch.testing.assert_close(bank[:,slices[f'B{b:03d}']],self.bank['native'][f'B{b:03d}']['K'].double())

    def test_bank_rejects_group_or_geometry_overlap(self):
        self.bank['geometry']['ids'][0]=0
        with self.assertRaises(ValueError):validate_bank(self.bank,input_dim=self.d,output_dim=self.o,native_count=2,geometry_count=3)
        self.bank['geometry']['ids'][0]=100
        self.bank['native']['B001']['key_weights']=[1/6]*6
        with self.assertRaises(ValueError):validate_bank(self.bank,input_dim=self.d,output_dim=self.o,native_count=2,geometry_count=3)

    def test_pilot_dependency_and_no_duplicate_histories(self):
        gate=dict(status='PASS',C01='PASS')
        self.assertEqual(validate_dependencies([0,10],gate,[])['histories'],[0,10])
        with self.assertRaises(ValueError):validate_dependencies([5],gate,[])
        with self.assertRaises(ValueError):validate_dependencies([0,0],gate,[])
        receipts=[dict(status='PASS',histories=[dict(history_batch=h,status='PASS') for h in [0,1,10,100]])]
        self.assertEqual(validate_dependencies([5,90],gate,receipts)['pilot_histories_verified'],[0,1,10,100])
        with self.assertRaises(ValueError):validate_dependencies([0],dict(status='PASS',C01='FAILED'),[])

    def test_residual_uses_physical_fp32_and_bare_not_mean_key(self):
        capture=self.bank['native']['B002'];w0=self.rand(self.o,self.d);we=w0+.01*self.rand(self.o,self.d)
        affine=capture['h0'].double()+(we.double()-w0.double())@capture['bare_K'].double()
        capture.update(h_entry=affine.float(),entry_batch=1,entry_weight_sha256=tensor_sha(we))
        z=self.rand(self.o,self.n)
        r,receipt=residual_from_capture(z,capture,we,w0,1)
        torch.testing.assert_close(r,z-capture['h_entry'],rtol=0,atol=0)
        self.assertEqual(receipt['h_kind'],'PHYSICAL_FP32_ENTRY_H')
        self.assertTrue(receipt['affine_physical_parity']['passed'])
        capture['entry_weight_sha256']='wrong'
        with self.assertRaises(ValueError):residual_from_capture(z,capture,we,w0,1)

    def test_fallback_affine_is_labelled_not_original_physical(self):
        capture=self.bank['native']['B002'];w0=self.rand(self.o,self.d);we=w0+.02*self.rand(self.o,self.d)
        z=self.rand(self.o,self.n);_,receipt=residual_from_capture(z,capture,we,w0,1)
        self.assertFalse(receipt['original_fp32_entry_capture'])
        self.assertIn('RECONSTRUCTED',receipt['h_kind'])

    def test_m0_small_engine_and_counterfactual_count(self):
        m=torch.zeros_like(self.p)
        k=self.bank['native']['B001']['K']
        actual,_=native_dense(self.p,m,k,self.residuals[1])
        result=compute_history(self.p.double(),m.double(),self.bank,self.residuals,0,
                              actual_b1=actual,original_dense_inputs=dict(P=self.p,M=m),w0_norm=5.)
        self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['probe_rows']),3)
        self.assertEqual(len(result['native_block_diagnostics']),7)
        self.assertEqual(len(result['counterfactual_rows']),42)
        self.assertEqual(result['reconstruction_rows'][0]['status'],'PASS')
        self.assertEqual(len(result['mode_rows']),2)
        self.assertEqual(result['Y'].shape,(self.d,17));self.assertEqual(result['factorization_count'],1)

    def test_late_dense_mode_and_missing_targets_local(self):
        a=self.rand(self.d,self.d);m=a@a.T
        result=compute_history(self.p.double(),m.double(),self.bank,self.residuals,90,
                              original_dense_inputs=dict(P=self.p,M=m))
        self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['counterfactual_rows']),0)
        self.assertEqual(result['reconstruction_rows'][0]['target_batch'],91)
        self.assertEqual(result['reconstruction_rows'][0]['status'],'PASS')
        missing=dict(self.residuals);del missing[91]
        result2=compute_history(self.p.double(),m.double(),self.bank,missing,90,
                               original_dense_inputs=dict(P=self.p,M=m))
        self.assertEqual(result2['status'],'PASS')
        self.assertEqual(result2['reconstruction_rows'][0]['status'],'BLOCKED')
        self.assertEqual(len(result2['mode_rows']),0)

    def test_failed_actual_attribution_does_not_fake_operator_failure(self):
        m=torch.zeros_like(self.p)
        result=compute_history(self.p.double(),m.double(),self.bank,self.residuals,0,
                              actual_b1=torch.zeros(self.o,self.d),original_dense_inputs=dict(P=self.p,M=m))
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['reconstruction_rows'][0]['status'],'FAILED')
        self.assertEqual(result['mode_rows'],[])

    def test_analysis_cache_cannot_store_weights_and_is_create_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.pt'
            with self.assertRaises(ValueError):save_analysis_tensor(path,dict(weights=torch.zeros(1)))
            receipt=save_analysis_tensor(path,dict(Y=torch.ones(2,3),history_batch=0))
            self.assertEqual(receipt['artifact_kind'],'ANALYSIS_KEY_RESPONSE_NOT_MODEL_CHECKPOINT')
            with self.assertRaises(FileExistsError):save_analysis_tensor(path,dict(Y=torch.ones(2,3)))


if __name__=='__main__':unittest.main()
