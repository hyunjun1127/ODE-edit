import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from .common import EASY,module
from .numerics import load_c0,native_project,validate

class TestPrep(unittest.TestCase):
    def test_native_strict_threshold(self):
        c=torch.diag(torch.tensor([.001,.019,.02,.021,1.],dtype=torch.float32))
        p,s=native_project(c,(EASY/'easyeditor/models/alphaedit/AlphaEdit_main.py').read_text())
        self.assertTrue(torch.equal(p,torch.diag(torch.tensor([1.,1.,0.,0.,0.]))))
        self.assertEqual(int((s<.02).sum()),2)
    def test_native_sum_count(self):
        raw=np.array([[4.,2.],[2.,10.]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'toy.npz'
            np.savez(path,**{'mom2.constructor':'easyeditor.util.runningstats.SecondMoment()','mom2.count':2,'mom2.mom2':raw,'sample_size':100000})
            c,count=load_c0(path,width=2)
            rs=module(EASY/'easyeditor/util/runningstats.py','test_native_rs')
            native=rs.SecondMoment();native.load_state_dict(dict(count=2,mom2=raw))
            self.assertTrue(torch.equal(c,native.moment()));self.assertEqual(count,2)
    def test_sampler_determinism(self):
        rs=module(EASY/'easyeditor/util/runningstats.py','test_native_rs')
        a=list(rs.FixedRandomSubsetSampler(range(1000),end=100,seed=1));b=list(rs.FixedRandomSubsetSampler(range(1000),end=100,seed=1))
        self.assertEqual(a,b);self.assertEqual(len(set(a)),100)
    def test_nonzero_retained_response_is_valid(self):
        tol=dict(c0_symmetry_relative=1e-6,P_symmetry_maxabs=1e-5,P_idempotence_relative=1e-4,P_trace_abs=.1,retained_RMS_slack=1e-4,c0_psd_relative=1e-6)
        c=torch.diag(torch.tensor([.01,1.],dtype=torch.float32));p=torch.diag(torch.tensor([1.,0.]))
        result=validate(c,p,1,tol);self.assertGreater(result['C0P_retained_RMS_Fro'],0)
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):validate(c*float('nan'),p,1,tol)
        with self.assertRaisesRegex(RuntimeError,'P_IDEMPOTENCE'):validate(c,p*.5,1,tol)

if __name__=='__main__':unittest.main()
