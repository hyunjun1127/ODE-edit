import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from project.run_scripts.baseline_mechanism_first.full_geometry import run,load_native_moment


class FullGeometryTests(unittest.TestCase):
    def test_exact_diagnostic_basis_and_native_operands_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'moment.npz'
            np.savez(path,**{'mom2.count':np.array(8),'mom2.mom2':np.diag([1,2,3]).astype(np.float32)*8})
            C,count=load_native_moment(path);self.assertEqual(count,8);self.assertEqual(C.dtype,torch.float32)
            P=torch.diag(torch.tensor([1.,1.,0.]));M=torch.eye(3);K=torch.ones(3,1)
            before=[x.clone() for x in (P,M,K)];W=torch.ones(2,3)
            ref=run(P,M,K,W*.01,W,torch.zeros_like(W),path,root/'output')
            data=json.loads(Path(ref['path']).read_text())
            self.assertEqual(data['geometry']['P_spectrum']['rank'],2)
            self.assertEqual(data['geometry']['P_spectrum']['method'],'EXACT_SVD')
            self.assertFalse(data['native_inputs_changed'])
            for a,b in zip((P,M,K),before):self.assertTrue(torch.equal(a,b))
