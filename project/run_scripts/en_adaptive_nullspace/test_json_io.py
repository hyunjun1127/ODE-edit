import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .json_io import save
from .geometry import build_geometry
from .controller import run_controller

class ReceiptTests(unittest.TestCase):
    def test_complete_controller_ledgers(self):
        geo=build_geometry(np.eye(2), [.5,.5], [0,1], np.eye(2))
        wn=np.zeros((1,2),np.float32); g=np.array([[1.,0.]])
        cache=[]
        results=[]
        for name, d, objective in [('accepted',-.1*g,.9),('fallback',-.1*g,2.),('no-step',0*g,1.)]:
            r=run_controller(wn,g,d,1.,lambda w:dict(J=objective),geo,
                native_norm=1.,native_action=100.,input_identity=name,objective_cache=cache)
            results.append((name,r))
        alias=run_controller(wn,g,-.1*g,1.,lambda w: self.fail('cache miss'),geo,
            native_norm=1.,native_action=100.,input_identity='accepted',objective_cache=cache)
        self.assertEqual(alias['evaluations'],0)
        results.append(('alias',alias))
        original_failed=False
        with tempfile.TemporaryDirectory() as directory:
            for name,result in results:
                value={k:v for k,v in result.items() if k!='weight'}
                try: json.dumps(value,allow_nan=False)
                except TypeError: original_failed=True
                path=Path(directory)/name
                save(path,value)
                decoded=json.loads(path.read_text())
                self.assertEqual(decoded['status'],result['status'])
                self.assertEqual(decoded['objective'],result['objective'])
                for before,after in zip(result['ledger'],decoded['ledger']):
                    self.assertEqual(before['geometry_checks'],after['geometry_checks'])
        self.assertTrue(original_failed)

    def test_reject_nonfinite_and_state_without_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            for value in [float('nan'),float('inf'),np.float64('nan'),np.array([1]),torch.ones(1),object(),np.complex128(1j)]:
                with self.assertRaises((TypeError,ValueError)):save(path,{'nested':[value]})
                self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_existing_partial_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'partial.json';path.write_bytes(b'{"ideal_norm":')
            with self.assertRaises(FileExistsError):save(path,{'ideal_norm':np.bool_(True)})
            self.assertEqual(path.read_bytes(),b'{"ideal_norm":')
            self.assertEqual(len(list(Path(directory).iterdir())),1)

    def test_builtin_scalar_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'values';save(path,[np.bool_(True),np.int64(7),np.float32(.5)])
            value=json.loads(path.read_text());self.assertEqual([type(x) for x in value],[bool,int,float])

if __name__=='__main__':unittest.main()
