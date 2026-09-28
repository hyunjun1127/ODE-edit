"""Narrow regression for virtual import files and nonmutating gate receipts."""
import os,sys,tempfile,unittest,random
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import numpy as np,torch
from .provenance import import_closure,auxiliary_state
from .io import save

class RepairTests(unittest.TestCase):
    def test_actual_torch_virtual_module_reproduces_old_failure(self):
        m=sys.modules['torch.classes'];self.assertEqual(m.__file__,'_classes.py')
        with tempfile.TemporaryDirectory() as d:
            old=Path.cwd()
            try:
                os.chdir(d)
                with self.assertRaises(FileNotFoundError):Path(m.__file__).resolve().read_bytes()
                r=import_closure({'torch.classes':m},[d]);self.assertEqual(r['files'],[])
                self.assertEqual(r['virtual_modules'][0]['module'],'torch.classes')
            finally:os.chdir(old)
    def test_real_missing_scoped_file_still_fails(self):
        with tempfile.TemporaryDirectory() as d:
            m=ModuleType('fixture');m.__file__=d+'/missing.py'
            with self.assertRaisesRegex(FileNotFoundError,'REAL_SCOPED'):import_closure({'fixture':m},[d])
    def test_absolute_spec_relative_file_and_create_once_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.py';p.write_text('x=1\n')
            m=ModuleType('native');m.__file__='native.py';m.__spec__=SimpleNamespace(origin=str(p))
            r=import_closure({'native':m,'torch.classes':sys.modules['torch.classes']},[d])
            self.assertEqual(r['files'][0]['path'],str(p));save(Path(d)/'closure.json',r)
    def test_fingerprint_observer_and_b1_b2_ledger(self):
        with patch.object(torch.cuda,'get_rng_state_all',return_value=[]):
            ctx=[['{}']];records=[{'case_id':i} for i in range(100)]
            before=auxiliary_state(ctx,[]);after=auxiliary_state(ctx,[]);self.assertEqual(before,after)
            commit=auxiliary_state(ctx,records);entry=auxiliary_state(ctx,records);self.assertEqual(commit,entry)
            self.assertNotEqual(before['ledger'],entry['ledger']);self.assertEqual(entry['ledger']['requests'],100)
            random.random();self.assertNotEqual(entry['rng'],auxiliary_state(ctx,records)['rng'])
if __name__=='__main__':unittest.main()
