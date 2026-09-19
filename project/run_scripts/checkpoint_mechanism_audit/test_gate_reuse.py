"""Reuse checks bind failed-attempt evidence; numerical failure stays failed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import write_json,sha256
from . import model_gate as g


class GateReuseTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);source=root/'prior';source.mkdir()
        write_json(root/'inputs/source-map.json',{})
        names=['runtime.json','C00.json','failure.json','B001-keys.pt','B002-keys.pt','geometry32-keys.pt',
               'W0-first100.json','B1-first100.json','evaluation-parity.json']
        for name in names:
            value={'status':'PASS'} if name=='C00.json' else {'stage':'B1_DENSE'} if name=='failure.json' else {'status':'FAILED','unchanged_tolerance':1e-4}
            write_json(source/name,value)
        p=root/'reuse.json'
        write_json(p,dict(status='SEALED_TECHNICAL_CONTINUATION',original_failure_stage='B1_DENSE',root=str(source),
            source_map_sha256=sha256(root/'inputs/source-map.json'),members=[dict(name=n,bytes=(source/n).stat().st_size,sha256=sha256(source/n)) for n in names]))
        return root,source,p

    def test_failed_evaluator_remains_reusable_failed_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root,source,p=self.fixture(td)
            with patch.object(g,'ATTEMPT',root):
                self.assertEqual(g.verified_reuse(p)['root'],str(source))
                self.assertEqual(g.read(source/'evaluation-parity.json')['status'],'FAILED')

    def test_changed_member_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root,source,p=self.fixture(td)
            (source/'runtime.json').write_text('{}')
            with patch.object(g,'ATTEMPT',root),self.assertRaises(AssertionError):g.verified_reuse(p)


if __name__=='__main__':unittest.main()
