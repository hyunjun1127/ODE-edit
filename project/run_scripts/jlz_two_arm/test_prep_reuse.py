import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .common import write,member
from . import prep_reuse as reuse

class PrepReuseTests(unittest.TestCase):
    def fixture(self,root):
        old=root/'old';out=root/'new'/'prep';old.mkdir();out.mkdir(parents=True)
        original=dict(instruction_id='task',config_sha256='old-config',source_sha='old-lock')
        for name in reuse.OUTPUTS:
            if name=='w0-teachers.pt':(old/name).write_bytes(b'finite immutable teacher fixture')
            else:write(old/name,dict(original,route='dense'))
        write(old/'lock.json',{'source':'original'});write(old/'config.json',{'input':'same'})
        return {'prep_reuse':dict(W0_state={'weights':'same'},outputs={n:member(old/n) for n in reuse.OUTPUTS},
                   original_lock=member(old/'lock.json'),original_config=member(old/'config.json'))},out,original

    def test_explicit_reuse_keeps_original_bytes_and_binding(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,original=self.fixture(Path(d));before=member(Path(d)/'old/route.json')
            common=dict(instruction_id='task',config_sha256='new-config',source_sha='new-lock')
            self.assertEqual(reuse.consume(out.parent,config,common,{'weights':'same'}),'dense')
            row=json.loads((out/'route.json').read_text())
            self.assertEqual(row['original_binding'],original);self.assertFalse(row['new_computation'])
            self.assertEqual(row['source_sha'],'new-lock');self.assertEqual(before,member(Path(d)/'old/route.json'))
            self.assertTrue((out/'w0-teachers.pt').is_symlink())
            self.assertTrue((out/'W00-observations.json').is_symlink())

    def test_different_actual_W0_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,_=self.fixture(Path(d))
            with self.assertRaisesRegex(RuntimeError,'ACTUAL_W0_STATE'):
                reuse.consume(out.parent,config,{}, {'weights':'wrong'})
            self.assertEqual(list(out.iterdir()),[])

    def test_changed_input_bytes_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,_=self.fixture(Path(d))
            (Path(d)/'old/teacher.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'OUTPUT_CHANGED'):
                reuse.consume(out.parent,config,{}, {'weights':'same'})

    def test_function_body_change_not_path_controls_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'function.py';p.write_text('def f():\n    return 1\n')
            before=reuse.function_hash(p,['f']);p.write_text('# relocated\ndef f():\n    return 1\n')
            self.assertEqual(before,reuse.function_hash(p,['f']))
            p.write_text('def f():\n    return 2\n');self.assertNotEqual(before,reuse.function_hash(p,['f']))

    def test_import_receipt_ignores_synthetic_relative_file(self):
        from . import run
        config={'source':'/nonexistent-source','baseline_pilot':{'native_root':'/nonexistent-native'}}
        with patch.dict(run.sys.modules,{'torch.ops.fixture':SimpleNamespace(__file__='_ops.py')}):
            result=run.import_receipt(config)
        self.assertEqual(result['modules'],[])
