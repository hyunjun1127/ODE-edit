"""Migration transport/launcher regressions; never actual GPU qualification."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import server3_submit as s

class MigrationTests(unittest.TestCase):
    def test_original_science_preserved(self):
        for path in ('entry.py','optimize.py','optimizer.py','writer.py','qualification.py','run.py','events.py','telemetry.py','collect.py'):
            rel='project/run_scripts/jlz_shared_budget/'+path
            frozen=subprocess.check_output(['git','show','7852d66ed5467f94ebda73507b6ccdbf5c921328:'+rel],cwd=s.ROOT)
            self.assertEqual((s.ROOT/rel).read_bytes(),frozen,rel)

    def test_cessation_scope_and_tamper(self):
        result=s.verify_source_cessation()
        self.assertEqual(result['sha256'],'74adc314f0b0a1f497aa1acefe415dd98867b9c0395d87369466cc2a150a3d22')
        with patch.object(s,'sha',return_value='invalid'), self.assertRaises(RuntimeError):
            s.verify_source_cessation()

    def test_two_gpu_stages_are_serial_collector_afterany(self):
        r=dict(host_mib=59392,collector_host_mib=24576,wall='7-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00')
        root=Path('/exact/attempt')
        for stage,dep in [('pilot','afterany:999'),('V12_MAIN','afterany:1000'),('collector','afterany:1000:1001')]:
            argv=s.arguments(stage,dep,root,r)
            self.assertIn('--nodelist=ubuntu',argv);self.assertIn('--qos=lab_gpu_s3',argv)
            self.assertIn('--export=NONE',argv);self.assertIn('--no-requeue',argv)
            self.assertIn('--dependency='+dep,argv);self.assertIn('--cpus-per-task=8',argv)
            self.assertIn('--mem='+('24576' if stage=='collector' else '59392')+'M',argv)
            self.assertEqual('--gres=gpu:1' in argv,stage!='collector')
            self.assertNotIn('server4',' '.join(argv))

    def test_launcher_uses_actual_python_and_isolated_overlay(self):
        for cpu in (False,True):
            script=s.launcher(Path('/sealed source'), 'a'*40, 'project.run_scripts.jlz_shared_budget.run',['--attempt','/exact attempt'],cpu)
            checked=subprocess.run(['bash','-n'],input=script,text=True,capture_output=True)
            self.assertEqual(checked.returncode,0,checked.stderr)
            self.assertIn(s.PYTHON,script)
            self.assertIn(str(s.LOCAL/'dependencies-r1'),script)
            self.assertIn('ODEEDIT_SOURCE_COMMIT='+('a'*40),script)
            self.assertEqual('export CUDA_VISIBLE_DEVICES=""' in script,cpu)
            self.assertNotIn('EasyEdit/.venv',script)

    def test_main_ready_lock_before_model_load(self):
        source=(s.ROOT/'project/run_scripts/jlz_shared_budget/run.py').read_text()
        self.assertIn("require(ready['status']=='CHAIN_COMPLETE'",source)
        self.assertLess(source.index("require(ready['status']=='CHAIN_COMPLETE'"),source.index('a,bench,records,H=setup'))
        self.assertIn("ready['source']==lock['source_commit']",source)
        self.assertIn("ready['config']==digest(config)",source)

if __name__=='__main__':unittest.main()
