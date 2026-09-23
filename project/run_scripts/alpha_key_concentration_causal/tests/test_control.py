"""CPU-only exact Slurm receipt parsing; never invokes a scheduler."""
import unittest
import io,json,tarfile,tempfile
from unittest.mock import patch
from project.run_scripts.alpha_key_concentration_causal.control import dependency_matches,attempt_paths,freeze
from project.run_scripts.alpha_key_concentration_causal.common import save,file_sha
from pathlib import Path


class DependencyInspection(unittest.TestCase):
    def test_freeze_archive_lock_and_phase_routing_end_to_end_CPU(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'root';repo=Path(d)/'repo';root.mkdir();repo.mkdir()
            save(root/'receipts/preflight-r1/preflight.json',{'status':'PASS'})
            save(root/'receipts/preflight-r1/resource-plan.json',{'status':'PASS','planned_future_total':1})
            save(root/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json',{'dependencies':'/pinned/deps'})
            binding=root/'receipts/native-input-binding-r2.json'
            save(binding,{'status':'PASS','unresolved':[],'members':[]})
            rel='project/run_scripts/alpha_key_concentration_causal/dummy.py';payload=b'# CPU freeze fixture\n'
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode='w') as archive:
                info=tarfile.TarInfo(rel);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
            proc=SimpleNamespace(stdout=io.BytesIO(stream.getvalue()),wait=lambda:0)
            def cmd(argv):
                if 'status' in argv:return ''
                if 'ls-tree' in argv:return rel
                if 'rev-parse' in argv:return 'tree-fixture' if argv[-1]=='HEAD^{tree}' else 'source-fixture'
                raise AssertionError(argv)
            def checksum(p):
                return '500cd93224d4d2cb0daf880dad67aa18acf91bc0a12d127a2a23c42e7805ded1' if Path(p)==binding else file_sha(p)
            with patch('project.run_scripts.alpha_key_concentration_causal.control.command',side_effect=cmd), \
                 patch('project.run_scripts.alpha_key_concentration_causal.control.subprocess.Popen',return_value=proc), \
                 patch('project.run_scripts.alpha_key_concentration_causal.control.file_sha',side_effect=checksum), \
                 patch('project.run_scripts.alpha_key_concentration_causal.token_binding.load_reference',return_value={}):
                lock=freeze(root,repo,'attempt-r3')
                with self.assertRaises(AssertionError):freeze(root,repo,'attempt-r3')
            control=root/'controls/attempt-r3'
            self.assertEqual(lock['execution_lock_path'],str(control/'execution.lock.json'))
            self.assertEqual(lock['report_subdir'],'generated-r3')
            self.assertEqual(lock['repo'],str(root/'execution-source-r3'))
            self.assertEqual(len(lock['execution_source_members']),1)
            for phase in ('gate','geometry','writers','reduce'):
                script=(control/(phase+'.sh')).read_text()
                self.assertIn(str(control/'execution.lock.json'),script)
                self.assertIn('--phase '+phase,script)
                self.assertNotIn('attempt-r1',script)
            self.assertFalse((root/'control').exists())

    def test_attempt_paths_isolated_and_legacy_unchanged(self):
        root=Path('/tmp/alpha-causal-test')
        old,new=attempt_paths(root,'attempt-r1'),attempt_paths(root,'attempt-r2')
        self.assertEqual(old['control'],root/'control')
        self.assertEqual(old['frozen'],root/'execution-source-r1')
        self.assertEqual(new['control'],root/'controls/attempt-r2')
        for key in old:self.assertNotEqual(old[key],new[key])
        self.assertEqual(new['report_subdir'],'generated-r2')

    def test_attempt_traversal_and_alias_fail(self):
        for value in ('../attempt-r1','attempt-r0','attempt-r01','attempt-r2/../r1','r2','attempt-r-1'):
            with self.assertRaises(ValueError):attempt_paths('/tmp/alpha-causal-test',value)

    def test_null(self):
        self.assertTrue(dependency_matches('JobId=1 Dependency=(null) X=1',None))
        self.assertFalse(dependency_matches('Dependency=afterok:2(unfulfilled)',None))

    def test_per_id_annotation(self):
        self.assertTrue(dependency_matches('Dependency=afterany:1(unfulfilled):2(unfulfilled):3(unfulfilled) X=1',
                                           'afterany:1:2:3'))
        self.assertTrue(dependency_matches('Dependency=afterany:1(unfulfilled),afterany:2(unfulfilled)',
                                           'afterany:1:2'))

    def test_wrong_type_id_duplicate_and_extra_rejected(self):
        for value in ('afterok:1:2','afterany:1:3','afterany:1:1','afterany:1:2:3','afterany:1?afterany:2'):
            self.assertFalse(dependency_matches('Dependency='+value,'afterany:1:2'))


if __name__=='__main__':unittest.main()
