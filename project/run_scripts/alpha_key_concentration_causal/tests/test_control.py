"""CPU-only exact Slurm receipt parsing; never invokes a scheduler."""
import unittest
from project.run_scripts.alpha_key_concentration_causal.control import dependency_matches,attempt_paths
from pathlib import Path


class DependencyInspection(unittest.TestCase):
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
