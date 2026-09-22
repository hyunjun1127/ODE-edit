"""CPU-only exact Slurm receipt parsing; never invokes a scheduler."""
import unittest
from project.run_scripts.alpha_key_concentration_causal.control import dependency_matches


class DependencyInspection(unittest.TestCase):
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
