import tempfile
import unittest
from pathlib import Path
from project.run_scripts.baseline_mechanism_first.contracts import Cell, ContractBoundary, execution_plan, save
from project.run_scripts.baseline_mechanism_first.panels import historical_ordinals, panel_manifest


class ContractTests(unittest.TestCase):
    def test_scope_and_nonduplicated_execution_count(self):
        p = execution_plan()
        self.assertEqual(len(p['cells']), 20)
        self.assertEqual(p['E0_including_E1_max_native_batches'], 155)
        with self.assertRaises(ContractBoundary): Cell(3, 1000)

    def test_historical_and_cardinality(self):
        records = [dict(case_id=i, paraphrase_prompts=['a','b'], neighborhood_prompts=['n']*10) for i in range(10000)]
        for n in (1000,5000,9000):
            self.assertEqual(historical_ordinals(n), historical_ordinals(n))
            p = panel_manifest(records,n)
            self.assertEqual(p['prompt_pairs'], 2964)
            self.assertEqual(p['candidate_sequences'], 5928)
            self.assertTrue(all(i<n for i in historical_ordinals(n)))
        self.assertEqual(panel_manifest(records,0)['prompt_pairs'],1300)

    def test_create_once_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); save(root/'x.json', {'a':1})
            with self.assertRaises(FileExistsError): save(root/'x.json', {'a':2})
            (root/'link').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ContractBoundary): save(root/'link'/'y.json', {})
