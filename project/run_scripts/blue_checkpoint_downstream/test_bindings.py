"""Minimal CPU source/data and selected-weight restoration fixtures."""
import json
from pathlib import Path
import unittest
import torch
from evaluator_adapter import DatasetBinding, bind_classes, TASKS
from restoration import W0Transaction, load_reference

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1')


class BindingTests(unittest.TestCase):
    def test_six_original_classes_exact_slice(self):
        expected = json.loads(Path('/mnt/raid5/janghj/ODE-edit/local/state/downstream-dataset-20260909-v1/source-audit.json').read_text())
        binding = DatasetBinding('/mnt/raid5/janghj/EasyEdit/glue_eval/dataset', expected)
        classes = bind_classes(ROOT/'evaluator-source', binding)
        self.assertEqual(set(classes), set(TASKS))
        for task in TASKS:
            evaluator = classes[task](None, None, number_of_tests=100, number_of_few_shots=0)
            self.assertEqual(evaluator.eval_dataset, binding.rows[task][10:110])
            self.assertEqual(evaluator.few_shots, [])
        with self.assertRaises(AssertionError):
            binding.split('../../dataset/rte.pkl',0,100)

    def test_singleton_does_not_carry_previous_pair_and_no_history_apply(self):
        ref = load_reference(ROOT/'source/closure/003-checkpoint_loader.py')
        model = torch.nn.Module()
        for key in ('l4','l8','other'):
            model.register_parameter(key,torch.nn.Parameter(torch.arange(4,dtype=torch.float32).reshape(2,2)))
        def entry(keys):
            return dict(weights={k:dict(sha256=ref.tensor_sha(getattr(model,k)+10),shape=[2,2],dtype='torch.float32') for k in keys},
                        base_selected_weights={k:dict(sha256=ref.tensor_sha(getattr(model,k))) for k in keys})
        pair, single = entry(['l4','l8']), entry(['l8'])
        guard = W0Transaction(model,ref,[pair,single])
        for e in (pair,single):
            cp=dict(weights={k:guard.w0[k]+10 for k in e['weights']}, cache_c=torch.ones(3),metadata={})
            cache_before=cp['cache_c'].clone()
            guard.apply(cp,e)
            self.assertTrue(torch.equal(cp['cache_c'],cache_before))
            if e is single:
                self.assertEqual(ref.tensor_sha(model.l4),guard.w0_hashes['l4'])
            guard.restore_w0()
        self.assertEqual(guard.hashes(),guard.w0_hashes)

    def test_nonselected_mutation_is_detected(self):
        ref = load_reference(ROOT/'source/closure/003-checkpoint_loader.py')
        model=torch.nn.Linear(2,2)
        entry=dict(weights={'weight':{}},base_selected_weights={'weight':dict(sha256=ref.tensor_sha(model.weight))})
        guard=W0Transaction(model,ref,[entry])
        with torch.no_grad():
            model.bias.add_(1)
        with self.assertRaises(AssertionError):
            guard.verify_endpoint()


if __name__ == '__main__':
    unittest.main()
