"""CPU source/metadata/RNG fixtures; not actual native GPU qualification."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from official.experiments import checkpoint
from official.runners.server2 import run


class RunConnectorTests(unittest.TestCase):
    def test_twelve_exact_gptj_configs_not_ours(self):
        configs = [run.configuration(method, dataset) for dataset in ('cf','zsre')
                   for method in run.METHODS]
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({row['config_sha256'] for row in configs}), 12)
        self.assertTrue(all(row['edit_seed'] == 0 and row['stream']['requests'] == 2000
                            and row['stream']['batch_size'] == 100 for row in configs))
        self.assertFalse(any('PRICE' in row['method'] for row in configs))

    def test_checkpoint_identity_is_transport_independent_and_cell_specific(self):
        manifest = dict(streams={name:{'lock':{'stream_sha256':name+'-stream'}}
                        for name in ('cf','zsre')}, code_commit='a'*40, official_tree_sha256='b'*40,
                        model_revision='c'*40, tokenizer_sha256='d'*64, assets_identity_sha256='e'*64)
        identity = run.checkpoint_identity(manifest, 'MEMIT', 'cf')
        checkpoint.validate_identity(identity)
        other = dict(manifest, registration_stage='qualification', Slurm_job='12345',
                     tracking_run_id='not-science', base_manifest_sha256='f'*64)
        self.assertEqual(identity, run.checkpoint_identity(other, 'MEMIT', 'cf'))
        self.assertNotEqual(identity, run.checkpoint_identity(manifest, 'MEMIT', 'zsre'))
        self.assertNotEqual(identity, run.checkpoint_identity(manifest, 'FT', 'cf'))

    def test_rng_exact_comparison_not_numerical_nearness(self):
        before = checkpoint.rng_snapshot()
        self.assertTrue(run.equal(before, deepcopy(before)))
        modified = deepcopy(before)
        modified['torch_cpu'][0] ^= 1
        self.assertFalse(run.equal(before, modified))
        self.assertFalse(run.equal(np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)))
        self.assertFalse(run.equal(torch.tensor([1.]), torch.tensor([1.000001])))

    def test_cpu_tensor_hash_is_shape_dtype_and_bytes_bound(self):
        self.assertEqual(run.tensor_hash(torch.tensor([[1.,2.]])), run.tensor_hash(torch.tensor([[1.,2.]])))
        self.assertNotEqual(run.tensor_hash(torch.tensor([[1.,2.]])), run.tensor_hash(torch.tensor([1.,2.])))
        self.assertNotEqual(run.tensor_hash(torch.tensor([1.])), run.tensor_hash(torch.tensor([1.], dtype=torch.float64)))

    def test_missing_shared_tracking_is_typed_not_offline_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'OFFICIAL_TRACKING_SHARED_API_NOT_BOUND'):
                run.tracker({}, run.configuration('MEMIT','cf'), 'qualification', Path(temp))

    def test_official_macro_payload_not_old_pair_rpn_relabel(self):
        value = run.evaluate_payload({'summary':{'Efficacy':25., 'Specificity':75.,
            'requests':100, 'Specificity_availability':'NOT_MEASURED'}}, 'W5', 500)
        self.assertEqual(value['all_seen/post/Efficacy'],25.)
        self.assertEqual(value['edits'],500)
        self.assertNotIn('all_seen/post/R/success_pct',value)
        self.assertNotIn('all_seen/post/Specificity_availability',value)

    def test_actual_qualification_and_durable_resume_code_are_present(self):
        text = Path(run.__file__).read_text()
        self.assertIn("checkpoint.load(out/'checkpoints', identity)",text)
        self.assertIn("'contexts_equal'", text.replace('contexts_equal=', "'contexts_equal':"))
        self.assertIn("current_commit_receipt",text)
        self.assertIn("metadata_recovered_from_durable_checkpoint",text)
        self.assertIn("ATTEMPT_ALREADY_EXECUTED",text)
        self.assertNotIn('torch.save(model',text)


if __name__ == '__main__':
    unittest.main()
