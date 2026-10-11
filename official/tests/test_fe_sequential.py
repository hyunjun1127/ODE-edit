"""Sequential scope, immutable covariance repair and observational parity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from official.baselines.fe_author_profile import profile, resolve
from official.baselines.fe_observation import observe_batch, summarize
from official.experiments.prepare import digest
from official.experiments import checkpoint
from official.runners import fe_author_history as runner, fe_covariance_assets
from official.runners.server1.common import member
from official.runners.server1.test_fe_author import config


class SequentialTests(unittest.TestCase):
    def test_effective_covariance_and_writer_identity_checkpoint_roundtrip(self):
        # The exact extension emitted by fe_author_history.run previously failed
        # at W0 because the common store accepted only the seven legacy fields.
        for backend in ('EASYEDIT_HISTORY', 'AUTHOR_WRITER_CURRENT_BATCH_Z'):
            identity = {k: 'a'*64 for k in checkpoint.IDENTITY_FIELDS}
            identity.update(covariance_sha256='b'*64, z_policy=runner.Z_POLICY, writer_backend=backend)
            with tempfile.TemporaryDirectory() as folder:
                for batch in (0, 1):
                    checkpoint.save(folder, batch=batch, weights={'w': torch.full((2, 3), float(batch))},
                        cache_c={'0': torch.eye(3)*batch}, contexts={'z_policy': runner.Z_POLICY},
                        evaluation_cursor={'completed_batch': batch}, identity=identity,
                        method='MEMIT_FE_HISTORY', evaluation_complete=True)
                saved = checkpoint.load(folder, identity)
                self.assertEqual(saved['batch'], 1)
                self.assertEqual(saved['identity'], identity)
                self.assertTrue(torch.equal(saved['cache_c']['0'], torch.eye(3)))
                for change in ({'covariance_sha256': 'c'*64},
                               {'writer_backend': 'EASYEDIT_HISTORY' if backend != 'EASYEDIT_HISTORY' else 'AUTHOR_WRITER_CURRENT_BATCH_Z'}):
                    with self.assertRaisesRegex(ValueError, 'IDENTITY_MISMATCH'):
                        checkpoint.load(folder, dict(identity, **change))

    def test_gptj_author_settings_are_existing_settings(self):
        hp = resolve('gptj')
        self.assertEqual((hp.clamp_norm_factor, hp.v_num_grad_steps, hp.v_lr,
                          hp.v_weight_decay, hp.layers), (.75, 25, .5, .5, list(range(3, 9))))
        self.assertEqual(profile('gptj')['changed_scientific_fields'], [])
        runner.validate_config(config('gptj'))

    def test_future_target_policies_and_caches_rejected(self):
        for change in ({'z_policy': 'W0_ALL_2000'}, {'z_cache': '/prior/z.pt'},
                       {'precomputed_targets': []}, {'covariance_policy': 'silent_clip'}):
            c = config('gptj')
            c.update(change)
            c.pop('config_sha256')
            c['config_sha256'] = digest(c)
            with self.assertRaises(AssertionError):
                runner.validate_config(c)

    def fixture(self):
        weight = torch.nn.Parameter(torch.eye(2))
        native = SimpleNamespace(
            compute_z=lambda model, tok, request: {0: weight.detach()[:, request['case_id']].clone()},
            compute_ks=lambda *args: torch.tensor([[1., 2.], [3., 4.]]),
            get_module_input_output_at_words=lambda *args, **kwargs: torch.eye(2))
        return native, weight, [{'case_id': 0}, {'case_id': 1}], SimpleNamespace(layers=[0])

    def test_observation_keeps_outputs_and_rng_exact(self):
        native, weight, requests, hp = self.fixture()
        originals = dict(vars(native))
        expected = [native.compute_z(None, None, r) for r in requests]
        rng = torch.get_rng_state()
        with observe_batch(native, requests, hp, {'w': weight}) as observed:
            actual = [native.compute_z(None, None, r) for r in requests]
            keys = native.compute_ks(None, None, requests, hp, 0)
            native.get_module_input_output_at_words(None, None, 0, track='out')
            native.compute_ks(None, None, requests, hp, 0)
        self.assertEqual(vars(native), originals)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a[0], b[0]) for a, b in zip(expected, actual)))
        self.assertTrue(torch.equal(keys, observed['keys']['0']))
        self.assertEqual(summarize(observed)['layers']['0']['write_history_key_max_abs_difference'], 0.)

    def test_future_request_or_changed_entry_model_fails_and_restores(self):
        for mutate in (False, True):
            native, weight, requests, hp = self.fixture()
            originals = dict(vars(native))
            with self.assertRaisesRegex(ValueError, 'FE_TARGET_'):
                with observe_batch(native, requests, hp, {'w': weight}):
                    native.compute_z(None, None, requests[0])
                    if mutate:
                        with torch.no_grad():
                            weight.add_(1)
                    native.compute_z(None, None, {'case_id': 1 if mutate else 1001})
            self.assertEqual(vars(native), originals)

    def test_missing_current_targets_fails(self):
        native, weight, requests, hp = self.fixture()
        with self.assertRaisesRegex(ValueError, 'TARGETS_MISSING'):
            with observe_batch(native, requests, hp, {'w': weight}):
                native.compute_z(None, None, requests[0])

    def test_repaired_prior_is_frozen_and_resume_checks_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            asset = root/'prior.npz'
            # Rank-deficient PSD moment rounded just into the indefinite region.
            np.savez(asset, **{'sample_size': 10, 'mom2.count': 1,
                'mom2.mom2': np.array([[1., 1.], [1., np.nextafter(np.float32(1), np.float32(0))]], dtype=np.float32)})
            assets = {'C0': {'0': dict(module='layers.0', member=member(asset),
                                      validation={'masked_token_vector_count': 1})},
                      'model': {'identity': {'intermediate': 2}}}
            c = dict(config_sha256='a'*64, hparams={'mom2_n_samples': 10}, covariance_policy='project_roundoff')
            cov, receipt = fe_covariance_assets.prepare(c, assets, root/'run')
            self.assertTrue(receipt['0']['repaired'])
            self.assertEqual(torch.linalg.cholesky_ex(cov['layers.0']).info.item(), 0)
            with patch.object(fe_covariance_assets, 'prepare_covariance', side_effect=AssertionError('must not recompute')):
                resumed, again = fe_covariance_assets.prepare(c, assets, root/'run', resume=True)
            self.assertEqual(receipt, again)
            self.assertTrue(torch.equal(cov['layers.0'], resumed['layers.0']))
            derived = Path(receipt['0']['derived']['path'])
            with derived.open('ab') as f:
                f.write(b'corruption')
            with self.assertRaises(ValueError):
                fe_covariance_assets.prepare(c, assets, root/'run', resume=True)

    def test_healthy_prior_has_no_derived_file_or_numerical_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            asset = root/'prior.npz'
            matrix = np.array([[4., 1.], [1., 6.]], dtype=np.float32)
            np.savez(asset, **{'sample_size': 10, 'mom2.count': 2, 'mom2.mom2': matrix})
            assets = {'C0': {'0': dict(module='layers.0', member=member(asset),
                                      validation={'masked_token_vector_count': 2})},
                      'model': {'identity': {'intermediate': 2}}}
            c = dict(config_sha256='a'*64, hparams={'mom2_n_samples': 10}, covariance_policy='project_roundoff')
            cov, receipt = fe_covariance_assets.prepare(c, assets, root/'run')
            self.assertFalse(receipt['0']['repaired'])
            self.assertNotIn('derived', receipt['0'])
            self.assertTrue(torch.equal(cov['layers.0'], torch.from_numpy(matrix)/2))


if __name__ == '__main__':
    unittest.main()
