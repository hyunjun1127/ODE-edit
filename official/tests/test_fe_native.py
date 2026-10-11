"""Native-entrypoint target timing and computation-neutral observation checks."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from official.diagnostics.fe_native import checkpoint_observer, sequential_z, patch as native_patch
from official.diagnostics.fe_native.evaluate import validate_payload, restore_weights
from official.diagnostics.fe_native.covariance import derive


class Tokens(dict):
    def to(self, device):
        return self


class Trace(dict):
    def __init__(self, module, layers, edit_output):
        super().__init__()
        self.handles = []
        for name in layers:
            def hook(_, inputs, output, name=name):
                output = edit_output(output, name)
                self[name] = SimpleNamespace(output=output)
                return output
            self.handles.append(module.get_submodule(name).register_forward_hook(hook))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False) for _ in range(2)])
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.eye(3))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def config():
    return OmegaConf.create(dict(bs=2, gpu=0, cache_dir='unused', cache_filename_suffix='',
        negetive_prompt_test=False, algs=dict(add_old_keys=True, L2=0),
        llms=dict(name='fixture', layers=[0, 1], rewrite_module_tmp='layers.{}',
                  layer_module_tmp='layers.{}', fact_token='subject_last', mom2_update_weight=[2., 2.])))


def tokenize(*args, **kwargs):
    return Tokens(inputs=torch.ones(1, 1, 3))


def lookup(*args, **kwargs):
    return 0


def fitted(model, tok, request, cfg, layer, contexts):
    return model.layers[layer].weight[:, 0].detach().clone() + request['case_id']/10.


def requests():
    return [dict(case_id=i, subject='subject', prompt='{} here', target_new='new') for i in (1, 2, 3, 4)]


class NativeTests(unittest.TestCase):
    def test_initial_reference_control_only_sees_current_requests_and_restores_editing_state(self):
        model, cfg = Model(), config()
        cfg.diagnostic_z_reference = 'initial_model_on_arrival'
        supply = sequential_z.TargetSupplier(model, cfg)
        with torch.no_grad():
            model.layers[0].weight.mul_(4)
            model.layers[1].weight.mul_(3)
        editing = {name:p.detach().clone() for name,p in model.named_parameters()}
        observed = []
        def fit(*args):
            observed.append(args[2]['case_id'])
            self.assertTrue(torch.equal(model.layers[0].weight, torch.eye(3)))
            return fitted(*args)
        targets = supply(model, tokenize, requests()[2:], cfg, [['{}']],
                         compute_z=fit, find_fact_lookup_idx=lookup, TraceDict=Trace)
        self.assertEqual(observed, [3, 4])
        expected = torch.stack([torch.tensor([1., 0., 0.]) + i/10. for i in (3, 4)], dim=1)
        self.assertTrue(torch.equal(targets[0], expected))
        self.assertTrue(torch.equal(targets[1], expected))
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p, editing[name]))

    def test_initial_reference_restores_weights_when_target_fit_fails(self):
        model, cfg = Model(), config()
        cfg.diagnostic_z_reference = 'initial_model_on_arrival'
        supply = sequential_z.TargetSupplier(model, cfg)
        with torch.no_grad():
            model.layers[0].weight.mul_(4)
        editing = {name:p.detach().clone() for name,p in model.named_parameters()}
        def failed(*args):
            raise RuntimeError('target-fit-failure')
        with self.assertRaisesRegex(RuntimeError, 'target-fit-failure'):
            supply(model, tokenize, requests()[:2], cfg, [['{}']],
                   compute_z=failed, find_fact_lookup_idx=lookup, TraceDict=Trace)
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p, editing[name]))

    def test_derived_covariance_native_roundtrip_preserves_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.npz'
            destination = Path(folder)/'derived.npz'
            matrix = np.array([[1., 1. + np.finfo(np.float32).eps],
                               [1. + np.finfo(np.float32).eps, 1.]], dtype=np.float32)
            np.savez(source, **{'mom2.count':np.array(1), 'mom2.mom2':matrix,
                              'sample_size':np.array(100000), 'batch_size':np.array(100)})
            before = source.read_bytes()
            sha = hashlib.sha256(before).hexdigest()
            receipt = derive(source, destination, expected_sha256=sha)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(receipt['repair']['corrected_eigenvalues'], 1)
            self.assertEqual(receipt['roundtrip_cholesky_info'], 0)
            with np.load(destination) as archive:
                # The actual author's SecondMoment loader has these same operations.
                moment = torch.from_numpy(archive['mom2.mom2']) / int(archive['mom2.count'])
                self.assertEqual(moment.dtype, torch.float64)
                self.assertTrue(torch.all(torch.linalg.eigvalsh(moment) > 0))
                self.assertEqual(int(archive['sample_size']), 100000)
            with self.assertRaisesRegex(ValueError, 'NEW_ASSET_REQUIRED'):
                derive(source, destination, expected_sha256=sha)

    def test_covariance_asset_refuses_wrong_source_and_healthy_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.npz'
            destination = Path(folder)/'derived.npz'
            np.savez(source, **{'mom2.count':np.array(1),
                              'mom2.mom2':np.eye(2, dtype=np.float32)})
            with self.assertRaisesRegex(ValueError, 'SOURCE_SHA'):
                derive(source, destination, expected_sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'HEALTHY_KEEP_ORIGINAL'):
                derive(source, destination, expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertFalse(destination.exists())

    def test_readonly_evaluation_rejects_reordered_checkpoint(self):
        model = Model()
        cfg = config()
        cfg.bs = 100
        cfg.num_edits = 2000
        cfg.model_dtype = 'bfloat16'
        cfg.diagnostic_identity_sha256 = 'a'*64
        rows = [dict(case_id=i) for i in range(2000)]
        payload = dict(schema='native-fe-sequential-diagnostic-v1', identity_sha256='a'*64,
            z_policy='CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS', completed_batches=5, completed_edits=500,
            seen_case_ids=list(range(500)), config=OmegaConf.to_container(cfg),
            weights={k:p.detach().clone() for k,p in model.named_parameters()})
        validate_payload(payload, {'identity_sha256':'a'*64}, rows, 5)
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_IDENTITY'):
            validate_payload(payload, {'identity_sha256':'a'*64,
                                       'z_reference':'initial_model_on_arrival'}, rows, 5)
        payload['seen_case_ids'][0], payload['seen_case_ids'][1] = 1, 0
        with self.assertRaisesRegex(ValueError, 'NATIVE_EVAL_STREAM_ORDER'):
            validate_payload(payload, {'identity_sha256':'a'*64}, rows, 5)

    def test_restore_validates_all_shapes_before_writing(self):
        model = Model()
        weights = {k:torch.ones_like(p) for k,p in model.named_parameters()}
        before = {k:p.detach().clone() for k,p in model.named_parameters()}
        invalid = dict(weights, **{'layers.1.weight':torch.ones(2,3)})
        with self.assertRaisesRegex(ValueError, 'NATIVE_EVAL_WEIGHT_SHAPE_DTYPE_FINITE'):
            restore_weights(model, invalid)
        self.assertTrue(all(torch.equal(p,before[k]) for k,p in model.named_parameters()))
        restore_weights(model, weights)
        self.assertTrue(all(torch.equal(p,weights[k]) for k,p in model.named_parameters()))

    def test_targets_use_only_this_batch_and_current_weights(self):
        model = Model()
        seen = []
        def target(*args):
            seen.append((args[2]['case_id'], args[0].layers[0].weight.detach().clone()))
            return fitted(*args)
        kwargs = dict(compute_z=target, find_fact_lookup_idx=lookup, TraceDict=Trace)
        first = sequential_z.current_batch_targets(model, tokenize, requests()[:2], config(), [['{}']], **kwargs)
        with torch.no_grad():
            model.layers[0].weight.add_(.25)
        entry = model.layers[0].weight.detach().clone()
        second = sequential_z.current_batch_targets(model, tokenize, requests()[2:], config(), [['{}']], **kwargs)
        self.assertEqual([r[0] for r in seen], [1, 2, 3, 4])
        self.assertTrue(all(torch.equal(r[1], entry) for r in seen[2:]))
        self.assertEqual(first[0].shape, (3, 2))
        self.assertTrue(torch.equal(second[1][:, 0], entry[:, 0]+.3))
        with self.assertRaisesRegex(ValueError, 'CURRENT_BATCH_ONLY'):
            sequential_z.current_batch_targets(model, tokenize, requests(), config(), [['{}']], **kwargs)

    def test_z_must_not_mutate_parameters(self):
        model = Model()
        def mutate(*args):
            with torch.no_grad():
                model.layers[0].weight.add_(1)
            return fitted(*args)
        with self.assertRaisesRegex(RuntimeError, 'MODEL_CHANGED_DURING_BATCH_TARGETS'):
            sequential_z.current_batch_targets(model, tokenize, requests()[:2], config(), [['{}']],
                compute_z=mutate, find_fact_lookup_idx=lookup, TraceDict=Trace)

    def test_native_snapshot_roundtrip_without_state_or_rng_change(self):
        model = Model()
        H = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        cfg = config()
        with tempfile.TemporaryDirectory() as folder, patch.object(torch.cuda, 'is_available', return_value=False):
            cfg.diagnostic_checkpoint_dir = folder
            cfg.diagnostic_identity_sha256 = 'a'*64
            cfg.diagnostic_reserve_bytes = 0
            cfg.bs = 1
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint_observer.snapshot(cfg, model, H, [['{}']], 0, requests())
            saved = torch.load(Path(folder)/'B00.pt', weights_only=False)
            self.assertEqual(saved['completed_edits'], 0)
            self.assertEqual(saved['seen_case_ids'], [])
            self.assertTrue(torch.equal(saved['history'], H))
            self.assertTrue(all(torch.equal(saved['weights'][k], v) for k, v in before.items()))
            self.assertTrue(all(torch.equal(p, before[k]) for k, p in model.named_parameters()))
            self.assertEqual(rng[0], random.getstate())
            self.assertTrue(np.array_equal(rng[1][1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(rng[2], torch.get_rng_state()))
            receipt = json.loads((Path(folder)/'B00.json').read_text())
            self.assertEqual(receipt['sha256'], checkpoint_observer.sha256(Path(folder)/'B00.pt'))

    def test_patch_rejects_unpinned_writer(self):
        with self.assertRaisesRegex(ValueError, 'UNPINNED_AUTHOR_FE_SOURCE'):
            native_patch.patched_source('def batch_edit(): pass\n')

    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'), 'optional pinned author checkout')
    def test_forward_targets_equal_author_precompute_at_same_model_state(self):
        root = Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])
        source = subprocess.check_output(['git', 'show', native_patch.AUTHOR_COMMIT+':precompute_z.py'], cwd=root, text=True)
        code = native_patch.function_text(source, 'compute_z_batch_first_forward')
        class TupleBlock(torch.nn.Linear):
            def forward(self, value):
                return (super().forward(value),)
        class TupleModel(Model):
            def forward(self, inputs):
                for layer in self.layers:
                    inputs = layer(inputs)[0]
                return inputs
        model = TupleModel()
        for i, layer in enumerate(model.layers):
            replacement = TupleBlock(3, 3, bias=False)
            replacement.load_state_dict(layer.state_dict())
            model.layers[i] = replacement
        def target(*args):
            self.assertEqual(args[2]['target_new'], ' new')
            return fitted(*args)
        namespace = dict(torch=torch, AutoModelForCausalLM=object, AutoTokenizer=object,
            DictConfig=object, List=list, Dict=dict, get_z_compute_function=lambda _: target,
            tqdm=lambda rows, **kw: rows, find_fact_lookup_idx=lookup, nethook=SimpleNamespace(TraceDict=Trace))
        exec(code, namespace)
        with patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            expected = namespace['compute_z_batch_first_forward'](model, tokenize, requests()[:2],
                config(), 'firstforward', [0, 1], 2, [['{}']])
            rows = [dict(row, target_new=' '+row['target_new']) for row in requests()[:2]]
            actual = sequential_z.current_batch_targets(model, tokenize, rows, config(), [['{}']],
                compute_z=target, find_fact_lookup_idx=lookup, TraceDict=Trace)
        self.assertTrue(all(torch.equal(actual[layer], expected[layer]) for layer in expected))

    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'), 'optional pinned author checkout')
    def test_actual_native_apply_preserves_history_and_batch_entry_z(self):
        root = Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])
        original = (root/'algs/memit/FE-memit_main.py').read_text()
        patched = native_patch.patched_source(original)
        self.assertEqual(native_patch.function_text(original, 'batch_edit'), native_patch.function_text(patched, 'batch_edit'))
        funcs = ('apply_memit_to_model', 'batch_edit', 'chunks', 'get_fc_dim', 'upd_matrix_match_shape')
        code = '\n\n'.join(native_patch.function_text(patched, name) for name in funcs)
        model = Model()
        observations = []
        Hs = []
        def target(*args):
            observations.append((args[2]['case_id'], model.layers[0].weight.detach().clone(), args[2]['target_new']))
            return fitted(*args)
        K = torch.tensor([[1., .2, .4], [.1, 1., .3]])
        native = dict(__package__='_native_fixture', torch=torch, deepcopy=copy.deepcopy,
            AutoModelForCausalLM=object, AutoTokenizer=object, List=list, Dict=dict,
            DictConfig=object, os=SimpleNamespace(path=SimpleNamespace(exists=lambda _: True)),
            nethook=SimpleNamespace(get_parameter=lambda model, name: model.get_parameter(name), TraceDict=Trace),
            ensure_file_directory=lambda _: None, set_random_seed=lambda _: None,
            load_cov=lambda *a: None, covs=[torch.eye(3), torch.eye(3)],
            get_context_templates=lambda *a: [['{}']], compute_z=target, find_fact_lookup_idx=lookup,
            compute_ks=lambda *a: K.clone(), time=__import__('time'),
            get_module_input_output_at_words=lambda model, tok, layer, **kw:
                (None, model.layers[layer].weight[:, 0].detach()[None, :].repeat(2, 1)))
        exec(code, native)
        observer = SimpleNamespace(snapshot=lambda cfg, model, H, *args: Hs.append(H.clone()))
        modules = {'_native_fixture.native_sequential_z': sequential_z,
                   '_native_fixture.native_checkpoint_observer': observer}
        rows = requests()
        with patch.dict(sys.modules, modules), patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            result = native['apply_memit_to_model'](model, tokenize, rows, config())
        self.assertIs(result, model)
        self.assertEqual([r[0] for r in observations], [1, 2, 3, 4])
        self.assertTrue(torch.equal(observations[0][1], observations[1][1]))
        self.assertTrue(torch.equal(observations[2][1], observations[3][1]))
        self.assertFalse(torch.equal(observations[1][1], observations[2][1]))
        self.assertTrue(all(r[2] == ' new' for r in observations))
        self.assertEqual(rows, requests())
        gram = (K.double().T@K.double()).float()
        self.assertTrue(torch.equal(Hs[-1], torch.stack([2*gram, 2*gram])))


if __name__ == '__main__':
    unittest.main()
