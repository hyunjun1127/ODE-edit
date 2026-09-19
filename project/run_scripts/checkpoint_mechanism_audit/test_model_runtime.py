"""CPU-only guard fixtures; these do not certify actual Llama numerics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from . import model_runtime as rt
from .common import ATTEMPT, REPO, WEIGHT, tensor_sha


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(3, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.eye(3, dtype=torch.float64) / 8)
        self.calls = 0
        self.fail = False

    def forward(self, value):
        self.calls += 1
        if self.fail:
            raise ValueError("OTHER_MODEL_FAILURE")
        return value + self.mlp.down_proj(value)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyLayer() for _ in range(7)])
        self.requires_grad_(False)

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class RuntimeGuardTests(unittest.TestCase):
    def test_prefix_retains_layer4_hook_output_and_skips_layer5(self):
        model = ToyModel(); seen = []
        x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        hook = model.model.layers[4].register_forward_hook(
            lambda module, inputs, output: seen.append(output.clone()))
        model(x)
        result = rt.PrefixProxy(model)(x)
        hook.remove()
        self.assertIsNone(result)
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertEqual([layer.calls for layer in model.model.layers], [2, 2, 2, 2, 2, 1, 1])
        self.assertEqual(len(model.model.layers[5]._forward_pre_hooks), 0)

    def test_prefix_proxy_forwards_module_and_parameter_discovery(self):
        model = ToyModel(); proxy = rt.PrefixProxy(model)
        self.assertEqual(list(dict(proxy.named_modules())), list(dict(model.named_modules())))
        self.assertIs(proxy.get_parameter(WEIGHT), model.get_parameter(WEIGHT))
        self.assertIs(next(proxy.parameters()), next(model.parameters()))

    def test_unrelated_exception_propagates_and_hook_is_removed(self):
        model = ToyModel(); model.model.layers[4].fail = True
        with self.assertRaisesRegex(ValueError, 'OTHER_MODEL_FAILURE'):
            rt.PrefixProxy(model)(torch.zeros(1, 3, dtype=torch.float64))
        self.assertEqual(len(model.model.layers[5]._forward_pre_hooks), 0)
        self.assertEqual(model.model.layers[5].calls, 0)

    def test_selected_copy_and_restore_preserve_other_parameter_versions(self):
        model = ToyModel(); before = rt.pointer_versions(model)
        entry = model.get_parameter(WEIGHT).detach().clone()
        selected_version = model.get_parameter(WEIGHT)._version
        rt.set_weight(model, entry + 1)
        self.assertEqual(before, rt.pointer_versions(model))
        self.assertGreater(model.get_parameter(WEIGHT)._version, selected_version)
        rt.set_weight(model, entry)
        self.assertTrue(torch.equal(model.get_parameter(WEIGHT), entry))
        self.assertEqual(before, rt.pointer_versions(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_block_affine_uses_same_bare_input_synthetic(self):
        model = ToyModel(); block = model.model.layers[4]
        bare = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        entry = model.get_parameter(WEIGHT).detach().clone()
        h0 = block(bare)
        delta = torch.tensor([[.5, 0, .1], [0, -.25, 0], [.3, 0, .1]], dtype=torch.float64)
        rt.set_weight(model, entry + delta)
        self.assertTrue(torch.allclose(block(bare), h0 + bare @ delta.T, rtol=0, atol=1e-14))

    def test_tensor_hash_has_original_dtype_shape_header(self):
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        contiguous = value.contiguous()
        expected = hashlib.sha256(str((str(contiguous.dtype), list(contiguous.shape))).encode())
        expected.update(contiguous.view(torch.uint8).numpy().reshape(-1).tobytes())
        self.assertEqual(tensor_sha(value), expected.hexdigest())
        self.assertNotEqual(tensor_sha(value), tensor_sha(value.reshape(-1)))

    def test_import_pinned_source_and_native_group_aggregation_cpu(self):
        """No model creation: import/metadata, original grouping, z-call guard."""
        receipt = ATTEMPT / 'inputs/source-ready.json'
        if not receipt.is_file():
            self.skipTest('local pinned source-ready unavailable; not a source import PASS')
        source = json.loads(receipt.read_text())
        deps = Path(source['deps'])
        if not deps.is_dir():
            self.skipTest('local pinned dependencies unavailable')
        code = r'''
import torch, transformers
from unittest.mock import patch
from project.run_scripts.checkpoint_mechanism_audit import model_runtime as rt
from project.run_scripts.blue_alphaedit_sequential_comparison import integrity
'''
        # Bind the helper package before importing its integrity module.
        code = code.replace('from project.run_scripts.blue_alphaedit_sequential_comparison import integrity', '''
b = rt.bind_sources()
from project.run_scripts.blue_alphaedit_sequential_comparison import integrity
assert transformers.__version__ == '4.44.2'
assert b[2].layers == [4] and b[2].L2 == 1 and b[2].blue
try:
    b[1].compute_z()
except RuntimeError as e:
    assert str(e) == 'NEW_Z_OPTIMIZATION_FORBIDDEN'
else:
    raise AssertionError('z optimizer not blocked')
requests = [dict(prompt='{} is here', subject='A'), dict(prompt='{} is there', subject='B')]
contexts = [['{}'], ['x {}', 'y {}', 'z {}', 'a {}', 'b {}']]
keys = torch.tensor([[100.], [0.], [2.], [4.], [6.], [8.],
                     [200.], [10.], [12.], [14.], [16.], [18.]])
with patch.object(b[0], 'get_module_input_output_at_words', return_value=(keys, keys)):
    result = b[0].compute_ks(None, None, requests, b[2], 4, contexts)
assert torch.equal(result, torch.tensor([[52.], [107.]])), result
x = torch.arange(6, dtype=torch.float32).reshape(2,3).T
assert rt.tensor_sha(x) == integrity.tensor_sha(x)
print('CPU_SOURCE_IMPORT_GROUPING_GUARDS_PASS_NO_MODEL')
''')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(deps) + os.pathsep + str(REPO)
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, env=env,
                                text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU_SOURCE_IMPORT_GROUPING_GUARDS_PASS_NO_MODEL', result.stdout)


if __name__ == '__main__':
    unittest.main()
