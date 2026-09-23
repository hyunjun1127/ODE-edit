"""Small deterministic CPU model fixtures for scoped component orchestration."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from project.run_scripts.alpha_key_concentration_causal.common import tensor_sha
from project.run_scripts.alpha_key_concentration_causal.component_runner import (
    InputBinding, KeyInterchange, _PrefixDone, component_context, fit_action_mean,
    full_hook_parity, run_components,
)


class TinyLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.eye(2) * scale)

    def forward(self, x):
        return x + self.mlp.down_proj(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(.01 * (i + 1)) for i in range(9)])
        self.eval()

    def forward(self, input_ids, attention_mask, **unused):
        hidden = torch.stack((input_ids.float() * .01 + .1, input_ids.float() * .003 + .2), -1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=torch.cat((hidden, hidden.sum(-1, keepdim=True)), -1))


class TinyTokenizer:
    def __call__(self, texts, *, padding, return_tensors):
        assert padding and return_tensors == 'pt'
        rows = [[(ord(c) % 7) + 1 for c in text] or [1] for text in texts]
        length = max(map(len, rows))
        return {'input_ids': torch.tensor([r + [0] * (length - len(r)) for r in rows]),
                'attention_mask': torch.tensor([[1] * len(r) + [0] * (length - len(r)) for r in rows])}


class TinyRuntime:
    def __init__(self):
        self.model, self.tok = TinyModel(), TinyTokenizer()
        self.weights = {l: self.model.model.layers[l].mlp.down_proj.weight for l in range(4, 9)}
        self.M = torch.eye(2).repeat(5, 1, 1)
        self.P = torch.eye(2).repeat(5, 1, 1)
        self.contexts = [['{}'], ['{}'] * 5]
        self.byid = {i: {'case_id': i, 'requested_rewrite': {'prompt': '{} z', 'subject': str(i % 19),
                                                          'target_new': {'str': 'new'}}} for i in range(4000)}
        self.hp = SimpleNamespace(layer_module_tmp='model.layers.{}', fact_token='subject_last')
        self.native = SimpleNamespace(compute_ks=self.compute_ks,
                                      get_module_input_output_at_words=self.get_io)

    def snapshot(self):
        return {'w': {l: w.detach().clone() for l, w in self.weights.items()}, 'm': self.M.clone()}

    def restore(self, saved):
        with torch.no_grad():
            for layer, weight in self.weights.items():
                weight.copy_(saved['w'][layer])
            self.M.copy_(saved['m'])

    def signature(self):
        return {'w': {str(l): tensor_sha(w) for l, w in self.weights.items()}, 'M': tensor_sha(self.M)}

    def _capture_io(self, prompts, words, layer):
        pack = self.tok([p.format(w) for p, w in zip(prompts, words)], padding=True, return_tensors='pt')
        result = {}
        def capture(module, args, output):
            last = pack['attention_mask'].sum(-1) - 1
            rows = torch.arange(len(last))
            result['input'] = args[0][rows, last].detach().clone()
            result['output'] = output[rows, last].detach().clone()
        handle = self.model.model.layers[layer].register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.model(**pack)
        finally:
            handle.remove()
        return result['input'], result['output']

    def compute_ks(self, model, tok, requests, hp, layer, contexts):
        return self._capture_io([r['prompt'] for r in requests], [r['subject'] for r in requests], layer)[0]

    def get_io(self, model, tok, layer, *, context_templates, words, **unused):
        return self._capture_io(context_templates, words, layer)

    def capture(self, records, *, contexts, features, full):
        keys, means = {}, {}
        for layer in range(4, 9):
            requests = [r['requested_rewrite'] for r in records]
            mean = self.compute_ks(self.model, self.tok, requests, self.hp, layer, self.contexts)
            means[layer] = mean
            keys[layer] = mean[:, None, :].repeat(1, 6, 1)
        return {'keys': keys, 'means': means, 'features': {}, 'token_receipt': {'fixture': True}, 'seconds': 0.0}


def _panels():
    return {'cohorts': {name: {'calibration_case_ids': list(range(i * 1000, i * 1000 + 128))}
                        for i, name in enumerate(('early', 'middle', 'onset', 'late'))}}


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.rt = TinyRuntime()

    def test_mask_binding_cleanup_and_prefix_exception(self):
        pack = self.rt.tok(['abc', 'x'], padding=True, return_tensors='pt')
        with InputBinding(self.rt.model) as binding:
            def abort(module, args):
                self.assertEqual(binding.mask().shape, pack['input_ids'].shape)
                raise _PrefixDone()
            handle = self.rt.model.model.layers[5].register_forward_pre_hook(abort)
            try:
                with self.assertRaises(_PrefixDone):
                    self.rt.model(**pack)
                self.assertEqual(binding.frames, [])
            finally:
                handle.remove()
        self.assertEqual(len(self.rt.model._forward_pre_hooks), 0)

    def test_full_hook_physical_exact_and_no_pad_action(self):
        pre = self.rt.snapshot()
        delta = torch.tensor([[.2, .1], [-.1, .05]])
        mean = torch.zeros(2)
        receipt = full_hook_parity(self.rt, 5, pre, delta, mean, [self.rt.byid[1], self.rt.byid[11]])
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['full_physical_max_abs'], 0)
        self.assertEqual(receipt['threshold'], 0)
        self.assertEqual(self.rt.signature()['M'], tensor_sha(pre['m']))

    def test_mean_every_canonical_prompt_and_valid_token(self):
        records = [self.rt.byid[i] for i in range(512)]
        with patch('project.run_scripts.alpha_key_concentration_causal.component_runner.rng_preserved', nullcontext):
            mean, receipt = fit_action_mean(self.rt, 5, torch.eye(2), records, microbatch=16)
        self.assertEqual(receipt['prompts'], 512)
        self.assertEqual(len(receipt['inputs']), 32)
        self.assertGreater(receipt['valid_tokens'], 512)
        self.assertTrue(torch.isfinite(mean).all())
        self.assertEqual(mean.dtype, torch.float32)

    def test_key_donor_prefix_bound_and_receiving_restored(self):
        pack = self.rt.tok(['abc', 'z'], padding=True, return_tensors='pt')
        before = self.rt.signature()
        donor = self.rt.weights[5].detach().clone() + .3 * torch.eye(2)
        basis = torch.eye(2)
        with torch.no_grad(), KeyInterchange(self.rt, 5, 6, donor, basis) as operation:
            value = self.rt.model(**pack).logits
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(len(operation.records), 1)
        self.assertEqual(operation.records[0]['donor_shape'][:2], list(pack['input_ids'].shape))
        self.assertEqual(self.rt.signature(), before)
        self.assertEqual(len(self.rt.model._forward_pre_hooks), 0)
        self.assertEqual(len(self.rt.model.model.layers[6].mlp.down_proj._forward_pre_hooks), 0)

    def test_no_component_is_identity(self):
        pack = self.rt.tok(['abcd', 'z'], padding=True, return_tensors='pt')
        with torch.no_grad():
            expected = self.rt.model(**pack).logits
            with component_context(self.rt, 5, self.rt.weights[5].detach(), torch.eye(2), torch.ones(2), 'no'):
                actual = self.rt.model(**pack).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_all_defined_family_branches_no_history_or_extra_z(self):
        pre = self.rt.snapshot()
        prestates = {5: pre, 6: pre}
        updates = {5: torch.eye(2) * .05, 6: torch.eye(2) * .04}
        requests = [dict(self.rt.byid[i]['requested_rewrite'], case_id=i) for i in range(100)]
        z = torch.ones((2, 100)) * 3
        original = self.rt.signature()
        calls = []
        def observe(endpoint, out):
            calls.append(endpoint)
            pack = self.rt.tok(['abc', 'x'], padding=True, return_tensors='pt')
            with torch.no_grad():
                self.rt.model(**pack)
            return {'fixture': True}
        with tempfile.TemporaryDirectory() as temp, patch('project.run_scripts.alpha_key_concentration_causal.component_runner.fit_action_mean', return_value=(torch.tensor([.1, .2]), {'fixture': True})):
            result = run_components(self.rt, 50, prestates, updates, z, requests, _panels(), observe, Path(temp) / 'out')
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['families'], 6)
            self.assertEqual(result['additional_z_requests'], 0)
            self.assertEqual(result['history_appends'], 0)
            self.assertEqual(sum('-KR-' in x for x in calls), 8)
            self.assertEqual(sum('-component-' in x for x in calls), 8)
            self.assertEqual(sum('-matrix-' in x for x in calls), 6)
            self.assertTrue((Path(temp) / 'out/terminal.json').exists())
        self.assertEqual(self.rt.signature(), original)


if __name__ == '__main__':
    unittest.main()
