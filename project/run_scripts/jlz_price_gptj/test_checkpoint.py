"""CPU-only regression for production masked checkpoint closure binding.

Uses tiny fake native modules, not a target-model load, fit, or quality test.
The exact production Adapter.masked and non-reentrant checkpoint are exercised.
"""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from .adapter import Adapter


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (hidden_states * 0.125,)


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = _Attention()
        self.mlp = nn.Module()
        self.mlp.fc_in = nn.Linear(width, width + 2)
        self.mlp.fc_out = nn.Linear(width + 2, width)
        self.mlp.act = nn.Tanh()

    def forward(self, x, **kwargs):
        normalized = self.ln_1(x)
        attention = self.attn(normalized, **kwargs)[0]
        key = self.mlp.act(self.mlp.fc_in(normalized))
        return ((attention + self.mlp.fc_out(key)) + x,)


class CheckpointBindingTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            width = 4
            self.adapter = Adapter.__new__(Adapter)
            self.adapter.model = SimpleNamespace(config=SimpleNamespace(n_embd=width))
            self.adapter.blocks = nn.ModuleList([_Block(width) for _ in range(10)])
            self.adapter.blocks.eval().requires_grad_(False)
            self.adapter.sites = (3, 4, 5, 6, 7, 8)
            self.adapter.first = 3
            self.adapter.device = torch.device('cpu')
            self.adapter.weights = {
                l: self.adapter.projection(l).weight for l in self.adapter.sites}
            self.input = torch.randn(7, 5, width)
        with torch.no_grad():
            key, residual = self.adapter.pre_projection(3, self.input, {})
        self.cache = {'key': key, 'residual': residual, 'kwargs': {}}
        self.rows = []
        for row in range(7):
            target = torch.full((5,), -100, dtype=torch.long)
            if row < 6:
                target[3:] = 0
            self.rows.append({'lookup': row % 3 + 1, 'global_row': (row + 2) % 7,
                              'kind': 'rewrite' if row < 6 else 'kl', 'target': target})
        self.adapter.checkpoint_enabled = False
        with torch.no_grad():
            reference, _, _, _ = self.adapter.masked(
                self.group(False), self.increments(0.0), capture=False)
        self.references = [reference[row, self.positions(row)].clone()
                           for row in range(7)]

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def positions(self, row):
        record = self.rows[row]
        return (torch.nonzero(record['target'] != -100).flatten()
                if record['kind'] == 'rewrite'
                else torch.tensor([record['lookup']]))

    def group(self, pending):
        group = {'cache': self.cache, 'rows': self.rows,
                 'native_c0_pending': pending}
        if pending:
            group['native_c0_selected'] = self.references
        return group

    def increments(self, scale):
        return {l: (torch.arange(28, dtype=torch.float32).reshape(7, 4) * scale
                    + l * scale).requires_grad_() for l in self.adapter.sites}

    def compare(self, pending, capture, scale=0.0):
        results = []
        for enabled in (False, True):
            self.adapter.checkpoint_enabled = enabled
            self.adapter.physical_calls = {'checkpoint_wrappers': 0,
                                           'function_invocations': 0}
            group = self.group(pending)
            increments = self.increments(scale)
            hidden, final, keys, bases = self.adapter.masked(group, increments, capture)
            loss = (hidden.square() * torch.arange(1, 5)).sum()
            gradients = torch.autograd.grad(loss, tuple(increments.values()))
            self.assertIs(hidden, final)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(all(bool((g != 0).any()) for g in gradients))
            if pending:
                self.assertFalse(group['native_c0_pending'])
                self.assertNotIn('native_c0_selected', group)
                self.assertEqual(group['native_c0_error_max'], 0.0)
            if capture:
                self.assertEqual(set(keys), set(self.adapter.sites))
                self.assertEqual(set(bases), set(self.adapter.sites))
            else:
                self.assertEqual(keys, {})
                self.assertEqual(bases, {})
            if enabled:
                self.assertGreater(self.adapter.physical_calls['function_invocations'],
                                   self.adapter.physical_calls['checkpoint_wrappers'])
            results.append((hidden.detach(), gradients,
                            {l: t.detach() for l, t in keys.items()},
                            {l: t.detach() for l, t in bases.items()}))
        for direct, replay in zip(results[0][:2], results[1][:2]):
            if isinstance(direct, tuple):
                for expected, actual in zip(direct, replay):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                torch.testing.assert_close(replay, direct, rtol=0, atol=0)
        for direct, replay in zip(results[0][2:], results[1][2:]):
            for layer in direct:
                torch.testing.assert_close(replay[layer], direct[layer], rtol=0, atol=0)

    def test_first_c0_backward_without_capture(self):
        self.compare(pending=True, capture=False)

    def test_first_c0_backward_with_capture(self):
        self.compare(pending=True, capture=True)

    def test_later_candidate_backward_without_capture(self):
        self.compare(pending=False, capture=False, scale=0.002)

    def test_later_candidate_backward_with_capture(self):
        self.compare(pending=False, capture=True, scale=0.002)


if __name__ == '__main__':
    unittest.main()
