"""Narrow CPU controller/gradient regressions; not a GPU qualification receipt."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .geometry import FrozenCD, build_geometry
from .calibration import CalibrationLock
from .optimize import fit, forward, common_stop
from project.run_scripts.jlz_shared_budget.optimizer import EfficiencyAdam


class FakeAdapter:
    def __init__(self, B):
        self.sites = (0, 1)
        self.dims = {l: (2, 3) for l in self.sites}
        self.profile = dict(anchor_layer=1, kl_factor=.0625)
        self.device = torch.device('cpu')
        self.last_virtual = {'stale': True}
        self.capture_virtual = False

    def head(self, x):
        return x

    def native_projected(self, group, Y, capture=False, local_rows=False):
        assert local_rows
        x = (Y[0] + Y[1]).T
        hidden = {l: Y[l].T for l in self.sites} if capture else {}
        if self.capture_virtual and capture:
            for j, row in enumerate(group['rows']):
                self.last_virtual[row['global_row']] = {l: h[j].detach().clone() for l, h in hidden.items()}
        return x[:, None, :].expand(-1, 2, -1), x[:, None, :].expand(-1, 2, -1), hidden


def fixture(B=2, *, counts=None):
    a = FakeAdapter(B)
    counts = counts or [1] * B
    rows, canonical = [], []
    for r, count in enumerate(counts):
        canonical.append(len(rows))
        for i in range(count + 1):
            target = torch.tensor([-100, 0]) if i < count else torch.tensor([-100, -100])
            rows.append(dict(global_row=len(rows), request=r, kind='rewrite' if i < count else 'kl',
                             lookup=0, target=target))
    groups = [dict(rows=[row for row in rows if row['request'] == r]) for r in range(B)]
    owners = torch.tensor([row['request'] for row in rows])
    q = len(rows)
    S = torch.full((B, q), .25 / max(1, B - 1), dtype=torch.float64)
    S[owners, torch.arange(q)] = .75 if B > 1 else 1.
    C = torch.eye(B, dtype=torch.float64) + .25 * torch.ones((B, B), dtype=torch.float64)
    g = {l: FrozenCD(S, C, torch.zeros(2, B, dtype=torch.float64), torch.linalg.cholesky(C), owners,
                    1 / (B * torch.bincount(owners)[owners].double()), 2.,
                    dict(delta_zero=True, operator_hash=f'layer{l}:B{B}:q{q}', full_operator_compatible=B == 1))
         for l in a.sites}
    entry = dict(pack=dict(n_requests=B, n_rw=counts[0], row_kind=[row['kind'] for row in rows],
                          record_ids=list(range(100, 100 + B)), canonical_rows=canonical), groups=groups,
                 anchors={l: torch.full((B,), 100.) for l in a.sites},
                 teachers={r: torch.full((2,), -torch.log(torch.tensor(2.))) for r in range(B)}, delta_zero=True)
    return a, entry, g


class FixedCalibration:
    value = 7.
    status = 'TEST_FIXED'
    receipt = {'receipt_hash': 'test'}

    def resolve(self, **kwargs):
        return self.value


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(25)

    def test_stop_is_all_owner_strict_before_backward(self):
        self.assertIsNone(common_stop([.01, .06], 0))
        self.assertIsNone(common_stop([.05], 0))
        self.assertEqual(common_stop([.01, .02], 0), 'ZERO_STEP')
        self.assertEqual(common_stop([.01, .02], 2), 'OBJECTIVE_THRESHOLD')
        self.assertEqual(common_stop([.01, .6], 24), 'EVALUATION_BUDGET')

    def controlled(self, a, entry, losses, gradient=.001):
        calls = []
        def native(a, entry, group, u, geometries, capture=True):
            candidate = len(calls) // (2 * len(entry['groups']))
            calls.append((candidate, torch.is_grad_enabled()))
            values, captures = {}, {}
            for r in {row['request'] for row in group['rows']}:
                task = sum(value.sum() * gradient for value in u.values())
                values[r] = (task + losses(candidate, r), task * 0)
                captures[r] = {l: torch.full((2,), float(candidate)) for l in a.sites} if capture else {}
            if capture and a.capture_virtual:
                for row in group['rows']:
                    a.last_virtual[row['global_row']] = {l: torch.full((2,), float(candidate)) for l in a.sites}
            return values, captures
        return native, calls

    def test_common_candidate_no_independent_freeze_terminal_capture(self):
        a, entry, g = fixture()
        native, calls = self.controlled(a, entry, lambda c, r: 0. if c == 2 else (.001 if r == 0 else .2))
        with tempfile.TemporaryDirectory() as temp, patch(__name__.rsplit('.', 1)[0] + '.optimize.forward', native):
            lock = CalibrationLock(Path(temp) / 'lambda.json', role='producer', identity={'test': 'common'})
            plan, receipt = fit(a, entry, g, lock)
        self.assertEqual(receipt['terminal_candidate'], 2)
        self.assertEqual(receipt['request_updates'], 4)
        self.assertEqual(receipt['physical_forward_calls'], 10)
        self.assertEqual(receipt['physical_backward_calls'], 4)
        self.assertTrue(all(v['candidate'] == 2 for v in plan['terminal'].values()))
        self.assertTrue(all(torch.equal(z, torch.full((2, 2), 2.)) for z in plan['z'].values()))
        self.assertEqual(set(a.last_virtual), {0, 1, 2, 3})
        self.assertTrue(all(not grad for c, grad in calls if c == 2))
        self.assertFalse(a.capture_virtual)

    def test_zero_step_and_cap_no_terminal_backward_or_forward(self):
        for loss, expected in ((0., 0), (2., 24)):
            a, entry, g = fixture()
            native, calls = self.controlled(a, entry, lambda c, r: loss)
            with patch(__name__.rsplit('.', 1)[0] + '.optimize.forward', native):
                _, receipt = fit(a, entry, g, FixedCalibration())
            self.assertEqual(receipt['terminal_candidate'], expected)
            self.assertEqual(receipt['logical_evaluations'], expected + 1)
            self.assertEqual(receipt['physical_backward_calls'], 2 * expected)
            self.assertEqual(receipt['terminal_extra_forward'], 0)
            self.assertEqual(receipt['terminal_extra_backward'], 0)
            self.assertTrue(all(not grad for c, grad in calls if c == expected))

    def test_native_SUM_plus_norm_and_allocation_exactly_once(self):
        a, entry, g = fixture()
        native, calls = self.controlled(a, entry, lambda c, r: 1. if c == 0 else 0., gradient=5.)
        recorded = []
        class RecordingAdam:
            def __init__(self, blocks, anchor):
                self.t = 0
            def step(self, blocks, grads):
                self.t += 1
                recorded.append([grad.clone() for grad in grads])
                # A zero proposal keeps the next threshold decision exact.
                return [torch.zeros_like(block) for block in blocks], dict(update=self.t)
        base = __name__.rsplit('.', 1)[0] + '.optimize.'
        def allocation(u, anchors, geometry, *, alpha, allocation_price):
            self.assertEqual(allocation_price, 7.)
            return torch.full_like(u, 3.), dict(Q=0., c=0., cumulative=0., Pi=0., weighted_cost=0.)
        with patch(base + 'forward', native), patch(base + 'EfficiencyAdam', RecordingAdam), \
             patch(base + 'norm_gradient', side_effect=lambda block, price: torch.full_like(block, 2.)) as ng, \
             patch(base + 'cost_gradient_u', side_effect=allocation) as ag:
            _, receipt = fit(a, entry, g, FixedCalibration())
        self.assertEqual(len(recorded), 2)
        self.assertTrue(all(torch.equal(value, torch.full_like(value, 15.)) for grads in recorded for value in grads))
        self.assertEqual(ng.call_count, 4)
        self.assertEqual(ag.call_count, 2)
        self.assertEqual(receipt['norm_gradient_additions'], 1)
        self.assertEqual(receipt['allocation_gradient_additions'], 1)

    def test_projected_off_owner_native_gradient_and_physical_group_SUM(self):
        a, entry, g = fixture(3, counts=[1, 2, 3])
        u = {l: (torch.randn(2, 3) * .001).requires_grad_() for l in a.sites}
        values, _ = forward(a, entry, entry['groups'][0], u, g)
        loss = sum(nll + .0625 * kl for nll, kl in values.values())
        single, = torch.autograd.grad(loss, u[0])
        self.assertGreater(float(single[:, 2].norm()), 0.)
        split = {l: value.detach().clone().requires_grad_() for l, value in u.items()}
        for group in entry['groups']:
            values, _ = forward(a, entry, group, split, g)
            sum(nll + .0625 * kl for nll, kl in values.values()).backward()
        compact = {l: value.detach().clone().requires_grad_() for l, value in u.items()}
        all_rows = dict(rows=[row for group in entry['groups'] for row in group['rows']])
        values, _ = forward(a, entry, all_rows, compact, g)
        sum(nll + .0625 * kl for nll, kl in values.values()).backward()
        for l in a.sites:
            torch.testing.assert_close(split[l].grad, compact[l].grad, atol=1e-6, rtol=2e-4)

    def test_tail_B1_empty_and_zero_layer_reentry(self):
        a, entry, g = fixture(1)
        native, _ = self.controlled(a, entry, lambda c, r: 0.)
        with patch(__name__.rsplit('.', 1)[0] + '.optimize.forward', native):
            plan, receipt = fit(a, entry, g, FixedCalibration())
        self.assertEqual(plan['Y'][0].shape, (2, 2))
        self.assertEqual(receipt['requests'], 1)
        empty = dict(pack={'n_requests': 0}, groups=[])
        _, receipt = fit(a, empty, {}, FixedCalibration())
        self.assertTrue(receipt['empty_noop'])
        blocks = [torch.ones(2) * .1, torch.zeros(2)]
        opt = EfficiencyAdam(blocks, torch.tensor(2.))
        first, _ = opt.step(blocks, [torch.ones(2), torch.zeros(2)])
        second, _ = opt.step(first, [torch.zeros(2), -torch.ones(2)])
        self.assertGreater(float(second[1].norm()), 0.)
        self.assertLessEqual(sum(float(block.double().norm()) for block in second), .750001)
        self.assertEqual(opt.t, 2)

    def test_qualified_witness_and_near_boundary_fallback(self):
        for loss, expect_witness, expected_F in ((2., True, 74), (.050005, False, 98)):
            a, entry, g = fixture()
            entry['qualified_witness'] = True
            frame = {'candidate': 0}
            class Sink:
                def emit(self, event, payload, request=None, candidate=None, layer=None):
                    if event == 'coupled_candidate' and payload['common_terminal_reason'] is None:
                        frame['candidate'] = candidate + 1
            def native(a, entry, group, u, geometries, capture=True):
                candidate = frame['candidate']
                task = sum(value.sum() * 0 for value in u.values())
                owners = {row['request'] for row in group['rows']}
                captures = {r: {l: torch.full((2,), float(candidate)) for l in a.sites}
                            if capture else {} for r in owners}
                if capture:
                    for row in group['rows']:
                        a.last_virtual[row['global_row']] = {l: torch.full((2,), float(candidate)) for l in a.sites}
                return {r: (task + loss, task) for r in owners}, captures
            with patch(__name__.rsplit('.', 1)[0] + '.optimize.forward', native):
                plan, receipt = fit(a, entry, g, FixedCalibration(), events=Sink(), schedule='qualified-witness')
            self.assertEqual(receipt['terminal_candidate'], 24)
            self.assertEqual(receipt['physical_forward_calls'], expected_F)
            self.assertEqual(receipt['physical_backward_calls'], 48)
            self.assertTrue(all(c['witness'] == expect_witness for c in receipt['candidate_trace'][:-1]))
            self.assertFalse(receipt['candidate_trace'][-1]['witness'])
            self.assertTrue(all(torch.equal(value, torch.full((2, 2), 24.)) for value in plan['z'].values()))

    def test_wrong_owner_geometry_and_unqualified_witness_fail_closed(self):
        a, entry, g = fixture()
        with self.assertRaisesRegex(RuntimeError, 'WITNESS_SCHEDULE_UNQUALIFIED'):
            fit(a, entry, g, FixedCalibration(), schedule='qualified-witness')
        from dataclasses import replace
        g[0] = replace(g[0], owners=g[0].owners.flip(0))
        with self.assertRaisesRegex(RuntimeError, 'FIT_GEOMETRY_ROW_OWNERS'):
            fit(a, entry, g, FixedCalibration())

    def test_actual_tiny_same_candidate_qualification(self):
        import numpy as np
        from transformers import LlamaConfig, LlamaForCausalLM
        from .adapter import Adapter
        from .qualification import qualify
        from project.run_scripts.jlz_shared_budget.entry import prepare_entry
        from project.run_scripts.jlz_realized_writer.capture import capture_native_sites
        config = LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=31,
            max_position_embeddings=64, _attn_implementation='eager')
        a = Adapter(LlamaForCausalLM(config), dict(eligible_layers=[0, 1], anchor_layer=1,
                                                 nll_layer=2, kl_factor=.0625))
        B = 2
        pack = dict(identity='CPU_SAME_CANDIDATE', n_requests=B, n_rw=2, record_ids=[11, 12],
            canonical_rows=[0, 3], context_group_slices=[(0, 1), (1, 2)], lookup=[1] * 6,
            row_kind=['rewrite', 'rewrite', 'kl'] * B, row_request=[i // 3 for i in range(6)],
            tokens=dict(input_ids=torch.arange(30).reshape(6, 5) % 29 + 1,
                        attention_mask=torch.ones(6, 5, dtype=torch.long)),
            targets=torch.full((6, 5), -100))
        for i in range(6):
            if pack['row_kind'][i] == 'rewrite':
                pack['targets'][i, 3:] = torch.tensor([2, 3])
        H = {l: torch.zeros(16, 16) for l in a.sites}
        W0 = {l: weight.detach().cpu().clone() for l, weight in a.weights.items()}
        bench = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        with tempfile.TemporaryDirectory() as temp:
            stats = {}
            for l in a.sites:
                path = Path(temp) / f'stats{l}.npz'
                # Fixture only: not a main experiment asset or durable checkpoint.
                np.savez(path, **{'mom2.mom2': np.eye(16, dtype=np.float32), 'mom2.count': np.array(1)})
                stats[str(l)] = str(path)
            entry = prepare_entry(a, bench, pack, H, stats, 1)
            initial = capture_native_sites(a, entry, a.sites)
            owners = torch.tensor(pack['row_request'])
            geometries = {l: build_geometry(initial['keys'][l], owners, torch.eye(16, dtype=torch.float64) * 2,
                torch.eye(16, dtype=torch.float64), a.weights[l], W0[l], lambda_cov=2., request_count=B)
                for l in a.sites}
            receipt = qualify(a, entry, geometries, initial, H, W0)
        self.assertTrue(receipt['pass_'])
        self.assertFalse(receipt['GPU_qualified'])
        self.assertTrue(receipt['CPU_does_not_qualify_GPU'])
        self.assertEqual(receipt['sealed_budget']['fit_calls'], 0)
        self.assertEqual(a.last_virtual, {})


if __name__ == '__main__':
    unittest.main()
