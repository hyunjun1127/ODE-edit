"""CPU synthetic SUM/fallback regressions; no model or GPU evidence is made."""
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from project.run_scripts.jlz_realized_writer.capture import cache_identity
from .geometry import build_geometry
from .qualification import (_all_pass, _compare_native, _native_pass, _parity,
                            _snapshot_u, _validate_original_owner_groups, qualify)
from .test_optimize import FakeAdapter, fixture


class TracedAdapter(FakeAdapter):
    """Tensor-only native action stub, not an actual language model."""

    def __init__(self, B):
        super().__init__(B)
        self.native_route = 'cached'
        self.weights = {l: torch.zeros(self.dims[l]) for l in self.sites}
        self.trace = []
        self.backward_rows = []
        self.bad_full_gradient = False
        self.omit_site_gradient = False

    def guard(self):
        return 'synthetic-frozen'

    def hook_signature(self):
        return ()

    def prefix(self, tokens):
        raise AssertionError('fallback must never construct a regrouped prefix')

    def native_projected(self, group, Y, capture=False, local_rows=False):
        indices = tuple(row['global_row'] for row in group['rows'])
        self.trace.append((self.native_route, indices, id(group['tokens']),
                           id(group['cache']), tuple(group['tokens']['input_ids'].shape)))
        for l, value in Y.items():
            if value.requires_grad:
                value.register_hook(lambda grad, l=l: self.backward_rows.append((indices, l)))
        if self.bad_full_gradient and self.native_route == 'full':
            Y = dict(Y)
            # Same forward values and subjects; intentionally wrong VJP.
            value = Y[0]
            Y[0] = value.detach() + 1.5 * (value - value.detach())
        if self.omit_site_gradient:
            Y = dict(Y)
            Y[1] = Y[1].detach()
        return super().native_projected(group, Y, capture, local_rows)

    def native(self, group, D, capture=False):
        owners = [row['request'] for row in group['rows']]
        return self.native_projected(group, {l: value[:, owners] for l, value in D.items()},
                                     capture, local_rows=True)


def synthetic_entry(B=3, counts=None):
    _, entry, geometries = fixture(B, counts=counts)
    a = TracedAdapter(B)
    entry['pack']['identity'] = 'SYNTHETIC_CPU_NOT_ACTUAL_MODEL_EVIDENCE'
    for owner, group in enumerate(entry['groups']):
        group['tokens'] = dict(input_ids=torch.full((len(group['rows']), 2), owner + 1),
                               attention_mask=torch.ones(len(group['rows']), 2, dtype=torch.long))
        group['cache'] = dict(marker=torch.tensor([owner]))
    return a, entry, geometries


def synthetic_qualification_entry(*, incompatible=False):
    a, entry, _ = synthetic_entry(2, counts=[1, 2])
    rows = [row for group in entry['groups'] for row in group['rows']]
    owners = torch.tensor([row['request'] for row in rows])
    keys = torch.zeros(3, len(rows))
    keys[owners, torch.arange(len(rows))] = 1.
    if incompatible:
        keys = torch.tensor([[1., 2., 1., 0., 1.],
                             [0., 1., 0., 1., 2.],
                             [1., 0., 2., 1., 0.]])
    history = {l: torch.zeros(3, 3) for l in a.sites}
    W0 = {l: value.clone() for l, value in a.weights.items()}
    geometries = {l: build_geometry(keys, owners, torch.eye(3, dtype=torch.float64) * 2,
        torch.eye(3, dtype=torch.float64), a.weights[l], W0[l], lambda_cov=2., request_count=2)
        for l in a.sites}
    entry['stats'] = {str(l): 'SYNTHETIC_NO_FILE' for l in a.sites}
    initial = dict(rows=rows, keys={l: keys.clone() for l in a.sites})
    return a, entry, geometries, initial, history, W0


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_joint_SUM_keeps_every_original_forward_and_off_owner_gradient(self):
        a, entry, g = synthetic_entry(3, counts=[1, 2, 3])
        u = _snapshot_u(a, 3)
        before = cache_identity(entry)
        backward = torch.Tensor.backward
        dispatches = []

        def counted_backward(value, *args, **kwargs):
            dispatches.append(value.shape)
            return backward(value, *args, **kwargs)

        with patch.object(torch.Tensor, 'backward', new=counted_backward):
            groupwise = _native_pass(a, entry, entry['groups'], g, u)
            groupwise_trace = list(a.trace)
            self.assertEqual(len(dispatches), 3)
            a.trace.clear()
            a.backward_rows.clear()
            dispatches.clear()
            joint = _native_pass(a, entry, entry['groups'], g, u, backward_schedule='logical-sum')
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(a.trace, groupwise_trace)
        self.assertEqual(joint['native_forward_calls'], 3)
        self.assertEqual(joint['native_backward_calls'], 1)
        self.assertEqual(groupwise['native_backward_calls'], 3)
        self.assertEqual(set(joint['hidden']), set(range(9)))
        self.assertEqual(set(joint['logprobs']), set(range(9)))
        self.assertEqual(set(a.backward_rows), {
            (tuple(row['global_row'] for row in group['rows']), l)
            for group in entry['groups'] for l in a.sites})
        parity = _compare_native(groupwise, joint, a.sites)
        self.assertTrue(_all_pass(parity))
        for record in parity['gradient'].values():
            self.assertEqual((record['atol'], record['rtol'], record['reduction']),
                             (1e-6, 2e-4, 'elementwise'))
        self.assertEqual(cache_identity(entry), before)
        # A single owner's loss must still differentiate every other owner
        # through full S; neither the reference nor production may use E.
        single = _native_pass(a, entry, entry['groups'][:1], g, u)
        for gradient in single['gradients'].values():
            self.assertGreater(float(gradient[:, 2].norm()), 0.)
        self.assertFalse(any(value.grad is not None for value in u.values()))

    def test_missing_forward_row_or_site_gradient_is_not_silently_accepted(self):
        a, entry, g = synthetic_entry()
        u = _snapshot_u(a, 3)
        full = _native_pass(a, entry, entry['groups'], g, u)
        omitted = _native_pass(a, entry, entry['groups'][:-1], g, u,
                               backward_schedule='logical-sum')
        with self.assertRaisesRegex(RuntimeError, 'QUALIFICATION_NATIVE_ROW_COVERAGE'):
            _compare_native(omitted, full, a.sites)
        a.omit_site_gradient = True
        with self.assertRaisesRegex(RuntimeError, 'QUALIFICATION_MISSING_NATIVE_GRADIENT'):
            _native_pass(a, entry, entry['groups'], g, u, backward_schedule='logical-sum')
        with self.assertRaisesRegex(RuntimeError, 'QUALIFICATION_PARITY_SHAPE'):
            _parity(torch.ones(1), torch.ones(2))

    def test_mixed_or_split_owner_grouping_blocks_before_forward(self):
        a, entry, g = synthetic_entry(2)
        _validate_original_owner_groups(a, entry, g)
        original = entry['groups']
        entry['groups'] = [dict(rows=[row for group in original for row in group['rows']])]
        with self.assertRaisesRegex(RuntimeError, 'QUALIFICATION_FIXED_ORIGINAL_OWNER_GROUPS_ONLY'):
            qualify(a, entry, g, {}, {l: torch.zeros(3, 3) for l in a.sites}, a.weights)
        entry['groups'] = [dict(rows=original[0]['rows'][:1]),
                           dict(rows=original[0]['rows'][1:]), original[1]]
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_REQUEST_GRAPH'):
            _validate_original_owner_groups(a, entry, g)
        self.assertEqual(a.trace, [])

    def synthetic_qualify(self, data, **kwargs):
        # Mock only stat loading. Real small dense/compact arithmetic and all
        # synthetic native graphs/gates execute; nothing is persisted.
        base = __name__.rsplit('.', 1)[0] + '.qualification.'
        with patch(base + 'build_prior_from_npz',
                   side_effect=lambda *args, **kw: torch.eye(3, dtype=torch.float64) * 2):
            return qualify(*data, **kwargs)

    def test_receipt_is_fallback_only_and_never_upgrades_CPU_or_old_failure(self):
        data = synthetic_qualification_entry()
        a, entry, _, _, _, _ = data
        old_virtual = dict(a.last_virtual)
        entry['source_identity'] = 'SYNTHETIC_SOURCE'
        entry['qualification_repair_receipt'] = dict(path='SYNTHETIC_PRIOR_FAILURE', sha256='fixture')
        receipt = self.synthetic_qualify(data)
        self.assertTrue(receipt['pass_'])
        self.assertFalse(receipt['GPU_qualified'])
        self.assertTrue(receipt['CPU_does_not_qualify_GPU'])
        self.assertEqual(receipt['qualification_scope'], 'FIXED_ORIGINAL_OWNER_GROUPS_ONLY')
        self.assertEqual(receipt['requests_per_group'], 1)
        self.assertEqual(receipt['regrouped_group_rows'], [])
        self.assertEqual(receipt['original_group_rows'], receipt['logical_SUM_reference_group_rows'])
        self.assertFalse(receipt['physical_regrouping_qualified'])
        self.assertNotIn('logical_SUM_microbatch', receipt['checks'])
        self.assertIn('original_group_logical_SUM', receipt['checks'])
        rejected = receipt['rejected_physical_regrouping']
        self.assertEqual((rejected['status'], rejected['usage']), ('NOT_QUALIFIED', 'NOT_USED'))
        self.assertFalse(rejected['old_failure_relabelled_PASS'])
        self.assertFalse(rejected['retried'])
        self.assertFalse(rejected['qualified_by_original_group_SUM'])
        self.assertEqual(rejected['immutable_prior_failure_link'], entry['qualification_repair_receipt'])
        self.assertEqual(receipt['qualification_repair_receipt'], entry['qualification_repair_receipt'])
        self.assertEqual(receipt['source_identity'], 'SYNTHETIC_SOURCE')
        self.assertEqual(receipt['actual_calls'], dict(native_forward=9, native_backward=7,
            prefix_calls=0, independent_qr_svd=2, independent_cholesky=2))
        for key, count in receipt['actual_calls'].items():
            self.assertEqual(count, receipt['sealed_budget'][key + '_max'])
        self.assertEqual(a.last_virtual, old_virtual)
        self.assertFalse(a.capture_virtual)
        self.assertEqual(a.native_route, 'cached')

    def test_fresh_native_route_gradient_failure_still_fails_closed(self):
        data = synthetic_qualification_entry()
        data[0].bad_full_gradient = True
        saved = []
        base = __name__.rsplit('.', 1)[0] + '.qualification.'
        with patch(base + 'write', side_effect=lambda path, receipt: saved.append(receipt)):
            with self.assertRaisesRegex(RuntimeError, 'SAME_CANDIDATE_QUALIFICATION_FAILED'):
                self.synthetic_qualify(data, out=Path('SYNTHETIC_NOT_WRITTEN'))
        self.assertEqual(len(saved), 1)
        self.assertFalse(saved[0]['pass_'])
        self.assertFalse(saved[0]['GPU_qualified'])
        self.assertTrue(saved[0]['checks']['cached_vs_full_native']['loss']['pass_'])
        self.assertTrue(any(not record['pass_'] for record in
                            saved[0]['checks']['cached_vs_full_native']['gradient'].values()))
        self.assertIsNone(saved[0]['qualification_repair_receipt'])

    def test_incompatible_operator_retains_projected_S_and_exact_bounded_calls(self):
        data = synthetic_qualification_entry(incompatible=True)
        self.assertTrue(all(not g.full_operator_compatible for g in data[2].values()))
        receipt = self.synthetic_qualify(data)
        self.assertTrue(receipt['pass_'])
        direct = receipt['checks']['compatible_direct_vs_projected']
        self.assertEqual(direct['status'], 'NOT_CLAIMED_RANK_INCOMPATIBLE_OPERATOR')
        self.assertTrue(direct['projected_S_path_required'])
        self.assertTrue(_all_pass(receipt['checks']['original_group_logical_SUM']))
        self.assertEqual(receipt['actual_calls'], dict(native_forward=7, native_backward=5,
            prefix_calls=0, independent_qr_svd=2, independent_cholesky=2))
        for key, count in receipt['actual_calls'].items():
            self.assertEqual(count, receipt['sealed_budget'][key + '_max'])
        self.assertFalse(receipt['GPU_qualified'])


if __name__ == '__main__':
    unittest.main()
